"""Persistent standalone daily/weekly scheduler for JSON ETL jobs."""

from __future__ import annotations

import json
import logging
import signal
from datetime import datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from .json_automation import DEFAULT_LOG_DIRECTORY, PROJECT_ROOT, run_json_pipeline


DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "data" / "config" / "json_schedule.json"
LOGGER = logging.getLogger("json_etl.scheduler")
_SECRET_MARKERS = ("password", "secret", "token", "credential")


def _read_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> dict[str, Any]:
    schedule_path = Path(path)
    if not schedule_path.exists():
        return {
            "enabled": False,
            "frequency": "daily",
            "execution_time": "00:00",
            "weekday": 0,
            "last_run_at": None,
            "pipeline_options": {},
        }
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read JSON schedule: {error}") from error
    if not isinstance(schedule, dict):
        raise ValueError("JSON schedule configuration must be a JSON object.")
    return schedule


def _write_schedule(
    schedule: Mapping[str, Any], path: str | Path = DEFAULT_SCHEDULE_PATH
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return dict(schedule)


def _reject_secrets(value: Any, path: str = "schedule") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
                raise ValueError(
                    f"Do not persist credentials in schedule settings ({path}.{key})."
                )
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def save_json_schedule(
    *,
    frequency: str,
    execution_time: str,
    source_directory: str | Path,
    table_name: str,
    unique_key_fields: list[str] | None = None,
    weekday: int = 0,
    enabled: bool = True,
    pipeline_options: Mapping[str, Any] | None = None,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
) -> dict[str, Any]:
    """Persist schedule timing and non-secret pipeline options."""
    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'.")
    try:
        parsed_time = datetime.strptime(execution_time, "%H:%M").time()
    except ValueError as error:
        raise ValueError("execution_time must use HH:MM (24-hour) format.") from error
    if frequency == "weekly" and not 0 <= weekday <= 6:
        raise ValueError("weekday must be in the range 0-6 (Monday-Sunday).")
    if not str(source_directory).strip() or not table_name.strip():
        raise ValueError("source_directory and table_name are required.")
    options = dict(pipeline_options or {})
    _reject_secrets(options)
    now = datetime.now().astimezone()
    schedule = {
        "enabled": bool(enabled),
        "frequency": frequency,
        "execution_time": parsed_time.strftime("%H:%M"),
        "weekday": int(weekday),
        "source_directory": str(Path(source_directory).expanduser().resolve()),
        "last_run_at": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "pipeline_options": {
            **options,
            "table_name": table_name,
            "unique_key_fields": list(unique_key_fields or []),
        },
    }
    return _write_schedule(schedule, path)


def set_json_schedule_enabled(
    enabled: bool, *, path: str | Path = DEFAULT_SCHEDULE_PATH
) -> dict[str, Any]:
    """Enable or disable the persisted JSON schedule."""
    schedule = _read_schedule(path)
    if enabled and not schedule.get("source_directory"):
        raise ValueError("Configure a JSON schedule before enabling it.")
    schedule["enabled"] = bool(enabled)
    schedule["updated_at"] = datetime.now(timezone.utc).isoformat()
    return _write_schedule(schedule, path)


def get_json_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> dict[str, Any]:
    """Load persistent schedule configuration."""
    return _read_schedule(path)


def get_next_run_time(
    schedule: Mapping[str, Any], *, now: datetime | None = None
) -> datetime | None:
    """Calculate the next local run while respecting a persisted last-run time."""
    if not schedule.get("enabled"):
        return None
    try:
        hour, minute = map(int, str(schedule["execution_time"]).split(":"))
        target = wall_time(hour, minute)
        frequency = schedule["frequency"]
        weekday = int(schedule.get("weekday", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved JSON schedule has invalid timing fields.") from error
    current = now or datetime.now().astimezone()
    current = current.astimezone() if current.tzinfo else current.astimezone()
    previous_value = schedule.get("last_run_at") or schedule.get("created_at")
    if previous_value:
        previous = datetime.fromisoformat(str(previous_value))
        if previous.tzinfo is None:
            previous = previous.astimezone()
        if frequency == "daily":
            due_date = previous.date()
            candidate = datetime.combine(due_date, target).astimezone()
            if candidate <= previous:
                due_date += timedelta(days=1)
        elif frequency == "weekly":
            days_ahead = (weekday - previous.weekday()) % 7
            due_date = previous.date() + timedelta(days=days_ahead)
            candidate = datetime.combine(due_date, target).astimezone()
            if candidate <= previous:
                due_date += timedelta(days=7)
        else:
            raise ValueError("Saved JSON schedule frequency must be daily or weekly.")
        candidate = datetime.combine(due_date, target).astimezone()
        while candidate <= current:
            due_date += timedelta(days=1 if frequency == "daily" else 7)
            candidate = datetime.combine(due_date, target).astimezone()
        return candidate

    if frequency == "daily":
        candidate = datetime.combine(current.date(), target).astimezone()
        return candidate if candidate > current else candidate + timedelta(days=1)
    if frequency == "weekly" and 0 <= weekday <= 6:
        days = (weekday - current.weekday()) % 7
        candidate = datetime.combine(current.date() + timedelta(days=days), target).astimezone()
        return candidate if candidate > current else candidate + timedelta(days=7)
    raise ValueError("Saved JSON schedule frequency must be daily or weekly.")


def run_saved_schedule_once(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    runner: Callable[..., dict[str, Any]] = run_json_pipeline,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Run a due schedule at most once, persisting the due marker before launch."""
    schedule = _read_schedule(path)
    if not schedule.get("enabled"):
        return None
    current = now or datetime.now().astimezone()
    current = current.astimezone() if current.tzinfo else current.astimezone()
    try:
        hour, minute = map(int, str(schedule["execution_time"]).split(":"))
        target = wall_time(hour, minute)
        frequency = schedule["frequency"]
        weekday = int(schedule.get("weekday", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved JSON schedule has invalid timing fields.") from error
    previous_value = schedule.get("last_run_at") or schedule.get("created_at")
    if previous_value:
        previous = datetime.fromisoformat(str(previous_value))
        if previous.tzinfo is None:
            previous = previous.astimezone()
        if frequency == "daily":
            due_date = previous.date()
            if datetime.combine(due_date, target).astimezone() <= previous:
                due_date += timedelta(days=1)
        elif frequency == "weekly":
            days = (weekday - previous.weekday()) % 7
            due_date = previous.date() + timedelta(days=days)
            if datetime.combine(due_date, target).astimezone() <= previous:
                due_date += timedelta(days=7)
        else:
            raise ValueError("Saved JSON schedule frequency must be daily or weekly.")
    elif frequency == "daily":
        due_date = current.date()
    elif frequency == "weekly":
        due_date = current.date() - timedelta(
            days=(current.weekday() - weekday) % 7
        )
    else:
        raise ValueError("Saved JSON schedule frequency must be daily or weekly.")
    due_at = datetime.combine(due_date, target).astimezone()
    if due_at > current:
        return None

    schedule["last_run_at"] = due_at.isoformat()
    _write_schedule(schedule, path)
    source_directory = Path(schedule["source_directory"])
    sources = sorted(source_directory.glob("*.json"))
    if not sources:
        raise FileNotFoundError(
            f"No JSON files found in scheduled source directory {source_directory}."
        )
    return runner(sources, **dict(schedule.get("pipeline_options") or {}))


def serve_json_scheduler(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    poll_interval_seconds: float = 20,
    stop_event: Event | None = None,
    runner: Callable[..., dict[str, Any]] = run_json_pipeline,
) -> None:
    """Poll the saved schedule until shutdown; scheduler errors are logged."""
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")
    stopping = stop_event or Event()
    while not stopping.is_set():
        try:
            run_saved_schedule_once(path=path, runner=runner)
        except Exception:
            LOGGER.exception("JSON scheduled execution failed.")
        stopping.wait(poll_interval_seconds)


def main() -> None:
    """Standalone service entry point."""
    DEFAULT_LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=DEFAULT_LOG_DIRECTORY / "scheduler.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    stopping = Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    LOGGER.info("JSON scheduler started; schedule file: %s", DEFAULT_SCHEDULE_PATH)
    serve_json_scheduler(stop_event=stopping)


if __name__ == "__main__":
    main()
