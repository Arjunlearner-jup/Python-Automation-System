"""Persistent daily/weekly scheduler for standalone CSV ETL execution."""

from __future__ import annotations

import json
import logging
import signal
from datetime import datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from .csv_automation import (
    DEFAULT_LOG_DIRECTORY,
    PROJECT_ROOT,
    run_csv_pipeline,
)


DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "data" / "config" / "csv_schedule.json"
LOGGER = logging.getLogger("csv_etl.scheduler")
_SECRET_FIELD_NAMES = {"password", "secret", "token", "credential"}
_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


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
        content = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read CSV schedule: {error}") from error
    if not isinstance(content, dict):
        raise ValueError("CSV schedule configuration must be a JSON object.")
    return content


def _write_schedule(
    schedule: Mapping[str, Any], path: str | Path = DEFAULT_SCHEDULE_PATH
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return dict(schedule)


def _reject_secrets(value: Any, path: str = "schedule") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised_key = str(key).lower()
            if any(secret_word in normalised_key for secret_word in _SECRET_FIELD_NAMES):
                raise ValueError(
                    f"Do not persist credentials in schedule settings ({path}.{key})."
                )
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def save_csv_schedule(
    *,
    frequency: str,
    execution_time: str,
    source_directory: str | Path,
    table_name: str,
    unique_key_columns: list[str],
    weekday: int = 0,
    enabled: bool = True,
    pipeline_options: Mapping[str, Any] | None = None,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
) -> dict[str, Any]:
    """Create or replace the persistent CSV schedule without storing credentials."""
    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'.")
    try:
        parsed_time = datetime.strptime(execution_time, "%H:%M").time()
    except ValueError as error:
        raise ValueError("execution_time must use HH:MM (24-hour) format.") from error
    if frequency == "weekly" and not 0 <= weekday <= 6:
        raise ValueError("weekday must be in the range 0-6 (Monday-Sunday).")
    if not str(source_directory).strip():
        raise ValueError("source_directory is required.")
    if not table_name.strip():
        raise ValueError("table_name is required.")
    if not unique_key_columns:
        raise ValueError("At least one unique key column is required.")
    options = dict(pipeline_options or {})
    _reject_secrets(options)
    created_at = datetime.now().astimezone()
    schedule = {
        "enabled": bool(enabled),
        "frequency": frequency,
        "execution_time": parsed_time.strftime("%H:%M"),
        "weekday": int(weekday),
        "source_directory": str(Path(source_directory).expanduser().resolve()),
        "last_run_at": None,
        "created_at": created_at.isoformat(),
        "pipeline_options": {
            **options,
            "table_name": table_name,
            "unique_key_columns": list(unique_key_columns),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return _write_schedule(schedule, path)


def set_csv_schedule_enabled(
    enabled: bool, *, path: str | Path = DEFAULT_SCHEDULE_PATH
) -> dict[str, Any]:
    """Enable or disable the saved schedule while retaining its configuration."""
    schedule = _read_schedule(path)
    if enabled and not schedule.get("source_directory"):
        raise ValueError("Configure a CSV schedule before enabling it.")
    schedule["enabled"] = bool(enabled)
    schedule["updated_at"] = datetime.now(timezone.utc).isoformat()
    return _write_schedule(schedule, path)


def get_csv_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> dict[str, Any]:
    """Return saved schedule settings."""
    return _read_schedule(path)


def get_next_run_time(
    schedule: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Calculate the next local execution time after the last scheduled run."""
    if not schedule.get("enabled"):
        return None
    try:
        hour, minute = map(int, str(schedule["execution_time"]).split(":"))
        target_time = wall_time(hour, minute)
        frequency = schedule["frequency"]
        weekday = int(schedule.get("weekday", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved CSV schedule has invalid timing fields.") from error
    current = now or datetime.now().astimezone()
    current = current.astimezone() if current.tzinfo else current.astimezone()
    last_run_value = schedule.get("last_run_at")
    last_run = None
    if last_run_value:
        last_run = datetime.fromisoformat(str(last_run_value))
        if last_run.tzinfo is None:
            last_run = last_run.astimezone()
    if frequency == "daily":
        candidate = datetime.combine(current.date(), target_time).astimezone()
        if candidate <= current:
            candidate += timedelta(days=1)
        if last_run is not None and candidate <= last_run:
            candidate = datetime.combine(last_run.date() + timedelta(days=1), target_time).astimezone()
        return candidate
    if frequency == "weekly" and 0 <= weekday <= 6:
        days_ahead = (weekday - current.weekday()) % 7
        candidate = datetime.combine(current.date() + timedelta(days=days_ahead), target_time).astimezone()
        if candidate <= current:
            candidate += timedelta(days=7)
        if last_run is not None and candidate <= last_run:
            candidate += timedelta(days=7)
        return candidate
    raise ValueError("Saved CSV schedule frequency must be daily or weekly.")


def run_saved_schedule_once(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    runner: Callable[..., dict[str, Any]] = run_csv_pipeline,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Run a due schedule at most once, recording the due time before execution."""
    schedule = _read_schedule(path)
    current = now or datetime.now().astimezone()
    current = current.astimezone() if current.tzinfo else current.astimezone()
    if not schedule.get("enabled"):
        return None
    try:
        hour, minute = map(int, str(schedule["execution_time"]).split(":"))
        target_time = wall_time(hour, minute)
        frequency = schedule["frequency"]
        weekday = int(schedule.get("weekday", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved CSV schedule has invalid timing fields.") from error
    last_run_value = schedule.get("last_run_at")
    if last_run_value or schedule.get("created_at"):
        previous = datetime.fromisoformat(
            str(last_run_value or schedule["created_at"])
        )
        if previous.tzinfo is None:
            previous = previous.astimezone()
        if frequency == "daily":
            due_date = previous.date()
            candidate = datetime.combine(due_date, target_time).astimezone()
            if candidate <= previous:
                due_date += timedelta(days=1)
        elif frequency == "weekly":
            days_ahead = (weekday - previous.weekday()) % 7
            due_date = previous.date() + timedelta(days=days_ahead)
            candidate = datetime.combine(due_date, target_time).astimezone()
            if candidate <= previous:
                due_date += timedelta(days=7)
        else:
            raise ValueError("Saved CSV schedule frequency must be daily or weekly.")
    elif frequency == "daily":
        due_date = current.date()
    elif frequency == "weekly":
        days_behind = (current.weekday() - weekday) % 7
        due_date = current.date() - timedelta(days=days_behind)
    else:
        raise ValueError("Saved CSV schedule frequency must be daily or weekly.")
    due_at = datetime.combine(due_date, target_time).astimezone()
    if due_at > current:
        return None

    schedule["last_run_at"] = current.isoformat()
    _write_schedule(schedule, path)
    source_directory = Path(schedule["source_directory"])
    sources = sorted(source_directory.glob("*.csv"))
    if not sources:
        raise FileNotFoundError(
            f"No CSV files found in scheduled source directory {source_directory}."
        )
    options = dict(schedule.get("pipeline_options") or {})
    return runner(sources, **options)


def serve_csv_scheduler(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    poll_interval_seconds: float = 20,
    stop_event: Event | None = None,
    runner: Callable[..., dict[str, Any]] = run_csv_pipeline,
) -> None:
    """Run the standalone scheduler loop until stopped by the caller or signal."""
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")
    stopping = stop_event or Event()
    while not stopping.is_set():
        try:
            run_saved_schedule_once(path=path, runner=runner)
        except Exception:
            LOGGER.exception("CSV scheduled execution failed.")
        stopping.wait(poll_interval_seconds)


def main() -> None:
    """CLI entry point: ``python -m SRC.automation.csv_scheduler``."""
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
    LOGGER.info("CSV scheduler started; schedule file: %s", DEFAULT_SCHEDULE_PATH)
    serve_csv_scheduler(stop_event=stopping)


if __name__ == "__main__":
    main()
