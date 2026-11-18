"""Persistent standalone daily/weekly scheduler for API ETL jobs.

Configure schedules through this module, then run the service separately with
``python -m SRC.automation.api_scheduler``. Importing the module never starts
the scheduler, which keeps Streamlit reruns from triggering API requests.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
from contextlib import contextmanager
from datetime import datetime, time as wall_time, timedelta
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .api_automation import (
    DEFAULT_LOG_DIRECTORY,
    PROJECT_ROOT,
    run_api_pipeline,
)


DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "data" / "config" / "api_schedule.json"
LOGGER = logging.getLogger("api_etl.scheduler")
_SECRET_MARKERS = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "authorization",
)
_ENV_REFERENCE_PREFIX = "${ENV:"


def _lock_file(path: Path, *, blocking: bool = True):
    """Lock one byte in a file, using the platform's native file-lock API."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b"\0")
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(descriptor, mode, 1)
        else:
            import fcntl

            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(descriptor, operation)
    except (OSError, BlockingIOError):
        os.close(descriptor)
        raise
    return descriptor


def _unlock_file(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


@contextmanager
def _schedule_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(f"{Path(path)}.lock")
    descriptor = _lock_file(lock_path)
    try:
        yield
    finally:
        _unlock_file(descriptor)


def _read_schedule_unlocked(path: str | Path) -> dict[str, Any]:
    schedule_path = Path(path)
    if not schedule_path.exists():
        return {
            "enabled": False,
            "frequency": "daily",
            "execution_time": "00:00",
            "weekday": 0,
            "last_run_at": None,
            "pipeline_options": {},
            "api_requests": [],
        }
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read API schedule: {error}") from error
    if not isinstance(schedule, dict):
        raise ValueError("API schedule configuration must be a JSON object.")
    return schedule


def _write_schedule_unlocked(
    schedule: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(schedule, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return dict(schedule)


def _is_env_reference(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_ENV_REFERENCE_PREFIX)
        and value.endswith("}")
        and bool(value[len(_ENV_REFERENCE_PREFIX) : -1].strip())
    )


def _reject_persisted_secrets(value: Any, path: str = "schedule") -> None:
    """Reject literal credentials; environment references remain safe to store."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized.endswith("_env"):
                if not isinstance(child, str) or not child.strip():
                    raise ValueError(f"{path}.{key} must name an environment variable.")
                continue
            if any(marker in normalized for marker in _SECRET_MARKERS):
                if not _is_env_reference(child):
                    raise ValueError(
                        f"Do not persist credentials in schedule settings "
                        f"({path}.{key}); use '${{ENV:VARIABLE_NAME}}'."
                    )
            _reject_persisted_secrets(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_persisted_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        parts = urlsplit(value)
        if parts.username is not None or parts.password is not None:
            raise ValueError(f"Do not persist credentials in URL authority ({path}).")
        for key, query_value in parse_qsl(parts.query, keep_blank_values=True):
            if any(marker in key.lower() for marker in _SECRET_MARKERS):
                if not _is_env_reference(query_value):
                    raise ValueError(
                        f"Do not persist credentials in URL query parameters "
                        f"({path}); use '${{ENV:VARIABLE_NAME}}'."
                    )


def _resolve_environment_references(value: Any) -> Any:
    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name.lower().endswith("_env"):
                target_key = name[:-4]
                variable = str(child).strip()
                if variable not in os.environ:
                    raise RuntimeError(
                        f"Required environment variable {variable!r} is not set."
                    )
                resolved[target_key] = os.environ[variable]
            else:
                resolved[name] = _resolve_environment_references(child)
        return resolved
    if isinstance(value, list):
        return [_resolve_environment_references(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_resolve_environment_references(child) for child in value)
    if _is_env_reference(value):
        variable = value[len(_ENV_REFERENCE_PREFIX) : -1].strip()
        if variable not in os.environ:
            raise RuntimeError(
                f"Required environment variable {variable!r} is not set."
            )
        return os.environ[variable]
    if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        parts = urlsplit(value)
        if parts.username is not None or parts.password is not None:
            raise ValueError("Scheduled API URLs must not contain embedded credentials.")
        query = []
        for key, item in parse_qsl(parts.query, keep_blank_values=True):
            query.append((key, _resolve_environment_references(item)))
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    return value


def save_api_schedule(
    *,
    api_requests: Mapping[str, Any] | list[Mapping[str, Any]],
    frequency: str,
    execution_time: str,
    table_name: str,
    unique_key_columns: list[str] | None = None,
    weekday: int = 0,
    enabled: bool = True,
    pipeline_options: Mapping[str, Any] | None = None,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
) -> dict[str, Any]:
    """Create or update the persistent API schedule without storing credentials.

    Sensitive request values may use ``${ENV:VARIABLE_NAME}`` or a sibling
    ``*_env`` field (for example ``api_key_env: API_TOKEN``). Database
    connection credentials are read from the normal API loader environment
    configuration when the scheduled job runs.
    """
    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'.")
    try:
        parsed_time = datetime.strptime(execution_time, "%H:%M").time()
    except ValueError as error:
        raise ValueError("execution_time must use HH:MM (24-hour) format.") from error
    if frequency == "weekly" and not 0 <= weekday <= 6:
        raise ValueError("weekday must be in the range 0-6 (Monday-Sunday).")
    if not table_name.strip():
        raise ValueError("table_name is required.")
    requests = (
        [dict(api_requests)]
        if isinstance(api_requests, Mapping)
        else [dict(item) for item in api_requests]
    )
    if not requests or any(not request for request in requests):
        raise ValueError("Provide at least one non-empty API request mapping.")
    options = dict(pipeline_options or {})
    options["table_name"] = table_name
    options["unique_key_columns"] = list(unique_key_columns or [])
    if any(key in options for key in ("connection_config", "smtp_config")):
        raise ValueError(
            "Do not persist database or SMTP credentials; configure them through "
            "the environment or Streamlit secrets."
        )
    _reject_persisted_secrets({"api_requests": requests, "pipeline_options": options})

    with _schedule_lock(path):
        now = datetime.now().astimezone().isoformat()
        schedule = {
            "enabled": bool(enabled),
            "frequency": frequency,
            "execution_time": parsed_time.strftime("%H:%M"),
            "weekday": int(weekday),
            "api_requests": requests,
            "pipeline_options": options,
            "created_at": now,
            "updated_at": now,
            "last_run_at": None,
        }
        return _write_schedule_unlocked(schedule, path)


def get_api_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> dict[str, Any]:
    """Return the persisted API schedule."""
    with _schedule_lock(path):
        return _read_schedule_unlocked(path)


def set_api_schedule_enabled(
    enabled: bool, *, path: str | Path = DEFAULT_SCHEDULE_PATH
) -> dict[str, Any]:
    """Enable or disable the saved schedule without discarding its settings."""
    with _schedule_lock(path):
        schedule = _read_schedule_unlocked(path)
        if enabled and not schedule.get("api_requests"):
            raise ValueError("Configure an API schedule before enabling it.")
        schedule["enabled"] = bool(enabled)
        if enabled:
            schedule["created_at"] = datetime.now().astimezone().isoformat()
            schedule["last_run_at"] = None
        schedule["updated_at"] = datetime.now().astimezone().isoformat()
        return _write_schedule_unlocked(schedule, path)


def _next_occurrence(
    schedule: Mapping[str, Any], reference: datetime, *, inclusive: bool = False
) -> datetime:
    try:
        hour, minute = map(int, str(schedule["execution_time"]).split(":"))
        target = wall_time(hour, minute)
        frequency = schedule["frequency"]
        weekday = int(schedule.get("weekday", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved API schedule has invalid timing fields.") from error
    current = reference.astimezone() if reference.tzinfo else reference.astimezone()
    if frequency == "daily":
        candidate = datetime.combine(current.date(), target).astimezone()
        if candidate < current or (candidate == current and not inclusive):
            candidate += timedelta(days=1)
        return candidate
    if frequency == "weekly":
        if not 0 <= weekday <= 6:
            raise ValueError("Saved API schedule weekday must be in the range 0-6.")
        days = (weekday - current.weekday()) % 7
        candidate = datetime.combine(
            current.date() + timedelta(days=days), target
        ).astimezone()
        if candidate < current or (candidate == current and not inclusive):
            candidate += timedelta(days=7)
        return candidate
    raise ValueError("Saved API schedule frequency must be daily or weekly.")


def get_next_run_time(
    schedule: Mapping[str, Any], *, now: datetime | None = None
) -> datetime | None:
    """Return the next scheduled local execution, or ``None`` if disabled."""
    if not schedule.get("enabled"):
        return None
    current = now or datetime.now().astimezone()
    previous_value = schedule.get("last_run_at")
    if previous_value:
        previous = datetime.fromisoformat(str(previous_value))
        if previous.tzinfo is None:
            previous = previous.astimezone()
        candidate = _next_occurrence(schedule, previous)
        while candidate <= current:
            candidate = _next_occurrence(schedule, candidate)
        return candidate
    created_value = schedule.get("created_at")
    if created_value:
        created = datetime.fromisoformat(str(created_value))
        if created.tzinfo is None:
            created = created.astimezone()
        candidate = _next_occurrence(schedule, created, inclusive=True)
        while candidate <= current:
            candidate = _next_occurrence(schedule, candidate)
        return candidate
    return _next_occurrence(schedule, current)


def _due_time(schedule: Mapping[str, Any], current: datetime) -> datetime:
    previous_value = schedule.get("last_run_at")
    if previous_value:
        previous = datetime.fromisoformat(str(previous_value))
        if previous.tzinfo is None:
            previous = previous.astimezone()
        return _next_occurrence(schedule, previous)
    created_value = schedule.get("created_at")
    if created_value:
        created = datetime.fromisoformat(str(created_value))
        if created.tzinfo is None:
            created = created.astimezone()
        return _next_occurrence(schedule, created, inclusive=True)
    return _next_occurrence(schedule, current, inclusive=True)


def run_saved_schedule_once(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    runner: Callable[..., dict[str, Any]] = run_api_pipeline,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Claim a due occurrence before executing it, so restarts cannot replay it."""
    current = now or datetime.now().astimezone()
    current = current.astimezone() if current.tzinfo else current.astimezone()
    with _schedule_lock(path):
        schedule = _read_schedule_unlocked(path)
        if not schedule.get("enabled"):
            return None
        due_at = _due_time(schedule, current)
        if due_at > current:
            return None
        schedule["last_run_at"] = current.isoformat()
        _write_schedule_unlocked(schedule, path)

    requests = _resolve_environment_references(schedule["api_requests"])
    options = _resolve_environment_references(
        dict(schedule.get("pipeline_options") or {})
    )
    return runner(requests, **options)


@contextmanager
def _scheduler_instance_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(f"{Path(path)}.service.lock")
    try:
        descriptor = _lock_file(lock_path, blocking=False)
    except (OSError, BlockingIOError) as error:
        raise RuntimeError(
            f"An API scheduler is already running for schedule {Path(path)}."
        ) from error
    try:
        yield
    finally:
        _unlock_file(descriptor)


def serve_api_scheduler(
    *,
    path: str | Path = DEFAULT_SCHEDULE_PATH,
    poll_interval_seconds: float = 15,
    stop_event: Event | None = None,
    runner: Callable[..., dict[str, Any]] = run_api_pipeline,
) -> None:
    """Run the persistent scheduler service until shutdown is requested."""
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")
    stopping = stop_event or Event()
    with _scheduler_instance_lock(path):
        LOGGER.info("API scheduler started; schedule file: %s", path)
        while not stopping.is_set():
            try:
                execution = run_saved_schedule_once(path=path, runner=runner)
                if execution is not None:
                    LOGGER.info(
                        "Scheduled API ETL completed with status %s.",
                        execution.get("status", "unknown"),
                    )
            except Exception:
                LOGGER.exception("Scheduled API ETL execution failed.")
            stopping.wait(poll_interval_seconds)


def main() -> None:
    """Standalone service entry point; no Streamlit session is required."""
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
    try:
        serve_api_scheduler(stop_event=stopping)
    except RuntimeError as error:
        LOGGER.error("%s", error)
        raise


if __name__ == "__main__":
    main()