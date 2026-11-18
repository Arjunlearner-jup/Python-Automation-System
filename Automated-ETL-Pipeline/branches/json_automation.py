"""End-to-end JSON ETL orchestration with durable logs and safe retries."""

from __future__ import annotations

import json
import math
import os
import smtplib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..cleaning.json_cleaner import clean_json_data
from ..database.json_loader import (
    get_connection_config_from_environment,
    load_validated_json_data,
    suggest_unique_key_fields,
)
from ..ingestion.json_extractor import extract_json_file
from ..integration.json_integrator import integrate_multiple_json_datasets
from ..profiling.json_profiler import profile_json
from ..validation.json_validator import validate_json_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "data" / "logs" / "json"
DEFAULT_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "automation" / "json"
)
_PROCESS_LOCK = threading.Lock()


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another process or thread owns the JSON pipeline lock."""


class PipelineStageError(RuntimeError):
    """Raised internally when a JSON ETL stage fails."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    return value


def _safe_error(error: BaseException, config: Mapping[str, Any]) -> str:
    message = str(error)
    password = str(config.get("password") or "")
    return message.replace(password, "[redacted]") if password else message


def _write_json(destination: Path, value: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _safe_stem(source: str) -> str:
    stem = Path(source).stem
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in stem
    )
    return safe.strip("._") or "dataset"


@contextmanager
def _execution_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a non-blocking thread and cross-process execution lock."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise PipelineAlreadyRunning("A JSON pipeline execution is already running.")
    lock_file = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+b")
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            raise PipelineAlreadyRunning(
                "A JSON pipeline execution is already running."
            ) from error
        yield
    finally:
        if lock_file is not None:
            try:
                if acquired:
                    lock_file.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_file.close()
        _PROCESS_LOCK.release()


def _record_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, Mapping):
        object_arrays = [
            item
            for item in value.values()
            if isinstance(item, list)
            and all(isinstance(record, Mapping) for record in item)
        ]
        if len(object_arrays) == 1:
            return len(object_arrays[0])
        return 1
    return 1


def _run_stage(
    *,
    name: str,
    operation: Callable[[], Any],
    record_count: Callable[[Any], int],
    input_records: int,
    retry_attempts: int,
    retry_on: tuple[type[BaseException], ...],
    connection_config: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> Any:
    started = _utcnow()
    entry: dict[str, Any] = {
        "stage": name,
        "status": "running",
        "started_at": started.isoformat(),
        "attempts": 0,
        "records_processed": input_records,
        "error_message": None,
    }
    stages.append(entry)

    def finish(status: str) -> None:
        ended = _utcnow()
        entry["status"] = status
        entry["ended_at"] = ended.isoformat()
        entry["duration_seconds"] = max(0.0, (ended - started).total_seconds())

    last_error: BaseException | None = None
    for attempt in range(retry_attempts + 1):
        entry["attempts"] = attempt + 1
        try:
            result = operation()
            if isinstance(result, Mapping) and result.get("execution_status") in {
                "failed",
                "error",
            }:
                entry["error_message"] = result.get(
                    "error_message", "Stage returned a failed status."
                )
                error_type = result.get("error_type")
                if error_type in {
                    "TimeoutError",
                    "ConnectionError",
                    "OperationalError",
                    "InterfaceError",
                }:
                    raise TimeoutError(entry["error_message"])
                raise PipelineStageError(entry["error_message"])
            entry["records_processed"] = record_count(result)
            if isinstance(result, Mapping):
                for key in ("inserted_rows", "updated_rows", "skipped_rows", "failed_rows"):
                    entry[key] = int(result.get(key, 0) or 0)
            finish("success")
            return result
        except Exception as error:
            last_error = error
            if attempt >= retry_attempts or not isinstance(error, retry_on):
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))

    assert last_error is not None
    finish("failed")
    entry["error_message"] = _safe_error(last_error, connection_config)
    entry["error_type"] = type(last_error).__name__
    if not entry.get("failed_rows"):
        entry["failed_rows"] = input_records
    raise PipelineStageError(f"{name} failed: {entry['error_message']}") from last_error


def send_email_notification(
    execution: Mapping[str, Any],
    *,
    smtp_config: Mapping[str, Any] | None = None,
) -> None:
    """Send an execution notification using environment-based SMTP settings."""
    config = dict(smtp_config or {})
    host = config.get("host") or os.getenv("SMTP_HOST")
    port = int(config.get("port") or os.getenv("SMTP_PORT", "587"))
    username = config.get("username") or os.getenv("SMTP_USER", "")
    password = config.get("password") or os.getenv("SMTP_PASSWORD", "")
    sender = config.get("from_address") or os.getenv("SMTP_FROM", username)
    recipients_value = config.get("to") or os.getenv("SMTP_TO", "")
    recipients = (
        [part.strip() for part in recipients_value.split(",") if part.strip()]
        if isinstance(recipients_value, str)
        else list(recipients_value)
    )
    if not host or not sender or not recipients:
        raise ValueError(
            "SMTP_HOST, SMTP_FROM, and SMTP_TO must be configured for notifications."
        )
    message = EmailMessage()
    message["Subject"] = (
        f"JSON ETL {execution.get('execution_status', 'unknown')}: "
        f"{execution.get('dataset_name', 'dataset')}"
    )
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(
        "\n".join(
            (
                f"Dataset: {execution.get('dataset_name', 'dataset')}",
                f"Status: {execution.get('execution_status', 'unknown')}",
                f"Execution ID: {execution.get('execution_id', '')}",
                f"Started: {execution.get('start_time', '')}",
                f"Ended: {execution.get('end_time', '')}",
                f"Error summary: {execution.get('error_message') or 'None'}",
            )
        )
    )
    with smtplib.SMTP(str(host), port, timeout=20) as server:
        server.starttls()
        if username:
            server.login(str(username), str(password))
        server.send_message(message)


def run_json_pipeline(
    sources: Sequence[str | Path],
    *,
    table_name: str,
    unique_key_fields: Sequence[str] | None = None,
    connection_config: Mapping[str, Any] | None = None,
    duplicate_strategy: str = "skip",
    batch_size: int = 1000,
    cleaning_options: Mapping[str, Any] | None = None,
    validation_options: Mapping[str, Any] | None = None,
    integration_options: Mapping[str, Any] | None = None,
    retry_attempts: int = 0,
    retry_on: tuple[type[BaseException], ...] = (OSError, TimeoutError, ConnectionError),
    notify: bool = False,
    smtp_config: Mapping[str, Any] | None = None,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    raw_archive_directory: str | Path | None = None,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run JSON ingestion through database loading without mutating source files."""
    source_paths = [Path(source).expanduser().resolve() for source in sources]
    if not source_paths:
        raise ValueError("Provide at least one JSON source path.")
    if retry_attempts < 0:
        raise ValueError("retry_attempts cannot be negative.")
    config = dict(connection_config or get_connection_config_from_environment())
    run_id = str(uuid.uuid4())
    start = _utcnow()
    names = [path.name for path in source_paths]
    dataset_name = names[0] if len(names) == 1 else f"{len(names)} JSON files"
    log_root = Path(log_directory)
    artifacts = Path(artifact_directory) / run_id
    archive = (
        Path(raw_archive_directory)
        if raw_archive_directory is not None
        else artifacts / "originals"
    )
    stages: list[dict[str, Any]] = []
    execution: dict[str, Any] = {
        "execution_id": run_id,
        "dataset_name": dataset_name,
        "source_filenames": names,
        "source_paths": [str(path) for path in source_paths],
        "start_time": start.isoformat(),
        "end_time": None,
        "execution_status": "running",
        "stage_records": stages,
        "processed_records": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
        "artifact_directory": str(artifacts),
        "execution_options": {
            "table_name": table_name,
            "unique_key_fields": list(unique_key_fields or []),
            "duplicate_strategy": duplicate_strategy,
            "batch_size": batch_size,
            "cleaning_options": dict(cleaning_options or {}),
            "validation_options": dict(validation_options or {}),
            "integration_options": dict(integration_options or {}),
            "retry_attempts": retry_attempts,
            "raw_archive_directory": str(archive),
        },
    }

    try:
        with _execution_lock(log_root / ".json_pipeline.lock"):
            extractions: list[dict[str, Any]] = []

            def ingest() -> list[dict[str, Any]]:
                nonlocal extractions
                extractions = [
                    extract_json_file(source, raw_directory=archive)
                    for source in source_paths
                ]
                failed = [
                    item
                    for item in extractions
                    if item["metadata"].get("status") != "success"
                ]
                if failed:
                    raise PipelineStageError(
                        "; ".join(
                            f"{item['metadata'].get('source')}: "
                            f"{item['metadata'].get('error', 'ingestion failed')}"
                            for item in failed
                        )
                    )
                return extractions

            extractions = _run_stage(
                name="ingestion",
                operation=ingest,
                record_count=lambda items: sum(
                    _record_count(item["json_data"]) for item in items
                ),
                input_records=0,
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["processed_records"] = sum(
                _record_count(item["json_data"]) for item in extractions
            )
            identifiers = [
                f"{index}:{item['metadata']['source']}"
                for index, item in enumerate(extractions)
            ]
            names_by_id = {
                identifier: item["metadata"]["source"]
                for identifier, item in zip(identifiers, extractions)
            }
            for identifier, item in zip(identifiers, extractions):
                _write_json(
                    artifacts / "ingested" / f"{_safe_stem(identifier)}.json",
                    item["json_data"],
                )

            profiles: dict[str, Any] = {}

            def profile() -> dict[str, Any]:
                nonlocal profiles
                profiles = {
                    identifier: profile_json(
                        item["data"],
                        item["metadata"]["source"],
                        parsed_json=item["json_data"],
                    )
                    for identifier, item in zip(identifiers, extractions)
                }
                return profiles

            _run_stage(
                name="profiling",
                operation=profile,
                record_count=lambda results: sum(
                    result["dimensions"]["number_of_records"]
                    for result in results.values()
                ),
                input_records=execution["processed_records"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["profiling_reports"] = {
                key: value["quality_summary"] for key, value in profiles.items()
            }
            for key, value in profiles.items():
                duplicates = value["duplicates"]["duplicate_records"]
                if not duplicates.empty:
                    _write_json(
                        artifacts / "profiled" / f"{_safe_stem(key)}_duplicates.json",
                        duplicates.to_dict(orient="records"),
                    )

            cleaned: dict[str, dict[str, Any]] = {}

            def clean() -> dict[str, dict[str, Any]]:
                nonlocal cleaned
                cleaned = {
                    key: clean_json_data(
                        profile["parsed_json"],
                        names_by_id[key],
                        **dict(cleaning_options or {}),
                    )
                    for key, profile in profiles.items()
                }
                if any(
                    result["report"].get("cleaning_status") == "error"
                    for result in cleaned.values()
                ):
                    raise PipelineStageError("JSON cleaning returned an error status.")
                return cleaned

            cleaned = _run_stage(
                name="cleaning",
                operation=clean,
                record_count=lambda results: sum(
                    _record_count(result["data"]) for result in results.values()
                ),
                input_records=execution["processed_records"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["cleaning_reports"] = {
                key: result["report"] for key, result in cleaned.items()
            }
            for key, result in cleaned.items():
                _write_json(
                    artifacts / "cleaned" / f"{_safe_stem(key)}.json",
                    result["data"],
                )

            validation_results: dict[str, dict[str, Any]] = {}

            def validate() -> dict[str, dict[str, Any]]:
                nonlocal validation_results
                validation_results = {
                    key: validate_json_data(
                        result["data"],
                        names_by_id[key],
                        **dict(validation_options or {}),
                    )
                    for key, result in cleaned.items()
                }
                failed = [
                    key
                    for key, result in validation_results.items()
                    if result["validation_status"]
                    not in {"passed", "passed_with_warnings"}
                    or "valid_data" not in result
                ]
                if failed:
                    raise PipelineStageError(
                        "JSON validation failed for: "
                        + ", ".join(names_by_id[key] for key in failed)
                    )
                return validation_results

            validation_results = _run_stage(
                name="validation",
                operation=validate,
                record_count=lambda results: sum(
                    int(result["validation_report"].get("valid_records") or 0)
                    for result in results.values()
                ),
                input_records=sum(
                    _record_count(result["data"]) for result in cleaned.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["validation_reports"] = {
                key: result["validation_report"]
                for key, result in validation_results.items()
            }
            for key, result in validation_results.items():
                _write_json(
                    artifacts / "validated" / f"{_safe_stem(key)}.json",
                    result["valid_data"],
                )
                _write_json(
                    artifacts / "invalid" / f"{_safe_stem(key)}.json",
                    result["invalid_data"],
                )

            integration_result: dict[str, Any] = {}

            def integrate() -> dict[str, Any]:
                nonlocal integration_result
                datasets = {
                    key: {
                        "validation_result": validation_results[key],
                        "source": names_by_id[key],
                        "metadata": extractions[index]["metadata"],
                    }
                    for index, key in enumerate(identifiers)
                }
                integration_result = integrate_multiple_json_datasets(
                    datasets,
                    save_outputs=False,
                    **dict(integration_options or {}),
                )
                if "data" not in integration_result:
                    raise PipelineStageError(
                        "JSON integration did not return integrated data."
                    )
                return integration_result

            integration_result = _run_stage(
                name="integration",
                operation=integrate,
                record_count=lambda result: _record_count(result["data"]),
                input_records=sum(
                    int(result["validation_report"].get("valid_records") or 0)
                    for result in validation_results.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            integrated_data = integration_result["data"]
            _write_json(artifacts / "integrated" / "integrated.json", integrated_data)
            execution["integration_report"] = integration_result.get(
                "integration_report", {}
            )

            integrated_validation: dict[str, Any] = {}

            def validate_integrated() -> dict[str, Any]:
                nonlocal integrated_validation
                integrated_validation = validate_json_data(
                    integrated_data,
                    dataset_name,
                    **dict(validation_options or {}),
                )
                if (
                    integrated_validation["validation_status"]
                    not in {"passed", "passed_with_warnings"}
                    or "valid_data" not in integrated_validation
                ):
                    raise PipelineStageError(
                        "Integrated JSON did not pass Stage 4 validation."
                    )
                return integrated_validation

            _run_stage(
                name="integrated_validation",
                operation=validate_integrated,
                record_count=lambda result: int(
                    result["validation_report"].get("valid_records") or 0
                ),
                input_records=_record_count(integrated_data),
                retry_attempts=0,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )

            def load() -> dict[str, Any]:
                result = load_validated_json_data(
                    integrated_validation["valid_data"],
                    connection_config=config,
                    table_name=table_name,
                    unique_key_fields=unique_key_fields,
                    duplicate_strategy=duplicate_strategy,
                    dataset_name=dataset_name,
                    source_filename=", ".join(names),
                    batch_size=batch_size,
                    connect_factory=connect_factory,
                )
                execution["database_load"] = result
                for count in ("inserted_rows", "updated_rows", "skipped_rows", "failed_rows"):
                    execution[count] = result.get(count, 0)
                return result

            load_result = _run_stage(
                name="database_loading",
                operation=load,
                record_count=lambda result: int(result.get("processed_rows", 0)),
                input_records=_record_count(integrated_validation["valid_data"]),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["execution_status"] = load_result["execution_status"]
            if execution["execution_status"] != "success":
                execution["error_message"] = load_result.get(
                    "error_message", "Database loading failed."
                )
            execution["integrated_data"] = integrated_data
    except PipelineAlreadyRunning as error:
        execution["execution_status"] = "skipped"
        execution["error_message"] = str(error)
    except Exception as error:
        execution["execution_status"] = "failed"
        execution["error_message"] = _safe_error(error, config)
        if not execution.get("failed_rows"):
            execution["failed_rows"] = max(
                0,
                execution["processed_records"]
                - execution["inserted_rows"]
                - execution["updated_rows"]
                - execution["skipped_rows"],
            )
    finally:
        execution["end_time"] = _utcnow().isoformat()
        execution["duration_seconds"] = max(
            0.0,
            (
                datetime.fromisoformat(execution["end_time"])
                - datetime.fromisoformat(execution["start_time"])
            ).total_seconds(),
        )
        try:
            log_path = log_root / f"{run_id}.json"
            _write_json(
                log_path,
                {
                    key: value
                    for key, value in execution.items()
                    if key != "integrated_data"
                },
            )
            execution["log_path"] = str(log_path)
        except OSError as error:
            execution["log_write_error"] = str(error)

    if notify:
        try:
            send_email_notification(execution, smtp_config=smtp_config)
            execution["notification_status"] = "sent"
        except Exception as error:
            message = str(error)
            for secret in (
                str((smtp_config or {}).get("password") or ""),
                str(config.get("password") or ""),
            ):
                if secret:
                    message = message.replace(secret, "[redacted]")
            execution["notification_status"] = "failed"
            execution["notification_error"] = message
        try:
            _write_json(
                Path(execution.get("log_path", log_root / f"{run_id}.json")),
                {
                    key: value
                    for key, value in execution.items()
                    if key not in {"integrated_data", "smtp_config"}
                },
            )
        except OSError:
            pass
    return execution


def get_json_execution_history(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read recent JSON execution logs; invalid logs are surfaced as errors."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    root = Path(log_directory)
    if not root.exists():
        return []
    history = []
    paths = sorted(
        root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read JSON execution log {path}: {error}") from error
        if isinstance(record, dict):
            record["log_path"] = str(path)
            history.append(record)
            if len(history) >= limit:
                break
    return history


def rerun_failed_execution(
    execution_id: str,
    *,
    connection_config: Mapping[str, Any] | None = None,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    **overrides: Any,
) -> dict[str, Any]:
    """Re-run a failed JSON execution from its original source files."""
    record = next(
        (
            item
            for item in get_json_execution_history(log_directory, limit=10000)
            if item.get("execution_id") == execution_id
        ),
        None,
    )
    if record is None:
        raise KeyError(f"No JSON execution log found for {execution_id!r}.")
    if record.get("execution_status") != "failed":
        raise ValueError("Only failed JSON executions can be rerun.")
    options = dict(record.get("execution_options") or {})
    options.update(overrides)
    return run_json_pipeline(
        record.get("source_paths", []),
        connection_config=connection_config,
        log_directory=log_directory,
        table_name=options.pop("table_name"),
        unique_key_fields=options.pop("unique_key_fields"),
        **options,
    )


def suggest_json_pipeline_keys(data: Any) -> list[str]:
    """Return unique root-level key suggestions from the JSON loader."""
    return suggest_unique_key_fields(data)
