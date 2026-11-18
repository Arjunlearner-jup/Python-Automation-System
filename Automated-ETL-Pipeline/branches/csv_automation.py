"""End-to-end CSV ETL orchestration, execution logging, and notifications."""

from __future__ import annotations

import json
import math
import os
import smtplib
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from ..cleaning.csv_cleaner import clean_csv_dataframe
from ..database.csv_loader import (
    get_connection_config_from_environment,
    load_validated_csv_dataframe,
)
from ..ingestion.csv_extractor import extract_csv_file
from ..integration.csv_integrator import integrate_multiple_csv_datasets
from ..profiling.csv_profiler import profile_csv
from ..validation.csv_validator import validate_csv_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "data" / "logs" / "csv"
DEFAULT_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "automation" / "csv"
)
_PROCESS_LOCK = threading.Lock()


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another process or thread owns the CSV pipeline lock."""


class PipelineStageError(RuntimeError):
    """Raised internally when a stage returns an unsuccessful result."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_error(error: BaseException, connection_config: Mapping[str, Any]) -> str:
    message = str(error)
    password = str(connection_config.get("password") or "")
    return message.replace(password, "[redacted]") if password else message


def _atomic_json_write(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


@contextmanager
def _execution_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a non-blocking cross-process lock for the complete pipeline."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise PipelineAlreadyRunning("A CSV pipeline execution is already running.")
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
        except (OSError, BlockingIOError) as error:
            raise PipelineAlreadyRunning(
                "A CSV pipeline execution is already running."
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


def _safe_stem(source: str) -> str:
    stem = Path(source).stem
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)
    return safe.strip("._") or "dataset"


def _write_frame(
    dataframe: pd.DataFrame, artifact_directory: Path, filename: str
) -> str:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    path = artifact_directory / filename
    dataframe.to_csv(path, index=False)
    return str(path)


def _run_stage(
    *,
    name: str,
    operation: Callable[[], Any],
    record_count: Callable[[Any], int],
    retry_attempts: int,
    retry_on: tuple[type[BaseException], ...],
    connection_config: Mapping[str, Any],
    stage_records: list[dict[str, Any]],
    input_record_count: int = 0,
) -> Any:
    started = _utcnow()
    stage_record: dict[str, Any] = {
        "stage": name,
        "status": "running",
        "started_at": started.isoformat(),
        "attempts": 0,
        "records_processed": input_record_count,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
    }
    stage_records.append(stage_record)

    def finish(status: str) -> None:
        ended = _utcnow()
        stage_record["ended_at"] = ended.isoformat()
        stage_record["duration_seconds"] = max(
            0.0, (ended - started).total_seconds()
        )
        stage_record["status"] = status

    last_error: BaseException | None = None
    for attempt in range(retry_attempts + 1):
        stage_record["attempts"] = attempt + 1
        try:
            result = operation()
            stage_record["records_processed"] = record_count(result)
            if isinstance(result, Mapping):
                for count_key in (
                    "inserted_rows",
                    "updated_rows",
                    "skipped_rows",
                    "failed_rows",
                ):
                    stage_record[count_key] = int(result.get(count_key, 0) or 0)
                if result.get("execution_status") in {"failed", "error"}:
                    stage_record["error_message"] = result.get(
                        "error_message", "Stage returned a failed execution status."
                    )
                    if result.get("error_type") in {
                        "TimeoutError",
                        "ConnectionError",
                        "OperationalError",
                        "InterfaceError",
                    }:
                        raise TimeoutError(stage_record["error_message"])
                    raise PipelineStageError(stage_record["error_message"])
            finish("success")
            return result
        except Exception as error:
            last_error = error
            if attempt >= retry_attempts or not isinstance(error, retry_on):
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))
    assert last_error is not None
    finish("failed")
    stage_record["error_message"] = _safe_error(last_error, connection_config)
    stage_record["error_type"] = type(last_error).__name__
    if not stage_record.get("failed_rows"):
        stage_record["failed_rows"] = int(stage_record["records_processed"] or 0)
    raise PipelineStageError(f"{name} failed: {stage_record['error_message']}") from last_error


def send_email_notification(
    execution: Mapping[str, Any],
    *,
    smtp_config: Mapping[str, Any] | None = None,
) -> None:
    """Send a notification using SMTP settings supplied through environment vars."""
    config = dict(smtp_config or {})
    host = config.get("host") or os.getenv("SMTP_HOST")
    port = int(config.get("port") or os.getenv("SMTP_PORT", "587"))
    username = config.get("username") or os.getenv("SMTP_USER", "")
    password = config.get("password") or os.getenv("SMTP_PASSWORD", "")
    sender = config.get("from_address") or os.getenv("SMTP_FROM", username)
    recipients_value = config.get("to") or os.getenv("SMTP_TO", "")
    recipients = (
        [str(item).strip() for item in recipients_value.split(",") if item.strip()]
        if isinstance(recipients_value, str)
        else list(recipients_value)
    )
    if not host or not sender or not recipients:
        raise ValueError(
            "SMTP_HOST, SMTP_FROM, and SMTP_TO must be configured for email notifications."
        )

    message = EmailMessage()
    message["Subject"] = (
        f"CSV ETL {execution.get('execution_status', 'unknown')}: "
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


def run_csv_pipeline(
    sources: Sequence[str | Path],
    *,
    table_name: str,
    unique_key_columns: Sequence[str],
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
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run CSV ingestion, profiling, cleaning, validation, integration, and MySQL loading.

    Each invocation receives a unique run directory; source files are never
    modified. A non-blocking process lock prevents overlapping executions.
    """
    source_list = list(sources)
    if not source_list:
        raise ValueError("Provide at least one CSV source path.")
    if retry_attempts < 0:
        raise ValueError("retry_attempts cannot be negative.")
    if not unique_key_columns:
        raise ValueError("Select at least one unique key column for safe loading.")
    db_config = dict(
        connection_config or get_connection_config_from_environment()
    )
    run_id = str(uuid.uuid4())
    started = _utcnow()
    dataset_names = [Path(source).name for source in source_list]
    dataset_name = (
        dataset_names[0] if len(dataset_names) == 1 else f"{len(dataset_names)} CSV files"
    )
    log_root = Path(log_directory)
    run_artifacts = Path(artifact_directory) / run_id
    stage_records: list[dict[str, Any]] = []
    execution: dict[str, Any] = {
        "execution_id": run_id,
        "dataset_name": dataset_name,
        "source_filenames": dataset_names,
        "source_paths": [str(Path(source).resolve()) for source in source_list],
        "start_time": started.isoformat(),
        "end_time": None,
        "execution_status": "running",
        "stage_records": stage_records,
        "processed_rows": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
        "artifact_directory": str(run_artifacts),
        "execution_options": {
            "table_name": table_name,
            "unique_key_columns": list(unique_key_columns),
            "duplicate_strategy": duplicate_strategy,
            "batch_size": batch_size,
            "cleaning_options": dict(cleaning_options or {}),
            "validation_options": dict(validation_options or {}),
            "integration_options": dict(integration_options or {}),
            "retry_attempts": retry_attempts,
        },
    }
    lock_path = log_root / ".csv_pipeline.lock"

    try:
        with _execution_lock(lock_path):
            extraction_results = []

            def ingest() -> list[dict[str, Any]]:
                nonlocal extraction_results
                extraction_results = [extract_csv_file(source) for source in source_list]
                failures = [
                    item
                    for item in extraction_results
                    if item["metadata"]["status"] != "success"
                    or not isinstance(item.get("data"), pd.DataFrame)
                ]
                if failures:
                    details = "; ".join(
                        f"{item['metadata']['source']}: "
                        f"{item['metadata'].get('error', 'extraction failed')}"
                        for item in failures
                    )
                    raise PipelineStageError(details)
                return extraction_results

            extraction_results = _run_stage(
                name="ingestion",
                operation=ingest,
                record_count=lambda items: sum(len(item["data"]) for item in items),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
            )
            source_ids = [
                f"{index}:{item['metadata']['source']}"
                for index, item in enumerate(extraction_results)
            ]
            source_names = {
                source_id: item["metadata"]["source"]
                for source_id, item in zip(source_ids, extraction_results)
            }
            execution["processed_rows"] = sum(
                len(item["data"]) for item in extraction_results
            )
            for index, item in enumerate(extraction_results):
                _write_frame(
                    item["data"],
                    run_artifacts / "ingested",
                    f"{index}_{_safe_stem(item['metadata']['source'])}.csv",
                )

            profiles: dict[str, Any] = {}

            def profile() -> dict[str, Any]:
                nonlocal profiles
                profiles = {
                    source_id: profile_csv(
                        item["data"], item["metadata"]["source"]
                    )
                    for source_id, item in zip(source_ids, extraction_results)
                }
                return profiles

            _run_stage(
                name="profiling",
                operation=profile,
                record_count=lambda items: sum(
                    item["dimensions"]["total_rows"] for item in items.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
                input_record_count=execution["processed_rows"],
            )
            execution["profiling_reports"] = {
                name: result["quality_summary"]
                for name, result in profiles.items()
            }
            for name, result in profiles.items():
                duplicate_rows = result["duplicates"]["duplicate_records"]
                if not duplicate_rows.empty:
                    _write_frame(
                        duplicate_rows,
                        run_artifacts / "profiled",
                        f"{_safe_stem(name)}_duplicate_rows.csv",
                    )

            cleaned_results: dict[str, dict[str, Any]] = {}

            def clean() -> dict[str, dict[str, Any]]:
                nonlocal cleaned_results
                cleaned_results = {
                    source_id: clean_csv_dataframe(
                        profile_result["data"],
                        source_names[source_id],
                        **dict(cleaning_options or {}),
                    )
                    for source_id, profile_result in profiles.items()
                }
                failures = [
                    name
                    for name, result in cleaned_results.items()
                    if result.get("report", {}).get("cleaning_status") == "error"
                ]
                if failures:
                    raise PipelineStageError(
                        "Cleaning failed for: " + ", ".join(failures)
                    )
                return cleaned_results

            cleaned_results = _run_stage(
                name="cleaning",
                operation=clean,
                record_count=lambda items: sum(
                    len(item["data"]) for item in items.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
                input_record_count=execution["processed_rows"],
            )
            for name, result in cleaned_results.items():
                _write_frame(
                    result["data"],
                    run_artifacts / "cleaned",
                    f"{_safe_stem(name)}.csv",
                )
                execution.setdefault("cleaning_reports", {})[name] = result["report"]

            validation_results: dict[str, dict[str, Any]] = {}

            def validate() -> dict[str, dict[str, Any]]:
                nonlocal validation_results
                validation_results = {
                    name: validate_csv_dataframe(
                        result["data"],
                        source_names[name],
                        **dict(validation_options or {}),
                    )
                    for name, result in cleaned_results.items()
                }
                failed = [
                    name
                    for name, result in validation_results.items()
                    if result["validation_status"] not in {
                        "passed",
                        "passed_with_warnings",
                    }
                    or result["valid_dataframe"].empty
                ]
                if failed:
                    raise PipelineStageError(
                        "Validation failed or produced no valid records for: "
                        + ", ".join(failed)
                    )
                return validation_results

            validation_results = _run_stage(
                name="validation",
                operation=validate,
                record_count=lambda items: sum(
                    len(item["valid_dataframe"]) for item in items.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
                input_record_count=sum(
                    len(item["data"]) for item in cleaned_results.values()
                ),
            )
            validated_inputs = {
                name: {
                    "validation_result": result,
                    "report": cleaned_results[name]["report"],
                    "source": source_names[name],
                }
                for name, result in validation_results.items()
            }
            for name, result in validation_results.items():
                _write_frame(
                    result["valid_dataframe"],
                    run_artifacts / "validated",
                    f"{_safe_stem(name)}.csv",
                )
                _write_frame(
                    result["invalid_dataframe"],
                    run_artifacts / "invalid",
                    f"{_safe_stem(name)}.csv",
                )
                _write_frame(
                    result["validation_errors"],
                    run_artifacts / "validation_errors",
                    f"{_safe_stem(name)}.csv",
                )
            execution["validation_reports"] = {
                name: {
                    "validation_report": result["validation_report"],
                    "validation_errors": result["validation_errors"].to_dict(
                        orient="records"
                    ),
                    "validation_warnings": result["validation_warnings"].to_dict(
                        orient="records"
                    ),
                }
                for name, result in validation_results.items()
            }

            integration_result: dict[str, Any] = {}

            def integrate() -> dict[str, Any]:
                nonlocal integration_result
                integration_result = integrate_multiple_csv_datasets(
                    validated_inputs,
                    save_outputs=False,
                    **dict(integration_options or {}),
                )
                if not isinstance(integration_result.get("data"), pd.DataFrame):
                    raise PipelineStageError(
                        "CSV integration did not return an integrated DataFrame."
                    )
                return integration_result

            integration_result = _run_stage(
                name="integration",
                operation=integrate,
                record_count=lambda result: len(result["data"]),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
                input_record_count=sum(
                    len(item["valid_dataframe"])
                    for item in validation_results.values()
                ),
            )
            integrated_data = integration_result["data"]
            _write_frame(
                integrated_data,
                run_artifacts / "integrated",
                "integrated.csv",
            )

            def load() -> dict[str, Any]:
                suggestions = list(unique_key_columns)
                if not set(suggestions).issubset(integrated_data.columns):
                    raise KeyError("A selected database key is absent after integration.")
                if integrated_data[suggestions].isna().any(axis=None):
                    raise ValueError("Database key values cannot contain nulls.")
                if integrated_data.duplicated(subset=suggestions).any():
                    raise ValueError(
                        "Selected database key columns are not unique after integration."
                    )
                result = load_validated_csv_dataframe(
                    integrated_data,
                    connection_config=db_config,
                    table_name=table_name,
                    unique_key_columns=suggestions,
                    duplicate_strategy=duplicate_strategy,
                    dataset_name=dataset_name,
                    source_filename=", ".join(dataset_names),
                    batch_size=batch_size,
                    connect_factory=connect_factory,
                )
                execution["database_load"] = result
                execution["inserted_rows"] = result.get("inserted_rows", 0)
                execution["updated_rows"] = result.get("updated_rows", 0)
                execution["skipped_rows"] = result.get("skipped_rows", 0)
                execution["failed_rows"] = result.get("failed_rows", 0)
                return result

            load_result = _run_stage(
                name="database_loading",
                operation=load,
                record_count=lambda result: int(result.get("processed_rows", 0)),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=db_config,
                stage_records=stage_records,
                input_record_count=len(integrated_data),
            )
            execution.update(
                {
                    "execution_status": load_result["execution_status"],
                    "inserted_rows": load_result.get("inserted_rows", 0),
                    "updated_rows": load_result.get("updated_rows", 0),
                    "skipped_rows": load_result.get("skipped_rows", 0),
                    "failed_rows": load_result.get("failed_rows", 0),
                    "integration_report": integration_result.get(
                        "integration_report", {}
                    ),
                    "integrated_data": integrated_data,
                }
            )
            if load_result.get("execution_status") != "success":
                execution["execution_status"] = "failed"
                execution["error_message"] = load_result.get(
                    "error_message", "Database loading failed."
                )
    except PipelineAlreadyRunning as error:
        execution["execution_status"] = "skipped"
        execution["error_message"] = str(error)
    except Exception as error:
        execution["execution_status"] = "failed"
        execution["error_message"] = _safe_error(error, db_config)
        execution["failed_rows"] = max(
            0,
            execution["processed_rows"]
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
        execution["stage_records"] = stage_records
        log_path = log_root / f"{run_id}.json"
        try:
            _atomic_json_write(
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
            execution["notification_status"] = "failed"
            smtp_password = str((smtp_config or {}).get("password") or "")
            notification_error = str(error)
            if smtp_password:
                notification_error = notification_error.replace(
                    smtp_password, "[redacted]"
                )
            mysql_password = str(db_config.get("password") or "")
            execution["notification_error"] = (
                notification_error.replace(mysql_password, "[redacted]")
                if mysql_password
                else notification_error
            )
        try:
            _atomic_json_write(
                Path(execution.get("log_path", log_root / f"{run_id}.json")),
                {
                    key: value
                    for key, value in execution.items()
                    if key != "integrated_data"
                },
            )
        except OSError:
            pass
    return execution


def get_csv_execution_history(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Load recent structured CSV pipeline execution logs."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    root = Path(log_directory)
    if not root.exists():
        return []
    history = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read CSV execution log {path}: {error}") from error
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
    """Rerun a failed log entry using its unchanged source paths and safe options."""
    match = next(
        (
            item
            for item in get_csv_execution_history(log_directory, limit=10000)
            if item.get("execution_id") == execution_id
        ),
        None,
    )
    if match is None:
        raise KeyError(f"No CSV execution log found for {execution_id!r}.")
    if match.get("execution_status") != "failed":
        raise ValueError("Only failed CSV executions can be rerun.")
    options = dict(match.get("execution_options") or {})
    options.update(overrides)
    return run_csv_pipeline(
        match.get("source_paths", []),
        connection_config=connection_config,
        log_directory=log_directory,
        table_name=options.pop("table_name"),
        unique_key_columns=options.pop("unique_key_columns"),
        **options,
    )


def suggest_pipeline_key_columns(dataframe: pd.DataFrame) -> list[str]:
    """Expose the existing CSV loader's verified candidate-key suggestion logic."""
    if dataframe.empty:
        return []
    for column in dataframe.columns:
        values = dataframe[[column]]
        if not values.isna().any(axis=None) and not values.duplicated().any():
            return [str(column)]
    for size in range(2, len(dataframe.columns) + 1):
        candidate = list(dataframe.columns[:size])
        values = dataframe[candidate]
        if not values.isna().any(axis=None) and not values.duplicated().any():
            return [str(column) for column in candidate]
    return []
