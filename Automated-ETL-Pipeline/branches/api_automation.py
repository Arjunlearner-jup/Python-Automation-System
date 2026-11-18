"""End-to-end API ETL orchestration using the existing pipeline stages."""

from __future__ import annotations

import json
import os
import re
import smtplib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests as http

from ..cleaning.api_cleaner import clean_api_dataframe
from ..database.api_loader import (
    get_connection_config_from_environment,
    load_validated_api_dataframe,
    suggest_api_unique_key_columns,
)
from ..ingestion.api_extractor import extract_api_data
from ..integration.api_integrator import integrate_multiple_api_datasets
from ..profiling.api_profiler import profile_api
from ..validation.api_validator import validate_api_dataframe
from .api_monitor import (
    get_api_execution_log,
    notify_api_execution,
    sanitize_for_log,
    save_api_execution_log,
    send_api_email_notification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "data" / "logs" / "api"
DEFAULT_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "automation" / "api"
)
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "automation" / "api"
_PROCESS_LOCK = threading.Lock()
_RETRYABLE_HTTP_STATUSES = {408, 425, 429}


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another process or thread owns the API pipeline lock."""


class PipelineStageError(RuntimeError):
    """Raised internally when an API ETL stage fails."""


class RetryableAPIError(RuntimeError):
    """A temporary API failure eligible for retry."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


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
    if isinstance(value, float) and pd.isna(value):
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


def _atomic_json_write(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str)
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
    """Acquire a non-blocking thread and cross-process pipeline lock."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise PipelineAlreadyRunning("An API pipeline execution is already running.")
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
                "An API pipeline execution is already running."
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


def _retry_after_seconds(value: Any, *, now: datetime | None = None) -> float | None:
    """Parse Retry-After as either delta seconds or an HTTP date."""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - (now or _utcnow())).total_seconds())


def _is_retryable_status(status_code: Any) -> bool:
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return False
    return status in _RETRYABLE_HTTP_STATUSES or 500 <= status <= 599


class _CaptureSession:
    """Adapt requests so raw attempts and Retry-After survive extraction."""

    def __init__(self, *, delegate: Any = None, attempts: list[dict[str, Any]]):
        self._delegate = delegate
        self._attempts = attempts
        self.last_response: Any = None
        self.last_exception: BaseException | None = None
        self.retry_after: float | None = None

    def get(self, url: str, **kwargs: Any) -> Any:
        try:
            response = (
                self._delegate.get(url, **kwargs)
                if self._delegate is not None
                else http.get(url, **kwargs)
            )
        except http.RequestException as error:
            self.last_exception = error
            self._attempts.append(
                {
                    "url": url,
                    "parameters": kwargs.get("params"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timestamp": _utcnow().isoformat(),
                }
            )
            raise

        self.last_response = response
        status = getattr(response, "status_code", None)
        headers = getattr(response, "headers", {}) or {}
        self.retry_after = _retry_after_seconds(
            headers.get("Retry-After") if isinstance(headers, Mapping) else None
        )
        try:
            body = response.text
        except (AttributeError, http.RequestException):
            body = None
        self._attempts.append(
            {
                "url": url,
                "parameters": kwargs.get("params"),
                "status_code": status,
                "retry_after": self.retry_after,
                "headers": {
                    str(key): str(value)
                    for key, value in headers.items()
                    if str(key).lower()
                    in {
                        "content-type",
                        "retry-after",
                        "date",
                        "x-ratelimit-limit",
                        "x-ratelimit-remaining",
                        "x-ratelimit-reset",
                    }
                },
                "body": body,
                "timestamp": _utcnow().isoformat(),
            }
        )
        return response


def _error_is_retryable(
    metadata: Mapping[str, Any], session: _CaptureSession
) -> tuple[bool, float | None]:
    status = metadata.get("http_status_code")
    if _is_retryable_status(status):
        return True, session.retry_after
    if status is not None:
        return False, None
    if isinstance(session.last_exception, (http.Timeout, http.ConnectionError)):
        return True, session.retry_after
    error = str(metadata.get("error", "")).lower()
    if any(
        marker in error
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporary failure",
            "name resolution",
        )
    ):
        return True, session.retry_after
    match = re.search(r"\b(?:http(?:error)?\s*)?([45]\d\d)\b", error)
    if match:
        return _is_retryable_status(int(match.group(1))), session.retry_after
    return False, None


def _run_stage(
    *,
    name: str,
    operation: Callable[[], Any],
    record_count: Callable[[Any], int],
    input_records: int,
    retry_attempts: int,
    retry_on: tuple[type[BaseException], ...],
    connection_config: Mapping[str, Any],
    stage_records: list[dict[str, Any]],
    retry_backoff_seconds: float,
    max_retry_backoff_seconds: float,
) -> Any:
    started = _utcnow()
    entry: dict[str, Any] = {
        "stage": name,
        "status": "running",
        "started_at": started.isoformat(),
        "attempts": 0,
        "records_processed": input_records,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
    }
    stage_records.append(entry)

    def finish(status: str) -> None:
        ended = _utcnow()
        entry["ended_at"] = ended.isoformat()
        entry["duration_seconds"] = max(0.0, (ended - started).total_seconds())
        entry["status"] = status

    last_error: BaseException | None = None
    for attempt in range(retry_attempts + 1):
        entry["attempts"] = attempt + 1
        try:
            result = operation()
            if isinstance(result, Mapping):
                for count_key in (
                    "inserted_rows",
                    "updated_rows",
                    "skipped_rows",
                    "failed_rows",
                ):
                    entry[count_key] = int(result.get(count_key, 0) or 0)
                if result.get("execution_status") in {"failed", "error"}:
                    message = result.get(
                        "error_message", "Stage returned an unsuccessful status."
                    )
                    entry["error_message"] = message
                    if result.get("error_type") in {
                        "TimeoutError",
                        "ConnectionError",
                        "OperationalError",
                        "InterfaceError",
                    }:
                        raise TimeoutError(message)
                    raise PipelineStageError(message)
            entry["records_processed"] = record_count(result)
            finish("success")
            return result
        except Exception as error:
            last_error = error
            if attempt >= retry_attempts or not isinstance(error, retry_on):
                break
            requested_delay = (
                error.retry_after if isinstance(error, RetryableAPIError) else None
            )
            delay = (
                requested_delay
                if requested_delay is not None
                else min(
                    retry_backoff_seconds * (2**attempt),
                    max_retry_backoff_seconds,
                )
            )
            time.sleep(max(0.0, delay))

    assert last_error is not None
    finish("failed")
    entry["error_message"] = _safe_error(last_error, connection_config)
    entry["error_type"] = type(last_error).__name__
    if not entry.get("failed_rows"):
        entry["failed_rows"] = input_records
    raise PipelineStageError(
        f"{name} failed: {entry['error_message']}"
    ) from last_error


def _write_dataframe(dataframe: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(
        destination.with_suffix(".json"),
        dataframe.to_dict(orient="records"),
    )
    dataframe.to_csv(destination.with_suffix(".csv"), index=False)


def _request_list(
    api_requests: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(api_requests, Mapping):
        requests_list = [dict(api_requests)]
    else:
        requests_list = [dict(item) for item in api_requests]
    if not requests_list:
        raise ValueError("Provide at least one API request parameter mapping.")
    for index, parameters in enumerate(requests_list):
        if not parameters:
            raise ValueError(f"API request {index + 1} has no parameters.")
    return requests_list


def _dataset_identifier(metadata: Mapping[str, Any], index: int) -> str:
    components = [
        str(metadata.get("database_id") or f"request_{index + 1}")
    ]
    for key in ("indicator_id", "country_reference_area"):
        if metadata.get(key) is not None:
            components.append(str(metadata[key]))
    return "_".join(components)


def _combined_api_metadata(
    extractions: Sequence[Mapping[str, Any]],
    identifiers: Sequence[str],
    integration_result: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = [dict(item["metadata"]) for item in extractions]
    if len(metadata) == 1:
        return metadata[0]
    return {
        "provider": metadata[0].get("provider"),
        "api_name": metadata[0].get("api_name"),
        "api_base_url": metadata[0].get("api_base_url"),
        "api_endpoint": metadata[0].get("api_endpoint"),
        "http_method": metadata[0].get("http_method"),
        "request_timestamp": max(
            (str(item.get("request_timestamp", "")) for item in metadata),
            default=None,
        ),
        "http_status_code": metadata[-1].get("http_status_code"),
        "api_response_record_count": sum(
            int(item.get("api_response_record_count") or 0) for item in metadata
        ),
        "records_retrieved": len(integration_result["data"].index),
        "pagination": {
            "sources": [item.get("pagination", {}) for item in metadata]
        },
        "source_metadata": [
            {"dataset_id": identifier, **item}
            for identifier, item in zip(identifiers, metadata)
        ],
        "integration_report": integration_result.get("integration_report", {}),
        "status": "success",
    }


def send_email_notification(
    execution: Mapping[str, Any],
    *,
    smtp_config: Mapping[str, Any] | None = None,
) -> None:
    """Compatibility wrapper for the shared API monitoring notification."""
    send_api_email_notification(execution, smtp_config=smtp_config)


def run_api_pipeline(
    api_requests: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    table_name: str,
    unique_key_columns: Sequence[str] | None = None,
    connection_config: Mapping[str, Any] | None = None,
    duplicate_strategy: str = "skip",
    batch_size: int = 1000,
    extraction_options: Mapping[str, Any] | None = None,
    cleaning_options: Mapping[str, Any] | None = None,
    validation_options: Mapping[str, Any] | None = None,
    integration_options: Mapping[str, Any] | None = None,
    retry_attempts: int = 2,
    retry_on: tuple[type[BaseException], ...] = (
        RetryableAPIError,
        http.Timeout,
        http.ConnectionError,
        OSError,
        TimeoutError,
        ConnectionError,
    ),
    retry_backoff_seconds: float = 1.0,
    max_retry_backoff_seconds: float = 30.0,
    notify: bool = False,
    smtp_config: Mapping[str, Any] | None = None,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run configured API requests through all six ETL stages.

    Pass one parameter mapping for a single Data360 query or a sequence of
    mappings to retrieve and integrate multiple queries. API retries are
    restricted to temporary transport errors, HTTP 408/425/429 and HTTP 5xx.
    """
    request_parameters = _request_list(api_requests)
    if retry_attempts < 0:
        raise ValueError("retry_attempts cannot be negative.")
    if retry_backoff_seconds < 0 or max_retry_backoff_seconds < 0:
        raise ValueError("Retry backoff values cannot be negative.")
    config = dict(connection_config or get_connection_config_from_environment())
    extractor_options = dict(extraction_options or {})
    allowed_options = {"specification_url", "timeout", "max_records", "page_size"}
    unknown_options = set(extractor_options) - allowed_options
    if unknown_options:
        raise ValueError(
            "Unsupported extraction option(s): "
            + ", ".join(sorted(unknown_options))
        )

    run_id = str(uuid.uuid4())
    started = _utcnow()
    log_root = Path(log_directory)
    run_artifacts = Path(artifact_directory) / run_id
    run_raw = Path(raw_directory) / run_id
    dataset_name = (
        str(request_parameters[0].get("DATABASE_ID") or "API dataset")
        if len(request_parameters) == 1
        else f"{len(request_parameters)} API requests"
    )
    stage_records: list[dict[str, Any]] = []
    http_attempts: list[dict[str, Any]] = []
    execution: dict[str, Any] = {
        "execution_id": run_id,
        "dataset_name": dataset_name,
        "source_endpoint": None,
        "request_parameters": _json_safe(request_parameters),
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
        "raw_directory": str(run_raw),
        "execution_options": {
            "table_name": table_name,
            "unique_key_columns": list(unique_key_columns or []),
            "duplicate_strategy": duplicate_strategy,
            "batch_size": batch_size,
            "extraction_options": extractor_options,
            "cleaning_options": dict(cleaning_options or {}),
            "validation_options": dict(validation_options or {}),
            "integration_options": dict(integration_options or {}),
            "retry_attempts": retry_attempts,
            "retry_backoff_seconds": retry_backoff_seconds,
            "max_retry_backoff_seconds": max_retry_backoff_seconds,
            "notify": notify,
        },
    }

    try:
        with _execution_lock(log_root / ".api_pipeline.lock"):
            extractions: list[dict[str, Any]] = []
            sessions: list[_CaptureSession] = []

            def ingest() -> list[dict[str, Any]]:
                nonlocal extractions
                extractions = []
                sessions.clear()
                execution["api_metadata"] = []
                for parameters in request_parameters:
                    capture = _CaptureSession(attempts=http_attempts)
                    sessions.append(capture)
                    result = extract_api_data(
                        parameters,
                        raw_directory=run_raw,
                        session=capture,
                        **extractor_options,
                    )
                    metadata = result.get("metadata", {})
                    archived_metadata = metadata.get("raw_file_path")
                    if archived_metadata:
                        archived_metadata_path = (
                            Path(archived_metadata) / "metadata.json"
                        )
                        if archived_metadata_path.is_file():
                            _atomic_json_write(
                                archived_metadata_path,
                                sanitize_for_log(metadata),
                            )
                    execution["api_metadata"].append(
                        {
                            "dataset_id": _dataset_identifier(
                                metadata, len(execution["api_metadata"])
                            ),
                            **_json_safe(metadata),
                        }
                    )
                    execution["records_retrieved"] = sum(
                        int(item.get("records_retrieved") or 0)
                        for item in execution["api_metadata"]
                    )
                    execution["api_provider"] = metadata.get("provider")
                    execution["api_endpoint"] = metadata.get("api_endpoint")
                    if metadata.get("status") != "success" or not isinstance(
                        result.get("data"), pd.DataFrame
                    ):
                        retryable, retry_after = _error_is_retryable(metadata, capture)
                        message = str(
                            metadata.get("error")
                            or "API ingestion returned no observation DataFrame."
                        )
                        if retryable:
                            raise RetryableAPIError(message, retry_after)
                        status = metadata.get("http_status_code")
                        if status in {401, 403}:
                            raise PipelineStageError(
                                f"API authentication failed (HTTP {status}): {message}"
                            )
                        raise PipelineStageError(f"API ingestion failed: {message}")
                    extractions.append(result)
                return extractions

            extractions = _run_stage(
                name="ingestion",
                operation=ingest,
                record_count=lambda results: sum(
                    len(item["data"].index) for item in results
                ),
                input_records=0,
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            identifiers = [
                _dataset_identifier(item["metadata"], index)
                for index, item in enumerate(extractions)
            ]
            if len(set(identifiers)) != len(identifiers):
                identifiers = [
                    f"{identifier}_{index + 1}"
                    for index, identifier in enumerate(identifiers)
                ]
            names_by_id = {
                identifier: str(
                    item["metadata"].get("database_id")
                    or item["metadata"].get("indicator_id")
                    or identifier
                )
                for identifier, item in zip(identifiers, extractions)
            }
            endpoints = list(
                dict.fromkeys(
                    str(item["metadata"].get("api_endpoint") or "")
                    for item in extractions
                )
            )
            execution["source_endpoint"] = ", ".join(
                endpoint for endpoint in endpoints if endpoint
            )
            execution["dataset_name"] = (
                names_by_id[identifiers[0]]
                if len(identifiers) == 1
                else f"{len(identifiers)} API datasets"
            )
            execution["processed_rows"] = sum(
                len(item["data"].index) for item in extractions
            )
            execution["api_metadata"] = [
                {"dataset_id": identifier, **_json_safe(item["metadata"])}
                for identifier, item in zip(identifiers, extractions)
            ]
            execution["api_provider"] = extractions[0]["metadata"].get("provider")
            execution["api_endpoint"] = execution["source_endpoint"]
            execution["database_ids"] = [
                item["metadata"].get("database_id")
                for item in extractions
                if item["metadata"].get("database_id") is not None
            ]
            execution["requested_indicators"] = [
                item["metadata"].get("indicator_id")
                for item in extractions
                if item["metadata"].get("indicator_id") is not None
            ]
            execution["country_reference_areas"] = [
                item["metadata"].get("country_reference_area")
                for item in extractions
                if item["metadata"].get("country_reference_area") is not None
            ]
            execution["records_retrieved"] = sum(
                int(item["metadata"].get("records_retrieved") or 0)
                for item in extractions
            )
            execution["http_status_codes"] = [
                item["metadata"].get("http_status_code")
                for item in extractions
                if item["metadata"].get("http_status_code") is not None
            ]
            execution["pagination"] = [
                item["metadata"].get("pagination", {}) for item in extractions
            ]
            for identifier, item in zip(identifiers, extractions):
                source_stem = _safe_stem(identifier)
                _write_dataframe(
                    item["data"], run_artifacts / "ingested" / f"{source_stem}.json"
                )
                _atomic_json_write(
                    run_artifacts / "ingested" / f"{source_stem}_metadata.json",
                    item["metadata"],
                )
                _atomic_json_write(
                    run_artifacts / "ingested" / f"{source_stem}_response.json",
                    item.get("api_response", {}),
                )
                _atomic_json_write(
                    run_artifacts / "ingested" / f"{source_stem}_pages.json",
                    item.get("page_responses", []),
                )

            profiles: dict[str, dict[str, Any]] = {}

            def profile() -> dict[str, dict[str, Any]]:
                nonlocal profiles
                profiles = {
                    identifier: profile_api(
                        item["data"],
                        api_metadata=item["metadata"],
                        api_response=item.get("api_response"),
                    )
                    for identifier, item in zip(identifiers, extractions)
                }
                return profiles

            profiles = _run_stage(
                name="profiling",
                operation=profile,
                record_count=lambda values: sum(
                    int(value["dimensions"].get("number_of_records", 0))
                    for value in values.values()
                ),
                input_records=execution["processed_rows"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            execution["profiling_reports"] = {
                key: value["quality_summary"] for key, value in profiles.items()
            }

            cleaned: dict[str, dict[str, Any]] = {}

            def clean() -> dict[str, dict[str, Any]]:
                nonlocal cleaned
                cleaned = {
                    key: clean_api_dataframe(
                        profiles[key]["data"],
                        api_metadata=profiles[key]["api_metadata"],
                        dataset_name=names_by_id[key],
                        api_response=profiles[key]["api_response"],
                        **dict(cleaning_options or {}),
                    )
                    for key in identifiers
                }
                if any(
                    item["report"].get("cleaning_status") == "error"
                    for item in cleaned.values()
                ):
                    raise PipelineStageError("API cleaning returned an error status.")
                return cleaned

            cleaned = _run_stage(
                name="cleaning",
                operation=clean,
                record_count=lambda values: sum(
                    len(item["data"].index) for item in values.values()
                ),
                input_records=execution["processed_rows"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            execution["cleaning_reports"] = {
                key: item["report"] for key, item in cleaned.items()
            }
            for key, item in cleaned.items():
                _write_dataframe(
                    item["data"], run_artifacts / "cleaned" / f"{_safe_stem(key)}.json"
                )

            validation_results: dict[str, dict[str, Any]] = {}

            def validate() -> dict[str, dict[str, Any]]:
                nonlocal validation_results
                validation_results = {
                    key: validate_api_dataframe(
                        cleaned[key],
                        dataset_name=names_by_id[key],
                        **dict(validation_options or {}),
                    )
                    for key in identifiers
                }
                failed = [
                    key
                    for key, result in validation_results.items()
                    if result.get("validation_status")
                    not in {"passed", "passed_with_warnings"}
                ]
                if failed:
                    raise PipelineStageError(
                        "API data validation failed for: "
                        + ", ".join(names_by_id[key] for key in failed)
                    )
                return validation_results

            validation_results = _run_stage(
                name="validation",
                operation=validate,
                record_count=lambda values: sum(
                    int(item["validation_report"].get("valid_records") or 0)
                    for item in values.values()
                ),
                input_records=sum(len(item["data"].index) for item in cleaned.values()),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            execution["validation_reports"] = {
                key: item["validation_report"]
                for key, item in validation_results.items()
            }
            for key, item in validation_results.items():
                _write_dataframe(
                    item["valid_dataframe"],
                    run_artifacts / "validated" / f"{_safe_stem(key)}_valid.json",
                )
                _write_dataframe(
                    item["invalid_dataframe"],
                    run_artifacts / "validated" / f"{_safe_stem(key)}_invalid.json",
                )

            integration_result: dict[str, Any] = {}

            def integrate() -> dict[str, Any]:
                nonlocal integration_result
                datasets = {
                    key: {
                        "validation_result": validation_results[key],
                        "api_metadata": extractions[index]["metadata"],
                        "source": names_by_id[key],
                    }
                    for index, key in enumerate(identifiers)
                }
                integration_result = integrate_multiple_api_datasets(
                    datasets,
                    save_outputs=False,
                    **dict(integration_options or {}),
                )
                if not isinstance(integration_result.get("data"), pd.DataFrame):
                    raise PipelineStageError(
                        "API integration did not return an integrated DataFrame."
                    )
                return integration_result

            integration_result = _run_stage(
                name="integration",
                operation=integrate,
                record_count=lambda result: len(result["data"].index),
                input_records=sum(
                    len(item["valid_dataframe"].index)
                    for item in validation_results.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            integrated_data = integration_result["data"]
            _write_dataframe(
                integrated_data, run_artifacts / "integrated" / "integrated.json"
            )
            api_metadata = _combined_api_metadata(
                extractions, identifiers, integration_result
            )
            _atomic_json_write(
                run_artifacts / "integrated" / "api_metadata.json", api_metadata
            )
            execution["integration_report"] = integration_result.get(
                "integration_report", {}
            )
            execution["integrated_metadata"] = api_metadata

            integrated_validation: dict[str, Any] = {}

            def validate_integrated() -> dict[str, Any]:
                nonlocal integrated_validation
                integrated_validation = validate_api_dataframe(
                    integrated_data,
                    api_metadata=api_metadata,
                    dataset_name=dataset_name,
                    **dict(validation_options or {}),
                )
                if integrated_validation.get("validation_status") not in {
                    "passed",
                    "passed_with_warnings",
                }:
                    raise PipelineStageError(
                        "Integrated API data did not pass Stage 4 validation."
                    )
                return integrated_validation

            _run_stage(
                name="integrated_validation",
                operation=validate_integrated,
                record_count=lambda result: int(
                    result["validation_report"].get("valid_records") or 0
                ),
                input_records=len(integrated_data.index),
                retry_attempts=0,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            integrated_valid_data = integrated_validation["valid_dataframe"]
            keys = list(
                unique_key_columns
                or suggest_api_unique_key_columns(integrated_valid_data)
            )
            execution["selected_unique_key_columns"] = keys
            _write_dataframe(
                integrated_valid_data,
                run_artifacts / "validated" / "integrated_valid.json",
            )

            def load() -> dict[str, Any]:
                result = load_validated_api_dataframe(
                    integrated_valid_data,
                    api_metadata=api_metadata,
                    connection_config=config,
                    table_name=table_name,
                    unique_key_columns=keys,
                    duplicate_strategy=duplicate_strategy,
                    dataset_name=dataset_name,
                    source_endpoint=execution["source_endpoint"],
                    batch_size=batch_size,
                    connect_factory=connect_factory,
                )
                execution["database_load"] = result
                for count in (
                    "inserted_rows",
                    "updated_rows",
                    "skipped_rows",
                    "failed_rows",
                ):
                    execution[count] = result.get(count, 0)
                return result

            load_result = _run_stage(
                name="database_loading",
                operation=load,
                record_count=lambda result: int(result.get("processed_rows", 0)),
                input_records=len(integrated_valid_data.index),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stage_records=stage_records,
                retry_backoff_seconds=retry_backoff_seconds,
                max_retry_backoff_seconds=max_retry_backoff_seconds,
            )
            execution["execution_status"] = load_result.get(
                "execution_status", "failed"
            )
            if execution["execution_status"] != "success":
                execution["error_message"] = load_result.get(
                    "error_message", "API database loading failed."
                )
            execution["integrated_data"] = integrated_data
    except PipelineAlreadyRunning as error:
        execution["execution_status"] = "skipped"
        execution["error_message"] = str(error)
    except Exception as error:
        execution["execution_status"] = "failed"
        execution["error_message"] = _safe_error(error, config)
        execution["error_type"] = type(error).__name__
        execution["failure_details"] = {
            "message": execution["error_message"],
            "stage": (
                stage_records[-1].get("stage")
                if stage_records and stage_records[-1].get("status") == "failed"
                else None
            ),
        }
        if stage_records and stage_records[-1].get("status") == "failed":
            execution["error_category"] = (
                "api_ingestion"
                if stage_records[-1].get("stage") == "ingestion"
                else "etl_stage"
            )
        if not execution.get("failed_rows"):
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
        if http_attempts:
            try:
                _atomic_json_write(
                    run_artifacts / "ingested" / "http_attempts.json",
                    sanitize_for_log(http_attempts),
                )
            except OSError as error:
                execution["raw_attempt_write_error"] = str(error)
        try:
            log_path = save_api_execution_log(
                execution,
                log_root,
                secrets=(
                    str(config.get("password") or ""),
                    str((smtp_config or {}).get("password") or ""),
                ),
            )
            execution["log_path"] = str(log_path)
        except OSError as error:
            execution["log_write_error"] = str(error)

    if notify:
        execution.update(
            notify_api_execution(
                execution,
                enabled=True,
                smtp_config=smtp_config,
            )
        )
        try:
            log_path = save_api_execution_log(
                execution,
                log_root,
                secrets=(
                    str(config.get("password") or ""),
                    str((smtp_config or {}).get("password") or ""),
                ),
            )
            execution["log_path"] = str(log_path)
        except OSError as error:
            execution["notification_log_write_error"] = str(error)
    return execution


def rerun_failed_api_execution(
    execution_id: str,
    *,
    connection_config: Mapping[str, Any] | None = None,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    **overrides: Any,
) -> dict[str, Any]:
    """Rerun a failed API execution using its recorded request and ETL options.

    The API is queried again, but the normal idempotent database duplicate
    handling and the pipeline execution lock remain active. Credentials are
    supplied at runtime and are never restored from an execution log.
    """
    record = get_api_execution_log(execution_id, log_directory)
    if record.get("execution_status") != "failed":
        raise ValueError("Only failed API executions can be rerun.")
    requests = record.get("request_parameters")
    if not requests:
        raise ValueError("The failed API execution has no recorded request parameters.")
    options = dict(record.get("execution_options") or {})
    options.update(overrides)
    if not options.get("table_name"):
        raise ValueError(
            "The failed API log does not contain its destination table; "
            "start a new run with the current settings instead."
        )
    return run_api_pipeline(
        requests,
        connection_config=connection_config,
        log_directory=log_directory,
        **options,
    )


_BACKGROUND_EXECUTOR: Any = None
_BACKGROUND_EXECUTOR_LOCK = threading.Lock()


def run_api_pipeline_background(*args: Any, **kwargs: Any):
    """Submit an independent background execution and return its Future."""
    global _BACKGROUND_EXECUTOR
    from concurrent.futures import ThreadPoolExecutor

    with _BACKGROUND_EXECUTOR_LOCK:
        if _BACKGROUND_EXECUTOR is None:
            _BACKGROUND_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="api-etl"
            )
        executor = _BACKGROUND_EXECUTOR
    return executor.submit(run_api_pipeline, *args, **kwargs)