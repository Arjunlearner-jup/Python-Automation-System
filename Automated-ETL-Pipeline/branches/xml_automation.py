"""End-to-end XML ETL orchestration with durable logs and safe retries."""

from __future__ import annotations

import json
import os
import smtplib
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..cleaning.xml_cleaner import clean_xml_data
from ..database.xml_loader import (
    get_connection_config_from_environment,
    load_validated_xml_data,
)
from ..ingestion.xml_extractor import extract_xml_file
from ..integration.xml_integrator import integrate_multiple_xml_datasets
from ..profiling.xml_profiler import profile_xml
from ..validation.xml_validator import validate_xml_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "data" / "logs" / "xml"
DEFAULT_ARTIFACT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "automation" / "xml"
_PROCESS_LOCK = threading.Lock()


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another process or thread owns the XML pipeline lock."""


class PipelineStageError(RuntimeError):
    """Raised internally when an XML ETL stage fails."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_error(error: BaseException, config: Mapping[str, Any]) -> str:
    message = str(error)
    password = str(config.get("password") or "")
    return message.replace(password, "[redacted]") if password else message


def _write_json(destination: Path, value: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
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


def _element_count(root: ET.Element) -> int:
    return sum(1 for _ in root.iter())


def _record_count(value: Any) -> int:
    if isinstance(value, ET.Element):
        return _element_count(value)
    if isinstance(value, pd.DataFrame):
        return len(value.index)
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return 0


@contextmanager
def _execution_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a non-blocking thread and cross-process execution lock."""
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise PipelineAlreadyRunning("An XML pipeline execution is already running.")
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
                "An XML pipeline execution is already running."
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
                message = result.get("error_message", "Stage returned a failed status.")
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


def _write_xml(
    destination: Path,
    root: ET.Element,
    *,
    namespaces: Mapping[str, str] | None = None,
    xml_declaration: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for prefix, uri in (namespaces or {}).items():
        try:
            ET.register_namespace(prefix, uri)
        except ValueError:
            continue
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=xml_declaration)
    ET.parse(destination)


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
        f"XML ETL {execution.get('execution_status', 'unknown')}: "
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


def run_xml_pipeline(
    sources: Sequence[str | Path],
    *,
    table_name: str,
    unique_key_field: str | None = None,
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
    """Run XML ingestion through database loading without mutating source files."""
    source_paths = [Path(source).expanduser().resolve() for source in sources]
    if not source_paths:
        raise ValueError("Provide at least one XML source path.")
    if retry_attempts < 0:
        raise ValueError("retry_attempts cannot be negative.")
    config = dict(connection_config or get_connection_config_from_environment())
    run_id = str(uuid.uuid4())
    start = _utcnow()
    names = [path.name for path in source_paths]
    dataset_name = names[0] if len(names) == 1 else f"{len(names)} XML files"
    log_root = Path(log_directory)
    artifacts = Path(artifact_directory) / run_id
    archive = Path(raw_archive_directory) if raw_archive_directory else artifacts / "originals"
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
            "unique_key_field": unique_key_field,
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
        with _execution_lock(log_root / ".xml_pipeline.lock"):
            extractions: list[dict[str, Any]] = []

            def ingest() -> list[dict[str, Any]]:
                nonlocal extractions
                extractions = [
                    extract_xml_file(source, raw_directory=archive)
                    for source in source_paths
                ]
                failed = [
                    item
                    for item in extractions
                    if item["metadata"].get("status") != "success"
                    or not isinstance(item.get("xml_tree"), ET.Element)
                ]
                if failed:
                    details = []
                    for item in failed:
                        error = item["metadata"].get("error", "XML parsing failed.")
                        item["metadata"]["error_category"] = "xml_parsing"
                        details.append(f"{item['metadata'].get('source')}: {error}")
                    raise PipelineStageError("; ".join(details))
                return extractions

            extractions = _run_stage(
                name="ingestion",
                operation=ingest,
                record_count=lambda items: sum(
                    _element_count(item["xml_tree"]) for item in items
                ),
                input_records=0,
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["processed_records"] = sum(
                _element_count(item["xml_tree"]) for item in extractions
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
                _write_xml(
                    artifacts / "ingested" / f"{_safe_stem(identifier)}.xml",
                    item["xml_tree"],
                    namespaces=item["metadata"].get("namespaces"),
                )

            profiles: dict[str, dict[str, Any]] = {}

            def profile() -> dict[str, dict[str, Any]]:
                nonlocal profiles
                profiles = {
                    identifier: profile_xml(
                        item["data"],
                        item["metadata"]["source"],
                        xml_metadata={
                            **item["metadata"],
                            "dataframes": item.get("dataframes", {}),
                        },
                    )
                    for identifier, item in zip(identifiers, extractions)
                }
                return profiles

            profiles = _run_stage(
                name="profiling",
                operation=profile,
                record_count=lambda result: sum(
                    int(item["dimensions"].get("number_of_records", 0))
                    for item in result.values()
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

            cleaned: dict[str, dict[str, Any]] = {}

            def clean() -> dict[str, dict[str, Any]]:
                nonlocal cleaned
                cleaned = {
                    key: clean_xml_data(
                        extraction,
                        names_by_id[key],
                        **dict(cleaning_options or {}),
                    )
                    for key, extraction in zip(identifiers, extractions)
                }
                if any(
                    result["report"].get("cleaning_status") == "error"
                    for result in cleaned.values()
                ):
                    raise PipelineStageError("XML cleaning returned an error status.")
                return cleaned

            cleaned = _run_stage(
                name="cleaning",
                operation=clean,
                record_count=lambda result: sum(
                    _element_count(item["xml_tree"]) for item in result.values()
                ),
                input_records=execution["processed_records"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["cleaning_reports"] = {
                key: item["report"] for key, item in cleaned.items()
            }
            for key, item in cleaned.items():
                _write_xml(
                    artifacts / "cleaned" / f"{_safe_stem(key)}.xml",
                    item["xml_tree"],
                    namespaces=item.get("namespaces"),
                    xml_declaration=item.get("xml_declaration", True),
                )

            validation_results: dict[str, dict[str, Any]] = {}

            def validate() -> dict[str, dict[str, Any]]:
                nonlocal validation_results
                validation_results = {
                    key: validate_xml_data(
                        item,
                        names_by_id[key],
                        **dict(validation_options or {}),
                    )
                    for key, item in cleaned.items()
                }
                failed = [
                    key
                    for key, result in validation_results.items()
                    if result["validation_status"]
                    not in {"passed", "passed_with_warnings"}
                    or not isinstance(result.get("valid_data"), ET.Element)
                ]
                if failed:
                    raise PipelineStageError(
                        "XML validation failed for: "
                        + ", ".join(names_by_id[key] for key in failed)
                    )
                return validation_results

            validation_results = _run_stage(
                name="validation",
                operation=validate,
                record_count=lambda result: sum(
                    int(item["validation_report"].get("valid_records") or 0)
                    for item in result.values()
                ),
                input_records=execution["processed_records"],
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["validation_reports"] = {
                key: item["validation_report"]
                for key, item in validation_results.items()
            }
            for key, item in validation_results.items():
                metadata = extractions[identifiers.index(key)]["metadata"]
                _write_xml(
                    artifacts / "validated" / f"{_safe_stem(key)}.xml",
                    item["valid_data"],
                    namespaces=item.get("namespaces") or metadata.get("namespaces"),
                    xml_declaration=item.get("xml_declaration", True),
                )
                if isinstance(item.get("invalid_data"), ET.Element):
                    _write_xml(
                        artifacts / "invalid" / f"{_safe_stem(key)}.xml",
                        item["invalid_data"],
                        namespaces=item.get("namespaces") or metadata.get("namespaces"),
                        xml_declaration=item.get("xml_declaration", True),
                    )

            integration_result: dict[str, Any] = {}

            def integrate() -> dict[str, Any]:
                nonlocal integration_result
                datasets = {
                    key: {
                        "validation_result": validation_results[key],
                        "source": names_by_id[key],
                        "metadata": extractions[index]["metadata"],
                        "namespaces": extractions[index]["metadata"].get("namespaces", {}),
                        "xml_declaration": validation_results[key].get(
                            "xml_declaration", True
                        ),
                    }
                    for index, key in enumerate(identifiers)
                }
                integration_result = integrate_multiple_xml_datasets(
                    datasets, save_outputs=False, **dict(integration_options or {})
                )
                if not isinstance(integration_result.get("data"), ET.Element):
                    raise PipelineStageError(
                        "XML integration did not return an XML document."
                    )
                return integration_result

            integration_result = _run_stage(
                name="integration",
                operation=integrate,
                record_count=lambda result: _element_count(result["data"]),
                input_records=sum(
                    int(item["validation_report"].get("valid_records") or 0)
                    for item in validation_results.values()
                ),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            integrated_root = integration_result["data"]
            integrated_namespaces = integration_result.get("namespaces") or {}
            integrated_declaration = integration_result.get("xml_declaration", True)
            _write_xml(
                artifacts / "integrated" / "integrated.xml",
                integrated_root,
                namespaces=integrated_namespaces,
                xml_declaration=integrated_declaration,
            )
            execution["integration_report"] = integration_result.get(
                "integration_report", {}
            )

            integrated_validation: dict[str, Any] = {}

            def validate_integrated() -> dict[str, Any]:
                nonlocal integrated_validation
                integrated_validation = validate_xml_data(
                    integrated_root,
                    dataset_name,
                    **dict(validation_options or {}),
                )
                if (
                    integrated_validation["validation_status"]
                    not in {"passed", "passed_with_warnings"}
                    or not isinstance(integrated_validation.get("valid_data"), ET.Element)
                ):
                    raise PipelineStageError(
                        "Integrated XML did not pass Stage 4 validation."
                    )
                return integrated_validation

            _run_stage(
                name="integrated_validation",
                operation=validate_integrated,
                record_count=lambda result: int(
                    result["validation_report"].get("valid_records") or 0
                ),
                input_records=_element_count(integrated_root),
                retry_attempts=0,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )

            def load() -> dict[str, Any]:
                result = load_validated_xml_data(
                    integrated_validation["valid_data"],
                    connection_config=config,
                    table_name=table_name,
                    unique_key_field=unique_key_field,
                    duplicate_strategy=duplicate_strategy,
                    dataset_name=dataset_name,
                    source_filename=", ".join(names),
                    namespaces=integrated_validation.get("namespaces")
                    or integrated_namespaces,
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
                input_records=_element_count(integrated_validation["valid_data"]),
                retry_attempts=retry_attempts,
                retry_on=retry_on,
                connection_config=config,
                stages=stages,
            )
            execution["execution_status"] = load_result.get(
                "execution_status", "failed"
            )
            if execution["execution_status"] != "success":
                execution["error_message"] = load_result.get(
                    "error_message", "XML database loading failed."
                )
            execution["integrated_xml"] = integrated_root
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
        if stages and stages[-1].get("status") == "failed":
            execution["error_category"] = (
                "xml_parsing"
                if stages[-1]["stage"] == "ingestion"
                else "xml_validation"
                if "validation" in stages[-1]["stage"]
                else "pipeline"
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
        execution["stage_records"] = stages
        log_path = log_root / f"{run_id}.json"
        try:
            _write_json(
                log_path,
                {
                    key: value
                    for key, value in execution.items()
                    if key != "integrated_xml"
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
                    if key != "integrated_xml"
                },
            )
        except OSError as error:
            execution["notification_log_write_error"] = str(error)
    return execution


def get_xml_execution_history(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Load recent structured XML pipeline execution logs."""
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
            raise ValueError(f"Unable to read XML execution log {path}: {error}") from error
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
    """Rerun a failed XML execution from its original, unchanged source files."""
    record = next(
        (
            item
            for item in get_xml_execution_history(log_directory, limit=10000)
            if item.get("execution_id") == execution_id
        ),
        None,
    )
    if record is None:
        raise KeyError(f"No XML execution log found for {execution_id!r}.")
    if record.get("execution_status") != "failed":
        raise ValueError("Only failed XML executions can be rerun.")
    options = dict(record.get("execution_options") or {})
    options.update(overrides)
    return run_xml_pipeline(
        record.get("source_paths", []),
        connection_config=connection_config,
        log_directory=log_directory,
        table_name=options.pop("table_name"),
        unique_key_field=options.pop("unique_key_field"),
        **options,
    )
