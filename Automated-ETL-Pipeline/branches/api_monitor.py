"""Structured execution-log and notification helpers for API ETL runs."""

from __future__ import annotations

import json
import os
import re
import smtplib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "data" / "logs" / "api"
_SENSITIVE_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "authorization",
    "credential",
    "client_secret",
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)([?&;\s\"'](?:api[_-]?key|access[_-]?token|token|password|"
    r"client[_-]?secret|authorization)=)([^&;\s\"']+)"
)


def _is_sensitive_key(key: Any) -> bool:
    lowered = str(key).strip().lower().replace("-", "_")
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _known_secrets() -> tuple[str, ...]:
    return tuple(
        value
        for key in (
            "SMTP_PASSWORD",
            "API_KEY",
            "API_TOKEN",
            "MYSQL_PASSWORD",
            "MYSQL_PWD",
        )
        if (value := os.getenv(key))
    )


def _redact_string(value: str, secrets: Sequence[str] = ()) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    result = _SENSITIVE_ASSIGNMENT.sub(r"\1[REDACTED]", result)
    if "://" in result:
        try:
            parsed = urlsplit(result)
            query = [
                (key, "[REDACTED]" if _is_sensitive_key(key) else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ]
            result = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    urlencode(query),
                    parsed.fragment,
                )
            )
        except ValueError:
            pass
    return result


def sanitize_for_log(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    """Recursively redact credentials and sensitive query parameters."""
    known_secrets = tuple(dict.fromkeys((*_known_secrets(), *secrets)))
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_key(key)
                else sanitize_for_log(child, secrets=known_secrets)
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_log(item, secrets=known_secrets) for item in value]
    if isinstance(value, str):
        return _redact_string(value, known_secrets)
    if isinstance(value, Path):
        return _redact_string(str(value), known_secrets)
    return value


def _metadata_items(execution: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    metadata = execution.get("api_metadata")
    if not isinstance(metadata, list):
        metadata = [execution.get("integrated_metadata", {})]
    return [item for item in metadata if isinstance(item, Mapping)]


def _sensitive_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_sensitive_key(key) and child is not None:
                if isinstance(child, (str, int, float)):
                    found.append(str(child))
            else:
                found.extend(_sensitive_values(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.extend(_sensitive_values(child))
    return found


def build_api_execution_log(
    execution: Mapping[str, Any],
    *,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize an execution into a stable API monitoring log schema."""
    metadata_items = _metadata_items(execution)
    first_metadata = metadata_items[0] if metadata_items else {}
    retrieved = execution.get("records_retrieved")
    if retrieved is None:
        retrieved = sum(
            int(item.get("records_retrieved") or 0) for item in metadata_items
        )
    http_status_codes = execution.get("http_status_codes")
    if http_status_codes is None:
        http_status_codes = [
            item.get("http_status_code")
            for item in metadata_items
            if item.get("http_status_code") is not None
        ]
    database_ids = execution.get("database_ids")
    if database_ids is None:
        database_ids = [
            item.get("database_id")
            for item in metadata_items
            if item.get("database_id") is not None
        ]
    indicators = execution.get("requested_indicators")
    if indicators is None:
        indicators = [
            item.get("indicator_id")
            for item in metadata_items
            if item.get("indicator_id") is not None
        ]
    reference_areas = execution.get("country_reference_areas")
    if reference_areas is None:
        reference_areas = [
            item.get("country_reference_area")
            for item in metadata_items
            if item.get("country_reference_area") is not None
        ]
    request_parameters = execution.get("request_parameters")
    if request_parameters is None:
        request_parameters = [
            item.get("request_parameters", {}) for item in metadata_items
        ]
    endpoint = execution.get("api_endpoint") or execution.get("source_endpoint")
    if not endpoint:
        endpoints = list(
            dict.fromkeys(
                str(item.get("api_endpoint"))
                for item in metadata_items
                if item.get("api_endpoint")
            )
        )
        endpoint = ", ".join(endpoints) or None
    stages = execution.get("stage_records", [])
    failed_stage = next(
        (
            stage.get("stage")
            for stage in stages
            if isinstance(stage, Mapping) and stage.get("status") == "failed"
        ),
        None,
    )
    record = {
        "execution_id": execution.get("execution_id"),
        "api_provider": execution.get("api_provider") or first_metadata.get("provider"),
        "api_endpoint": endpoint,
        "dataset_name": execution.get("dataset_name"),
        "database_ids": database_ids,
        "requested_indicators": indicators,
        "country_reference_areas": reference_areas,
        "request_parameters": request_parameters,
        "start_time": execution.get("start_time"),
        "end_time": execution.get("end_time"),
        "duration_seconds": execution.get("duration_seconds"),
        "execution_status": execution.get("execution_status"),
        "stage_records": stages,
        "http_status_codes": http_status_codes,
        "api_response_record_count": sum(
            int(item.get("api_response_record_count") or 0)
            for item in metadata_items
        ),
        "records_retrieved": int(retrieved or 0),
        "pagination": execution.get(
            "pagination", [item.get("pagination", {}) for item in metadata_items]
        ),
        "processed_rows": execution.get("processed_rows", 0),
        "inserted_rows": execution.get("inserted_rows", 0),
        "updated_rows": execution.get("updated_rows", 0),
        "skipped_rows": execution.get("skipped_rows", 0),
        "failed_rows": execution.get("failed_rows", 0),
        "error_message": execution.get("error_message"),
        "error_type": execution.get("error_type"),
        "error_category": execution.get("error_category"),
        "failure_stage": execution.get("failure_stage") or failed_stage,
        "failure_details": execution.get("failure_details"),
        "execution_options": execution.get("execution_options", {}),
        "artifact_directory": execution.get("artifact_directory"),
        "log_path": execution.get("log_path"),
        "notification_status": execution.get("notification_status"),
        "notification_error": execution.get("notification_error"),
    }
    return sanitize_for_log(
        record,
        secrets=tuple(secrets) + tuple(_sensitive_values(execution)),
    )


def save_api_execution_log(
    execution: Mapping[str, Any],
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    *,
    secrets: Sequence[str] = (),
) -> Path:
    """Atomically save a normalized, redacted execution record."""
    record = build_api_execution_log(execution, secrets=secrets)
    execution_id = record.get("execution_id")
    if not execution_id or Path(str(execution_id)).name != str(execution_id):
        raise ValueError("execution must contain a valid execution_id.")
    root = Path(log_directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{execution_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def get_api_execution_history(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Read recent API execution logs, surfacing malformed log files."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    root = Path(log_directory)
    if not root.exists():
        return []
    records = []
    for path in sorted(
        root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read API execution log {path}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"API execution log is not a JSON object: {path}")
        record["log_path"] = str(path)
        records.append(sanitize_for_log(record))
        if len(records) >= limit:
            break
    return records


def get_api_execution_log(
    execution_id: str,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
) -> dict[str, Any]:
    """Load one execution log by ID and redact any legacy credentials."""
    if not execution_id or Path(str(execution_id)).name != str(execution_id):
        raise ValueError("execution_id must be a simple identifier.")
    path = Path(log_directory) / f"{execution_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"API execution log not found: {execution_id}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"API execution log is invalid JSON: {path}") from error
    if not isinstance(record, dict):
        raise ValueError(f"API execution log is not a JSON object: {path}")
    return sanitize_for_log({**record, "log_path": str(path)})


def get_failed_api_executions(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Return recent failed API pipeline executions."""
    return [
        record
        for record in get_api_execution_history(log_directory, limit=limit)
        if record.get("execution_status") == "failed"
    ]


def get_api_log_text(
    execution_id: str,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
) -> str:
    """Return a formatted and redacted JSON log for viewing or download."""
    return json.dumps(
        get_api_execution_log(execution_id, log_directory),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _smtp_settings(
    supplied: Mapping[str, Any] | None,
    streamlit_secrets: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = dict(supplied or {})
    secrets = streamlit_secrets
    if secrets is None:
        try:
            import streamlit as st

            secrets = dict(st.secrets)
        except Exception:
            secrets = {}
    nested = secrets.get("smtp", {}) if isinstance(secrets, Mapping) else {}
    nested = nested if isinstance(nested, Mapping) else {}

    def setting(key: str, env_name: str, default: Any = "") -> Any:
        return (
            supplied.get(key)
            or os.getenv(env_name)
            or nested.get(key)
            or (secrets.get(env_name) if isinstance(secrets, Mapping) else None)
            or default
        )

    recipients = setting("to", "SMTP_TO")
    if isinstance(recipients, str):
        recipients = [part.strip() for part in recipients.split(",") if part.strip()]
    elif recipients:
        recipients = list(recipients)
    else:
        recipients = []
    return {
        "host": setting("host", "SMTP_HOST"),
        "port": int(setting("port", "SMTP_PORT", 587)),
        "username": setting("username", "SMTP_USER"),
        "password": setting("password", "SMTP_PASSWORD"),
        "from_address": setting("from_address", "SMTP_FROM"),
        "to": recipients,
    }


def send_api_email_notification(
    execution: Mapping[str, Any],
    *,
    enabled: bool = True,
    smtp_config: Mapping[str, Any] | None = None,
    streamlit_secrets: Mapping[str, Any] | None = None,
) -> bool:
    """Send an API execution email; return False without sending if disabled."""
    if not enabled:
        return False
    config = _smtp_settings(smtp_config, streamlit_secrets)
    if not config["host"] or not config["from_address"] or not config["to"]:
        raise ValueError(
            "Configure SMTP_HOST, SMTP_FROM, and SMTP_TO (or Streamlit secrets)."
        )
    message = EmailMessage()
    status = str(execution.get("execution_status") or "unknown")
    dataset = str(execution.get("dataset_name") or "API dataset")
    endpoint = str(
        execution.get("api_endpoint")
        or execution.get("source_endpoint")
        or "unknown"
    )
    try:
        records = int(execution.get("records_retrieved", 0) or 0)
    except (TypeError, ValueError):
        records = 0
    duration = execution.get("duration_seconds")
    start = execution.get("start_time") or "unknown"
    end = execution.get("end_time") or "unknown"
    error_summary = sanitize_for_log(str(execution.get("error_message") or "None"))
    message["Subject"] = f"API ETL {status}: {dataset}"
    message["From"] = config["from_address"]
    message["To"] = ", ".join(config["to"])
    message.set_content(
        "\n".join(
            (
                f"API endpoint: {endpoint}",
                f"Dataset: {dataset}",
                f"Execution status: {status}",
                f"Execution ID: {execution.get('execution_id', '')}",
                f"Execution time: {start} to {end} (duration: {duration} seconds)",
                f"Records retrieved: {records}",
                f"Error summary: {error_summary}",
            )
        )
    )
    with smtplib.SMTP(str(config["host"]), config["port"], timeout=20) as server:
        server.starttls()
        if config["username"]:
            server.login(str(config["username"]), str(config["password"]))
        server.send_message(message)
    return True


def notify_api_execution(
    execution: Mapping[str, Any],
    *,
    enabled: bool = True,
    smtp_config: Mapping[str, Any] | None = None,
    streamlit_secrets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an optional notification and return a non-secret delivery status."""
    if not enabled:
        return {"notification_status": "disabled"}
    try:
        send_api_email_notification(
            execution,
            enabled=True,
            smtp_config=smtp_config,
            streamlit_secrets=streamlit_secrets,
        )
        return {"notification_status": "sent"}
    except Exception as error:
        safe_message = sanitize_for_log(str(error))
        for key, value in (smtp_config or {}).items():
            if _is_sensitive_key(key) and value:
                safe_message = str(safe_message).replace(str(value), "[REDACTED]")
        return {
            "notification_status": "failed",
            "notification_error": safe_message,
        }