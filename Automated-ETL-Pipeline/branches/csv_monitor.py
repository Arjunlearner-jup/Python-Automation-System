"""Monitoring and log inspection helpers for automated CSV executions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .csv_automation import DEFAULT_LOG_DIRECTORY, get_csv_execution_history


def get_csv_execution_log(
    execution_id: str,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
) -> dict[str, Any]:
    """Return a specific structured execution log."""
    path = Path(log_directory) / f"{execution_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"CSV execution log not found: {execution_id}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"CSV execution log is invalid JSON: {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"CSV execution log is not a JSON object: {path}")
    result["log_path"] = str(path)
    return result


def get_failed_csv_executions(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent failed pipeline runs."""
    return [
        execution
        for execution in get_csv_execution_history(log_directory, limit=limit)
        if execution.get("execution_status") == "failed"
    ]


def get_csv_log_text(
    execution_id: str,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
) -> str:
    """Read an execution log as formatted JSON for display or download."""
    record = get_csv_execution_log(execution_id, log_directory)
    return json.dumps(record, ensure_ascii=False, indent=2, default=str)
