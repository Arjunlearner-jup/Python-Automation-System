"""History and structured log access for automated XML pipeline executions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .xml_automation import DEFAULT_LOG_DIRECTORY, get_xml_execution_history


def get_xml_execution_log(
    execution_id: str,
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY,
) -> dict[str, Any]:
    """Load one structured XML pipeline execution log."""
    path = Path(log_directory) / f"{execution_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"XML execution log not found: {execution_id}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"XML execution log is invalid JSON: {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"XML execution log is not a JSON object: {path}")
    result["log_path"] = str(path)
    return result


def get_failed_xml_executions(
    log_directory: str | Path = DEFAULT_LOG_DIRECTORY, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Return recent failed XML pipeline executions."""
    return [
        execution
        for execution in get_xml_execution_history(log_directory, limit=limit)
        if execution.get("execution_status") == "failed"
    ]


def get_xml_log_text(
    execution_id: str, log_directory: str | Path = DEFAULT_LOG_DIRECTORY
) -> str:
    """Return a formatted JSON representation suitable for display/download."""
    return json.dumps(
        get_xml_execution_log(execution_id, log_directory),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
