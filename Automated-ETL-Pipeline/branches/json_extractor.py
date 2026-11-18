"""JSON-only ingestion helpers for local paths and uploaded file objects."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


DEFAULT_RAW_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "json"
)


def _source_name(source: str | Path | BinaryIO) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return Path(str(name)).name if name else "uploaded.json"


def _read_source(source: str | Path | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")
        return path.read_bytes()

    try:
        original_position = source.tell()
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise OSError("Uploaded JSON file must support seeking.") from exc

    try:
        contents = source.read()
    finally:
        source.seek(original_position)

    if not isinstance(contents, bytes):
        raise OSError("Could not read uploaded JSON file as bytes.")
    return contents


def _save_original(
    contents: bytes, filename: str, raw_directory: str | Path
) -> Path:
    """Save bytes without overwriting an existing file, even on concurrent uploads."""
    directory = Path(raw_directory)
    directory.mkdir(parents=True, exist_ok=True)
    original_path = Path(filename)
    stem = original_path.stem or "uploaded"
    suffix = original_path.suffix or ".json"

    index = 0
    while True:
        candidate_name = (
            f"{stem}{suffix}" if index == 0 else f"{stem}_{index}{suffix}"
        )
        destination = directory / candidate_name
        try:
            with destination.open("xb") as output:
                output.write(contents)
            return destination
        except FileExistsError:
            index += 1


def convert_json_to_dataframe(json_data: Any) -> pd.DataFrame:
    """Represent parsed JSON as a DataFrame without flattening nested values.

    A top-level ``currencies`` array is treated as the record collection.
    Arrays of objects become rows and preserve their key order. For other JSON
    objects, nested dictionaries and lists remain intact in a single-row frame.
    """
    records: Any = json_data
    if isinstance(json_data, Mapping) and isinstance(
        json_data.get("currencies"), list
    ):
        records = json_data["currencies"]

    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(record, Mapping) for record in records):
            return pd.DataFrame(records)
        return pd.DataFrame({"value": records})
    if isinstance(records, Mapping):
        return pd.DataFrame([records])
    return pd.DataFrame({"value": [records]})


def get_json_metadata(
    json_data: Any,
    data: pd.DataFrame,
    source: str,
    file_size: int,
    status: str = "success",
    error: str | None = None,
    raw_file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build ingestion metadata dynamically from the source and DataFrame."""
    if isinstance(json_data, Mapping):
        structure_type = "object"
        top_level_keys = list(json_data.keys())
    elif isinstance(json_data, list):
        structure_type = "array"
        top_level_keys = []
    else:
        structure_type = "scalar"
        top_level_keys = []

    metadata: dict[str, Any] = {
        "source": source,
        "source_type": "File",
        "format": "JSON",
        "file_size": file_size,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "json_structure_type": structure_type,
        "top_level_keys": top_level_keys,
        "records": len(data.index),
        "columns": len(data.columns),
        "column_names": list(data.columns),
        "status": status,
    }
    if raw_file_path is not None:
        metadata["raw_file_path"] = str(raw_file_path)
    if error is not None:
        metadata["error"] = error
    return metadata


def _failed_result(
    source_name: str,
    file_size: int,
    error: Exception,
    raw_file_path: Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source_name,
        "source_type": "File",
        "format": "JSON",
        "file_size": file_size,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "json_structure_type": None,
        "top_level_keys": [],
        "records": 0,
        "columns": 0,
        "column_names": [],
        "status": "error",
        "error": str(error),
    }
    if raw_file_path is not None:
        metadata["raw_file_path"] = str(raw_file_path)
    return {"data": None, "json_data": None, "metadata": metadata}


def extract_json_file(
    source: str | Path | BinaryIO,
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> dict[str, Any]:
    """Extract one JSON file, archive its original bytes, and return its data.

    ``source`` can be a local ``.json`` path or a seekable upload object such as
    Streamlit's ``UploadedFile``. Results include the parsed source under
    ``json_data`` and a non-flattened DataFrame under ``data``. File-level
    failures are returned as an error result so multi-file ingestion can continue.
    """
    source_name = _source_name(source)
    file_size = 0
    raw_file_path: Path | None = None

    try:
        if Path(source_name).suffix.lower() != ".json":
            raise ValueError(
                f"Unsupported file format for '{source_name}'; expected .json."
            )

        contents = _read_source(source)
        file_size = len(contents)
        raw_file_path = _save_original(contents, source_name, raw_directory)
        if not contents.strip():
            raise ValueError("JSON file is empty.")

        try:
            json_data = json.loads(contents.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ValueError("JSON file must use UTF-8 or UTF-8 BOM encoding.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        data = convert_json_to_dataframe(json_data)
        metadata = get_json_metadata(
            json_data,
            data,
            source_name,
            file_size,
            raw_file_path=raw_file_path,
        )
        return {"data": data, "json_data": json_data, "metadata": metadata}
    except (OSError, ValueError, TypeError) as error:
        return _failed_result(source_name, file_size, error, raw_file_path)


def extract_multiple_json_files(
    sources: Sequence[str | Path | BinaryIO],
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> list[dict[str, Any]]:
    """Extract each JSON source independently, preserving one result per input."""
    return [
        extract_json_file(source, raw_directory=raw_directory) for source in sources
    ]
