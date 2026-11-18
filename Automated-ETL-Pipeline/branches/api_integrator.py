"""Stage 5 integration operations for cleaned and validated API observations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from numbers import Number
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.api import types as pandas_types


DEFAULT_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "integrated" / "api"
)
DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "integration" / "api"
)
SOURCE_COLUMN = "_api_source_dataset"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._") or "integrated_api"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _column_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pandas_types.is_bool_dtype(dtype):
        return "boolean"
    if pandas_types.is_numeric_dtype(dtype):
        return "numeric"
    if pandas_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    values = series.dropna()
    if values.empty:
        return "empty"
    if values.map(lambda value: isinstance(value, bool)).all():
        return "boolean"
    if values.map(lambda value: isinstance(value, Number) and not isinstance(value, bool)).all():
        return "numeric"
    if values.map(lambda value: isinstance(value, (date, datetime, pd.Timestamp))).all():
        return "datetime"
    if values.map(lambda value: isinstance(value, str)).all():
        return "text"
    return "mixed"


def _types_compatible(left: pd.Series, right: pd.Series) -> bool:
    left_type, right_type = _column_type(left), _column_type(right)
    if "empty" in {left_type, right_type}:
        return True
    if "mixed" in {left_type, right_type}:
        return False
    return left_type == right_type


def _source_parts(dataset: Any, identifier: str) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy(deep=True), identifier, {}
    if not isinstance(dataset, Mapping):
        raise TypeError("API datasets must be DataFrames or Stage 3/4 result mappings.")

    validation = dataset.get("validation_result")
    if validation is None and isinstance(dataset.get("valid_dataframe"), pd.DataFrame):
        validation = dataset
    if isinstance(validation, Mapping):
        status = validation.get("validation_status")
        if status in {"failed", "error"}:
            raise ValueError(f"API dataset {identifier!r} did not pass validation.")
        frame = validation.get("valid_dataframe", validation.get("valid_data"))
    else:
        frame = dataset.get("data")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"API dataset {identifier!r} does not contain a DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError(f"API DataFrame {identifier!r} has duplicate column names.")

    report = dataset.get("report", {})
    if not isinstance(report, Mapping) and isinstance(validation, Mapping):
        report = validation.get("validation_report", {})
    if not isinstance(report, Mapping):
        report = {}
    metadata = (
        dataset.get("api_metadata")
        or dataset.get("metadata")
        or (validation.get("api_metadata") if isinstance(validation, Mapping) else None)
        or report.get("api_metadata")
        or {}
    )
    if not isinstance(metadata, Mapping):
        raise TypeError(f"API metadata for dataset {identifier!r} must be a mapping.")
    name = str(
        report.get("dataset_name")
        or metadata.get("dataset_id")
        or dataset.get("source")
        or identifier
    )
    return frame.copy(deep=True), name, dict(metadata)


def _dataset_items(datasets: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    items = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if not items:
        raise ValueError("Select at least one API dataset for integration.")
    return [(str(key), value) for key, value in items]


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in metadata.items()}
    for name in names:
        value = metadata.get(name)
        if value is None:
            value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _source_metadata(
    identifier: str, name: str, metadata: Mapping[str, Any], row_count: int
) -> dict[str, Any]:
    base_url = _metadata_value(metadata, "api_base_url", "base_url")
    endpoint = _metadata_value(metadata, "api_endpoint", "endpoint", "url")
    if base_url and endpoint and not str(endpoint).startswith(("http://", "https://")):
        endpoint = f"{str(base_url).rstrip('/')}/{str(endpoint).lstrip('/')}"
    return {
        "dataset_id": identifier,
        "dataset_name": name,
        "record_count": row_count,
        "api_provider": _metadata_value(metadata, "api_provider", "provider", "source_api"),
        "api_name": _metadata_value(metadata, "api_name", "api"),
        "endpoint": endpoint,
        "database_id": _metadata_value(metadata, "database_id", "DATABASE_ID"),
        "indicator_id": _metadata_value(metadata, "indicator_id", "INDICATOR"),
        "reference_area": _metadata_value(
            metadata, "country_reference_area", "reference_area", "ref_area", "REF_AREA"
        ),
        "request_parameters": _json_safe(
            _metadata_value(metadata, "request_parameters", "parameters") or {}
        ),
        "api_metadata": _json_safe(dict(metadata)),
    }


def _indicator_ids(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> set[str]:
    identifiers = set()
    metadata_indicator = _metadata_value(metadata, "indicator_id", "INDICATOR")
    if metadata_indicator is not None:
        identifiers.add(str(metadata_indicator))
    indicator_column = next(
        (column for column in frame.columns if str(column).upper() == "INDICATOR"),
        None,
    )
    if indicator_column is not None:
        identifiers.update(
            str(value)
            for value in frame[indicator_column].dropna().unique()
            if str(value).strip()
        )
    return identifiers


def _unique_name(columns: Sequence[Any], preferred: str) -> str:
    existing = {str(column) for column in columns}
    candidate = preferred
    suffix = 2
    while candidate in existing:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    return candidate


def match_common_columns(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    column_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Find common columns, normalized-name suggestions, and type compatibility."""
    items = _dataset_items(datasets)
    if len(items) < 2:
        raise ValueError("At least two API datasets are needed to match columns.")
    left, left_name, _ = _source_parts(items[0][1], items[0][0])
    right, right_name, _ = _source_parts(items[1][1], items[1][0])
    left_names = [str(column) for column in left.columns]
    right_names = [str(column) for column in right.columns]
    left_positions = {name: index for index, name in enumerate(left_names)}
    right_positions = {name: index for index, name in enumerate(right_names)}
    common = [name for name in left_names if name in right_positions]

    def normalized(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    normalized_right: dict[str, list[str]] = {}
    for name in right_names:
        normalized_right.setdefault(normalized(name), []).append(name)
    suggestions = []
    for left_column in left_names:
        candidates = [
            right_column
            for right_column in normalized_right.get(normalized(left_column), [])
            if right_column != left_column
        ]
        if candidates:
            suggestions.append(
                {
                    "left_column": left_column,
                    "right_column_candidates": candidates,
                    "compatible_types": all(
                        _types_compatible(
                            left.iloc[:, left_positions[left_column]],
                            right.iloc[:, right_positions[right_column]],
                        )
                        for right_column in candidates
                    ),
                }
            )
    mapped = []
    for left_column, right_column in (column_mapping or {}).items():
        exists = left_column in left_positions and right_column in right_positions
        compatible = (
            _types_compatible(
                left.iloc[:, left_positions[left_column]],
                right.iloc[:, right_positions[right_column]],
            )
            if exists
            else False
        )
        mapped.append(
            {
                "left_column": left_column,
                "right_column": right_column,
                "exists": exists,
                "compatible_types": compatible,
            }
        )
    valid_mappings = {
        item["left_column"]: item["right_column"]
        for item in mapped
        if item["exists"] and item["compatible_types"]
    }
    return {
        "left_dataset": left_name,
        "right_dataset": right_name,
        "common_columns": common,
        "possible_matching_columns": suggestions,
        "manual_mappings": mapped,
        "incompatible_common_columns": [
            name
            for name in common
            if not _types_compatible(
                left.iloc[:, left_positions[name]],
                right.iloc[:, right_positions[name]],
            )
        ],
        "unmatched_left_columns": sorted(set(left_names) - set(common) - set(valid_mappings)),
        "unmatched_right_columns": sorted(
            set(right_names) - set(common) - set(valid_mappings.values())
        ),
    }


def handle_inconsistent_schemas(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    column_mappings: Mapping[str, Mapping[str, str]] | None = None,
    fill_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Align columns in copies and report missing, additional, and incompatible fields."""
    items = _dataset_items(datasets)
    frames: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    rename_maps: dict[str, dict[str, str]] = {}
    for identifier, dataset in items:
        frame, name, api_metadata = _source_parts(dataset, identifier)
        rename_map = dict((column_mappings or {}).get(identifier, {}))
        unknown = set(rename_map) - set(map(str, frame.columns))
        if unknown:
            raise KeyError(f"Unknown source columns in mapping for {name!r}: {sorted(unknown)}")
        renamed = frame.rename(columns=rename_map)
        if not renamed.columns.is_unique:
            raise ValueError(f"Column mapping creates duplicate columns in {name!r}.")
        frames[identifier] = renamed
        names[identifier] = name
        metadata[identifier] = api_metadata
        rename_maps[identifier] = rename_map

    all_columns = list(dict.fromkeys(column for frame in frames.values() for column in frame.columns))
    schema_rows = []
    for identifier, frame in frames.items():
        present = list(frame.columns)
        schema_rows.append(
            {
                "dataset_id": identifier,
                "dataset_name": names[identifier],
                "columns": [str(column) for column in present],
                "missing_columns": [str(column) for column in all_columns if column not in present],
                "additional_columns": [
                    str(column)
                    for column in present
                    if any(
                        column not in other.columns
                        for other_id, other in frames.items()
                        if other_id != identifier
                    )
                ],
                "row_count": len(frame.index),
                "column_count": len(present),
                "column_mapping": rename_maps[identifier],
            }
        )
    aligned: dict[str, pd.DataFrame] = {}
    for identifier, frame in frames.items():
        result = frame.reindex(columns=all_columns).copy(deep=True)
        for column, value in (fill_values or {}).items():
            if column not in all_columns:
                raise KeyError(f"Fill-value rule references unknown column {column!r}.")
            if column in result.columns:
                result[column] = result[column].astype("object").fillna(value)
        aligned[identifier] = result

    dtype_report = []
    for column in all_columns:
        definitions = [
            {
                "dataset_id": identifier,
                "dtype": str(frame[column].dtype),
                "value_type": _column_type(frame[column]),
            }
            for identifier, frame in frames.items()
            if column in frame.columns
        ]
        value_types = {
            entry["value_type"]
            for entry in definitions
            if entry["value_type"] != "empty"
        }
        dtype_report.append(
            {
                "column_name": str(column),
                "definitions": definitions,
                "compatible_types": (
                    len(value_types) <= 1 or value_types <= {"numeric"}
                ),
            }
        )
    return {
        "dataframes": aligned,
        "schema_report": schema_rows,
        "column_order": [str(column) for column in all_columns],
        "dtype_report": dtype_report,
        "column_mappings": {key: value for key, value in rename_maps.items() if value},
        "fill_values": _json_safe(dict(fill_values or {})),
        "metadata": metadata,
    }


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value or [])


def _is_missing_key(value: Any) -> bool:
    if isinstance(value, str) and not value.strip():
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if getattr(result, "ndim", 0) == 0 else False


def _key_tuples(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[Any, ...] | None]:
    output = []
    for values in frame.loc[:, list(keys)].itertuples(index=False, name=None):
        if any(_is_missing_key(value) for value in values):
            output.append(None)
        else:
            output.append(tuple(values))
    return output


def _canonical_join_key(value: Any, side: str, position: int) -> str:
    if _is_missing_key(value):
        return json.dumps(["missing", side, position], ensure_ascii=False)
    if isinstance(value, bool):
        value_type, normalized = "boolean", value
    elif isinstance(value, Number):
        try:
            value_type, normalized = "numeric", str(Decimal(str(value)).normalize())
        except (InvalidOperation, ValueError):
            value_type, normalized = "numeric", str(value)
    elif isinstance(value, (date, datetime, pd.Timestamp)):
        value_type, normalized = "datetime", value.isoformat()
    else:
        value_type, normalized = type(value).__name__, str(value)
    return json.dumps(["value", value_type, normalized], ensure_ascii=False)


def _duplicate_keys(
    frame: pd.DataFrame, keys: Sequence[str]
) -> dict[tuple[Any, ...], list[int]]:
    duplicates: dict[tuple[Any, ...], list[int]] = {}
    for position, key in enumerate(_key_tuples(frame, keys)):
        if key is not None:
            duplicates.setdefault(key, []).append(position)
    return {key: positions for key, positions in duplicates.items() if len(positions) > 1}


def _column_by_name(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame.iloc[:, next(index for index, column in enumerate(frame.columns) if str(column) == name)]


def combine_api_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    selected_datasets: Sequence[str] | None = None,
    allow_mixed_indicators: bool = False,
    column_mappings: Mapping[str, Mapping[str, str]] | None = None,
    fill_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Concatenate API observations, retaining source markers and API metadata."""
    items = _dataset_items(datasets)
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        items = [(key, dataset) for key, dataset in items if key in selected]
    if not items:
        raise ValueError("No selected API datasets are available to combine.")
    sources = []
    for identifier, dataset in items:
        frame, name, metadata = _source_parts(dataset, identifier)
        sources.append((identifier, frame, name, metadata))
    indicators = set().union(
        *(_indicator_ids(frame, metadata) for _, frame, _, metadata in sources)
    )
    if len(indicators) > 1 and not allow_mixed_indicators:
        raise ValueError(
            "Selected API datasets describe different indicators. Enable "
            "allow_mixed_indicators explicitly to combine them."
        )

    schema = handle_inconsistent_schemas(
        {
            identifier: {
                "data": frame,
                "api_metadata": metadata,
                "report": {"dataset_name": name},
            }
            for identifier, frame, name, metadata in sources
        },
        column_mappings=column_mappings,
        fill_values=fill_values,
    )
    source_column = _unique_name(schema["column_order"], SOURCE_COLUMN)
    contributions = []
    frames = []
    source_information = []
    for identifier, _, name, metadata in sources:
        aligned = schema["dataframes"][identifier].copy(deep=True)
        aligned[source_column] = name
        frames.append(aligned)
        source_entry = _source_metadata(
            identifier, name, metadata, len(aligned.index)
        )
        contributions.append(
            {
                "dataset_id": identifier,
                "dataset_name": name,
                "records_contributed": len(aligned.index),
            }
        )
        source_information.append(source_entry)
    integrated = pd.concat(frames, ignore_index=True, sort=False)
    report = {
        "operation": "combine",
        "integration_type": "combine",
        "dataset_name": "integrated_api",
        "input_dataset_count": len(sources),
        "input_record_count": sum(item["records_contributed"] for item in contributions),
        "output_record_count": len(integrated.index),
        "source_column": source_column,
        "source_record_counts": contributions,
        "mixed_indicators_allowed": allow_mixed_indicators,
        "indicator_ids": sorted(indicators),
        "column_mappings": schema["column_mappings"],
        "fill_values": schema["fill_values"],
    }
    return {
        "data": integrated,
        "integration_report": report,
        "source_datasets": source_information,
        "matched_records": integrated.iloc[0:0].copy(),
        "unmatched_records": {"left": integrated.iloc[0:0].copy(), "right": integrated.iloc[0:0].copy()},
        "schema_report": {
            "schema_report": schema["schema_report"],
            "column_order": schema["column_order"],
            "dtype_report": schema["dtype_report"],
        },
        "integration_status": "success",
    }


def merge_api_datasets(
    left_dataset: Any,
    right_dataset: Any,
    *,
    join_type: str = "inner",
    key_columns: Sequence[str] | str | None = None,
    right_key_columns: Sequence[str] | str | None = None,
    column_mapping: Mapping[str, str] | None = None,
    many_to_many: str = "raise",
    allow_mixed_indicators: bool = False,
    fill_values: Mapping[str, Any] | None = None,
    left_name: str = "left_api_dataset",
    right_name: str = "right_api_dataset",
) -> dict[str, Any]:
    """Join two validated API DataFrames and identify matched/unmatched rows."""
    join = join_type.lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join_type must be inner, left, right, or outer.")
    if many_to_many not in {"raise", "allow"}:
        raise ValueError("many_to_many must be 'raise' or 'allow'.")
    left, resolved_left_name, left_metadata = _source_parts(left_dataset, left_name)
    right, resolved_right_name, right_metadata = _source_parts(right_dataset, right_name)
    left_indicators = _indicator_ids(left, left_metadata)
    right_indicators = _indicator_ids(right, right_metadata)
    left_indicator = _metadata_value(left_metadata, "indicator_id", "INDICATOR")
    right_indicator = _metadata_value(right_metadata, "indicator_id", "INDICATOR")
    if (
        left_indicators
        and right_indicators
        and left_indicators != right_indicators
        and not allow_mixed_indicators
    ):
        raise ValueError(
            "Selected API datasets describe different indicators. Enable "
            "allow_mixed_indicators explicitly to merge them."
        )
    if not left.columns.is_unique or not right.columns.is_unique:
        raise ValueError("API DataFrames must have unique column names before merging.")
    left_names, right_names = list(map(str, left.columns)), list(map(str, right.columns))
    mappings = dict(column_mapping or {})
    left_keys = _as_list(key_columns)
    right_keys = _as_list(right_key_columns)
    if not left_keys and mappings:
        left_keys = list(mappings)
    if not left_keys:
        default_identifiers = (
            "DATABASE_ID",
            "INDICATOR",
            "REF_AREA",
            "TIME_PERIOD",
        )
        left_keys = [
            column
            for column in default_identifiers
            if column in left_names and (mappings.get(column, column) in right_names)
        ]
    if not left_keys:
        raise ValueError("Select one or more common API identifier columns as merge keys.")
    if not right_keys:
        right_keys = [mappings.get(key, key) for key in left_keys]
    if len(left_keys) != len(right_keys):
        raise ValueError("Left and right API key lists must have equal lengths.")
    missing_keys = [key for key in left_keys if key not in left_names]
    missing_keys.extend(key for key in right_keys if key not in right_names)
    if missing_keys:
        raise KeyError(f"Unknown API merge key column(s): {missing_keys}.")
    if len(set(left_keys)) != len(left_keys) or len(set(right_keys)) != len(right_keys):
        raise ValueError("API merge keys must not contain duplicate column names.")
    incompatible = [
        (left_key, right_key)
        for left_key, right_key in zip(left_keys, right_keys)
        if not _types_compatible(
            _column_by_name(left, left_key), _column_by_name(right, right_key)
        )
    ]
    if incompatible:
        raise TypeError(f"Incompatible API merge key data types: {incompatible}.")

    left_duplicates = _duplicate_keys(left, left_keys)
    right_duplicates = _duplicate_keys(right, right_keys)
    overlapping_duplicate_keys = set(left_duplicates) & set(right_duplicates)
    if overlapping_duplicate_keys and many_to_many == "raise":
        raise ValueError(
            "Many-to-many API merge would expand duplicate matching keys. "
            "Resolve duplicate keys or set many_to_many='allow'."
        )

    left_work = left.copy(deep=True)
    right_work = right.copy(deep=True)
    helper_columns = []
    left_join_keys = []
    right_join_keys = []
    for number, (left_key, right_key) in enumerate(zip(left_keys, right_keys)):
        left_helper = _unique_name([*left_work.columns, *right_work.columns, *helper_columns], f"_api_left_key_{number}")
        helper_columns.append(left_helper)
        right_helper = _unique_name([*left_work.columns, *right_work.columns, *helper_columns], f"_api_right_key_{number}")
        helper_columns.append(right_helper)
        left_values = _column_by_name(left_work, left_key).tolist()
        right_values = _column_by_name(right_work, right_key).tolist()
        left_work[left_helper] = [
            _canonical_join_key(value, "left", position)
            for position, value in enumerate(left_values)
        ]
        right_work[right_helper] = [
            _canonical_join_key(value, "right", position)
            for position, value in enumerate(right_values)
        ]
        left_join_keys.append(left_helper)
        right_join_keys.append(right_helper)

    left_position = _unique_name([*left_work.columns, *right_work.columns], "_api_left_position")
    right_position = _unique_name([*left_work.columns, *right_work.columns, left_position], "_api_right_position")
    left_source = _unique_name([*left_work.columns, *right_work.columns, left_position, right_position], "_api_left_source")
    right_source = _unique_name([*left_work.columns, *right_work.columns, left_position, right_position, left_source], "_api_right_source")
    left_work[left_position] = range(len(left_work.index))
    right_work[right_position] = range(len(right_work.index))
    left_work[left_source] = resolved_left_name
    right_work[right_source] = resolved_right_name
    indicator = _unique_name([*left_work.columns, *right_work.columns], "_api_merge_status")
    merged = left_work.merge(
        right_work,
        how=join,
        left_on=left_join_keys,
        right_on=right_join_keys,
        suffixes=("_left", "_right"),
        indicator=indicator,
        sort=False,
    )
    status = merged[indicator]
    matched = merged.loc[status == "both"].copy()
    internal_columns = [*helper_columns, left_position, right_position, indicator]
    def remove_helpers(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.drop(columns=internal_columns).copy()
        for left_key, right_key in zip(left_keys, right_keys):
            if left_key == right_key:
                left_label = f"{left_key}_left"
                right_label = f"{right_key}_right"
                if left_label in cleaned.columns and right_label in cleaned.columns:
                    cleaned[left_key] = cleaned[left_label].combine_first(
                        cleaned[right_label]
                    )
                    cleaned = cleaned.drop(columns=[left_label, right_label])
        return cleaned

    result_data = remove_helpers(merged)
    for column, value in (fill_values or {}).items():
        if column not in result_data.columns:
            raise KeyError(f"Fill-value rule references unknown column {column!r}.")
        result_data[column] = result_data[column].astype("object").fillna(value)
    matched_records = remove_helpers(matched)
    left_counterpart_keys = {
        key for key in _key_tuples(right, right_keys) if key is not None
    }
    right_counterpart_keys = {
        key for key in _key_tuples(left, left_keys) if key is not None
    }
    left_unmatched_mask = [
        key is None or key not in left_counterpart_keys
        for key in _key_tuples(left, left_keys)
    ]
    right_unmatched_mask = [
        key is None or key not in right_counterpart_keys
        for key in _key_tuples(right, right_keys)
    ]
    left_unmatched = left.loc[left_unmatched_mask].copy()
    right_unmatched = right.loc[right_unmatched_mask].copy()
    left_unmatched[left_source] = resolved_left_name
    right_unmatched[right_source] = resolved_right_name
    unmatched_records = {
        "left": left_unmatched,
        "right": right_unmatched,
    }
    source_information = [
        _source_metadata(resolved_left_name, resolved_left_name, left_metadata, len(left.index)),
        _source_metadata(resolved_right_name, resolved_right_name, right_metadata, len(right.index)),
    ]
    matching = match_common_columns(
        {
            "left": {
                "data": left,
                "api_metadata": left_metadata,
                "report": {"dataset_name": resolved_left_name},
            },
            "right": {
                "data": right,
                "api_metadata": right_metadata,
                "report": {"dataset_name": resolved_right_name},
            },
        },
        column_mapping=mappings,
    )
    schema = handle_inconsistent_schemas(
        {
            "left": {
                "data": left,
                "api_metadata": left_metadata,
                "report": {"dataset_name": resolved_left_name},
            },
            "right": {
                "data": right,
                "api_metadata": right_metadata,
                "report": {"dataset_name": resolved_right_name},
            },
        }
    )
    report = {
        "operation": "merge",
        "integration_type": "merge",
        "dataset_name": "integrated_api",
        "join_type": join,
        "left_dataset": resolved_left_name,
        "right_dataset": resolved_right_name,
        "left_key_columns": left_keys,
        "right_key_columns": right_keys,
        "left_input_records": len(left.index),
        "right_input_records": len(right.index),
        "output_records": len(result_data.index),
        "matched_records": len(matched_records.index),
        "unmatched_left_records": len(unmatched_records["left"].index),
        "unmatched_right_records": len(unmatched_records["right"].index),
        "duplicate_left_keys": len(left_duplicates),
        "duplicate_right_keys": len(right_duplicates),
        "many_to_many_allowed": many_to_many == "allow",
        "column_mapping": mappings,
        "mixed_indicators_allowed": allow_mixed_indicators,
        "left_indicator_id": left_indicator,
        "right_indicator_id": right_indicator,
        "left_indicator_ids": sorted(left_indicators),
        "right_indicator_ids": sorted(right_indicators),
        "fill_values": _json_safe(dict(fill_values or {})),
    }
    return {
        "data": result_data,
        "integration_report": report,
        "source_datasets": source_information,
        "matched_records": matched_records,
        "unmatched_records": unmatched_records,
        "schema_report": {
            "column_matching": matching,
            "schema_report": schema["schema_report"],
            "dtype_report": schema["dtype_report"],
            "left_columns": left_names,
            "right_columns": right_names,
        },
        "integration_status": "success",
    }


def _save_integration(
    result: Mapping[str, Any],
    output_directory: str | Path,
    report_directory: str | Path,
) -> dict[str, str]:
    dataframe = result.get("data")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("API integration result must contain a DataFrame under 'data'.")
    data_root, reports_root = Path(output_directory), Path(report_directory)
    data_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex[:8]
    name = _safe_name(str(result["integration_report"].get("dataset_name", "integrated_api")))
    csv_path = data_root / f"{name}_{identifier}.csv"
    json_path = data_root / f"{name}_{identifier}.json"
    metadata_path = data_root / f"{name}_{identifier}_metadata.json"
    report_path = reports_root / f"{name}_{identifier}_report.json"
    report_csv_path = reports_root / f"{name}_{identifier}_report.csv"
    schema_path = reports_root / f"{name}_{identifier}_schema.csv"
    records_path = reports_root / f"{name}_{identifier}_records.csv"

    with csv_path.open("x", encoding="utf-8", newline="") as file:
        dataframe.to_csv(file, index=False)
    with json_path.open("x", encoding="utf-8") as file:
        file.write(dataframe.to_json(orient="records", date_format="iso", force_ascii=False, indent=2))
        file.write("\n")

    payload = {
        "integration_report": _json_safe(result.get("integration_report", {})),
        "source_datasets": _json_safe(result.get("source_datasets", [])),
        "schema_report": _json_safe(result.get("schema_report", {})),
        "integration_status": result.get("integration_status", "success"),
    }
    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "api_sources": payload["source_datasets"],
                "integration_report": payload["integration_report"],
                "data_files": {"csv": csv_path.name, "json": json_path.name},
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        file.write("\n")
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
    with report_csv_path.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame([payload["integration_report"]]).to_csv(file, index=False)
    schema_data = payload["schema_report"]
    schema_rows = (
        schema_data.get("schema_report", [])
        if isinstance(schema_data, Mapping)
        else schema_data
    )
    with schema_path.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame(schema_rows).to_csv(file, index=False)
    record_rows = []
    matched = result.get("matched_records")
    if isinstance(matched, pd.DataFrame):
        record_rows.extend(
            {"match_status": "matched", **_json_safe(row)}
            for row in matched.to_dict(orient="records")
        )
    unmatched = result.get("unmatched_records", {})
    if isinstance(unmatched, Mapping):
        for side, frame in unmatched.items():
            if isinstance(frame, pd.DataFrame):
                record_rows.extend(
                    {"match_status": f"unmatched_{side}", **_json_safe(row)}
                    for row in frame.to_dict(orient="records")
                )
    with records_path.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame(record_rows).to_csv(file, index=False)
    return {
        "data_csv": str(csv_path),
        "data_json": str(json_path),
        "metadata_json": str(metadata_path),
        "report_json": str(report_path),
        "report_csv": str(report_csv_path),
        "schema_csv": str(schema_path),
        "records_csv": str(records_path),
        "data_directory": str(data_root),
        "report_directory": str(reports_root),
    }


def integrate_multiple_api_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    integration_type: str = "combine",
    selected_datasets: Sequence[str] | None = None,
    join_type: str = "inner",
    key_columns: Sequence[str] | str | None = None,
    right_key_columns: Sequence[str] | str | None = None,
    column_mapping: Mapping[str, str] | None = None,
    column_mappings: Mapping[str, Mapping[str, str]] | None = None,
    fill_values: Mapping[str, Any] | None = None,
    many_to_many: str = "raise",
    allow_mixed_indicators: bool = False,
    output_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    report_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Integrate selected validated API datasets and optionally persist outputs."""
    items = _dataset_items(datasets)
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        items = [(key, dataset) for key, dataset in items if key in selected]
    if not items:
        raise ValueError("No selected API datasets are available for integration.")
    operation = integration_type.lower()
    if operation in {"combine", "append", "row-wise"}:
        result = combine_api_datasets(
            dict(items),
            allow_mixed_indicators=allow_mixed_indicators,
            column_mappings=column_mappings,
            fill_values=fill_values,
        )
    elif operation in {"merge", "join"}:
        if len(items) < 2:
            raise ValueError("Select at least two API datasets to merge.")
        result = merge_api_datasets(
            items[0][1],
            items[1][1],
            join_type=join_type,
            key_columns=key_columns,
            right_key_columns=right_key_columns,
            column_mapping=column_mapping,
            many_to_many=many_to_many,
            allow_mixed_indicators=allow_mixed_indicators,
            fill_values=fill_values,
            left_name=items[0][0],
            right_name=items[1][0],
        )
        if len(items) > 2:
            merged = result["data"]
            merge_steps = [dict(result["integration_report"])]
            final_step = result
            for index, (dataset_id, dataset) in enumerate(items[2:], start=2):
                _, dataset_name, _ = _source_parts(dataset, dataset_id)
                final_step = merge_api_datasets(
                    merged,
                    dataset,
                    join_type=join_type,
                    key_columns=key_columns,
                    right_key_columns=right_key_columns or key_columns,
                    column_mapping=column_mapping,
                    many_to_many=many_to_many,
                    allow_mixed_indicators=allow_mixed_indicators,
                    fill_values=fill_values,
                    left_name=f"integrated_{index}",
                    right_name=dataset_name,
                )
                merged = final_step["data"]
                merge_steps.append(dict(final_step["integration_report"]))
            result["data"] = merged
            result["matched_records"] = final_step["matched_records"]
            result["unmatched_records"] = final_step["unmatched_records"]
            result["integration_report"]["merge_steps"] = merge_steps
            result["source_datasets"] = []
            for dataset_id, dataset in items:
                frame, dataset_name, metadata = _source_parts(dataset, dataset_id)
                result["source_datasets"].append(
                    _source_metadata(dataset_id, dataset_name, metadata, len(frame.index))
                )
            result["integration_report"]["input_dataset_count"] = len(items)
            result["integration_report"]["output_records"] = len(merged.index)
            result["integration_report"]["final_step_matched_records"] = len(
                final_step["matched_records"].index
            )
            result["integration_report"]["final_step_unmatched_left_records"] = len(
                final_step["unmatched_records"]["left"].index
            )
            result["integration_report"]["final_step_unmatched_right_records"] = len(
                final_step["unmatched_records"]["right"].index
            )
    else:
        raise ValueError("integration_type must be 'combine' or 'merge'.")

    result["integration_report"]["integration_type"] = operation
    result["integration_report"]["dataset_name"] = "integrated_api"
    result["integration_report"]["schema_comparison"] = result.get("schema_report")
    result["integration_status"] = "success"
    if save_outputs:
        result["saved_paths"] = _save_integration(
            result, output_directory, report_directory
        )
    return result
