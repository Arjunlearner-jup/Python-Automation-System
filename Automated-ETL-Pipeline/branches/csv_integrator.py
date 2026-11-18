"""Stage 5 integration operations for cleaned and validated CSV datasets."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Number
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.api import types as pandas_types


DEFAULT_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "integrated" / "csv"
)
DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "integration" / "csv"
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._") or "dataset"


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


def _normalise_column_name(column: Any) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column).strip())
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return name or "column"


def _column_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pandas_types.is_bool_dtype(dtype):
        return "boolean"
    if pandas_types.is_integer_dtype(dtype):
        return "numeric"
    if pandas_types.is_float_dtype(dtype):
        return "numeric"
    if pandas_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    non_missing = series.dropna()
    if non_missing.empty:
        return "empty"
    if non_missing.map(lambda value: isinstance(value, bool)).all():
        return "boolean"
    if non_missing.map(
        lambda value: isinstance(value, Number) and not isinstance(value, bool)
    ).all():
        return "numeric"
    if non_missing.map(lambda value: isinstance(value, (date, datetime, pd.Timestamp))).all():
        return "datetime"
    if non_missing.map(lambda value: isinstance(value, str)).all():
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
        return dataset, identifier, {}
    if not isinstance(dataset, Mapping):
        raise TypeError(
            "CSV datasets must be DataFrames or Stage 3/4 result mappings."
        )
    validation = dataset.get("validation_result")
    if validation is None and (
        isinstance(dataset.get("valid_dataframe"), pd.DataFrame)
        or isinstance(dataset.get("valid_data"), pd.DataFrame)
    ):
        validation = dataset
    if isinstance(validation, Mapping):
        status = validation.get("validation_status")
        if status in {"failed", "error"}:
            raise ValueError(f"Dataset {identifier!r} did not pass validation.")
        frame = validation.get(
            "valid_dataframe",
            validation.get("valid_data"),
        )
        if isinstance(frame, pd.DataFrame):
            helper_columns = [
                column
                for column in frame.columns
                if str(column).startswith("_validation_status")
                or str(column).startswith("_validation_errors")
            ]
            frame = frame.drop(columns=helper_columns).copy(deep=True)
    else:
        frame = dataset.get("data")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Dataset {identifier!r} does not contain a DataFrame.")
    report = dataset.get("report", {})
    name = str(
        (report.get("dataset_name") if isinstance(report, Mapping) else None)
        or dataset.get("source")
        or dataset.get("filename")
        or identifier
    )
    metadata = dict(dataset.get("metadata") or {})
    return frame, name, metadata


def _dataset_items(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> list[tuple[str, Any]]:
    items = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if not items:
        raise ValueError("Select at least one CSV dataset for integration.")
    return [(str(key), value) for key, value in items]


def _unique_column_name(columns: Sequence[Any], preferred: str) -> str:
    existing = {str(column) for column in columns}
    candidate = preferred
    suffix = 2
    while candidate in existing:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    return candidate


def match_common_columns(
    datasets: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    *,
    column_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Find exact common columns, normalized-name candidates, and type matches.

    Mapping keys are interpreted as left-column to right-column names for a
    two-dataset comparison. Column names are not renamed in either input.
    """
    items = _dataset_items(datasets)
    if len(items) < 2:
        raise ValueError("At least two datasets are needed to match columns.")
    frames = []
    names = []
    for identifier, dataset in items:
        frame, name, _ = _source_parts(dataset, identifier)
        frames.append(frame)
        names.append(name)
    left, right = frames[0], frames[1]
    left_names = [str(column) for column in left.columns]
    right_names = [str(column) for column in right.columns]
    common = [column for column in left_names if column in right_names]

    normalized_right: dict[str, list[str]] = {}
    for column in right_names:
        normalized_right.setdefault(_normalise_column_name(column), []).append(column)
    possible = []
    for left_name in left_names:
        normalized = _normalise_column_name(left_name)
        matches = [
            right_name
            for right_name in normalized_right.get(normalized, [])
            if right_name != left_name
        ]
        if matches:
            possible.append(
                {
                    "left_column": left_name,
                    "right_column_candidates": matches,
                    "compatible_types": all(
                        _types_compatible(
                            left.iloc[:, left_names.index(left_name)],
                            right.iloc[:, right_names.index(right_name)],
                        )
                        for right_name in matches
                    ),
                }
            )
    mapping = dict(column_mapping or {})
    mapped = []
    for left_name, right_name in mapping.items():
        exists = left_name in left_names and right_name in right_names
        compatible = (
            _types_compatible(
                left.iloc[:, left_names.index(left_name)],
                right.iloc[:, right_names.index(right_name)],
            )
            if exists
            else False
        )
        mapped.append(
            {
                "left_column": left_name,
                "right_column": right_name,
                "exists": exists,
                "compatible_types": compatible,
            }
        )
    valid_mappings = {
        entry["left_column"]: entry["right_column"]
        for entry in mapped
        if entry["exists"] and entry["compatible_types"]
    }
    matched_left = set(common) | set(valid_mappings)
    matched_right = set(common) | set(valid_mappings.values())
    return {
        "left_dataset": names[0],
        "right_dataset": names[1],
        "common_columns": common,
        "possible_matching_columns": possible,
        "manual_mappings": mapped,
        "unmatched_left_columns": [name for name in left_names if name not in matched_left],
        "unmatched_right_columns": [name for name in right_names if name not in matched_right],
    }


def handle_inconsistent_schemas(
    datasets: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
    *,
    standardize_column_names: bool = False,
) -> dict[str, Any]:
    """Compare schemas and align columns without mutating values or source frames."""
    items = _dataset_items(datasets)
    frames: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    original_columns: dict[str, list[str]] = {}
    rename_maps: dict[str, dict[str, str]] = {}
    for identifier, dataset in items:
        frame, name, _ = _source_parts(dataset, identifier)
        original_columns[identifier] = [str(column) for column in frame.columns]
        renamed = {}
        if standardize_column_names:
            for column in frame.columns:
                renamed[str(column)] = _normalise_column_name(column)
            targets = list(renamed.values())
            if len(targets) != len(set(targets)):
                raise ValueError(
                    f"Column-name standardization creates duplicate columns in {name!r}."
                )
            copy = frame.copy(deep=True).rename(columns=renamed)
        else:
            copy = frame.copy(deep=True)
        frames[identifier] = copy
        names[identifier] = name
        rename_maps[identifier] = renamed

    all_columns = list(
        dict.fromkeys(str(column) for frame in frames.values() for column in frame.columns)
    )
    schema_rows = []
    for identifier, frame in frames.items():
        columns = [str(column) for column in frame.columns]
        missing = [column for column in all_columns if column not in columns]
        additional = [
            column
            for column in columns
            if any(
                column not in [str(item) for item in other.columns]
                for other_id, other in frames.items()
                if other_id != identifier
            )
        ]
        schema_rows.append(
            {
                "dataset_id": identifier,
                "dataset_name": names[identifier],
                "columns": columns,
                "missing_columns": missing,
                "additional_columns": list(dict.fromkeys(additional)),
                "column_count": len(columns),
                "row_count": len(frame.index),
                "renamed_columns": rename_maps[identifier],
            }
        )
    aligned = {
        identifier: frame.reindex(columns=all_columns).copy(deep=True)
        for identifier, frame in frames.items()
    }
    dtype_report = []
    for column in all_columns:
        definitions = []
        for identifier, frame in frames.items():
            if column in frame.columns:
                definitions.append(
                    {
                        "dataset_id": identifier,
                        "dtype": str(frame[column].dtype),
                        "value_type": _column_type(frame[column]),
                    }
                )
        value_types = {entry["value_type"] for entry in definitions if entry["value_type"] != "empty"}
        dtype_report.append(
            {
                "column_name": column,
                "definitions": definitions,
                "compatible_types": len(value_types) <= 1 or value_types <= {"numeric"},
                "resolved_dtype": str(aligned[next(iter(aligned))][column].dtype),
            }
        )
    return {
        "dataframes": aligned,
        "schema_report": schema_rows,
        "column_order": all_columns,
        "dtype_report": dtype_report,
        "standardize_column_names": standardize_column_names,
    }


def combine_csv_files(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    selected_datasets: Sequence[str] | None = None,
    standardize_column_names: bool = False,
    source_column: str = "_source_file",
    source_row_column: str = "_source_row_index",
) -> dict[str, Any]:
    """Append CSV datasets row-wise with schema union and file provenance."""
    items = _dataset_items(datasets)
    selected = set(map(str, selected_datasets)) if selected_datasets is not None else None
    chosen = [(key, value) for key, value in items if selected is None or key in selected]
    if not chosen:
        raise ValueError("No selected datasets are available to combine.")
    schema = handle_inconsistent_schemas(
        dict(chosen), standardize_column_names=standardize_column_names
    )
    frames = schema["dataframes"]
    first = next(iter(frames.values()))
    source_column_name = _unique_column_name(first.columns, source_column)
    source_row_name = _unique_column_name(
        [*first.columns, source_column_name], source_row_column
    )
    combined_parts = []
    source_counts = []
    input_rows = 0
    for key, dataset in chosen:
        frame, source_name, metadata = _source_parts(dataset, key)
        output_frame = frames[key].copy(deep=True)
        output_frame[source_column_name] = source_name
        output_frame[source_row_name] = list(frame.index)
        combined_parts.append(output_frame)
        source_counts.append(
            {
                "dataset_id": key,
                "source_file": source_name,
                "records_contributed": len(frame.index),
                "metadata": metadata,
            }
        )
        input_rows += len(frame.index)
    combined = pd.concat(combined_parts, ignore_index=True, sort=False)
    report = {
        "operation": "append",
        "input_dataset_count": len(chosen),
        "input_record_count": input_rows,
        "output_record_count": len(combined.index),
        "source_record_counts": source_counts,
        "source_column": source_column_name,
        "source_row_column": source_row_name,
        "schema_report": schema["schema_report"],
        "column_order": [str(column) for column in combined.columns],
        "dtype_report": schema["dtype_report"],
    }
    return {
        "data": combined,
        "integration_report": report,
        "source_datasets": source_counts,
        "matched_records": [],
        "unmatched_records": [],
        "schema_report": schema["schema_report"],
    }


def _key_lists(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: Sequence[str] | str | None,
    right_key_columns: Sequence[str] | str | None,
    column_mapping: Mapping[str, str] | None,
) -> tuple[list[str], list[str]]:
    def as_list(value: Sequence[str] | str | None) -> list[str]:
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value or []]

    left_keys = as_list(key_columns)
    right_keys = as_list(right_key_columns)
    mapping = dict(column_mapping or {})
    if not left_keys and mapping:
        left_keys = list(mapping)
        right_keys = [mapping[key] for key in left_keys]
    if not left_keys:
        common = [str(column) for column in left.columns if column in right.columns]
        left_keys = common[:1]
    if not right_keys:
        right_keys = [mapping.get(key, key) for key in left_keys]
    if not left_keys or len(left_keys) != len(right_keys):
        raise ValueError("Provide the same number of left and right merge key columns.")
    missing_left = set(left_keys).difference(map(str, left.columns))
    missing_right = set(right_keys).difference(map(str, right.columns))
    if missing_left or missing_right:
        raise KeyError(
            f"Missing merge key columns; left={sorted(missing_left)}, right={sorted(missing_right)}."
        )
    return left_keys, right_keys


def merge_csv_datasets(
    left_dataset: Any,
    right_dataset: Any,
    *,
    join_type: str = "inner",
    key_columns: Sequence[str] | str | None = None,
    right_key_columns: Sequence[str] | str | None = None,
    column_mapping: Mapping[str, str] | None = None,
    many_to_many: str = "raise",
    left_name: str = "left.csv",
    right_name: str = "right.csv",
) -> dict[str, Any]:
    """Merge two CSV datasets and report matched and unmatched records.

    The default ``many_to_many='raise'`` rejects duplicate keys on both sides;
    pass ``'allow'`` explicitly when a Cartesian expansion is intended.
    """
    join = join_type.lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join_type must be inner, left, right, or outer.")
    if many_to_many not in {"raise", "allow"}:
        raise ValueError("many_to_many must be 'raise' or 'allow'.")
    left, resolved_left_name, left_metadata = _source_parts(left_dataset, left_name)
    right, resolved_right_name, right_metadata = _source_parts(right_dataset, right_name)
    left = left.copy(deep=True)
    right = right.copy(deep=True)
    left_keys, right_keys = _key_lists(
        left, right, key_columns, right_key_columns, column_mapping
    )
    incompatible = [
        (left_key, right_key)
        for left_key, right_key in zip(left_keys, right_keys)
        if not _types_compatible(
            left[next(column for column in left.columns if str(column) == left_key)],
            right[next(column for column in right.columns if str(column) == right_key)],
        )
    ]
    if incompatible:
        raise TypeError(f"Incompatible merge key data types: {incompatible}.")

    for frame, keys, side in ((left, left_keys, "left"), (right, right_keys, "right")):
        duplicate_mask = frame.duplicated(subset=keys, keep=False)
        if duplicate_mask.any():
            other = right if side == "left" else left
            other_keys = right_keys if side == "left" else left_keys
            has_many = other.duplicated(subset=other_keys, keep=False).any()
            if has_many and many_to_many == "raise":
                raise ValueError(
                    "Many-to-many merge would duplicate records on both sides; "
                    "deduplicate keys or set many_to_many='allow'."
                )

    left_marker = _unique_column_name([*left.columns, *right.columns], "_left_row_index")
    right_marker = _unique_column_name(
        [*left.columns, *right.columns, left_marker], "_right_row_index"
    )
    left_source_col = _unique_column_name(
        [*left.columns, *right.columns, left_marker, right_marker], "_left_source_file"
    )
    right_source_col = _unique_column_name(
        [*left.columns, *right.columns, left_marker, right_marker, left_source_col],
        "_right_source_file",
    )
    left[left_marker] = list(left.index)
    right[right_marker] = list(right.index)
    left[left_source_col] = resolved_left_name
    right[right_source_col] = resolved_right_name
    indicator = _unique_column_name(
        [*left.columns, *right.columns], "_merge_status"
    )
    merged = left.merge(
        right,
        how=join,
        left_on=left_keys,
        right_on=right_keys,
        suffixes=("_left", "_right"),
        indicator=indicator,
        validate=None,
        sort=False,
    )
    status = merged[indicator]
    matched = merged.loc[status == "both"].copy()
    left_unmatched = merged.loc[status == "left_only"].copy()
    right_unmatched = merged.loc[status == "right_only"].copy()
    matched = matched.drop(columns=[indicator])
    left_unmatched = left_unmatched.drop(columns=[indicator])
    right_unmatched = right_unmatched.drop(columns=[indicator])
    report = {
        "operation": "merge",
        "join_type": join,
        "left_dataset": resolved_left_name,
        "right_dataset": resolved_right_name,
        "left_key_columns": left_keys,
        "right_key_columns": right_keys,
        "left_input_records": len(left.index),
        "right_input_records": len(right.index),
        "output_records": len(merged.index),
        "matched_records": len(matched.index),
        "unmatched_left_records": len(left_unmatched.index),
        "unmatched_right_records": len(right_unmatched.index),
        "many_to_many_allowed": many_to_many == "allow",
        "column_mapping": dict(column_mapping or {}),
    }
    return {
        "data": merged.drop(columns=[indicator]).copy(),
        "integration_report": report,
        "source_datasets": [
            {
                "dataset_name": resolved_left_name,
                "records": len(left.index),
                "metadata": left_metadata,
            },
            {
                "dataset_name": resolved_right_name,
                "records": len(right.index),
                "metadata": right_metadata,
            },
        ],
        "matched_records": matched,
        "unmatched_records": {
            "left": left_unmatched,
            "right": right_unmatched,
        },
        "schema_report": {
            "column_matching": match_common_columns(
                {resolved_left_name: left_dataset, resolved_right_name: right_dataset},
                column_mapping=column_mapping,
            ),
            "left_columns": [str(column) for column in left.columns],
            "right_columns": [str(column) for column in right.columns],
        },
    }


def _save_integration(
    result: Mapping[str, Any],
    output_directory: str | Path,
    report_directory: str | Path,
) -> dict[str, str]:
    report = result.get("integration_report")
    if not isinstance(report, Mapping):
        raise TypeError("Integration result must contain 'integration_report'.")
    data = result.get("data")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Integration result must contain a DataFrame under 'data'.")
    data_root = Path(output_directory)
    reports_root = Path(report_directory)
    data_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex[:8]
    name = _safe_name(str(report.get("dataset_name") or report.get("operation", "integrated_csv")))
    data_path = data_root / f"{name}_{identifier}.csv"
    report_path = reports_root / f"{name}_{identifier}_report.json"
    schema_path = reports_root / f"{name}_{identifier}_schema.csv"
    issues_path = reports_root / f"{name}_{identifier}_records.csv"
    data.to_csv(data_path, index=False)
    report_payload = {
        "integration_report": _json_safe(report),
        "source_datasets": _json_safe(result.get("source_datasets", [])),
        "schema_report": _json_safe(result.get("schema_report", {})),
        "matched_records": (
            _json_safe(result["matched_records"].to_dict(orient="records"))
            if isinstance(result.get("matched_records"), pd.DataFrame)
            else _json_safe(result.get("matched_records", []))
        ),
        "unmatched_records": (
            {
                key: _json_safe(value.to_dict(orient="records"))
                for key, value in result["unmatched_records"].items()
                if isinstance(value, pd.DataFrame)
            }
            if isinstance(result.get("unmatched_records"), Mapping)
            else _json_safe(result.get("unmatched_records", []))
        ),
    }
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    schema = result.get("schema_report", {})
    if isinstance(schema, Mapping) and isinstance(schema.get("schema_report"), list):
        schema_rows = schema["schema_report"]
    elif isinstance(schema, list):
        schema_rows = schema
    else:
        schema_rows = [dict(schema)] if schema else []
    pd.DataFrame(schema_rows).to_csv(schema_path, index=False)
    records: list[dict[str, Any]] = []
    matched = result.get("matched_records")
    if isinstance(matched, pd.DataFrame):
        records.extend(
            {"match_status": "matched", **_json_safe(row)}
            for row in matched.to_dict(orient="records")
        )
    unmatched = result.get("unmatched_records")
    if isinstance(unmatched, Mapping):
        for side, frame in unmatched.items():
            if isinstance(frame, pd.DataFrame):
                records.extend(
                    {"match_status": f"unmatched_{side}", **_json_safe(row)}
                    for row in frame.to_dict(orient="records")
                )
    pd.DataFrame(records).to_csv(issues_path, index=False)
    return {
        "data": str(data_path),
        "report_json": str(report_path),
        "schema_csv": str(schema_path),
        "records_csv": str(issues_path),
        "data_directory": str(data_root),
        "report_directory": str(reports_root),
    }


def integrate_multiple_csv_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    integration_type: str = "append",
    selected_datasets: Sequence[str] | None = None,
    join_type: str = "inner",
    key_columns: Sequence[str] | str | None = None,
    right_key_columns: Sequence[str] | str | None = None,
    column_mapping: Mapping[str, str] | None = None,
    standardize_column_names: bool = False,
    many_to_many: str = "raise",
    output_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    report_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Run selected CSV integration on cleaned/validated datasets independently."""
    items = _dataset_items(datasets)
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        items = [(key, value) for key, value in items if key in selected]
    if not items:
        raise ValueError("No selected CSV datasets are available for integration.")
    kind = integration_type.lower()
    if kind in {"append", "combine", "combine_rows", "row-wise"}:
        result = combine_csv_files(
            dict(items),
            standardize_column_names=standardize_column_names,
        )
    elif kind in {"merge", "join"}:
        if len(items) < 2:
            raise ValueError("Select at least two datasets to merge.")
        result = merge_csv_datasets(
            items[0][1],
            items[1][1],
            join_type=join_type,
            key_columns=key_columns,
            right_key_columns=right_key_columns,
            column_mapping=column_mapping,
            many_to_many=many_to_many,
            left_name=items[0][0],
            right_name=items[1][0],
        )
        if len(items) > 2:
            merged_data = result["data"]
            additional_reports = []
            last_merge_result = result
            for index, (key, dataset) in enumerate(items[2:], start=2):
                frame, name, _ = _source_parts(dataset, key)
                merge_result = merge_csv_datasets(
                    merged_data,
                    frame,
                    join_type=join_type,
                    key_columns=key_columns,
                    right_key_columns=right_key_columns or key_columns,
                    column_mapping=column_mapping,
                    many_to_many=many_to_many,
                    left_name=f"integrated_{index}",
                    right_name=name,
                )
                merged_data = merge_result["data"]
                additional_reports.append(merge_result["integration_report"])
                last_merge_result = merge_result
            result["data"] = merged_data
            result["matched_records"] = last_merge_result["matched_records"]
            result["unmatched_records"] = last_merge_result["unmatched_records"]
            result["integration_report"]["additional_merge_steps"] = additional_reports
            initial_merge_report = dict(result["integration_report"])
            result["integration_report"]["merge_steps"] = [
                initial_merge_report,
                *additional_reports,
            ]
            result["integration_report"]["final_step_matched_records"] = len(
                last_merge_result["matched_records"]
            )
            result["integration_report"]["final_step_unmatched_left_records"] = len(
                last_merge_result["unmatched_records"]["left"]
            )
            result["integration_report"]["final_step_unmatched_right_records"] = len(
                last_merge_result["unmatched_records"]["right"]
            )
            result["integration_report"]["input_dataset_count"] = len(items)
            result["source_datasets"] = []
            for key, dataset in items:
                frame, source_name, metadata = _source_parts(dataset, key)
                result["source_datasets"].append(
                    {
                        "dataset_name": source_name,
                        "records": len(frame.index),
                        "metadata": metadata,
                    }
                )
    else:
        raise ValueError("integration_type must be 'append' or 'merge'.")
    result["integration_report"]["dataset_name"] = "integrated_csv"
    result["integration_report"]["integration_type"] = kind
    result["integration_report"]["schema_comparison"] = result.get("schema_report")
    if save_outputs:
        result["saved_paths"] = _save_integration(
            result, output_directory, report_directory
        )
    return result
