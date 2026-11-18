"""Configurable, read-only validation for cleaned CSV DataFrames."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "validation" / "csv"
)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if getattr(result, "ndim", 0) == 0 else False


def _row_position(index: Any, position: int) -> dict[str, Any]:
    return {"row_index": index, "row_number": position + 1}


def _make_issue(
    dataset_name: str,
    row_index: Any,
    row_number: int | None,
    column: str,
    value: Any,
    rule: str,
    message: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "row_index": row_index,
        "row_number": row_number,
        "column_name": column,
        "invalid_value": value,
        "validation_rule": rule,
        "error_message": message,
        "severity": severity,
    }


def validate_required_fields(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
    required_fields: Sequence[str] | None = None,
    *,
    warning_fields: Sequence[str] | None = None,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Check required column existence and required-field values.

    ``required_columns`` only checks schema presence. ``required_fields`` checks
    both presence and per-record non-missing values.
    """
    columns_required = list(dict.fromkeys(required_columns or []))
    fields_required = list(dict.fromkeys(required_fields or []))
    fields_warn = list(dict.fromkeys(warning_fields or []))
    required_names = set(columns_required + fields_required)
    missing_columns = [
        column
        for column in dict.fromkeys(columns_required + fields_required)
        if column not in df.columns
    ]
    missing_warning_columns = [
        column
        for column in fields_warn
        if column not in required_names and column not in df.columns
    ]
    errors = []
    field_summary = []
    for column in dict.fromkeys(columns_required + fields_required + fields_warn):
        exists = column in df.columns
        check_values = exists and column in fields_required + fields_warn
        severity = "error" if column in required_names else "warning"
        missing_column = column not in df.columns
        missing_mask = (
            df[column].map(_is_missing)
            if check_values
            else pd.Series(False, index=df.index, dtype=bool)
        )
        field_summary.append(
            {
                "column_name": column,
                "validation_rule": (
                    "required_field" if severity == "error" else "optional_field"
                ),
                "total_values_checked": len(df.index) if exists else 0,
                "valid_values": int(len(df.index) - missing_mask.sum())
                if exists
                else 0,
                "invalid_values": int(missing_mask.sum()),
                "missing_values": int(missing_mask.sum()),
                "error_count": (
                    int(missing_mask.sum()) + int(missing_column)
                    if severity == "error"
                    else 0
                ),
                "warning_count": (
                    int(missing_mask.sum()) + int(missing_column)
                    if severity == "warning"
                    else 0
                ),
            }
        )
        if column in missing_columns:
            errors.append(
                {
                    "dataset_name": dataset_name,
                    "row_index": None,
                    "row_number": None,
                    "column_name": column,
                    "invalid_value": None,
                    "validation_rule": "required_column",
                    "error_message": f"Required column {column!r} is missing.",
                    "severity": "error",
                }
            )
        elif missing_column:
            errors.append(
                _make_issue(
                    dataset_name,
                    None,
                    None,
                    column,
                    None,
                    "optional_column",
                    f"Optional column {column!r} is missing.",
                    "warning",
                )
            )
        if check_values:
            for position, (index, is_missing) in enumerate(missing_mask.items()):
                if is_missing:
                    errors.append(
                        _make_issue(
                            dataset_name,
                            index,
                            position + 1,
                            column,
                            df[column].iloc[position],
                            "required_field" if severity == "error" else "optional_field",
                            (
                                f"Required field {column!r} is missing."
                                if severity == "error"
                                else f"Optional field {column!r} is missing."
                            ),
                            severity,
                        )
                    )

    return {
        "missing_required_columns": missing_columns,
        "missing_warning_columns": missing_warning_columns,
        "validation_errors": errors,
        "field_summary": field_summary,
    }


def _number(value: Any) -> float | int | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        try:
            return float(value) if math.isfinite(float(value)) else None
        except OverflowError:
            return None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def validate_numerical_values(
    df: pd.DataFrame,
    numerical_rules: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate configured numeric columns without coercing or changing them.

    Each column rule may contain ``min``, ``max``, ``allow_negative`` (default
    true), and ``integer_only`` (default false).
    """
    errors = []
    field_summary = []
    missing_columns = []
    for column, rule in (numerical_rules or {}).items():
        initial_error_count = len(errors)
        bounds = {}
        for bound in ("min", "max"):
            configured_bound = rule.get(bound)
            if configured_bound is None:
                continue
            try:
                if isinstance(configured_bound, bool):
                    raise ValueError
                numeric_bound = float(str(configured_bound))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{bound!r} bound for {column!r} must be numeric."
                ) from exc
            if not math.isfinite(numeric_bound):
                raise ValueError(f"{bound!r} bound for {column!r} must be finite.")
            bounds[bound] = numeric_bound
        if (
            "min" in bounds
            and "max" in bounds
            and bounds["min"] > bounds["max"]
        ):
            raise ValueError(f"Minimum bound exceeds maximum for {column!r}.")
        if column not in df.columns:
            missing_columns.append(column)
            errors.append(
                _make_issue(
                    dataset_name,
                    None,
                    None,
                    column,
                    None,
                    "numerical_column",
                    f"Configured numerical column {column!r} is missing.",
                )
            )
            field_summary.append(
                {
                    "column_name": column,
                    "validation_rule": "numerical",
                    "total_values_checked": 0,
                    "valid_values": 0,
                    "invalid_values": 0,
                    "missing_values": 0,
                    "error_count": 1,
                }
            )
            continue

        invalid_count = 0
        missing_count = 0
        values_checked = 0
        for position, (index, value) in enumerate(df[column].items()):
            if _is_missing(value):
                missing_count += 1
                continue
            values_checked += 1
            number = _number(value)
            messages = []
            if number is None:
                messages.append(f"Value {value!r} is not a finite number.")
            else:
                if not rule.get("allow_negative", True) and number < 0:
                    messages.append(f"Negative value {value!r} is not permitted.")
                if rule.get("integer_only", False) and not float(number).is_integer():
                    messages.append(f"Value {value!r} must be an integer.")
                if "min" in bounds and number < bounds["min"]:
                    messages.append(
                        f"Value {value!r} is below minimum {bounds['min']}."
                    )
                if "max" in bounds and number > bounds["max"]:
                    messages.append(
                        f"Value {value!r} is above maximum {bounds['max']}."
                    )
            if messages:
                invalid_count += 1
                for message in messages:
                    errors.append(
                        _make_issue(
                            dataset_name,
                            index,
                            position + 1,
                            column,
                            value,
                            "numerical",
                            message,
                        )
                    )
        field_summary.append(
            {
                "column_name": column,
                "validation_rule": "numerical",
                "total_values_checked": len(df.index),
                "valid_values": len(df.index) - missing_count - invalid_count,
                "invalid_values": invalid_count,
                "missing_values": missing_count,
                "error_count": len(errors) - initial_error_count,
            }
        )
    return {
        "missing_configured_columns": missing_columns,
        "validation_errors": errors,
        "field_summary": field_summary,
    }


def _date_matches(value: Any, formats: str | Sequence[str] | None) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return not pd.isna(value)
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if formats is None:
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    elif isinstance(formats, str):
        formats_to_try = [formats]
    else:
        formats_to_try = list(formats)
    for date_format in formats_to_try:
        try:
            datetime.strptime(text, date_format)
            return True
        except ValueError:
            continue
    return False


def validate_date_formats(
    df: pd.DataFrame,
    date_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
    *,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate explicitly configured date columns without modifying values.

    A date rule may be a format string or a mapping with ``format`` or
    ``formats``. Set ``year_only=True`` to accept only four-digit years.
    """
    errors = []
    field_summary = []
    missing_columns = []
    for column, configured_rule in (date_rules or {}).items():
        rule = {"format": configured_rule} if isinstance(configured_rule, str) else (
            dict(configured_rule) if configured_rule is not None else {}
        )
        if column not in df.columns:
            missing_columns.append(column)
            errors.append(
                _make_issue(
                    dataset_name,
                    None,
                    None,
                    column,
                    None,
                    "date_column",
                    f"Configured date column {column!r} is missing.",
                )
            )
            field_summary.append(
                {
                    "column_name": column,
                    "validation_rule": "date_format",
                    "total_values_checked": 0,
                    "valid_values": 0,
                    "invalid_values": 0,
                    "missing_values": 0,
                    "error_count": 1,
                }
            )
            continue

        formats = rule.get("formats", rule.get("format"))
        year_only = bool(rule.get("year_only", False))
        invalid_count = 0
        missing_count = 0
        values_checked = 0
        for position, (index, value) in enumerate(df[column].items()):
            if _is_missing(value):
                missing_count += 1
                continue
            values_checked += 1
            if year_only:
                valid = isinstance(value, (str, int)) and re.fullmatch(
                    r"\d{4}", str(value).strip()
                ) is not None
            else:
                valid = _date_matches(value, formats)
            if not valid:
                invalid_count += 1
                expected = "a four-digit year" if year_only else (
                    formats or "an ISO 8601 date"
                )
                errors.append(
                    _make_issue(
                        dataset_name,
                        index,
                        position + 1,
                        column,
                        value,
                        "date_format",
                        f"Value {value!r} is not a valid date matching {expected!r}.",
                    )
                )
        field_summary.append(
            {
                "column_name": column,
                "validation_rule": "date_format",
                "total_values_checked": len(df.index),
                "valid_values": len(df.index) - missing_count - invalid_count,
                "invalid_values": invalid_count,
                "missing_values": missing_count,
                "error_count": invalid_count,
            }
        )
    return {
        "missing_configured_columns": missing_columns,
        "validation_errors": errors,
        "field_summary": field_summary,
    }


def _records_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=[
        "dataset_name",
        "row_index",
        "row_number",
        "column_name",
        "invalid_value",
        "validation_rule",
        "error_message",
        "severity",
    ])


def validate_csv_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    *,
    required_columns: Sequence[str] | None = None,
    required_fields: Sequence[str] | None = None,
    warning_fields: Sequence[str] | None = None,
    numerical_rules: Mapping[str, Mapping[str, Any]] | None = None,
    date_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
    validate_required: bool = True,
    validate_numerical: bool = True,
    validate_dates: bool = True,
) -> dict[str, Any]:
    """Validate a cleaned CSV DataFrame and return separated records and reports.

    The input is never modified. Original index labels are retained, and
    ``row_number`` is the one-based position in the input.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_required_columns: list[str] = []
    missing_warning_columns: list[str] = []
    missing_configured_columns: list[str] = []

    if validate_required:
        result = validate_required_fields(
            df,
            required_columns,
            required_fields,
            warning_fields=warning_fields,
            dataset_name=dataset_name,
        )
        errors.extend(
            issue for issue in result["validation_errors"] if issue["severity"] == "error"
        )
        warnings.extend(
            issue
            for issue in result["validation_errors"]
            if issue["severity"] == "warning"
        )
        summaries.extend(result["field_summary"])
        missing_required_columns.extend(result["missing_required_columns"])
        missing_warning_columns.extend(result["missing_warning_columns"])
    if validate_numerical:
        result = validate_numerical_values(
            df, numerical_rules, dataset_name=dataset_name
        )
        errors.extend(result["validation_errors"])
        summaries.extend(result["field_summary"])
        missing_configured_columns.extend(result["missing_configured_columns"])
    if validate_dates:
        result = validate_date_formats(df, date_rules, dataset_name=dataset_name)
        errors.extend(result["validation_errors"])
        summaries.extend(result["field_summary"])
        missing_configured_columns.extend(result["missing_configured_columns"])

    errors_by_position: dict[int, list[str]] = {}
    for issue in errors:
        if issue["row_number"] is not None:
            errors_by_position.setdefault(issue["row_number"] - 1, []).append(
                issue["error_message"]
            )
    invalid_positions = set(errors_by_position)
    valid_positions = [
        position
        for position in range(len(df.index))
        if position not in invalid_positions
    ]
    invalid_position_list = sorted(invalid_positions)
    warning_positions = {
        issue["row_number"] - 1
        for issue in warnings
        if issue["row_number"] is not None
    }
    valid_df = df.iloc[valid_positions].copy(deep=True)
    invalid_df = df.iloc[invalid_position_list].copy(deep=True)
    status_column = "_validation_status"
    while status_column in df.columns:
        status_column += "_"
    errors_column = "_validation_errors"
    while errors_column in df.columns or errors_column == status_column:
        errors_column += "_"
    valid_df[status_column] = [
        "valid_with_warnings" if position in warning_positions else "valid"
        for position in valid_positions
    ]
    invalid_df[status_column] = "invalid"
    invalid_df[errors_column] = [
        "; ".join(errors_by_position[position])
        for position in invalid_position_list
    ]

    valid_count = len(valid_positions)
    invalid_count = len(invalid_position_list)
    error_count = sum(issue["severity"] == "error" for issue in errors)
    warning_count = sum(issue["severity"] == "warning" for issue in warnings)
    status = "failed" if error_count else "passed_with_warnings" if warning_count else "passed"
    report = {
        "dataset_name": dataset_name,
        "total_rows": len(df.index),
        "total_columns": len(df.columns),
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "total_validation_errors": error_count,
        "total_validation_warnings": warning_count,
        "validation_status": status,
        "missing_required_columns": list(dict.fromkeys(missing_required_columns)),
        "missing_warning_columns": list(dict.fromkeys(missing_warning_columns)),
        "missing_configured_columns": list(dict.fromkeys(missing_configured_columns)),
        "field_summary": summaries,
        "record_level_errors": errors,
        "record_level_warnings": warnings,
        "record_status_column": status_column,
        "record_errors_column": errors_column,
    }
    return {
        "valid_dataframe": valid_df,
        "invalid_dataframe": invalid_df,
        "validation_errors": _records_frame(errors),
        "validation_warnings": _records_frame(warnings),
        "validation_report": report,
        "validation_status": status,
    }


def _dataset_parts(dataset: Any, identifier: str) -> tuple[pd.DataFrame, str]:
    if isinstance(dataset, pd.DataFrame):
        return dataset, identifier
    if isinstance(dataset, Mapping):
        dataframe = dataset.get("data")
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Cleaned CSV dataset {identifier!r} has no DataFrame.")
        report = dataset.get("report", {})
        name = str(report.get("dataset_name") or identifier) if isinstance(
            report, Mapping
        ) else identifier
        return dataframe, name
    raise TypeError("Datasets must be DataFrames or Stage 3 cleaning results.")


def validate_multiple_csv_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    rules_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
    **validation_options: Any,
) -> dict[str, dict[str, Any]]:
    """Validate cleaned datasets independently, retaining their keys and names."""
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(
        enumerate(datasets)
    )
    results = {}
    for identifier, dataset in items:
        key = str(identifier)
        if isinstance(dataset, Mapping):
            cleaning_report = dataset.get("report", {})
            if isinstance(cleaning_report, Mapping) and cleaning_report.get(
                "cleaning_status"
            ) == "error":
                continue
        dataframe, name = _dataset_parts(dataset, key)
        options = dict(validation_options)
        options.update((rules_by_dataset or {}).get(key, {}))
        results[key] = validate_csv_dataframe(dataframe, name, **options)
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_validation_report(
    result: Mapping[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, str]:
    """Export a validation report and record-level issues as JSON and CSV."""
    report = result.get("validation_report")
    if not isinstance(report, Mapping):
        raise TypeError("Validation result must contain 'validation_report'.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset_name = str(report.get("dataset_name") or "dataset")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dataset_name).stem).strip(
        "._"
    ) or "dataset"
    run_directory = directory / f"{safe_name}_{uuid4().hex[:8]}"
    run_directory.mkdir()
    json_path = run_directory / "validation_report.json"
    csv_path = run_directory / "validation_errors.csv"
    summary_csv_path = run_directory / "field_summary.csv"
    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    issue_frames = [
        frame
        for frame in (
            result.get("validation_errors"),
            result.get("validation_warnings"),
        )
        if isinstance(frame, pd.DataFrame)
    ]
    issue_frame = (
        pd.concat(issue_frames, ignore_index=True)
        if issue_frames
        else _records_frame(
            list(report.get("record_level_errors", []))
            + list(report.get("record_level_warnings", []))
        )
    )
    issue_frame.to_csv(csv_path, index=False)
    pd.DataFrame(report.get("field_summary", [])).to_csv(
        summary_csv_path, index=False
    )
    return {
        "directory": str(run_directory),
        "json": str(json_path),
        "csv": str(csv_path),
        "field_summary_csv": str(summary_csv_path),
    }
