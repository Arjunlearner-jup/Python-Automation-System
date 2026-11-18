"""Configurable, read-only validation for cleaned API observation DataFrames."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

DEFAULT_REPORT_DIRECTORY = (
        Path(__file__).resolve().parents[2] / "data" / "reports" / "validation" / "api"
)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if getattr(missing, "ndim", 0) == 0 else False


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _issue(
        dataset_name: str,
        row_index: Any,
        row_number: int | None,
        column_name: str,
        value: Any,
        rule: str,
        message: str,
        severity: str = "error",
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "row_index": _json_safe(row_index),
        "row_number": row_number,
        "column_name": column_name,
        "invalid_value": _json_safe(value),
        "validation_rule": rule,
        "error_message": message,
        "severity": severity,
    }


def _column_summary(
        summaries: dict[str, dict[str, Any]],
        column: str,
        rule: str,
        row_count: int,
) -> dict[str, Any]:
    key = f"{column}:{rule}"
    return summaries.setdefault(
        key,
        {
            "column_name": column,
            "validation_rule": rule,
            "total_values_checked": row_count,
            "valid_values": 0,
            "invalid_values": 0,
            "missing_values": 0,
            "error_count": 0,
            "warning_count": 0,
        },
    )


def validate_required_fields(
        dataframe: pd.DataFrame,
        required_columns: Sequence[str] | None = None,
        required_fields: Sequence[str] | None = None,
        *,
        warning_fields: Sequence[str] | None = None,
        dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate required column existence and required non-empty row values."""
    required_column_names = list(dict.fromkeys(map(str, required_columns or [])))
    required_field_names = list(dict.fromkeys(map(str, required_fields or [])))
    warning_field_names = list(dict.fromkeys(map(str, warning_fields or [])))
    required_names = set(required_column_names + required_field_names)
    configured = list(dict.fromkeys(
        required_column_names + required_field_names + warning_field_names
    ))
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    missing_required_columns: list[str] = []
    missing_warning_columns: list[str] = []
    invalid_positions: set[int] = set()

    for column in configured:
        exists = column in dataframe.columns
        severity = "error" if column in required_names else "warning"
        checks_values = column in required_field_names or column in warning_field_names
        summary = _column_summary(summaries, column, "required_field", len(dataframe))
        if not exists:
            summary["total_values_checked"] = 0
            summary["invalid_values"] = 1
            summary["error_count" if severity == "error" else "warning_count"] = 1
            target = missing_required_columns if severity == "error" else missing_warning_columns
            target.append(column)
            issue = _issue(
                dataset_name,
                None,
                None,
                column,
                None,
                "required_column" if severity == "error" else "optional_column",
                f"{'Required' if severity == 'error' else 'Optional'} column {column!r} is missing.",
                severity,
            )
            (errors if severity == "error" else warnings).append(issue)
            if severity == "error":
                invalid_positions.update(range(len(dataframe.index)))
            continue

        if not checks_values:
            summary["valid_values"] = len(dataframe.index)
            continue
        for position, (row_index, value) in enumerate(dataframe[column].items()):
            if _is_missing(value):
                summary["missing_values"] += 1
                summary["invalid_values"] += 1
                summary["error_count" if severity == "error" else "warning_count"] += 1
                (errors if severity == "error" else warnings).append(
                    _issue(
                        dataset_name,
                        row_index,
                        position + 1,
                        column,
                        value,
                        "required_field" if severity == "error" else "optional_field",
                        f"{'Required' if severity == 'error' else 'Optional'} field {column!r} is null or empty.",
                        severity,
                    )
                )
                if severity == "error":
                    invalid_positions.add(position)
            else:
                summary["valid_values"] += 1

    return {
        "missing_required_columns": missing_required_columns,
        "missing_warning_columns": missing_warning_columns,
        "invalid_positions": invalid_positions,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "field_summary": list(summaries.values()),
    }


def _number(value: Any) -> float | int | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _numeric_bounds(
        column: str, rule: Mapping[str, Any]
) -> dict[str, float | None]:
    bounds = {}
    for bound_name in ("min", "max", "minimum", "maximum"):
        if rule.get(bound_name) is None:
            continue
        value = rule[bound_name]
        if isinstance(value, bool):
            raise ValueError(f"{bound_name!r} bound for {column!r} must be numeric.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{bound_name!r} bound for {column!r} must be numeric."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{bound_name!r} bound for {column!r} must be finite.")
        bounds[bound_name] = numeric
    minimum = bounds.get("min", bounds.get("minimum"))
    maximum = bounds.get("max", bounds.get("maximum"))
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"Minimum bound exceeds maximum for {column!r}.")
    return {"min": minimum, "max": maximum}


def validate_numerical_values(
        dataframe: pd.DataFrame,
        numerical_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
        *,
        dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Check configured numeric columns without coercing or modifying values.

    Rules accept ``type`` (``integer`` or ``decimal``), ``min``/``max``
    (or ``minimum``/``maximum``), ``allow_negative``, ``integer_only``, and
    ``severity``. Missing values are counted but left to required-field rules.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_columns: list[str] = []
    invalid_positions: set[int] = set()

    for column, configured_rule in (numerical_rules or {}).items():
        column = str(column)
        rule = {"type": configured_rule} if isinstance(configured_rule, str) else dict(configured_rule)
        number_type = str(rule.get("type", "decimal")).lower()
        if number_type not in {"integer", "int", "decimal", "float", "number"}:
            raise ValueError(f"Unsupported numeric type {number_type!r} for {column!r}.")
        integer_only = bool(rule.get("integer_only", number_type in {"integer", "int"}))
        bounds = _numeric_bounds(column, rule)
        severity = str(rule.get("severity", "error")).lower()
        if severity not in {"error", "warning"}:
            raise ValueError(f"Severity for {column!r} must be 'error' or 'warning'.")
        summary = _column_summary(
            {}, column, "numerical", len(dataframe.index)
        )
        if column not in dataframe.columns:
            summary.update(
                total_values_checked=0,
                invalid_values=1,
                error_count=int(severity == "error"),
                warning_count=int(severity == "warning"),
            )
            missing_columns.append(column)
            (errors if severity == "error" else warnings).append(
                _issue(
                    dataset_name, None, None, column, None,
                    "numerical_column",
                    f"Configured numerical column {column!r} is missing.",
                    severity,
                )
            )
            if severity == "error":
                invalid_positions.update(range(len(dataframe.index)))
            summaries.append(summary)
            continue

        for position, (row_index, value) in enumerate(dataframe[column].items()):
            if _is_missing(value):
                summary["missing_values"] += 1
                continue
            number = _number(value)
            problems = []
            if number is None:
                problems.append(f"Value {value!r} is not a finite number.")
            else:
                if integer_only and not float(number).is_integer():
                    problems.append(f"Value {value!r} must be an integer.")
                if rule.get("allow_negative", True) is False and number < 0:
                    problems.append(f"Negative value {value!r} is not permitted.")
                if bounds["min"] is not None and number < bounds["min"]:
                    problems.append(f"Value {value!r} is below minimum {bounds['min']}.")
                if bounds["max"] is not None and number > bounds["max"]:
                    problems.append(f"Value {value!r} is above maximum {bounds['max']}.")
            if problems:
                summary["invalid_values"] += 1
                summary["error_count" if severity == "error" else "warning_count"] += len(problems)
                (errors if severity == "error" else warnings).extend(
                    _issue(
                        dataset_name, row_index, position + 1, column, value,
                        "numerical", message, severity,
                    )
                    for message in problems
                )
                if severity == "error":
                    invalid_positions.add(position)
            else:
                summary["valid_values"] += 1
        summaries.append(summary)
    return {
        "missing_configured_columns": missing_columns,
        "invalid_positions": invalid_positions,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "field_summary": summaries,
    }


def _date_matches(value: Any, formats: str | Sequence[str] | None) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return not pd.isna(value)
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if formats is None:
        if re.fullmatch(r"\d{4}", text):
            return 1 <= int(text) <= 9999
        if re.fullmatch(r"\d{4}-\d{2}", text):
            try:
                datetime.strptime(text, "%Y-%m")
                return True
            except ValueError:
                return False
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return True
        except ValueError:
            try:
                date.fromisoformat(text)
                return True
            except ValueError:
                return False
    formats_to_try = [formats] if isinstance(formats, str) else list(formats)
    for fmt in formats_to_try:
        if fmt == "%Y":
            matched = re.fullmatch(r"\d{4}", text)
        elif fmt == "%Y-%m":
            matched = re.fullmatch(r"\d{4}-\d{2}", text)
        else:
            matched = True
        if not matched:
            continue
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def validate_date_formats(
        dataframe: pd.DataFrame,
        date_rules: Mapping[str, Mapping[str, Any] | str | Sequence[str]] | None = None,
        *,
        dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate explicit date/time-period columns, preserving original values.

    With no format specified the accepted forms are a four-digit year,
    valid ``YYYY-MM``, ISO date, or ISO datetime. Configure formats explicitly
    for any other representation. A rule can specify ``required`` and
    ``severity``.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_columns: list[str] = []
    invalid_positions: set[int] = set()

    for column, configured_rule in (date_rules or {}).items():
        column = str(column)
        if isinstance(configured_rule, str):
            rule: dict[str, Any] = {"format": configured_rule}
        elif isinstance(configured_rule, Mapping):
            rule = dict(configured_rule)
        else:
            rule = {"formats": list(configured_rule)}
        formats = rule.get("formats", rule.get("format"))
        if formats is not None and not isinstance(formats, str):
            formats = [str(fmt) for fmt in formats]
        severity = str(rule.get("severity", "error")).lower()
        if severity not in {"error", "warning"}:
            raise ValueError(f"Severity for {column!r} must be 'error' or 'warning'.")
        summary = _column_summary({}, column, "date", len(dataframe.index))
        if column not in dataframe.columns:
            summary.update(
                total_values_checked=0,
                invalid_values=int(bool(rule.get("required", False))),
                missing_values=1,
                error_count=int(bool(rule.get("required", False)) and severity == "error"),
                warning_count=int(bool(rule.get("required", False)) and severity == "warning"),
            )
            missing_columns.append(column)
            if rule.get("required", False):
                (errors if severity == "error" else warnings).append(
                    _issue(
                        dataset_name, None, None, column, None, "required_date_column",
                        f"Required date column {column!r} is missing.", severity,
                    )
                )
                if severity == "error":
                    invalid_positions.update(range(len(dataframe.index)))
            summaries.append(summary)
            continue

        for position, (row_index, value) in enumerate(dataframe[column].items()):
            if _is_missing(value):
                summary["missing_values"] += 1
                if not rule.get("required", False):
                    continue
                summary["invalid_values"] += 1
                summary["error_count" if severity == "error" else "warning_count"] += 1
                (errors if severity == "error" else warnings).append(
                    _issue(
                        dataset_name, row_index, position + 1, column, value,
                        "required_date", f"Required date value in {column!r} is missing.",
                        severity,
                    )
                )
                if severity == "error":
                    invalid_positions.add(position)
                continue
            if _date_matches(value, formats):
                summary["valid_values"] += 1
                continue
            summary["invalid_values"] += 1
            summary["error_count" if severity == "error" else "warning_count"] += 1
            expected_formats = (
                [formats] if isinstance(formats, str) else list(formats or [])
            )
            expected = (
                ", ".join(map(str, expected_formats))
                if expected_formats
                else "YYYY, YYYY-MM, or ISO 8601"
            )
            (errors if severity == "error" else warnings).append(
                _issue(
                    dataset_name, row_index, position + 1, column, value,
                    "date_format",
                    f"Value {value!r} is not a valid date/time period; expected {expected}.",
                    severity,
                )
            )
            if severity == "error":
                invalid_positions.add(position)
        summaries.append(summary)
    return {
        "missing_configured_columns": missing_columns,
        "invalid_positions": invalid_positions,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "field_summary": summaries,
    }


def _dataset_parts(
        dataset: Any,
        api_metadata: Mapping[str, Any] | None,
        dataset_name: str | None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if isinstance(dataset, pd.DataFrame):
        dataframe = dataset
        embedded_metadata: Mapping[str, Any] = {}
        embedded_report: Mapping[str, Any] = {}
    elif isinstance(dataset, Mapping):
        dataframe = dataset.get("data")
        embedded_metadata = dataset.get("api_metadata") or dataset.get("metadata") or {}
        embedded_report = dataset.get("report") or {}
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("API validation result must contain a DataFrame under 'data'.")
    else:
        raise TypeError("API validation requires a DataFrame or Stage 3 cleaning result.")
    if not isinstance(embedded_metadata, Mapping):
        embedded_metadata = {}
    if not isinstance(embedded_report, Mapping):
        embedded_report = {}
    metadata = (
        deepcopy(dict(api_metadata))
        if api_metadata is not None
        else deepcopy(dict(embedded_metadata))
    )
    name = dataset_name or str(
        embedded_report.get("dataset_name")
        or metadata.get("dataset_id")
        or metadata.get("database_id")
        or "api_dataset"
    )
    return dataframe, metadata, name


def validate_api_dataframe(
        cleaned_dataframe: pd.DataFrame | Mapping[str, Any],
        api_metadata: Mapping[str, Any] | None = None,
        dataset_name: str | None = None,
        *,
        required_columns: Sequence[str] | None = None,
        required_fields: Sequence[str] | None = None,
        warning_fields: Sequence[str] | None = None,
        numerical_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
        date_rules: Mapping[str, Mapping[str, Any] | str | Sequence[str]] | None = None,
        validate_required: bool = True,
        validate_numerical: bool = True,
        validate_dates: bool = True,
) -> dict[str, Any]:
    """Validate a Stage 3 API DataFrame without modifying it or its metadata."""
    dataframe, metadata, name = _dataset_parts(
        cleaned_dataframe, api_metadata, dataset_name
    )
    original = dataframe.copy(deep=True)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_positions: set[int] = set()
    missing_required_columns: list[str] = []
    missing_warning_columns: list[str] = []
    missing_configured_columns: list[str] = []

    checks = []
    if validate_required:
        checks.append(
            validate_required_fields(
                dataframe,
                required_columns,
                required_fields,
                warning_fields=warning_fields,
                dataset_name=name,
            )
        )
    if validate_numerical:
        checks.append(
            validate_numerical_values(
                dataframe, numerical_rules, dataset_name=name
            )
        )
    if validate_dates:
        checks.append(
            validate_date_formats(dataframe, date_rules, dataset_name=name)
        )
    for result in checks:
        errors.extend(result["validation_errors"])
        warnings.extend(result["validation_warnings"])
        invalid_positions.update(result["invalid_positions"])
        missing_required_columns.extend(result.get("missing_required_columns", []))
        missing_warning_columns.extend(result.get("missing_warning_columns", []))
        missing_configured_columns.extend(result.get("missing_configured_columns", []))
        for item in result["field_summary"]:
            summaries[(item["column_name"], item["validation_rule"])] = item

    if not dataframe.equals(original) or not dataframe.index.equals(original.index):
        raise RuntimeError("API validation unexpectedly modified its input DataFrame.")
    valid_positions = [
        position
        for position in range(len(dataframe.index))
        if position not in invalid_positions
    ]
    invalid_position_list = sorted(invalid_positions)
    valid_df = dataframe.iloc[valid_positions].copy(deep=True)
    invalid_df = dataframe.iloc[invalid_position_list].copy(deep=True)
    valid_df.attrs = dict(dataframe.attrs)
    invalid_df.attrs = dict(dataframe.attrs)
    error_count = sum(issue["severity"] == "error" for issue in errors)
    warning_count = sum(issue["severity"] == "warning" for issue in warnings)
    status = (
        "failed"
        if error_count
        else "passed_with_warnings"
        if warning_count
        else "passed"
    )
    column_summary = list(summaries.values())
    report = {
        "dataset_name": name,
        "api_metadata": _json_safe(metadata),
        "database_id": metadata.get("database_id"),
        "indicator_id": metadata.get("indicator_id"),
        "reference_area": metadata.get("country_reference_area"),
        "total_rows": len(dataframe.index),
        "total_columns": len(dataframe.columns),
        "column_names": [str(column) for column in dataframe.columns],
        "valid_records": len(valid_positions),
        "invalid_records": len(invalid_position_list),
        "total_validation_errors": error_count,
        "total_validation_warnings": warning_count,
        "validation_status": status,
        "missing_required_columns": list(dict.fromkeys(missing_required_columns)),
        "missing_warning_columns": list(dict.fromkeys(missing_warning_columns)),
        "missing_configured_columns": list(dict.fromkeys(missing_configured_columns)),
        "column_summary": column_summary,
        "record_level_errors": errors,
        "record_level_warnings": warnings,
    }
    return {
        "valid_dataframe": valid_df,
        "invalid_dataframe": invalid_df,
        "valid_data": valid_df,
        "invalid_data": invalid_df,
        "validation_errors": pd.DataFrame(
            errors,
            columns=[
                "dataset_name", "row_index", "row_number", "column_name",
                "invalid_value", "validation_rule", "error_message", "severity",
            ],
        ),
        "validation_warnings": pd.DataFrame(
            warnings,
            columns=[
                "dataset_name", "row_index", "row_number", "column_name",
                "invalid_value", "validation_rule", "error_message", "severity",
            ],
        ),
        "validation_report": report,
        "validation_status": status,
        "api_metadata": metadata,
    }


def validate_multiple_api_datasets(
        datasets: Mapping[str, Any] | Sequence[Any],
        *,
        rules_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
        **validation_options: Any,
) -> dict[str, dict[str, Any]]:
    """Validate cleaned API DataFrames independently, preserving dataset keys."""
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(enumerate(datasets))
    results: dict[str, dict[str, Any]] = {}
    for identifier, dataset in items:
        key = str(identifier)
        report: Mapping[str, Any] = {}
        if isinstance(dataset, Mapping):
            candidate_report = dataset.get("report", {})
            report = candidate_report if isinstance(candidate_report, Mapping) else {}
            metadata = dataset.get("api_metadata", dataset.get("metadata", {}))
            if report.get("cleaning_status") == "error":
                continue
            if isinstance(metadata, Mapping) and metadata.get("status") == "error":
                continue
        options = dict(validation_options)
        options.update((rules_by_dataset or {}).get(key, {}))
        name = str(report.get("dataset_name") or key)
        results[key] = validate_api_dataframe(dataset, dataset_name=name, **options)
    return results


def export_api_validation_report(
        result: Mapping[str, Any],
        output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, str]:
    """Save detailed JSON and CSV reports in a unique API validation directory."""
    report = result.get("validation_report")
    if not isinstance(report, Mapping):
        raise TypeError("Validation result must contain 'validation_report'.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        Path(str(report.get("dataset_name") or "api_dataset")).stem,
    ).strip("._") or "api_dataset"
    run_directory = directory / f"{name}_{uuid4().hex[:8]}"
    run_directory.mkdir()
    json_path = run_directory / "validation_report.json"
    errors_path = run_directory / "validation_errors.csv"
    summary_path = run_directory / "column_summary.csv"
    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    issues = pd.concat(
        [
            frame
            for frame in (
            result.get("validation_errors"),
            result.get("validation_warnings"),
        )
            if isinstance(frame, pd.DataFrame)
        ],
        ignore_index=True,
    ) if any(
        isinstance(result.get(key), pd.DataFrame)
        for key in ("validation_errors", "validation_warnings")
    ) else pd.DataFrame(
        list(report.get("record_level_errors", []))
        + list(report.get("record_level_warnings", []))
    )
    issues.to_csv(errors_path, index=False)
    pd.DataFrame(report.get("column_summary", [])).to_csv(summary_path, index=False)
    return {
        "directory": str(run_directory),
        "json": str(json_path),
        "csv": str(errors_path),
        "column_summary_csv": str(summary_path),
    }


def export_api_validation_data(
        result: Mapping[str, Any],
        output_directory: str | Path,
) -> dict[str, str]:
    """Write valid and invalid observations separately as CSV files."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        Path(str(result.get("validation_report", {}).get("dataset_name", "api_dataset"))).stem,
    ).strip("._") or "api_dataset"
    paths = {}
    for key, field, filename in (
            ("valid_csv", "valid_dataframe", f"{name}_valid.csv"),
            ("invalid_csv", "invalid_dataframe", f"{name}_invalid.csv"),
    ):
        frame = result.get(field)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Validation result must contain DataFrame {field!r}.")
        target = directory / filename
        frame.to_csv(target, index=True, index_label="row_index")
        paths[key] = str(target)
    return paths
