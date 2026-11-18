"""Configurable, read-only validation for cleaned nested JSON data."""

from __future__ import annotations

import copy
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
    Path(__file__).resolve().parents[2] / "data" / "reports" / "validation" / "json"
)


def _is_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and not value.strip()
    )


def _walk(value: Any, path: str = "$"):
    """Yield every node, including containers, with its stable JSON path."""
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _normalise_pattern(pattern: str) -> str:
    if pattern == "$" or pattern.startswith("$.") or pattern.startswith("$["):
        return pattern
    if pattern.startswith("["):
        return "$" + pattern
    return "$." + pattern


def _matches(path: str, pattern: str) -> bool:
    pattern = _normalise_pattern(pattern)
    expression = re.escape(pattern).replace(r"\[\*\]", r"\[\d+\]")
    expression = expression.replace(r"\*", r"[^.\[\]]+")
    return re.fullmatch(expression, path) is not None


def _matching_nodes(data: Any, pattern: str) -> list[tuple[str, Any]]:
    return [
        (path, value)
        for path, value in _walk(data)
        if _matches(path, pattern)
    ]


def discover_json_structure(data: Any) -> dict[str, Any]:
    """Return canonical object, array, and scalar path choices for user interfaces."""
    object_fields: dict[str, set[str]] = {}
    arrays = set()
    scalar_paths = set()
    for path, value in _walk(data):
        canonical = re.sub(r"\[\d+\]", "[*]", path)
        if isinstance(value, Mapping):
            object_fields.setdefault(canonical, set()).update(map(str, value.keys()))
        elif isinstance(value, list):
            arrays.add(canonical)
        else:
            scalar_paths.add(canonical)
    return {
        "object_fields": {
            path: sorted(fields) for path, fields in object_fields.items()
        },
        "array_paths": sorted(arrays),
        "scalar_paths": sorted(scalar_paths),
    }


def _issue(
    dataset_name: str,
    path: str,
    value: Any,
    rule: str,
    message: str,
    severity: str = "error",
    record_path: str | None = None,
    record_index: int | None = None,
) -> dict[str, Any]:
    terminal = path.rsplit(".", 1)[-1]
    field_match = re.match(r"([^[]+)", terminal)
    if field_match is None and "." in path:
        field_match = re.match(r"([^[]+)", path.rsplit(".", 2)[-2])
    return {
        "dataset_name": dataset_name,
        "record_index": record_index,
        "record_path": record_path,
        "json_path": path,
        "field_name": (
            field_match.group(1).lstrip("$") if field_match else ""
        ),
        "invalid_value": copy.deepcopy(value),
        "validation_rule": rule,
        "error_message": message,
        "severity": severity,
    }


def _record_arrays(data: Any) -> list[tuple[str, list[Any]]]:
    return [
        (path, value)
        for path, value in _walk(data)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value)
    ]


def _record_owner(path: str, arrays: Sequence[tuple[str, list[Any]]]):
    owners = []
    for array_path, records in arrays:
        is_object_array = all(isinstance(item, Mapping) for item in records)
        for index in range(len(records)):
            record_path = f"{array_path}[{index}]"
            if path == record_path or path.startswith(record_path + ".") or path.startswith(
                record_path + "["
            ):
                owners.append(
                    (int(is_object_array), len(record_path), record_path, index)
                )
    if not owners:
        return "$", None
    _, _, record_path, index = max(owners)
    return record_path, index


def _attach_owners(
    issues: list[dict[str, Any]], arrays: Sequence[tuple[str, list[Any]]]
) -> list[dict[str, Any]]:
    for issue in issues:
        owner_path, index = _record_owner(issue["json_path"], arrays)
        issue["record_path"] = owner_path
        issue["record_index"] = index
    return issues


def validate_required_fields(
    data: Any,
    required_fields: Mapping[str, Sequence[str]],
    *,
    warning_fields: Mapping[str, Sequence[str]] | None = None,
    required_arrays: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Check configured keys on matching objects and required array structure.

    ``required_fields`` maps an object path (wildcards support array indexes)
    to required keys, e.g. ``{"currencies[*]": ["code", "name"]}``.
    ``required_arrays`` accepts path rules with ``required``, ``allow_empty``,
    ``item_type`` and ``items_type``.
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    seen_parent_patterns = set()

    configured_fields = [
        (parent_pattern, fields, "error")
        for parent_pattern, fields in (required_fields or {}).items()
    ] + [
        (parent_pattern, fields, "warning")
        for parent_pattern, fields in (warning_fields or {}).items()
    ]
    for parent_pattern, fields, severity in configured_fields:
        normalized = _normalise_pattern(parent_pattern)
        parents = _matching_nodes(data, normalized)
        seen_parent_patterns.add(normalized)
        if not parents:
            errors.append(
                _issue(
                    dataset_name,
                    normalized,
                    None,
                    "required_object",
                    f"Required object location {normalized!r} was not found.",
                )
            )
            continue
        for parent_path, parent in parents:
            if not isinstance(parent, Mapping):
                errors.append(
                    _issue(
                        dataset_name,
                        parent_path,
                        parent,
                        "object_type",
                        f"Expected an object at {parent_path!r}.",
                    )
                )
                continue
            for field in fields:
                field_path = f"{parent_path}.{field}"
                key = (str(field), "required_field")
                item = summary.setdefault(
                    key,
                    {
                        "field_name": str(field),
                        "json_path_pattern": f"{normalized}.{field}",
                        "validation_rule": "required_field",
                        "total_values_checked": 0,
                        "valid_values": 0,
                        "invalid_values": 0,
                        "missing_values": 0,
                        "error_count": 0,
                        "warning_count": 0,
                    },
                )
                item["total_values_checked"] += 1
                if field not in parent:
                    item["invalid_values"] += 1
                    item["missing_values"] += 1
                    item["error_count" if severity == "error" else "warning_count"] += 1
                    issue_list = errors if severity == "error" else warnings
                    issue_list.append(
                        _issue(
                            dataset_name,
                            field_path,
                            None,
                            "required_key",
                            f"Required key {field!r} is missing.",
                            severity,
                        )
                    )
                elif _is_missing(parent[field]):
                    item["invalid_values"] += 1
                    item["missing_values"] += 1
                    item["error_count" if severity == "error" else "warning_count"] += 1
                    issue_list = errors if severity == "error" else warnings
                    issue_list.append(
                        _issue(
                            dataset_name,
                            field_path,
                            parent[field],
                            "required_value",
                            f"Field {field!r} is null, empty, or whitespace-only.",
                            severity,
                        )
                    )
                else:
                    item["valid_values"] += 1

    for pattern, rule in (required_arrays or {}).items():
        normalized = _normalise_pattern(pattern)
        nodes = _matching_nodes(data, normalized)
        if not nodes:
            if rule.get("required", True):
                errors.append(
                    _issue(
                        dataset_name,
                        normalized,
                        None,
                        "required_array",
                        f"Required array {normalized!r} is missing.",
                    )
                )
            continue
        for path, value in nodes:
            invalid = False
            message = ""
            if not isinstance(value, list):
                invalid, message = True, f"Expected an array at {path!r}."
            elif not value and not rule.get("allow_empty", True):
                invalid, message = True, f"Array at {path!r} must not be empty."
            else:
                expected = rule.get("item_type")
                if expected:
                    expected_type = {
                        "object": Mapping,
                        "array": list,
                        "string": str,
                        "integer": int,
                        "number": Real,
                        "boolean": bool,
                    }.get(str(expected).lower())
                    if expected_type is None:
                        raise ValueError(f"Unsupported JSON item_type: {expected!r}")
                    if any(
                        not isinstance(item, expected_type)
                        or (expected_type is int and isinstance(item, bool))
                        or (expected_type is Real and isinstance(item, bool))
                        for item in value
                    ):
                        invalid = True
                        message = f"Every item in {path!r} must be a {expected}."
            if invalid:
                errors.append(
                    _issue(dataset_name, path, value, "array_structure", message)
                )

        if rule.get("items_type"):
            expected = str(rule["items_type"]).lower()
            expected_type = {
                "object": Mapping,
                "array": list,
                "string": str,
                "integer": int,
                "number": Real,
                "boolean": bool,
            }.get(expected)
            if expected_type is None:
                raise ValueError(f"Unsupported JSON items_type: {expected!r}")
            for path, value in nodes:
                if not isinstance(value, list):
                    continue
                for index, item in enumerate(value):
                    if not isinstance(item, expected_type) or (
                        expected_type is int and isinstance(item, bool)
                    ):
                        errors.append(
                            _issue(
                                dataset_name,
                                f"{path}[{index}]",
                                item,
                                "array_item_type",
                                f"Array item must be a {expected}.",
                            )
                        )

    arrays = [
        (path, value) for path, value in _walk(data) if isinstance(value, list)
    ]
    _attach_owners(errors, arrays)
    _attach_owners(warnings, arrays)
    return {
        "validation_errors": errors,
        "validation_warnings": warnings,
        "field_summary": list(summary.values()),
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or _is_missing(value):
        return None
    if isinstance(value, Real):
        try:
            number = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def validate_numerical_values(
    data: Any,
    numerical_rules: Mapping[str, Mapping[str, Any]],
    *,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Check configured nested numeric paths without converting source values."""
    errors: list[dict[str, Any]] = []
    summary = []
    for pattern, rule in (numerical_rules or {}).items():
        nodes = _matching_nodes(data, pattern)
        initial_errors = len(errors)
        if not nodes:
            errors.append(
                _issue(
                    dataset_name,
                    _normalise_pattern(pattern),
                    None,
                    "numerical_field",
                    f"Configured numerical field {pattern!r} was not found.",
                )
            )
            summary.append(
                {
                    "field_name": pattern.rsplit(".", 1)[-1],
                    "json_path_pattern": _normalise_pattern(pattern),
                    "validation_rule": "numerical",
                    "total_values_checked": 0,
                    "valid_values": 0,
                    "invalid_values": 0,
                    "missing_values": 0,
                    "error_count": 1,
                }
            )
            continue
        missing_count = 0
        invalid_count = 0
        for path, value in nodes:
            if _is_missing(value):
                missing_count += 1
                if rule.get("required", False):
                    invalid_count += 1
                    errors.append(
                        _issue(
                            dataset_name,
                            path,
                            value,
                            "numerical_required",
                            "A numerical value is required.",
                        )
                    )
                continue
            number = _as_number(value)
            messages = []
            if number is None:
                messages.append(f"Value {value!r} is not a finite number.")
            else:
                if rule.get("type") == "integer" or rule.get("integer_only", False):
                    if not float(number).is_integer():
                        messages.append(f"Value {value!r} must be an integer.")
                if not rule.get("allow_negative", True) and number < 0:
                    messages.append(f"Negative value {value!r} is not permitted.")
                for bound in ("min", "max"):
                    if rule.get(bound) is not None:
                        try:
                            limit = float(rule[bound])
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"{bound!r} bound for {pattern!r} must be numeric."
                            ) from exc
                        if not math.isfinite(limit):
                            raise ValueError(
                                f"{bound!r} bound for {pattern!r} must be finite."
                            )
                        if (
                            bound == "max"
                            and rule.get("min") is not None
                            and limit < float(rule["min"])
                        ):
                            raise ValueError(
                                f"Minimum bound exceeds maximum for {pattern!r}."
                            )
                        if bound == "min" and number < limit:
                            messages.append(
                                f"Value {value!r} is below minimum {limit}."
                            )
                        if bound == "max" and number > limit:
                            messages.append(
                                f"Value {value!r} is above maximum {limit}."
                            )
            if messages:
                invalid_count += 1
                errors.extend(
                    _issue(dataset_name, path, value, "numerical", message)
                    for message in messages
                )
        summary.append(
            {
                "field_name": pattern.rsplit(".", 1)[-1],
                "json_path_pattern": _normalise_pattern(pattern),
                "validation_rule": "numerical",
                "total_values_checked": len(nodes),
                "valid_values": len(nodes) - missing_count - invalid_count,
                "invalid_values": invalid_count,
                "missing_values": missing_count,
                "error_count": len(errors) - initial_errors,
            }
        )
    all_lists = [
        (path, value) for path, value in _walk(data) if isinstance(value, list)
    ]
    _attach_owners(errors, all_lists)
    return {"validation_errors": errors, "field_summary": summary}


def _date_valid(value: Any, formats: str | Sequence[str] | None) -> bool:
    if isinstance(value, (datetime, date)):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if formats is None:
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    selected_formats = [formats] if isinstance(formats, str) else list(formats)
    for date_format in selected_formats:
        try:
            datetime.strptime(text, date_format)
            return True
        except ValueError:
            continue
    return False


def validate_date_formats(
    data: Any,
    date_rules: Mapping[str, Mapping[str, Any] | str],
    *,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Validate configured nested date fields, including ISO 8601 values."""
    errors: list[dict[str, Any]] = []
    summary = []
    for pattern, configured in (date_rules or {}).items():
        rule = {"format": configured} if isinstance(configured, str) else dict(configured)
        nodes = _matching_nodes(data, pattern)
        initial_errors = len(errors)
        if not nodes:
            errors.append(
                _issue(
                    dataset_name,
                    _normalise_pattern(pattern),
                    None,
                    "date_field",
                    f"Configured date field {pattern!r} was not found.",
                )
            )
            summary.append(
                {
                    "field_name": pattern.rsplit(".", 1)[-1],
                    "json_path_pattern": _normalise_pattern(pattern),
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
        missing_count = 0
        invalid_count = 0
        for path, value in nodes:
            if _is_missing(value):
                missing_count += 1
                if rule.get("required", False):
                    invalid_count += 1
                    errors.append(
                        _issue(
                            dataset_name,
                            path,
                            value,
                            "date_required",
                            "A date value is required.",
                        )
                    )
                continue
            if not _date_valid(value, formats):
                invalid_count += 1
                errors.append(
                    _issue(
                        dataset_name,
                        path,
                        value,
                        "date_format",
                        f"Value {value!r} does not match expected date format "
                        f"{formats or 'ISO 8601'!r}.",
                    )
                )
        summary.append(
            {
                "field_name": pattern.rsplit(".", 1)[-1],
                "json_path_pattern": _normalise_pattern(pattern),
                "validation_rule": "date_format",
                "total_values_checked": len(nodes),
                "valid_values": len(nodes) - missing_count - invalid_count,
                "invalid_values": invalid_count,
                "missing_values": missing_count,
                "error_count": len(errors) - initial_errors,
            }
        )
    arrays = [
        (path, value) for path, value in _walk(data) if isinstance(value, list)
    ]
    _attach_owners(errors, arrays)
    return {"validation_errors": errors, "field_summary": summary}


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


def _json_paths_for_records(data: Any) -> list[tuple[str, int, Any]]:
    records: list[tuple[str, int, Any]] = []
    for array_path, values in _record_arrays(data):
        records.extend(
            (f"{array_path}[{index}]", index, value)
            for index, value in enumerate(values)
        )
    if isinstance(data, list) and not all(isinstance(item, Mapping) for item in data):
        return records
    return records


def _split_data(
    data: Any,
    path: str,
    issue_records: set[str],
    mode: str,
) -> tuple[Any, bool]:
    """Split record arrays recursively without marking ancestors for child errors."""
    if isinstance(data, Mapping):
        output = {}
        any_records = False
        for key, value in data.items():
            child, has_selected = _split_data(
                value, f"{path}.{key}", issue_records, mode
            )
            output[key] = child
            any_records = any_records or has_selected
        return output, any_records
    if isinstance(data, list):
        if all(isinstance(item, Mapping) for item in data):
            output = []
            for index, item in enumerate(data):
                item_path = f"{path}[{index}]"
                directly_invalid = item_path in issue_records
                if mode == "valid" and directly_invalid:
                    continue
                if mode == "invalid" and not directly_invalid:
                    nested, has_selected = _split_data(
                        item, item_path, issue_records, mode
                    )
                    if has_selected:
                        output.append(nested)
                    continue
                if directly_invalid:
                    output.append(copy.deepcopy(item))
                else:
                    nested, has_selected = _split_data(
                        item, item_path, issue_records, mode
                    )
                    output.append(nested)
            return output, bool(output)
        output = []
        has_selected = False
        for index, item in enumerate(data):
            child_path = f"{path}[{index}]"
            if mode == "invalid" and child_path in issue_records:
                output.append(copy.deepcopy(item))
                has_selected = True
            elif mode == "valid" and child_path in issue_records:
                continue
            elif isinstance(item, (Mapping, list)):
                child, selected = _split_data(item, child_path, issue_records, mode)
                output.append(child)
                has_selected = has_selected or selected
            elif mode == "valid":
                output.append(copy.deepcopy(item))
                has_selected = True
        return output, has_selected
    return copy.deepcopy(data), mode == "valid"


def _top_record_count(data: Any) -> int | None:
    arrays = _record_arrays(data)
    if arrays:
        return sum(len(records) for _, records in arrays)
    if isinstance(data, list):
        return len(data)
    return None


def validate_json_data(
    data: Any,
    dataset_name: str = "dataset.json",
    *,
    required_fields: Mapping[str, Sequence[str]] | None = None,
    warning_fields: Mapping[str, Sequence[str]] | None = None,
    required_arrays: Mapping[str, Mapping[str, Any]] | None = None,
    numerical_rules: Mapping[str, Mapping[str, Any]] | None = None,
    date_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
    validate_required: bool = True,
    validate_numerical: bool = True,
    validate_dates: bool = True,
) -> dict[str, Any]:
    """Validate cleaned JSON and preserve the original input and hierarchy."""
    source = copy.deepcopy(data)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    field_summary = []

    if validate_required:
        result = validate_required_fields(
            source,
            required_fields or {},
            warning_fields=warning_fields,
            required_arrays=required_arrays,
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
        warnings.extend(result["validation_warnings"])
        field_summary.extend(result["field_summary"])
    if validate_numerical:
        result = validate_numerical_values(
            source, numerical_rules or {}, dataset_name=dataset_name
        )
        errors.extend(result["validation_errors"])
        field_summary.extend(result["field_summary"])
    if validate_dates:
        result = validate_date_formats(
            source, date_rules or {}, dataset_name=dataset_name
        )
        errors.extend(result["validation_errors"])
        field_summary.extend(result["field_summary"])

    arrays = [
        (path, value) for path, value in _walk(source) if isinstance(value, list)
    ]
    _attach_owners(errors, arrays)
    _attach_owners(warnings, arrays)
    invalid_record_paths = {
        issue["record_path"]
        for issue in errors
        if issue.get("record_path") is not None
    }
    valid_data, _ = _split_data(
        source, "$", invalid_record_paths, "valid"
    )
    invalid_data, _ = _split_data(
        source, "$", invalid_record_paths, "invalid"
    )
    if "$" in invalid_record_paths:
        valid_data = None
        invalid_data = copy.deepcopy(source)

    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    total_records = _top_record_count(source)
    if total_records is None:
        valid_count = 0 if errors else 1
        invalid_count = 1 if errors else 0
        valid_data = None if errors else copy.deepcopy(source)
        invalid_data = copy.deepcopy(source) if errors else None
    else:
        invalid_count = sum(
            record_path != "$" for record_path in invalid_record_paths
        )
        valid_count = max(total_records - invalid_count, 0)
        if "$" in invalid_record_paths and not invalid_record_paths.difference({"$"}):
            valid_count, invalid_count = 0, total_records

    report = {
        "dataset_name": dataset_name,
        "top_level_structure": (
            "object"
            if isinstance(source, Mapping)
            else "array"
            if isinstance(source, list)
            else "scalar"
        ),
        "total_records": total_records,
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "total_validation_errors": len(errors),
        "total_validation_warnings": len(warnings),
        "validation_status": status,
        "field_summary": field_summary,
        "record_level_errors": errors,
        "record_level_warnings": warnings,
    }
    return {
        "valid_data": valid_data,
        "invalid_data": invalid_data,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "validation_report": report,
        "validation_status": status,
    }


def validate_multiple_json_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    rules_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
    **validation_options: Any,
) -> dict[str, dict[str, Any]]:
    """Validate independent Stage 3 JSON results or parsed values."""
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(
        enumerate(datasets)
    )
    results = {}
    for identifier, item in items:
        key = str(identifier)
        if (
            isinstance(item, Mapping)
            and isinstance(item.get("report"), Mapping)
            and item["report"].get("cleaning_status") == "error"
        ):
            continue
        if (
            isinstance(item, Mapping)
            and "data" in item
            and isinstance(item.get("report"), Mapping)
        ):
            data = item["data"]
            name = str(item["report"].get("dataset_name") or key)
        else:
            data = item
            name = key
        options = dict(validation_options)
        options.update((rules_by_dataset or {}).get(key, {}))
        results[key] = validate_json_data(data, name, **options)
    return results


def export_json_validation_result(
    result: Mapping[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, str]:
    """Save report/issues and hierarchy-preserving valid/invalid JSON outputs."""
    report = result.get("validation_report")
    if not isinstance(report, Mapping):
        raise TypeError("Validation result must contain 'validation_report'.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        Path(str(report.get("dataset_name") or "dataset")).stem,
    ).strip("._") or "dataset"
    run_directory = directory / f"{name}_{uuid4().hex[:8]}"
    run_directory.mkdir()
    report_path = run_directory / "validation_report.json"
    errors_path = run_directory / "validation_errors.csv"
    valid_path = run_directory / "valid_data.json"
    invalid_path = run_directory / "invalid_data.json"
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    error_records = list(result.get("validation_errors", [])) + list(
        result.get("validation_warnings", [])
    )
    pd.DataFrame(
        [
            {
                **issue,
                "invalid_value": json.dumps(
                    _json_safe(issue.get("invalid_value")),
                    ensure_ascii=False,
                    default=str,
                ),
            }
            for issue in error_records
        ],
        columns=[
            "dataset_name",
            "record_index",
            "record_path",
            "json_path",
            "field_name",
            "invalid_value",
            "validation_rule",
            "error_message",
            "severity",
        ],
    ).to_csv(errors_path, index=False)
    valid_path.write_text(
        json.dumps(_json_safe(result.get("valid_data")), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    invalid_path.write_text(
        json.dumps(
            _json_safe(result.get("invalid_data")), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return {
        "directory": str(run_directory),
        "report_json": str(report_path),
        "errors_csv": str(errors_path),
        "valid_json": str(valid_path),
        "invalid_json": str(invalid_path),
    }
