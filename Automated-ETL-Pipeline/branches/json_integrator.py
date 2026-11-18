"""Stage 5 integration operations for cleaned and validated JSON datasets."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


DEFAULT_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "integrated" / "json"
)
DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "integration" / "json"
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._") or "integrated_json"


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


def _items(datasets: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    entries = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if not entries:
        raise ValueError("Select at least one JSON dataset for integration.")
    return [(str(key), value) for key, value in entries]


def _extract_source(dataset: Any, identifier: str) -> tuple[Any, str, dict[str, Any]]:
    if not isinstance(dataset, Mapping):
        return copy.deepcopy(dataset), identifier, {}

    validation = dataset.get("validation_result")
    if validation is None and "valid_data" in dataset and (
        "validation_status" in dataset or "validation_report" in dataset
    ):
        validation = dataset
    if isinstance(validation, Mapping):
        status = validation.get("validation_status")
        if status in {"failed", "error"}:
            raise ValueError(f"Dataset {identifier!r} did not pass JSON validation.")
        if "valid_data" not in validation:
            raise ValueError(f"Validated dataset {identifier!r} has no valid_data.")
        data = validation.get("valid_data")
        report = validation.get("validation_report", {})
        name = str(
            (report.get("dataset_name") if isinstance(report, Mapping) else None)
            or dataset.get("source")
            or identifier
        )
        return copy.deepcopy(data), name, dict(dataset.get("metadata") or {})

    if "data" in dataset and isinstance(dataset.get("report"), Mapping) and (
        "cleaning_status" in dataset["report"]
        or "dataset_name" in dataset["report"]
    ):
        report = dataset["report"]
        if report.get("cleaning_status") == "error":
            raise ValueError(f"Dataset {identifier!r} has a failed cleaning result.")
        name = str(
            report.get("dataset_name")
            or dataset.get("source")
            or dataset.get("filename")
            or identifier
        )
        return copy.deepcopy(dataset["data"]), name, dict(dataset.get("metadata") or {})

    return copy.deepcopy(dataset), identifier, {}


def _path_tokens(path: str) -> list[str | int]:
    path = path.strip()
    if path in {"", "$"}:
        return []
    if path.startswith("$"):
        path = path[1:]
    if path.startswith("."):
        path = path[1:]
    tokens: list[str | int] = []
    position = 0
    while position < len(path):
        if path[position] == ".":
            position += 1
            continue
        if path[position] == "[":
            end = path.find("]", position)
            if end < 0:
                raise ValueError(f"Invalid JSON path: {path!r}.")
            token = path[position + 1 : end]
            tokens.append(int(token) if token.isdigit() else token)
            position = end + 1
            continue
        end = position
        while end < len(path) and path[end] not in ".[":
            end += 1
        tokens.append(path[position:end])
        position = end
    return tokens


def _get_path(value: Any, path: str) -> Any:
    current = value
    for token in _path_tokens(path):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return _MISSING
            current = current[token]
        else:
            if not isinstance(current, Mapping) or token not in current:
                return _MISSING
            current = current[token]
    return current


def _set_path(value: Any, path: str, replacement: Any) -> Any:
    tokens = _path_tokens(path)
    if not tokens:
        return replacement
    current = value
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = replacement
    return value


def _canonical_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def _walk(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _record_groups(data: Any) -> list[dict[str, Any]]:
    groups = []
    for path, value in _walk(data):
        if isinstance(value, list) and (
            not value or all(isinstance(item, Mapping) for item in value)
        ):
            groups.append(
                {"path": path, "canonical_path": _canonical_path(path), "records": value}
            )
    if isinstance(data, Mapping) and not groups:
        groups.append({"path": "$", "canonical_path": "$", "records": [data]})
    elif isinstance(data, list) and not groups:
        groups.append({"path": "$", "canonical_path": "$", "records": data})
    return groups


def _choose_group(data: Any, record_path: str | None = None) -> dict[str, Any]:
    groups = _record_groups(data)
    if record_path is not None:
        requested = _canonical_path(record_path if record_path.startswith("$") else f"$.{record_path}")
        matches = [
            group
            for group in groups
            if group["path"] == record_path or group["canonical_path"] == requested
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Record path {record_path!r} matched {len(matches)} object arrays."
            )
        return matches[0]
    populated_groups = [group for group in groups if group["records"]]
    if populated_groups:
        groups = populated_groups
    outermost_groups = [
        group
        for group in groups
        if not any(
            parent is not group
            and (
                parent["canonical_path"] == "$"
                or group["canonical_path"].startswith(
                    parent["canonical_path"] + "[*]."
                )
                or group["canonical_path"].startswith(
                    parent["canonical_path"] + "[*]["
                )
            )
            for parent in groups
        )
    ]
    if outermost_groups:
        groups = outermost_groups
    if len(groups) != 1:
        paths = [group["canonical_path"] for group in groups]
        raise ValueError(
            f"JSON contains multiple record arrays {paths}; select a record_path."
        )
    return groups[0]


def get_json_record_paths(data: Any) -> list[str]:
    """Return concrete JSON paths of arrays that contain only objects."""
    groups = _record_groups(data)
    populated_groups = [group for group in groups if group["records"]]
    return [
        group["path"] for group in (populated_groups or groups)
    ]


_MISSING = object()


def _kind(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _compatible_types(left: set[str], right: set[str]) -> bool:
    left = left.difference({"missing", "null"})
    right = right.difference({"missing", "null"})
    if not left or not right:
        return True
    return left == right or (left | right) <= {"number"}


def _record_fields(records: Sequence[Any]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for path, value in _walk(record, "$"):
            if path != "$":
                relative_path = path[2:] if path.startswith("$.") else path
                relative_path = re.sub(r"\[\d+\]", "[*]", relative_path)
                fields.setdefault(relative_path, set()).add(_kind(value))
    return fields


def _relative_key_path(path: str) -> str:
    value = path.strip()
    value = re.sub(r"^\$\.?", "", value)
    value = re.sub(r"^\*\.", "", value)
    return value


def _field_value(record: Any, field_path: str) -> Any:
    path = _relative_key_path(field_path)
    if not isinstance(record, Mapping):
        return _MISSING
    tokens = _path_tokens(path)

    def resolve(current: Any, remaining: list[str | int]) -> Any:
        if not remaining:
            return current
        token, rest = remaining[0], remaining[1:]
        if token == "*":
            if not isinstance(current, list):
                return _MISSING
            return [resolve(item, rest) for item in current]
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return _MISSING
            return resolve(current[token], rest)
        if not isinstance(current, Mapping) or token not in current:
            return _MISSING
        return resolve(current[token], rest)

    return resolve(record, tokens)


def _fields_for_records(records: Sequence[Any]) -> dict[str, set[str]]:
    return _record_fields(records)


def _normalise_field(path: str) -> str:
    return re.sub(r"[^a-z0-9]", "", path.lower())


def match_common_fields(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    field_mapping: Mapping[str, str] | None = None,
    record_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare nested field paths in the first two JSON datasets."""
    entries = _items(datasets)
    if len(entries) < 2:
        raise ValueError("At least two datasets are needed to match fields.")
    extracted = []
    names = []
    for key, dataset in entries[:2]:
        data, name, _ = _extract_source(dataset, key)
        group = _choose_group(data, (record_paths or {}).get(key))
        extracted.append(_fields_for_records(group["records"]))
        names.append(name)

    left_fields, right_fields = extracted
    common = sorted(set(left_fields).intersection(right_fields))
    right_normalized: dict[str, list[str]] = {}
    for path in right_fields:
        right_normalized.setdefault(_normalise_field(path), []).append(path)
    suggestions = []
    for left_path in sorted(left_fields):
        candidates = [
            path
            for path in right_normalized.get(_normalise_field(left_path), [])
            if path != left_path
        ]
        if candidates:
            suggestions.append(
                {
                    "left_field": left_path,
                    "right_field_candidates": candidates,
                    "compatible_types": all(
                        _compatible_types(left_fields[left_path], right_fields[path])
                        for path in candidates
                    ),
                }
            )
    mappings = []
    for left_path, right_path in (field_mapping or {}).items():
        exists = left_path in left_fields and right_path in right_fields
        mappings.append(
            {
                "left_field": left_path,
                "right_field": right_path,
                "exists": exists,
                "compatible_types": (
                    _compatible_types(left_fields[left_path], right_fields[right_path])
                    if exists
                    else False
                ),
            }
        )
    valid_mappings = {
        entry["left_field"]: entry["right_field"]
        for entry in mappings
        if entry["exists"] and entry["compatible_types"]
    }
    matched_left = set(common) | set(valid_mappings)
    matched_right = set(common) | set(valid_mappings.values())
    return {
        "left_dataset": names[0],
        "right_dataset": names[1],
        "common_fields": common,
        "possible_matching_fields": suggestions,
        "manual_mappings": mappings,
        "unmatched_left_fields": sorted(set(left_fields) - matched_left),
        "unmatched_right_fields": sorted(set(right_fields) - matched_right),
        "left_field_types": {key: sorted(value) for key, value in left_fields.items()},
        "right_field_types": {key: sorted(value) for key, value in right_fields.items()},
    }


def handle_inconsistent_schemas(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    record_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare nested JSON schemas while leaving all input structures intact."""
    per_dataset = []
    field_types: dict[str, dict[str, list[str]]] = {}
    for key, dataset in _items(datasets):
        data, name, metadata = _extract_source(dataset, key)
        group = _choose_group(data, (record_paths or {}).get(key))
        fields = _fields_for_records(group["records"])
        per_dataset.append(
            {
                "dataset_id": key,
                "dataset_name": name,
                "record_path": group["path"],
                "record_count": len(group["records"]),
                "fields": sorted(fields),
                "field_types": {path: sorted(kinds) for path, kinds in fields.items()},
                "metadata": metadata,
            }
        )
        field_types[key] = {path: sorted(kinds) for path, kinds in fields.items()}
    all_fields = sorted(
        {field for dataset in per_dataset for field in dataset["fields"]}
    )
    for dataset in per_dataset:
        present = set(dataset["fields"])
        dataset["missing_fields"] = sorted(set(all_fields) - present)
        dataset["additional_fields"] = sorted(
            present
            - set.intersection(
                *(set(item["fields"]) for item in per_dataset)
            )
            if len(per_dataset) > 1
            else set()
        )
    types_report = []
    for field in all_fields:
        definitions = [
            {
                "dataset_id": dataset["dataset_id"],
                "types": dataset["field_types"].get(field, []),
            }
            for dataset in per_dataset
            if field in dataset["field_types"]
        ]
        combined_types = {
            kind for definition in definitions for kind in definition["types"]
        }
        types_report.append(
            {
                "field_path": field,
                "definitions": definitions,
                "compatible_types": (
                    combined_types <= {"number"}
                    or len(combined_types.difference({"null"})) <= 1
                ),
            }
        )
    return {
        "schema_report": per_dataset,
        "all_fields": all_fields,
        "field_type_report": types_report,
        "field_types_by_dataset": field_types,
    }


def _record_path_argument(
    record_path: str | Mapping[str, str] | None, key: str
) -> str | None:
    if isinstance(record_path, Mapping):
        return record_path.get(key)
    return record_path


def _merge_values(left: Any, right: Any, left_suffix: str, right_suffix: str) -> Any:
    if left == right:
        return copy.deepcopy(left)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged: dict[str, Any] = copy.deepcopy(dict(left))
        for key, value in right.items():
            if key not in merged:
                merged[key] = copy.deepcopy(value)
            elif merged[key] != value:
                combined = _merge_values(merged[key], value, left_suffix, right_suffix)
                if isinstance(combined, Mapping) and set(combined) == {"left", "right"}:
                    del merged[key]
                    merged[f"{key}_{left_suffix}"] = combined["left"]
                    merged[f"{key}_{right_suffix}"] = combined["right"]
                else:
                    merged[key] = combined
        return merged
    return {"left": copy.deepcopy(left), "right": copy.deepcopy(right)}


def _merged_record(
    left: Mapping[str, Any], right: Mapping[str, Any], left_suffix: str, right_suffix: str
) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(left))
    for key, value in right.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
            continue
        if result[key] == value:
            continue
        combined = _merge_values(result[key], value, left_suffix, right_suffix)
        if isinstance(combined, Mapping) and set(combined) == {"left", "right"}:
            del result[key]
            result[f"{key}_{left_suffix}"] = combined["left"]
            result[f"{key}_{right_suffix}"] = combined["right"]
        else:
            result[key] = combined
    return result


def _key_value(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...] | None:
    values = tuple(_field_value(record, field) for field in fields)
    if any(value is _MISSING or value is None for value in values):
        return None
    return tuple(_freeze(value) for value in values)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _unique_key_values(records: Sequence[Any], fields: Sequence[str]) -> dict[Any, list[int]]:
    values: dict[Any, list[int]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("JSON record arrays must contain objects.")
        key = _key_value(record, fields)
        if key is not None:
            values.setdefault(key, []).append(index)
    return values


def _replace_group(data: Any, path: str, records: list[Any]) -> Any:
    if path == "$":
        if isinstance(data, Mapping) and len(records) == 1:
            return copy.deepcopy(records[0])
        return copy.deepcopy(records)
    result = copy.deepcopy(data)
    _set_path(result, path, copy.deepcopy(records))
    return result


def merge_json_datasets(
    left_dataset: Any,
    right_dataset: Any,
    *,
    join_type: str = "inner",
    key_fields: Sequence[str] | str | None = None,
    right_key_fields: Sequence[str] | str | None = None,
    field_mapping: Mapping[str, str] | None = None,
    left_record_path: str | None = None,
    right_record_path: str | None = None,
    many_to_many: str = "raise",
    left_name: str = "left.json",
    right_name: str = "right.json",
) -> dict[str, Any]:
    """Join two JSON record groups without flattening their objects or wrappers."""
    join = join_type.lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join_type must be inner, left, right, or outer.")
    if many_to_many not in {"raise", "allow"}:
        raise ValueError("many_to_many must be 'raise' or 'allow'.")
    left_data, left_name, left_metadata = _extract_source(left_dataset, left_name)
    right_data, right_name, right_metadata = _extract_source(right_dataset, right_name)
    left_group = _choose_group(left_data, left_record_path)
    right_group = _choose_group(right_data, right_record_path)
    left_records = left_group["records"]
    right_records = right_group["records"]
    if isinstance(key_fields, str):
        left_keys = [key_fields]
    else:
        left_keys = list(key_fields or [])
    if isinstance(right_key_fields, str):
        right_keys = [right_key_fields]
    else:
        right_keys = list(right_key_fields or [])
    mappings = dict(field_mapping or {})
    if not left_keys and mappings:
        left_keys = list(mappings)
    if not left_keys:
        raise ValueError("Select at least one JSON field as a merge key.")
    if not right_keys:
        right_keys = [mappings.get(key, key) for key in left_keys]
    if len(left_keys) != len(right_keys):
        raise ValueError("Left and right merge key lists must have equal length.")
    left_types, right_types = _record_fields(left_records), _record_fields(right_records)
    incompatible = [
        (left_key, right_key)
        for left_key, right_key in zip(left_keys, right_keys)
        if (left_records and left_key not in left_types)
        or (right_records and right_key not in right_types)
        or not _compatible_types(left_types.get(left_key, set()), right_types.get(right_key, set()))
    ]
    if incompatible:
        raise TypeError(f"Incompatible or missing JSON merge key fields: {incompatible}.")

    left_index = _unique_key_values(left_records, left_keys)
    right_index = _unique_key_values(right_records, right_keys)
    duplicate_left = {
        key: indexes for key, indexes in left_index.items() if len(indexes) > 1
    }
    duplicate_right = {
        key: indexes for key, indexes in right_index.items() if len(indexes) > 1
    }
    many_to_many_keys = set(duplicate_left).intersection(duplicate_right)
    if many_to_many_keys and many_to_many == "raise":
        raise ValueError(
            "Many-to-many JSON merge would expand duplicate keys on both sides; "
            "resolve duplicate keys or set many_to_many='allow'."
        )

    right_keys_lookup: dict[Any, list[int]] = right_index
    output_records: list[dict[str, Any]] = []
    matched_records: list[dict[str, Any]] = []
    unmatched_left: list[dict[str, Any]] = []
    matched_right_indexes: set[int] = set()
    left_matched_count = 0
    for left_position, left_record in enumerate(left_records):
        if not isinstance(left_record, Mapping):
            raise TypeError("JSON record arrays must contain objects.")
        key = _key_value(left_record, left_keys)
        right_positions = right_keys_lookup.get(key, []) if key is not None else []
        if right_positions:
            left_matched_count += 1
            for right_position in right_positions:
                merged = _merged_record(
                    left_record,
                    right_records[right_position],
                    _safe_name(left_name),
                    _safe_name(right_name),
                )
                matched_records.append(
                    {
                        "left_record_index": left_position,
                        "right_record_index": right_position,
                        "left_source_file": left_name,
                        "right_source_file": right_name,
                        "record": copy.deepcopy(merged),
                    }
                )
                matched_right_indexes.add(right_position)
                if join in {"inner", "left", "outer", "right"}:
                    output_records.append(merged)
        else:
            unmatched_left.append(
                {
                    "record_index": left_position,
                    "source_file": left_name,
                    "record": copy.deepcopy(left_record),
                }
            )
            if join in {"left", "outer"}:
                output_records.append(copy.deepcopy(dict(left_record)))

    unmatched_right = []
    for right_position, right_record in enumerate(right_records):
        if right_position not in matched_right_indexes:
            unmatched_right.append(
                {
                    "record_index": right_position,
                    "source_file": right_name,
                    "record": copy.deepcopy(right_record),
                }
            )
            if join in {"right", "outer"}:
                output_records.append(copy.deepcopy(dict(right_record)))

    output = _replace_group(left_data, left_group["path"], output_records)
    schema_result = handle_inconsistent_schemas(
        {
            left_name: {"data": left_data, "report": {"dataset_name": left_name}},
            right_name: {"data": right_data, "report": {"dataset_name": right_name}},
        },
        record_paths={left_name: left_group["path"], right_name: right_group["path"]},
    )
    report = {
        "operation": "merge",
        "join_type": join,
        "left_dataset": left_name,
        "right_dataset": right_name,
        "left_record_path": left_group["path"],
        "right_record_path": right_group["path"],
        "left_key_fields": left_keys,
        "right_key_fields": right_keys,
        "left_input_records": len(left_records),
        "right_input_records": len(right_records),
        "output_records": len(output_records),
        "matched_records": len(matched_records),
        "unmatched_left_records": len(unmatched_left),
        "unmatched_right_records": len(unmatched_right),
        "duplicate_left_keys": len(duplicate_left),
        "duplicate_right_keys": len(duplicate_right),
        "many_to_many_allowed": many_to_many == "allow",
        "field_mapping": mappings,
        "left_metadata": left_metadata,
        "right_metadata": right_metadata,
    }
    return {
        "data": output,
        "integration_report": report,
        "source_datasets": [
            {"dataset_name": left_name, "records": len(left_records), "metadata": left_metadata},
            {"dataset_name": right_name, "records": len(right_records), "metadata": right_metadata},
        ],
        "matched_records": matched_records,
        "unmatched_records": {
            "left": unmatched_left,
            "right": unmatched_right,
        },
        "schema_report": schema_result,
    }


def combine_json_files(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    selected_datasets: Sequence[str] | None = None,
    record_path: str | Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Append object records from JSON arrays while preserving the first wrapper."""
    entries = _items(datasets)
    selected = set(map(str, selected_datasets)) if selected_datasets is not None else None
    entries = [(key, value) for key, value in entries if selected is None or key in selected]
    if not entries:
        raise ValueError("No selected JSON datasets are available to combine.")
    extracted = []
    for key, dataset in entries:
        data, name, metadata = _extract_source(dataset, key)
        group = _choose_group(data, _record_path_argument(record_path, key))
        extracted.append((key, data, name, metadata, group))
    if len(entries) > 1:
        canonical_paths = [group["canonical_path"] for _, _, _, _, group in extracted]
        if len(set(canonical_paths)) > 1 and not record_path:
            raise ValueError(
                "Selected JSON record arrays use different paths; select a record_path "
                "for each dataset to confirm they belong together."
            )

    output_records = []
    source_information = []
    record_provenance = []
    for key, _, name, metadata, group in extracted:
        records = group["records"]
        if not all(isinstance(record, Mapping) for record in records):
            raise TypeError("JSON record groups must contain objects.")
        output_start = len(output_records)
        output_records.extend(copy.deepcopy(records))
        record_provenance.extend(
            {
                "output_record_index": output_start + index,
                "source_record_index": index,
                "source_file": name,
            }
            for index in range(len(records))
        )
        source_information.append(
            {
                "dataset_id": key,
                "source_file": name,
                "records_contributed": len(records),
                "record_path": group["path"],
                "metadata": metadata,
            }
        )
    first_data = extracted[0][1]
    first_path = extracted[0][4]["path"]
    output = _replace_group(first_data, first_path, output_records)
    schema = handle_inconsistent_schemas(
        {
            key: {"data": data, "report": {"dataset_name": name}}
            for key, data, name, _, _ in extracted
        },
        record_paths={
            key: group["path"] for key, _, _, _, group in extracted
        },
    )
    report = {
        "operation": "combine",
        "dataset_name": "integrated_json",
        "input_dataset_count": len(entries),
        "input_record_count": sum(item["records_contributed"] for item in source_information),
        "output_record_count": len(output_records),
        "source_record_counts": source_information,
        "record_provenance": record_provenance,
        "output_record_path": first_path,
        "schema_comparison": schema,
    }
    return {
        "data": output,
        "integration_report": report,
        "source_datasets": source_information,
        "matched_records": [],
        "unmatched_records": {"left": [], "right": []},
        "schema_report": schema,
    }


def _save_integration(
    result: Mapping[str, Any],
    output_directory: str | Path,
    report_directory: str | Path,
) -> dict[str, str]:
    data_directory = Path(output_directory)
    reports_directory = Path(report_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex[:8]
    name = _safe_name(
        str(result.get("integration_report", {}).get("dataset_name", "integrated_json"))
    )
    data_path = data_directory / f"{name}_{identifier}.json"
    report_path = reports_directory / f"{name}_{identifier}_report.json"
    report_csv_path = reports_directory / f"{name}_{identifier}_report.csv"
    schema_path = reports_directory / f"{name}_{identifier}_schema.csv"
    records_path = reports_directory / f"{name}_{identifier}_records.csv"
    with data_path.open("x", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(_json_safe(result["data"]), ensure_ascii=False, indent=2)
            + "\n"
        )
    payload = {
        "integration_report": _json_safe(result.get("integration_report", {})),
        "source_datasets": _json_safe(result.get("source_datasets", [])),
        "matched_records": _json_safe(result.get("matched_records", [])),
        "unmatched_records": _json_safe(result.get("unmatched_records", {})),
        "schema_report": _json_safe(result.get("schema_report", {})),
    }
    with report_path.open("x", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with report_csv_path.open("x", encoding="utf-8", newline="") as output_file:
        pd.DataFrame(
            [{"section": "integration_report", **payload["integration_report"]}]
        ).to_csv(output_file, index=False)
    schema = result.get("schema_report", {})
    schema_rows = schema.get("schema_report", []) if isinstance(schema, Mapping) else []
    with schema_path.open("x", encoding="utf-8", newline="") as output_file:
        pd.DataFrame(schema_rows).to_csv(output_file, index=False)
    record_rows = []
    for match in result.get("matched_records", []):
        record_rows.append(
            {
                "status": "matched",
                "left_record_index": match.get("left_record_index"),
                "right_record_index": match.get("right_record_index"),
                "left_source_file": match.get("left_source_file"),
                "right_source_file": match.get("right_source_file"),
                "record": json.dumps(_json_safe(match.get("record")), ensure_ascii=False),
            }
        )
    unmatched = result.get("unmatched_records", {})
    if isinstance(unmatched, Mapping):
        for side, records in unmatched.items():
            for item in records:
                record_rows.append(
                    {
                        "status": f"unmatched_{side}",
                        "left_record_index": (
                            item.get("record_index") if side == "left" else None
                        ),
                        "right_record_index": (
                            item.get("record_index") if side == "right" else None
                        ),
                        "left_source_file": (
                            item.get("source_file") if side == "left" else None
                        ),
                        "right_source_file": (
                            item.get("source_file") if side == "right" else None
                        ),
                        "record": json.dumps(
                            _json_safe(item.get("record")), ensure_ascii=False
                        ),
                    }
                )
    with records_path.open("x", encoding="utf-8", newline="") as output_file:
        pd.DataFrame(
            record_rows,
            columns=[
                "status",
                "left_record_index",
                "right_record_index",
                "left_source_file",
                "right_source_file",
                "record",
            ],
        ).to_csv(output_file, index=False)
    return {
        "data": str(data_path),
        "report_json": str(report_path),
        "report_csv": str(report_csv_path),
        "schema_csv": str(schema_path),
        "records_csv": str(records_path),
        "data_directory": str(data_directory),
        "report_directory": str(reports_directory),
    }


def integrate_multiple_json_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    integration_type: str = "combine",
    selected_datasets: Sequence[str] | None = None,
    record_path: str | Mapping[str, str] | None = None,
    join_type: str = "inner",
    key_fields: Sequence[str] | str | None = None,
    right_key_fields: Sequence[str] | str | None = None,
    field_mapping: Mapping[str, str] | None = None,
    left_record_path: str | None = None,
    right_record_path: str | None = None,
    many_to_many: str = "raise",
    output_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    report_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Integrate selected cleaned/validated JSON documents and save outputs."""
    entries = _items(datasets)
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        entries = [(key, dataset) for key, dataset in entries if key in selected]
    if not entries:
        raise ValueError("No selected JSON datasets are available for integration.")
    operation = integration_type.lower()
    if operation in {"combine", "append", "concatenate"}:
        result = combine_json_files(dict(entries), record_path=record_path)
    elif operation in {"merge", "join"}:
        if len(entries) < 2:
            raise ValueError("Select at least two JSON datasets to merge.")
        result = merge_json_datasets(
            entries[0][1],
            entries[1][1],
            join_type=join_type,
            key_fields=key_fields,
            right_key_fields=right_key_fields,
            field_mapping=field_mapping,
            left_record_path=left_record_path or _record_path_argument(record_path, entries[0][0]),
            right_record_path=right_record_path or _record_path_argument(record_path, entries[1][0]),
            many_to_many=many_to_many,
            left_name=entries[0][0],
            right_name=entries[1][0],
        )
        if len(entries) > 2:
            merged = result["data"]
            reports = []
            last_result = result
            for index, (key, dataset) in enumerate(entries[2:], start=2):
                next_path = _record_path_argument(record_path, key)
                step = merge_json_datasets(
                    merged,
                    dataset,
                    join_type=join_type,
                    key_fields=key_fields,
                    right_key_fields=right_key_fields or key_fields,
                    field_mapping=field_mapping,
                    left_record_path=left_record_path or result["integration_report"]["left_record_path"],
                    right_record_path=right_record_path or next_path,
                    many_to_many=many_to_many,
                    left_name=f"integrated_{index}",
                    right_name=key,
                )
                merged = step["data"]
                reports.append(step["integration_report"])
                last_result = step
            initial_report = dict(result["integration_report"])
            result["data"] = merged
            result["matched_records"] = last_result["matched_records"]
            result["unmatched_records"] = last_result["unmatched_records"]
            result["integration_report"]["merge_steps"] = [
                initial_report,
                *reports,
            ]
            result["integration_report"]["additional_merge_steps"] = reports
            result["integration_report"]["input_dataset_count"] = len(entries)
            result["source_datasets"] = []
            for key, dataset in entries:
                data, name, metadata = _extract_source(dataset, key)
                group = _choose_group(data, _record_path_argument(record_path, key))
                result["source_datasets"].append(
                    {"dataset_name": name, "records": len(group["records"]), "metadata": metadata}
                )
    else:
        raise ValueError("integration_type must be 'combine' or 'merge'.")
    result["integration_report"]["integration_type"] = operation
    result["integration_report"].setdefault("dataset_name", "integrated_json")
    result["integration_report"]["schema_comparison"] = result.get("schema_report")
    if save_outputs:
        result["saved_paths"] = _save_integration(
            result, output_directory, report_directory
        )
    return result
