"""Non-mutating cleaning helpers for arbitrary JSON-compatible values."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PROCESSED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "processed" / "json"
_DATE_NAME = re.compile(r"date|time|timestamp|_at$|_on$", re.I)
_CURRENCY_NAME = re.compile(r"currency|amount|price|cost|balance|total|_value$", re.I)
_CODE = re.compile(r"^(EUR|USD|GBP|INR)$", re.I)
_SYMBOL = re.compile(r"^[€$£₹]$")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ("dict", tuple(sorted(((str(k), _freeze(v)) for k, v in value.items()), key=repr)))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_freeze(v) for v in value))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield child_path, str(index), child
            yield from _walk(child, child_path)


def _record_arrays(value: Any, path: str = "$"):
    if isinstance(value, list):
        if all(isinstance(item, Mapping) for item in value):
            yield path, value
        for index, child in enumerate(value):
            yield from _record_arrays(child, f"{path}[{index}]")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _record_arrays(child, f"{path}.{key}")


def identify_duplicate_records(data: Any, subset: Sequence[str] | None = None) -> dict[str, Any]:
    found = []
    for path, records in _record_arrays(data):
        seen = set()
        for index, record in enumerate(records):
            value = {key: record.get(key) for key in subset} if subset else record
            frozen = _freeze(value)
            if frozen in seen:
                found.append({"path": path, "index": index, "record": copy.deepcopy(record)})
            else:
                seen.add(frozen)
    return {"duplicate_records_detected": len(found), "duplicates": found}


def remove_duplicate_records(data: Any, subset: Sequence[str] | None = None) -> tuple[Any, dict[str, Any]]:
    result = copy.deepcopy(data)
    report = identify_duplicate_records(result, subset)
    removed = []
    def clean(node: Any, path: str = "$") -> Any:
        if isinstance(node, list):
            if all(isinstance(item, Mapping) for item in node):
                seen, kept = set(), []
                for index, item in enumerate(node):
                    identity = _freeze({key: item.get(key) for key in subset} if subset else item)
                    if identity in seen:
                        removed.append({"path": path, "index": index, "record": copy.deepcopy(item)})
                    else:
                        seen.add(identity); kept.append(clean(item, f"{path}[{index}]"))
                return kept
            return [clean(item, f"{path}[{i}]") for i, item in enumerate(node)]
        if isinstance(node, dict):
            return {key: clean(value, f"{path}.{key}") for key, value in node.items()}
        return node
    result = clean(result)
    return result, {**report, "duplicate_records_removed": len(removed), "removed_records": removed, "subset": list(subset) if subset else None}


def identify_missing_values(data: Any) -> dict[str, Any]:
    counts = {"null": [], "empty_string": [], "whitespace_string": [], "empty_list": [], "empty_dictionary": []}
    for path, _, value in _walk(data):
        if value is None: counts["null"].append(path)
        elif isinstance(value, str) and value == "": counts["empty_string"].append(path)
        elif isinstance(value, str) and not value.strip(): counts["whitespace_string"].append(path)
        elif isinstance(value, list) and not value: counts["empty_list"].append(path)
        elif isinstance(value, dict) and not value: counts["empty_dictionary"].append(path)
    return {"counts": {key: len(paths) for key, paths in counts.items()}, "paths": counts, "total": sum(map(len, counts.values()))}


def _is_missing(value: Any, include_empty: bool = True) -> bool:
    return value is None or (isinstance(value, str) and (not value.strip())) or (include_empty and isinstance(value, (list, dict)) and not value)


def handle_missing_values(data: Any, strategy: str = "leave", fill_value: Any = None, *, field_strategies: Mapping[str, Any] | None = None, selected_fields: Sequence[str] | None = None, empty_containers: bool = False) -> tuple[Any, dict[str, Any]]:
    before = identify_missing_values(data)
    selected = set(selected_fields) if selected_fields is not None else None
    def transform(node: Any, path: str = "$") -> Any:
        if isinstance(node, dict):
            output = {}
            for key, val in node.items():
                child = f"{path}.{key}"
                spec = (field_strategies or {}).get(child, (field_strategies or {}).get(key, strategy))
                local_strategy, local_fill = (spec.get("strategy", "leave"), spec.get("value")) if isinstance(spec, Mapping) else (spec, fill_value)
                enabled = selected is None or key in selected or child in selected
                missing = _is_missing(val, empty_containers)
                if enabled and missing and local_strategy == "drop_records":
                    return _DROP
                if enabled and missing and local_strategy == "replace": output[key] = copy.deepcopy(local_fill)
                else:
                    transformed = transform(val, child)
                    if transformed is not _DROP: output[key] = transformed
            return output
        if isinstance(node, list):
            output = []
            for i, val in enumerate(node):
                child = f"{path}[{i}]"
                spec = (field_strategies or {}).get(child, strategy)
                local_strategy, local_fill = (spec.get("strategy", "leave"), spec.get("value")) if isinstance(spec, Mapping) else (spec, fill_value)
                missing = _is_missing(val, empty_containers)
                if missing and local_strategy == "drop_records": continue
                if missing and local_strategy == "replace": output.append(copy.deepcopy(local_fill))
                else:
                    transformed = transform(val, child)
                    if transformed is not _DROP: output.append(transformed)
            return output
        return node
    _DROP = object()
    result = copy.deepcopy(data) if strategy == "leave" and not field_strategies else transform(copy.deepcopy(data))
    if result is _DROP:
        result = None
    after = identify_missing_values(result)
    return result, {"strategy": strategy, "missing_values_before": before, "missing_values_after": after, "empty_containers_are_missing": empty_containers}


def standardise_field_names(data: Any, recursive: bool = False) -> tuple[Any, dict[str, Any]]:
    renamed = []
    def visit(node: Any, path: str = "$") -> Any:
        if isinstance(node, list): return [visit(x, f"{path}[{i}]") for i, x in enumerate(node)]
        if not isinstance(node, dict): return node
        result = {}
        for key, value in node.items():
            normalized = re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s-]+", "_", str(key).strip().lower()))
            normalized = re.sub(r"_+", "_", normalized).strip("_") or "field"
            name, suffix = normalized, 2
            while name in result:
                name = f"{normalized}_{suffix}"; suffix += 1
            if name != key: renamed.append({"path": path, "original": key, "standardised": name})
            result[name] = visit(value, f"{path}.{name}") if recursive else value
        return result
    return visit(copy.deepcopy(data)), {"field_names_changed": renamed}


def _map_fields(data: Any, fn, selected: Sequence[str] | None = None, parent: str = "$"):
    chosen = set(selected) if selected is not None else None
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            path = f"{parent}.{key}"
            out[key] = fn(value, key, path) if chosen is None or key in chosen or path in chosen else _map_fields(value, fn, selected, path)
        return out
    if isinstance(data, list): return [_map_fields(v, fn, selected, f"{parent}[{i}]") for i, v in enumerate(data)]
    return data


def standardise_dates(data: Any, fields: Sequence[str] | None = None, source_format: str | None = None) -> tuple[Any, dict[str, Any]]:
    converted, invalid = [], []
    def convert(value, key, path):
        if not isinstance(value, str) or not (_DATE_NAME.search(key) or fields is not None): return value
        try:
            parsed = datetime.strptime(value, source_format) if source_format else datetime.fromisoformat(value.replace("Z", "+00:00"))
            result = parsed.isoformat().replace("+00:00", "Z")
            converted.append({"path": path, "from": value, "to": result}); return result
        except (ValueError, OverflowError): invalid.append({"path": path, "value": value}); return value
    output = _map_fields(copy.deepcopy(data), convert, fields)
    return output, {"converted": converted, "invalid_values": invalid}


def standardise_currencies(data: Any, fields: Sequence[str] | None = None) -> tuple[Any, dict[str, Any]]:
    standardized, failed = [], []
    def convert(value, key, path):
        if not isinstance(value, str): return value
        if fields is None and not re.search(r"currency|code|symbol|amount|price|cost|balance|total", key, re.I): return value
        stripped = value.strip()
        if _CODE.fullmatch(stripped):
            result = stripped.upper()
            if result != value: standardized.append({"path": path, "from": value, "to": result})
            return result
        if _SYMBOL.fullmatch(stripped): return stripped
        if re.search(r"amount|price|cost|balance|total", key, re.I):
            numeric = re.sub(r"[\s,€$£₹]", "", stripped)
            if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", numeric):
                try:
                    result = float(numeric) if "." in numeric else int(numeric)
                    standardized.append({"path": path, "from": value, "to": result}); return result
                except ValueError: pass
            failed.append({"path": path, "value": value})
        return value
    output = _map_fields(copy.deepcopy(data), convert, fields)
    return output, {"standardised": standardized, "unparsed_values": failed}


def detect_data_types(data: Any) -> dict[str, str]:
    def typ(v):
        if v is None: return "null"
        if isinstance(v, bool): return "boolean"
        if isinstance(v, int): return "integer"
        if isinstance(v, float): return "float"
        if isinstance(v, str): return "string"
        if isinstance(v, list): return "list"
        if isinstance(v, dict): return "dictionary"
        return type(v).__name__
    return {path: typ(value) for path, _, value in _walk(data)}


def convert_data_types(data: Any, conversions: Mapping[str, str]) -> tuple[Any, dict[str, Any]]:
    performed, failed = [], []
    def cast(value, target):
        if target in {"dictionary", "dict"}: return value if isinstance(value, dict) else json.loads(value) if isinstance(value, str) else (_ for _ in ()).throw(ValueError())
        if target == "list": return value if isinstance(value, list) else json.loads(value) if isinstance(value, str) and isinstance(json.loads(value), list) else (_ for _ in ()).throw(ValueError())
        if target in {"string", "str"}: return value if isinstance(value, str) else str(value)
        if target == "integer":
            if isinstance(value, bool) or isinstance(value, (list, dict)): raise ValueError()
            if isinstance(value, str) and not re.fullmatch(r"[-+]?\d+", value.strip()): raise ValueError()
            return int(value)
        if target == "float":
            if isinstance(value, bool) or isinstance(value, (list, dict)): raise ValueError()
            return float(value)
        if target == "boolean":
            if isinstance(value, bool): return value
            if str(value).strip().lower() in {"true", "1", "yes", "y"}: return True
            if str(value).strip().lower() in {"false", "0", "no", "n"}: return False
            raise ValueError()
        if target in {"date", "datetime"}:
            if not isinstance(value, str): raise ValueError()
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat().replace("+00:00", "Z")
        if target == "null": return None
        raise ValueError(f"Unsupported target type: {target}")
    def walk(node: Any, path: str = "$"):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                p = f"{path}.{key}"
                target = conversions.get(p, conversions.get(str(key)))
                if target and not (_CODE.fullmatch(str(value)) or re.search(r"country|identifier|(^|_)id$", str(key), re.I)):
                    try:
                        out[key] = cast(value, target); performed.append({"path": p, "target_type": target, "from": value, "to": out[key]})
                    except (ValueError, TypeError, OverflowError) as exc:
                        out[key] = value; failed.append({"path": p, "target_type": target, "value": value, "error": str(exc) or "Value cannot be converted"})
                else: out[key] = walk(value, p)
            return out
        if isinstance(node, list): return [walk(v, f"{path}[{i}]") for i, v in enumerate(node)]
        return node
    result = walk(copy.deepcopy(data))
    return result, {"conversions": performed, "failed_conversions": failed}


def _structure(data: Any) -> str:
    if isinstance(data, dict): return "object"
    if isinstance(data, list): return "array"
    return "scalar"


def _record_count(data: Any) -> int | None:
    if isinstance(data, list): return len(data)
    if isinstance(data, dict):
        arrays = [v for v in data.values() if isinstance(v, list) and all(isinstance(x, Mapping) for x in v)]
        if len(arrays) == 1: return len(arrays[0])
    return None


def clean_json_data(data: Any, dataset_name: str = "dataset.json", *, operations: Mapping[str, bool] | None = None, remove_duplicates: bool = False, duplicate_subset: Sequence[str] | None = None, handle_missing: bool = False, missing_strategy: str = "leave", missing_fill_value: Any = None, field_strategies: Mapping[str, Any] | None = None, missing_fields: Sequence[str] | None = None, empty_containers_missing: bool = False, standardise_names: bool = False, recursive_names: bool = False, standardise_date_values: bool = False, date_fields: Sequence[str] | None = None, date_format: str | None = None, standardise_currency_values: bool = False, currency_fields: Sequence[str] | None = None, type_conversions: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run selected operations on a deep copy and return ``data`` plus a report."""
    flags = dict(operations or {})
    remove_duplicates = flags.get("duplicates", remove_duplicates)
    handle_missing = flags.get("missing", handle_missing)
    standardise_names = flags.get("field_names", standardise_names)
    standardise_date_values = flags.get("dates", standardise_date_values)
    standardise_currency_values = flags.get("currencies", standardise_currency_values)
    if flags.get("types", bool(type_conversions)) is False: type_conversions = None
    cleaned = copy.deepcopy(data)
    report = {"dataset_name": dataset_name, "original_record_count": _record_count(data), "original_structure": _structure(data), "duplicate_records_detected": 0, "duplicate_records_removed": 0, "missing_values_before": identify_missing_values(data), "missing_values_after": identify_missing_values(data), "empty_strings_detected": identify_missing_values(data)["counts"]["empty_string"] + identify_missing_values(data)["counts"]["whitespace_string"], "field_names_changed": [], "date_values_converted": [], "currency_values_standardised": [], "data_type_conversions_performed": [], "failed_conversions": [], "cleaning_status": "success"}
    if remove_duplicates:
        cleaned, r = remove_duplicate_records(cleaned, duplicate_subset); report.update({k: r[k] for k in ("duplicate_records_detected", "duplicate_records_removed")})
    if handle_missing:
        cleaned, r = handle_missing_values(cleaned, missing_strategy, missing_fill_value, field_strategies=field_strategies, selected_fields=missing_fields, empty_containers=empty_containers_missing); report["missing_values_before"] = r["missing_values_before"]; report["missing_values_after"] = r["missing_values_after"]
    if standardise_names:
        cleaned, r = standardise_field_names(cleaned, recursive_names); report["field_names_changed"] = r["field_names_changed"]
    if standardise_date_values:
        cleaned, r = standardise_dates(cleaned, date_fields, date_format); report["date_values_converted"] = r["converted"]; report["invalid_date_values"] = r["invalid_values"]
    if standardise_currency_values:
        cleaned, r = standardise_currencies(cleaned, currency_fields); report["currency_values_standardised"] = r["standardised"]; report["unparsed_currency_values"] = r["unparsed_values"]
    if type_conversions:
        cleaned, r = convert_data_types(cleaned, type_conversions); report["data_type_conversions_performed"] = r["conversions"]; report["failed_conversions"] = r["failed_conversions"]
    report["final_record_count"] = _record_count(cleaned); report["final_structure"] = _structure(cleaned)
    report["missing_values_after"] = identify_missing_values(cleaned)
    if report["failed_conversions"] or report.get("invalid_date_values") or report.get("unparsed_currency_values"): report["cleaning_status"] = "partial"
    return {"data": cleaned, "report": report}


def save_cleaned_json(data: Any, dataset_name: str, output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY) -> Path:
    directory = Path(output_directory); directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dataset_name).stem) or "dataset"
    destination = directory / f"{stem}_cleaned.json"; suffix = 2
    while destination.exists(): destination = directory / f"{stem}_cleaned_{suffix}.json"; suffix += 1
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
