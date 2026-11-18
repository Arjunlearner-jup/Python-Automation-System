"""Non-mutating Stage 3 cleaning operations for hierarchical XML documents."""

from __future__ import annotations

import copy
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

try:
    from ..ingestion.xml_extractor import convert_xml_to_dataframe, get_xml_metadata
except ImportError:
    from ingestion.xml_extractor import convert_xml_to_dataframe, get_xml_metadata


DEFAULT_PROCESSED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "processed" / "xml"
_DATE_NAME = re.compile(r"date|time|timestamp|_at$|_on$", re.I)
_CODE = re.compile(r"^(EUR|USD|GBP|INR)$", re.I)
_SYMBOL = re.compile(r"^[\u20ac$\u00a3\u20b9]$")
_IDENTIFIER = re.compile(r"(^|_)(id|identifier|code|isbn|uuid|key)($|_)", re.I)


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _tag_for_name(original: str, name: str) -> str:
    if original.startswith("{"):
        namespace = original[1:].split("}", 1)[0]
        return f"{{{namespace}}}{name}"
    return name


def _path_map(root: ET.Element) -> dict[str, list[ET.Element]]:
    result: dict[str, list[ET.Element]] = {}
    def visit(node: ET.Element, path: str) -> None:
        result.setdefault(path, []).append(node)
        counts: dict[str, int] = {}
        for child in list(node):
            name = _local_name(str(child.tag)); counts[name] = counts.get(name, 0) + 1
            visit(child, f"{path}/{name}")
    visit(root, f"/{_local_name(str(root.tag))}")
    return result


def _all_nodes(root: ET.Element):
    def visit(node: ET.Element, path: str):
        yield path, node
        counts: dict[str, int] = {}
        for child in list(node):
            name = _local_name(str(child.tag)); counts[name] = counts.get(name, 0) + 1
            yield from visit(child, f"{path}/{name}[{counts[name]}]")
    yield from visit(root, f"/{_local_name(str(root.tag))}[1]")


def _group_path(root: ET.Element, selected: str) -> str:
    normalized = "/" + selected.strip("/")
    if normalized.startswith(f"/{_local_name(str(root.tag))}/") or normalized == f"/{_local_name(str(root.tag))}":
        return normalized
    return f"/{_local_name(str(root.tag))}{normalized}"


def _records_for_group(root: ET.Element, group: str | None) -> list[tuple[ET.Element, list[ET.Element], str]]:
    result = []
    def visit(parent: ET.Element, path: str):
        buckets: dict[str, list[ET.Element]] = {}
        for child in list(parent):
            buckets.setdefault(str(child.tag), []).append(child)
        for qualified_name, children in buckets.items():
            name = _local_name(qualified_name)
            candidate = f"{path}/{name}"
            if len(children) > 1 and (group is None or _group_path(root, group) == candidate):
                result.append((parent, children, candidate))
            for child in children:
                visit(child, candidate)
    visit(root, f"/{_local_name(str(root.tag))}")
    return result


def get_record_groups(root: ET.Element) -> list[dict[str, Any]]:
    """List repeated sibling groups using readable local-name paths."""
    return [{"path": path, "element": path.rsplit("/", 1)[-1], "count": len(records)}
            for _, records, path in _records_for_group(root, None)]


def _element_signature(element: ET.Element) -> Any:
    return (str(element.tag), tuple(sorted(element.attrib.items())), (element.text or "").strip(), tuple((_element_signature(child), (child.tail or "").strip()) for child in list(element)))


def _field_value(element: ET.Element, field: str) -> Any:
    if field.startswith("@"):
        return element.attrib.get(field[1:])
    path = [part for part in field.strip("/").split("/") if part]
    nodes = [element]
    for part in path:
        nodes = [child for node in nodes for child in list(node) if _local_name(str(child.tag)) == part]
    return tuple((node.text or "").strip() for node in nodes)


def identify_duplicate_records(root: ET.Element, record_group: str | None = None, fields: Sequence[str] | None = None) -> dict[str, Any]:
    duplicates = []
    for parent, records, path in _records_for_group(root, record_group):
        seen: set[Any] = set()
        for position, record in enumerate(records):
            identity = tuple(_field_value(record, field) for field in fields) if fields else _element_signature(record)
            if identity in seen:
                duplicates.append({"group": path, "position": position, "record": ET.tostring(record, encoding="unicode")})
            else:
                seen.add(identity)
    return {"duplicate_records_detected": len(duplicates), "duplicates": duplicates, "record_group": record_group, "fields": list(fields) if fields else None}


def remove_duplicate_records(root: ET.Element, record_group: str | None = None, fields: Sequence[str] | None = None) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root)
    detected = identify_duplicate_records(cleaned, record_group, fields)
    removed = []
    for parent, records, path in _records_for_group(cleaned, record_group):
        seen: set[Any] = set()
        for position, record in enumerate(list(records)):
            identity = tuple(_field_value(record, field) for field in fields) if fields else _element_signature(record)
            if identity in seen:
                removed.append({"group": path, "position": position, "record": ET.tostring(record, encoding="unicode")})
                parent.remove(record)
            else:
                seen.add(identity)
    return cleaned, {**detected, "duplicate_records_removed": len(removed), "removed_records": removed}


def identify_missing_values(root: ET.Element, selected_elements: Sequence[str] | None = None, selected_attributes: Sequence[str] | None = None) -> dict[str, Any]:
    missing_elements, empty_elements, whitespace_elements = [], [], []
    missing_attributes, empty_attributes = [], []
    element_selection = set(selected_elements or [])
    attribute_selection = set(selected_attributes or [])
    for path, node in _all_nodes(root):
        name = _local_name(str(node.tag))
        if (not element_selection or name in element_selection or path in element_selection):
            if node.text is None and not list(node): empty_elements.append(path)
            elif node.text == "": empty_elements.append(path)
            elif node.text is not None and not node.text.strip(): whitespace_elements.append(path)
        for attr in attribute_selection or node.attrib:
            if attr not in node.attrib:
                missing_attributes.append({"path": path, "attribute": attr})
            elif node.attrib[attr] == "":
                empty_attributes.append({"path": path, "attribute": attr})
    if element_selection:
        for path, parent in _all_nodes(root):
            children = {_local_name(str(child.tag)) for child in list(parent)}
            for expected in element_selection:
                if "/" not in expected and "[" not in expected and expected not in children:
                    missing_elements.append(f"{path}/{expected}")
    return {"missing_elements": missing_elements, "empty_elements": empty_elements, "whitespace_elements": whitespace_elements, "missing_attributes": missing_attributes, "empty_attributes": empty_attributes, "counts": {"missing_elements": len(missing_elements), "empty_elements": len(empty_elements), "whitespace_elements": len(whitespace_elements), "missing_attributes": len(missing_attributes), "empty_attributes": len(empty_attributes)}}


def handle_missing_values(root: ET.Element, strategy: str = "leave", value: str = "", *, elements: Sequence[str] | None = None, attributes: Sequence[str] | None = None, record_group: str | None = None, drop_fields: Sequence[str] | None = None, element_fill_value: str | None = None, attribute_fill_value: str | None = None) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root)
    before = identify_missing_values(cleaned, elements, attributes)
    selected_elements, selected_attributes = set(elements or []), set(attributes or [])
    fill = element_fill_value if element_fill_value is not None else value
    attr_fill = attribute_fill_value if attribute_fill_value is not None else value
    removed = []
    if strategy == "drop_records" and drop_fields:
        for parent, records, path in _records_for_group(cleaned, record_group):
            for record in list(records):
                if any(_missing_field(record, field) for field in drop_fields):
                    removed.append({"group": path, "record": ET.tostring(record, encoding="unicode")})
                    parent.remove(record)
    elif strategy not in {"leave", "replace"}:
        raise ValueError("strategy must be 'leave', 'replace', or 'drop_records'.")
    if strategy == "replace":
        for path, node in _all_nodes(cleaned):
            name = _local_name(str(node.tag))
            if (not selected_elements or name in selected_elements or path in selected_elements) and (node.text is None or not node.text.strip()) and not list(node):
                node.text = fill
        for _, node in _all_nodes(cleaned):
            for attr in selected_attributes:
                if attr not in node.attrib or not node.attrib[attr].strip(): node.set(attr, attr_fill)
    after = identify_missing_values(cleaned, elements, attributes)
    return cleaned, {"strategy": strategy, "missing_values_before": before, "missing_values_after": after, "records_removed": removed}


def _missing_field(record: ET.Element, field: str) -> bool:
    if field.startswith("@"):
        value = record.attrib.get(field[1:]); return value is None or not value.strip()
    nodes = [node for node in record.iter() if _local_name(str(node.tag)) == field.strip("/").split("/")[-1]]
    return not nodes or all(node.text is None or not node.text.strip() for node in nodes)


def standardise_names(root: ET.Element, recursive: bool = True, elements: bool = True, attributes: bool = True) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root)
    renamed_elements, renamed_attributes = [], []
    def normalize(name: str) -> str:
        value = re.sub(r"[\s-]+", "_", name.strip().lower())
        value = re.sub(r"[^a-z0-9_.]", "", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "field"
    def visit(parent: ET.Element, path: str):
        if attributes:
            seen: set[str] = set()
            for old in list(parent.attrib):
                normalized = normalize(_local_name(old)); new, suffix = normalized, 2
                while new in seen or (new in parent.attrib and new != old): new = f"{normalized}_{suffix}"; suffix += 1
                seen.add(new)
                if old != new:
                    value = parent.attrib.pop(old); new_qname = _tag_for_name(old, new); parent.set(new_qname, value)
                    renamed_attributes.append({"path": path, "original": old, "standardised": new_qname})
        if not recursive: return
        children = list(parent); used_sources: dict[str, str] = {}; source_names: dict[str, str] = {}
        for child in children:
            old = str(child.tag); normalized = normalize(_local_name(old)); new, suffix = normalized, 2
            source_local = _local_name(old)
            if source_local in source_names:
                new = source_names[source_local]
            else:
                while new in used_sources and used_sources[new] != source_local:
                    new = f"{normalized}_{suffix}"; suffix += 1
                used_sources[new] = source_local; source_names[source_local] = new
            if elements and old != _tag_for_name(old, new):
                child.tag = _tag_for_name(old, new)
                renamed_elements.append({"path": path, "original": old, "standardised": str(child.tag)})
            visit(child, f"{path}/{_local_name(str(child.tag))}")
    if elements:
        old_root = str(cleaned.tag); new_root = _tag_for_name(old_root, normalize(_local_name(old_root)))
        if new_root != old_root:
            cleaned.tag = new_root
            renamed_elements.append({"path": "/", "original": old_root, "standardised": new_root})
    visit(cleaned, f"/{_local_name(str(cleaned.tag))}")
    return cleaned, {"element_names_changed": renamed_elements, "attribute_names_changed": renamed_attributes}


def standardise_dates(root: ET.Element, elements: Sequence[str] | None = None, attributes: Sequence[str] | None = None, source_format: str | None = None) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root); converted, invalid = [], []
    element_selection, attr_selection = set(elements or []), set(attributes or [])
    def convert(value: str, name: str, path: str, kind: str) -> str:
        if not (_DATE_NAME.search(name) or name in element_selection or path in element_selection or name in attr_selection): return value
        try:
            parsed = datetime.strptime(value.strip(), source_format) if source_format else datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            result = parsed.isoformat().replace("+00:00", "Z")
            converted.append({"path": path, "kind": kind, "from": value, "to": result}); return result
        except (ValueError, OverflowError): invalid.append({"path": path, "kind": kind, "value": value}); return value
    for path, node in _all_nodes(cleaned):
        if node.text is not None: node.text = convert(node.text, _local_name(str(node.tag)), path, "element")
        for attr, value in list(node.attrib.items()): node.set(attr, convert(value, _local_name(attr), f"{path}/@{_local_name(attr)}", "attribute"))
    return cleaned, {"converted": converted, "invalid_values": invalid}


def standardise_currencies(root: ET.Element, elements: Sequence[str] | None = None, attributes: Sequence[str] | None = None) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root); standardized, invalid = [], []
    selected_elements, selected_attributes = set(elements or []), set(attributes or [])
    def convert(value: str, name: str, path: str, kind: str) -> str:
        lower = name.lower()
        explicit = name in selected_elements or path in selected_elements or name in selected_attributes
        text = value.strip()
        if not explicit and not re.search(r"currency|amount|price|cost|balance|total|symbol", lower) and not _CODE.fullmatch(text): return value
        if "symbol" in lower or _SYMBOL.fullmatch(text): return value
        if _CODE.fullmatch(text):
            result = text.upper()
            if result != value: standardized.append({"path": path, "kind": kind, "from": value, "to": result})
            return result
        # Currency metadata is not a monetary amount; only parse fields clearly
        # named for transaction values and only when the string has currency syntax.
        if re.search(r"amount|price|cost|balance|total", lower) and re.search(r"[\u20ac$\u00a3\u20b9,]", text):
            numeric = re.sub(r"[\s,\u20ac$\u00a3\u20b9]", "", text)
            try:
                parsed = Decimal(numeric)
                result = format(parsed, "f")
                if "." in result: result = result.rstrip("0").rstrip(".")
                standardized.append({"path": path, "kind": kind, "from": value, "to": result}); return result
            except (InvalidOperation, ValueError): invalid.append({"path": path, "kind": kind, "value": value})
        elif explicit and not ("symbol" in lower or "currency" in lower or "code" in lower):
            invalid.append({"path": path, "kind": kind, "value": value})
        return value
    for path, node in _all_nodes(cleaned):
        if node.text is not None: node.text = convert(node.text, _local_name(str(node.tag)), path, "element")
        for attr, value in list(node.attrib.items()): node.set(attr, convert(value, _local_name(attr), f"{path}/@{_local_name(attr)}", "attribute"))
    return cleaned, {"standardised": standardized, "unparsed_values": invalid}


def detect_data_types(root: ET.Element) -> dict[str, str]:
    types = {}
    for path, node in _all_nodes(root):
        if node.text is not None:
            value = node.text.strip()
            kind = "empty" if not value else "boolean" if value.lower() in {"true", "false"} else "integer" if re.fullmatch(r"[-+]?\d+", value) else "float" if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value) else "datetime" if _is_iso_datetime(value) else "string"
            types[path] = kind
        for attr, value in node.attrib.items():
            value = value.strip(); kind = "empty" if not value else "boolean" if value.lower() in {"true", "false"} else "integer" if re.fullmatch(r"[-+]?\d+", value) else "float" if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value) else "datetime" if _is_iso_datetime(value) else "string"
            types[f"{path}/@{_local_name(attr)}"] = kind
    return types


def _is_iso_datetime(value: str) -> bool:
    if "-" not in value or not re.search(r"\d{4}-\d{2}-\d{2}", value): return False
    try: datetime.fromisoformat(value.replace("Z", "+00:00")); return True
    except ValueError: return False


def convert_data_types(root: ET.Element, conversions: Mapping[str, str]) -> tuple[ET.Element, dict[str, Any]]:
    cleaned = copy.deepcopy(root); performed, failed = [], []
    def convert(value: str, target: str, path: str, field_name: str) -> str:
        if _IDENTIFIER.search(field_name):
            failed.append({"path": path, "target_type": target, "value": value, "error": "Identifier and code fields are not converted."}); return value
        try:
            raw = value.strip()
            if target == "integer":
                if not re.fullmatch(r"[-+]?\d+", raw): raise ValueError("Not an integer")
                result = str(int(raw))
            elif target == "float": result = repr(float(raw))
            elif target == "string": result = value
            elif target == "boolean":
                if raw.lower() in {"true", "1", "yes", "y"}: result = "true"
                elif raw.lower() in {"false", "0", "no", "n"}: result = "false"
                else: raise ValueError("Not a recognised boolean")
            elif target in {"date", "datetime"}:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")); result = parsed.isoformat().replace("+00:00", "Z")
            else: raise ValueError(f"Unsupported target type: {target}")
            performed.append({"path": path, "target_type": target, "from": value, "to": result}); return result
        except (ValueError, TypeError, OverflowError) as exc:
            failed.append({"path": path, "target_type": target, "value": value, "error": str(exc)}); return value
    for path, node in _all_nodes(cleaned):
        name = _local_name(str(node.tag)); target = conversions.get(path, conversions.get(name))
        if target and node.text is not None: node.text = convert(node.text, target, path, name)
        for attr in list(node.attrib):
            attr_name = _local_name(attr); attr_path = f"{path}/@{attr_name}"; target = conversions.get(attr_path, conversions.get(f"@{attr_name}"))
            if target: node.set(attr, convert(node.attrib[attr], target, attr_path, attr_name))
    return cleaned, {"conversions": performed, "failed_conversions": failed}


def _counts(root: ET.Element) -> tuple[int, int]:
    elements = list(root.iter()); return len(elements), sum(len(node.attrib) for node in elements)


def _declaration_from_extraction(extraction: Mapping[str, Any] | None) -> bool:
    path = (extraction or {}).get("metadata", {}).get("raw_file_path")
    if path:
        try: return Path(path).read_bytes().lstrip().startswith(b"<?xml")
        except OSError: pass
    return True


def clean_xml_data(data: ET.Element | Mapping[str, Any], dataset_name: str | None = None, *, operations: Mapping[str, bool] | None = None, remove_duplicates: bool = False, duplicate_group: str | None = None, duplicate_fields: Sequence[str] | None = None, handle_missing: bool = False, missing_strategy: str = "leave", missing_elements: Sequence[str] | None = None, missing_attributes: Sequence[str] | None = None, drop_fields: Sequence[str] | None = None, missing_fill_value: str = "", standardise_names_enabled: bool = False, recursive_names: bool = True, standardise_elements: bool = True, standardise_attributes: bool = True, standardise_date_values: bool = False, date_elements: Sequence[str] | None = None, date_attributes: Sequence[str] | None = None, date_format: str | None = None, standardise_currency_values: bool = False, currency_elements: Sequence[str] | None = None, currency_attributes: Sequence[str] | None = None, type_conversions: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Clean an XML root or an XML ingestion result without mutating its source."""
    extraction = data if isinstance(data, Mapping) else None
    root = data.get("xml_tree") if extraction is not None else data
    if not isinstance(root, ET.Element): raise TypeError("data must be an ElementTree root or XML ingestion result containing 'xml_tree'.")
    name = dataset_name or (extraction or {}).get("metadata", {}).get("source", "dataset.xml")
    flags = dict(operations or {})
    remove_duplicates = flags.get("duplicates", remove_duplicates)
    handle_missing = flags.get("missing", handle_missing)
    standardise_names_enabled = flags.get("names", standardise_names_enabled)
    standardise_date_values = flags.get("dates", standardise_date_values)
    standardise_currency_values = flags.get("currencies", standardise_currency_values)
    if flags.get("types", bool(type_conversions)) is False: type_conversions = None
    cleaned = copy.deepcopy(root); original_elements, original_attributes = _counts(root)
    report: dict[str, Any] = {"dataset_name": name, "root_element": str(root.tag), "original_element_count": original_elements, "original_attribute_count": original_attributes, "missing_values_before": identify_missing_values(root), "missing_values_after": identify_missing_values(root), "duplicate_records_detected": 0, "duplicate_records_removed": 0, "element_names_changed": [], "attribute_names_changed": [], "date_values_converted": [], "invalid_date_values": [], "currency_values_standardised": [], "unparsed_currency_values": [], "data_type_conversions_performed": [], "failed_conversions": [], "cleaning_status": "success"}
    if remove_duplicates:
        cleaned, duplicate = remove_duplicate_records(cleaned, duplicate_group, duplicate_fields); report.update({k: duplicate[k] for k in ("duplicate_records_detected", "duplicate_records_removed")}); report["duplicate_group"] = duplicate_group; report["duplicate_details"] = duplicate
    if handle_missing:
        cleaned, missing = handle_missing_values(cleaned, missing_strategy, elements=missing_elements, attributes=missing_attributes, record_group=duplicate_group, drop_fields=drop_fields, element_fill_value=missing_fill_value, attribute_fill_value=missing_fill_value); report["missing_values_before"] = missing["missing_values_before"]; report["missing_values_after"] = missing["missing_values_after"]; report["missing_records_removed"] = missing["records_removed"]
    if standardise_names_enabled:
        cleaned, names = standardise_names(cleaned, recursive_names, standardise_elements, standardise_attributes); report.update(names)
    if standardise_date_values:
        cleaned, dates = standardise_dates(cleaned, date_elements, date_attributes, date_format); report["date_values_converted"] = dates["converted"]; report["invalid_date_values"] = dates["invalid_values"]
    if standardise_currency_values:
        cleaned, currencies = standardise_currencies(cleaned, currency_elements, currency_attributes); report["currency_values_standardised"] = currencies["standardised"]; report["unparsed_currency_values"] = currencies["unparsed_values"]
    if type_conversions:
        cleaned, types = convert_data_types(cleaned, type_conversions); report["data_type_conversions_performed"] = types["conversions"]; report["failed_conversions"] = types["failed_conversions"]
    report["final_element_count"], report["final_attribute_count"] = _counts(cleaned); report["missing_values_after"] = identify_missing_values(cleaned)
    if report["failed_conversions"] or report["invalid_date_values"] or report["unparsed_currency_values"]: report["cleaning_status"] = "partial"
    result: dict[str, Any] = {"data": cleaned, "xml_tree": cleaned, "report": report, "xml_declaration": _declaration_from_extraction(extraction), "namespaces": copy.deepcopy((extraction or {}).get("metadata", {}).get("namespaces", {}))}
    if extraction is not None:
        frames = convert_xml_to_dataframe(cleaned)
        result["dataframes"] = frames
        result["data"] = frames.get("records")
        metadata = copy.deepcopy(extraction.get("metadata", {}))
        refreshed = get_xml_metadata(cleaned, frames, str(metadata.get("source", name)), int(metadata.get("file_size", 0)), namespaces=metadata.get("namespaces", {}))
        metadata.update({key: refreshed[key] for key in ("root_element", "element_count", "attribute_count", "repeating_record_elements", "dataframes")})
        result["metadata"] = metadata
    return result


def save_cleaned_xml(root: ET.Element, dataset_name: str, output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY, *, xml_declaration: bool = True, namespaces: Mapping[str, str] | None = None) -> Path:
    """Serialize an XML tree as a valid UTF-8 document without overwriting files."""
    directory = Path(output_directory); directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dataset_name).stem) or "dataset"
    destination = directory / f"{stem}_cleaned.xml"; suffix = 2
    while destination.exists(): destination = directory / f"{stem}_cleaned_{suffix}.xml"; suffix += 1
    for prefix, uri in (namespaces or {}).items():
        try: ET.register_namespace(prefix, uri)
        except ValueError: continue
    tree = ET.ElementTree(copy.deepcopy(root))
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=xml_declaration)
    ET.parse(destination)
    return destination
