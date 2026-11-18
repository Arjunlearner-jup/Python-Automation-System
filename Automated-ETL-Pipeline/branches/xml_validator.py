"""Configurable, read-only validation for cleaned hierarchical XML documents."""

from __future__ import annotations

import copy
import csv
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from uuid import uuid4
from xml.etree import ElementTree as ET

import pandas as pd

DEFAULT_REPORT_DIRECTORY = (
        Path(__file__).resolve().parents[2] / "data" / "reports" / "validation" / "xml"
)


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _path_name(tag: str) -> str:
    if tag.startswith("{"):
        namespace, local = tag[1:].split("}", 1)
        return f"{{{quote(namespace, safe='')}}}{local}"
    return tag


def _path_segment_local(segment: str) -> str:
    return segment.split("}", 1)[1] if segment.startswith("{") else segment


def _attribute_matches(name: str, pattern: str) -> bool:
    if pattern.startswith("{"):
        return _path_name(name) == pattern
    return _local_name(name) == pattern


def _element_paths(root: ET.Element):
    """Yield elements with namespace-aware paths and stable sibling indexes."""

    def visit(
            node: ET.Element,
            path: str,
            location: tuple[int, ...],
            parent_location: tuple[int, ...] | None,
    ):
        yield node, path, location, parent_location
        counts: defaultdict[str, int] = defaultdict(int)
        for index, child in enumerate(list(node)):
            name = _path_name(str(child.tag))
            counts[name] += 1
            child_path = f"{path}/{name}[{counts[name]}]"
            yield from visit(child, child_path, location + (index,), location)

    root_name = _path_name(str(root.tag))
    yield from visit(root, f"/{root_name}[1]", (), None)


def _structural_path(path: str) -> str:
    return re.sub(r"\[\d+]", "", path)


def _path_matches(path: str, pattern: str) -> bool:
    """Match local-name XML paths; '*' matches one element path segment."""
    actual = _structural_path(path).strip("/").split("/")
    expected = pattern.strip("/").split("/")
    if len(actual) != len(expected):
        return False
    for actual_part, expected_part in zip(actual, expected):
        if expected_part == "*":
            continue
        if expected_part.startswith("{") and actual_part != expected_part:
            return False
        elif not expected_part.startswith("{") and _path_segment_local(actual_part) != expected_part:
            return False
    return True


def _record_groups(root: ET.Element) -> list[dict[str, Any]]:
    """Find every repeated-sibling group without flattening its hierarchy."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entries = list(_element_paths(root))
    by_parent: dict[tuple[int, ...], list[tuple[ET.Element, str, tuple[int, ...]]]] = (
        defaultdict(list)
    )
    for node, path, location, parent_location in entries:
        if parent_location is not None:
            by_parent[parent_location].append((node, path, location))
    for children in by_parent.values():
        counts: defaultdict[str, int] = defaultdict(int)
        for node, path, location in children:
            name = _path_name(str(node.tag))
            counts[name] += 1
        if any(count > 1 for count in counts.values()):
            seen: defaultdict[str, int] = defaultdict(int)
            for node, path, location in children:
                name = _path_name(str(node.tag))
                seen[name] += 1
                if counts[name] > 1:
                    key = _structural_path(path)
                    grouped[key].append(
                        {
                            "element": node,
                            "path": path,
                            "location": location,
                            "index": seen[name],
                        }
                    )
    return [
        {
            "path": path,
            "record_type": _path_segment_local(path.rsplit("/", 1)[-1]),
            "records": records,
        }
        for path, records in grouped.items()
    ]


def _configured_record_groups(
    root: ET.Element, configured_paths: Sequence[str]
) -> list[dict[str, Any]]:
    """Include configured singleton record paths alongside repeated groups."""
    groups = _record_groups(root)
    entries = list(_element_paths(root))
    for pattern in dict.fromkeys(configured_paths):
        if pattern == "*" or any(
            _path_matches(group["path"], pattern) for group in groups
        ):
            continue
        by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node, path, location, _ in entries:
            if _path_matches(path, pattern):
                by_path[_structural_path(path)].append(
                    {
                        "element": node,
                        "path": path,
                        "location": location,
                        "index": len(by_path[_structural_path(path)]) + 1,
                    }
                )
        for path, records in by_path.items():
            groups.append(
                {
                    "path": path,
                    "record_type": _path_segment_local(path.rsplit("/", 1)[-1]),
                    "records": records,
                }
            )
    return groups


def _normalise_field_rules(
        rules: Mapping[str, Sequence[str] | str] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for group_path, fields in (rules or {}).items():
        result[str(group_path)] = (
            [fields] if isinstance(fields, str) else [str(field) for field in fields]
        )
    return result


def _owner(
        location: tuple[int, ...],
        groups: Sequence[Mapping[str, Any]],
        root: ET.Element,
) -> tuple[str, str, int | None, str | None]:
    candidates = []
    for group in groups:
        for record in group["records"]:
            record_location = record["location"]
            if location[: len(record_location)] == record_location:
                candidates.append(
                    (
                        len(record_location),
                        str(group["record_type"]),
                        int(record["index"]),
                        str(record["path"]),
                        record["element"],
                        record_location,
                    )
                )
    if not candidates:
        return _local_name(str(root.tag)), "/", 1, None
    _, record_type, index, path, element, _ = max(candidates, key=lambda item: item[0])
    identifier = (
            element.get("id")
            or element.get("ID")
            or element.get("isbn")
            or element.get("code")
    )
    return record_type, path, index, identifier


def _issue(
        root: ET.Element,
        groups: Sequence[Mapping[str, Any]],
        dataset_name: str,
        path: str,
        location: tuple[int, ...],
        field_name: str,
        value: Any,
        rule: str,
        message: str,
        severity: str = "error",
) -> dict[str, Any]:
    record_type, record_path, record_index, record_id = _owner(location, groups, root)
    return {
        "dataset_name": dataset_name,
        "record_type": record_type,
        "record_index": record_index,
        "record_id": record_id,
        "record_path": record_path,
        "xml_path": path,
        "element_or_attribute": field_name,
        "invalid_value": value,
        "validation_rule": rule,
        "error_message": message,
        "severity": severity,
    }


def _relative_matches(
        record: ET.Element,
        record_path: str,
        field_path: str,
        all_entries: Sequence[tuple[ET.Element, str, tuple[int, ...], tuple[int, ...] | None]],
):
    relative = field_path.strip("/")
    if relative.startswith("@"):
        expected = []
        attribute = relative[1:]
    elif "/@" in relative:
        element_path, attribute = relative.rsplit("/@", 1)
        expected = [part for part in element_path.split("/") if part]
    else:
        expected = [part for part in relative.split("/") if part]
        attribute = None
    record_location = next(
        location for node, path, location, _ in all_entries
        if node is record and path == record_path
    )
    for node, path, location, _ in all_entries:
        if location[: len(record_location)] != record_location:
            continue
        parts = _structural_path(path).strip("/").split("/")
        base = _structural_path(record_path).strip("/").split("/")
        tail = parts[len(base):]
        matches_element = len(tail) == len(expected) and all(
                rule == "*"
                or (
                        actual == rule
                        if rule.startswith("{")
                        else _path_segment_local(actual) == rule
                )
                for actual, rule in zip(tail, expected)
        )
        if attribute is not None and matches_element:
                for name, value in node.attrib.items():
                    if _attribute_matches(name, attribute):
                        yield (
                            node,
                            f"{path}/@{_path_name(name)}",
                            location,
                            _local_name(name),
                            value,
                        )
        elif attribute is None and matches_element:
                yield node, path, location, _local_name(str(node.tag)), node.text


def validate_required_fields(
        xml_document: ET.Element,
        required_elements: Mapping[str, Sequence[str] | str] | None = None,
        required_attributes: Mapping[str, Sequence[str] | str] | None = None,
        *,
        warning_elements: Mapping[str, Sequence[str] | str] | None = None,
        warning_attributes: Mapping[str, Sequence[str] | str] | None = None,
        dataset_name: str = "dataset.xml",
) -> dict[str, Any]:
    """Check configured relative fields for each matching repeated record.

    Rule keys identify record-group paths, such as ``/library/book``. Values
    are element paths relative to each record (for example ``author/name``),
    or attribute names prefixed with ``@``.
    """
    rules = [
        required_elements or {},
        required_attributes or {},
        warning_elements or {},
        warning_attributes or {},
    ]
    groups = _configured_record_groups(
        xml_document,
        [str(path) for rule_map in rules for path in rule_map],
    )
    entries = list(_element_paths(xml_document))
    issues: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}

    for rule_map, severity in (
            (_normalise_field_rules(required_elements), "error"),
            (_normalise_field_rules(required_attributes), "error"),
            (_normalise_field_rules(warning_elements), "warning"),
            (_normalise_field_rules(warning_attributes), "warning"),
    ):
        for group_pattern, fields in rule_map.items():
            matching_groups = [
                group for group in groups
                if _path_matches(group["path"], group_pattern)
            ]
            if not matching_groups:
                rule = f"required group {group_pattern}"
                issues.append(
                    _issue(
                        xml_document, groups, dataset_name, group_pattern, (),
                        group_pattern.rsplit("/", 1)[-1], None, rule,
                        f"Configured record group {group_pattern!r} was not found.",
                        severity,
                    )
                )
                continue
            for group in matching_groups:
                for record in group["records"]:
                    for field in fields:
                        is_attribute = field.startswith("@") or "/@" in field
                        matches = list(
                            _relative_matches(
                                record["element"],
                                record["path"],
                                field,
                                entries,
                            )
                        )
                        key = f"{group['path']}/{field}"
                        counts = summary.setdefault(
                            key,
                            {
                                "total_values_checked": 0,
                                "valid_values": 0,
                                "invalid_values": 0,
                                "missing_values": 0,
                                "error_count": 0,
                            },
                        )
                        counts["total_values_checked"] += 1
                        if not matches:
                            missing_path = f"{record['path']}/{field}"
                            counts["missing_values"] += 1
                            counts["invalid_values"] += 1
                            counts["error_count"] += int(severity == "error")
                            issues.append(
                                _issue(
                                    xml_document, groups, dataset_name,
                                    missing_path, record["location"],
                                    (
                                        f"@{field.rsplit('/@', 1)[-1]}"
                                        if is_attribute
                                        else field.rsplit("/", 1)[-1]
                                    ),
                                    None, f"required_{'attribute' if is_attribute else 'element'}",
                                    f"Required {'attribute' if is_attribute else 'element'} "
                                    f"{field!r} is missing from record {record['path']}.",
                                    severity,
                                )
                            )
                            continue
                        for node, path, location, name, value in matches:
                            if value is None or not value.strip():
                                counts["invalid_values"] += 1
                                counts["error_count"] += int(severity == "error")
                                state = "whitespace-only" if value else "empty"
                                issues.append(
                                    _issue(
                                        xml_document, groups, dataset_name, path,
                                        location, f"@{name}" if is_attribute else name,
                                        value, f"required_{'attribute' if is_attribute else 'element'}",
                                        f"Required {'attribute' if is_attribute else 'element'} "
                                        f"{field!r} contains {state} text.",
                                        severity,
                                    )
                                )
                            else:
                                counts["valid_values"] += 1
    return {
        "validation_errors": [item for item in issues if item["severity"] == "error"],
        "validation_warnings": [item for item in issues if item["severity"] == "warning"],
        "field_summary": summary,
    }


def _numeric_rule(rule: Mapping[str, Any] | str) -> dict[str, Any]:
    return {"type": rule} if isinstance(rule, str) else dict(rule)


def _numeric_bound(value: Any, rule_path: str, bound_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(
            f"Numeric {bound_name} for {rule_path!r} must be a number."
        )
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Numeric {bound_name} for {rule_path!r} must be a number."
        ) from exc


def _configured_value_matches(
        entries: Sequence[tuple[ET.Element, str, tuple[int, ...], tuple[int, ...] | None]],
        pattern: str,
):
    attribute = pattern.rsplit("/@", 1)[1] if "/@" in pattern else None
    element_pattern = pattern.rsplit("/@", 1)[0] if attribute else pattern
    for node, path, location, _ in entries:
        if _path_matches(path, element_pattern):
            if attribute is None:
                yield node, path, location, _local_name(str(node.tag)), node.text
            else:
                for name, value in node.attrib.items():
                    if _attribute_matches(name, attribute):
                        yield (
                            node,
                            f"{path}/@{_path_name(name)}",
                            location,
                            f"@{_local_name(name)}",
                            value,
                        )


def validate_numerical_values(
        xml_document: ET.Element,
        numerical_rules: Mapping[str, Mapping[str, Any] | str],
        *,
        dataset_name: str = "dataset.xml",
) -> dict[str, Any]:
    """Validate configured numeric element or attribute values without conversion."""
    groups = _configured_record_groups(
        xml_document,
        [
            str(path).rsplit("/@", 1)[0].rsplit("/", 1)[0]
            for path in (numerical_rules or {})
        ],
    )
    entries = list(_element_paths(xml_document))
    issues: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for path_pattern, raw_rule in (numerical_rules or {}).items():
        rule = _numeric_rule(raw_rule)
        kind = str(rule.get("type", "float")).lower()
        if kind not in {"int", "integer", "float", "decimal"}:
            raise ValueError(f"Unsupported numeric type {kind!r} for {path_pattern}.")
        matches = list(_configured_value_matches(entries, str(path_pattern)))
        counts = summary.setdefault(
            str(path_pattern),
            {"total_values_checked": 0, "valid_values": 0, "invalid_values": 0,
             "missing_values": 0, "error_count": 0},
        )
        if not matches:
            counts["missing_values"] += 1
            issues.append(
                _issue(xml_document, groups, dataset_name, str(path_pattern), (),
                       str(path_pattern).rsplit("/", 1)[-1], None,
                       "numeric_field_present", "Configured numeric field was not found.")
            )
            continue
        for node, path, location, name, raw_value in matches:
            counts["total_values_checked"] += 1
            value_text = raw_value.strip() if raw_value is not None else ""
            message = None
            parsed: int | float | None = None
            if not value_text:
                message = "Numeric value is empty."
            else:
                try:
                    parsed = int(value_text) if kind in {"int", "integer"} else float(value_text)
                    if isinstance(parsed, float) and not math.isfinite(parsed):
                        raise ValueError("not finite")
                except (ValueError, OverflowError):
                    message = f"Value is not a valid {kind}."
            if message is None and parsed is not None:
                minimum = rule.get("minimum", rule.get("min"))
                maximum = rule.get("maximum", rule.get("max"))
                if minimum is not None and not isinstance(minimum, (int, float)):
                    raise ValueError(
                        f"Numeric minimum for {path_pattern!r} must be a number."
                    )
                if maximum is not None and not isinstance(maximum, (int, float)):
                    raise ValueError(
                        f"Numeric maximum for {path_pattern!r} must be a number."
                    )
                if rule.get("allow_negative") is False and parsed < 0:
                    message = "Negative values are not permitted."
                elif minimum is not None and parsed < minimum:
                    message = f"Value is below the minimum {minimum}."
                elif maximum is not None and parsed > maximum:
                    message = f"Value is above the maximum {maximum}."
            if message:
                counts["invalid_values"] += 1
                severity = str(rule.get("severity", "error"))
                counts["error_count"] += int(severity == "error")
                issues.append(
                    _issue(xml_document, groups, dataset_name, path, location,
                           name, raw_value, f"numeric_{kind}", message,
                           severity)
                )
            else:
                counts["valid_values"] += 1
    return {
        "validation_errors": [item for item in issues if item["severity"] == "error"],
        "validation_warnings": [item for item in issues if item["severity"] == "warning"],
        "field_summary": summary,
    }


def validate_date_formats(
        xml_document: ET.Element,
        date_rules: Mapping[str, Mapping[str, Any] | str | Sequence[str]],
        *,
        dataset_name: str = "dataset.xml",
) -> dict[str, Any]:
    """Validate configured date values using ISO 8601 or explicit strptime formats."""
    groups = _configured_record_groups(
        xml_document,
        [
            str(path).rsplit("/@", 1)[0].rsplit("/", 1)[0]
            for path in (date_rules or {})
        ],
    )
    entries = list(_element_paths(xml_document))
    issues: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for path_pattern, raw_rule in (date_rules or {}).items():
        if isinstance(raw_rule, str):
            formats = [raw_rule]
            severity = "error"
            required = False
        elif isinstance(raw_rule, Mapping):
            configured = raw_rule.get("formats", raw_rule.get("format", []))
            formats = [configured] if isinstance(configured, str) else list(configured)
            severity = str(raw_rule.get("severity", "error"))
            required = bool(raw_rule.get("required", False))
        else:
            formats = list(raw_rule)
            severity = "error"
            required = False
        formats = [str(fmt) for fmt in formats]
        matches = list(_configured_value_matches(entries, str(path_pattern)))
        counts = summary.setdefault(
            str(path_pattern),
            {"total_values_checked": 0, "valid_values": 0, "invalid_values": 0,
             "missing_values": 0, "error_count": 0},
        )
        if not matches:
            counts["missing_values"] += 1
            if required:
                issues.append(
                    _issue(xml_document, groups, dataset_name, str(path_pattern), (),
                           str(path_pattern).rsplit("/", 1)[-1], None,
                           "required_date", "Required date field was not found.", severity)
                )
            continue
        for node, path, location, name, value in matches:
            counts["total_values_checked"] += 1
            text = (value or "").strip()
            valid = False
            if text:
                try:
                    if not formats:
                        _parse_iso_date(text)
                    else:
                        for fmt in formats:
                            try:
                                datetime.strptime(text, fmt)
                                valid = True
                                break
                            except ValueError:
                                continue
                    if not formats:
                        valid = True
                except ValueError:
                    valid = False
            if not valid:
                counts["invalid_values"] += 1
                counts["error_count"] += int(severity == "error")
                expected = ", ".join(formats) if formats else "ISO 8601"
                issues.append(
                    _issue(xml_document, groups, dataset_name, path, location,
                           name, value, "date_format",
                           f"Date value is empty or invalid; expected {expected}.",
                           severity)
                )
            else:
                counts["valid_values"] += 1
    return {
        "validation_errors": [item for item in issues if item["severity"] == "error"],
        "validation_warnings": [item for item in issues if item["severity"] == "warning"],
        "field_summary": summary,
    }


def _root_from_input(data: Any) -> tuple[ET.Element, Mapping[str, Any] | None]:
    if isinstance(data, ET.Element):
        return data, None
    if isinstance(data, Mapping):
        root = data.get("xml_tree")
        if isinstance(root, ET.Element):
            return root, data
        root = data.get("data")
        if isinstance(root, ET.Element):
            return root, data
    raise TypeError("XML validation requires an ElementTree root or Stage 3 result.")


def _pruned_copy(
        root: ET.Element,
        selected_locations: set[tuple[int, ...]],
        groups: Sequence[Mapping[str, Any]],
        mode: str,
) -> ET.Element | None:
    """Create hierarchy-preserving valid/invalid XML record subsets."""
    record_locations = {
        record["location"]
        for group in groups
        for record in group["records"]
    }
    if not groups:
        return copy.deepcopy(root) if (() in selected_locations) == (mode == "invalid") else (
            None if mode == "invalid" else copy.deepcopy(root)
        )

    def has_selected_descendant(location: tuple[int, ...]) -> bool:
        return any(
            candidate[: len(location)] == location
            for candidate in selected_locations
        )

    if () in selected_locations:
        return copy.deepcopy(root) if mode == "invalid" else None

    def copy_node(
            node: ET.Element,
            location: tuple[int, ...],
            ancestor_invalid: bool,
            ancestor_selected: bool,
    ):
        is_record = location in record_locations
        own_selected = location in selected_locations
        if mode == "valid" and (ancestor_invalid or own_selected):
            return None
        if mode == "invalid" and not ancestor_selected:
            if is_record and not own_selected and not has_selected_descendant(location):
                return None
            if location and not is_record and not has_selected_descendant(location):
                return None
        new_node = ET.Element(node.tag, dict(node.attrib))
        new_node.text = node.text
        new_node.tail = node.tail
        child_counts: defaultdict[str, int] = defaultdict(int)
        for index, child in enumerate(list(node)):
            child_location = location + (index,)
            clone = copy_node(
                child,
                child_location,
                ancestor_invalid or (is_record and own_selected),
                ancestor_selected or (is_record and own_selected),
            )
            if clone is not None:
                new_node.append(clone)
            child_counts[_local_name(str(child.tag))] += 1
        return new_node

    return copy_node(root, (), False, False)


def get_xml_metadata(xml_document: ET.Element, dataset_name: str) -> dict[str, Any]:
    """Return structural counts and repeated record-group details."""
    groups = _record_groups(xml_document)
    nodes = list(xml_document.iter())
    return {
        "dataset_name": dataset_name,
        "root_element": str(xml_document.tag),
        "total_element_count": len(nodes),
        "total_attribute_count": sum(len(node.attrib) for node in nodes),
        "record_groups": [
            {"record_type": item["record_type"], "xml_path": item["path"],
             "record_count": len(item["records"])}
            for item in groups
        ],
    }


def get_xml_validation_paths(xml_document: ET.Element) -> dict[str, list[str]]:
    """Return local-name paths suitable for numeric and date rule configuration."""
    elements = sorted(
        {_structural_path(path) for _, path, _, _ in _element_paths(xml_document)}
    )
    attributes = sorted(
        f"{_structural_path(path)}/@{_path_name(name)}"
        for node, path, _, _ in _element_paths(xml_document)
        for name in node.attrib
    )
    return {"elements": elements, "attributes": attributes}


def validate_xml_data(
        cleaned_xml: Any,
        dataset_name: str = "dataset.xml",
        *,
        required_elements: Mapping[str, Sequence[str] | str] | None = None,
        required_attributes: Mapping[str, Sequence[str] | str] | None = None,
        warning_elements: Mapping[str, Sequence[str] | str] | None = None,
        warning_attributes: Mapping[str, Sequence[str] | str] | None = None,
        numerical_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
        date_rules: Mapping[str, Mapping[str, Any] | str | Sequence[str]] | None = None,
        validate_required: bool = True,
        validate_numerical: bool = True,
        validate_dates: bool = True,
) -> dict[str, Any]:
    """Validate Stage 3 XML without modifying it and retain record relationships."""
    root, cleaning_result = _root_from_input(cleaned_xml)
    source = root
    before = ET.tostring(source, encoding="utf-8")
    configured_record_paths = [
        str(path)
        for rule_map in (
            required_elements or {},
            required_attributes or {},
            warning_elements or {},
            warning_attributes or {},
        )
        for path in rule_map
    ]
    for path in (numerical_rules or {}):
        element_path = str(path).rsplit("/@", 1)[0]
        configured_record_paths.append(element_path.rsplit("/", 1)[0])
    for path in (date_rules or {}):
        element_path = str(path).rsplit("/@", 1)[0]
        configured_record_paths.append(element_path.rsplit("/", 1)[0])
    groups = _configured_record_groups(source, configured_record_paths)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    field_summaries: dict[str, dict[str, int]] = {}

    if validate_required:
        result = validate_required_fields(
            source, required_elements, required_attributes,
            warning_elements=warning_elements,
            warning_attributes=warning_attributes,
            dataset_name=dataset_name,
        )
        errors.extend(result["validation_errors"])
        warnings.extend(result["validation_warnings"])
        field_summaries.update(result["field_summary"])
    if validate_numerical:
        result = validate_numerical_values(
            source, numerical_rules or {}, dataset_name=dataset_name
        )
        errors.extend(result["validation_errors"])
        warnings.extend(result["validation_warnings"])
        field_summaries.update(result["field_summary"])
    if validate_dates:
        result = validate_date_formats(
            source, date_rules or {}, dataset_name=dataset_name
        )
        errors.extend(result["validation_errors"])
        warnings.extend(result["validation_warnings"])
        field_summaries.update(result["field_summary"])

    if ET.tostring(source, encoding="utf-8") != before:
        raise RuntimeError("XML validation unexpectedly modified its input.")
    invalid_locations = {
        next(
            (
                record["location"]
                for group in groups
                for record in group["records"]
                if issue["record_path"] == record["path"]
                   and issue["record_type"] == group["record_type"]
                   and issue["record_index"] == record["index"]
            ),
            (),
        )
        for issue in errors
    }
    if any(issue["record_path"] == "/" for issue in errors):
        invalid_locations.add(())
    if not groups and errors:
        invalid_locations.add(())
    valid_xml = _pruned_copy(source, invalid_locations, groups, "valid")
    invalid_xml = _pruned_copy(source, invalid_locations, groups, "invalid")
    records = [
        (str(group["record_type"]), record)
        for group in groups
        for record in group["records"]
    ]
    if records:
        total_records = len(records)
        invalid_record_count = (
            total_records
            if () in invalid_locations
            else sum(
                record["location"] in invalid_locations for _, record in records
            )
        )
        valid_record_count = total_records - invalid_record_count
    else:
        total_records = 1
        invalid_record_count = int(bool(errors))
        valid_record_count = total_records - invalid_record_count
        valid_xml = copy.deepcopy(source) if valid_record_count else None
        invalid_xml = copy.deepcopy(source) if invalid_record_count else None

    element_summary = [
        {
            "element_name": path.rsplit("/", 1)[-1],
            "xml_path": path,
            **counts,
        }
        for path, counts in field_summaries.items()
        if not path.rsplit("/", 1)[-1].startswith("@")
    ]
    attribute_summary = [
        {"xml_path": path, "attribute_name": path.rsplit("@", 1)[-1], **counts}
        for path, counts in field_summaries.items()
        if "/@" in path or path.rsplit("/", 1)[-1].startswith("@")
    ]
    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    report = {
        **get_xml_metadata(source, dataset_name),
        "total_records_checked": total_records,
        "valid_records": valid_record_count,
        "invalid_records": invalid_record_count,
        "total_validation_errors": len(errors),
        "total_validation_warnings": len(warnings),
        "validation_status": status,
        "element_summary": element_summary,
        "attribute_summary": attribute_summary,
        "record_level_errors": errors,
        "record_level_warnings": warnings,
    }
    result = {
        "valid_data": valid_xml,
        "invalid_data": invalid_xml,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "validation_report": report,
        "validation_status": status,
        "xml_declaration": bool((cleaning_result or {}).get("xml_declaration", True)),
        "namespaces": copy.deepcopy((cleaning_result or {}).get("namespaces", {})),
    }
    return result


def validate_multiple_xml_datasets(
        datasets: Mapping[str, Any] | Sequence[Any],
        *,
        rules_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
        **validation_options: Any,
) -> dict[str, dict[str, Any]]:
    """Validate independent cleaned XML trees or Stage 3 XML results."""
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(enumerate(datasets))
    results = {}
    for identifier, item in items:
        key = str(identifier)
        if (
                isinstance(item, Mapping)
                and isinstance(item.get("report"), Mapping)
                and item["report"].get("cleaning_status") == "error"
        ):
            continue
        name = key
        if isinstance(item, Mapping):
            report = item.get("report")
            metadata = item.get("metadata")
            name = str(
                (report or {}).get("dataset_name")
                or (metadata or {}).get("source")
                or key
            )
        options = dict(validation_options)
        options.update((rules_by_dataset or {}).get(key, {}))
        if isinstance(item, pd.DataFrame):
            results[key] = validate_xml_dataframe(item, key, dataset_name=name, **options)
        else:
            results[key] = validate_xml_data(item, name, **options)
    return results


def _frame_rule_field(rule_path: str, table_name: str) -> str:
    parts = rule_path.strip("/").split("/")
    if parts and parts[0] == table_name:
        parts = parts[1:]
    if parts and parts[0] in {"record", "records"}:
        parts = parts[1:]
    return "/".join(parts)


def _frame_value(row: Mapping[str, Any], field_path: str) -> tuple[bool, Any]:
    parts = field_path.strip("/").split("/")
    if parts and parts[0].startswith("@"):
        attributes = row.get("@attributes", {})
        name = parts[0][1:]
        if isinstance(attributes, Mapping) and name in attributes:
            return True, attributes[name]
        return False, None
    if not parts or parts[0] not in row:
        return False, None
    value = row[parts[0]]
    for part in parts[1:]:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return False, None
    return True, value


def _frame_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if getattr(missing, "ndim", 0) == 0 else False


def validate_xml_dataframe(
        dataframe: pd.DataFrame,
        record_group_name: str,
        *,
        dataset_name: str | None = None,
        required_elements: Mapping[str, Sequence[str] | str] | None = None,
        required_attributes: Mapping[str, Sequence[str] | str] | None = None,
        warning_elements: Mapping[str, Sequence[str] | str] | None = None,
        warning_attributes: Mapping[str, Sequence[str] | str] | None = None,
        numerical_rules: Mapping[str, Mapping[str, Any] | str] | None = None,
        date_rules: Mapping[str, Mapping[str, Any] | str | Sequence[str]] | None = None,
        validate_required: bool = True,
        validate_numerical: bool = True,
        validate_dates: bool = True,
) -> dict[str, Any]:
    """Validate one XML-derived table independently while retaining link columns.

    Rule keys can use column names or XML-like paths ending in ``/record/name``
    and ``/record/@attribute``. Nested dictionary values use slash-separated
    paths. The returned valid/invalid data remain DataFrames with their
    original ``_element_id`` and ``_parent_id`` relationship columns intact.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a Pandas DataFrame.")
    name = dataset_name or record_group_name
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    field_summary: dict[str, dict[str, int]] = {}
    invalid_rows: set[int] = set()

    field_requirements = [
        (required_elements or {}, "element", "error"),
        (required_attributes or {}, "attribute", "error"),
        (warning_elements or {}, "element", "warning"),
        (warning_attributes or {}, "attribute", "warning"),
    ]
    for rules, kind, severity in field_requirements:
        if not validate_required:
            break
        for _, fields in rules.items():
            fields = [fields] if isinstance(fields, str) else list(fields)
            for field in fields:
                field_path = str(field)
                if kind == "attribute" and not field_path.startswith("@"):
                    field_path = f"@{field_path}"
                for position, (_, series) in enumerate(dataframe.iterrows()):
                    row = series.to_dict()
                    found, value = _frame_value(row, field_path)
                    location = str(row.get("_element_id") or f"{record_group_name}[{position + 1}]")
                    xml_path = f"{location}/{field_path}"
                    key = f"{record_group_name}/{field_path}"
                    counts = field_summary.setdefault(
                        key,
                        {"total_values_checked": 0, "valid_values": 0,
                         "invalid_values": 0, "missing_values": 0,
                         "error_count": 0},
                    )
                    counts["total_values_checked"] += 1
                    if not found or _frame_missing(value):
                        counts["missing_values"] += int(not found)
                        counts["invalid_values"] += 1
                        counts["error_count"] += int(severity == "error")
                        index_value = row.get("id", row.get("_element_id"))
                        issue = {
                            "dataset_name": name,
                            "record_type": record_group_name,
                            "record_index": position + 1,
                            "record_id": index_value,
                            "record_path": location,
                            "xml_path": xml_path,
                            "element_or_attribute": field_path.rsplit("/", 1)[-1],
                            "invalid_value": value,
                            "validation_rule": f"required_{kind}",
                            "error_message": (
                                f"Required {kind} {field_path!r} is missing."
                                if not found
                                else f"Required {kind} {field_path!r} is empty."
                            ),
                            "severity": severity,
                        }
                        (errors if severity == "error" else warnings).append(issue)
                        if severity == "error":
                            invalid_rows.add(position)
                    else:
                        counts["valid_values"] += 1

    configured_rules = []
    if validate_numerical:
        configured_rules.append((numerical_rules or {}, "numeric"))
    if validate_dates:
        configured_rules.append((date_rules or {}, "date"))
    for rules, kind in configured_rules:
        for rule_path, raw_rule in rules.items():
            field_path = _frame_rule_field(str(rule_path), record_group_name)
            for position, (_, series) in enumerate(dataframe.iterrows()):
                row = series.to_dict()
                found, value = _frame_value(row, field_path)
                key = f"{record_group_name}/{field_path}"
                counts = field_summary.setdefault(
                    key,
                    {"total_values_checked": 0, "valid_values": 0,
                     "invalid_values": 0, "missing_values": 0, "error_count": 0},
                )
                counts["total_values_checked"] += 1
                message = None
                if not found or _frame_missing(value):
                    message = "Configured value is missing or empty."
                    counts["missing_values"] += 1
                elif kind == "numeric":
                    rule = _numeric_rule(raw_rule)
                    value_type = str(rule.get("type", "float")).lower()
                    minimum = rule.get("minimum", rule.get("min"))
                    maximum = rule.get("maximum", rule.get("max"))
                    minimum = _numeric_bound(minimum, str(rule_path), "minimum")
                    maximum = _numeric_bound(maximum, str(rule_path), "maximum")
                    try:
                        if value_type not in {"int", "integer", "float", "decimal"}:
                            raise ValueError(f"Unsupported numeric type {value_type!r}.")
                        parsed = int(str(value).strip()) if value_type in {"int", "integer"} else float(
                            str(value).strip())
                        if isinstance(parsed, float) and not math.isfinite(parsed):
                            raise ValueError
                        if rule.get("allow_negative") is False and parsed < 0:
                            message = "Negative values are not permitted."
                        if message is None and minimum is not None and parsed < minimum:
                            message = f"Value is below the minimum {minimum}."
                        if message is None and maximum is not None and parsed > maximum:
                            message = f"Value is above the maximum {maximum}."
                    except (ValueError, TypeError, OverflowError):
                        message = f"Value is not a valid {value_type}."
                else:
                    if isinstance(raw_rule, str):
                        formats = [raw_rule]
                        date_required = False
                    elif isinstance(raw_rule, Mapping):
                        configured_formats = raw_rule.get("formats", raw_rule.get("format", []))
                        formats = [configured_formats] if isinstance(configured_formats, str) else list(
                            configured_formats)
                        date_required = bool(raw_rule.get("required", False))
                    else:
                        formats, date_required = list(raw_rule), False
                    text = str(value).strip() if found else ""
                    if not text and not date_required:
                        counts["missing_values"] += 1
                        continue
                    try:
                        if formats:
                            valid_date = any(
                                _date_with_format(text, str(fmt)) for fmt in formats
                            )
                        else:
                            _parse_iso_date(text)
                            valid_date = True
                    except ValueError:
                        valid_date = False
                    if not text or not valid_date:
                        expected = ", ".join(map(str, formats)) if formats else "ISO 8601"
                        message = f"Date value is invalid; expected {expected}."
                if message:
                    severity = str(raw_rule.get("severity", "error")) if isinstance(raw_rule, Mapping) else "error"
                    counts["invalid_values"] += 1
                    counts["error_count"] += int(severity == "error")
                    row_location = str(
                        row.get("_element_id")
                        or f"{record_group_name}[{position + 1}]"
                    )
                    errors_or_warnings = errors if severity == "error" else warnings
                    errors_or_warnings.append(
                        {
                            "dataset_name": name,
                            "record_type": record_group_name,
                            "record_index": position + 1,
                            "record_id": row.get("id", row.get("_element_id")),
                            "record_path": row_location,
                            "xml_path": f"{row_location}/{field_path}",
                            "element_or_attribute": field_path.rsplit("/", 1)[-1],
                            "invalid_value": value if found else None,
                            "validation_rule": f"{kind}_{field_path}",
                            "error_message": message,
                            "severity": severity,
                        }
                    )
                    if severity == "error":
                        invalid_rows.add(position)
                else:
                    counts["valid_values"] += 1

    valid_data = dataframe.iloc[
        [position for position in range(len(dataframe)) if position not in invalid_rows]
    ].copy()
    invalid_data = dataframe.iloc[sorted(invalid_rows)].copy()
    total = len(dataframe)
    invalid_count = len(invalid_rows)
    status = "failed" if errors else "passed_with_warnings" if warnings else "passed"
    element_summary = [
        {"element_name": path.rsplit("/", 1)[-1], "xml_path": path, **counts}
        for path, counts in field_summary.items()
        if not path.rsplit("/", 1)[-1].startswith("@")
    ]
    attribute_summary = [
        {"attribute_name": path.rsplit("@", 1)[-1], "xml_path": path, **counts}
        for path, counts in field_summary.items()
        if "/@" in path
    ]
    report = {
        "dataset_name": name,
        "root_element": record_group_name,
        "total_element_count": 0,
        "total_attribute_count": 0,
        "record_groups": [
            {"record_type": record_group_name, "xml_path": f"/{record_group_name}",
             "record_count": total}
        ],
        "total_records_checked": total,
        "valid_records": total - invalid_count,
        "invalid_records": invalid_count,
        "total_validation_errors": len(errors),
        "total_validation_warnings": len(warnings),
        "validation_status": status,
        "element_summary": element_summary,
        "attribute_summary": attribute_summary,
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


def _date_with_format(value: str, fmt: str) -> bool:
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False


def _parse_iso_date(value: str) -> date | datetime:
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return date.fromisoformat(text)


def _safe_json(value: Any) -> Any:
    if isinstance(value, ET.Element):
        return ET.tostring(value, encoding="unicode")
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def export_xml_validation_result(
        result: Mapping[str, Any],
        output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> dict[str, str]:
    """Export a JSON summary and CSV issue report in a unique run directory."""
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
    issues_path = run_directory / "validation_errors.csv"
    report_path.write_text(
        json.dumps(_safe_json(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    issues = list(result.get("validation_errors", [])) + list(
        result.get("validation_warnings", [])
    )
    columns = [
        "dataset_name", "record_type", "record_index", "record_id", "record_path",
        "xml_path", "element_or_attribute", "invalid_value", "validation_rule",
        "error_message", "severity",
    ]
    pd.DataFrame(
        [
            {
                **issue,
                "invalid_value": json.dumps(
                    _safe_json(issue.get("invalid_value")),
                    ensure_ascii=False,
                    default=str,
                ),
            }
            for issue in issues
        ],
        columns=columns,
    ).to_csv(issues_path, index=False, quoting=csv.QUOTE_MINIMAL)

    paths = {
        "directory": str(run_directory),
        "report_json": str(report_path),
        "errors_csv": str(issues_path),
    }
    for key, filename in (("valid_xml", "valid.xml"), ("invalid_xml", "invalid.xml")):
        xml_root = result.get("valid_data" if key == "valid_xml" else "invalid_data")
        if isinstance(xml_root, ET.Element):
            for prefix, uri in result.get("namespaces", {}).items():
                try:
                    ET.register_namespace(prefix, uri)
                except ValueError:
                    continue
            tree = ET.ElementTree(copy.deepcopy(xml_root))
            ET.indent(tree, space="  ")
            target = run_directory / filename
            tree.write(
                target,
                encoding="utf-8",
                xml_declaration=bool(result.get("xml_declaration", True)),
            )
            ET.parse(target)
            paths[key] = str(target)
    return paths
