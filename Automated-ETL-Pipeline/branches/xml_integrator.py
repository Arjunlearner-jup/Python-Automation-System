"""Hierarchy-preserving integration operations for cleaned, validated XML."""

from __future__ import annotations

import copy
import difflib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote
from uuid import uuid4
from xml.etree import ElementTree as ET

import pandas as pd


DEFAULT_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "integrated" / "xml"
)
DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "integration" / "xml"
)
INTEGRATION_NAMESPACE = "urn:automated-etl:integration"
INTEGRATION_PREFIX = "etl"
ET.register_namespace(INTEGRATION_PREFIX, INTEGRATION_NAMESPACE)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem).strip("._") or "dataset"


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1] if name.startswith("{") else name


def _path_name(name: str) -> str:
    if name.startswith("{"):
        namespace, local = name[1:].split("}", 1)
        return f"{{{quote(namespace, safe='')}}}{local}"
    return name


def _decode_path_name(name: str) -> str:
    if name.startswith("{"):
        namespace, local = name[1:].split("}", 1)
        return f"{{{unquote(namespace)}}}{local}"
    return name


def _split_path(path: str) -> list[str]:
    parts = []
    start = 0
    in_namespace = False
    for index, character in enumerate(path):
        if character == "{":
            in_namespace = True
        elif character == "}":
            in_namespace = False
        elif character == "/" and not in_namespace:
            if index > start:
                parts.append(path[start:index])
            start = index + 1
    if start < len(path):
        parts.append(path[start:])
    return parts


def _normalize_field(field: str) -> str:
    field = field.strip()
    if field.startswith("@"):
        return "@" + _path_name(_decode_path_name(field[1:]))
    parts = _split_path(field.strip("/"))
    return "/" + "/".join(_path_name(_decode_path_name(part)) for part in parts)


def _name_matches(actual: str, expected: str) -> bool:
    expected = _decode_path_name(expected)
    if expected.startswith("{"):
        return actual == expected
    return _local_name(actual) == expected


def _coerce_root(value: Any) -> ET.Element:
    if isinstance(value, ET.ElementTree):
        return value.getroot()
    if isinstance(value, ET.Element):
        return value
    if isinstance(value, bytes | str):
        try:
            return ET.fromstring(value)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML document: {exc}") from exc
    raise TypeError("XML input must be an Element, ElementTree, or XML text/bytes.")


def _extract_dataset(
    dataset: Any, identifier: str
) -> tuple[ET.Element, str, dict[str, str], bool, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    declaration = True
    namespaces: dict[str, str] = {}
    source_name = identifier
    root_value = dataset

    if isinstance(dataset, Mapping):
        validation = dataset.get("validation_result")
        if validation is None and (
            "valid_data" in dataset
            and ("validation_status" in dataset or "validation_report" in dataset)
        ):
            validation = dataset
        if isinstance(validation, Mapping):
            status = validation.get("validation_status")
            if status in {"failed", "error"}:
                raise ValueError(f"XML dataset {identifier!r} did not pass validation.")
            if not isinstance(validation.get("valid_data"), ET.Element):
                raise ValueError(
                    f"Validated XML dataset {identifier!r} has no valid XML root."
                )
            root_value = validation["valid_data"]
            report = validation.get("validation_report", {})
            source_name = str(
                (report.get("dataset_name") if isinstance(report, Mapping) else None)
                or dataset.get("source")
                or identifier
            )
        elif isinstance(dataset.get("xml_tree"), ET.Element):
            root_value = dataset["xml_tree"]
            report = dataset.get("report", {})
            source_name = str(
                (report.get("dataset_name") if isinstance(report, Mapping) else None)
                or dataset.get("source")
                or dataset.get("metadata", {}).get("source")
                or identifier
            )
            declaration = bool(dataset.get("xml_declaration", True))
        elif isinstance(dataset.get("data"), ET.Element):
            root_value = dataset["data"]
            report = dataset.get("report", {})
            source_name = str(
                (report.get("dataset_name") if isinstance(report, Mapping) else None)
                or dataset.get("source")
                or identifier
            )
            declaration = bool(dataset.get("xml_declaration", True))
        metadata = dict(dataset.get("metadata") or {})
        namespaces = dict(dataset.get("namespaces") or metadata.get("namespaces") or {})
        declaration = bool(
            dataset.get("xml_declaration", validation.get("xml_declaration", declaration)
                        if isinstance(validation, Mapping) else declaration)
        )
    root = _coerce_root(root_value)
    if not namespaces:
        namespaces = dict(dataset.get("namespaces") or {}) if isinstance(dataset, Mapping) else {}
    return copy.deepcopy(root), source_name, namespaces, declaration, metadata


def _walk(root: ET.Element):
    def visit(
        node: ET.Element,
        path: str,
        locations: tuple[int, ...],
        parent_location: tuple[int, ...] | None,
    ):
        yield node, path, locations, parent_location
        counts: defaultdict[str, int] = defaultdict(int)
        for index, child in enumerate(list(node)):
            child_name = _path_name(str(child.tag))
            counts[child_name] += 1
            child_path = f"{path}/{child_name}[{counts[child_name]}]"
            yield from visit(child, child_path, locations + (index,), locations)

    yield from visit(root, f"/{_path_name(str(root.tag))}[1]", (), None)


def _structural_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path)


def _record_group_candidates(root: ET.Element) -> list[dict[str, Any]]:
    entries = list(_walk(root))
    children_by_parent: dict[
        tuple[int, ...], list[tuple[ET.Element, str, tuple[int, ...]]]
    ] = defaultdict(list)
    for node, path, location, parent_location in entries:
        if parent_location is not None:
            children_by_parent[parent_location].append((node, path, location))
    groups: dict[str, dict[str, Any]] = {}
    for parent_location, children in children_by_parent.items():
        by_tag: dict[str, list[tuple[ET.Element, str, tuple[int, ...]]]] = defaultdict(list)
        for item in children:
            by_tag[str(item[0].tag)].append(item)
        for items in by_tag.values():
            if len(items) < 2:
                continue
            path = _structural_path(items[0][1])
            group = groups.setdefault(
                path,
                {
                    "path": path,
                    "element_name": str(items[0][0].tag),
                    "records": [],
                    "parent_locations": set(),
                    "record_locations": [],
                },
            )
            group["parent_locations"].add(parent_location)
            for node, _, location in items:
                group["records"].append(node)
                group["record_locations"].append(location)
    for group in groups.values():
        group["parent_count"] = len(group["parent_locations"])
    return list(groups.values())


def get_xml_record_groups(root: ET.Element) -> list[dict[str, Any]]:
    """Describe repeated same-tag sibling groups without flattening the tree."""
    parsed = _coerce_root(root)
    return [
        {
            "path": group["path"],
            "element_name": group["element_name"],
            "record_count": len(group["records"]),
            "parent_count": group["parent_count"],
        }
        for group in _record_group_candidates(parsed)
    ]


def _resolve_group(root: ET.Element, record_path: str | None) -> dict[str, Any]:
    groups = _record_group_candidates(root)
    if record_path:
        requested = record_path.rstrip("/")
        exact = [group for group in groups if group["path"] == requested]
        if not exact:
            requested_parts = [
                _local_name(_decode_path_name(part))
                for part in requested.strip("/").split("/")
            ]
            exact = [
                group
                for group in groups
                if [
                    _local_name(_decode_path_name(part))
                    for part in group["path"].strip("/").split("/")
                ] == requested_parts
            ]
        if not exact:
            requested_structural = _structural_path(requested)
            entries = [
                (node, path, location, parent)
                for node, path, location, parent in _walk(root)
                if _structural_path(path) == requested_structural
                or [
                    _local_name(_decode_path_name(part))
                    for part in _split_path(_structural_path(path).strip("/"))
                ]
                == [
                    _local_name(_decode_path_name(part))
                    for part in _split_path(requested_structural.strip("/"))
                ]
            ]
            if not entries:
                raise ValueError(
                    f"Record path {record_path!r} did not match an XML element."
                )
            parent_locations = {item[3] for item in entries}
            selected = {
                "path": requested_structural,
                "element_name": str(entries[0][0].tag),
                "records": [item[0] for item in entries],
                "parent_locations": parent_locations,
                "record_locations": [item[2] for item in entries],
                "parent_count": len(parent_locations),
            }
        elif len(exact) != 1:
            raise ValueError(
                f"Record group {record_path!r} matched {len(exact)} repeating groups."
            )
        else:
            selected = exact[0]
    else:
        eligible = [group for group in groups if group["parent_count"] == 1]
        if groups and eligible:
            minimum_depth = min(len(_split_path(group["path"])) for group in eligible)
            shallowest = [
                group
                for group in eligible
                if len(_split_path(group["path"])) == minimum_depth
            ]
        else:
            shallowest = []
        if len(shallowest) == 1:
            selected = shallowest[0]
        elif len(shallowest) > 1:
            raise ValueError(
                "XML contains multiple equally shallow record groups; provide "
                "record_path explicitly."
            )
        elif not groups:
            root_children = list(root)
            if len(root_children) != 1:
                raise ValueError(
                    "XML has no unique repeating record group. Provide record_path "
                    "for a singleton record group."
                )
            selected = {
                "path": f"/{_path_name(str(root.tag))}/{_path_name(str(root_children[0].tag))}",
                "element_name": str(root_children[0].tag),
                "records": [root_children[0]],
                "parent_locations": {()},
                "record_locations": [(0,)],
                "parent_count": 1,
            }
        else:
            raise ValueError(
                "XML has no unambiguous repeating record group. Provide record_path."
            )
    if selected["parent_count"] != 1:
        raise ValueError(
            f"Record group {selected['path']!r} occurs beneath multiple parent "
            "elements. Select a group with one parent to avoid breaking relationships."
        )
    return selected


def _field_type(values: Sequence[str]) -> set[str]:
    kinds = set()
    for value in values:
        text = value.strip()
        if not text:
            kinds.add("empty")
        elif re.fullmatch(r"[-+]?\d+", text):
            kinds.add("integer")
        elif re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
            kinds.add("decimal")
        elif text.lower() in {"true", "false"}:
            kinds.add("boolean")
        else:
            try:
                datetime.fromisoformat(text.replace("Z", "+00:00"))
                kinds.add("datetime")
            except ValueError:
                kinds.add("text")
    return kinds


def _schema_for_group(group: Mapping[str, Any]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    for record in group["records"]:
        for name, value in record.attrib.items():
            path = f"@{_path_name(str(name))}"
            fields.setdefault(path, set()).update(_field_type([value]))
        for node, path, _, _ in _walk(record):
            relative = path
            prefix = f"/{_path_name(str(record.tag))}[1]"
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
            relative = _structural_path(relative)
            if node.text is not None:
                fields.setdefault(relative or "/", set()).update(_field_type([node.text]))
            if node is not record:
                for name, value in node.attrib.items():
                    fields.setdefault(
                        f"{relative}/@{_path_name(str(name))}", set()
                    ).update(_field_type([value]))
    return fields


def _compatible_types(left: set[str], right: set[str]) -> bool:
    left = left - {"empty"}
    right = right - {"empty"}
    if not left or not right:
        return True
    numeric = {"integer", "decimal"}
    return left == right or (left | right) <= numeric


def match_common_elements(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    record_paths: Mapping[str, str] | None = None,
    field_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare nested element and attribute paths in two XML datasets."""
    entries = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if len(entries) < 2:
        raise ValueError("At least two XML datasets are needed to match fields.")
    left_id, left_dataset = entries[0]
    right_id, right_dataset = entries[1]
    left_root, left_name, _, _, _ = _extract_dataset(left_dataset, str(left_id))
    right_root, right_name, _, _, _ = _extract_dataset(right_dataset, str(right_id))
    left_group = _resolve_group(left_root, (record_paths or {}).get(str(left_id)))
    right_group = _resolve_group(right_root, (record_paths or {}).get(str(right_id)))
    left_fields = _schema_for_group(left_group)
    right_fields = _schema_for_group(right_group)
    common = sorted(set(left_fields) & set(right_fields))
    right_by_local: dict[str, list[str]] = defaultdict(list)
    for path in right_fields:
        terminal = path.rsplit("/", 1)[-1]
        name = terminal[1:] if terminal.startswith("@") else terminal
        right_by_local[_local_name(_decode_path_name(name)).lower()].append(path)
    suggestions = []
    for path in left_fields:
        terminal = path.rsplit("/", 1)[-1]
        name = terminal[1:] if terminal.startswith("@") else terminal
        left_local_name = _local_name(_decode_path_name(name)).lower()
        left_tokens = set(re.split(r"[^a-z0-9]+", left_local_name)) - {""}
        candidates = [
            candidate
            for candidate in right_fields
            if candidate != path
            and candidate not in common
            and (
                _local_name(
                    _decode_path_name(
                        candidate.rsplit("/", 1)[-1].lstrip("@")
                    )
                ).lower()
                in right_by_local
                and (
                    left_tokens
                    & set(
                        re.split(
                            r"[^a-z0-9]+",
                            _local_name(
                                _decode_path_name(
                                    candidate.rsplit("/", 1)[-1].lstrip("@")
                                )
                            ).lower(),
                        )
                    )
                    - {""}
                )
                or difflib.SequenceMatcher(
                    None,
                    left_local_name,
                    _local_name(
                        _decode_path_name(candidate.rsplit("/", 1)[-1].lstrip("@"))
                    ).lower(),
                ).ratio()
                >= 0.6
            )
        ]
        if candidates:
            suggestions.append(
                {
                    "left_field": path,
                    "right_field_candidates": candidates,
                    "compatible_types": all(
                        _compatible_types(left_fields[path], right_fields[candidate])
                        for candidate in candidates
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
        item["left_field"]: item["right_field"]
        for item in mappings
        if item["exists"] and item["compatible_types"]
    }
    return {
        "left_dataset": left_name,
        "right_dataset": right_name,
        "left_record_group": left_group["path"],
        "right_record_group": right_group["path"],
        "common_fields": common,
        "common_elements": [field for field in common if not field.rsplit("/", 1)[-1].startswith("@")],
        "common_attributes": [field for field in common if field.rsplit("/", 1)[-1].startswith("@")],
        "possible_matching_fields": suggestions,
        "manual_mappings": mappings,
        "unmatched_left_fields": sorted(set(left_fields) - set(common) - set(valid_mappings)),
        "unmatched_right_fields": sorted(
            set(right_fields) - set(common) - set(valid_mappings.values())
        ),
        "incompatible_common_fields": [
            path
            for path in common
            if not _compatible_types(left_fields[path], right_fields[path])
        ],
    }


def handle_inconsistent_schemas(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    record_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report per-document XML element/attribute differences without rewriting trees."""
    entries = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    schema_rows = []
    schemas: dict[str, dict[str, set[str]]] = {}
    for key, dataset in entries:
        root, name, _, _, _ = _extract_dataset(dataset, str(key))
        group = _resolve_group(root, (record_paths or {}).get(str(key)))
        schema = _schema_for_group(group)
        schemas[str(key)] = schema
        schema_rows.append(
            {
                "dataset_id": str(key),
                "dataset_name": name,
                "root_element": str(root.tag),
                "record_group": group["path"],
                "record_type": group["element_name"],
                "record_count": len(group["records"]),
                "elements": sorted(
                    field for field in schema if not field.rsplit("/", 1)[-1].startswith("@")
                ),
                "attributes": sorted(
                    field for field in schema if field.rsplit("/", 1)[-1].startswith("@")
                ),
                "field_types": {path: sorted(kinds) for path, kinds in schema.items()},
            }
        )
    all_fields = set().union(*(set(schema) for schema in schemas.values()))
    common_fields = set.intersection(*(set(schema) for schema in schemas.values())) if schemas else set()
    for row in schema_rows:
        present = set(row["field_types"])
        row["missing_fields"] = sorted(all_fields - present)
        row["additional_fields"] = sorted(present - common_fields)
    type_report = []
    for field in sorted(all_fields):
        definitions = [
            {"dataset_id": key, "types": sorted(schema[field])}
            for key, schema in schemas.items()
            if field in schema
        ]
        types = {kind for definition in definitions for kind in definition["types"]}
        compatible = len(types - {"empty"}) <= 1 or types <= {"integer", "decimal", "empty"}
        type_report.append(
            {
                "field_path": field,
                "definitions": definitions,
                "compatible_types": compatible,
            }
        )
    return {
        "schema_report": schema_rows,
        "all_fields": sorted(all_fields),
        "common_fields": sorted(common_fields),
        "field_type_report": type_report,
    }


def _field_value(record: ET.Element, field: str) -> str | None:
    path = field.strip()
    if path.startswith("@") and "/" not in path:
        return record.attrib.get(_decode_path_name(path[1:]))
    if path.startswith("@"):
        path = path[1:]
    path = path.replace("/text()", "")
    parts = [part for part in _split_path(path.strip("/")) if part and part != "."]
    nodes = [record]
    for part in parts:
        if part.startswith("@"):
            return nodes[0].attrib.get(_decode_path_name(part[1:])) if nodes else None
        matched = []
        for node in nodes:
            matched.extend(child for child in list(node) if _name_matches(str(child.tag), part))
        nodes = matched
    if not nodes:
        return None
    values = [(node.text or "").strip() for node in nodes]
    return json.dumps(values, ensure_ascii=False) if len(values) > 1 else values[0]


def _key_value(record: ET.Element, keys: Sequence[str]) -> tuple[str | None, ...] | None:
    values = tuple(_field_value(record, key) for key in keys)
    if any(value is None or value == "" for value in values):
        return None
    return values


def _duplicate_keys(records: Sequence[ET.Element], keys: Sequence[str]) -> dict[Any, list[int]]:
    found: dict[Any, list[int]] = {}
    for index, record in enumerate(records):
        key = _key_value(record, keys)
        if key is not None:
            found.setdefault(key, []).append(index)
    return {key: indexes for key, indexes in found.items() if len(indexes) > 1}


def _merge_element(
    left: ET.Element,
    right: ET.Element,
    conflicts: list[dict[str, Any]],
    location: str,
) -> ET.Element:
    merged = copy.deepcopy(left)
    for name, value in right.attrib.items():
        if name not in merged.attrib:
            merged.set(name, value)
        elif merged.attrib[name] != value:
            conflicts.append(
                {
                    "location": location,
                    "field": f"@{name}",
                    "left_value": merged.attrib[name],
                    "right_value": value,
                    "resolution": "kept_left_and_preserved_conflict_in_report",
                }
            )
    left_text = (merged.text or "").strip()
    right_text = (right.text or "").strip()
    if not left_text and right_text:
        merged.text = right.text
    elif left_text and right_text and left_text != right_text:
        conflicts.append(
            {
                "location": location,
                "field": "text()",
                "left_value": left_text,
                "right_value": right_text,
                "resolution": "preserved_both_as_repeating_element_values",
            }
        )
        duplicate = copy.deepcopy(right)
        duplicate.tail = right.tail
        merged.append(duplicate)
    left_children = list(merged)
    for right_child in list(right):
        equivalent = [
            child
            for child in left_children
            if child.tag == right_child.tag
            and ET.tostring(child, encoding="utf-8")
            == ET.tostring(right_child, encoding="utf-8")
        ]
        if equivalent:
            continue
        same_tag = [child for child in left_children if child.tag == right_child.tag]
        if same_tag and len(same_tag) == 1:
            candidate = _merge_element(
                same_tag[0], right_child, conflicts, f"{location}/{right_child.tag}"
            )
            index = list(merged).index(same_tag[0])
            merged.remove(same_tag[0])
            merged.insert(index, candidate)
            left_children = list(merged)
        else:
            merged.append(copy.deepcopy(right_child))
            left_children.append(merged[-1])
    return merged


def _replace_records(
    root: ET.Element, group: Mapping[str, Any], records: Sequence[ET.Element]
) -> ET.Element:
    result = copy.deepcopy(root)
    parent = result
    for index in next(iter(group["parent_locations"])):
        parent = list(parent)[index]
    target_tag = group["element_name"]
    existing_positions = [
        index for index, child in enumerate(list(parent)) if child.tag == target_tag
    ]
    insert_at = min(existing_positions) if existing_positions else len(parent)
    for index in reversed(existing_positions):
        del parent[index]
    for offset, record in enumerate(records):
        parent.insert(insert_at + offset, copy.deepcopy(record))
    return result


def _source_marker(element: ET.Element, filenames: Sequence[str]) -> None:
    element.set(
        f"{{{INTEGRATION_NAMESPACE}}}source-file",
        ";".join(dict.fromkeys(filenames)),
    )


def _register_namespaces(namespaces: Sequence[Mapping[str, str]]) -> None:
    for mapping in namespaces:
        for prefix, uri in mapping.items():
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                continue


def merge_xml_datasets(
    left_dataset: Any,
    right_dataset: Any,
    *,
    join_type: str = "inner",
    key_elements: Sequence[str] | str | None = None,
    key_attributes: Sequence[str] | str | None = None,
    right_key_elements: Sequence[str] | str | None = None,
    right_key_attributes: Sequence[str] | str | None = None,
    key_fields: Sequence[str] | str | None = None,
    right_key_fields: Sequence[str] | str | None = None,
    field_mapping: Mapping[str, str] | None = None,
    left_record_path: str | None = None,
    right_record_path: str | None = None,
    many_to_many: str = "raise",
    left_name: str = "left.xml",
    right_name: str = "right.xml",
) -> dict[str, Any]:
    """Join two repeating XML record groups, preserving nested nodes and attributes."""
    join = join_type.lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise ValueError("join_type must be inner, left, right, or outer.")
    if many_to_many not in {"raise", "allow"}:
        raise ValueError("many_to_many must be 'raise' or 'allow'.")
    left_root, left_name, left_ns, left_decl, left_meta = _extract_dataset(
        left_dataset, left_name
    )
    right_root, right_name, right_ns, right_decl, right_meta = _extract_dataset(
        right_dataset, right_name
    )
    left_group = _resolve_group(left_root, left_record_path)
    right_group = _resolve_group(right_root, right_record_path)
    left_records = left_group["records"]
    right_records = right_group["records"]

    def as_list(value: Sequence[str] | str | None) -> list[str]:
        if isinstance(value, str):
            return [value]
        return list(value or [])

    mappings = dict(field_mapping or {})
    left_keys = as_list(key_fields) if key_fields is not None else []
    right_keys = as_list(right_key_fields) if right_key_fields is not None else []
    left_keys.extend("@" + key.lstrip("@") for key in as_list(key_attributes))
    right_keys.extend("@" + key.lstrip("@") for key in as_list(right_key_attributes))
    left_keys.extend(as_list(key_elements))
    right_keys.extend(as_list(right_key_elements))
    if not left_keys and mappings:
        left_keys = list(mappings)
    if not left_keys:
        raise ValueError("Select one or more XML elements or attributes as merge keys.")
    if not right_keys:
        right_keys = [mappings.get(key, key) for key in left_keys]
    left_keys = [_normalize_field(key) for key in left_keys]
    right_keys = [_normalize_field(key) for key in right_keys]
    if len(left_keys) != len(right_keys):
        raise ValueError("Left and right XML key lists must have equal lengths.")

    left_schema = _schema_for_group(left_group)
    right_schema = _schema_for_group(right_group)
    incompatible = [
        (left_key, right_key)
        for left_key, right_key in zip(left_keys, right_keys)
        if left_key not in left_schema
        or right_key not in right_schema
        or not _compatible_types(left_schema.get(left_key, set()), right_schema.get(right_key, set()))
    ]
    if incompatible:
        raise TypeError(f"Incompatible or missing XML merge keys: {incompatible}.")

    left_duplicates = _duplicate_keys(left_records, left_keys)
    right_duplicates = _duplicate_keys(right_records, right_keys)
    duplicate_both = set(left_duplicates).intersection(right_duplicates)
    if duplicate_both and many_to_many == "raise":
        raise ValueError(
            "Many-to-many XML merge would expand duplicate keys on both sides; "
            "resolve duplicate keys or set many_to_many='allow'."
        )

    right_index: dict[Any, list[int]] = {}
    for index, record in enumerate(right_records):
        key = _key_value(record, right_keys)
        if key is not None:
            right_index.setdefault(key, []).append(index)
    matched: list[dict[str, Any]] = []
    unmatched_left: list[dict[str, Any]] = []
    matched_right: set[int] = set()
    output_records: list[ET.Element] = []
    conflicts: list[dict[str, Any]] = []
    for left_index, left_record in enumerate(left_records):
        key = _key_value(left_record, left_keys)
        matches = right_index.get(key, []) if key is not None else []
        if matches:
            for right_index_value in matches:
                merged = _merge_element(
                    left_record,
                    right_records[right_index_value],
                    conflicts,
                    left_group["path"],
                )
                _source_marker(merged, [left_name, right_name])
                matched.append(
                    {
                        "left_record_index": left_index,
                        "right_record_index": right_index_value,
                        "left_source_file": left_name,
                        "right_source_file": right_name,
                        "key": list(key or ()),
                        "record_xml": ET.tostring(merged, encoding="unicode"),
                    }
                )
                output_records.append(merged)
                matched_right.add(right_index_value)
        else:
            unmatched_left.append(
                {
                    "record_index": left_index,
                    "source_file": left_name,
                    "record_xml": ET.tostring(left_record, encoding="unicode"),
                }
            )
            if join in {"left", "outer"}:
                record = copy.deepcopy(left_record)
                _source_marker(record, [left_name])
                output_records.append(record)
    unmatched_right = []
    for index, record in enumerate(right_records):
        if index not in matched_right:
            unmatched_right.append(
                {
                    "record_index": index,
                    "source_file": right_name,
                    "record_xml": ET.tostring(record, encoding="unicode"),
                }
            )
            if join in {"right", "outer"}:
                candidate = copy.deepcopy(record)
                _source_marker(candidate, [right_name])
                output_records.append(candidate)

    integrated = _replace_records(left_root, left_group, output_records)
    _register_namespaces([left_ns, right_ns])
    schema = handle_inconsistent_schemas(
        {
            "left": {"xml_tree": left_root, "namespaces": left_ns},
            "right": {"xml_tree": right_root, "namespaces": right_ns},
        },
        record_paths={
            "left": left_group["path"],
            "right": right_group["path"],
        },
    )
    report = {
        "operation": "merge",
        "integration_type": "merge",
        "dataset_name": "integrated_xml",
        "join_type": join,
        "left_dataset": left_name,
        "right_dataset": right_name,
        "left_root_element": str(left_root.tag),
        "right_root_element": str(right_root.tag),
        "output_root_element": str(integrated.tag),
        "left_record_group": left_group["path"],
        "right_record_group": right_group["path"],
        "left_key_fields": left_keys,
        "right_key_fields": right_keys,
        "left_input_records": len(left_records),
        "right_input_records": len(right_records),
        "output_records": len(output_records),
        "matched_records": len(matched),
        "unmatched_left_records": len(unmatched_left),
        "unmatched_right_records": len(unmatched_right),
        "duplicate_left_keys": len(left_duplicates),
        "duplicate_right_keys": len(right_duplicates),
        "many_to_many_allowed": many_to_many == "allow",
        "merge_conflicts": conflicts,
        "left_metadata": left_meta,
        "right_metadata": right_meta,
    }
    return {
        "data": integrated,
        "xml_document": integrated,
        "integration_report": report,
        "source_datasets": [
            {"dataset_name": left_name, "records": len(left_records), "metadata": left_meta},
            {"dataset_name": right_name, "records": len(right_records), "metadata": right_meta},
        ],
        "matched_records": matched,
        "unmatched_records": {"left": unmatched_left, "right": unmatched_right},
        "schema_report": schema,
        "namespaces": {**left_ns, **right_ns},
        "xml_declaration": left_decl or right_decl,
        "integration_status": "success",
    }


def combine_xml_files(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    selected_datasets: Sequence[str] | None = None,
    record_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Append selected repeating XML records to the first document's record group."""
    entries = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        entries = [(key, value) for key, value in entries if str(key) in selected]
    if not entries:
        raise ValueError("Select at least one XML dataset to combine.")
    extracted = []
    for key, value in entries:
        root, name, namespaces, declaration, metadata = _extract_dataset(value, str(key))
        group = _resolve_group(root, (record_paths or {}).get(str(key)))
        extracted.append((str(key), root, name, namespaces, declaration, metadata, group))
    target_tag = extracted[0][6]["element_name"]
    if any(item[6]["element_name"] != target_tag for item in extracted[1:]):
        raise ValueError(
            "Selected record groups have different qualified element names. "
            "Use separate integrations or align the XML schemas first."
        )
    base_group = extracted[0][6]
    output_records: list[ET.Element] = []
    source_counts = []
    provenance = []
    conflicts = []
    for _, _, source_name, _, _, metadata, group in extracted:
        count = len(group["records"])
        source_counts.append(
            {
                "dataset_name": source_name,
                "records_contributed": count,
                "record_group": group["path"],
                "metadata": metadata,
            }
        )
        for record_index, record in enumerate(group["records"]):
            clone = copy.deepcopy(record)
            _source_marker(clone, [source_name])
            output_index = len(output_records)
            output_records.append(clone)
            provenance.append(
                {
                    "output_record_index": output_index,
                    "source_record_index": record_index,
                    "source_file": source_name,
                }
            )
    integrated = _replace_records(extracted[0][1], base_group, output_records)
    namespaces = {}
    for item in extracted:
        namespaces.update(item[3])
    _register_namespaces([namespaces])
    schema = handle_inconsistent_schemas(
        {
            key: {"xml_tree": root, "namespaces": ns}
            for key, root, _, ns, _, _, _ in extracted
        },
        record_paths={key: group["path"] for key, _, _, _, _, _, group in extracted},
    )
    report = {
        "operation": "combine",
        "integration_type": "combine",
        "dataset_name": "integrated_xml",
        "input_dataset_count": len(extracted),
        "input_record_count": sum(item["records_contributed"] for item in source_counts),
        "output_record_count": len(output_records),
        "source_record_counts": source_counts,
        "record_provenance": provenance,
        "root_elements": [str(item[1].tag) for item in extracted],
        "output_root_element": str(integrated.tag),
        "output_record_group": base_group["path"],
        "merge_conflicts": conflicts,
    }
    return {
        "data": integrated,
        "xml_document": integrated,
        "integration_report": report,
        "source_datasets": source_counts,
        "matched_records": [],
        "unmatched_records": {"left": [], "right": []},
        "schema_report": schema,
        "namespaces": namespaces,
        "xml_declaration": any(item[4] for item in extracted),
        "integration_status": "success",
    }


def _serialize(root: ET.Element, namespaces: Mapping[str, str], declaration: bool) -> bytes:
    _register_namespaces([namespaces])
    return ET.tostring(
        copy.deepcopy(root),
        encoding="utf-8",
        xml_declaration=declaration,
    )


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
        str(result.get("integration_report", {}).get("dataset_name", "integrated_xml"))
    )
    xml_path = data_directory / f"{name}_{identifier}.xml"
    report_json = reports_directory / f"{name}_{identifier}_report.json"
    report_csv = reports_directory / f"{name}_{identifier}_report.csv"
    schema_csv = reports_directory / f"{name}_{identifier}_schema.csv"
    records_csv = reports_directory / f"{name}_{identifier}_records.csv"
    xml_bytes = _serialize(
        result["data"],
        result.get("namespaces", {}),
        bool(result.get("xml_declaration", True)),
    )
    with xml_path.open("xb") as file:
        file.write(xml_bytes)
    ET.parse(xml_path)

    payload = {
        "integration_report": result.get("integration_report", {}),
        "source_datasets": result.get("source_datasets", []),
        "matched_records": result.get("matched_records", []),
        "unmatched_records": result.get("unmatched_records", {}),
        "schema_report": result.get("schema_report", {}),
        "integration_status": result.get("integration_status", "success"),
    }
    with report_json.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
    with report_csv.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame(
            [
                {
                    "operation": payload["integration_report"].get("operation"),
                    "integration_status": payload["integration_status"],
                    **{
                        key: value
                        for key, value in payload["integration_report"].items()
                        if key not in {"operation"}
                    },
                }
            ]
        ).to_csv(file, index=False)
    schema = payload["schema_report"]
    schema_rows = schema.get("schema_report", []) if isinstance(schema, Mapping) else []
    with schema_csv.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame(schema_rows).to_csv(file, index=False)
    records = []
    for item in payload["matched_records"]:
        records.append(
            {
                "status": "matched",
                "left_record_index": item.get("left_record_index"),
                "right_record_index": item.get("right_record_index"),
                "left_source_file": item.get("left_source_file"),
                "right_source_file": item.get("right_source_file"),
                "record": item.get("record_xml"),
            }
        )
    for side, values in payload["unmatched_records"].items():
        for item in values:
            records.append(
                {
                    "status": f"unmatched_{side}",
                    "left_record_index": item.get("record_index") if side == "left" else None,
                    "right_record_index": item.get("record_index") if side == "right" else None,
                    "left_source_file": item.get("source_file") if side == "left" else None,
                    "right_source_file": item.get("source_file") if side == "right" else None,
                    "record": item.get("record_xml"),
                }
            )
    with records_csv.open("x", encoding="utf-8", newline="") as file:
        pd.DataFrame(
            records,
            columns=[
                "status",
                "left_record_index",
                "right_record_index",
                "left_source_file",
                "right_source_file",
                "record",
            ],
        ).to_csv(file, index=False)
    return {
        "data": str(xml_path),
        "report_json": str(report_json),
        "report_csv": str(report_csv),
        "schema_csv": str(schema_csv),
        "records_csv": str(records_csv),
        "data_directory": str(data_directory),
        "report_directory": str(reports_directory),
    }


def integrate_multiple_xml_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    *,
    integration_type: str = "combine",
    selected_datasets: Sequence[str] | None = None,
    record_paths: Mapping[str, str] | None = None,
    join_type: str = "inner",
    key_elements: Sequence[str] | str | None = None,
    key_attributes: Sequence[str] | str | None = None,
    right_key_elements: Sequence[str] | str | None = None,
    right_key_attributes: Sequence[str] | str | None = None,
    key_fields: Sequence[str] | str | None = None,
    right_key_fields: Sequence[str] | str | None = None,
    field_mapping: Mapping[str, str] | None = None,
    many_to_many: str = "raise",
    output_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    report_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Integrate selected validated XML documents and save non-overwriting outputs."""
    entries = list(datasets.items()) if isinstance(datasets, Mapping) else [
        (str(index), value) for index, value in enumerate(datasets)
    ]
    if selected_datasets is not None:
        selected = set(map(str, selected_datasets))
        entries = [(key, value) for key, value in entries if str(key) in selected]
    if not entries:
        raise ValueError("No selected XML datasets are available for integration.")
    operation = integration_type.lower()
    if operation in {"combine", "append"}:
        result = combine_xml_files(dict(entries), record_paths=record_paths)
    elif operation in {"merge", "join"}:
        if len(entries) < 2:
            raise ValueError("Select at least two XML datasets to merge.")
        first_key, first_dataset = entries[0]
        second_key, second_dataset = entries[1]
        result = merge_xml_datasets(
            first_dataset,
            second_dataset,
            join_type=join_type,
            key_elements=key_elements,
            key_attributes=key_attributes,
            right_key_elements=right_key_elements,
            right_key_attributes=right_key_attributes,
            key_fields=key_fields,
            right_key_fields=right_key_fields,
            field_mapping=field_mapping,
            left_record_path=(record_paths or {}).get(str(first_key)),
            right_record_path=(record_paths or {}).get(str(second_key)),
            many_to_many=many_to_many,
            left_name=str(first_key),
            right_name=str(second_key),
        )
        if len(entries) > 2:
            merged = result["data"]
            reports = []
            last_result = result
            for index, (key, dataset) in enumerate(entries[2:], start=2):
                step = merge_xml_datasets(
                    merged,
                    dataset,
                    join_type=join_type,
                    key_elements=key_elements,
                    key_attributes=key_attributes,
                    right_key_elements=right_key_elements,
                    right_key_attributes=right_key_attributes,
                    key_fields=key_fields,
                    right_key_fields=right_key_fields or key_fields,
                    field_mapping=field_mapping,
                    left_record_path=result["integration_report"]["left_record_group"],
                    right_record_path=(record_paths or {}).get(str(key)),
                    many_to_many=many_to_many,
                    left_name=f"integrated_{index}",
                    right_name=str(key),
                )
                merged = step["data"]
                reports.append(step["integration_report"])
                last_result = step
            initial_report = dict(result["integration_report"])
            result["data"] = merged
            result["xml_document"] = merged
            result["matched_records"] = last_result["matched_records"]
            result["unmatched_records"] = last_result["unmatched_records"]
            result["integration_report"]["merge_steps"] = [initial_report, *reports]
            result["integration_report"]["additional_merge_steps"] = reports
            result["integration_report"]["input_dataset_count"] = len(entries)
            result["source_datasets"] = []
            namespaces = {}
            declarations = []
            for key, dataset in entries:
                root, name, ns, declaration, metadata = _extract_dataset(dataset, str(key))
                group = _resolve_group(root, (record_paths or {}).get(str(key)))
                result["source_datasets"].append(
                    {
                        "dataset_name": name,
                        "records": len(group["records"]),
                        "metadata": metadata,
                    }
                )
                namespaces.update(ns)
                declarations.append(declaration)
            result["namespaces"] = namespaces
            result["xml_declaration"] = any(declarations)
    else:
        raise ValueError("integration_type must be 'combine' or 'merge'.")
    result["integration_report"]["integration_type"] = operation
    result["integration_report"].setdefault("dataset_name", "integrated_xml")
    result["integration_report"]["schema_comparison"] = result.get("schema_report")
    result["integration_status"] = "success"
    if save_outputs:
        result["saved_paths"] = _save_integration(
            result, output_directory, report_directory
        )
    return result
