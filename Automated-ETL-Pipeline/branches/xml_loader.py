"""Transactional loading of validated XML documents into related MySQL tables."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .csv_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRACKING_TABLE,
    _insert_tracking_record,
    _redact_password,
    create_tracking_table,
    get_connection_config_from_environment,
    get_pipeline_execution_history,
    prepare_mysql_database,
    quote_identifier,
    validate_identifier,
)


_INTERNAL_COLUMNS = {
    "_etl_record_key",
    "_etl_parent_key",
    "_etl_root_key",
    "_etl_position",
}
_XML_NAMESPACE_TABLE = "_etl_xml_namespaces"
_XML_TABLE_REGISTRY = "_etl_xml_tables"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _split_qname(value: str) -> tuple[str, str]:
    if value.startswith("{") and "}" in value:
        uri, local_name = value[1:].split("}", 1)
        return uri, local_name
    return "", value


def _table_segment(tag: str) -> str:
    uri, local_name = _split_qname(tag)
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", local_name).strip("_") or "element"
    if safe[0].isdigit():
        safe = f"element_{safe}"
    if uri:
        safe = f"{safe}_{hashlib.sha1(uri.encode('utf-8')).hexdigest()[:8]}"
    return safe.lower()


def _table_name(base: str, tags: Sequence[str]) -> str:
    if len(tags) <= 1:
        return validate_identifier(base, kind="table")
    suffix = "__".join(_table_segment(tag) for tag in tags[1:])
    candidate = f"{base}__{suffix}" if suffix else base
    if len(candidate) > 64:
        marker = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{marker}"
    return validate_identifier(candidate, kind="table")


def _element_count(root: ET.Element) -> int:
    return sum(1 for _ in root.iter())


def _root_key(root: ET.Element, unique_key_field: str | None) -> str:
    if unique_key_field:
        if unique_key_field.startswith("attr:"):
            field = unique_key_field[5:]
            value = root.attrib.get(field)
        elif unique_key_field.startswith("child:"):
            field = unique_key_field[6:]
            value = next(
                (child.text for child in list(root) if child.tag == field), None
            )
        else:
            raise ValueError("XML unique key must select a root attribute or child.")
        if value is None or not value.strip():
            raise ValueError(
                f"Selected XML unique key {unique_key_field!r} is missing or empty."
            )
        return _digest([root.tag, unique_key_field, value.strip()])
    return _digest([root.tag, ET.tostring(root, encoding="unicode")])


def get_xml_key_candidates(root: ET.Element) -> list[dict[str, str]]:
    """Describe actual root attributes and scalar children usable as stable keys."""
    candidates = []
    for name in root.attrib:
        uri, local = _split_qname(str(name))
        candidates.append(
            {
                "field": f"attr:{name}",
                "label": f"Root attribute @{local}"
                + (f" (namespace {uri})" if uri else ""),
            }
        )
    child_tag_counts: dict[str, int] = {}
    for child in list(root):
        child_tag_counts[str(child.tag)] = child_tag_counts.get(str(child.tag), 0) + 1
    seen_child_tags: set[str] = set()
    for child in list(root):
        if (
            child_tag_counts[str(child.tag)] != 1
            or str(child.tag) in seen_child_tags
        ):
            continue
        seen_child_tags.add(str(child.tag))
        if not list(child) and child.text and child.text.strip():
            uri, local = _split_qname(str(child.tag))
            candidates.append(
                {
                    "field": f"child:{child.tag}",
                    "label": f"Root child {local}"
                    + (f" (namespace {uri})" if uri else ""),
                }
            )
    return candidates


def get_xml_structure_summary(root: ET.Element, table_name: str) -> dict[str, Any]:
    """Summarize actual XML paths, counts, attributes, and proposed tables."""
    validate_identifier(table_name, kind="table")
    plans, parent_tables, path_labels, _ = _build_xml_plan(
        root,
        table_name,
        unique_key_field=None,
        namespaces={},
    )
    relationships = []
    for current_table, rows in plans.items():
        relationships.append(
            {
                "table": current_table,
                "parent_table": parent_tables[current_table],
                "xml_path": "/" + "/".join(path_labels[current_table]),
                "element_count": len(rows),
                "attributes": sorted(
                    {
                        attribute
                        for row in rows
                        for attribute in row["attributes"]
                    }
                ),
            }
        )
    return {
        "root_element": str(root.tag),
        "total_elements": _element_count(root),
        "table_relationships": relationships,
        "key_candidates": get_xml_key_candidates(root),
    }


def _namespace_for(value: str) -> str:
    return _split_qname(value)[0]


def _build_xml_plan(
    root: ET.Element,
    table_name: str,
    *,
    unique_key_field: str | None,
    namespaces: Mapping[str, str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, str | None],
    dict[str, tuple[str, ...]],
    str,
]:
    root_key = _root_key(root, unique_key_field)
    plans: dict[str, list[dict[str, Any]]] = {}
    parent_tables: dict[str, str | None] = {}
    path_labels: dict[str, tuple[str, ...]] = {}

    def visit(
        element: ET.Element,
        path: tuple[str, ...],
        parent_key: str | None,
        sibling_index: int,
    ) -> None:
        table = _table_name(table_name, path)
        parent_table = (
            _table_name(table_name, path[:-1]) if len(path) > 1 else None
        )
        previous_parent = parent_tables.setdefault(table, parent_table)
        if previous_parent != parent_table:
            raise ValueError(
                "Distinct XML element paths map to the same MySQL table name."
            )
        path_labels.setdefault(table, path)
        node_key = (
            root_key
            if parent_key is None
            else _digest([parent_key, element.tag, sibling_index])
        )
        row = {
            "_etl_record_key": node_key,
            "_etl_parent_key": parent_key,
            "_etl_root_key": root_key,
            "_etl_position": sibling_index,
            "tag_name": str(element.tag),
            "namespace_uri": _namespace_for(str(element.tag)),
            "text_content": element.text,
            "tail_content": element.tail,
            "attributes": dict(element.attrib),
        }
        plans.setdefault(table, []).append(row)

        child_counts: dict[str, int] = {}
        for child in list(element):
            tag = str(child.tag)
            index = child_counts.get(tag, 0)
            child_counts[tag] = index + 1
            visit(child, (*path, tag), node_key, index)

    visit(root, (str(root.tag),), None, 0)
    return plans, parent_tables, path_labels, root_key


def _create_xml_tables(
    connection: Any,
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_tables: Mapping[str, str | None],
    path_labels: Mapping[str, tuple[str, ...]],
    *,
    root_table: str,
) -> None:
    cursor = connection.cursor()
    try:
        for table_name in plans:
            definitions = [
                "`_etl_record_key` VARCHAR(64) NOT NULL PRIMARY KEY",
                "`_etl_parent_key` VARCHAR(64) NULL",
                "`_etl_root_key` VARCHAR(64) NOT NULL",
                "`_etl_position` BIGINT NOT NULL",
                "`tag_name` VARCHAR(1024) NOT NULL",
                "`namespace_uri` LONGTEXT NULL",
                "`text_content` LONGTEXT NULL",
                "`tail_content` LONGTEXT NULL",
                "`attributes` JSON NOT NULL",
            ]
            parent_table = parent_tables[table_name]
            if parent_table:
                fk_name = "fk_" + hashlib.sha1(
                    f"{table_name}_{parent_table}".encode("utf-8")
                ).hexdigest()[:12]
                definitions.append(
                    f"CONSTRAINT `{fk_name}` FOREIGN KEY (`_etl_parent_key`) "
                    f"REFERENCES {quote_identifier(parent_table, kind='table')} "
                    "(`_etl_record_key`)"
                )
            elif table_name != root_table:
                raise ValueError(
                    f"XML hierarchy table {table_name!r} has no parent table."
                )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS "
                f"{quote_identifier(table_name, kind='table')} "
                f"({', '.join(definitions)})"
            )
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(_XML_NAMESPACE_TABLE, kind='table')} "
            "(root_table VARCHAR(64) CHARACTER SET ascii NOT NULL, "
            "source_digest CHAR(64) CHARACTER SET ascii NOT NULL, "
            "source_filename LONGTEXT NOT NULL, "
            "prefix_digest CHAR(64) CHARACTER SET ascii NOT NULL, "
            "prefix_name LONGTEXT NOT NULL, namespace_uri LONGTEXT NOT NULL, "
            "PRIMARY KEY (root_table, source_digest, prefix_digest))"
        )
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(_XML_TABLE_REGISTRY, kind='table')} "
            "(root_table VARCHAR(64) NOT NULL, child_table VARCHAR(64) NOT NULL, "
            "depth BIGINT NOT NULL, PRIMARY KEY (root_table, child_table))"
        )
        for child_table, parent_table in parent_tables.items():
            if parent_table:
                cursor.execute(
                    f"INSERT IGNORE INTO "
                    f"{quote_identifier(_XML_TABLE_REGISTRY, kind='table')} "
                    "(root_table, child_table, depth) VALUES (%s, %s, %s)",
                    (root_table, child_table, len(path_labels[child_table])),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def _insert_namespace_mappings(
    connection: Any,
    root_table: str,
    source_filename: str,
    namespaces: Mapping[str, str],
) -> None:
    cursor = connection.cursor()
    try:
        cursor.executemany(
            f"INSERT INTO {quote_identifier(_XML_NAMESPACE_TABLE, kind='table')} "
            "(root_table, source_digest, source_filename, prefix_digest, "
            "prefix_name, namespace_uri) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
            "namespace_uri = VALUES(namespace_uri)",
            [
                (
                    root_table,
                    _digest(source_filename),
                    source_filename,
                    _digest(str(prefix)),
                    str(prefix),
                    str(uri),
                )
                for prefix, uri in namespaces.items()
            ],
        )
    finally:
        cursor.close()


def _persist_nodes(
    connection: Any,
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    batch_size: int,
) -> None:
    cursor = connection.cursor()
    try:
        for table_name, rows in plans.items():
            if not rows:
                continue
            columns = list(rows[0])
            quoted_columns = ", ".join(
                quote_identifier(column, kind="column") for column in columns
            )
            placeholders = ", ".join(["%s"] * len(columns))
            query = (
                f"INSERT INTO {quote_identifier(table_name, kind='table')} "
                f"({quoted_columns}) VALUES ({placeholders})"
            )
            values = [
                tuple(
                    _json_text(row[column])
                    if column == "attributes"
                    else row[column]
                    for column in columns
                )
                for row in rows
            ]
            for start in range(0, len(values), batch_size):
                cursor.executemany(query, values[start : start + batch_size])
    finally:
        cursor.close()


def load_validated_xml_data(
    root: ET.Element,
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_field: str | None = None,
    duplicate_strategy: str = "skip",
    dataset_name: str = "xml_dataset",
    source_filename: str | None = None,
    namespaces: Mapping[str, str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load one validated XML document while preserving its element hierarchy."""
    if not isinstance(root, ET.Element):
        raise TypeError("XML data must be an ElementTree Element.")
    validate_identifier(table_name, kind="table")
    validate_identifier(tracking_table, kind="table")
    if table_name in {tracking_table, _XML_NAMESPACE_TABLE, _XML_TABLE_REGISTRY}:
        raise ValueError(f"{table_name!r} is reserved for pipeline metadata.")
    if tracking_table in {_XML_NAMESPACE_TABLE, _XML_TABLE_REGISTRY}:
        raise ValueError(f"{tracking_table!r} is reserved for XML metadata.")
    if duplicate_strategy not in {"skip", "update", "reject"}:
        raise ValueError("duplicate_strategy must be skip, update, or reject.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    source_label = source_filename or str(dataset_name)
    namespace_map = dict(namespaces or {})
    plans, parent_tables, path_labels, root_key = _build_xml_plan(
        root,
        table_name,
        unique_key_field=unique_key_field,
        namespaces=namespace_map,
    )
    processed_count = _element_count(root)
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    execution: dict[str, Any] = {
        "execution_id": str(uuid.uuid4()),
        "dataset_name": str(dataset_name),
        "source_filename": source_filename,
        "destination_table": f"xml:{table_name}"[:64],
        "start_time": started,
        "end_time": None,
        "execution_status": "running",
        "processed_rows": processed_count,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
    }

    connection = None
    tracking_available = False
    try:
        prepared = prepare_mysql_database(
            connection_config, connect_factory=connect_factory
        )
        connection = prepared["connection"]
        create_tracking_table(connection, tracking_table=tracking_table)
        tracking_available = True
        _create_xml_tables(
            connection,
            plans,
            parent_tables,
            path_labels,
            root_table=table_name,
        )

        cursor = connection.cursor()
        try:
            cursor.execute(
                f"SELECT 1 FROM {quote_identifier(table_name, kind='table')} "
                "WHERE `_etl_record_key` = %s LIMIT 1 FOR UPDATE",
                (root_key,),
            )
            exists = cursor.fetchone() is not None
        finally:
            cursor.close()

        if exists and duplicate_strategy == "reject":
            raise ValueError("XML document with this key already exists.")
        if exists and duplicate_strategy == "skip":
            execution["skipped_rows"] = processed_count
        else:
            if exists:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"SELECT child_table FROM "
                        f"{quote_identifier(_XML_TABLE_REGISTRY, kind='table')} "
                        "WHERE root_table = %s ORDER BY depth DESC",
                        (table_name,),
                    )
                    registered = cursor.fetchall()
                    for item in registered:
                        child_table = (
                            item.get("child_table")
                            if isinstance(item, Mapping)
                            else item[0]
                        )
                        cursor.execute(
                            f"DELETE FROM "
                            f"{quote_identifier(child_table, kind='table')} "
                            "WHERE `_etl_root_key` = %s",
                            (root_key,),
                        )
                finally:
                    cursor.close()
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        f"DELETE FROM {quote_identifier(table_name, kind='table')} "
                        "WHERE `_etl_record_key` = %s",
                        (root_key,),
                    )
                finally:
                    cursor.close()
                execution["updated_rows"] = processed_count
            _persist_nodes(connection, plans, batch_size=batch_size)
            _insert_namespace_mappings(
                connection, table_name, source_label, namespace_map
            )
            connection.commit()
            if not exists:
                execution["inserted_rows"] = processed_count
        execution["execution_status"] = "success"
    except Exception as error:
        if connection is not None:
            try:
                connection.rollback()
            except Exception as rollback_error:
                execution["rollback_error"] = _redact_password(
                    rollback_error, connection_config
                )
        execution["execution_status"] = "failed"
        execution["failed_rows"] = max(
            0,
            execution["processed_rows"]
            - execution["inserted_rows"]
            - execution["updated_rows"]
            - execution["skipped_rows"],
        )
        execution["error_message"] = _redact_password(error, connection_config)
        execution["error_type"] = type(error).__name__
        if connection is not None and tracking_available:
            try:
                create_tracking_table(connection, tracking_table=tracking_table)
                execution["end_time"] = datetime.now(timezone.utc).replace(
                    tzinfo=None
                )
                _insert_tracking_record(connection, tracking_table, execution)
                execution["tracking_saved"] = True
            except Exception as tracking_error:
                execution["tracking_saved"] = False
                execution["tracking_error"] = _redact_password(
                    tracking_error, connection_config
                )
        else:
            execution["tracking_saved"] = False
        if connection is not None:
            connection.close()
        return execution

    execution["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        _insert_tracking_record(connection, tracking_table, execution)
        execution["tracking_saved"] = True
    except Exception as error:
        connection.rollback()
        execution["execution_status"] = "failed"
        execution["tracking_saved"] = False
        execution["tracking_error"] = _redact_password(error, connection_config)
        execution["error_message"] = (
            "XML data was committed, but execution tracking failed: "
            + _redact_password(error, connection_config)
        )
    finally:
        connection.close()
    execution["table_relationships"] = [
        {
            "table": child_table,
            "parent_table": parent_tables[child_table],
            "xml_path": "/" + "/".join(path_labels[child_table]),
        }
        for child_table in plans
    ]
    execution["root_key"] = root_key
    execution["namespace_mappings"] = namespace_map
    return execution


def load_validated_xml_dataset(
    dataset: Mapping[str, Any],
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_field: str | None = None,
    duplicate_strategy: str = "skip",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load an XML document only when Stage 4 validation passed."""
    validation = dataset.get("validation_result", dataset)
    if not isinstance(validation, Mapping):
        raise TypeError("XML dataset must contain a validation-result mapping.")
    if validation.get("validation_status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("Only successfully validated XML datasets can be loaded.")
    root = validation.get("valid_data")
    if not isinstance(root, ET.Element):
        raise TypeError("Validated dataset has no valid XML Element.")
    report = validation.get("validation_report", {})
    source = dataset.get("source") or dataset.get("filename")
    dataset_name = (
        report.get("dataset_name") if isinstance(report, Mapping) else None
    ) or source or table_name
    return load_validated_xml_data(
        root,
        connection_config=connection_config,
        table_name=table_name,
        unique_key_field=unique_key_field,
        duplicate_strategy=duplicate_strategy,
        dataset_name=str(dataset_name),
        source_filename=str(source) if source else None,
        namespaces=dataset.get("namespaces") or validation.get("namespaces") or {},
        batch_size=batch_size,
        tracking_table=tracking_table,
        connect_factory=connect_factory,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRACKING_TABLE",
    "get_connection_config_from_environment",
    "get_pipeline_execution_history",
    "get_xml_key_candidates",
    "get_xml_structure_summary",
    "load_validated_xml_data",
    "load_validated_xml_dataset",
    "prepare_mysql_database",
    "validate_identifier",
]
