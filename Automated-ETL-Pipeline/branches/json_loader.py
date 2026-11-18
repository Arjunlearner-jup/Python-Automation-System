"""Transactional loading of validated hierarchical JSON into MySQL."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from typing import Any

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
_JSON_TABLE_REGISTRY = "_etl_json_table_registry"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_column_name(value: str) -> str:
    if value in _INTERNAL_COLUMNS or value == "_etl_source_context":
        return validate_identifier(value, kind="column")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value.strip())
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"column_{name}"
    return validate_identifier(name[:64], kind="column")


def _table_for_path(base: str, parts: Sequence[str]) -> str:
    suffix = "__".join(_safe_column_name(part).lower() for part in parts)
    candidate = f"{base}__{suffix}" if suffix else base
    if len(candidate) > 64:
        marker = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{marker}"
    return validate_identifier(candidate, kind="table")


def _contains_array(value: Mapping[str, Any]) -> bool:
    for child in value.values():
        if isinstance(child, list):
            return True
        if isinstance(child, Mapping) and _contains_array(child):
            return True
    return False


def _primary_records(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose top-level records while retaining a single array's wrapper fields."""
    if isinstance(data, list):
        return [
            dict(item) if isinstance(item, Mapping) else {"value": item}
            for item in data
        ], {}
    if isinstance(data, Mapping):
        object_arrays = [
            (str(key), value)
            for key, value in data.items()
            if isinstance(value, list)
            and value
            and all(isinstance(item, Mapping) for item in value)
        ]
        if len(object_arrays) == 1:
            collection_name, values = object_arrays[0]
            context = {
                str(key): value
                for key, value in data.items()
                if str(key) != collection_name
            }
            return [dict(item) for item in values], context
        return [dict(data)], {}
    return [{"value": data}], {}


def suggest_unique_key_fields(data: Any) -> list[str]:
    """Suggest root-record scalar fields that are present and unique."""
    records, _ = _primary_records(data)
    if len(records) < 2:
        return []
    common = set(records[0])
    for record in records[1:]:
        common.intersection_update(record)
    suggestions = []
    for field in sorted(common):
        values = [record[field] for record in records]
        if (
            all(
                value is not None
                and isinstance(value, (str, int, float, bool))
                and not (isinstance(value, float) and not value == value)
                for value in values
            )
            and len({_canonical_json(value) for value in values}) == len(values)
        ):
            suggestions.append(field)
    return sorted(
        suggestions,
        key=lambda candidate: (
            0
            if candidate.lower() in {"id", "code", "key"}
            or candidate.lower().endswith(("_id", "_code", "_key"))
            else 1,
            candidate.lower(),
        ),
    )


def _check_key_fields(
    records: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    if len(set(fields)) != len(fields):
        raise ValueError("Unique key fields must not contain duplicates.")
    if not fields:
        return
    keys = []
    for index, record in enumerate(records):
        missing = [field for field in fields if field not in record]
        if missing:
            raise KeyError(
                f"Unique key field(s) missing from record {index}: {missing}"
            )
        values = [record[field] for field in fields]
        if any(value is None or isinstance(value, (Mapping, list)) for value in values):
            raise ValueError("Unique key fields must contain non-null scalar values.")
        keys.append(_digest(values))


def _record_key(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    return _digest([record[field] for field in fields]) if fields else _digest(record)


def _column_type(values: Sequence[Any]) -> str:
    present = [value for value in values if value is not None]
    if any(isinstance(value, (Mapping, list)) for value in present):
        return "JSON"
    if present and all(isinstance(value, bool) for value in present):
        return "BOOLEAN"
    if present and all(
        isinstance(value, int) and not isinstance(value, bool) for value in present
    ):
        return "BIGINT"
    if present and all(
        isinstance(value, Real) and not isinstance(value, bool) for value in present
    ):
        return "DOUBLE"
    return "LONGTEXT"


def _mysql_value(value: Any, column_type: str) -> Any:
    if value is None:
        return None
    if column_type == "JSON":
        return _canonical_json(value)
    return value


def _extract_record(
    record: Mapping[str, Any],
    *,
    table: str,
    key: str,
    root_key: str,
    parent_key: str | None,
    position: int | None,
    table_parts: Sequence[str],
    root_table: str,
    table_path_registry: dict[str, tuple[str, ...]],
    plans: dict[str, list[dict[str, Any]]],
    source_context: Mapping[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "_etl_record_key": key,
        "_etl_parent_key": parent_key,
        "_etl_root_key": root_key,
        "_etl_position": position,
    }
    if source_context:
        if "_etl_source_context" in record:
            raise ValueError(
                "The JSON data uses the reserved field '_etl_source_context'."
            )
        row["_etl_source_context"] = dict(source_context)

    normalized_fields: set[str] = set()
    for raw_field, value in record.items():
        field = str(raw_field)
        if field in _INTERNAL_COLUMNS or field == "_etl_source_context":
            raise ValueError(f"JSON field {field!r} is reserved for loader metadata.")
        safe_field = _safe_column_name(field)
        if safe_field in normalized_fields:
            raise ValueError(
                "JSON fields collide after conversion to MySQL identifiers."
            )
        normalized_fields.add(safe_field)
        if isinstance(value, list):
            child_parts = [*table_parts, field]
            child_table = _table_for_path(root_table, child_parts)
            _register_table_path(child_table, child_parts, table_path_registry)
            for index, item in enumerate(value):
                child_record = (
                    dict(item) if isinstance(item, Mapping) else {"value": item}
                )
                child_key = _digest([key, child_parts, index])
                _extract_record(
                    child_record,
                    table=child_table,
                    key=child_key,
                    root_key=root_key,
                    parent_key=key,
                    position=index,
                    table_parts=child_parts,
                    root_table=root_table,
                    table_path_registry=table_path_registry,
                    plans=plans,
                )
        elif isinstance(value, Mapping) and _contains_array(value):
            child_parts = [*table_parts, field]
            child_table = _table_for_path(root_table, child_parts)
            _register_table_path(child_table, child_parts, table_path_registry)
            child_key = _digest([key, child_parts, 0])
            _extract_record(
                value,
                table=child_table,
                key=child_key,
                root_key=root_key,
                parent_key=key,
                position=0,
                table_parts=child_parts,
                root_table=root_table,
                table_path_registry=table_path_registry,
                plans=plans,
            )
        else:
            row[field] = value
    plans.setdefault(table, []).append(row)


def _register_table_path(
    table_name: str,
    path: Sequence[str],
    registry: dict[str, tuple[str, ...]],
) -> None:
    canonical_path = tuple(path)
    previous_path = registry.setdefault(table_name, canonical_path)
    if previous_path != canonical_path:
        raise ValueError(
            "Distinct nested JSON paths map to the same MySQL child table name."
        )


def _build_plans(
    data: Any,
    table_name: str,
    unique_key_fields: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    records, context = _primary_records(data)
    _check_key_fields(records, unique_key_fields)
    return _build_plans_from_records(
        records, table_name, unique_key_fields, context
    )


def _build_plans_from_records(
    records: Sequence[Mapping[str, Any]],
    table_name: str,
    unique_key_fields: Sequence[str],
    context: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    plans: dict[str, list[dict[str, Any]]] = {}
    root_records: list[tuple[str, dict[str, Any]]] = []
    table_path_registry: dict[str, tuple[str, ...]] = {}
    for index, record in enumerate(records):
        record_copy = dict(record)
        key = _record_key(record_copy, unique_key_fields)
        root_records.append((key, record_copy))
        _extract_record(
            record_copy,
            table=table_name,
            key=key,
            root_key=key,
            parent_key=None,
            position=index,
            table_parts=[],
            root_table=table_name,
            table_path_registry=table_path_registry,
            plans=plans,
            source_context=context,
        )
    if not records:
        plans[table_name] = []
    return plans, root_records


def _table_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns = list(_INTERNAL_COLUMNS)
    if any("_etl_source_context" in row for row in rows):
        columns.append("_etl_source_context")
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return columns


def _create_json_tables(
    connection: Any,
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    root_table: str,
) -> dict[str, dict[str, str]]:
    schemas: dict[str, dict[str, str]] = {}
    cursor = connection.cursor()
    try:
        for table_name, rows in plans.items():
            columns = _table_columns(rows)
            safe_columns: dict[str, str] = {}
            for column in columns:
                if column in _INTERNAL_COLUMNS:
                    continue
                safe_column = _safe_column_name(column)
                previous = safe_columns.setdefault(safe_column, column)
                if previous != column:
                    raise ValueError(
                        "JSON fields collide after conversion to MySQL identifiers."
                    )
            schemas[table_name] = {}
            for column in columns:
                values = [row.get(column) for row in rows]
                schemas[table_name][column] = (
                    "VARCHAR(64)"
                    if column in {"_etl_record_key", "_etl_parent_key", "_etl_root_key"}
                    else "BIGINT"
                    if column == "_etl_position"
                    else _column_type(values)
                )
            definitions = [
                f"{quote_identifier('_etl_record_key', kind='column')} "
                "VARCHAR(64) NOT NULL PRIMARY KEY",
                f"{quote_identifier('_etl_parent_key', kind='column')} VARCHAR(64) NULL",
                f"{quote_identifier('_etl_root_key', kind='column')} VARCHAR(64) NOT NULL",
                f"{quote_identifier('_etl_position', kind='column')} BIGINT NULL",
            ]
            for column in columns:
                if column in _INTERNAL_COLUMNS:
                    continue
                definitions.append(
                    f"{quote_identifier(_safe_column_name(column), kind='column')} "
                    f"{schemas[table_name][column]} NULL"
                )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {quote_identifier(table_name, kind='table')} "
                f"({', '.join(definitions)})"
            )
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(_JSON_TABLE_REGISTRY, kind='table')} "
            "(root_table VARCHAR(64) NOT NULL, child_table VARCHAR(64) NOT NULL, "
            "PRIMARY KEY (root_table, child_table))"
        )
        for child_table in plans:
            if child_table != root_table:
                cursor.execute(
                    f"INSERT IGNORE INTO "
                    f"{quote_identifier(_JSON_TABLE_REGISTRY, kind='table')} "
                    "(root_table, child_table) VALUES (%s, %s)",
                    (root_table, child_table),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
    return schemas


def _execute_batch(
    cursor: Any,
    query: str,
    values: Sequence[Sequence[Any]],
    batch_size: int,
) -> None:
    for start in range(0, len(values), batch_size):
        cursor.executemany(query, values[start : start + batch_size])


def _write_rows(
    connection: Any,
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    schemas: Mapping[str, Mapping[str, str]],
    *,
    root_table: str,
    existing_root_keys: set[str],
    accepted_root_keys: set[str],
    duplicate_strategy: str,
    batch_size: int,
) -> None:
    cursor = connection.cursor()
    try:
        root_rows = plans.get(root_table, [])
        root_columns = list(schemas[root_table])
        user_columns = [column for column in root_columns if column != "_etl_record_key"]
        insert_columns = list(schemas[root_table])
        sql_columns = ", ".join(
            quote_identifier(_safe_column_name(column), kind="column")
            for column in insert_columns
        )
        placeholders = ", ".join(["%s"] * len(insert_columns))
        insert_query = (
            f"INSERT INTO {quote_identifier(root_table, kind='table')} "
            f"({sql_columns}) VALUES ({placeholders})"
        )
        insert_values = []
        for row in root_rows:
            key = row["_etl_record_key"]
            if key not in accepted_root_keys:
                continue
            values = [
                _mysql_value(row.get(column), schemas[root_table][column])
                for column in user_columns
            ]
            if key in existing_root_keys and duplicate_strategy == "update":
                assignments = ", ".join(
                    f"{quote_identifier(_safe_column_name(column), kind='column')} = %s"
                    for column in user_columns
                )
                cursor.execute(
                    f"UPDATE {quote_identifier(root_table, kind='table')} SET "
                    f"{assignments} WHERE `_etl_record_key` = %s",
                    (*values, key),
                )
            else:
                insert_values.append(
                    tuple(
                        _mysql_value(row.get(column), schemas[root_table][column])
                        for column in insert_columns
                    )
                )
        _execute_batch(cursor, insert_query, insert_values, batch_size)

        updated_keys = existing_root_keys & accepted_root_keys
        if duplicate_strategy == "update" and updated_keys:
            cursor.execute(
                f"SELECT child_table FROM "
                f"{quote_identifier(_JSON_TABLE_REGISTRY, kind='table')} "
                "WHERE root_table = %s",
                (root_table,),
            )
            child_tables = []
            for entry in cursor.fetchall():
                child_name = (
                    entry.get("child_table")
                    if isinstance(entry, Mapping)
                    else entry[0]
                )
                child_tables.append(validate_identifier(child_name, kind="table"))
            for child_table in child_tables:
                for root_key in updated_keys:
                    cursor.execute(
                        f"DELETE FROM {quote_identifier(child_table, kind='table')} "
                        "WHERE `_etl_root_key` = %s",
                        (root_key,),
                    )

        for table_name, rows in plans.items():
            if table_name == root_table:
                continue
            columns = list(schemas[table_name])
            selected = [
                row
                for row in rows
                if row["_etl_root_key"] in accepted_root_keys
            ]
            if not selected:
                continue
            sql_columns = ", ".join(
                quote_identifier(_safe_column_name(column), kind="column")
                for column in columns
            )
            placeholders = ", ".join(["%s"] * len(columns))
            query = (
                f"INSERT INTO {quote_identifier(table_name, kind='table')} "
                f"({sql_columns}) VALUES ({placeholders})"
            )
            values = [
                tuple(
                    _mysql_value(row.get(column), schemas[table_name][column])
                    for column in columns
                )
                for row in selected
            ]
            _execute_batch(cursor, query, values, batch_size)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def load_validated_json_data(
    data: Any,
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_fields: Sequence[str] | None = None,
    duplicate_strategy: str = "skip",
    dataset_name: str = "json_dataset",
    source_filename: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load JSON data that has already passed Stage 4 validation."""
    validate_identifier(table_name, kind="table")
    validate_identifier(tracking_table, kind="table")
    if table_name in {tracking_table, _JSON_TABLE_REGISTRY}:
        raise ValueError(f"{table_name!r} is reserved for pipeline metadata.")
    if tracking_table == _JSON_TABLE_REGISTRY:
        raise ValueError(f"{tracking_table!r} is reserved for JSON table metadata.")
    if duplicate_strategy not in {"skip", "update", "reject"}:
        raise ValueError("duplicate_strategy must be skip, update, or reject.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    keys = list(unique_key_fields or [])
    plans, root_records = _build_plans(data, table_name, keys)
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    record: dict[str, Any] = {
        "execution_id": str(uuid.uuid4()),
        "dataset_name": str(dataset_name),
        "source_filename": source_filename,
        "destination_table": f"json:{table_name}"[:64],
        "start_time": started,
        "end_time": None,
        "execution_status": "running",
        "processed_rows": len(root_records),
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
        schemas = _create_json_tables(
            connection, plans, root_table=table_name
        )
        cursor = connection.cursor()
        try:
            existing: set[str] = set()
            for key, _ in root_records:
                cursor.execute(
                    f"SELECT 1 FROM {quote_identifier(table_name, kind='table')} "
                    "WHERE `_etl_record_key` = %s LIMIT 1 FOR UPDATE",
                    (key,),
                )
                if cursor.fetchone() is not None:
                    existing.add(key)
        finally:
            cursor.close()

        selected: dict[str, dict[str, Any]] = {}
        repeated = 0
        for key, root in root_records:
            if key in selected:
                repeated += 1
                if duplicate_strategy == "reject":
                    raise ValueError(
                        "Duplicate record key encountered within this JSON dataset."
                    )
                if duplicate_strategy == "update":
                    selected[key] = root
                continue
            selected[key] = root
        if duplicate_strategy == "reject" and existing:
            raise ValueError("Duplicate record already exists in the destination.")

        selected_keys = set(selected)
        records, context = _primary_records(data)
        selected_records = list(selected.values())
        plans, _ = _build_plans_from_records(
            selected_records, table_name, keys, context
        )
        duplicate_existing = existing & selected_keys
        accepted = (
            selected_keys - duplicate_existing
            if duplicate_strategy == "skip"
            else selected_keys
        )
        for table_rows in plans.values():
            table_rows[:] = [
                row
                for row in table_rows
                if row["_etl_root_key"] in accepted
            ]
        _write_rows(
            connection,
            plans,
            schemas,
            root_table=table_name,
            existing_root_keys=existing,
            accepted_root_keys=accepted,
            duplicate_strategy=duplicate_strategy,
            batch_size=batch_size,
        )
        record["inserted_rows"] = len(accepted - existing)
        record["updated_rows"] = (
            len(accepted & existing) + repeated
            if duplicate_strategy == "update"
            else 0
        )
        record["skipped_rows"] = (
            len(duplicate_existing) + repeated
            if duplicate_strategy == "skip"
            else 0
        )
        record["execution_status"] = "success"
    except Exception as exc:
        if connection is not None:
            try:
                connection.rollback()
            except Exception as rollback_error:
                record["rollback_error"] = _redact_password(
                    rollback_error, connection_config
                )
        record["execution_status"] = "failed"
        record["failed_rows"] = max(
            0,
            record["processed_rows"]
            - record["inserted_rows"]
            - record["updated_rows"]
            - record["skipped_rows"],
        )
        record["error_message"] = _redact_password(exc, connection_config)
        record["error_type"] = type(exc).__name__
        if connection is not None and tracking_available:
            try:
                create_tracking_table(connection, tracking_table=tracking_table)
                record["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                _insert_tracking_record(connection, tracking_table, record)
                record["tracking_saved"] = True
            except Exception as tracking_error:
                record["tracking_saved"] = False
                record["tracking_error"] = _redact_password(
                    tracking_error, connection_config
                )
        else:
            record["tracking_saved"] = False
        if connection is not None:
            connection.close()
        return record

    record["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        _insert_tracking_record(connection, tracking_table, record)
        record["tracking_saved"] = True
    except Exception as exc:
        connection.rollback()
        record["execution_status"] = "failed"
        record["tracking_saved"] = False
        record["tracking_error"] = _redact_password(exc, connection_config)
        record["error_message"] = (
            "Data was committed, but execution tracking failed: "
            + _redact_password(exc, connection_config)
        )
    finally:
        connection.close()
    return record


def load_validated_json_dataset(
    dataset: Mapping[str, Any],
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_fields: Sequence[str] | None = None,
    duplicate_strategy: str = "skip",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load only successful Stage 4 JSON validation results."""
    validation = dataset.get("validation_result", dataset)
    if not isinstance(validation, Mapping):
        raise TypeError("JSON dataset must contain a validation-result mapping.")
    if validation.get("validation_status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("Only successfully validated JSON datasets can be loaded.")
    if "valid_data" not in validation:
        raise TypeError("Validated dataset has no valid_data.")
    report = validation.get("validation_report", {})
    source = dataset.get("source") or dataset.get("filename")
    dataset_name = (
        report.get("dataset_name")
        if isinstance(report, Mapping)
        else None
    ) or source or table_name
    return load_validated_json_data(
        validation["valid_data"],
        connection_config=connection_config,
        table_name=table_name,
        unique_key_fields=unique_key_fields,
        duplicate_strategy=duplicate_strategy,
        dataset_name=str(dataset_name),
        source_filename=str(source) if source else None,
        batch_size=batch_size,
        tracking_table=tracking_table,
        connect_factory=connect_factory,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRACKING_TABLE",
    "get_connection_config_from_environment",
    "get_pipeline_execution_history",
    "load_validated_json_data",
    "load_validated_json_dataset",
    "prepare_mysql_database",
    "suggest_unique_key_fields",
    "validate_identifier",
]
