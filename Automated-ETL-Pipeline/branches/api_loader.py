"""Transactional loading of validated API observations and request metadata."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types

from .csv_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRACKING_TABLE,
    _insert_tracking_record,
    _redact_password,
    _to_mysql_value,
    create_tracking_table,
    get_connection_config_from_environment,
    get_pipeline_execution_history,
    prepare_mysql_database,
    quote_identifier,
    validate_identifier,
)


_INTERNAL_COLUMNS = {"_etl_observation_key", "_etl_ingestion_id"}
_METADATA_SUFFIX = "_ingestions"
def _canonical_json(value: Any) -> str:
    def safe(item: Any) -> Any:
        if item is None:
            return None
        if isinstance(item, Mapping):
            return {str(key): safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [safe(child) for child in item]
        if isinstance(item, (datetime, date, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            try:
                item = item.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(item, float) and not pd.notna(item):
            return None
        return item

    return json.dumps(
        safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_column_name(column: Any) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column).strip())
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"column_{name}"
    return validate_identifier(name[:64], kind="column")


def get_api_metadata_table_name(observation_table: str) -> str:
    """Derive a validated related metadata table name from the selected table."""
    validate_identifier(observation_table, kind="table")
    suffix = _METADATA_SUFFIX
    available = 64 - len(suffix)
    return validate_identifier(observation_table[:available] + suffix, kind="table")


def suggest_api_unique_key_columns(dataframe: pd.DataFrame) -> list[str]:
    """Suggest dimension columns that are jointly unique in these observations."""
    if dataframe.empty:
        return []
    columns = {str(column): column for column in dataframe.columns}
    preferred_order = [
        "INDICATOR",
        "REF_AREA",
        "TIME_PERIOD",
        "DATABASE_ID",
        "FREQ",
        "SEX",
        "AGE",
        "UNIT_MEASURE",
        "URBANISATION",
    ]
    ordered = [name for name in preferred_order if name in columns]
    ordered.extend(
        str(column)
        for column in dataframe.columns
        if str(column) not in ordered and str(column).upper() in {
            "INDICATOR",
            "REF_AREA",
            "TIME_PERIOD",
            "DATABASE_ID",
        }
    )
    if not ordered:
        return []
    for size in range(1, len(ordered) + 1):
        candidate = ordered[:size]
        values = dataframe[[columns[name] for name in candidate]]
        if not values.isna().any(axis=None) and not values.duplicated().any():
            return candidate
    return []


def _observation_key(
    row: Mapping[str, Any],
    original_columns: Sequence[str],
    key_columns: Sequence[str],
) -> str:
    selected = list(key_columns) if key_columns else list(original_columns)
    values = {column: row.get(column) for column in selected}
    return _digest(values)


def _sql_type(series: pd.Series) -> str:
    if pandas_types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if pandas_types.is_integer_dtype(series.dtype):
        return "BIGINT"
    if pandas_types.is_float_dtype(series.dtype):
        return "DOUBLE"
    if pandas_types.is_datetime64_any_dtype(series.dtype):
        return "DATETIME(6)"
    values = series.dropna()
    if not values.empty and values.map(
        lambda value: isinstance(value, (date, datetime, pd.Timestamp))
    ).all():
        return "DATETIME(6)"
    if not values.empty and values.map(
        lambda value: isinstance(value, (Mapping, list, tuple))
    ).any():
        return "JSON"
    if not values.empty and values.map(
        lambda value: isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
    ).all():
        return "DOUBLE"
    return "LONGTEXT"


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    return _canonical_json(dict(metadata))


def _metadata_row(
    metadata: Mapping[str, Any],
    *,
    dataset_name: str,
    source_endpoint: str | None,
    retrieved_count: int,
    ingestion_id: str,
) -> dict[str, Any]:
    pagination = metadata.get("pagination") or {}
    if not isinstance(pagination, Mapping):
        pagination = {"raw": pagination}
    return {
        "ingestion_id": ingestion_id,
        "dataset_name": dataset_name,
        "provider": metadata.get("provider"),
        "endpoint": source_endpoint or metadata.get("api_endpoint"),
        "database_id": metadata.get("database_id"),
        "indicator_id": metadata.get("indicator_id"),
        "reference_area": metadata.get("country_reference_area"),
        "request_timestamp": metadata.get("request_timestamp"),
        "http_status_code": metadata.get("http_status_code"),
        "api_response_record_count": metadata.get("api_response_record_count"),
        "records_retrieved": metadata.get("records_retrieved", retrieved_count),
        "pages_fetched": pagination.get("pages_fetched"),
        "pagination_complete": pagination.get("complete"),
        "ingestion_status": metadata.get("status"),
        "pagination_json": _metadata_json(pagination),
        "metadata_json": _metadata_json(metadata),
    }


def _create_api_tables(
    connection: Any,
    dataframe: pd.DataFrame,
    observation_table: str,
    metadata_table: str,
    *,
    key_columns: Sequence[str],
) -> dict[str, str]:
    if not dataframe.columns.is_unique:
        raise ValueError("API DataFrame has duplicate column names.")
    if any(not isinstance(column, str) for column in dataframe.columns):
        raise ValueError("API DataFrame column names must be strings.")
    if dataframe.empty and not len(dataframe.columns):
        raise ValueError("Cannot create API observation tables without columns.")
    mapping = {
        str(column): _normalise_column_name(column)
        for column in dataframe.columns
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("API columns collide after conversion to SQL identifiers.")
    if set(mapping.values()).intersection(_INTERNAL_COLUMNS):
        raise ValueError("API columns use reserved loader field names.")
    unknown_keys = set(key_columns).difference(mapping)
    if unknown_keys:
        raise KeyError(f"Unknown API key columns: {sorted(unknown_keys)}")

    definitions = [
        "`_etl_observation_key` CHAR(64) NOT NULL PRIMARY KEY",
        "`_etl_ingestion_id` CHAR(64) NOT NULL",
    ]
    for column in dataframe.columns:
        definitions.append(
            f"{quote_identifier(mapping[str(column)], kind='column')} "
            f"{_sql_type(dataframe[column])} NULL"
        )
    definitions.append(
        "KEY `idx_etl_ingestion_id` (`_etl_ingestion_id`)"
    )
    definitions.append(
        f"CONSTRAINT `fk_{hashlib.sha1(observation_table.encode()).hexdigest()[:12]}` "
        f"FOREIGN KEY (`_etl_ingestion_id`) REFERENCES "
        f"{quote_identifier(metadata_table, kind='table')} (`ingestion_id`)"
    )
    metadata_definitions = [
        "`ingestion_id` CHAR(64) NOT NULL PRIMARY KEY",
        "`dataset_name` VARCHAR(512) NOT NULL",
        "`provider` VARCHAR(512) NULL",
        "`endpoint` LONGTEXT NULL",
        "`database_id` VARCHAR(512) NULL",
        "`indicator_id` VARCHAR(512) NULL",
        "`reference_area` VARCHAR(512) NULL",
        "`request_timestamp` VARCHAR(128) NULL",
        "`http_status_code` INT NULL",
        "`api_response_record_count` BIGINT NULL",
        "`records_retrieved` BIGINT NULL",
        "`pages_fetched` BIGINT NULL",
        "`pagination_complete` BOOLEAN NULL",
        "`ingestion_status` VARCHAR(64) NULL",
        "`pagination_json` JSON NOT NULL",
        "`metadata_json` JSON NOT NULL",
    ]

    cursor = connection.cursor()
    try:
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(metadata_table, kind='table')} "
            f"({', '.join(metadata_definitions)})"
        )
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(observation_table, kind='table')} "
            f"({', '.join(definitions)})"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
    return mapping


def _insert_metadata(
    connection: Any,
    metadata_table: str,
    row: Mapping[str, Any],
) -> None:
    columns = list(row)
    sql_columns = ", ".join(
        quote_identifier(column, kind="column") for column in columns
    )
    placeholders = ", ".join(["%s"] * len(columns))
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {quote_identifier(metadata_table, kind='table')} "
            f"({sql_columns}) VALUES ({placeholders}) "
            "ON DUPLICATE KEY UPDATE ingestion_id = VALUES(ingestion_id)",
            tuple(row[column] for column in columns),
        )
    finally:
        cursor.close()


def _load_observations(
    connection: Any,
    dataframe: pd.DataFrame,
    observation_table: str,
    column_mapping: Mapping[str, str],
    ingestion_id: str,
    key_columns: Sequence[str],
    duplicate_strategy: str,
    batch_size: int,
) -> dict[str, int]:
    original_columns = list(map(str, dataframe.columns))
    selected_rows: dict[str, dict[str, Any]] = {}
    duplicates_in_input = 0
    for values in dataframe.to_dict(orient="records"):
        row = {str(key): value for key, value in values.items()}
        key = _observation_key(row, original_columns, key_columns)
        if key in selected_rows:
            duplicates_in_input += 1
            if duplicate_strategy == "reject":
                raise ValueError("Duplicate observation key found in API data.")
            if duplicate_strategy == "update":
                selected_rows[key] = row
            continue
        selected_rows[key] = row

    cursor = connection.cursor()
    existing: set[str] = set()
    try:
        for observation_key in selected_rows:
            cursor.execute(
                f"SELECT 1 FROM {quote_identifier(observation_table, kind='table')} "
                "WHERE `_etl_observation_key` = %s LIMIT 1 FOR UPDATE",
                (observation_key,),
            )
            if cursor.fetchone() is not None:
                existing.add(observation_key)
    finally:
        cursor.close()
    if duplicate_strategy == "reject" and existing:
        raise ValueError("Duplicate observation already exists in destination.")

    accepted = (
        set(selected_rows) - existing
        if duplicate_strategy == "skip"
        else set(selected_rows)
    )
    insert_columns = ["_etl_observation_key", "_etl_ingestion_id"] + [
        column_mapping[column] for column in original_columns
    ]
    query = (
        f"INSERT INTO {quote_identifier(observation_table, kind='table')} "
        f"({', '.join(quote_identifier(column, kind='column') for column in insert_columns)}) "
        f"VALUES ({', '.join(['%s'] * len(insert_columns))})"
    )
    updated = 0
    inserts = []
    for observation_key, row in selected_rows.items():
        if observation_key not in accepted:
            continue
        values = [_to_mysql_value(row.get(column)) for column in original_columns]
        if observation_key in existing and duplicate_strategy == "update":
            assignments = ", ".join(
                f"{quote_identifier(column_mapping[column], kind='column')} = %s"
                for column in original_columns
            )
            cursor = connection.cursor()
            try:
                cursor.execute(
                    f"UPDATE {quote_identifier(observation_table, kind='table')} "
                    f"SET `_etl_ingestion_id` = %s, {assignments} "
                    "WHERE `_etl_observation_key` = %s",
                    (ingestion_id, *values, observation_key),
                )
            finally:
                cursor.close()
            updated += 1
        else:
            inserts.append((observation_key, ingestion_id, *values))
    cursor = connection.cursor()
    try:
        for start in range(0, len(inserts), batch_size):
            cursor.executemany(query, inserts[start : start + batch_size])
    finally:
        cursor.close()
    connection.commit()
    return {
        "processed_rows": len(dataframe.index),
        "inserted_rows": len(inserts),
        "updated_rows": updated + (
            duplicates_in_input if duplicate_strategy == "update" else 0
        ),
        "skipped_rows": len(existing - accepted) + (
            duplicates_in_input if duplicate_strategy == "skip" else 0
        ),
    }


def load_validated_api_dataframe(
    dataframe: pd.DataFrame,
    *,
    api_metadata: Mapping[str, Any],
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_columns: Sequence[str] | None = None,
    duplicate_strategy: str = "skip",
    dataset_name: str = "api_dataset",
    source_endpoint: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Persist validated API observations and metadata; performs no API requests."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Only a pandas DataFrame can be loaded.")
    if not isinstance(api_metadata, Mapping):
        raise TypeError("API metadata must be a mapping.")
    validate_identifier(table_name, kind="table")
    validate_identifier(tracking_table, kind="table")
    metadata_table = get_api_metadata_table_name(table_name)
    if table_name in {tracking_table, metadata_table}:
        raise ValueError("Observation table conflicts with pipeline metadata.")
    if tracking_table == metadata_table:
        raise ValueError("Tracking and API metadata tables must be distinct.")
    if duplicate_strategy not in {"skip", "update", "reject"}:
        raise ValueError("duplicate_strategy must be skip, update, or reject.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not dataframe.columns.is_unique:
        raise ValueError("API DataFrame has duplicate column names.")
    if any(not isinstance(column, str) for column in dataframe.columns):
        raise ValueError("API DataFrame column names must be strings.")
    keys = list(unique_key_columns or [])
    if len(set(keys)) != len(keys):
        raise ValueError("Unique key columns must not contain duplicates.")
    missing_keys = set(keys).difference(map(str, dataframe.columns))
    if missing_keys:
        raise KeyError(f"Unknown unique key columns: {sorted(missing_keys)}")
    if keys and dataframe[keys].isna().any(axis=None):
        raise ValueError("Unique key columns cannot contain null values.")
    source_endpoint = (
        source_endpoint
        or str(api_metadata.get("api_endpoint") or "")
        or None
    )
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    record: dict[str, Any] = {
        "execution_id": str(uuid.uuid4()),
        "dataset_name": str(dataset_name),
        "source_filename": source_endpoint,
        "destination_table": f"api:{table_name}"[:64],
        "start_time": started,
        "end_time": None,
        "execution_status": "running",
        "processed_rows": len(dataframe.index),
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "error_message": None,
    }
    connection = None
    tracking_available = False
    ingestion_id = _digest(
        {
            "metadata": dict(api_metadata),
            "dataset_name": dataset_name,
            "endpoint": source_endpoint,
        }
    )
    try:
        prepared = prepare_mysql_database(
            connection_config, connect_factory=connect_factory
        )
        connection = prepared["connection"]
        create_tracking_table(connection, tracking_table=tracking_table)
        tracking_available = True
        mapping = _create_api_tables(
            connection,
            dataframe,
            table_name,
            metadata_table,
            key_columns=keys,
        )
        metadata_row = _metadata_row(
            api_metadata,
            dataset_name=str(dataset_name),
            source_endpoint=source_endpoint,
            retrieved_count=len(dataframe.index),
            ingestion_id=ingestion_id,
        )
        _insert_metadata(connection, metadata_table, metadata_row)
        stats = _load_observations(
            connection,
            dataframe,
            table_name,
            mapping,
            ingestion_id,
            keys,
            duplicate_strategy,
            batch_size,
        )
        record.update(stats)
        record["execution_status"] = "success"
    except Exception as error:
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
        record["error_message"] = _redact_password(error, connection_config)
        record["error_type"] = type(error).__name__
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
    except Exception as error:
        connection.rollback()
        record["execution_status"] = "failed"
        record["tracking_saved"] = False
        record["tracking_error"] = _redact_password(error, connection_config)
        record["error_message"] = (
            "API data was committed, but execution tracking failed: "
            + _redact_password(error, connection_config)
        )
    finally:
        connection.close()
    record["metadata_table"] = metadata_table
    record["ingestion_id"] = ingestion_id
    record["api_metadata"] = dict(api_metadata)
    return record


def load_validated_api_dataset(
    dataset: Mapping[str, Any],
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_columns: Sequence[str] | None = None,
    duplicate_strategy: str = "skip",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load only Stage 4 validated API DataFrames and their Stage 1 metadata."""
    validation = dataset.get("validation_result", dataset)
    if not isinstance(validation, Mapping):
        raise TypeError("API dataset must contain a validation-result mapping.")
    if validation.get("validation_status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("Only successfully validated API datasets can be loaded.")
    dataframe = validation.get("valid_dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Validated API dataset has no valid_dataframe.")
    api_metadata = (
        dataset.get("api_metadata")
        or validation.get("api_metadata")
        or validation.get("validation_report", {}).get("api_metadata")
        or {}
    )
    if not isinstance(api_metadata, Mapping):
        raise TypeError("Validated API metadata must be a mapping.")
    report = validation.get("validation_report", {})
    dataset_name = (
        report.get("dataset_name")
        if isinstance(report, Mapping)
        else None
    ) or dataset.get("source") or table_name
    endpoint = dataset.get("source_endpoint") or api_metadata.get("api_endpoint")
    return load_validated_api_dataframe(
        dataframe,
        api_metadata=api_metadata,
        connection_config=connection_config,
        table_name=table_name,
        unique_key_columns=unique_key_columns,
        duplicate_strategy=duplicate_strategy,
        dataset_name=str(dataset_name),
        source_endpoint=str(endpoint) if endpoint else None,
        batch_size=batch_size,
        tracking_table=tracking_table,
        connect_factory=connect_factory,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRACKING_TABLE",
    "get_api_metadata_table_name",
    "get_connection_config_from_environment",
    "get_pipeline_execution_history",
    "load_validated_api_dataframe",
    "load_validated_api_dataset",
    "prepare_mysql_database",
    "suggest_api_unique_key_columns",
    "validate_identifier",
]
