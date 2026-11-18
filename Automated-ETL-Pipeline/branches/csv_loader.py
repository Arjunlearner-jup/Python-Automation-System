"""Transactional loading of validated CSV observations into MySQL."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


DEFAULT_TRACKING_TABLE = "etl_pipeline_executions"
DEFAULT_BATCH_SIZE = 1000
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "csv"
)


def validate_identifier(identifier: str, *, kind: str = "SQL") -> str:
    """Validate an unquoted MySQL identifier before embedding it in SQL."""
    if not isinstance(identifier, str) or not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(
            f"Invalid {kind} identifier {identifier!r}; use 1-64 letters, "
            "digits, or underscores, beginning with a letter or underscore."
        )
    return identifier


def quote_identifier(identifier: str, *, kind: str = "SQL") -> str:
    """Validate and quote a MySQL identifier."""
    return f"`{validate_identifier(identifier, kind=kind)}`"


def _mysql_connect(**kwargs: Any) -> Any:
    try:
        import mysql.connector
    except ImportError as exc:
        raise RuntimeError(
            "MySQL support requires mysql-connector-python. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc
    return mysql.connector.connect(**kwargs)


def _connection_factory(
    connect_factory: Callable[..., Any] | None,
) -> Callable[..., Any]:
    return connect_factory or _mysql_connect


def _redact_password(message: Any, config: Mapping[str, Any]) -> str:
    """Keep connection secrets out of errors returned to the caller."""
    text = str(message)
    password = str(config.get("password") or "")
    return text.replace(password, "[redacted]") if password else text


def _connection_options(config: Mapping[str, Any]) -> dict[str, Any]:
    host = str(config.get("host") or "").strip()
    username = str(config.get("username") or "").strip()
    if not host or not username:
        raise ValueError("MySQL host and username are required.")
    try:
        port = int(config.get("port", 3306))
    except (TypeError, ValueError) as exc:
        raise ValueError("MySQL port must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("MySQL port must be between 1 and 65535.")
    return {
        "host": host,
        "port": port,
        "user": username,
        "password": str(config.get("password") or ""),
    }


def prepare_mysql_database(
    config: Mapping[str, Any],
    *,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Authenticate without a database, then create or reuse the requested DB."""
    database = validate_identifier(str(config.get("database") or ""), kind="database")
    options = _connection_options(config)
    factory = _connection_factory(connect_factory)
    server_connection = None
    try:
        server_connection = factory(**options)
        cursor = server_connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database, kind='database')}")
        server_connection.commit()
        cursor.close()
    except Exception:
        if server_connection is not None:
            try:
                server_connection.rollback()
            finally:
                server_connection.close()
        raise
    else:
        server_connection.close()

    database_connection = factory(**options, database=database)
    return {
        "connection": database_connection,
        "database": database,
        "connection_status": "connected",
        "database_status": "created_or_reused",
    }


def _normalise_column_name(column: Any) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column).strip())
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"column_{name}"
    if len(name) > 64:
        name = name[:64]
    return validate_identifier(name, kind="column")


def _column_mapping(columns: Sequence[Any]) -> dict[str, str]:
    mapping = {str(column): _normalise_column_name(column) for column in columns}
    destinations = list(mapping.values())
    if len(destinations) != len(set(destinations)):
        raise ValueError(
            "CSV column names collide after conversion to safe MySQL identifiers."
        )
    return mapping


def _sql_type(series: pd.Series, *, is_key: bool) -> str:
    if is_key:
        return "VARCHAR(512)"
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
        lambda value: isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
    ).all():
        return "DOUBLE"
    return "LONGTEXT"


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if getattr(result, "ndim", 0) == 0 else False


def _to_mysql_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (datetime, date)):
        return value
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def create_tracking_table(
    connection: Any,
    *,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
) -> None:
    """Create the append-only pipeline execution history table if absent."""
    table_sql = quote_identifier(tracking_table, kind="table")
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {table_sql} (
                execution_id CHAR(36) NOT NULL PRIMARY KEY,
                dataset_name VARCHAR(512) NOT NULL,
                source_filename VARCHAR(1024) NULL,
                destination_table VARCHAR(64) NOT NULL,
                start_time DATETIME(6) NOT NULL,
                end_time DATETIME(6) NULL,
                execution_status VARCHAR(32) NOT NULL,
                processed_rows BIGINT NOT NULL DEFAULT 0,
                inserted_rows BIGINT NOT NULL DEFAULT 0,
                updated_rows BIGINT NOT NULL DEFAULT 0,
                skipped_rows BIGINT NOT NULL DEFAULT 0,
                failed_rows BIGINT NOT NULL DEFAULT 0,
                error_message LONGTEXT NULL
            )"""
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()


def create_destination_table(
    connection: Any,
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    unique_key_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a destination table if missing, based on the DataFrame schema."""
    if len(dataframe.columns) == 0:
        raise ValueError("Cannot create a MySQL table from a DataFrame with no columns.")
    if not dataframe.columns.is_unique:
        raise ValueError("CSV DataFrame has duplicate column names.")
    table_sql = quote_identifier(table_name, kind="table")
    mapping = _column_mapping(list(dataframe.columns))
    keys = list(unique_key_columns or [])
    unknown_keys = set(keys).difference(mapping)
    if unknown_keys:
        raise KeyError(f"Unknown unique key column(s): {sorted(unknown_keys)}")
    if len(set(keys)) != len(keys):
        raise ValueError("Unique key columns must not contain duplicates.")
    sql_keys = [mapping[key] for key in keys]
    declarations = [
        f"{quote_identifier(mapping[str(column)], kind='column')} "
        f"{_sql_type(dataframe[column], is_key=str(column) in keys)} NULL"
        for column in dataframe.columns
    ]
    if sql_keys:
        key_sql = ", ".join(quote_identifier(key, kind="column") for key in sql_keys)
        declarations.append(
            f"UNIQUE KEY {quote_identifier('uq_' + table_name[:48], kind='index')} "
            f"({key_sql})"
        )
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {table_sql} ({', '.join(declarations)})"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
    return {"table_name": table_name, "column_mapping": mapping, "unique_key_columns": keys}


def _insert_tracking_record(connection: Any, table_name: str, record: Mapping[str, Any]) -> None:
    sql = f"""INSERT INTO {quote_identifier(table_name, kind='table')}
        (execution_id, dataset_name, source_filename, destination_table,
         start_time, end_time, execution_status, processed_rows, inserted_rows,
         updated_rows, skipped_rows, failed_rows, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = (
        record["execution_id"],
        record["dataset_name"],
        record["source_filename"],
        record["destination_table"],
        record["start_time"],
        record["end_time"],
        record["execution_status"],
        record["processed_rows"],
        record["inserted_rows"],
        record["updated_rows"],
        record["skipped_rows"],
        record["failed_rows"],
        record["error_message"],
    )
    cursor = connection.cursor()
    try:
        cursor.execute(sql, values)
        connection.commit()
    finally:
        cursor.close()


def _key_value(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    values = tuple(_to_mysql_value(row[key]) for key in keys)
    if any(_is_missing(value) for value in values):
        raise ValueError("Unique key columns cannot contain null or empty values.")
    return values


def _load_batch(
    connection: Any,
    dataframe: pd.DataFrame,
    table_name: str,
    column_mapping: Mapping[str, str],
    key_columns: Sequence[str],
    strategy: str,
    batch_size: int,
) -> dict[str, int]:
    table_sql = quote_identifier(table_name, kind="table")
    source_columns = [str(column) for column in dataframe.columns]
    sql_columns = [column_mapping[column] for column in source_columns]
    quoted_columns = ", ".join(quote_identifier(column, kind="column") for column in sql_columns)
    placeholders = ", ".join(["%s"] * len(sql_columns))
    insert_sql = f"INSERT INTO {table_sql} ({quoted_columns}) VALUES ({placeholders})"
    update_columns = [column for column in source_columns if column not in key_columns]
    sql_keys = [column_mapping[column] for column in key_columns]
    update_sql = ""
    if update_columns:
        assignments = ", ".join(
            f"{quote_identifier(column_mapping[column], kind='column')} = %s"
            for column in update_columns
        )
        predicates = " AND ".join(
            f"{quote_identifier(column_mapping[column], kind='column')} = %s"
            for column in key_columns
        )
        update_sql = f"UPDATE {table_sql} SET {assignments} WHERE {predicates}"
    predicates = " AND ".join(
        f"{quote_identifier(column_mapping[column], kind='column')} = %s"
        for column in key_columns
    )
    select_sql = f"SELECT 1 FROM {table_sql} WHERE {predicates} LIMIT 1 FOR UPDATE"
    stats = {"processed_rows": len(dataframe.index), "inserted_rows": 0, "updated_rows": 0, "skipped_rows": 0}
    seen_in_run: dict[tuple[Any, ...], tuple[str, int | None]] = {}
    cursor = connection.cursor()
    try:
        for start in range(0, len(dataframe.index), batch_size):
            part = dataframe.iloc[start : start + batch_size]
            inserts: list[tuple[Any, ...]] = []
            updates: list[tuple[Any, ...]] = []
            for row in part.to_dict(orient="records"):
                key = _key_value(row, key_columns)
                values = tuple(_to_mysql_value(row[column]) for column in source_columns)
                if key in seen_in_run:
                    if strategy == "reject":
                        raise ValueError(f"Duplicate key encountered within this dataset: {key!r}.")
                    if strategy == "skip":
                        stats["skipped_rows"] += 1
                    else:
                        if update_columns:
                            state, insert_index = seen_in_run[key]
                            if state == "pending":
                                inserts[insert_index] = values
                            else:
                                updates.append(
                                    tuple(_to_mysql_value(row[column]) for column in update_columns)
                                    + key
                                )
                        stats["updated_rows"] += 1
                    continue

                cursor.execute(select_sql, key)
                exists = cursor.fetchone() is not None
                if exists:
                    if strategy == "reject":
                        raise ValueError(f"Duplicate key already exists in destination: {key!r}.")
                    if strategy == "skip":
                        stats["skipped_rows"] += 1
                    else:
                        if update_columns:
                            updates.append(
                                tuple(_to_mysql_value(row[column]) for column in update_columns)
                                + key
                            )
                        stats["updated_rows"] += 1
                    seen_in_run[key] = ("existing", None)
                    continue

                inserts.append(values)
                seen_in_run[key] = ("pending", len(inserts) - 1)

            if inserts:
                cursor.executemany(insert_sql, inserts)
                stats["inserted_rows"] += len(inserts)
                for key, (state, _) in list(seen_in_run.items()):
                    if state == "pending":
                        seen_in_run[key] = ("existing", None)
            if updates:
                if not update_sql:
                    raise ValueError(
                        "Update existing records requires at least one non-key column."
                    )
                cursor.executemany(update_sql, updates)
        connection.commit()
        return stats
    finally:
        cursor.close()


def load_validated_csv_dataframe(
    dataframe: pd.DataFrame,
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_columns: Sequence[str],
    duplicate_strategy: str = "skip",
    dataset_name: str = "csv_dataset",
    source_filename: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load Stage 4 validated rows and persist a complete execution record.

    Duplicate strategies are ``skip``, ``update``, and ``reject``. Key fields
    are required so reprocessing the same dataset cannot silently duplicate rows.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Only a pandas DataFrame can be loaded.")
    if duplicate_strategy not in {"skip", "update", "reject"}:
        raise ValueError("duplicate_strategy must be skip, update, or reject.")
    if not unique_key_columns:
        raise ValueError("Select at least one unique key column.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    validate_identifier(table_name, kind="table")
    validate_identifier(tracking_table, kind="table")
    keys = list(unique_key_columns)
    _column_mapping(list(dataframe.columns))
    if not dataframe.columns.is_unique:
        raise ValueError("CSV DataFrame has duplicate column names.")
    if any(not isinstance(column, str) for column in dataframe.columns):
        raise ValueError("CSV DataFrame column names must be strings.")
    if not dataframe.columns.size:
        raise ValueError("Cannot load a DataFrame with no columns.")
    if not set(keys).issubset(set(map(str, dataframe.columns))):
        raise KeyError(
            f"Unknown unique key column(s): {sorted(set(keys) - set(map(str, dataframe.columns)))}"
        )

    started = datetime.now(timezone.utc).replace(tzinfo=None)
    record: dict[str, Any] = {
        "execution_id": str(uuid.uuid4()),
        "dataset_name": str(dataset_name),
        "source_filename": source_filename,
        "destination_table": table_name,
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
    db_connection = None
    tracker_available = False
    try:
        prepared = prepare_mysql_database(
            connection_config, connect_factory=connect_factory
        )
        db_connection = prepared["connection"]
        create_tracking_table(db_connection, tracking_table=tracking_table)
        tracker_available = True
        schema = create_destination_table(
            db_connection,
            dataframe,
            table_name,
            unique_key_columns=keys,
        )
        stats = _load_batch(
            db_connection,
            dataframe,
            table_name,
            schema["column_mapping"],
            keys,
            duplicate_strategy,
            batch_size,
        )
        record.update(stats)
        record["execution_status"] = "success"
    except Exception as exc:
        safe_message = _redact_password(exc, connection_config)
        if db_connection is not None:
            try:
                db_connection.rollback()
            except Exception as rollback_error:
                record["rollback_error"] = (
                    f"{type(rollback_error).__name__}: "
                    f"{_redact_password(rollback_error, connection_config)}"
                )
        record["execution_status"] = "failed"
        record["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
        record["failed_rows"] = max(
            0,
            record["processed_rows"]
            - record["inserted_rows"]
            - record["updated_rows"]
            - record["skipped_rows"],
        )
        record["error_message"] = safe_message
        record["error_type"] = type(exc).__name__
        if db_connection is not None and tracker_available:
            try:
                create_tracking_table(db_connection, tracking_table=tracking_table)
                _insert_tracking_record(db_connection, tracking_table, record)
                record["tracking_saved"] = True
            except Exception as tracking_error:
                record["tracking_saved"] = False
                record["tracking_error"] = (
                    f"{type(tracking_error).__name__}: "
                    f"{_redact_password(tracking_error, connection_config)}"
                )
        else:
            record["tracking_saved"] = False
        if db_connection is not None:
            db_connection.close()
        return record

    record["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        _insert_tracking_record(db_connection, tracking_table, record)
        record["tracking_saved"] = True
    except Exception as exc:
        db_connection.rollback()
        record["tracking_saved"] = False
        record["tracking_error"] = (
            f"{type(exc).__name__}: {_redact_password(exc, connection_config)}"
        )
        record["execution_status"] = "failed"
        record["error_message"] = (
            "Data was committed, but execution tracking failed: "
            + _redact_password(exc, connection_config)
        )
    finally:
        db_connection.close()
    return record


def load_validated_csv_dataset(
    dataset: Mapping[str, Any],
    *,
    connection_config: Mapping[str, Any],
    table_name: str,
    unique_key_columns: Sequence[str],
    duplicate_strategy: str = "skip",
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load only Stage 4 valid records from a validation result mapping."""
    validation = dataset.get("validation_result", dataset)
    if not isinstance(validation, Mapping):
        raise TypeError("CSV dataset must contain a validation-result mapping.")
    if validation.get("validation_status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("Only successfully validated CSV datasets can be loaded.")
    dataframe = validation.get("valid_dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Validated dataset has no valid_dataframe.")
    helper_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith("_validation_status")
        or str(column).startswith("_validation_errors")
    ]
    if helper_columns:
        dataframe = dataframe.drop(columns=helper_columns).copy(deep=True)
    report = validation.get("validation_report", {})
    cleaned = dataset.get("cleaning_result", {})
    cleaning_report = cleaned.get("report", {}) if isinstance(cleaned, Mapping) else {}
    dataset_name = str(
        (report.get("dataset_name") if isinstance(report, Mapping) else None)
        or (cleaning_report.get("dataset_name") if isinstance(cleaning_report, Mapping) else None)
        or dataset.get("source")
        or table_name
    )
    source_filename = (
        dataset.get("source")
        or dataset.get("filename")
        or (cleaning_report.get("source") if isinstance(cleaning_report, Mapping) else None)
    )
    return load_validated_csv_dataframe(
        dataframe,
        connection_config=connection_config,
        table_name=table_name,
        unique_key_columns=unique_key_columns,
        duplicate_strategy=duplicate_strategy,
        dataset_name=dataset_name,
        source_filename=str(source_filename) if source_filename else None,
        batch_size=batch_size,
        tracking_table=tracking_table,
        connect_factory=connect_factory,
    )


def get_pipeline_execution_history(
    connection_config: Mapping[str, Any],
    *,
    limit: int = 100,
    tracking_table: str = DEFAULT_TRACKING_TABLE,
    connect_factory: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Return recent loading executions without revealing connection secrets."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")
    validate_identifier(tracking_table, kind="table")
    prepared = prepare_mysql_database(
        connection_config, connect_factory=connect_factory
    )
    connection = prepared["connection"]
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            f"SELECT execution_id, dataset_name, source_filename, destination_table, "
            f"start_time, end_time, execution_status, processed_rows, inserted_rows, "
            f"updated_rows, skipped_rows, failed_rows, error_message FROM "
            f"{quote_identifier(tracking_table, kind='table')} "
            "ORDER BY start_time DESC LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        cursor.close()
        connection.close()


def get_connection_config_from_environment() -> dict[str, Any]:
    """Read optional connection defaults from environment variables."""
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "username": os.getenv("MYSQL_USER", ""),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", ""),
    }
