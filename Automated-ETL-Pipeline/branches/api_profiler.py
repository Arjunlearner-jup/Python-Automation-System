"""Stage 2 profiling functions for already-ingested API DataFrames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return False
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if getattr(missing, "ndim", 0) == 0 else False


def profile_dimensions(
    df: pd.DataFrame,
    api_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report DataFrame dimensions and available API dataset identifiers."""
    rows, columns = df.shape
    dimensions: dict[str, Any] = {
        "total_rows": rows,
        "total_columns": columns,
        "dimensions": (rows, columns),
        "column_names": list(df.columns),
        "number_of_records": len(df.index),
    }
    if api_metadata is not None:
        for key in (
            "provider",
            "api_name",
            "database_id",
            "indicator_id",
            "country_reference_area",
            "api_response_record_count",
            "records_retrieved",
        ):
            dimensions[key] = api_metadata.get(key)
    return dimensions


def profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Count null, empty, and whitespace-only values without modifying data."""
    columns: dict[str, dict[str, int | float]] = {}
    total_missing = 0
    for column in df.columns:
        values = df[column]
        missing_count = sum(_is_missing(value) for value in values)
        total_values = len(values)
        columns[str(column)] = {
            "total_values": total_values,
            "missing_count": missing_count,
            "missing_percentage": (
                missing_count / total_values * 100 if total_values else 0.0
            ),
            "non_missing_count": total_values - missing_count,
        }
        total_missing += missing_count

    total_values = len(df.index) * len(df.columns)
    return {
        "columns": columns,
        "total_missing_values": total_missing,
        "total_values": total_values,
        "overall_missing_percentage": (
            total_missing / total_values * 100 if total_values else 0.0
        ),
    }


def _freeze(value: Any) -> Any:
    """Convert nested observations into hashable representations."""
    if _is_missing(value):
        return ("__missing__",)
    if isinstance(value, Mapping):
        pairs = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return ("__dict__", tuple(sorted(pairs, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("__list__", tuple(_freeze(item) for item in value))
    if isinstance(value, set):
        return ("__set__", tuple(sorted((_freeze(item) for item in value), key=repr)))
    try:
        hash(value)
    except TypeError:
        return ("__repr__", repr(value))
    return value


def profile_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Report exact full-row duplicates, including observations with nested data."""
    seen: set[Any] = set()
    duplicate_positions: list[int] = []
    for position, row in enumerate(df.itertuples(index=False, name=None)):
        frozen_row = tuple(_freeze(value) for value in row)
        if frozen_row in seen:
            duplicate_positions.append(position)
        else:
            seen.add(frozen_row)

    duplicate_count = len(duplicate_positions)
    row_count = len(df.index)
    return {
        "total_duplicate_rows": duplicate_count,
        "unique_rows": row_count - duplicate_count,
        "duplicate_percentage": (
            duplicate_count / row_count * 100 if row_count else 0.0
        ),
        "duplicate_records": df.iloc[duplicate_positions].copy(),
    }


def _detected_type(series: pd.Series) -> str:
    values = [value for value in series if not _is_missing(value)]
    if values and all(isinstance(value, list) for value in values):
        return "list"
    if values and all(isinstance(value, dict) for value in values):
        return "dictionary"

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pandas_types.is_bool_dtype(dtype):
        return "boolean"
    if pandas_types.is_integer_dtype(dtype):
        return "integer"
    if pandas_types.is_float_dtype(dtype):
        return "float"
    if pandas_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pandas_types.is_object_dtype(dtype):
        if not values or all(isinstance(value, str) for value in values):
            return "string/object"
        if all(isinstance(value, bool) for value in values):
            return "boolean"
        if all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in values
        ):
            return "integer"
        if all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in values
        ):
            return "float"
        if all(isinstance(value, (date, datetime, pd.Timestamp)) for value in values):
            return "datetime"
        return "other"
    if pandas_types.is_string_dtype(dtype):
        return "string/object"
    return "other"


def profile_data_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify each column's current data type and non-missing value count."""
    return [
        {
            "column_name": str(column),
            "data_type": _detected_type(df[column]),
            "pandas_dtype": str(df[column].dtype),
            "non_null_values": sum(not _is_missing(value) for value in df[column]),
        }
        for column in df.columns
    ]


def generate_quality_summary(
    df: pd.DataFrame,
    api_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a structured dataset and column quality report with API context."""
    metadata = api_metadata or {}
    dimensions = profile_dimensions(df, metadata)
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    types = {item["column_name"]: item for item in profile_data_types(df)}

    columns = []
    for column in dimensions["column_names"]:
        name = str(column)
        columns.append(
            {
                "column_name": name,
                "data_type": types[name]["data_type"],
                "missing_count": missing["columns"][name]["missing_count"],
                "missing_percentage": missing["columns"][name][
                    "missing_percentage"
                ],
                "non_missing_count": missing["columns"][name][
                    "non_missing_count"
                ],
            }
        )

    summary: dict[str, Any] = {
        "provider": metadata.get("provider"),
        "api_name": metadata.get("api_name"),
        "api_endpoint": metadata.get("api_endpoint"),
        "database_id": metadata.get("database_id"),
        "indicator_id": metadata.get("indicator_id"),
        "country_reference_area": metadata.get("country_reference_area"),
        "total_rows": dimensions["total_rows"],
        "total_columns": dimensions["total_columns"],
        "total_missing_values": missing["total_missing_values"],
        "overall_missing_percentage": missing["overall_missing_percentage"],
        "total_duplicate_records": duplicates["total_duplicate_rows"],
        "number_of_unique_records": duplicates["unique_rows"],
        "columns": columns,
        "request_timestamp": metadata.get("request_timestamp"),
        "http_status_code": metadata.get("http_status_code"),
        "api_response_record_count": metadata.get("api_response_record_count"),
        "records_retrieved": metadata.get("records_retrieved"),
        "pagination": metadata.get("pagination"),
        "ingestion_status": metadata.get("status"),
    }
    return summary


def profile_api(
    df: pd.DataFrame,
    api_metadata: Mapping[str, Any] | None = None,
    api_response: Any = None,
) -> dict[str, Any]:
    """Run all five profiling operations on an already-extracted API DataFrame.

    Metadata and the original JSON response are retained by reference; this
    function neither requests API data nor transforms the input DataFrame.
    """
    dimensions = profile_dimensions(df, api_metadata)
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    data_types = profile_data_types(df)
    return {
        "data": df,
        "api_metadata": api_metadata,
        "api_response": api_response,
        "dimensions": dimensions,
        "missing_values": missing,
        "duplicates": duplicates,
        "data_types": data_types,
        "quality_summary": generate_quality_summary(df, api_metadata),
    }


def _profile_extraction(
    dataset: Mapping[str, Any],
    dataset_id: str,
) -> dict[str, Any] | None:
    metadata = dataset.get("metadata", dataset.get("api_metadata", {}))
    if isinstance(metadata, Mapping) and metadata.get("status") == "error":
        return None
    dataframe = dataset.get("data")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"API dataset {dataset_id!r} does not contain a DataFrame.")
    profile = profile_api(
        dataframe,
        api_metadata=metadata if isinstance(metadata, Mapping) else None,
        api_response=dataset.get("api_response"),
    )
    profile["dataset_id"] = dataset_id
    return profile


def profile_multiple_api_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """Profile separate DataFrames or API extractor results without combining.

    Mapping keys are retained as dataset IDs. Sequence entries use the
    extractor's database/indicator/source metadata when present, otherwise
    their position in the sequence.
    """
    has_mapping_ids = isinstance(datasets, Mapping)
    items = list(datasets.items()) if has_mapping_ids else list(enumerate(datasets))
    profiles: dict[str, dict[str, Any]] = {}
    for identifier, dataset in items:
        dataset_id = str(identifier)
        if isinstance(dataset, pd.DataFrame):
            profiles[dataset_id] = profile_api(dataset)
            profiles[dataset_id]["dataset_id"] = dataset_id
            continue
        if not isinstance(dataset, Mapping):
            raise TypeError(
                "API datasets must be DataFrames or API extraction result mappings."
            )

        metadata = dataset.get("metadata", dataset.get("api_metadata", {}))
        if not has_mapping_ids and isinstance(metadata, Mapping):
            dataset_id = str(
                metadata.get("dataset_id")
                or metadata.get("source")
                or metadata.get("database_id")
                or identifier
            )
            if metadata.get("indicator_id"):
                dataset_id = f"{dataset_id}::{metadata['indicator_id']}"
            if dataset_id in profiles:
                dataset_id = f"{dataset_id}::{identifier}"
        profile = _profile_extraction(dataset, dataset_id)
        if profile is not None:
            profiles[dataset_id] = profile
    return profiles
