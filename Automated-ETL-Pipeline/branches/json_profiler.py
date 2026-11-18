"""Stage 2 profiling functions for JSON-derived Pandas DataFrames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


def profile_dimensions(df: pd.DataFrame) -> dict[str, Any]:
    """Return dynamic row, column, dimension, record, and column-name data."""
    row_count, column_count = df.shape
    return {
        "total_rows": row_count,
        "total_columns": column_count,
        "dimensions": (row_count, column_count),
        "column_names": list(df.columns),
        "number_of_records": len(df.index),
    }


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


def profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Report scalar missing values by column without changing nested values."""
    column_profiles: dict[str, dict[str, int | float]] = {}
    total_missing = 0

    for column in df.columns:
        values = df[column]
        missing_count = sum(_is_missing(value) for value in values)
        total_values = len(values)
        non_missing_count = total_values - missing_count
        column_profiles[str(column)] = {
            "total_values": total_values,
            "missing_count": missing_count,
            "missing_percentage": (
                missing_count / total_values * 100 if total_values else 0.0
            ),
            "non_missing_count": non_missing_count,
        }
        total_missing += missing_count

    total_values = len(df.index) * len(df.columns)
    return {
        "columns": column_profiles,
        "total_missing_values": total_missing,
        "total_values": total_values,
        "overall_missing_percentage": (
            total_missing / total_values * 100 if total_values else 0.0
        ),
    }


def _freeze(value: Any) -> Any:
    """Convert nested JSON-like values into stable, hashable row values."""
    if _is_missing(value):
        return ("__missing__",)
    if isinstance(value, Mapping):
        return (
            "__dict__",
            tuple(
                sorted(
                    ((_freeze(key), _freeze(item)) for key, item in value.items()),
                    key=repr,
                )
            ),
        )
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
    """Report full-row duplicates, including rows containing lists/dictionaries."""
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
    duplicate_records = df.iloc[duplicate_positions].copy()
    return {
        "total_duplicate_rows": duplicate_count,
        "unique_rows": row_count - duplicate_count,
        "duplicate_percentage": (
            duplicate_count / row_count * 100 if row_count else 0.0
        ),
        "duplicate_records": duplicate_records,
    }


def _detected_type(series: pd.Series) -> str:
    non_null_values = [value for value in series if not _is_missing(value)]
    if non_null_values:
        if all(isinstance(value, list) for value in non_null_values):
            return "list"
        if all(isinstance(value, dict) for value in non_null_values):
            return "dictionary"

    dtype = series.dtype
    if pandas_types.is_object_dtype(dtype):
        if not non_null_values or all(
            isinstance(value, str) for value in non_null_values
        ):
            return "string/object"
        return "other"
    if pandas_types.is_bool_dtype(dtype):
        return "boolean"
    if pandas_types.is_integer_dtype(dtype):
        return "integer"
    if pandas_types.is_float_dtype(dtype):
        return "float"
    if pandas_types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pandas_types.is_string_dtype(dtype):
        return "string/object"
    return "other"


def profile_data_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Report the inferred value family, original Pandas dtype, and count."""
    return [
        {
            "column_name": str(column),
            "data_type": _detected_type(df[column]),
            "pandas_dtype": str(df[column].dtype),
            "non_null_values": int(sum(not _is_missing(value) for value in df[column])),
        }
        for column in df.columns
    ]


def generate_quality_summary(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Return structured dataset- and column-level quality information."""
    dimensions = profile_dimensions(df)
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

    return {
        "filename": filename,
        "total_rows": dimensions["total_rows"],
        "total_columns": dimensions["total_columns"],
        "total_missing_values": missing["total_missing_values"],
        "overall_missing_percentage": missing["overall_missing_percentage"],
        "total_duplicate_records": duplicates["total_duplicate_rows"],
        "number_of_unique_records": duplicates["unique_rows"],
        "columns": columns,
    }


def profile_json(
    df: pd.DataFrame,
    filename: str,
    parsed_json: Any = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Run all five profiling operations without mutating input data.

    ``parsed_json`` is optional and, when provided by the ingestion result, is
    retained by reference so later stages can inspect the complete source JSON.
    """
    dimensions = profile_dimensions(df)
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    data_types = profile_data_types(df)
    quality_summary = generate_quality_summary(df, filename)
    result = {
        "filename": filename,
        "data": df,
        "parsed_json": parsed_json,
        "dimensions": dimensions,
        "missing_values": missing,
        "duplicates": duplicates,
        "data_types": data_types,
        "quality_summary": quality_summary,
    }
    if table_name is not None:
        result["table_name"] = table_name
    return result


def _profile_extraction(
    extraction: Mapping[str, Any], fallback_filename: str
) -> list[dict[str, Any]]:
    metadata = extraction.get("metadata", {})
    if metadata.get("status", "success") != "success":
        return []
    filename = str(metadata.get("source", fallback_filename))
    parsed_json = extraction.get("json_data")
    tables = extraction.get("dataframes")

    if isinstance(tables, Mapping) and tables:
        results = []
        for table_name, dataframe in tables.items():
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"JSON table {table_name!r} in {filename!r} is not a DataFrame."
                )
            results.append(
                profile_json(
                    dataframe,
                    filename,
                    parsed_json=parsed_json,
                    table_name=str(table_name),
                )
            )
        return results

    dataframe = extraction.get("data")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"JSON dataset {filename!r} has no DataFrame.")
    return [profile_json(dataframe, filename, parsed_json=parsed_json)]


def profile_multiple_json_files(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> list[dict[str, Any]]:
    """Profile separate JSON DataFrames or ingestion results independently.

    Mapping inputs may map filenames to DataFrames or extraction results.
    Sequence inputs may contain extraction results from
    ``extract_multiple_json_files``. If an extraction result includes multiple
    DataFrames, each table is profiled separately with its table name retained.
    """
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(
        enumerate(datasets)
    )
    results: list[dict[str, Any]] = []
    for identifier, dataset in items:
        if isinstance(dataset, pd.DataFrame):
            results.append(profile_json(dataset, str(identifier)))
        elif isinstance(dataset, Mapping):
            results.extend(_profile_extraction(dataset, str(identifier)))
        else:
            raise TypeError(
                "Datasets must map filenames to DataFrames or contain JSON "
                "extraction result dictionaries."
            )
    return results
