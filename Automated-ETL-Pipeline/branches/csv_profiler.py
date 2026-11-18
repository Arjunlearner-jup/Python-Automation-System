"""Stage 2 profiling functions for CSV DataFrames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


def profile_dimensions(df: pd.DataFrame) -> dict[str, Any]:
    """Return dynamic row, column, dimension, and column-name information."""
    row_count, column_count = df.shape
    return {
        "total_rows": row_count,
        "total_columns": column_count,
        "dimensions": (row_count, column_count),
        "column_names": list(df.columns),
    }


def _missing_mask(series: pd.Series) -> pd.Series:
    """Mark nulls, empty strings, and strings containing only whitespace."""
    return series.isna() | series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Report missing values by column without changing the DataFrame."""
    columns: dict[str, dict[str, int | float]] = {}
    total_missing = 0

    for column in df.columns:
        values = df[column]
        missing_count = int(_missing_mask(values).sum())
        total_values = len(values)
        non_missing_count = total_values - missing_count
        columns[str(column)] = {
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
        "columns": columns,
        "total_missing_values": total_missing,
        "total_values": total_values,
        "overall_missing_percentage": (
            total_missing / total_values * 100 if total_values else 0.0
        ),
    }


def profile_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Report full-row duplicates and unique rows without removing anything."""
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    row_count = len(df.index)
    return {
        "total_duplicate_rows": duplicate_count,
        "unique_rows": row_count - duplicate_count,
        "duplicate_percentage": (
            duplicate_count / row_count * 100 if row_count else 0.0
        ),
        "duplicate_records": df.loc[duplicate_mask].copy(),
    }


def _detected_type(dtype: Any) -> str:
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
    """Report each column's existing Pandas dtype and non-null count."""
    return [
        {
            "column_name": str(column),
            "data_type": str(df[column].dtype),
            "detected_type": _detected_type(df[column].dtype),
            "non_null_values": int(df[column].notna().sum()),
        }
        for column in df.columns
    ]


def generate_quality_summary(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Return a structured dataset- and column-level quality report."""
    dimensions = profile_dimensions(df)
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    data_types = {
        item["column_name"]: item for item in profile_data_types(df)
    }

    columns = []
    for column_name in dimensions["column_names"]:
        name = str(column_name)
        missing_info = missing["columns"][name]
        type_info = data_types[name]
        columns.append(
            {
                "column_name": name,
                "data_type": type_info["data_type"],
                "missing_count": missing_info["missing_count"],
                "missing_percentage": missing_info["missing_percentage"],
                "non_missing_count": missing_info["non_missing_count"],
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


def profile_csv(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Run all CSV profiling operations and retain the original DataFrame.

    The ``data`` reference is the input DataFrame itself; all profile operations
    are read-only, so callers can pass it to Stage 3 without copying or mutation.
    """
    dimensions = profile_dimensions(df)
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    data_types = profile_data_types(df)
    quality_summary = generate_quality_summary(df, filename)
    return {
        "filename": filename,
        "data": df,
        "dimensions": dimensions,
        "missing_values": missing,
        "duplicates": duplicates,
        "data_types": data_types,
        "quality_summary": quality_summary,
    }


def _get_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(datasets, Mapping):
        items = list(datasets.items())
    else:
        items = list(enumerate(datasets))

    extracted: list[tuple[str, pd.DataFrame]] = []
    for identifier, item in items:
        if isinstance(item, pd.DataFrame):
            filename, dataframe = str(identifier), item
        elif isinstance(item, Mapping):
            metadata = item.get("metadata", {})
            if metadata.get("status", "success") != "success":
                continue
            dataframe = item.get("data")
            filename = str(metadata.get("source", identifier))
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"CSV dataset {filename!r} has no DataFrame.")
        else:
            raise TypeError(
                "Datasets must map filenames to DataFrames or contain CSV "
                "extraction result dictionaries."
            )
        extracted.append((filename or Path(str(identifier)).name, dataframe))
    return extracted


def profile_multiple_csv_files(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> list[dict[str, Any]]:
    """Profile separate DataFrames or CSV extraction results independently."""
    return [
        profile_csv(dataframe, filename)
        for filename, dataframe in _get_datasets(datasets)
    ]
