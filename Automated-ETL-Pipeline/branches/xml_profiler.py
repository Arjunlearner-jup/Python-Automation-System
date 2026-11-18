"""Stage 2 profiling functions for XML-derived Pandas DataFrames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


def profile_dimensions(df: pd.DataFrame) -> dict[str, Any]:
    """Return dynamic DataFrame dimensions and XML document-level counts."""
    rows, columns = df.shape
    return {
        "total_rows": rows,
        "total_columns": columns,
        "dimensions": (rows, columns),
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
    """Count null and empty text values without flattening nested XML cells."""
    column_profiles: dict[str, dict[str, int | float]] = {}
    total_missing = 0

    for column in df.columns:
        values = df[column]
        missing_count = sum(_is_missing(value) for value in values)
        total_values = len(values)
        column_profiles[str(column)] = {
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
        "columns": column_profiles,
        "total_missing_values": total_missing,
        "total_values": total_values,
        "overall_missing_percentage": (
            total_missing / total_values * 100 if total_values else 0.0
        ),
    }


def _freeze(value: Any) -> Any:
    """Create hashable values recursively for full-row duplicate detection."""
    if _is_missing(value):
        return ("__missing__",)
    if isinstance(value, Mapping):
        frozen_pairs = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        return ("__dict__", tuple(sorted(frozen_pairs, key=repr)))
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
    """Report full-row duplicates, including rows with nested lists/dictionaries."""
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
    if pandas_types.is_object_dtype(dtype):
        if not values or all(isinstance(value, str) for value in values):
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
    """Report each extracted XML column's value type and non-null count."""
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
    filename: str,
    table_name: str | None = None,
    xml_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build DataFrame quality statistics and available XML document metadata."""
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

    summary: dict[str, Any] = {
        "filename": filename,
        "table_name": table_name,
        "total_rows": dimensions["total_rows"],
        "total_columns": dimensions["total_columns"],
        "total_missing_values": missing["total_missing_values"],
        "overall_missing_percentage": missing["overall_missing_percentage"],
        "total_duplicate_records": duplicates["total_duplicate_rows"],
        "number_of_unique_records": duplicates["unique_rows"],
        "columns": columns,
    }
    if xml_metadata is not None:
        extracted_tables = xml_metadata.get("dataframes", {})
        summary.update(
            {
                "root_element": xml_metadata.get("root_element"),
                "xml_element_count": xml_metadata.get("element_count"),
                "xml_attribute_count": xml_metadata.get("attribute_count"),
                "extracted_table_count": len(extracted_tables),
                "extracted_table_names": list(extracted_tables),
            }
        )
    return summary


def profile_xml(
    df: pd.DataFrame,
    filename: str,
    xml_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all five profiling operations for one extracted XML DataFrame."""
    dimensions = profile_dimensions(df)
    if xml_metadata is not None:
        dimensions.update(
            {
                "root_element": xml_metadata.get("root_element"),
                "xml_element_count": xml_metadata.get("element_count"),
                "xml_attribute_count": xml_metadata.get("attribute_count"),
                "extracted_record_group_count": len(
                    xml_metadata.get("repeating_record_elements", [])
                ),
            }
        )
    missing = profile_missing_values(df)
    duplicates = profile_duplicates(df)
    data_types = profile_data_types(df)
    table_name = (
        xml_metadata.get("table_name") if xml_metadata is not None else None
    )
    return {
        "filename": filename,
        "table_name": table_name,
        "data": df,
        "xml_metadata": xml_metadata,
        "dimensions": dimensions,
        "missing_values": missing,
        "duplicates": duplicates,
        "data_types": data_types,
        "quality_summary": generate_quality_summary(
            df, filename, table_name=table_name, xml_metadata=xml_metadata
        ),
    }


def _profile_extraction(
    extraction: Mapping[str, Any], fallback_filename: str
) -> list[dict[str, Any]]:
    metadata = extraction.get("metadata", {})
    if metadata.get("status", "success") != "success":
        return []
    filename = str(metadata.get("source", fallback_filename))
    source_tables = extraction.get("dataframes")
    if isinstance(source_tables, Mapping) and source_tables:
        profiles = []
        for table_name, dataframe in source_tables.items():
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(
                    f"XML table {table_name!r} in {filename!r} is not a DataFrame."
                )
            table_metadata = dict(metadata)
            table_metadata["table_name"] = str(table_name)
            profile = profile_xml(dataframe, filename, table_metadata)
            profile["xml_tree"] = extraction.get("xml_tree")
            profiles.append(profile)
        return profiles

    dataframe = extraction.get("data")
    if not isinstance(dataframe, pd.DataFrame):
        return []
    profile = profile_xml(dataframe, filename, metadata)
    profile["xml_tree"] = extraction.get("xml_tree")
    return [profile]


def profile_multiple_xml_files(
    datasets: Mapping[str, Any] | Sequence[Any],
) -> list[dict[str, Any]]:
    """Profile each XML dataset and extracted table independently.

    Inputs can be a filename-to-DataFrame mapping or extraction results from
    ``extract_multiple_xml_files``. Failed extraction results are skipped.
    """
    items = (
        list(datasets.items())
        if isinstance(datasets, Mapping)
        else list(enumerate(datasets))
    )
    profiles: list[dict[str, Any]] = []
    for identifier, dataset in items:
        if isinstance(dataset, pd.DataFrame):
            profiles.append(profile_xml(dataset, str(identifier)))
        elif isinstance(dataset, Mapping):
            profiles.extend(_profile_extraction(dataset, str(identifier)))
        else:
            raise TypeError(
                "Datasets must map names to DataFrames or contain XML "
                "extraction result dictionaries."
            )
    return profiles
