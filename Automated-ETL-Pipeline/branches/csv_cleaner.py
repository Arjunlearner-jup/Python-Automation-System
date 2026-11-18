"""Read-only-input CSV cleaning operations for Stage 3 of the ETL pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pandas_types


DEFAULT_PROCESSED_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "csv"
)
DATE_NAME_PATTERN = re.compile(r"(date|datetime|timestamp)", re.IGNORECASE)
YEAR_COLUMN_PATTERN = re.compile(r"^(?:18|19|20|21)\d{2}$")
CURRENCY_NAME_PATTERN = re.compile(
    r"(currency|price|amount|cost|revenue|income|salary|wage|budget)",
    re.IGNORECASE,
)
CURRENCY_TOKENS = {
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "₹": "INR",
    "INR": "INR",
}


def identify_duplicate_rows(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> dict[str, Any]:
    """Identify exact duplicate records without changing the input."""
    duplicate_mask = df.duplicated(subset=list(subset) if subset else None)
    return {
        "duplicate_rows_detected": int(duplicate_mask.sum()),
        "duplicate_records": df.loc[duplicate_mask].copy(),
        "duplicate_mask": duplicate_mask,
    }


def remove_duplicate_records(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a duplicate-free copy plus counts and removed records."""
    duplicate_info = identify_duplicate_rows(df, subset=subset)
    cleaned = df.loc[~duplicate_info["duplicate_mask"]].copy()
    return cleaned, {
        "duplicate_rows_detected": duplicate_info["duplicate_rows_detected"],
        "rows_removed": len(df.index) - len(cleaned.index),
        "duplicate_records": duplicate_info["duplicate_records"],
        "subset": list(subset) if subset else None,
    }


def count_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count null, empty, and whitespace-only values in each column."""
    counts = {}
    for column in df.columns:
        series = df[column]
        blank_text = series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        counts[str(column)] = int((series.isna() | blank_text).sum())
    return counts


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "leave",
    fill_value: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one explicit missing-value strategy to a copy of the DataFrame."""
    allowed = {"leave", "drop_rows", "mean", "median", "mode", "fill_value"}
    if strategy not in allowed:
        raise ValueError(f"Unknown missing-value strategy: {strategy!r}.")

    before = count_missing_values(df)
    cleaned = df.copy(deep=True)
    if strategy == "drop_rows":
        missing_mask = cleaned.apply(
            lambda column_values: column_values.isna()
            | column_values.map(
                lambda item: isinstance(item, str) and not item.strip()
            )
        )
        cleaned = cleaned.loc[~missing_mask.any(axis=1)].copy()
    elif strategy in {"mean", "median"}:
        for column in cleaned.columns:
            series = cleaned[column]
            if not pandas_types.is_numeric_dtype(series.dtype):
                continue
            value = series.mean() if strategy == "mean" else series.median()
            if pd.notna(value):
                cleaned[column] = series.fillna(value)
    elif strategy == "mode":
        for column in cleaned.columns:
            series = cleaned[column]
            if pandas_types.is_numeric_dtype(series.dtype):
                continue
            nonblank = series.mask(
                series.map(
                    lambda item: isinstance(item, str) and not item.strip()
                )
            )
            modes = nonblank.dropna().mode()
            if not modes.empty:
                value = modes.iloc[0]
                if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
                    if value not in cleaned[column].cat.categories:
                        cleaned[column] = cleaned[column].cat.add_categories([value])
                missing_mask = cleaned[column].isna() | cleaned[column].map(
                    lambda item: isinstance(item, str) and not item.strip()
                )
                cleaned[column] = cleaned[column].mask(missing_mask, value)
    elif strategy == "fill_value":
        if fill_value is None:
            raise ValueError("fill_value strategy requires a non-None fill_value.")
        for column in cleaned.columns:
            missing_mask = cleaned[column].isna() | cleaned[column].map(
                lambda item: isinstance(item, str) and not item.strip()
            )
            if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
                if fill_value not in cleaned[column].cat.categories:
                    cleaned[column] = cleaned[column].cat.add_categories([fill_value])
            cleaned[column] = cleaned[column].mask(missing_mask, fill_value)

    after = count_missing_values(cleaned)
    return cleaned, {
        "strategy": strategy,
        "missing_values_before": before,
        "missing_values_after": after,
        "total_missing_before": sum(before.values()),
        "total_missing_after": sum(after.values()),
    }


def standardise_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize column labels and make collisions unique, without changing data."""
    cleaned = df.copy(deep=True)
    original_names = list(cleaned.columns)
    seen: set[str] = set()
    new_names: list[str] = []

    for original in original_names:
        name = str(original).strip().lower()
        name = re.sub(r"[\s\-]+", "_", name)
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name).strip("_") or "column"
        unique_name = name
        suffix = 2
        while unique_name in seen:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        seen.add(unique_name)
        new_names.append(unique_name)

    cleaned.columns = new_names
    return cleaned, {
        "original_column_names": original_names,
        "standardised_column_names": new_names,
        "column_names_changed": [
            {"original": old, "standardised": new}
            for old, new in zip(original_names, new_names)
            if str(old) != new
        ],
    }


def identify_date_columns(df: pd.DataFrame) -> list[str]:
    """Find likely date columns by label and representative values.

    Numeric year fields and columns named only as a four-digit year are excluded.
    """
    candidates: list[str] = []
    for column in df.columns:
        name = str(column)
        series = df[column]
        if YEAR_COLUMN_PATTERN.fullmatch(name.strip()):
            continue
        if pandas_types.is_datetime64_any_dtype(series.dtype):
            candidates.append(name)
            continue
        if DATE_NAME_PATTERN.search(name):
            candidates.append(name)
            continue
        if pandas_types.is_numeric_dtype(series.dtype):
            continue

        values = series.dropna()
        if values.empty:
            continue
        sample = values.astype(str).head(20)
        date_shaped = sample.str.contains(r"[-/T:]", regex=True).mean()
        if date_shaped < 0.8:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8:
            candidates.append(name)
    return candidates


def standardise_date_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse selected/recognized date columns; report invalid source values."""
    selected_columns = (
        list(columns) if columns is not None else identify_date_columns(df)
    )
    unknown_columns = set(selected_columns).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown date column(s): {sorted(unknown_columns)}")

    cleaned = df.copy(deep=True)
    reports = []
    for column_name in selected_columns:
        column = next(col for col in cleaned.columns if str(col) == column_name)
        original = cleaned[column].copy()
        if YEAR_COLUMN_PATTERN.fullmatch(column_name.strip()) or re.search(
            r"(^|_)year($|_)", column_name, re.IGNORECASE
        ):
            reports.append(
                {
                    "column": column_name,
                    "date_format": date_format or "automatic",
                    "values_converted": 0,
                    "invalid_value_count": 0,
                    "invalid_values": [],
                    "status": "skipped",
                    "reason": "Ordinary year columns are not converted to dates.",
                }
            )
            continue
        nonempty = original.notna() & original.astype("string").str.strip().ne("")
        if date_format:
            parsed = pd.to_datetime(
                original, format=date_format, errors="coerce"
            )
        else:
            parsed = pd.to_datetime(original, errors="coerce", format="mixed")
        invalid_mask = nonempty & parsed.isna()
        invalid_values = original.loc[invalid_mask].tolist()
        cleaned[column] = parsed
        reports.append(
            {
                "column": column_name,
                "date_format": date_format or "automatic",
                "values_converted": int(parsed.notna().sum()),
                "invalid_value_count": int(invalid_mask.sum()),
                "invalid_values": invalid_values,
            }
        )
    return cleaned, {"columns_processed": reports}


def identify_currency_columns(df: pd.DataFrame) -> list[str]:
    """Find potential currency fields by their names, not by numeric contents."""
    return [
        str(column)
        for column in df.columns
        if CURRENCY_NAME_PATTERN.search(str(column))
    ]


def _parse_currency_value(value: Any, selected_currency: str | None) -> tuple[Any, Any, bool]:
    if value is None or value is pd.NA or value is pd.NaT:
        return value, None, True
    if isinstance(value, float) and pd.isna(value):
        return value, None, True
    if isinstance(value, bool):
        return value, None, False
    if isinstance(value, Real):
        return value, selected_currency, True
    if not isinstance(value, str):
        return value, None, False

    text = value.strip()
    if not text:
        return value, None, True
    currency_code = selected_currency
    for token, code in sorted(CURRENCY_TOKENS.items(), key=lambda item: -len(item[0])):
        if token in text:
            currency_code = code
            text = text.replace(token, "")
    text = text.strip().replace(" ", "")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        pieces = text.split(",")
        text = (
            "".join(pieces)
            if len(pieces[-1]) == 3 and all(piece.isdigit() for piece in pieces)
            else ".".join(pieces)
        )
    text = re.sub(r"[^\d.eE+\-]", "", text)
    try:
        amount = float(text)
    except ValueError:
        return value, currency_code, False
    if negative:
        amount = -amount
    return amount, currency_code, True


def standardise_currency_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    currency_format: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse selected currency strings to numbers without any FX conversion.

    Detected symbols/codes (or a user-selected currency) are retained in a
    separate companion column named ``<column>_currency``.
    """
    allowed_currencies = {"auto", "USD", "EUR", "GBP", "INR"}
    if currency_format not in allowed_currencies:
        raise ValueError(
            f"currency_format must be one of {sorted(allowed_currencies)}."
        )
    selected_columns = (
        list(columns) if columns is not None else identify_currency_columns(df)
    )
    unknown_columns = set(selected_columns).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown currency column(s): {sorted(unknown_columns)}")

    cleaned = df.copy(deep=True)
    reports = []
    selected_code = None if currency_format == "auto" else currency_format
    for column_name in selected_columns:
        column = next(col for col in cleaned.columns if str(col) == column_name)
        values: list[Any] = []
        currencies: list[Any] = []
        invalid_values: list[Any] = []
        for value in cleaned[column].tolist():
            amount, code, valid = _parse_currency_value(value, selected_code)
            values.append(amount)
            currencies.append(code)
            if not valid:
                invalid_values.append(value)
        cleaned[column] = values
        if any(currency is not None for currency in currencies):
            companion = f"{column_name}_currency"
            suffix = 2
            while companion in cleaned.columns:
                companion = f"{column_name}_currency_{suffix}"
                suffix += 1
            cleaned[companion] = currencies
        reports.append(
            {
                "column": column_name,
                "currency_format": currency_format,
                "values_parsed": len(values) - len(invalid_values),
                "invalid_value_count": len(invalid_values),
                "invalid_values": invalid_values,
            }
        )
    return cleaned, {"columns_processed": reports}


def _is_identifier_column(column: Any) -> bool:
    return bool(re.search(r"(^|[_\s])(id|code|identifier)([_\s]|$)", str(column), re.I))


def convert_data_types(
    df: pd.DataFrame,
    conversions: Mapping[str, str],
    date_format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert explicitly selected columns and report values that did not parse."""
    allowed = {"integer", "float", "string", "boolean", "datetime"}
    unknown_columns = set(conversions).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown conversion column(s): {sorted(unknown_columns)}")
    invalid_types = set(conversions.values()).difference(allowed)
    if invalid_types:
        raise ValueError(f"Unsupported target type(s): {sorted(invalid_types)}")

    cleaned = df.copy(deep=True)
    reports = []
    for column_name, target_type in conversions.items():
        column = next(col for col in cleaned.columns if str(col) == column_name)
        original = cleaned[column].copy()
        if target_type in {"integer", "float"} and _is_identifier_column(column):
            reports.append(
                {
                    "column": column_name,
                    "target_type": target_type,
                    "status": "skipped",
                    "reason": "Identifier-like columns are not converted to numeric.",
                    "failed_count": 0,
                    "failed_values": [],
                }
            )
            continue

        if target_type in {"integer", "float"}:
            numeric = pd.to_numeric(original, errors="coerce")
            invalid_mask = original.notna() & numeric.isna()
            if target_type == "integer":
                non_integral = numeric.notna() & (numeric % 1 != 0)
                invalid_mask = invalid_mask | non_integral
                numeric = numeric.mask(non_integral)
                converted = numeric.astype("Int64")
            else:
                converted = numeric.astype("Float64")
        elif target_type == "string":
            converted = original.astype("string")
            invalid_mask = pd.Series(False, index=original.index)
        elif target_type == "boolean":
            true_values = {"true", "1", "yes", "y"}
            false_values = {"false", "0", "no", "n"}

            def to_boolean(value: Any) -> Any:
                if pd.isna(value):
                    return pd.NA
                if isinstance(value, bool):
                    return value
                normalized = str(value).strip().lower()
                if normalized in true_values:
                    return True
                if normalized in false_values:
                    return False
                return pd.NA

            converted = original.map(to_boolean).astype("boolean")
            invalid_mask = original.notna() & converted.isna()
        else:
            if YEAR_COLUMN_PATTERN.fullmatch(column_name.strip()) or re.search(
                r"(^|_)year($|_)", column_name, re.IGNORECASE
            ):
                reports.append(
                    {
                        "column": column_name,
                        "target_type": target_type,
                        "status": "skipped",
                        "reason": "Ordinary year columns are not converted to dates.",
                        "failed_count": 0,
                        "failed_values": [],
                    }
                )
                continue
            converted = pd.to_datetime(
                original,
                errors="coerce",
                format=date_format or "mixed",
            )
            invalid_mask = (
                original.notna()
                & original.astype("string").str.strip().ne("")
                & converted.isna()
            )

        cleaned[column] = converted
        failed_values = original.loc[invalid_mask].tolist()
        reports.append(
            {
                "column": column_name,
                "target_type": target_type,
                "status": "success" if not failed_values else "partial",
                "failed_count": len(failed_values),
                "failed_values": failed_values,
            }
        )
    return cleaned, {"conversions": reports}


def clean_csv_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    *,
    remove_duplicates: bool = False,
    duplicate_subset: Sequence[str] | None = None,
    missing_strategy: str = "leave",
    missing_fill_value: Any = None,
    standardise_names: bool = False,
    standardise_dates: bool = False,
    date_columns: Sequence[str] | None = None,
    date_format: str | None = None,
    standardise_currencies: bool = False,
    currency_columns: Sequence[str] | None = None,
    currency_format: str = "auto",
    type_conversions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply only selected cleaning operations to a copy of the source frame."""
    cleaned = df.copy(deep=True)
    original_rows, original_columns = df.shape
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "original_row_count": original_rows,
        "original_column_count": original_columns,
        "duplicate_rows_detected": 0,
        "duplicate_rows_removed": 0,
        "missing_values_before": count_missing_values(df),
        "missing_values_after": count_missing_values(df),
        "missing_values_total_before": sum(count_missing_values(df).values()),
        "missing_values_total_after": sum(count_missing_values(df).values()),
        "column_names_changed": [],
        "original_column_names": list(df.columns),
        "date_conversions_performed": [],
        "currency_parsing_results": [],
        "data_type_conversions_performed": [],
        "failed_conversions": [],
        "cleaning_status": "success",
    }

    if remove_duplicates:
        cleaned, duplicate_report = remove_duplicate_records(
            cleaned, subset=duplicate_subset
        )
        report["duplicate_rows_detected"] = duplicate_report[
            "duplicate_rows_detected"
        ]
        report["duplicate_rows_removed"] = duplicate_report["rows_removed"]
        report["duplicate_subset"] = duplicate_report["subset"]

    cleaned, missing_report = handle_missing_values(
        cleaned, missing_strategy, missing_fill_value
    )
    report["missing_values_before"] = missing_report["missing_values_before"]
    report["missing_values_after"] = missing_report["missing_values_after"]
    report["missing_strategy"] = missing_report["strategy"]

    if standardise_names:
        cleaned, names_report = standardise_column_names(cleaned)
        report["column_names_changed"] = names_report["column_names_changed"]
        report["standardised_column_names"] = names_report[
            "standardised_column_names"
        ]

    if standardise_dates:
        dates_to_process = list(date_columns) if date_columns is not None else None
        if dates_to_process is not None and standardise_names:
            renames = {
                str(change["original"]): str(change["standardised"])
                for change in report["column_names_changed"]
            }
            dates_to_process = [renames.get(name, name) for name in dates_to_process]
        cleaned, date_report = standardise_date_columns(
            cleaned, dates_to_process, date_format
        )
        report["date_conversions_performed"] = date_report["columns_processed"]

    if standardise_currencies:
        currencies_to_process = (
            list(currency_columns) if currency_columns is not None else None
        )
        if currencies_to_process is not None and standardise_names:
            renames = {
                str(change["original"]): str(change["standardised"])
                for change in report["column_names_changed"]
            }
            currencies_to_process = [
                renames.get(name, name) for name in currencies_to_process
            ]
        cleaned, currency_report = standardise_currency_columns(
            cleaned, currencies_to_process, currency_format
        )
        report["currency_parsing_results"] = currency_report["columns_processed"]

    if type_conversions:
        conversions_to_apply = dict(type_conversions)
        if standardise_names:
            renames = {
                str(change["original"]): str(change["standardised"])
                for change in report["column_names_changed"]
            }
            conversions_to_apply = {
                renames.get(name, name): target_type
                for name, target_type in conversions_to_apply.items()
            }
        cleaned, conversion_report = convert_data_types(
            cleaned, conversions_to_apply, date_format=date_format
        )
        report["data_type_conversions_performed"] = conversion_report[
            "conversions"
        ]
        report["failed_conversions"] = [
            item
            for item in conversion_report["conversions"]
            if item.get("failed_count", 0) > 0
        ]

    report["final_row_count"] = len(cleaned.index)
    report["final_column_count"] = len(cleaned.columns)
    report["final_column_names"] = list(cleaned.columns)
    report["missing_values_after"] = count_missing_values(cleaned)
    report["missing_values_total_after"] = sum(
        report["missing_values_after"].values()
    )
    has_failed_conversion = bool(report["failed_conversions"]) or any(
        item.get("invalid_value_count", 0) > 0
        for item in (
            report["date_conversions_performed"]
            + report["currency_parsing_results"]
        )
    )
    report["cleaning_status"] = "partial" if has_failed_conversion else "success"
    return {"data": cleaned, "report": report}


def save_cleaned_csv(
    df: pd.DataFrame,
    dataset_name: str,
    output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY,
) -> Path:
    """Save a cleaned copy under the processed CSV directory without overwrites."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dataset_name).stem)
    destination = directory / f"{safe_stem}_cleaned.csv"
    suffix = 2
    while destination.exists():
        destination = directory / f"{safe_stem}_cleaned_{suffix}.csv"
        suffix += 1
    df.to_csv(destination, index=False)
    return destination
