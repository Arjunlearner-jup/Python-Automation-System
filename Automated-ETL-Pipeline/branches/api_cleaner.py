"""Stage 3 cleaning operations for already-ingested API observation data."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.api import types as pandas_types


DEFAULT_PROCESSED_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "api"
)
IDENTIFIER_PATTERN = re.compile(
    r"(^|[_\s])(id|code|identifier|ref_area|indicator|database)([_\s]|$)",
    re.IGNORECASE,
)
DATE_NAME_PATTERN = re.compile(r"(date|datetime|timestamp|time_period)", re.I)
CURRENCY_NAME_PATTERN = re.compile(
    r"(currency|amount|price|cost|revenue|income|salary|wage|budget)", re.I
)
YEAR_PATTERN = re.compile(r"^\d{4}$")
YEAR_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
CURRENCY_TOKENS = {
    "US$": "USD",
    "USD": "USD",
    "$": "USD",
    "EUR": "EUR",
    "€": "EUR",
    "GBP": "GBP",
    "£": "GBP",
    "INR": "INR",
    "₹": "INR",
}


def identify_duplicate_records(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> dict[str, Any]:
    """Identify exact duplicate observations by full row or selected key columns."""
    missing_columns = set(subset or []).difference(map(str, df.columns))
    if missing_columns:
        raise KeyError(f"Unknown duplicate key column(s): {sorted(missing_columns)}")
    if subset:
        columns = [next(col for col in df.columns if str(col) == name) for name in subset]
    else:
        columns = list(df.columns)
    if not columns:
        duplicate_mask = pd.Series(False, index=df.index, dtype=bool)
    else:
        seen: set[Any] = set()
        duplicate_positions = []
        for position, row in enumerate(
            df.loc[:, columns].itertuples(index=False, name=None)
        ):
            frozen = tuple(_freeze(value) for value in row)
            if frozen in seen:
                duplicate_positions.append(position)
            else:
                seen.add(frozen)
        duplicate_mask = pd.Series(False, index=df.index, dtype=bool)
        if duplicate_positions:
            duplicate_mask.iloc[duplicate_positions] = True
    return {
        "duplicate_records_detected": int(duplicate_mask.sum()),
        "duplicate_records": df.loc[duplicate_mask].copy(),
        "duplicate_mask": duplicate_mask,
    }


def remove_duplicate_records(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove full-row or selected-key duplicates from a copy of the input."""
    duplicate_info = identify_duplicate_records(df, subset)
    cleaned = df.loc[~duplicate_info["duplicate_mask"]].copy()
    return cleaned, {
        "duplicate_records_detected": duplicate_info["duplicate_records_detected"],
        "duplicate_records_removed": len(df.index) - len(cleaned.index),
        "duplicate_records": duplicate_info["duplicate_records"],
        "duplicate_key_columns": list(subset) if subset else None,
    }


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if getattr(result, "ndim", 0) == 0 else False


def _freeze(value: Any) -> Any:
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


def count_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count nulls, empty strings, and whitespace-only strings by column."""
    return {
        str(column): sum(_is_missing(value) for value in df[column])
        for column in df.columns
    }


def missing_value_percentages(df: pd.DataFrame) -> dict[str, float]:
    """Return each column's missing-value percentage using the row count."""
    row_count = len(df.index)
    if not row_count:
        return {str(column): 0.0 for column in df.columns}
    return {
        column: count * 100.0 / row_count
        for column, count in count_missing_values(df).items()
    }


def _column_missing_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_missing).astype(bool)


def _numeric_series(series: pd.Series) -> pd.Series | None:
    if pandas_types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce")
    nonmissing = series.loc[~_column_missing_mask(series)]
    if nonmissing.empty:
        return None
    parsed = pd.to_numeric(nonmissing, errors="coerce")
    if parsed.notna().all():
        return pd.to_numeric(series, errors="coerce")
    return None


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "leave",
    *,
    column_strategies: Mapping[str, str] | None = None,
    fill_value: Any = None,
    column_fill_values: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit global or per-column missing-value strategies.

    Identifier fields are not imputed unless a specific per-column strategy or
    fill value is supplied by the caller.
    """
    strategies = {
        "leave",
        "drop_rows",
        "mean",
        "median",
        "mode",
        "fill_value",
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown missing-value strategy: {strategy!r}.")
    selected_strategies = dict(column_strategies or {})
    selected_fill_values = dict(column_fill_values or {})
    unknown_columns = (set(selected_strategies) | set(selected_fill_values)).difference(
        map(str, df.columns)
    )
    if unknown_columns:
        raise KeyError(f"Unknown missing-value column(s): {sorted(unknown_columns)}")
    if any(value not in strategies for value in selected_strategies.values()):
        raise ValueError("A per-column missing strategy is not supported.")
    if strategy == "fill_value" and fill_value is None and not selected_fill_values:
        raise ValueError("fill_value strategy requires a non-None fill value.")

    before = count_missing_values(df)
    cleaned = df.copy(deep=True)
    if strategy == "drop_rows":
        row_has_missing = pd.DataFrame(
            {column: _column_missing_mask(cleaned[column]) for column in cleaned}
        ).any(axis=1)
        cleaned = cleaned.loc[~row_has_missing].copy()
    else:
        for column in cleaned.columns:
            column_name = str(column)
            selected = selected_strategies.get(column_name, strategy)
            has_explicit_column_rule = column_name in selected_strategies
            if selected == "leave" or (
                IDENTIFIER_PATTERN.search(column_name)
                and not has_explicit_column_rule
                and column_name not in selected_fill_values
            ):
                continue
            missing_mask = _column_missing_mask(cleaned[column])
            if not missing_mask.any():
                continue
            if selected == "drop_rows":
                cleaned = cleaned.loc[~missing_mask].copy()
                continue
            if selected in {"mean", "median"}:
                numeric = _numeric_series(cleaned[column])
                if numeric is None:
                    continue
                replacement = numeric.mean() if selected == "mean" else numeric.median()
                if pd.notna(replacement):
                    cleaned.loc[missing_mask, column] = replacement
                continue
            if selected == "mode":
                observed = cleaned.loc[~missing_mask, column]
                numeric = _numeric_series(cleaned[column])
                if numeric is not None:
                    continue
                modes = observed.mode()
                if not modes.empty:
                    cleaned.loc[missing_mask, column] = modes.iloc[0]
                continue
            if selected == "fill_value":
                replacement = selected_fill_values.get(column_name, fill_value)
                if replacement is None:
                    raise ValueError(
                        f"Column {column_name!r} requires an explicit fill value."
                    )
                cleaned.loc[missing_mask, column] = replacement

    after = count_missing_values(cleaned)
    return cleaned, {
        "strategy": strategy,
        "column_strategies": selected_strategies,
        "missing_values_before": before,
        "missing_values_after": after,
        "missing_value_percentages_before": missing_value_percentages(df),
        "missing_value_percentages_after": missing_value_percentages(cleaned),
        "total_missing_before": sum(before.values()),
        "total_missing_after": sum(after.values()),
    }


def standardise_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize DataFrame labels and return a source-to-clean mapping."""
    cleaned = df.copy(deep=True)
    original_names = list(cleaned.columns)
    seen: set[str] = set()
    cleaned_names = []
    for original in original_names:
        name = re.sub(r"[\s\-]+", "_", str(original).strip().lower())
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name).strip("_") or "column"
        unique_name = name
        suffix = 2
        while unique_name in seen:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        seen.add(unique_name)
        cleaned_names.append(unique_name)

    cleaned.columns = cleaned_names
    mapping = [
        {"original": original, "standardised": new}
        for original, new in zip(original_names, cleaned_names)
    ]
    return cleaned, {
        "original_column_names": original_names,
        "standardised_column_names": cleaned_names,
        "column_name_mapping": mapping,
        "column_names_changed": [item for item in mapping if str(item["original"]) != item["standardised"]],
    }


def identify_date_columns(df: pd.DataFrame) -> list[str]:
    """Identify temporal columns by name and value shape, excluding plain years."""
    candidates = []
    for column in df.columns:
        name = str(column)
        series = df[column]
        if not DATE_NAME_PATTERN.search(name):
            sample = series.dropna().astype(str).head(30)
            date_like = sample.str.contains(r"[-/T:]", regex=True).mean() if len(sample) else 0
            if date_like < 0.8:
                continue
        candidates.append(name)
    return candidates


def _has_full_date_precision(value: Any, date_format: str | None = None) -> bool:
    if not isinstance(value, str):
        return isinstance(value, (datetime, pd.Timestamp))
    text = value.strip()
    if YEAR_PATTERN.fullmatch(text) or YEAR_MONTH_PATTERN.fullmatch(text):
        return False
    return True


def _is_year_or_month_precision(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(YEAR_PATTERN.fullmatch(text) or YEAR_MONTH_PATTERN.fullmatch(text))


def _parse_dates(values: pd.Series, date_format: str | None) -> pd.Series:
    return pd.to_datetime(
        values,
        format=date_format or "mixed",
        errors="coerce",
    )


def standardise_date_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert full-precision date fields; preserve year/year-month observations."""
    selected_columns = list(columns) if columns is not None else identify_date_columns(df)
    unknown_columns = set(selected_columns).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown date column(s): {sorted(unknown_columns)}")
    cleaned = df.copy(deep=True)
    reports = []
    for name in selected_columns:
        column = next(col for col in cleaned.columns if str(col) == name)
        original = cleaned[column].copy()
        temporal_values = original.map(
            lambda value: not _is_missing(value)
            and _has_full_date_precision(value, date_format)
        )
        has_precision_only_values = original.map(_is_year_or_month_precision).any()
        if has_precision_only_values:
            parsed = _parse_dates(original.where(temporal_values), date_format)
            invalid_mask = temporal_values & parsed.isna()
            reports.append(
                {
                    "column": name,
                    "values_converted": 0,
                    "invalid_value_count": int(invalid_mask.sum()),
                    "invalid_values": original.loc[invalid_mask].tolist(),
                    "preserved_year_or_year_month_count": int(
                        original.map(_is_year_or_month_precision).sum()
                    ),
                    "status": "preserved",
                    "reason": (
                        "Mixed date precision was retained to avoid assigning "
                        "fabricated day or month values."
                    ),
                }
            )
            continue
        if not temporal_values.any():
            reports.append(
                {
                    "column": name,
                    "values_converted": 0,
                    "invalid_value_count": 0,
                    "invalid_values": [],
                    "preserved_year_or_year_month_count": int(
                        original.map(_is_year_or_month_precision).sum()
                    ),
                    "status": "preserved",
                }
            )
            continue

        parsed = _parse_dates(original.where(temporal_values), date_format)
        invalid_mask = temporal_values & parsed.isna()
        cleaned[column] = parsed
        reports.append(
            {
                "column": name,
                "values_converted": int(parsed.notna().sum()),
                "invalid_value_count": int(invalid_mask.sum()),
                "invalid_values": original.loc[invalid_mask].tolist(),
                "preserved_year_or_year_month_count": int(
                    (~temporal_values & original.notna()).sum()
                ),
                "status": "partial" if invalid_mask.any() else "success",
            }
        )
    return cleaned, {"columns_processed": reports}


def identify_currency_columns(
    df: pd.DataFrame,
    api_metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """Find likely monetary fields using labels and explicit unit metadata."""
    candidates = [
        str(column)
        for column in df.columns
        if CURRENCY_NAME_PATTERN.search(str(column))
    ]
    currency_codes = set(CURRENCY_TOKENS.values())
    metadata = api_metadata or {}
    hints = [
        metadata.get("unit_measure"),
        metadata.get("UNIT_MEASURE"),
        metadata.get("unit_type"),
        metadata.get("UNIT_TYPE"),
    ]
    request_params = metadata.get("request_parameters", {})
    if isinstance(request_params, Mapping):
        hints.extend(
            [request_params.get("UNIT_MEASURE"), request_params.get("UNIT_TYPE")]
        )
    if any(str(hint).strip().upper() in currency_codes for hint in hints if hint):
        if "OBS_VALUE" in map(str, df.columns) and "OBS_VALUE" not in candidates:
            candidates.append("OBS_VALUE")
    if "UNIT_MEASURE" in map(str, df.columns):
        unit_column = next(col for col in df.columns if str(col) == "UNIT_MEASURE")
        units = {str(value).strip().upper() for value in df[unit_column] if not _is_missing(value)}
        if units and units.issubset(currency_codes):
            if "OBS_VALUE" in map(str, df.columns) and "OBS_VALUE" not in candidates:
                candidates.append("OBS_VALUE")
    return candidates


def standardise_currency_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    currency_code: str | None = None,
    api_metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize selected currency values and preserve a separate currency code.

    No exchange-rate calculations or conversion are performed.
    """
    normalized_code = currency_code.upper() if currency_code else None
    allowed_codes = set(CURRENCY_TOKENS.values())
    if normalized_code is not None and normalized_code not in allowed_codes:
        raise ValueError(f"Unsupported currency code: {currency_code!r}.")
    selected_columns = (
        list(columns)
        if columns is not None
        else identify_currency_columns(df, api_metadata)
    )
    unknown_columns = set(selected_columns).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown currency column(s): {sorted(unknown_columns)}")

    cleaned = df.copy(deep=True)
    reports = []
    for name in selected_columns:
        column = next(col for col in cleaned.columns if str(col) == name)
        amounts = []
        codes = []
        invalid = []
        for value in cleaned[column].tolist():
            if _is_missing(value):
                amounts.append(value)
                codes.append(normalized_code)
                continue
            if isinstance(value, bool):
                amounts.append(value)
                codes.append(normalized_code)
                invalid.append(value)
                continue
            if isinstance(value, Real):
                amounts.append(value)
                codes.append(normalized_code)
                continue
            if not isinstance(value, str):
                amounts.append(value)
                codes.append(normalized_code)
                invalid.append(value)
                continue

            text = value.strip()
            detected_code = None
            for token, code in sorted(
                CURRENCY_TOKENS.items(), key=lambda item: -len(item[0])
            ):
                if token.upper() in text.upper():
                    text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)
                    detected_code = code
            detected_code = detected_code or normalized_code
            text = text.strip().replace(" ", "")
            negative = text.startswith("(") and text.endswith(")")
            if negative:
                text = text[1:-1]
            if "," in text and "." in text:
                text = (
                    text.replace(".", "").replace(",", ".")
                    if text.rfind(",") > text.rfind(".")
                    else text.replace(",", "")
                )
            elif "," in text:
                parts = text.split(",")
                text = (
                    "".join(parts)
                    if len(parts[-1]) == 3 and all(part.isdigit() for part in parts)
                    else ".".join(parts)
                )
            text = re.sub(r"[^\d.eE+\-]", "", text)
            try:
                amount = float(text)
            except ValueError:
                amounts.append(value)
                codes.append(detected_code)
                invalid.append(value)
                continue
            amounts.append(-amount if negative else amount)
            codes.append(detected_code)

        cleaned[column] = amounts
        companion_column = f"{name}_currency"
        if any(code is not None for code in codes):
            suffix = 2
            while companion_column in cleaned.columns:
                companion_column = f"{name}_currency_{suffix}"
                suffix += 1
            cleaned[companion_column] = codes
        reports.append(
            {
                "column": name,
                "currency_code": normalized_code,
                "values_standardised": len(amounts) - len(invalid),
                "invalid_value_count": len(invalid),
                "invalid_values": invalid,
            }
        )
    return cleaned, {"columns_processed": reports}


def _is_identifier_column(column: Any) -> bool:
    return bool(IDENTIFIER_PATTERN.search(str(column)))


def detect_data_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Report each API observation column's existing Pandas type."""
    results = []
    for column in df.columns:
        series = df[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            detected = "category"
        elif pandas_types.is_bool_dtype(series.dtype):
            detected = "boolean"
        elif pandas_types.is_integer_dtype(series.dtype):
            detected = "integer"
        elif pandas_types.is_float_dtype(series.dtype):
            detected = "float"
        elif pandas_types.is_datetime64_any_dtype(series.dtype):
            detected = "datetime"
        elif pandas_types.is_string_dtype(series.dtype) and all(
            isinstance(value, str) or _is_missing(value) for value in series
        ):
            detected = "string"
        else:
            detected = "object"
        results.append(
            {
                "column_name": str(column),
                "data_type": detected,
                "pandas_dtype": str(series.dtype),
                "non_null_values": int(series.notna().sum()),
            }
        )
    return results


def convert_data_types(
    df: pd.DataFrame,
    conversions: Mapping[str, str],
    date_format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert explicitly requested columns with nullable dtypes and reports."""
    allowed = {"integer", "float", "string", "boolean", "datetime"}
    unknown_columns = set(conversions).difference(map(str, df.columns))
    if unknown_columns:
        raise KeyError(f"Unknown conversion column(s): {sorted(unknown_columns)}")
    invalid_targets = set(conversions.values()).difference(allowed)
    if invalid_targets:
        raise ValueError(f"Unsupported target type(s): {sorted(invalid_targets)}")

    cleaned = df.copy(deep=True)
    reports = []
    for name, target in conversions.items():
        column = next(col for col in cleaned.columns if str(col) == name)
        original = cleaned[column].copy()
        if target in {"integer", "float"} and _is_identifier_column(column):
            reports.append(
                {
                    "column": name,
                    "target_type": target,
                    "status": "skipped",
                    "reason": "Identifier-like columns are preserved.",
                    "failed_count": 0,
                    "failed_values": [],
                }
            )
            continue

        if target in {"integer", "float"}:
            numeric = pd.to_numeric(original, errors="coerce")
            invalid_mask = original.notna() & numeric.isna()
            if target == "integer":
                non_integral = numeric.notna() & (numeric % 1 != 0)
                invalid_mask |= non_integral
                numeric = numeric.mask(non_integral)
                converted = numeric.astype("Int64")
            else:
                converted = numeric.astype("Float64")
        elif target == "string":
            converted = original.astype("string")
            invalid_mask = pd.Series(False, index=original.index)
        elif target == "boolean":
            mapping = {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }
            converted = original.map(
                lambda value: (
                    pd.NA
                    if _is_missing(value)
                    else value
                    if isinstance(value, bool)
                    else mapping.get(str(value).strip().lower(), pd.NA)
                )
            ).astype("boolean")
            invalid_mask = original.notna() & converted.isna()
        else:
            # Avoid replacing coarse time periods with fabricated dates.
            temporal = original.map(
                lambda value: _has_full_date_precision(value, date_format)
            )
            coarse = original.map(_is_year_or_month_precision)
            if coarse.any():
                reports.append(
                    {
                        "column": name,
                        "target_type": target,
                        "status": "skipped",
                        "converted_count": 0,
                        "preserved_year_or_year_month_count": int(coarse.sum()),
                        "failed_count": 0,
                        "failed_values": [],
                    }
                )
                continue
            converted = _parse_dates(original.where(temporal), date_format)
            invalid_mask = temporal & converted.isna()
            preserved = (~temporal) & original.notna()
            reports.append(
                {
                    "column": name,
                    "target_type": target,
                    "status": "partial" if invalid_mask.any() else "success",
                    "converted_count": int(converted.notna().sum()),
                    "preserved_year_or_year_month_count": int(preserved.sum()),
                    "failed_count": int(invalid_mask.sum()),
                    "failed_values": original.loc[invalid_mask].tolist(),
                }
            )
            cleaned[column] = converted
            continue

        failed_values = original.loc[invalid_mask].tolist()
        cleaned[column] = converted
        reports.append(
            {
                "column": name,
                "target_type": target,
                "status": "partial" if failed_values else "success",
                "converted_count": int(converted.notna().sum()),
                "failed_count": len(failed_values),
                "failed_values": failed_values,
            }
        )
    return cleaned, {"conversions": reports}


def _rename_config_columns(
    columns: Sequence[str] | None,
    name_mapping: Mapping[str, str],
) -> list[str] | None:
    if columns is None:
        return None
    return [name_mapping.get(name, name) for name in columns]


def clean_api_dataframe(
    df: pd.DataFrame,
    api_metadata: Mapping[str, Any] | None = None,
    *,
    dataset_name: str = "api_dataset",
    remove_duplicates: bool = False,
    duplicate_subset: Sequence[str] | None = None,
    handle_missing: bool = False,
    missing_strategy: str = "leave",
    column_missing_strategies: Mapping[str, str] | None = None,
    missing_fill_value: Any = None,
    column_fill_values: Mapping[str, Any] | None = None,
    standardise_names: bool = False,
    standardise_dates: bool = False,
    date_columns: Sequence[str] | None = None,
    date_format: str | None = None,
    standardise_currencies: bool = False,
    currency_columns: Sequence[str] | None = None,
    currency_code: str | None = None,
    correct_data_types: bool = False,
    type_conversions: Mapping[str, str] | None = None,
    api_response: Any = None,
) -> dict[str, Any]:
    """Apply selected API observation cleaning steps to a separate DataFrame copy."""
    metadata = deepcopy(dict(api_metadata)) if api_metadata is not None else {}
    cleaned = df.copy(deep=True)
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "api_provider": metadata.get("provider"),
        "api_name": metadata.get("api_name"),
        "api_endpoint": metadata.get("api_endpoint"),
        "DATABASE_ID": metadata.get("database_id"),
        "INDICATOR": metadata.get("indicator_id"),
        "REF_AREA": metadata.get("country_reference_area"),
        "original_row_count": len(df.index),
        "original_column_count": len(df.columns),
        "duplicate_records_detected": 0,
        "duplicate_records_removed": 0,
        "missing_values_before": count_missing_values(df),
        "missing_values_after": count_missing_values(df),
        "missing_value_percentages_before": missing_value_percentages(df),
        "missing_value_percentages_after": missing_value_percentages(df),
        "column_names_changed": [],
        "column_name_mapping": [],
        "date_values_converted": [],
        "currency_values_standardised": [],
        "data_type_conversions_performed": [],
        "failed_conversions": [],
        "cleaning_status": "success",
    }

    if remove_duplicates:
        selected_subset = list(duplicate_subset) if duplicate_subset else None
        cleaned, duplicate_report = remove_duplicate_records(cleaned, selected_subset)
        report["duplicate_records_detected"] = duplicate_report[
            "duplicate_records_detected"
        ]
        report["duplicate_records_removed"] = duplicate_report[
            "duplicate_records_removed"
        ]
        report["duplicate_key_columns"] = duplicate_report["duplicate_key_columns"]

    if handle_missing:
        per_column_strategies = {
            name: value
            for name, value in (column_missing_strategies or {}).items()
        }
        per_column_fill = {
            name: value
            for name, value in (column_fill_values or {}).items()
        }
        cleaned, missing_report = handle_missing_values(
            cleaned,
            missing_strategy,
            column_strategies=per_column_strategies,
            fill_value=missing_fill_value,
            column_fill_values=per_column_fill,
        )
        report["missing_values_before"] = missing_report["missing_values_before"]
        report["missing_values_after"] = missing_report["missing_values_after"]
        report["missing_value_percentages_before"] = missing_report[
            "missing_value_percentages_before"
        ]
        report["missing_value_percentages_after"] = missing_report[
            "missing_value_percentages_after"
        ]
        report["missing_strategy"] = missing_report["strategy"]
        report["column_missing_strategies"] = per_column_strategies

    name_mapping: dict[str, str] = {}
    if standardise_names:
        cleaned, names_report = standardise_column_names(cleaned)
        report["column_names_changed"] = names_report["column_names_changed"]
        report["column_name_mapping"] = names_report["column_name_mapping"]
        name_mapping = {
            str(item["original"]): str(item["standardised"])
            for item in names_report["column_name_mapping"]
        }

    if standardise_dates:
        selected_dates = _rename_config_columns(date_columns, name_mapping)
        cleaned, date_report = standardise_date_columns(
            cleaned, selected_dates, date_format
        )
        report["date_values_converted"] = date_report["columns_processed"]

    if standardise_currencies:
        selected_currency_columns = _rename_config_columns(
            currency_columns, name_mapping
        )
        cleaned, currency_report = standardise_currency_columns(
            cleaned,
            selected_currency_columns,
            currency_code,
            metadata,
        )
        report["currency_values_standardised"] = currency_report[
            "columns_processed"
        ]

    if correct_data_types:
        conversions = {
            name_mapping.get(name, name): target
            for name, target in (type_conversions or {}).items()
        }
        cleaned, conversion_report = convert_data_types(
            cleaned, conversions, date_format
        )
        report["data_type_conversions_performed"] = conversion_report["conversions"]
        report["failed_conversions"] = [
            item
            for item in conversion_report["conversions"]
            if item.get("failed_count", 0)
        ]

    report["missing_values_after"] = count_missing_values(cleaned)
    report["missing_value_percentages_after"] = missing_value_percentages(cleaned)
    report["final_row_count"] = len(cleaned.index)
    report["final_column_count"] = len(cleaned.columns)
    report["final_column_names"] = list(cleaned.columns)
    failed_dates_or_currency = any(
        item.get("invalid_value_count", 0)
        for item in (
            report["date_values_converted"]
            + report["currency_values_standardised"]
        )
    )
    report["cleaning_status"] = (
        "partial" if report["failed_conversions"] or failed_dates_or_currency else "success"
    )
    return {
        "data": cleaned,
        "report": report,
        "api_metadata": metadata,
        "api_response": deepcopy(api_response),
    }


def _dataset_parts(
    dataset: Any,
) -> tuple[pd.DataFrame, Mapping[str, Any] | None, Any, str | None]:
    if isinstance(dataset, pd.DataFrame):
        return dataset, None, None, None
    if isinstance(dataset, Mapping):
        metadata = dataset.get("metadata", dataset.get("api_metadata"))
        response = dataset.get("api_response")
        dataframe = dataset.get("data")
        if dataframe is None:
            raw_response = response
            if raw_response is None and isinstance(dataset.get("value"), list):
                raw_response = dataset
            if isinstance(raw_response, Mapping):
                records = raw_response.get("value", [])
                if not isinstance(records, list):
                    raise ValueError("API response 'value' must be an array.")
                dataframe = pd.DataFrame(records, dtype=object)
                response = raw_response
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("API dataset must contain a Pandas DataFrame or value array.")
        name = (
            str(metadata.get("dataset_id") or metadata.get("source"))
            if isinstance(metadata, Mapping)
            and (metadata.get("dataset_id") or metadata.get("source"))
            else None
        )
        return dataframe, metadata if isinstance(metadata, Mapping) else None, response, name
    if isinstance(dataset, list):
        if not all(isinstance(item, Mapping) for item in dataset):
            raise TypeError("API observation arrays must contain JSON objects.")
        return pd.DataFrame(dataset, dtype=object), None, dataset, None
    raise TypeError(
        "API dataset must be a DataFrame, API extraction result, or observation array."
    )


def clean_multiple_api_datasets(
    datasets: Mapping[str, Any] | Sequence[Any],
    **cleaning_options: Any,
) -> dict[str, dict[str, Any]]:
    """Clean independent API datasets and retain each dataset's identifiers."""
    items = list(datasets.items()) if isinstance(datasets, Mapping) else list(
        enumerate(datasets)
    )
    results: dict[str, dict[str, Any]] = {}
    for identifier, dataset in items:
        if isinstance(dataset, Mapping):
            metadata = dataset.get("metadata", dataset.get("api_metadata", {}))
            if isinstance(metadata, Mapping) and metadata.get("status") == "error":
                continue
        dataframe, metadata, response, name = _dataset_parts(dataset)
        dataset_name = name or str(identifier)
        result = clean_api_dataframe(
            dataframe,
            api_metadata=metadata,
            dataset_name=dataset_name,
            api_response=response,
            **cleaning_options,
        )
        result["dataset_id"] = str(identifier)
        results[str(identifier)] = result
    return results


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe_df = df.astype(object).where(pd.notna(df), None)
    records = safe_df.to_dict(orient="records")
    return json.loads(json.dumps(records, default=str, ensure_ascii=False))


def save_cleaned_api_dataset(
    result: Mapping[str, Any],
    output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY,
) -> dict[str, str]:
    """Write cleaned observations, report, source metadata, and run metadata.

    Unique run directories keep processed files separate from the immutable
    ``data/raw/api`` archive.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", Path(str(result.get("report", {}).get("dataset_name", "api_dataset"))).stem
    ).strip("._") or "api_dataset"
    run_directory = directory / f"{safe_name}_{uuid4().hex[:8]}"
    run_directory.mkdir()

    data = result.get("data")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Cleaning result must contain a DataFrame under 'data'.")
    csv_path = run_directory / "cleaned_data.csv"
    json_path = run_directory / "cleaned_data.json"
    data.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(_json_safe_records(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = dict(result.get("report", {}))
    report["saved_at"] = timestamp
    report_path = run_directory / "cleaning_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    source_metadata = dict(result.get("api_metadata") or {})
    source_metadata_path = run_directory / "api_metadata.json"
    source_metadata_path.write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    execution_metadata = {
        "dataset_name": result.get("report", {}).get("dataset_name"),
        "saved_at": timestamp,
        "cleaning_status": result.get("report", {}).get("cleaning_status"),
        "cleaning_report": str(report_path),
        "api_metadata": str(source_metadata_path),
        "cleaned_csv": str(csv_path),
        "cleaned_json": str(json_path),
    }
    execution_path = run_directory / "execution_metadata.json"
    execution_path.write_text(
        json.dumps(execution_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "directory": str(run_directory),
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(report_path),
        "api_metadata": str(source_metadata_path),
        "execution_metadata": str(execution_path),
    }
