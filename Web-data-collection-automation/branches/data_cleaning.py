"""Clean and standardise the raw country JSON for downstream use."""

import argparse
import json
import logging
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

DEFAULT_INPUT_PATH = Path("data/raw/countries_raw.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/countries_cleaned.json")
OUTPUT_FIELDS = ["country_name", "capital", "population", "area_km2"]
FIELD_ALIASES = {
    "country_name": ("country_name", "country"),
    "capital": ("capital",),
    "population": ("population",),
    "area_km2": ("area_km2", "area_sq_km"),
}
MISSING_TEXT_VALUES = {"", "none", "null", "nan", "n/a"}

logger = logging.getLogger(__name__)


def load_raw_data(input_path=DEFAULT_INPUT_PATH):
    """Load a JSON array of country records from ``input_path``."""
    path = Path(input_path)
    try:
        with path.open("r", encoding="utf-8") as raw_file:
            records = json.load(raw_file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read valid JSON from %s", path)
        raise

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def standardise_text(value):
    """Normalize Unicode and whitespace while preserving the original casing."""
    if value is None or isinstance(value, (list, dict, bool)):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return None if text.casefold() in MISSING_TEXT_VALUES else text


def _record_to_output_fields(record):
    """Map supported input field names to the output schema."""
    result = {}
    for output_field, aliases in FIELD_ALIASES.items():
        value = next((record[key] for key in aliases if key in record), None)
        result[output_field] = value
    return result


def _convert_numeric_values(values, field):
    """Convert a column to numbers, preferring Pandas when it is available."""
    normalized = [
        None if value is None else value.replace(",", "")
        for value in (standardise_text(item) for item in values)
    ]

    if field == "area_km2" and pd is not None:
        converted = pd.to_numeric(pd.Series(normalized), errors="coerce")
        return [
            None if pd.isna(value) or not math.isfinite(float(value)) else float(value)
            for value in converted
        ]

    converted = []
    for value in normalized:
        try:
            number = Decimal(value) if value is not None else None
        except InvalidOperation:
            number = None
        if number is None or not number.is_finite():
            converted.append(None)
        elif field == "population" and number != number.to_integral_value():
            converted.append(None)
        elif field == "population":
            converted.append(int(number))
        else:
            converted.append(float(number))
    return converted


def clean_records(records):
    """Clean records and return ``(cleaned_records, cleaning_summary)``.

    Duplicate detection uses the normalized country name, case-insensitively;
    when duplicates occur, the first record is retained.
    """
    if pd is None:
        logger.warning("Pandas is unavailable; using standard-library numeric parsing")

    valid_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            logger.warning("Skipping non-object record at index %d", index)
            continue
        cleaned_record = _record_to_output_fields(record)
        cleaned_record["country_name"] = standardise_text(
            cleaned_record["country_name"]
        )
        cleaned_record["capital"] = standardise_text(cleaned_record["capital"])
        if cleaned_record["country_name"] is None:
            logger.warning("Skipping record at index %d without a country name", index)
            continue
        valid_records.append(cleaned_record)

    country_keys = set()
    cleaned_records = []
    duplicates_removed = 0
    for record in valid_records:
        key = record["country_name"].casefold()
        if key in country_keys:
            duplicates_removed += 1
            continue
        country_keys.add(key)
        cleaned_records.append(record)

    for field in ("population", "area_km2"):
        converted = _convert_numeric_values(
            [record[field] for record in cleaned_records], field
        )
        for record, value in zip(cleaned_records, converted):
            record[field] = value

    missing_values = {
        field: sum(record[field] is None for record in cleaned_records)
        for field in OUTPUT_FIELDS
    }

    summary = {
        "total_records": len(records),
        "duplicates_removed": duplicates_removed,
        "missing_values": missing_values,
        "cleaned_records": len(cleaned_records),
    }
    return cleaned_records, summary


def write_clean_data(records, output_path=DEFAULT_OUTPUT_PATH):
    """Write cleaned records as UTF-8 JSON, creating the output directory."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    except OSError:
        logger.exception("Could not write cleaned data to %s", path)
        raise


def clean_country_data(
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Run the full load, clean, summarize, and write workflow."""
    raw_records = load_raw_data(input_path)
    cleaned_records, summary = clean_records(raw_records)
    write_clean_data(cleaned_records, output_path)
    logger.info("Cleaning summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    """Parse optional file paths and run the cleaner as a command-line script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_country_data(args.input, args.output)


if __name__ == "__main__":
    main()
