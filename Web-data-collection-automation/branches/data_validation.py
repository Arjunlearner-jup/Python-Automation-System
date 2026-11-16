"""Validate cleaned country records and write validation results."""

import argparse
import json
import logging
import math
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/processed/countries_cleaned.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/validated")
EXPECTED_FIELDS = ("country_name", "capital", "population", "area_km2")
TEXT_FIELDS = ("country_name", "capital")
NUMERIC_FIELDS = ("population", "area_km2")
RULES = (
    "country_name_not_empty",
    "capital_not_empty",
    "population_positive_integer",
    "area_positive_numeric",
    "country_name_unique",
    "expected_data_types",
    "expected_record_structure",
)

logger = logging.getLogger(__name__)


def load_records(input_path=DEFAULT_INPUT_PATH):
    """Read a JSON array from disk without changing the source file."""
    path = Path(input_path)
    try:
        with path.open("r", encoding="utf-8") as input_file:
            records = json.load(input_file, parse_constant=lambda value: value)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read valid JSON from %s", path)
        raise

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def _validate_record(record, index, seen_country_names):
    """Return one annotated record and its rule and field-level failures."""
    failures = []
    missing_values = []
    invalid_values = []

    if not isinstance(record, dict):
        return (
            {
                "record": record,
                "status": "FAIL",
                "reasons": ["expected_record_structure: record must be a JSON object"],
            },
            ["expected_record_structure"],
            missing_values,
            invalid_values,
        )

    expected_field_set = set(EXPECTED_FIELDS)
    missing_keys = expected_field_set - record.keys()
    extra_keys = record.keys() - expected_field_set
    if missing_keys or extra_keys:
        details = []
        if missing_keys:
            details.append(f"missing fields: {', '.join(sorted(missing_keys))}")
        if extra_keys:
            details.append(f"unexpected fields: {', '.join(sorted(extra_keys))}")
        failures.append(
            ("expected_record_structure", "; ".join(details))
        )

    type_errors = []
    for field in TEXT_FIELDS:
        value = record.get(field)
        if field not in record or value is None or (
            isinstance(value, str) and not value.strip()
        ):
            rule = f"{field}_not_empty"
            failures.append((rule, f"{field} is missing or empty"))
            missing_values.append(field)
        elif not isinstance(value, str):
            type_errors.append(f"{field} must be a string")
            invalid_values.append(field)

    population = record.get("population")
    if "population" not in record or population is None:
        failures.append(
            ("population_positive_integer", "population is missing")
        )
        missing_values.append("population")
    elif isinstance(population, bool) or not isinstance(population, int):
        failures.append(
            (
                "population_positive_integer",
                "population must be a positive integer",
            )
        )
        type_errors.append("population must be an integer")
        invalid_values.append("population")
    elif population <= 0:
        failures.append(
            (
                "population_positive_integer",
                "population must be greater than zero",
            )
        )
        invalid_values.append("population")

    area = record.get("area_km2")
    if "area_km2" not in record or area is None:
        failures.append(("area_positive_numeric", "area_km2 is missing"))
        missing_values.append("area_km2")
    elif isinstance(area, bool) or not isinstance(area, (int, float)):
        failures.append(
            ("area_positive_numeric", "area_km2 must be a positive number")
        )
        type_errors.append("area_km2 must be numeric")
        invalid_values.append("area_km2")
    elif (isinstance(area, float) and not math.isfinite(area)) or area <= 0:
        failures.append(
            (
                "area_positive_numeric",
                "area_km2 must be finite and greater than zero",
            )
        )
        invalid_values.append("area_km2")

    country_name = record.get("country_name")
    if isinstance(country_name, str) and country_name.strip():
        country_key = country_name.strip().casefold()
        if country_key in seen_country_names:
            failures.append(
                ("country_name_unique", "country_name duplicates an earlier record")
            )
            invalid_values.append("country_name")
        else:
            seen_country_names.add(country_key)

    if type_errors:
        failures.append(("expected_data_types", "; ".join(type_errors)))

    validation_failures = [rule for rule, _ in failures]
    annotated_record = dict(record)
    annotated_record["status"] = "FAIL" if failures else "PASS"
    annotated_record["reasons"] = [
        f"{rule}: {reason}" for rule, reason in failures
    ]

    logger.debug("Validated record %d: %s", index, annotated_record["status"])
    return (
        annotated_record,
        validation_failures,
        missing_values,
        invalid_values,
    )


def validate_records(records):
    """Validate all records and return valid, rejected, and report data."""
    seen_country_names = set()
    valid_records = []
    rejected_records = []
    failure_counts = {rule: 0 for rule in RULES}
    missing_value_counts = {field: 0 for field in EXPECTED_FIELDS}
    invalid_value_counts = {field: 0 for field in EXPECTED_FIELDS}

    for index, record in enumerate(records):
        annotated, failures, missing, invalid = _validate_record(
            record, index, seen_country_names
        )
        for rule in failures:
            failure_counts[rule] += 1
        for field in missing:
            missing_value_counts[field] += 1
        for field in invalid:
            invalid_value_counts[field] += 1

        if annotated["status"] == "PASS":
            valid_records.append(annotated)
        else:
            rejected_records.append(annotated)

    total_records = len(records)
    total_valid = len(valid_records)
    total_invalid = len(rejected_records)
    report = {
        "total_records_processed": total_records,
        "total_valid_records": total_valid,
        "total_invalid_records": total_invalid,
        "validation_success_percentage": (
            round(total_valid / total_records * 100, 2)
            if total_records
            else 0.0
        ),
        "validation_failures_by_rule": failure_counts,
        "missing_values_by_field": missing_value_counts,
        "invalid_values_by_field": invalid_value_counts,
    }
    return valid_records, rejected_records, report


def _write_json(data, output_path):
    """Write JSON data to the requested path, creating parent directories."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    except OSError:
        logger.exception("Could not write validation output to %s", path)
        raise


def validate_country_data(
    input_path=DEFAULT_INPUT_PATH,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Validate the input and write valid, rejected, and summary JSON files."""
    source_path = Path(input_path).resolve()
    output_dir = Path(output_directory)
    output_paths = {
        "valid": output_dir / "countries_validated.json",
        "rejected": output_dir / "countries_rejected.json",
        "report": output_dir / "validation_report.json",
    }
    if source_path in {path.resolve() for path in output_paths.values()}:
        raise ValueError("Validation outputs must not overwrite the input file")
    if len({path.resolve() for path in output_paths.values()}) != len(output_paths):
        raise ValueError("Validation output paths must be distinct")

    records = load_records(source_path)
    valid_records, rejected_records, report = validate_records(records)
    _write_json(valid_records, output_paths["valid"])
    _write_json(rejected_records, output_paths["rejected"])
    _write_json(report, output_paths["report"])
    logger.info("Validation report: %s", json.dumps(report, ensure_ascii=False))
    return report


def main():
    """Run validation from the command line with optional input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_country_data(args.input, args.output_directory)


if __name__ == "__main__":
    main()
