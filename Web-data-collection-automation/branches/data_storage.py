"""Export validated country data to CSV, Excel, and a MySQL database."""

import argparse
import csv
import json
import logging
import os
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/validated/countries_validated.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/exports")
DATABASE_NAME = "web_data_automation"
TABLE_NAME = "countries"
FIELDS = ("country_name", "capital", "population", "area_km2")
MYSQL_BIGINT_MAX = 9_223_372_036_854_775_807
MYSQL_AREA_MAX = Decimal("9999999999999.99")
AREA_QUANTUM = Decimal("0.01")

logger = logging.getLogger(__name__)


def load_validated_records(input_path=DEFAULT_INPUT_PATH):
    """Load and check PASS records, including field types and SQL limits."""
    path = Path(input_path)
    try:
        with path.open("r", encoding="utf-8") as input_file:
            records = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read validated JSON from %s", path)
        raise

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")

    checked_records = []
    seen_names = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object")
        if record.get("status") != "PASS":
            raise ValueError(f"Record {index} is not marked PASS")
        missing_fields = set(FIELDS) - record.keys()
        if missing_fields:
            raise ValueError(
                f"Record {index} is missing fields: {', '.join(sorted(missing_fields))}"
            )

        country_name = record["country_name"]
        capital = record["capital"]
        population = record["population"]
        area = record["area_km2"]
        if (
            not isinstance(country_name, str)
            or not country_name.strip()
            or len(country_name) > 150
        ):
            raise ValueError(f"Record {index} has an invalid country_name")
        if (
            not isinstance(capital, str)
            or not capital.strip()
            or len(capital) > 150
        ):
            raise ValueError(f"Record {index} has an invalid capital")
        if isinstance(population, bool) or not isinstance(population, int):
            raise ValueError(f"Record {index} population must be an integer")
        if population <= 0 or population > MYSQL_BIGINT_MAX:
            raise ValueError(f"Record {index} population is outside MySQL BIGINT limits")
        if isinstance(area, bool) or not isinstance(area, (int, float)):
            raise ValueError(f"Record {index} area_km2 must be numeric")
        try:
            area_decimal = Decimal(str(area))
            stored_area = area_decimal.quantize(AREA_QUANTUM, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            raise ValueError(f"Record {index} area_km2 is not a valid decimal") from None
        if not area_decimal.is_finite() or stored_area <= 0 or stored_area > MYSQL_AREA_MAX:
            raise ValueError(f"Record {index} area_km2 is outside DECIMAL(15,2) limits")

        country_key = country_name.strip().casefold()
        if country_key in seen_names:
            raise ValueError(f"Record {index} duplicates country_name {country_name!r}")
        seen_names.add(country_key)

        checked_records.append(
            {
                "country_name": country_name,
                "capital": capital,
                "population": population,
                "area_km2": float(area_decimal),
            }
        )

    logger.info("Loaded %d validated country records from %s", len(checked_records), path)
    return checked_records


def export_to_csv(records, output_path):
    """Write all four data fields as CSV without an index column."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
    except OSError:
        logger.exception("Could not write CSV export to %s", path)
        raise
    logger.info("Exported %d records to CSV: %s", len(records), path)
    return len(records)


def export_to_excel(records, output_path):
    """Write an Excel workbook with a Countries sheet and useful formats."""
    try:
        from openpyxl import Workbook
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.exception("Excel export requires openpyxl; install it with pip")
        raise

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()
        worksheet = workbook.create_sheet("Countries")
        del workbook["Sheet"]
        worksheet.append(FIELDS)
        for record in records:
            worksheet.append([record[field] for field in FIELDS])

        for row in worksheet.iter_rows(min_row=2):
            row[2].number_format = "0"
            row[3].number_format = "#,##0.00"

        for column_index, field in enumerate(FIELDS, start=1):
            values = [field] + [
                "" if record[field] is None else str(record[field])
                for record in records
            ]
            width = max(len(value) for value in values) + 2
            worksheet.column_dimensions[get_column_letter(column_index)].width = min(
                max(width, 12), 40
            )

        workbook.save(path)
    except OSError:
        logger.exception("Could not write Excel export to %s", path)
        raise
    logger.info("Exported %d records to Excel: %s", len(records), path)
    return len(records)


def _mysql_connection_settings():
    """Read MySQL connection settings from environment variables."""
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "Set MYSQL_USER and MYSQL_PASSWORD environment variables before "
            "running the MySQL export"
        )
    try:
        port = int(os.getenv("MYSQL_PORT", "3306"))
    except ValueError as error:
        raise ValueError("MYSQL_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("MYSQL_PORT must be between 1 and 65535")
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": port,
        "user": user,
        "password": password,
        "database": DATABASE_NAME,
    }


def export_to_mysql(records):
    """Create the database/table if needed and transactionally upsert records."""
    try:
        import mysql.connector
    except ImportError:
        logger.exception(
            "MySQL export requires mysql-connector-python; install it with pip"
        )
        raise

    settings = _mysql_connection_settings()
    connection = None
    try:
        connection = mysql.connector.connect(**settings)
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    country_name VARCHAR(150) NOT NULL,
                    capital VARCHAR(150) NOT NULL,
                    population BIGINT NOT NULL,
                    area_km2 DECIMAL(15,2) NOT NULL,
                    UNIQUE KEY uq_countries_country_name (country_name)
                ) ENGINE=InnoDB
                """
            )
            cursor.execute(
                f"SHOW INDEX FROM `{TABLE_NAME}` "
                "WHERE Key_name = 'uq_countries_country_name'"
            )
            if not cursor.fetchall():
                cursor.execute(
                    f"ALTER TABLE `{TABLE_NAME}` "
                    "ADD UNIQUE KEY uq_countries_country_name (country_name)"
                )
            cursor.execute("START TRANSACTION")
            insert_query = f"""
                INSERT INTO `{TABLE_NAME}`
                    (country_name, capital, population, area_km2)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    capital = VALUES(capital),
                    population = VALUES(population),
                    area_km2 = VALUES(area_km2)
            """
            values = [
                (
                    record["country_name"],
                    record["capital"],
                    record["population"],
                    Decimal(str(record["area_km2"])).quantize(
                        AREA_QUANTUM, rounding=ROUND_HALF_UP
                    ),
                )
                for record in records
            ]
            if values:
                cursor.executemany(insert_query, values)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
    except mysql.connector.Error:
        logger.exception("MySQL storage failed; transaction was rolled back")
        raise
    finally:
        if connection is not None:
            connection.close()

    logger.info("Upserted %d records into %s.%s", len(records), DATABASE_NAME, TABLE_NAME)
    return len(records)


def _write_summary(summary, output_path):
    """Persist the per-destination storage counts as JSON."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
            output_file.write("\n")
    except OSError:
        logger.exception("Could not write storage summary to %s", path)
        raise


def store_country_data(
    input_path=DEFAULT_INPUT_PATH,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Export source records to CSV, Excel, and MySQL and report record counts."""
    source_path = Path(input_path).resolve()
    output_dir = Path(output_directory)
    output_paths = {
        "csv": output_dir / "countries.csv",
        "excel": output_dir / "countries.xlsx",
        "summary": output_dir / "storage_summary.json",
    }
    if source_path in {path.resolve() for path in output_paths.values()}:
        raise ValueError("Storage outputs must not overwrite the validated input")
    if len({path.resolve() for path in output_paths.values()}) != len(output_paths):
        raise ValueError("Storage output paths must be distinct")

    records = load_validated_records(source_path)
    summary: dict[str, int | str] = {
        "records_exported_to_csv": export_to_csv(records, output_paths["csv"]),
        "records_exported_to_excel": export_to_excel(records, output_paths["excel"]),
        "records_upserted_to_mysql": 0,
        "mysql_status": "pending",
    }
    try:
        summary["records_upserted_to_mysql"] = export_to_mysql(records)
        summary["mysql_status"] = "success"
    except Exception as error:
        summary["mysql_status"] = "failed"
        summary["mysql_error_type"] = type(error).__name__
        _write_summary(summary, output_paths["summary"])
        logger.exception(
            "Storage summary (MySQL failed): %s",
            json.dumps(summary, ensure_ascii=False),
        )
        raise

    _write_summary(summary, output_paths["summary"])
    logger.info("Storage summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    """Run all storage exports with optional input and output-directory paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        store_country_data(args.input, args.output_directory)
    except Exception as error:
        logger.error("Data storage failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
