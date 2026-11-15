import json
import os
import re
import uuid
from typing import Any

import mysql.connector
import streamlit as st

from Datacleaning import (
    _display_field_name,
    clean_invoice,
    invoice_to_csv,
)
from Datastandardisation import SUPPORTED_FALLBACK_CURRENCIES


def save_invoice_to_mysql(
    cleaned_invoice: dict[str, Any],
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError(
            "Database name must contain only letters, numbers, and underscores."
        )

    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    finally:
        if connection.is_connected():
            connection.close()

    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaned_invoice_data (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                submission_id CHAR(36) NOT NULL,
                section VARCHAR(64) NOT NULL,
                line_item INT NULL,
                `data` VARCHAR(255) NOT NULL,
                `value` LONGTEXT NULL,
                INDEX idx_submission_id (submission_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )

        submission_id = str(uuid.uuid4())
        rows = invoice_to_rows(cleaned_invoice, submission_id)
        cursor.executemany(
            """
            INSERT INTO cleaned_invoice_data
                (submission_id, section, line_item, `data`, `value`)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
        connection.commit()
        return submission_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def invoice_to_rows(
    invoice: dict[str, Any],
    submission_id: str,
) -> list[tuple[str, str, int | None, str, str | None]]:
    def database_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    rows = []
    for field_name, value in invoice.items():
        if field_name not in {"line_items", "original_values"}:
            rows.append(
                (
                    submission_id,
                    "Invoice details",
                    None,
                    _display_field_name(field_name),
                    database_value(value),
                )
            )

    for item_index, item in enumerate(invoice.get("line_items", []), start=1):
        for field_name, value in item.items():
            rows.append(
                (
                    submission_id,
                    "Line item",
                    item_index,
                    _display_field_name(field_name),
                    database_value(value),
                )
            )

    if not rows:
        raise ValueError("The cleaned invoice has no fields to save.")
    return rows


def main() -> None:
    st.set_page_config(page_title="Save cleaned invoice to MySQL", page_icon="📄")
    st.title("Clean and save invoice to MySQL")
    st.write(
        "Upload extracted invoice JSON or paste it below. The cleaned invoice "
        "details and line items will be saved to MySQL with field names and "
        "values in separate columns. You can also download the cleaned CSV."
    )

    with st.sidebar:
        st.header("MySQL connection")
        host = st.text_input("Host", value=os.getenv("MYSQL_HOST", "localhost"))
        port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=int(os.getenv("MYSQL_PORT", "3306")),
            step=1,
        )
        user = st.text_input("Username", value=os.getenv("MYSQL_USER", "root"))
        password = st.text_input(
            "Password",
            value=os.getenv("MYSQL_PASSWORD", ""),
            type="password",
            help="Set the MYSQL_PASSWORD environment variable to prefill this securely.",
        )
        database = st.text_input(
            "Database name",
            value=os.getenv("MYSQL_DATABASE", "invoice_processing"),
        )

    uploaded_file = st.file_uploader("Upload extracted invoice JSON", type=["json"])
    pasted_json = st.text_area("Or paste extracted JSON", height=240)
    fallback_currency = st.selectbox(
        "Fallback currency when no code or symbol is present",
        SUPPORTED_FALLBACK_CURRENCIES,
        index=0,
    )
    output_date_format = st.selectbox(
        "Cleaned date output format",
        ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"),
        format_func=lambda value: {
            "%m/%d/%Y": "MM/DD/YYYY",
            "%d/%m/%Y": "DD/MM/YYYY",
            "%Y-%m-%d": "YYYY-MM-DD (ISO)",
        }[value],
    )

    if not st.button("Clean and save to MySQL"):
        return
    if uploaded_file is None and not pasted_json.strip():
        st.error("Upload a JSON file or paste invoice JSON to continue.")
        return
    if not host.strip() or not user.strip():
        st.error("Enter the MySQL host and username.")
        return

    try:
        raw_json = (
            uploaded_file.getvalue().decode("utf-8-sig")
            if uploaded_file is not None
            else pasted_json
        )
        invoice = json.loads(raw_json)
        cleaned = clean_invoice(
            invoice,
            fallback_currency=fallback_currency,
            output_date_format=output_date_format,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not clean invoice data: {error}")
        return

    st.subheader("Cleaned invoice details")
    if cleaned["currency_code"] != "USD":
        st.warning(
            f"Currency is {cleaned['currency_code']}, not USD. "
            "Monetary amount values are hidden and shown as null."
        )
    invoice_fields = [
        {"Field": _display_field_name(field_name), "Value": value}
        for field_name, value in cleaned.items()
        if field_name not in {"line_items", "original_values"}
    ]
    st.table(invoice_fields)

    st.subheader("Line items")
    if cleaned.get("line_items"):
        line_items = [
            {
                _display_field_name(field_name): value
                for field_name, value in item.items()
            }
            for item in cleaned["line_items"]
        ]
        st.table(line_items)
    else:
        st.info("No line items were found in the invoice data.")

    original_values = cleaned.get("original_values")
    if isinstance(original_values, dict):
        with st.expander("Original extracted values"):
            st.table(
                [
                    {
                        "Field": _display_field_name(field_name),
                        "Value": value,
                    }
                    for field_name, value in original_values.items()
                ]
            )

    st.download_button(
        "Download cleaned CSV",
        data=invoice_to_csv(cleaned),
        file_name="cleaned_invoice.csv",
        mime="text/csv",
    )

    try:
        import mysql.connector

        record_id = save_invoice_to_mysql(
            cleaned,
            host=host.strip(),
            port=int(port),
            user=user.strip(),
            password=password,
            database=database.strip(),
        )
    except ValueError as error:
        st.error(f"Could not save invoice to MySQL: {error}")
        return
    except mysql.connector.Error as error:
        st.error(
            "Could not save invoice to MySQL. Check that the MySQL server is "
            f"reachable and the account has database/table permissions: {error}"
        )
        return

    st.success(
        f"Invoice data saved as separate field/value rows in "
        f"`{database}.cleaned_invoice_data` (submission `{record_id}`)."
    )


if __name__ == "__main__":
    main()