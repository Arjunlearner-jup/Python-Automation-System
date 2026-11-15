import json
import re
from typing import Any

import streamlit as st

from Datastandardisation import (
    SUPPORTED_FALLBACK_CURRENCIES,
    standardize_invoice,
)


def _clean_fields(value: Any, field_name: str = "") -> Any:
    if field_name == "original_values":
        return value
    if isinstance(value, dict):
        return {
            key: _clean_fields(child, key)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_clean_fields(child, field_name) for child in value]
    if not isinstance(value, str):
        return value

    if field_name == "seller_name":
        return re.sub(r"\s+", " ", value.replace("-", " ")).strip() or None
    if field_name in {"seller_tax_id", "client_tax_id"}:
        return re.sub(r"\s+", "", value) or None
    return value


def clean_invoice(
    invoice: dict[str, Any],
    fallback_currency: str = "USD",
    output_date_format: str = "%m/%d/%Y",
) -> dict[str, Any]:
    cleaned = standardize_invoice(
        invoice,
        fallback_currency=fallback_currency,
        date_format="auto",
        output_date_format=output_date_format,
    )
    return _clean_fields(cleaned)


def main() -> None:
    st.set_page_config(page_title="Invoice data cleaning", page_icon="📄")
    st.title("Invoice data cleaning")
    st.write(
        "Upload the JSON downloaded from Invoicedataextraction.py, or paste its "
        "extracted invoice details JSON below."
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

    if not st.button("Clean invoice"):
        return
    if uploaded_file is None and not pasted_json.strip():
        st.error("Upload a JSON file or paste invoice JSON to continue.")
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
        {"Field": field_name, "Value": value}
        for field_name, value in cleaned.items()
        if field_name not in {"line_items", "original_values"}
    ]
    st.table(invoice_fields)

    st.subheader("Line items")
    if cleaned.get("line_items"):
        st.table(cleaned["line_items"])
    else:
        st.info("No line items were found in the invoice data.")

    original_values = cleaned.get("original_values")
    if isinstance(original_values, dict):
        with st.expander("Original extracted values"):
            original_fields = [
                {
                    "Field": field_name,
                    "Value": (
                        json.dumps(value, ensure_ascii=False, default=str)
                        if isinstance(value, (dict, list))
                        else value
                    ),
                }
                for field_name, value in original_values.items()
            ]
            st.table(original_fields)

    st.download_button(
        "Download cleaned JSON",
        data=json.dumps(cleaned, indent=2, ensure_ascii=False),
        file_name="cleaned_invoice.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()