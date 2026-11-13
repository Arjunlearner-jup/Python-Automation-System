import json
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Optional

import streamlit as st


AMOUNT_FIELDS = {
    "net_amount",
    "vat_amount",
    "gross_amount",
    "net_price",
    "net_worth",
    "gross_worth",
}
NUMERIC_FIELDS = {"quantity", "vat_in_percent"}
DATE_FIELDS = {"invoice_date"}
IBAN_FIELDS = {"iban_number"}
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20,
    "BE": 16, "BG": 22, "BH": 22, "BI": 27, "BR": 29, "BY": 28,
    "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DJ": 27,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18,
    "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23,
    "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21,
    "LY": 25, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27,
    "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28,
    "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33,
    "SA": 24, "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24,
    "SM": 27, "SN": 28, "SO": 23, "ST": 25, "SV": 28, "TL": 23,
    "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}
SUPPORTED_FALLBACK_CURRENCIES = (
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "INR",
    "JPY",
    "CHF",
)
CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "CHF": "CHF",
}


def _standardize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", value).strip()


def _standardize_text_case(value: str, field_name: str) -> str:
    if field_name in {"seller_tax_id", "client_tax_id", "currency_code"}:
        return value.upper()
    if field_name == "unit_measure":
        return value.lower()
    value = value.title()
    return re.sub(
        r",\s*([A-Za-z]{2})(?=\s|$)",
        lambda match: f", {match[1].upper()}",
        value,
    )


def _standardize_invoice_number(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    invoice_number = _standardize_text(str(value))
    invoice_number = re.sub(
        r"^(?:INV\s*-\s*)+",
        "",
        invoice_number,
        flags=re.IGNORECASE,
    )
    if not invoice_number:
        return None
    return f"INV-{invoice_number}"


def _parse_decimal(value: Any, field_path: str) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must be a number, not a boolean.")

    text = _standardize_text(str(value))
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[\s$€£¥₹]", "", text)
    text = re.sub(r"(?i)(USD|EUR|GBP|CAD|AUD|INR|JPY|CHF)", "", text)
    text = text.replace("%", "")
    if negative:
        text = f"-{text[1:-1]}"

    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif "," in text:
        last_group = text.rsplit(",", 1)[1]
        if len(last_group) in (1, 2):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(".") > 1:
        last_dot = text.rfind(".")
        text = text[:last_dot].replace(".", "") + text[last_dot:]

    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        raise ValueError(f"{field_path} contains an invalid number: {value!r}.")
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"{field_path} contains an invalid number: {value!r}."
        ) from error


def _standardize_number(value: Any, field_path: str) -> Optional[int | float]:
    number = _parse_decimal(value, field_path)
    if number is None:
        return None
    if number == number.to_integral_value():
        return int(number)
    return float(number.normalize())


def _standardize_vat_percentage(value: Any, field_path: str) -> Optional[str]:
    number = _parse_decimal(value, field_path)
    if number is None:
        return None
    return f"{number.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def _standardize_amount(value: Any, field_path: str) -> Optional[int | float]:
    amount = _parse_decimal(value, field_path)
    if amount is None:
        return None
    amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if amount == amount.to_integral_value():
        return int(amount)
    return float(amount)


def _standardize_date(
    value: Any,
    field_path: str,
    date_format: str,
    output_date_format: str,
) -> Optional[str]:
    if value is None or value == "":
        return None
    parsed_date = value.date() if isinstance(value, datetime) else None
    text = _standardize_text(str(value)) if parsed_date is None else ""
    formats = {
        "auto": (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ),
        "MM/DD/YYYY": ("%m/%d/%Y", "%m-%d-%Y"),
        "DD/MM/YYYY": ("%d/%m/%Y", "%d-%m-%Y"),
    }
    for candidate_format in formats[date_format]:
        try:
            parsed_date = datetime.strptime(text, candidate_format).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        raise ValueError(
            f"{field_path} has unsupported date {value!r}; "
            "use YYYY-MM-DD or the selected date format."
        )
    return parsed_date.strftime(output_date_format)


def _standardize_iban(value: str) -> str:
    iban = re.sub(r"[^A-Z0-9]", "", value.upper())
    if len(iban) < 4 or not iban[:2].isalpha() or not iban[2:4].isdigit():
        raise ValueError(f"Invalid IBAN value: {value!r}.")
    expected_length = IBAN_LENGTHS.get(iban[:2])
    if expected_length is None:
        raise ValueError(f"Unsupported IBAN country code in {value!r}.")
    if len(iban) < expected_length:
        raise ValueError(f"Incomplete IBAN value: {value!r}.")
    return iban[:expected_length]


def _currency_code(invoice: dict[str, Any], fallback_currency: str) -> str:
    provided_code = invoice.get("currency_code")
    if provided_code:
        code = _standardize_text(str(provided_code)).upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(
                f"currency_code must be a three-letter ISO code, got {provided_code!r}."
            )
        return code

    observed_codes = set()
    for field_name in AMOUNT_FIELDS:
        value = invoice.get(field_name)
        if isinstance(value, str):
            observed_codes.update(
                CURRENCY_SYMBOLS[symbol]
                for symbol in CURRENCY_SYMBOLS
                if symbol in value
            )
        elif isinstance(value, (int, float)):
            continue
    for item in invoice.get("line_items", []):
        if not isinstance(item, dict):
            continue
        for field_name in AMOUNT_FIELDS:
            value = item.get(field_name)
            if isinstance(value, str):
                observed_codes.update(
                    CURRENCY_SYMBOLS[symbol]
                    for symbol in CURRENCY_SYMBOLS
                    if symbol in value
                )
    if len(observed_codes) > 1:
        raise ValueError(
            "The invoice contains multiple currency symbols. "
            "Set currency_code explicitly before standardizing."
        )
    return next(iter(observed_codes), fallback_currency)


def standardize_invoice(
    invoice: dict[str, Any],
    fallback_currency: str = "USD",
    date_format: str = "auto",
    output_date_format: str = "%m/%d/%Y",
) -> dict[str, Any]:
    """Return a normalized copy of one extracted invoice record."""
    if not isinstance(invoice, dict):
        raise ValueError("The input JSON must contain an invoice object.")
    line_items = invoice.get("line_items", [])
    if not isinstance(line_items, list):
        raise ValueError("line_items must be a JSON array.")
    if any(not isinstance(item, dict) for item in line_items):
        raise ValueError("Each line_items entry must be a JSON object.")
    if fallback_currency not in SUPPORTED_FALLBACK_CURRENCIES:
        raise ValueError(f"Unsupported fallback currency: {fallback_currency}.")
    if date_format not in {"auto", "MM/DD/YYYY", "DD/MM/YYYY"}:
        raise ValueError(f"Unsupported date format: {date_format}.")
    if output_date_format not in {"%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"}:
        raise ValueError(f"Unsupported output date format: {output_date_format}.")

    currency_code = _currency_code(invoice, fallback_currency)

    def normalize(value: Any, field_name: str, field_path: str) -> Any:
        if isinstance(value, dict):
            normalized = {}
            for key, child in value.items():
                child_path = f"{field_path}.{key}" if field_path else key
                normalized[key] = normalize(child, key, child_path)
            return normalized
        if isinstance(value, list):
            return [
                normalize(child, field_name, f"{field_path}[{index}]")
                for index, child in enumerate(value)
            ]
        if isinstance(value, str):
            value = _standardize_text(value)
            if field_name in IBAN_FIELDS:
                return _standardize_iban(value)
        if field_name == "invoice_number":
            return _standardize_invoice_number(value)
        if field_name in AMOUNT_FIELDS and currency_code != "USD":
            return None
        if field_name in AMOUNT_FIELDS:
            return _standardize_amount(value, field_path)
        if field_name == "vat_in_percent":
            return _standardize_vat_percentage(value, field_path)
        if field_name in NUMERIC_FIELDS:
            return _standardize_number(value, field_path)
        if field_name in DATE_FIELDS:
            return _standardize_date(
                value, field_path, date_format, output_date_format
            )
        if isinstance(value, str):
            return _standardize_text_case(value, field_name) or None
        return value

    standardized = normalize(invoice, "", "")
    standardized["currency_code"] = currency_code
    return standardized


def main() -> None:
    st.set_page_config(page_title="Invoice data standardisation", page_icon="📄")
    st.title("Invoice data standardisation")
    st.write(
        "Upload the JSON downloaded from Invoicedataextraction.py, or paste its "
        "Extracted invoice details JSON below."
    )

    uploaded_file = st.file_uploader("Upload extracted invoice JSON", type=["json"])
    pasted_json = st.text_area("Or paste extracted JSON", height=240)
    fallback_currency = st.selectbox(
        "Fallback currency when no code or symbol is present",
        SUPPORTED_FALLBACK_CURRENCIES,
        index=0,
    )
    output_date_format = st.selectbox(
        "Standardized date output format",
        ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"),
        format_func=lambda value: {
            "%m/%d/%Y": "MM/DD/YYYY",
            "%d/%m/%Y": "DD/MM/YYYY",
            "%Y-%m-%d": "YYYY-MM-DD (ISO)",
        }[value],
    )

    if not st.button("Standardize invoice"):
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
        standardized = standardize_invoice(
            invoice,
            fallback_currency=fallback_currency,
            date_format="auto",
            output_date_format=output_date_format,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not standardize invoice data: {error}")
        return

    st.subheader("Standardized invoice details")
    if standardized["currency_code"] != "USD":
        st.warning(
            f"Currency is {standardized['currency_code']}, not USD. "
            "Monetary amount values are hidden and shown as null."
        )
    st.json(standardized)
    st.download_button(
        "Download standardized JSON",
        data=json.dumps(standardized, indent=2, ensure_ascii=False),
        file_name="standardized_invoice.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()