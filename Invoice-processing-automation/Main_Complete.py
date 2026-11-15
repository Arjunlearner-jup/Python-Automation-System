"""Single-file Streamlit invoice processing application."""

import csv
import hashlib
import io
import json
import os
import re
import unicodedata
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Optional

import pdfplumber
import streamlit as st
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError


SUPPORTED_FALLBACK_CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD", "INR", "JPY", "CHF")
AMOUNT_FIELDS = {"net_amount", "vat_amount", "gross_amount", "net_price", "net_worth", "gross_worth"}
DATE_FORMATS = {"MM/DD/YYYY": "%m/%d/%Y", "DD/MM/YYYY": "%d/%m/%Y", "YYYY-MM-DD (ISO)": "%Y-%m-%d"}
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
FIELD_LABELS = {
    "invoice_number": (r"invoice\s*(?:number|no\.?|#|id)", r"inv\.?\s*no\.?"),
    "invoice_date": (r"invoice\s+date", r"date\s+of\s+issue", r"issue\s+date", r"date"),
    "seller_name": (r"seller", r"supplier", r"from"),
    "client_name": (r"client", r"customer", r"buyer", r"bill\s+to"),
    "net_amount": (r"net\s+(?:amount|total|worth|subtotal)", r"subtotal"),
    "vat_amount": (r"(?:vat|tax)(?:\s+amount|\s+total)?",),
    "gross_amount": (r"grand\s+total(?:\s+due)?", r"amount\s+due", r"gross\s+(?:amount|total|worth)", r"(?<!net\s)total(?:\s+due)?"),
}
LEGACY_FIELD_MAP = {
    "Invoice no": "invoice_number", "Date of issue": "invoice_date",
    "Seller": "seller_name", "Seller Tax ID": "seller_tax_id",
    "Seller IBAN": "iban_number", "Client": "client_name",
    "Client Tax ID": "client_tax_id",
}


def _display_field_name(field_name: str) -> str:
    return re.sub(r"_+", " ", field_name).strip().title()


def _parse_decimal(value: Any, field_path: str = "value") -> Optional[Decimal]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must be a number, not a boolean.")
    text = unicodedata.normalize("NFC", str(value)).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"(?i)(USD|EUR|GBP|CAD|AUD|INR|JPY|CHF)", "", text)
    text = re.sub(r"[\s$€£¥₹%\u00a0\u202f]", "", text)
    if negative:
        text = f"-{text[1:-1]}"
    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif "," in text:
        tail = text.rsplit(",", 1)[1]
        text = text.replace(",", ".") if len(tail) in (1, 2) else text.replace(",", "")
    elif text.count(".") > 1:
        last_dot = text.rfind(".")
        text = text[:last_dot].replace(".", "") + text[last_dot:]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        raise ValueError(f"{field_path} contains an invalid number: {value!r}.")
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{field_path} contains an invalid number: {value!r}.") from error


def _number(value: Any, path: str) -> Optional[int | float]:
    number = _parse_decimal(value, path)
    if number is None:
        return None
    return int(number) if number == number.to_integral_value() else float(number.normalize())


def _date(value: Any, path: str, output_format: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        parsed = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"{path} has unsupported date {value!r}.")
    return parsed.strftime(output_format)


def _standardize_iban(value: str) -> str:
    value = re.sub(r"(?i)ITEMS\s*NO\s*$", "", value).strip()
    iban = re.sub(r"[^A-Z0-9]", "", value.upper())
    if len(iban) < 4 or not iban[:2].isalpha() or not iban[2:4].isdigit():
        raise ValueError(f"Invalid IBAN value: {value!r}.")
    expected_length = IBAN_LENGTHS.get(iban[:2])
    if expected_length is None:
        raise ValueError(f"Unsupported IBAN country code in {value!r}.")
    if len(iban) != expected_length:
        raise ValueError(
            f"IBAN for {iban[:2]} must contain {expected_length} characters; "
            f"received {len(iban)}."
        )
    return iban


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


def _currency_code(invoice: dict[str, Any], fallback_currency: str) -> str:
    code = invoice.get("currency_code")
    if code:
        code = str(code).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError("currency_code must be a three-letter ISO code.")
        return code
    symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "CHF": "CHF"}
    found = set()
    def scan(value: Any, field: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                scan(child, key)
        elif isinstance(value, list):
            for child in value:
                scan(child, field)
        elif field in AMOUNT_FIELDS and isinstance(value, str):
            found.update(currency for symbol, currency in symbols.items() if symbol in value)
            found.update(code.upper() for code in re.findall(r"(?i)\b(?:USD|EUR|GBP|CAD|AUD|INR|JPY|CHF)\b", value))
    scan(invoice)
    if len(found) > 1:
        raise ValueError("The invoice contains multiple currency symbols. Set currency_code explicitly.")
    return next(iter(found), fallback_currency)


def standardize_invoice(invoice: dict[str, Any], fallback_currency: str = "USD", output_date_format: str = "%m/%d/%Y") -> dict[str, Any]:
    if not isinstance(invoice, dict):
        raise ValueError("The input must be an invoice object.")
    if not isinstance(invoice.get("line_items", []), list) or any(not isinstance(item, dict) for item in invoice.get("line_items", [])):
        raise ValueError("line_items must be an array of objects.")
    if fallback_currency not in SUPPORTED_FALLBACK_CURRENCIES:
        raise ValueError(f"Unsupported fallback currency: {fallback_currency}.")
    if output_date_format not in DATE_FORMATS.values():
        raise ValueError("Unsupported date output format.")
    currency = _currency_code(invoice, fallback_currency)

    def normalize(value: Any, key: str = "", path: str = "") -> Any:
        if key == "original_values":
            return value
        if isinstance(value, dict):
            return {k: normalize(v, k, f"{path}.{k}" if path else k) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v, key, f"{path}[{i}]") for i, v in enumerate(value)]
        if key == "invoice_number":
            text = re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""
            text = re.sub(r"^(?:INV\s*-\s*)+", "", text, flags=re.IGNORECASE)
            return f"INV-{text}" if text else None
        if key in AMOUNT_FIELDS:
            if currency != "USD":
                return None
            amount = _parse_decimal(value, path)
            if amount is None:
                return None
            amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            return int(amount) if amount == amount.to_integral_value() else float(amount)
        if key == "vat_in_percent":
            amount = _parse_decimal(value, path)
            return f"{amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}" if amount is not None else None
        if key == "quantity":
            return _number(value, path)
        if key == "invoice_date":
            return _date(value, path, output_date_format)
        if key == "iban_number" and isinstance(value, str):
            return _standardize_iban(value)
        if isinstance(value, str):
            text = unicodedata.normalize("NFC", value)
            text = re.sub(r"\s+", " ", text).strip()
            if key in {"seller_tax_id", "client_tax_id", "currency_code"}:
                return text.upper() or None
            if key == "unit_measure":
                return text.lower() or None
            return _standardize_text_case(text, key) or None
        return value

    cleaned = normalize(invoice)
    cleaned["currency_code"] = currency
    return cleaned


def clean_invoice(invoice: dict[str, Any], fallback_currency: str = "USD", output_date_format: str = "%m/%d/%Y") -> dict[str, Any]:
    """Standardize invoice fields and apply the application's field cleaning rules."""
    cleaned = standardize_invoice(invoice, fallback_currency, output_date_format)
    for field in ("seller_tax_id", "client_tax_id"):
        value = cleaned.get(field)
        if isinstance(value, str):
            cleaned[field] = re.sub(r"\D", "", value) or None
    for field in ("seller_name", "client_name"):
        name = cleaned.get(field)
        if isinstance(name, str):
            cleaned[field] = re.sub(
                r"\s+", " ", name.replace("-", " ")
            ).strip().title() or None
    return cleaned


def _field_value(text: str, field: str) -> Optional[str]:
    patterns = FIELD_LABELS[field]
    all_patterns = [
        pattern for aliases in FIELD_LABELS.values() for pattern in aliases
    ] + [
        r"tax\s*(?:identification\s*)?(?:id|no\.?|number)",
        r"vat\s*(?:registration\s*)?(?:id|no\.?|number)",
        r"\biban(?:\s*(?:no\.?|number))?",
    ]
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(rf"(?<!\w)(?:{pattern})(?!\w)\s*[:#-]?\s*(.*)", line, re.IGNORECASE)
            if not match:
                continue
            value = match.group(1)
            for other in all_patterns:
                value = re.split(rf"(?<!\w)(?:{other})(?!\w)\s*[:#-]?", value, maxsplit=1, flags=re.IGNORECASE)[0]
            value = value.strip(" \t:;,#-")
            if value:
                if field in AMOUNT_FIELDS:
                    value = re.sub(r"[$€£¥₹]", "", value).strip()
                return value
            for following_line in lines[line_index + 1:]:
                following_line = following_line.strip()
                if not following_line:
                    continue
                if any(
                    re.search(rf"(?<!\w)(?:{label})(?!\w)\s*[:#-]?", following_line, re.I)
                    for label in all_patterns
                ):
                    break
                return following_line
            break
    return None


def _legacy_extract_fields(text: str) -> dict[str, str]:
    labels = {
        "Invoice no": FIELD_LABELS["invoice_number"], "Date of issue": FIELD_LABELS["invoice_date"],
        "Seller": FIELD_LABELS["seller_name"], "Client": FIELD_LABELS["client_name"],
    }
    result = {}
    for output, patterns in labels.items():
        for line in text.splitlines():
            for pattern in patterns:
                match = re.search(rf"(?<!\w)(?:{pattern})(?!\w)\s*[:#-]?\s*(.*)", line, re.IGNORECASE)
                if match and match.group(1).strip(" :#-"):
                    result[output] = match.group(1).strip(" :#-")
                    break
            if output in result:
                break
    return result


def _labelled_text_value(text: str, labels: tuple[str, ...]) -> Optional[str]:
    label_pattern = "|".join(f"(?:{label})" for label in labels)
    for line_index, line in enumerate(text.splitlines()):
        match = re.search(
            rf"(?<!\w)(?:{label_pattern})(?!\w)\s*[:#-]?\s*(.*)",
            line,
            re.I,
        )
        if not match:
            continue
        value = match.group(1).strip(" \t:;,#-")
        if value:
            return value
        for following_line in text.splitlines()[line_index + 1:]:
            following_line = following_line.strip()
            if following_line:
                return following_line
    return None


def merge_legacy_fields(primary: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for legacy_name, value in legacy.items():
        field = LEGACY_FIELD_MAP.get(legacy_name, legacy_name)
        if field in merged and merged[field] not in (None, "", [], {}):
            continue
        if value not in (None, ""):
            merged[field] = value
    return merged


def _extract_number(value: Any) -> Optional[Decimal]:
    try:
        return _parse_decimal(value, "PDF value")
    except ValueError:
        return None


def _complete_line_item(item: dict[str, Any]) -> dict[str, Any]:
    net = item["net_worth"]
    gross = item["gross_worth"]
    vat = item["vat_amount"]
    rate = item["vat_in_percent"]
    if net is None and gross is not None and vat is not None:
        item["net_worth"] = (gross - vat).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        net = item["net_worth"]
    if gross is None and net is not None:
        if vat is not None:
            gross = net + vat
        elif rate is not None:
            vat = (net * rate / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            item["vat_amount"] = vat
            gross = net + vat
        else:
            gross = net
        item["gross_worth"] = gross.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    elif net is not None and gross is not None:
        difference = (gross - net).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if difference:
            item["vat_amount"] = difference
        elif rate is not None:
            item["vat_amount"] = (net * rate / Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            item["gross_worth"] = net + item["vat_amount"]
        elif vat is None:
            item["vat_amount"] = Decimal("0.00")
    if (
        item["quantity"] is None
        and item["net_price"] not in (None, Decimal("0"))
        and net is not None
    ):
        quantity = (net / item["net_price"]).quantize(Decimal("0.01"))
        if quantity > 0 and item["net_price"] * quantity == net:
            item["quantity"] = quantity
    return item


def _line_item_dict(description: str, cells: dict[str, Any]) -> dict[str, Any]:
    item = {
        "description": description,
        "quantity": _extract_number(cells.get("quantity")),
        "unit_measure": cells.get("unit_measure") or None,
        "net_price": _extract_number(cells.get("unit_price")),
        "net_worth": _extract_number(cells.get("net_amount")),
        "vat_in_percent": _extract_number(cells.get("vat_rate")),
        "vat_amount": _extract_number(cells.get("vat_amount")),
        "gross_worth": _extract_number(cells.get("gross_amount") or cells.get("amount")),
    }
    return _complete_line_item(item)


def _extract_line_items_from_tables(pages: Any) -> list[dict[str, Any]]:
    total_row = re.compile(r"^(?:sub\s*total|net\s*total|vat|tax|total|amount\s+due|grand\s+total)\b", re.I)
    for page in pages:
        tables = page.extract_tables() or []
        if not tables:
            try:
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"}) or []
            except TypeError:
                tables = []
        for table in tables:
            for header_index, row in enumerate(table):
                headers = [str(cell or "").strip().lower() for cell in row]
                last_header_index = header_index
                continuation_words = re.compile(
                    r"^(?:worth|amount|total|price|rate|description|qty|quantity|unit|vat|tax|gross|net|%|\[%\])$",
                    re.I,
                )
                for continuation_index in range(header_index + 1, min(header_index + 3, len(table))):
                    continuation = [str(cell or "").strip().lower() for cell in table[continuation_index]]
                    non_empty = [cell for cell in continuation if cell]
                    if (
                        not non_empty
                        or any(re.search(r"\d", cell) for cell in non_empty)
                        or not all(continuation_words.fullmatch(cell) for cell in non_empty)
                    ):
                        break
                    for column_index, cell in enumerate(continuation[:len(headers)]):
                        if cell:
                            headers[column_index] = f"{headers[column_index]} {cell}".strip()
                    last_header_index = continuation_index
                description_col = next((i for i, name in enumerate(headers) if re.search(r"description|details|product|item", name)), None)
                if description_col is None:
                    continue
                cols = {
                    "quantity": next((i for i, name in enumerate(headers) if re.search(r"\bqty\b|quantity", name)), None),
                    "unit_measure": next((i for i, name in enumerate(headers) if re.search(r"unit\s*measure|\bu\.?m\.?\b", name)), None),
                    "unit_price": next((i for i, name in enumerate(headers) if re.search(r"unit\s*price|price\s*each|net\s+price|\brate\b", name)), None),
                    "net_amount": next((i for i, name in enumerate(headers) if re.search(r"net\s*(?:worth|amount|total)", name)), None),
                    "vat_rate": next((i for i, name in enumerate(headers) if re.search(r"\bvat\b|\btax\b", name) and not re.search(r"amount", name)), None),
                    "vat_amount": next((i for i, name in enumerate(headers) if re.search(r"(?:vat|tax)\s+amount", name)), None),
                    "gross_amount": next((i for i, name in enumerate(headers) if re.search(r"gross\s*(?:worth|amount|total)", name)), None),
                    "amount": next((i for i, name in enumerate(headers) if re.search(r"line\s+total|\bamount\b|\btotal\b", name)), None),
                }
                if cols["gross_amount"] is None and cols["amount"] is None and cols["net_amount"] is None:
                    continue
                items = []
                for row_values in table[last_header_index + 1:]:
                    cells = [str(cell or "").strip() for cell in row_values]
                    description = cells[description_col] if description_col < len(cells) else ""
                    if not description or total_row.match(description):
                        continue
                    values = {
                        name: cells[index] if index is not None and index < len(cells) else None
                        for name, index in cols.items()
                    }
                    item = _line_item_dict(description, values)
                    if item["gross_worth"] is not None or item["net_worth"] is not None:
                        items.append(item)
                if items:
                    return items
    return []


def _extract_line_items_from_words(page: Any) -> list[dict[str, Any]]:
    words = page.extract_words()
    lines = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not lines or abs(lines[-1][0] - word["top"]) > 3:
            lines.append((word["top"], [word]))
        else:
            lines[-1][1].append(word)
    for header_index, (header_top, first_line_words) in enumerate(lines):
        header_words = list(first_line_words)
        last_header_index = header_index
        for next_index in range(header_index + 1, len(lines)):
            if lines[next_index][0] - header_top > 25:
                break
            next_text = " ".join(word["text"] for word in lines[next_index][1])
            if re.search(r"\d", next_text) or not re.search(
                r"description|details|product|item|qty|quantity|unit|price|net|worth|vat|tax|gross|amount|total|no\.?",
                next_text,
                re.I,
            ):
                break
            header_words.extend(lines[next_index][1])
            last_header_index = next_index
        header_words = sorted(header_words, key=lambda word: word["x0"])
        header_text = " ".join(word["text"] for word in header_words)
        if not re.search(r"description|details|product|item", header_text, re.I):
            continue
        if not re.search(r"amount|worth|line\s+total|gross|net|total", header_text, re.I):
            continue
        patterns = {
            "description": r"description|details|product|item",
            "quantity": r"qty|quantity",
            "unit_measure": r"unit\s+measure|u\.?m\.?",
            "unit_price": r"unit\s+price|net\s+price|price\s+each|rate",
            "net_amount": r"net\s+(?:worth|amount|total)",
            "vat_rate": r"(?:vat|tax)(?:\s*(?:\[%\]|rate|percent|%))?",
            "vat_amount": r"(?:vat|tax)\s+amount",
            "gross_amount": r"gross\s+(?:worth|amount|total)",
            "amount": r"line\s+total|amount|total",
        }
        centers = {}
        for name, pattern in patterns.items():
            for start in range(len(header_words)):
                for end in range(start + 1, min(start + 4, len(header_words) + 1)):
                    phrase = " ".join(word["text"].strip(":#") for word in header_words[start:end])
                    if name == "vat_rate" and re.search(r"\bamount\b", phrase, re.I):
                        continue
                    if name == "vat_rate" and phrase.lower() in {"vat", "tax"}:
                        following = header_words[end] if end < len(header_words) else None
                        if following and re.fullmatch(r"amount", following["text"].strip(":#"), re.I):
                            continue
                    if re.fullmatch(pattern, phrase, re.I):
                        group = header_words[start:end]
                        centers[name] = sum((word["x0"] + word["x1"]) / 2 for word in group) / len(group)
                        break
                if name in centers:
                    break
        amount_field = next((key for key in ("gross_amount", "amount", "net_amount") if key in centers), None)
        if amount_field is None or "description" not in centers:
            continue
        ordered = sorted(centers.items(), key=lambda entry: entry[1])
        bounds = [(ordered[i][1] + ordered[i + 1][1]) / 2 for i in range(len(ordered) - 1)]
        description_header = next(word for word in header_words if re.fullmatch(patterns["description"], word["text"].strip(":#"), re.I))
        quantity_header = next((word for word in header_words if re.fullmatch(patterns["quantity"], word["text"].strip(":#"), re.I)), None)
        desc_left = description_header["x0"]
        desc_right = quantity_header["x0"] if quantity_header else float("inf")
        result = []
        for _, row_words in lines[last_header_index + 1:]:
            row_text = " ".join(word["text"] for word in row_words).strip()
            if re.match(r"^(?:sub\s*total|net\s*total|vat|tax|total|amount\s+due|grand\s+total)\b", row_text, re.I):
                break
            columns = {name: [] for name, _ in ordered}
            for word in row_words:
                center = (word["x0"] + word["x1"]) / 2
                index = sum(center >= boundary for boundary in bounds)
                columns[ordered[index][0]].append(word["text"])
            description = " ".join(word["text"] for word in row_words if desc_left <= (word["x0"] + word["x1"]) / 2 < desc_right).strip()
            if not description:
                continue
            values = {name: " ".join(words_in_column) for name, words_in_column in columns.items()}
            if not values.get("gross_amount") and not values.get("amount") and not values.get("net_amount"):
                if result:
                    result[-1]["description"] = f"{result[-1]['description']} {description}".strip()
                continue
            result.append(_line_item_dict(description, values))
        if result:
            return result
    return []


def _extract_line_items(pages: Any) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for page in pages:
        page_items = _extract_line_items_from_words(page)
        if not page_items:
            page_items = _extract_line_items_from_tables([page])
        for item in page_items:
            signature = (item["description"], item["quantity"], item["net_price"], item["net_worth"], item["gross_worth"])
            if signature not in seen:
                seen.add(signature)
                items.append(item)
    return items


def _extract_party_details(pages: Any) -> dict[str, Optional[str]]:
    details = {
        "seller_name": None, "seller_address": None, "seller_tax_id": None,
        "client_name": None, "client_address": None, "client_tax_id": None,
        "iban_number": None,
    }
    party_labels = {"seller": "seller", "supplier": "seller", "client": "client", "customer": "client", "buyer": "client"}
    section_heading = re.compile(r"^(?:invoice items|items?|totals?|vat summary|payment|notes|terms|subtotal)\b", re.I)
    metadata = re.compile(r"^(tax\s*id|vat(?:\s*(?:id|no\.?|number))?|iban)\b\s*[:#-]?\s*(.*)$", re.I)
    for page in pages:
        words = page.extract_words()
        page_lines = []
        for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
            if not page_lines or abs(page_lines[-1][0] - word["top"]) > 3:
                page_lines.append((word["top"], [word]))
            else:
                page_lines[-1][1].append(word)
        headers = []
        for word in words:
            label = re.sub(r"[^a-z]", "", word["text"].lower())
            if label in party_labels:
                headers.append({"party": party_labels[label], "x": (word["x0"] + word["x1"]) / 2, "bottom": word["bottom"]})
        headers.sort(key=lambda header: header["x"])
        for index, header in enumerate(headers):
            party = header["party"]
            left = (headers[index - 1]["x"] + header["x"]) / 2 if index else 0
            right = (header["x"] + headers[index + 1]["x"]) / 2 if index + 1 < len(headers) else page.width
            name_and_address = []
            for top, line_words in page_lines:
                if top <= header["bottom"]:
                    continue
                if top > header["bottom"] + 350:
                    break
                whole_line = " ".join(word["text"] for word in line_words).strip()
                if section_heading.match(whole_line):
                    break
                line_text = " ".join(word["text"] for word in line_words if left <= (word["x0"] + word["x1"]) / 2 < right).strip()
                if not line_text:
                    continue
                meta = metadata.match(line_text)
                if meta:
                    if meta.group(1).lower().startswith("tax") or meta.group(1).lower().startswith("vat"):
                        details[f"{party}_tax_id"] = meta.group(2).strip() or None
                    elif party == "seller":
                        details["iban_number"] = meta.group(2).strip() or None
                    break
                name_and_address.append(line_text)
            if name_and_address:
                details[f"{party}_name"] = details[f"{party}_name"] or name_and_address[0]
                if len(name_and_address) > 1:
                    details[f"{party}_address"] = details[f"{party}_address"] or ", ".join(name_and_address[1:])
    return details


def _labeled_amount(text: str, labels: tuple[str, ...], excluded: Optional[re.Pattern] = None) -> Optional[Decimal]:
    label_pattern = "|".join(f"(?:{label})" for label in labels)
    for line in text.splitlines():
        if excluded and excluded.search(line):
            continue
        match = re.search(rf"(?<!\w)(?:{label_pattern})(?!\w)\s*[:#-]?\s*(.*)", line, re.I)
        if not match:
            continue
        amount_tokens = re.findall(r"[-+]?\s*(?:[$€£¥₹]\s*)?\d(?:[\d.,]|\s(?=\d))*", match.group(1))
        for token in amount_tokens:
            amount = _extract_number(token)
            if amount is not None:
                return amount
    return None


def _amount_from_tables(pages: Any, labels: tuple[str, ...]) -> Optional[Decimal]:
    pattern = re.compile(rf"^\s*(?:{'|'.join(labels)})\s*[:#-]?\s*$", re.I)
    for page in pages:
        for table in page.extract_tables() or []:
            for row in table:
                cells = [str(cell or "").strip() for cell in row]
                if cells and pattern.match(cells[0]):
                    for cell in reversed(cells[1:]):
                        amount = _extract_number(cell)
                        if amount is not None:
                            return amount
    return None


def extract_invoice_record(document_text: str, pages: Any = ()) -> dict[str, Any]:
    """Extract invoice fields, totals, and line items from selectable PDF text."""
    party = _extract_party_details(pages)
    invoice = {
        "invoice_number": _field_value(document_text, "invoice_number"),
        "invoice_date": _field_value(document_text, "invoice_date"),
        "seller_name": party["seller_name"] or _field_value(document_text, "seller_name"),
        "seller_address": party["seller_address"] or _labelled_text_value(
            document_text, (r"seller\s+address", r"supplier\s+address", r"from\s+address")
        ),
        "seller_tax_id": party["seller_tax_id"] or _labelled_text_value(
            document_text,
            (r"(?:seller|supplier)\s+(?:tax\s*(?:id|no\.?|number)|vat\s*(?:id|no\.?|number))",),
        ),
        "client_name": party["client_name"] or _field_value(document_text, "client_name"),
        "client_address": party["client_address"] or _labelled_text_value(
            document_text, (r"client\s+address", r"customer\s+address", r"buyer\s+address")
        ),
        "client_tax_id": party["client_tax_id"] or _labelled_text_value(
            document_text,
            (r"(?:client|customer|buyer)\s+(?:tax\s*(?:id|no\.?|number)|vat\s*(?:id|no\.?|number))",),
        ),
        "iban_number": party["iban_number"] or _labelled_text_value(
            document_text, (r"iban(?:\s*(?:no\.?|number))?",)
        ),
    }
    total_amounts = {
        "net_amount": _labeled_amount(document_text, (r"net\s*(?:amount|total|worth)", r"sub\s*total", r"taxable\s+amount")),
        "vat_amount": _labeled_amount(
            document_text,
            (r"vat(?!\s*(?:registration|number|no\.?|id|rate))(?:\s*(?:amount|total))?", r"tax(?!\s*(?:identification|number|no\.?|id|rate))(?:\s*(?:amount|total))?", r"sales\s+tax"),
            re.compile(r"\b(?:vat|tax)\s*(?:\[%\]|rate)(?:\s|$)", re.I),
        ),
        "gross_amount": _labeled_amount(
            document_text,
            (r"gross(?:\s*(?:amount|total|worth))?", r"grand\s*total", r"total\s+due", r"amount\s+due", r"balance\s+due", r"total"),
            re.compile(r"^\s*(?:net|sub\s*total|subtotal|vat|tax)\b", re.I),
        ),
    }
    if total_amounts["net_amount"] is None:
        total_amounts["net_amount"] = _amount_from_tables(pages, (r"net(?:\s+(?:amount|total|worth))?", r"sub\s*total", r"taxable\s+amount"))
    if total_amounts["vat_amount"] is None:
        total_amounts["vat_amount"] = _amount_from_tables(pages, (r"vat(?:\s*(?:amount|total))?", r"tax(?:\s*(?:amount|total))?"))
    if total_amounts["gross_amount"] is None:
        total_amounts["gross_amount"] = _amount_from_tables(pages, (r"gross(?:\s*(?:amount|total|worth))?", r"grand\s*total", r"total\s+due", r"amount\s+due", r"balance\s+due", r"total"))
    invoice.update({key: str(value) if value is not None else None for key, value in total_amounts.items()})

    if not invoice["seller_tax_id"] or not invoice["client_tax_id"]:
        tax_ids = re.findall(
            r"\b(?:tax\s*(?:identification\s*)?(?:id|no\.?|number)|vat\s*(?:id|no\.?|number))\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{4,})",
            document_text,
            re.I,
        )
        if tax_ids:
            invoice["seller_tax_id"] = invoice["seller_tax_id"] or tax_ids[0]
        if len(tax_ids) > 1:
            invoice["client_tax_id"] = invoice["client_tax_id"] or tax_ids[1]
    if not invoice["iban_number"]:
        match = re.search(r"\biban(?:\s*(?:no\.?|number))?\s*[:#-]?\s*([A-Z]{2}\d{2}(?:\s*[A-Z0-9]){10,32})", document_text, re.I)
        invoice["iban_number"] = re.sub(r"\s+", "", match.group(1)) if match else None

    currency_symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
    currencies = {currency for symbol, currency in currency_symbols.items() if symbol in document_text}
    currencies.update(code.upper() for code in re.findall(r"(?i)\b(?:USD|EUR|GBP|CAD|AUD|INR|JPY|CHF)\b", document_text))
    if len(currencies) == 1:
        invoice["currency_code"] = currencies.pop()

    items = _extract_line_items(pages)
    invoice["line_items"] = items
    if total_amounts["net_amount"] is None and items and all(item.get("net_worth") is not None for item in items):
        total_amounts["net_amount"] = sum((item["net_worth"] for item in items), Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if total_amounts["gross_amount"] is None and items and all(item.get("gross_worth") is not None for item in items):
        total_amounts["gross_amount"] = sum((item["gross_worth"] for item in items), Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if total_amounts["vat_amount"] is None and items and all(item.get("vat_amount") is not None for item in items):
        total_amounts["vat_amount"] = sum((item["vat_amount"] for item in items), Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    elif total_amounts["net_amount"] is not None and total_amounts["gross_amount"] is not None:
        total_amounts["vat_amount"] = (total_amounts["gross_amount"] - total_amounts["net_amount"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    for field in total_amounts:
        invoice[field] = str(total_amounts[field]) if total_amounts[field] is not None else None
    return invoice


def extract_pdf_invoice(pdf_bytes: bytes) -> tuple[dict[str, Any], str]:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The selected file does not appear to be a valid PDF.")
    from Invoicedataextraction import (
        _record_output as serialize_extracted_record,
        extract_invoice_record as extract_record,
    )

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not text.strip():
            raise ValueError("No selectable text found. Scanned PDFs require OCR.")
        record = extract_record(text, pdf.pages)
    invoice = serialize_extracted_record(record)
    iban = invoice.get("iban_number")
    if isinstance(iban, str):
        invoice["iban_number"] = re.sub(r"(?i)ITEMS\s*NO\s*$", "", iban).strip()
    return invoice, text


def validate_invoice(invoice: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for field in ("invoice_number", "invoice_date", "seller_name", "gross_amount"):
        if not invoice.get(field):
            errors.append(f"{_display_field_name(field)} is missing.")
    if invoice.get("invoice_date"):
        try:
            _date(invoice["invoice_date"], "invoice_date", "%Y-%m-%d")
        except ValueError as error:
            errors.append(str(error))
    amounts = []
    for field in ("net_amount", "vat_amount", "gross_amount"):
        if invoice.get(field) not in (None, ""):
            try:
                amounts.append((field, _parse_decimal(invoice[field], field)))
            except ValueError as error:
                errors.append(str(error))
    values = dict(amounts)
    if all(values.get(field) is not None for field in ("net_amount", "vat_amount", "gross_amount")):
        expected = values["net_amount"] + values["vat_amount"]
        if abs(expected - values["gross_amount"]) > Decimal("0.02"):
            warnings.append(f"Net amount plus VAT ({expected}) does not equal gross amount ({values['gross_amount']}).")
    if not invoice.get("line_items"):
        warnings.append("No line items were found.")
    return errors, warnings


def invoice_to_csv(invoice: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("Section", "Line item", "Field", "Value"))
    def csv_value(value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value
    for field, value in invoice.items():
        if field not in {"line_items", "original_values"}:
            writer.writerow(("Invoice details", "", _display_field_name(field), csv_value(value)))
    for index, item in enumerate(invoice.get("line_items", []), 1):
        for field, value in item.items():
            writer.writerow(("Line item", index, _display_field_name(field), csv_value(value)))
    return output.getvalue()


def invoice_report_csv(invoices: dict[str, dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("Invoice ID", "Filename", "Section", "Line item", "Field", "Value"))
    for invoice_id, record in invoices.items():
        cleaned = record.get("cleaned")
        if not cleaned:
            continue
        rows = csv.reader(io.StringIO(invoice_to_csv(cleaned)))
        next(rows, None)
        for section, line_item, field, value in rows:
            writer.writerow((invoice_id, record.get("filename", ""), section, line_item, field, value))
    return output.getvalue()


def register_upload(invoices: dict[str, dict[str, Any]], filename: str, pdf_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    if not pdf_bytes.startswith(b"%PDF-"):
        return None, "The selected file does not appear to be a valid PDF."
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    if any(record.get("sha256") == digest for record in invoices.values()):
        return None, "Duplicate upload: this PDF has already been added."
    invoice_id = str(uuid.uuid4())
    invoices[invoice_id] = {"filename": filename, "pdf_bytes": pdf_bytes, "sha256": digest, "extracted": None, "cleaned": None}
    return invoice_id, None


def invoice_to_rows(invoice: dict[str, Any], submission_id: str) -> list[tuple[str, str, int | None, str, str | None]]:
    def db_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    rows = []
    for field, value in invoice.items():
        if field not in {"line_items", "original_values"}:
            rows.append((submission_id, "Invoice details", None, _display_field_name(field), db_value(value)))
    for index, item in enumerate(invoice.get("line_items", []), 1):
        for field, value in item.items():
            rows.append((submission_id, "Line item", index, _display_field_name(field), db_value(value)))
    if not rows:
        raise ValueError("The cleaned invoice has no fields to save.")
    return rows


def save_invoice_to_mysql(cleaned_invoice: dict[str, Any], host: str, port: int, user: str, password: str, database: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError("Database name must contain only letters, numbers, and underscores.")
    import mysql.connector

    connection = mysql.connector.connect(host=host, port=port, user=user, password=password)
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    finally:
        connection.close()
    connection = mysql.connector.connect(host=host, port=port, user=user, password=password, database=database)
    try:
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS cleaned_invoice_data (
            id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            submission_id CHAR(36) NOT NULL,
            section VARCHAR(64) NOT NULL,
            line_item INT NULL,
            `data` VARCHAR(255) NOT NULL,
            `value` LONGTEXT NULL,
            INDEX idx_submission_id (submission_id)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci""")
        submission_id = str(uuid.uuid4())
        cursor.executemany("INSERT INTO cleaned_invoice_data (submission_id, section, line_item, `data`, `value`) VALUES (%s, %s, %s, %s, %s)", invoice_to_rows(cleaned_invoice, submission_id))
        connection.commit()
        return submission_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _page_upload(invoices: dict[str, dict[str, Any]]) -> None:
    st.title("Upload Invoice")
    st.write("Upload one or more PDF invoices. Duplicate files are detected by content.")
    files = st.file_uploader("Choose invoice PDF files", type=["pdf"], accept_multiple_files=True)
    if files:
        for uploaded in files:
            upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
            if upload_hash in st.session_state.get("processed_upload_hashes", set()):
                continue
            invoice_id, error = register_upload(invoices, uploaded.name, uploaded.getvalue())
            if error:
                st.warning(f"{uploaded.name}: {error}")
            else:
                st.success(f"Uploaded {uploaded.name}.")
            st.session_state.setdefault("processed_upload_hashes", set()).add(upload_hash)
    if invoices:
        st.subheader("Uploaded invoices")
        st.dataframe([{"Filename": item["filename"], "Invoice ID": key} for key, item in invoices.items()], hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Download original PDF", list(invoices), format_func=lambda key: invoices[key]["filename"])
        st.download_button("Download PDF", invoices[selected_id]["pdf_bytes"], file_name=invoices[selected_id]["filename"], mime="application/pdf")


def _select_invoice(invoices: dict[str, dict[str, Any]], label: str, require_extracted: bool = False, require_cleaned: bool = False) -> Optional[str]:
    available = [key for key, record in invoices.items() if (not require_extracted or record.get("extracted")) and (not require_cleaned or record.get("cleaned"))]
    if not available:
        st.info("Upload an invoice first." if not invoices else "Complete the previous workflow step first.")
        return None
    return st.selectbox(label, available, format_func=lambda key: invoices[key]["filename"])


def _page_extraction(invoices: dict[str, dict[str, Any]]) -> None:
    st.title("Invoice extraction")
    st.write("Extract invoice details and line items from text-based PDF files.")
    invoice_id = _select_invoice(invoices, "Select an uploaded invoice")
    if invoice_id is None:
        return
    record = invoices[invoice_id]
    if st.button("Extract invoice data", type="primary"):
        try:
            extracted, _ = extract_pdf_invoice(record["pdf_bytes"])
        except (PDFSyntaxError, PDFPasswordIncorrect, OSError, ValueError) as error:
            st.error(f"Could not extract invoice data: {error}")
            return
        record["extracted"] = extracted
        record["cleaned"] = None
    if record.get("extracted"):
        st.subheader("Extracted data")
        st.json(record["extracted"])
        missing = [field for field in ("invoice_number", "invoice_date", "seller_name", "gross_amount") if not record["extracted"].get(field)]
        if missing:
            st.warning("Could not find: " + ", ".join(_display_field_name(field) for field in missing) + ". Review the extracted data before cleaning.")
        st.download_button("Download extracted JSON", json.dumps(record["extracted"], indent=2, ensure_ascii=False), file_name=f"{os.path.splitext(record['filename'])[0]}_extracted.json", mime="application/json")
    else:
        st.caption("Select the file, then choose Extract invoice data.")


def _page_cleaning(invoices: dict[str, dict[str, Any]]) -> None:
    st.title("Data cleaning and extraction")
    invoice_id = _select_invoice(invoices, "Select an extracted invoice", require_extracted=True)
    if invoice_id is None:
        return
    fallback = st.selectbox("Fallback currency", SUPPORTED_FALLBACK_CURRENCIES, index=0)
    output_format = st.selectbox("Cleaned date format", list(DATE_FORMATS), index=0)
    if st.button("Clean and validate invoice", type="primary"):
        try:
            cleaned = clean_invoice(invoices[invoice_id]["extracted"], fallback, DATE_FORMATS[output_format])
        except ValueError as error:
            st.error(f"Could not clean invoice data: {error}")
            return
        invoices[invoice_id]["cleaned"] = cleaned
        errors, warnings = validate_invoice(cleaned)
        invoices[invoice_id]["validation"] = {"errors": errors, "warnings": warnings}
    cleaned = invoices[invoice_id].get("cleaned")
    if not cleaned:
        return
    errors, warnings = validate_invoice(cleaned)
    for error in errors:
        st.error(error)
    for warning in warnings:
        st.warning(warning)
    if cleaned.get("currency_code") != "USD":
        st.warning(f"Currency is {cleaned['currency_code']}, not USD. Monetary values are shown as null.")
    st.subheader("Cleaned invoice details")
    st.dataframe([{"Field": _display_field_name(key), "Value": value} for key, value in cleaned.items() if key not in {"line_items", "original_values"}], hide_index=True, use_container_width=True)
    st.subheader("Line items")
    if cleaned.get("line_items"):
        st.dataframe(cleaned["line_items"], hide_index=True, use_container_width=True)
    else:
        st.info("No line items were found in the invoice data.")
    st.download_button("Download cleaned CSV", invoice_to_csv(cleaned), file_name=f"{os.path.splitext(invoices[invoice_id]['filename'])[0]}_cleaned.csv", mime="text/csv")
    st.download_button("Download cleaned JSON", json.dumps(cleaned, indent=2, ensure_ascii=False), file_name=f"{os.path.splitext(invoices[invoice_id]['filename'])[0]}_cleaned.json", mime="application/json")


def _page_save_report(invoices: dict[str, dict[str, Any]]) -> None:
    st.title("Data save in Mysql and csv export or report")
    st.write("Save a cleaned invoice to MySQL or export a CSV report for all cleaned invoices.")
    with st.sidebar.expander("MySQL connection", expanded=False):
        host = st.text_input("Host", os.getenv("MYSQL_HOST", "localhost"), key="mysql_host")
        port = st.number_input("Port", 1, 65535, int(os.getenv("MYSQL_PORT", "3306")), key="mysql_port")
        user = st.text_input("Username", os.getenv("MYSQL_USER", "root"), key="mysql_user")
        password = st.text_input("Password", os.getenv("MYSQL_PASSWORD", ""), type="password", key="mysql_password")
        database = st.text_input("Database name", os.getenv("MYSQL_DATABASE", "invoice_processing"), key="mysql_database")
    invoice_id = _select_invoice(invoices, "Select a cleaned invoice", require_cleaned=True)
    if invoice_id is not None:
        if st.button("Save invoice to MySQL", type="primary"):
            if not host.strip() or not user.strip():
                st.error("Enter the MySQL host and username.")
            else:
                try:
                    submission_id = save_invoice_to_mysql(invoices[invoice_id]["cleaned"], host.strip(), int(port), user.strip(), password, database.strip())
                    st.success(f"Invoice saved in `{database}.cleaned_invoice_data` (submission `{submission_id}`).")
                except ValueError as error:
                    st.error(f"Could not save invoice to MySQL: {error}")
                except ImportError:
                    st.error("MySQL support is unavailable. Install mysql-connector-python.")
                except Exception as error:
                    st.error(f"Could not save invoice to MySQL. Check the server and account permissions: {error}")
    if any(record.get("cleaned") for record in invoices.values()):
        st.subheader("Cleaned invoice report")
        report = invoice_report_csv(invoices)
        st.download_button("Download CSV report", report, file_name="invoice_report.csv", mime="text/csv")
        st.dataframe([{"Filename": record["filename"], "Invoice Number": (record.get("cleaned") or {}).get("invoice_number"), "Gross Amount": (record.get("cleaned") or {}).get("gross_amount"), "Currency": (record.get("cleaned") or {}).get("currency_code")} for record in invoices.values() if record.get("cleaned")], hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Invoice Processing", page_icon="📄", layout="wide")
    st.sidebar.title("Invoice Processing")
    page = st.sidebar.radio("Navigation", ("Upload Invoice", "Invoice extraction", "Data cleaning and extraction", "Data save in Mysql and csv export or report"))
    invoices = st.session_state.setdefault("invoices", {})
    if page == "Upload Invoice":
        _page_upload(invoices)
    elif page == "Invoice extraction":
        _page_extraction(invoices)
    elif page == "Data cleaning and extraction":
        _page_cleaning(invoices)
    else:
        _page_save_report(invoices)


if __name__ == "__main__":
    main()
