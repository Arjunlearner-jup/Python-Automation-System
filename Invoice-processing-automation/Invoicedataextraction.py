import io
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import pdfplumber
import streamlit as st
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError


@dataclass
class InvoiceLineItem:
    description: str
    quantity: Optional[float] = None
    unit_measure: Optional[str] = None
    unit_price: Optional[float] = None
    vat_rate: Optional[float] = None
    net_amount: Optional[float] = None
    vat_amount: Optional[float] = None
    amount: Optional[float] = None


@dataclass
class InvoiceRecord:
    seller_name: Optional[str] = None
    seller_address: Optional[str] = None
    seller_tax_id: Optional[str] = None
    client_name: Optional[str] = None
    client_address: Optional[str] = None
    client_tax_id: Optional[str] = None
    iban_number: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    net_amount: Optional[float] = None
    vat_amount: Optional[float] = None
    gross_amount: Optional[float] = None
    line_items: list[InvoiceLineItem] = field(default_factory=list)


def _clean_value(value: str) -> Optional[str]:
    value = re.sub(r"\s+", " ", value).strip(" \t:;,#-")
    return value or None


def _extract_labeled_value(
    text: str,
    labels: tuple[str, ...],
    known_labels: tuple[str, ...],
) -> Optional[str]:
    label_pattern = "|".join(f"(?:{label})" for label in labels)
    stop_pattern = "|".join(f"(?:{label})" for label in known_labels)
    pattern = re.compile(
        rf"(?<!\w)(?:{label_pattern})(?!\w)\s*[:#-]?\s*(.*)",
        re.IGNORECASE,
    )

    for line_index, line in enumerate(text.splitlines()):
        match = pattern.search(line)
        if not match:
            continue

        value = match.group(1)
        if stop_pattern:
            value = re.split(
                rf"(?<!\w)(?:{stop_pattern})(?!\w)\s*[:#-]?",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
        value = _clean_value(value)
        if value:
            return value

        for following_line in text.splitlines()[line_index + 1:]:
            following_line = following_line.strip()
            if not following_line:
                continue
            if re.search(
                rf"(?<!\w)(?:{stop_pattern})(?!\w)\s*[:#-]?",
                following_line,
                re.IGNORECASE,
            ):
                break
            return _clean_value(following_line)
    return None


def _parse_number(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", value)
    if not re.search(r"\d", cleaned):
        return None

    if "," in cleaned and "." in cleaned:
        decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "")
        if decimal_separator == ",":
            cleaned = cleaned.replace(",", ".")
    elif "," in cleaned:
        # This invoice workflow treats commas as grouping separators, including
        # values such as "8,00" -> 800 (rather than locale decimal notation).
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        last_dot = cleaned.rfind(".")
        cleaned = cleaned[:last_dot].replace(".", "") + cleaned[last_dot:]

    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_percentage(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    percent_match = re.search(
        r"([-+]?\d+(?:[.,]\d+)?)\s*%",
        value,
    )
    if percent_match:
        return _parse_number(percent_match.group(1))

    parsed = _parse_number(value)
    if parsed is not None and 0 <= parsed <= 100:
        return parsed
    return None


def _complete_line_item(item: InvoiceLineItem) -> InvoiceLineItem:
    if item.net_amount is None and item.amount is not None and item.vat_amount is not None:
        item.net_amount = round(item.amount - item.vat_amount, 2)

    if item.amount is None and item.net_amount is not None:
        if item.vat_amount is not None:
            item.amount = round(item.net_amount + item.vat_amount, 2)
        elif item.vat_rate is not None:
            item.vat_amount = round(item.net_amount * item.vat_rate / 100, 2)
            item.amount = round(item.net_amount + item.vat_amount, 2)
        else:
            item.amount = item.net_amount
    elif item.net_amount is not None and item.amount is not None:
        difference = round(item.amount - item.net_amount, 2)
        if difference != 0:
            item.vat_amount = difference
        elif item.vat_rate is not None:
            item.vat_amount = round(item.net_amount * item.vat_rate / 100, 2)
            item.amount = round(item.net_amount + item.vat_amount, 2)
        elif item.vat_amount is None:
            item.vat_amount = 0.0
    return item


def _extract_labeled_amount(
    text: str,
    labels: tuple[str, ...],
    excluded_line: Optional[re.Pattern] = None,
) -> Optional[float]:
    label_pattern = "|".join(f"(?:{label})" for label in labels)
    for line in text.splitlines():
        if excluded_line and excluded_line.search(line):
            continue
        match = re.search(
            rf"(?<!\w)(?:{label_pattern})(?!\w)\s*[:#-]?\s*(.*)",
            line,
            re.IGNORECASE,
        )
        if match:
            amounts = re.findall(
                r"[-+]?\s*(?:[$€£]\s*)?\d(?:[\d.,]|\s(?=\d))*",
                match.group(1),
            )
            for amount in amounts:
                parsed = _parse_number(amount)
                if parsed is not None:
                    return parsed
    return None


def _extract_table_amount(pages, labels: tuple[str, ...]) -> Optional[float]:
    label_pattern = "|".join(f"(?:{label})" for label in labels)
    row_label = re.compile(rf"^\s*{label_pattern}\s*[:#-]?\s*$", re.IGNORECASE)

    for page in pages:
        for table in page.extract_tables() or []:
            for row in table:
                cells = [str(cell or "").strip() for cell in row]
                if not cells or not row_label.match(cells[0]):
                    continue
                for cell in reversed(cells[1:]):
                    amount = _parse_number(cell)
                    if amount is not None:
                        return amount
    return None


def _header_index(headers: list[str], patterns: tuple[str, ...]) -> Optional[int]:
    for index, header in enumerate(headers):
        if any(re.search(pattern, header, re.IGNORECASE) for pattern in patterns):
            return index
    return None


def _cell(row: list[str], index: Optional[int]) -> Optional[str]:
    if index is None or index >= len(row):
        return None
    return _clean_value(str(row[index] or ""))


def extract_line_items(pages) -> list[InvoiceLineItem]:
    items = []
    seen = set()
    total_row = re.compile(
        r"^(?:sub\s*total|net\s*total|vat|tax|total|amount\s+due|grand\s+total)\b",
        re.IGNORECASE,
    )

    for page in pages:
        word_items = _extract_line_items_from_words(page)
        if word_items:
            for item in word_items:
                _complete_line_item(item)
                key = (
                    item.description,
                    item.quantity,
                    item.unit_price,
                    item.net_amount,
                    item.amount,
                )
                if key not in seen:
                    seen.add(key)
                    items.append(item)
            continue

        tables = page.extract_tables() or []
        if not tables:
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }
            ) or []

        for table in tables:
            header_row_index = None
            for row_index, row in enumerate(table):
                headers = _combined_table_headers(table, row_index)
                description_index = _header_index(
                    headers, (r"description", r"details", r"item", r"product")
                )
                amount_index = _header_index(
                    headers,
                    (
                        r"gross\s*(?:worth|amount|total)",
                        r"line\s+total",
                        r"\bamount\b",
                        r"\btotal\b",
                        r"net\s*(?:worth|amount|total)",
                    ),
                )
                if description_index is not None and amount_index is not None:
                    header_row_index = row_index
                    break

            if header_row_index is None:
                continue

            headers = _combined_table_headers(table, header_row_index)
            description_index = _header_index(
                headers, (r"description", r"details", r"item", r"product")
            )
            quantity_index = _header_index(headers, (r"\bqty\b", r"quantity"))
            unit_measure_index = _header_index(
                headers, (r"unit\s*measure", r"\bu\.?m\.?\b")
            )
            unit_price_index = _header_index(
                headers,
                (r"unit\s*price", r"price\s*each", r"net\s+price", r"rate"),
            )
            vat_index = _header_index(headers, (r"\bvat\b", r"\btax\b"))
            net_index = _header_index(
                headers, (r"net\s*(?:worth|amount|total)",)
            )
            vat_amount_index = _header_index(
                headers, (r"vat\s+amount", r"tax\s+amount")
            )
            gross_index = _header_index(
                headers, (r"gross\s*(?:worth|amount|total)",)
            )
            line_total_index = _header_index(
                headers, (r"line\s+total", r"\bamount\b", r"\btotal\b")
            )

            for row in table[header_row_index + 1:]:
                row = [cell or "" for cell in row]
                description = _cell(row, description_index)
                if not description or total_row.match(description):
                    continue
                selected_amount_index = (
                    gross_index
                    if gross_index is not None
                    else line_total_index
                    if line_total_index is not None
                    else net_index
                )
                amount = _parse_number(_cell(row, selected_amount_index))
                net_amount = _parse_number(_cell(row, net_index))
                vat_amount = _parse_number(_cell(row, vat_amount_index))
                if amount is None and net_amount is not None:
                    amount = net_amount
                if amount is None:
                    continue

                item = InvoiceLineItem(
                    description=description,
                    quantity=_parse_number(_cell(row, quantity_index)),
                    unit_measure=_cell(row, unit_measure_index),
                    unit_price=_parse_number(_cell(row, unit_price_index)),
                    vat_rate=_parse_percentage(_cell(row, vat_index)),
                    net_amount=net_amount,
                    vat_amount=vat_amount,
                    amount=amount,
                )
                _complete_line_item(item)
                key = (
                    item.description,
                    item.quantity,
                    item.unit_price,
                    item.net_amount,
                    item.amount,
                )
                if key not in seen:
                    seen.add(key)
                    items.append(item)

    return items


def _combined_table_headers(table, header_row_index: int) -> list[str]:
    headers = [str(cell or "").strip() for cell in table[header_row_index]]
    continuation_pattern = re.compile(
        r"^(?:worth|amount|total|price|rate|description|qty|quantity|"
        r"vat|tax|gross|net|\[%\])$",
        re.IGNORECASE,
    )
    for following_row in table[header_row_index + 1:header_row_index + 3]:
        cells = [str(cell or "").strip() for cell in following_row]
        non_empty = [cell for cell in cells if cell]
        if not non_empty or any(re.search(r"\d", cell) for cell in non_empty):
            break
        if not all(continuation_pattern.fullmatch(cell) for cell in non_empty):
            break
        for index, cell in enumerate(cells[:len(headers)]):
            if cell:
                headers[index] = f"{headers[index]} {cell}".strip()
    return headers


def _extract_line_items_from_words(page) -> list[InvoiceLineItem]:
    words = page.extract_words()
    lines = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not lines or abs(lines[-1][0] - word["top"]) > 3:
            lines.append((word["top"], [word]))
        else:
            lines[-1][1].append(word)

    for header_index, (header_top, first_header_words) in enumerate(lines):
        header_words = list(first_header_words)
        last_header_index = header_index
        for next_index in range(header_index + 1, len(lines)):
            if lines[next_index][0] - header_top > 25:
                break
            next_line_text = " ".join(
                item["text"] for item in lines[next_index][1]
            )
            if re.search(r"\d", next_line_text) or not re.search(
                r"description|details|product|item|qty|quantity|unit|price|"
                r"net|worth|vat|tax|gross|amount|total|no\.?",
                next_line_text,
                re.IGNORECASE,
            ):
                break
            header_words.extend(lines[next_index][1])
            last_header_index = next_index
        header_words.sort(key=lambda word: word["x0"])
        header_text = " ".join(word["text"] for word in header_words)
        if not re.search(r"description|details|product|item", header_text, re.I):
            continue
        if not re.search(
            r"amount|worth|line\s+total|gross|net|total", header_text, re.I
        ):
            continue

        columns = {}

        def find_header_position(pattern: str, word_pattern: str) -> Optional[float]:
            for start in range(len(header_words)):
                for end in range(start + 1, min(start + 4, len(header_words) + 1)):
                    phrase = " ".join(
                        item["text"].strip(":#") for item in header_words[start:end]
                    )
                    if re.fullmatch(pattern, phrase, re.IGNORECASE):
                        matched_words = header_words[start:end]
                        return sum(
                            (item["x0"] + item["x1"]) / 2 for item in matched_words
                        ) / len(matched_words)
            for word in header_words:
                if word_pattern and re.fullmatch(
                    word_pattern, word["text"].strip(":#"), re.IGNORECASE
                ):
                    return (word["x0"] + word["x1"]) / 2
            return None

        header_patterns = {
            "line_number": (r"(?:no\.?|#|item\s+no\.?)", r""),
            "description": (r"description|details|product|item", r"description|details|product|item"),
            "quantity": (r"qty|quantity", r"qty|quantity"),
            "unit_measure": (r"unit\s+measure|u\.?m\.?", r"u\.?m\.?"),
            "unit_price": (
                r"(?:unit\s+price|net\s+price|price\s+each|rate)",
                r"price|rate",
            ),
            "net_amount": (r"net\s+(?:worth|amount|total)", r""),
            "vat_rate": (r"vat\s*(?:\[%\]|rate)?|tax\s*(?:\[%\]|rate)?", r"vat|tax"),
            "vat_amount": (r"(?:vat|tax)\s+amount", r""),
            "gross_amount": (r"gross\s+(?:worth|amount|total)", r""),
            "amount": (r"(?:line\s+total|amount|total)", r"amount|total"),
        }
        for column, (phrase_pattern, word_pattern) in header_patterns.items():
            position = find_header_position(phrase_pattern, word_pattern)
            if position is not None:
                columns[column] = position

        amount_column = next(
            (
                name for name in ("gross_amount", "amount", "net_amount")
                if name in columns
            ),
            None,
        )
        if "description" not in columns or amount_column is None:
            continue

        ordered_columns = sorted(columns.items(), key=lambda item: item[1])
        boundaries = [
            (ordered_columns[index][1] + ordered_columns[index + 1][1]) / 2
            for index in range(len(ordered_columns) - 1)
        ]
        description_index = next(
            index
            for index, (name, _) in enumerate(ordered_columns)
            if name == "description"
        )
        description_header_words = [
            word for word in header_words
            if re.fullmatch(
                r"description|details|product|item", word["text"].strip(":#"), re.I
            )
        ]
        description_left = (
            min(word["x0"] for word in description_header_words)
            if description_header_words else 0
        )
        # Header centers are poor column boundaries for a wide description field.
        # Stop at the left edge of the quantity column so wrapped item names survive.
        quantity_header_words = [
            word for word in header_words
            if re.fullmatch(r"qty|quantity", word["text"].strip(":#"), re.I)
        ]
        description_right = (
            min(word["x0"] for word in quantity_header_words)
            if quantity_header_words else float("inf")
        )
        found = []
        for _, row_words in lines[last_header_index + 1:]:
            row_text = " ".join(word["text"] for word in row_words).strip()
            if re.match(
                r"^(?:sub\s*total|net\s*total|vat|tax|total|amount\s+due|grand\s+total)\b",
                row_text,
                re.I,
            ):
                break

            cells = {column: [] for column in columns}
            for word in row_words:
                center = (word["x0"] + word["x1"]) / 2
                position = sum(center >= boundary for boundary in boundaries)
                column = ordered_columns[position][0]
                cells[column].append(word["text"])

            description_words = [
                word["text"] for word in row_words
                if description_left
                <= (word["x0"] + word["x1"]) / 2
                < description_right
            ]
            description = _clean_value(" ".join(description_words))
            if not description:
                description = _clean_value(" ".join(cells.get("description", [])))
            amount = _parse_number(" ".join(cells.get(amount_column, [])))
            net_amount = _parse_number(" ".join(cells.get("net_amount", [])))
            vat_amount = _parse_number(" ".join(cells.get("vat_amount", [])))
            if amount is None and net_amount is not None:
                amount = net_amount
            if not description:
                continue
            if amount is None:
                if found:
                    found[-1].description = _clean_value(
                        f"{found[-1].description} {description}"
                    )
                continue
            item = InvoiceLineItem(
                description=description,
                quantity=_parse_number(" ".join(cells.get("quantity", []))),
                unit_measure=_clean_value(" ".join(cells.get("unit_measure", []))),
                unit_price=_parse_number(" ".join(cells.get("unit_price", []))),
                vat_rate=_parse_percentage(" ".join(cells.get("vat_rate", []))),
                net_amount=net_amount,
                vat_amount=vat_amount,
                amount=amount,
            )
            _complete_line_item(item)
            found.append(item)
        if found:
            return found
    return []


def _extract_party_details_from_columns(pages) -> dict[str, Optional[str]]:
    details = {
        "seller_name": None,
        "seller_address": None,
        "seller_tax_id": None,
        "client_name": None,
        "client_address": None,
        "client_tax_id": None,
        "iban_number": None,
    }
    party_fields = {
        "supplier": "seller",
        "seller": "seller",
        "client": "client",
        "customer": "client",
        "buyer": "client",
    }
    tax_id_pattern = re.compile(
        r"\b(?:tax\s*(?:identification\s*)?(?:id|no\.?|number)|"
        r"vat\s*(?:id|no\.?|number))\s*[:#-]?\s*(.+)$",
        re.IGNORECASE,
    )
    iban_pattern = re.compile(
        r"\biban(?:\s*(?:no\.?|number))?\s*[:#-]?\s*([A-Z0-9 ]{8,})",
        re.IGNORECASE,
    )
    section_pattern = re.compile(
        r"^(?:items?|description|no\.?\s+description)\b", re.IGNORECASE
    )

    for page in pages:
        words = page.extract_words()
        lines = []
        for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
            if not lines or abs(lines[-1][0] - word["top"]) > 3:
                lines.append((word["top"], [word]))
            else:
                lines[-1][1].append(word)

        headers = []
        for word in words:
            normalized = re.sub(r"[^a-z]", "", word["text"].lower())
            if normalized in party_fields:
                headers.append({
                    "field": party_fields[normalized],
                    "x": (word["x0"] + word["x1"]) / 2,
                    "bottom": word["bottom"],
                })
        headers.sort(key=lambda header: header["x"])

        for index, header in enumerate(headers):
            left = (headers[index - 1]["x"] + header["x"]) / 2 if index else 0
            right = (
                (header["x"] + headers[index + 1]["x"]) / 2
                if index + 1 < len(headers)
                else page.width
            )
            party = header["field"]
            party_lines = []
            for top, line_words in lines:
                if top <= header["bottom"]:
                    continue
                value = " ".join(
                    word["text"] for word in line_words
                    if left <= (word["x0"] + word["x1"]) / 2 < right
                ).strip()
                if not value:
                    continue
                if section_pattern.match(value) or (
                    re.search(r"\b(?:qty|quantity)\b", value, re.IGNORECASE)
                    and re.search(
                        r"\b(?:net\s+(?:price|worth)|gross\s+worth|vat)\b",
                        value,
                        re.IGNORECASE,
                    )
                ):
                    break
                party_lines.append(value)

            name_key = f"{party}_name"
            address_key = f"{party}_address"
            tax_id_key = f"{party}_tax_id"
            address_lines = []
            for value in party_lines:
                tax_match = tax_id_pattern.search(value)
                iban_match = iban_pattern.search(value)
                if tax_match:
                    details[tax_id_key] = _clean_value(tax_match.group(1))
                    break
                elif iban_match:
                    details["iban_number"] = re.sub(
                        r"\s+", "", iban_match.group(1)
                    )
                elif not details[name_key]:
                    details[name_key] = _clean_value(value)
                else:
                    address_lines.append(value)

            if address_lines:
                details[address_key] = _clean_value(" ".join(address_lines))

    return details


def extract_invoice_record(text: str, pages) -> InvoiceRecord:
    common_labels = (
        r"invoice\s*(?:number|no\.?|#)",
        r"date\s+of\s+issue",
        r"issue\s+date",
        r"invoice\s+date",
        r"supplier(?:\s+name)?",
        r"seller(?:\s+name)?",
        r"client(?:\s+name)?",
        r"customer(?:\s+name)?",
        r"buyer(?:\s+name)?",
        r"bill\s+to",
        r"billed\s+to",
        r"sold\s+to",
        r"invoice\s+to",
        r"vat\s*(?:registration\s*)?(?:number|no\.?|id)",
        r"vat\s*no\.?",
        r"tax\s*(?:identification\s*)?(?:number|no\.?|id)",
        r"net\s*(?:amount|total|subtotal)",
        r"subtotal",
        r"vat\s*(?:amount|total)",
        r"tax\s*(?:amount|total)",
        r"gross\s*(?:amount|total)",
        r"grand\s*total",
        r"total\s+due",
        r"amount\s+due",
        r"balance\s+due",
        r"from",
        r"to",
        r"tax\s*id",
    )

    party_details = _extract_party_details_from_columns(pages)
    seller_name = party_details["seller_name"]
    if not seller_name:
        seller_name = _extract_labeled_value(
            text,
            (r"supplier(?:\s+name)?", r"seller(?:\s+name)?"),
            common_labels,
        )

    client_name = party_details["client_name"]
    if not client_name:
        client_name = _extract_labeled_value(
            text,
            (
                r"client(?:\s+name)?",
                r"customer(?:\s+name)?",
                r"buyer(?:\s+name)?",
                r"bill\s+to",
                r"billed\s+to",
                r"sold\s+to",
                r"invoice\s+to",
            ),
            common_labels,
        )

    net_amount = _extract_labeled_amount(
        text,
        (
            r"net\s*(?:amount|total|worth)",
            r"sub\s*total",
            r"taxable\s+amount",
            r"amount\s+before\s+tax",
            r"total\s+excluding\s+(?:vat|tax)",
            r"total\s+excl\.?\s*(?:vat|tax)",
        ),
    )
    if net_amount is None:
        net_amount = _extract_table_amount(
            pages,
            (
                r"net(?:\s+(?:amount|total|worth))?",
                r"sub\s*total",
                r"taxable\s+amount",
            ),
        )

    vat_amount = _extract_labeled_amount(
        text,
        (
            r"vat(?!\s*(?:registration|number|no\.?|id))(?:\s*(?:amount|total))?",
            r"tax(?!\s*(?:identification|number|no\.?|id))(?:\s*(?:amount|total))?",
            r"sales\s+tax",
        ),
        excluded_line=re.compile(
            r"\b(?:vat|tax)\s*(?:\[%\]|rate)(?:\s|$)", re.IGNORECASE
        ),
    )
    if vat_amount is None:
        vat_amount = _extract_table_amount(
            pages, (r"vat(?:\s*(?:amount|total))?", r"tax(?:\s*(?:amount|total))?")
        )

    gross_amount = _extract_labeled_amount(
        text,
        (
            r"gross(?:\s*(?:amount|total|worth))?",
            r"grand\s*total",
            r"total\s+due",
            r"amount\s+due",
            r"balance\s+due",
            r"total",
        ),
        excluded_line=re.compile(
            r"^\s*(?:net|sub\s*total|subtotal|vat|tax)\b", re.IGNORECASE
        ),
    )
    if gross_amount is None:
        gross_amount = _extract_table_amount(
            pages,
            (
                r"gross(?:\s*(?:amount|total|worth))?",
                r"grand\s*total",
                r"total\s+due",
                r"amount\s+due",
                r"balance\s+due",
                r"total",
            ),
        )

    line_items = extract_line_items(pages)
    if line_items and all(
        item.net_amount is not None for item in line_items
    ):
        net_amount = round(sum(item.net_amount or 0.0 for item in line_items), 2)
    if line_items and all(
        item.amount is not None for item in line_items
    ):
        gross_amount = round(sum(item.amount or 0.0 for item in line_items), 2)
    if line_items and all(item.vat_amount is not None for item in line_items):
        vat_amount = round(sum(item.vat_amount or 0.0 for item in line_items), 2)
    elif net_amount is not None and gross_amount is not None:
        vat_amount = round(gross_amount - net_amount, 2)

    tax_ids = re.findall(
        r"\b(?:tax\s*(?:identification\s*)?(?:id|no\.?|number)|"
        r"vat\s*(?:id|no\.?|number))\s*[:#-]?\s*"
        r"([A-Z0-9][A-Z0-9-]{4,})",
        text,
        re.IGNORECASE,
    )
    seller_tax_id = party_details["seller_tax_id"]
    client_tax_id = party_details["client_tax_id"]
    if tax_ids:
        seller_tax_id = seller_tax_id or tax_ids[0]
    if len(tax_ids) > 1:
        client_tax_id = client_tax_id or tax_ids[1]
    iban_match = re.search(
        r"\biban(?:\s*(?:no\.?|number))?\s*[:#-]?\s*"
        r"([A-Z]{2}\d{2}(?:\s*[A-Z0-9]){10,32})",
        text,
        re.IGNORECASE,
    )

    return InvoiceRecord(
        seller_name=seller_name,
        seller_address=party_details["seller_address"],
        seller_tax_id=seller_tax_id,
        client_name=client_name,
        client_address=party_details["client_address"],
        client_tax_id=client_tax_id,
        iban_number=(
            party_details["iban_number"]
            or (
                re.sub(r"\s+", "", iban_match.group(1))
                if iban_match
                else None
            )
        ),
        invoice_number=_extract_labeled_value(
            text,
            (r"invoice\s*(?:number|no\.?|#)",),
            common_labels,
        ),
        invoice_date=_extract_labeled_value(
            text,
            (r"date\s+of\s+issue", r"issue\s+date", r"invoice\s+date"),
            common_labels,
        ),
        net_amount=net_amount,
        vat_amount=vat_amount,
        gross_amount=gross_amount,
        line_items=line_items,
    )


def _line_item_output(item: InvoiceLineItem) -> dict:
    """Serialize a line item using names that mirror the PDF columns."""
    return {
        "description": item.description,
        "quantity": item.quantity,
        "unit_measure": item.unit_measure,
        "net_price": _format_currency(item.unit_price),
        "net_worth": _format_currency(item.net_amount),
        "vat_in_percent": item.vat_rate,
        "gross_worth": _format_currency(item.amount),
    }


def _format_currency(value: Optional[float]) -> Optional[str]:
    return f"${value:,.2f}" if value is not None else None


def _record_output(record: InvoiceRecord) -> dict:
    output = asdict(record)
    for field_name in ("net_amount", "vat_amount", "gross_amount"):
        output[field_name] = _format_currency(output[field_name])
    output["line_items"] = [_line_item_output(item) for item in record.line_items]
    return output


def main():
    st.set_page_config(page_title="Invoice data extraction", page_icon="📄")
    st.title("Invoice data extraction")
    st.write(
        "Upload a text-based invoice PDF to extract its details and line items. "
        "Scanned PDFs need OCR before their text can be extracted."
    )

    uploaded_pdf = st.file_uploader("Upload an invoice PDF", type=["pdf"])
    if uploaded_pdf is None:
        return

    try:
        with pdfplumber.open(io.BytesIO(uploaded_pdf.getvalue())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                st.error("No selectable text found. Scanned PDFs require OCR.")
                return
            record = extract_invoice_record(text, pdf.pages)
    except (PDFSyntaxError, PDFPasswordIncorrect, OSError, ValueError) as error:
        st.error(f"Could not read the PDF: {error}")
        return

    structured_record = _record_output(record)
    st.subheader("Extracted invoice details")
    st.json(structured_record)

    if record.line_items:
        st.subheader("Line items")
        st.dataframe(
            [_line_item_output(item) for item in record.line_items],
            use_container_width=True,
        )
    else:
        st.warning(
            "No line-item table was recognized. Extraction depends on the PDF "
            "having selectable text and a detectable table structure."
        )
    st.caption(
        "A null field means the extractor did not find a value in the PDF; "
        "an empty line_items list means no invoice rows were recognized."
    )

    st.download_button(
        "Download structured JSON",
        data=json.dumps(structured_record, indent=2, ensure_ascii=False),
        file_name="invoice_data.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
