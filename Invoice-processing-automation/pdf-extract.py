import io
import re

import pdfplumber
import streamlit as st
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError


FIELD_LABELS = {
    "Invoice no": (r"invoice\s*(?:number|no\.?|#)",),
    "Date of issue": (r"date\s+of\s+issue", r"issue\s+date"),
    "Seller": (r"seller", r"supplier"),
    "Client": (r"client", r"customer", r"buyer", r"bill\s+to"),
}


def extract_value(document_text, labels):
    lines = document_text.splitlines()
    all_labels = [
        (field, label)
        for field, aliases in FIELD_LABELS.items()
        for label in aliases
    ]

    for index, line in enumerate(lines):
        matching_labels = []
        for field, label in all_labels:
            match = re.search(
                rf"(?<!\w)(?:{label})(?![a-z])\s*[:#-]?\s*",
                line,
                re.IGNORECASE,
            )
            if match:
                matching_labels.append((match.start(), match.end(), field))

        for field, field_aliases in FIELD_LABELS.items():
            if field_aliases != labels:
                continue
            for label in field_aliases:
                match = re.search(
                    rf"(?<!\w)(?:{label})(?![a-z])\s*[:#-]?\s*",
                    line,
                    re.IGNORECASE,
                )
                if not match:
                    continue

                next_label_start = min(
                    (
                        start for start, _, next_field in matching_labels
                        if start >= match.end() and next_field != field
                    ),
                    default=len(line),
                )
                field_value = line[match.end():next_label_start].strip(" \t:;,#-")
                if field_value:
                    return field_value

                for following_line in lines[index + 1:]:
                    following_line = following_line.strip()
                    if not following_line:
                        continue
                    if any(
                        re.search(
                            rf"(?<!\w)(?:{other_label})(?![a-z])\s*[:#-]?",
                            following_line,
                            re.IGNORECASE,
                        )
                        for _, other_label in all_labels
                    ):
                        break
                    return following_line
    return ""


def extract_party_names(pages):
    party_labels = {
        "Seller": {"seller", "supplier"},
        "Client": {"client", "customer", "buyer"},
    }
    results = {field: "" for field in party_labels}
    section_heading = re.compile(
        r"^(?:INVOICE ITEMS|VAT SUMMARY|ITEMS?|TOTALS?|PAYMENT|NOTES|"
        r"TERMS|SUBTOTAL)(?:\s|$)",
        re.IGNORECASE,
    )

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
            normalized = re.sub(r"[^a-z]", "", word["text"].lower())
            for field, aliases in party_labels.items():
                if normalized in aliases:
                    headers.append({
                        "field": field,
                        "x": (word["x0"] + word["x1"]) / 2,
                        "bottom": word["bottom"],
                    })

        headers.sort(key=lambda header: header["x"])
        if not headers:
            continue

        for header_index, header in enumerate(headers):
            field = header["field"]
            if results[field]:
                continue

            left = (
                (headers[header_index - 1]["x"] + header["x"]) / 2
                if header_index > 0
                else 0
            )
            right = (
                (header["x"] + headers[header_index + 1]["x"]) / 2
                if header_index + 1 < len(headers)
                else page.width
            )
            column_lines = []
            for line_top, line_words in page_lines:
                if line_top <= header["bottom"]:
                    continue
                if line_top > header["bottom"] + 350:
                    break

                whole_line = " ".join(word["text"] for word in line_words).strip()
                if section_heading.match(whole_line):
                    break

                column_text = " ".join(
                    word["text"] for word in line_words
                    if left <= (word["x0"] + word["x1"]) / 2 < right
                ).strip()
                if column_text:
                    column_lines.append(column_text)

            results[field] = "\n".join(column_lines)

    return results


st.set_page_config(page_title="Invoice details", page_icon="📄")
st.title("Extract invoice details")

uploaded_pdf = st.file_uploader("Upload an invoice PDF", type=["pdf"])

if uploaded_pdf is not None:
    try:
        with pdfplumber.open(io.BytesIO(uploaded_pdf.getvalue())) as pdf:
            document_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
            party_names = extract_party_names(pdf.pages)

        if not document_text.strip():
            st.error("No selectable text found. Scanned PDFs require OCR.")
        else:
            result = {
                field: extract_value(document_text, labels)
                for field, labels in FIELD_LABELS.items()
            }
            result.update(party_names)

            st.subheader("Extracted data")
            for field, field_value in result.items():
                if field in party_names:
                    st.text_area(
                        field,
                        value=field_value or "Not found",
                        disabled=True,
                        height=150,
                    )
                else:
                    st.text_input(
                        field,
                        value=field_value or "Not found",
                        disabled=True,
                    )

            missing = [
                field for field, field_value in result.items() if not field_value
            ]
            if missing:
                st.warning(
                    "Could not find: " + ", ".join(missing)
                    + ". Check the PDF text and adjust the label patterns if needed."
                )
    except (PDFSyntaxError, PDFPasswordIncorrect, OSError, ValueError) as error:
        st.error(f"Could not read the PDF: {error}")
