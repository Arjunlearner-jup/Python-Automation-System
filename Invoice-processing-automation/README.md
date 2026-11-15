# Invoice Processing

A Streamlit application for uploading PDF invoices, extracting and cleaning invoice data, validating totals, and exporting results to CSV or MySQL.

## Features and processing logic

The streamlit output guides each invoice through four sidebar pages:

1. **Upload Invoice**
   - Accepts one or more PDF files.
   - Checks that the file begins with the PDF signature and uses SHA-256 to reject duplicate file contents.
   - Keeps uploaded PDF bytes and processing results in Streamlit session state for the current session.

2. **Invoice extraction**
   - Extracts selectable text from each PDF with `pdfplumber`. Scanned PDFs without selectable text are not OCR-processed and cannot be extracted.
   - Extracts common invoice fields, including seller/client details, invoice number and date, tax IDs, IBAN, totals, and line items from tables or positioned text.
   - Infers some missing line-item amounts and invoice totals when enough values are available.
   - Displays the structured result and offers an extracted JSON download.

3. **Data cleaning and extraction**
   - Normalizes invoice numbers to `INV-...`, parses dates and numbers, cleans text, and validates IBAN country codes and lengths.
   - Replaces hyphens in seller and client names with spaces and strips non-digits from tax IDs.
   - Checks required invoice fields and warns if net amount plus VAT does not match the gross amount.
   - Offers cleaned JSON and CSV downloads. CSV output has separate rows for invoice fields and line-item fields.
   - Detects currency from a currency code or symbol, or uses the chosen fallback. Amount fields are set to `null` for currencies other than USD; the app does not convert currency.

4. **Data save in Mysql and csv export or report**
   - Can save a selected cleaned invoice to a MySQL `cleaned_invoice_data` table, with a submission ID, section, field name, and value.
   - Provides a CSV report covering all cleaned invoices in the current session.

The output shows an extracted field as missing when it could not find a value in the PDF. Extraction depends on the invoice's labels and layout, so review the results before saving.

## Requirements

Install the packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

## Run the Main_Complete

The current project entry point is `Main_Complete.py`:

```powershell
streamlit run Main_Complete.py
```

The PDF extraction page calls the PDF parser and serializer in `Invoicedataextraction.py`; keep that module alongside the app.

## MySQL configuration

On the **Data save in Mysql and csv export or report** page, expand **MySQL connection** and enter the MySQL host, port, username, password, and database name. The database name may contain only letters, numbers, and underscores. The MySQL account must be allowed to create the database and table and insert rows.

Alternatively, set these environment variables before starting Streamlit:

| Variable | Default |
|---|---|
| `MYSQL_HOST` | `localhost` |
| `MYSQL_PORT` | `3306` |
| `MYSQL_USER` | `root` |
| `MYSQL_PASSWORD` | empty or your password |
| `MYSQL_DATABASE` | `invoice_processing` |

## Tests

Run the invoice-processing unit tests with:

```powershell
python -m unittest -v
```
