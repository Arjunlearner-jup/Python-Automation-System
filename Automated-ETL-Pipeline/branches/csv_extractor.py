"""CSV-only ingestion helpers for local paths and uploaded file objects."""

from __future__ import annotations

import codecs
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO

import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000
SUPPORTED_DELIMITERS = (",", ";", "\t")
SAMPLE_SIZE = 64 * 1024


@contextmanager
def _rewound_source(source: str | Path | BinaryIO | TextIO) -> Iterator[Any]:
    """Yield a source from its beginning and restore uploaded stream position."""
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as file_obj:
            yield file_obj
        return

    try:
        original_position = source.tell()
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise OSError("Uploaded CSV file must support seeking.") from exc

    try:
        yield source
    finally:
        source.seek(original_position)


def _source_name(source: str | Path | BinaryIO | TextIO) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return Path(str(name)).name if name else "uploaded.csv"


def _read_sample(source: str | Path | BinaryIO | TextIO) -> bytes:
    with _rewound_source(source) as file_obj:
        sample = file_obj.read(SAMPLE_SIZE)
    if isinstance(sample, str):
        return sample.encode("utf-8")
    return sample


def _detect_encoding(sample: bytes) -> str:
    bom_encodings = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )
    for bom, encoding in bom_encodings:
        if sample.startswith(bom):
            return encoding

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError("Could not decode the CSV sample using supported encodings.")


def detect_csv_delimiter(
    source: str | Path | BinaryIO | TextIO, encoding: str | None = None
) -> str:
    """Detect comma, semicolon, or tab delimiters from a file sample."""
    sample = _read_sample(source)
    if not sample.strip():
        raise ValueError("CSV file is empty.")

    selected_encoding = encoding or _detect_encoding(sample)
    text = sample.decode(selected_encoding)
    try:
        return csv.Sniffer().sniff(text, delimiters="".join(SUPPORTED_DELIMITERS)).delimiter
    except csv.Error:
        rows = [line for line in text.splitlines() if line.strip()]
        counts = {
            delimiter: sum(line.count(delimiter) for line in rows[:20])
            for delimiter in SUPPORTED_DELIMITERS
        }
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            raise ValueError(
                "Could not detect the CSV delimiter; specify delimiter explicitly."
            )
        return delimiter


def get_csv_metadata(
    data: pd.DataFrame,
    source: str,
    delimiter: str,
    encoding: str,
    status: str = "success",
    error: str | None = None,
) -> dict[str, Any]:
    """Build extraction metadata for a successfully extracted DataFrame."""
    metadata: dict[str, Any] = {
        "source": source,
        "source_type": "File",
        "format": "CSV",
        "delimiter": delimiter,
        "encoding": encoding,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": len(data.index),
        "columns": len(data.columns),
        "column_names": list(data.columns),
        "status": status,
    }
    if error is not None:
        metadata["error"] = error
    return metadata


def _failed_result(
    source_name: str,
    error: Exception,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> dict[str, Any]:
    return {
        "data": None,
        "metadata": {
            "source": source_name,
            "source_type": "File",
            "format": "CSV",
            "delimiter": delimiter,
            "encoding": encoding,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "status": "error",
            "error": str(error),
        },
    }


def extract_csv_file(
    source: str | Path | BinaryIO | TextIO,
    delimiter: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Extract one CSV into a DataFrame and return it with extraction metadata.

    ``source`` may be a local path or a seekable uploaded file object, such as a
    Streamlit ``UploadedFile``. Parsing is performed in chunks to limit temporary
    parser memory; the returned DataFrame still contains the complete dataset.
    Errors are represented in the result rather than raised to the caller.
    """
    source_name = _source_name(source)
    chosen_delimiter: str | None = delimiter
    chosen_encoding: str | None = None

    try:
        if Path(source_name).suffix.lower() != ".csv":
            raise ValueError(f"Unsupported file format for '{source_name}'; expected .csv.")
        if delimiter is not None and delimiter not in SUPPORTED_DELIMITERS:
            raise ValueError("Delimiter must be comma, semicolon, or tab.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if isinstance(source, (str, Path)) and not Path(source).is_file():
            raise FileNotFoundError(f"CSV file not found: {source}")

        sample = _read_sample(source)
        if not sample.strip():
            raise ValueError("CSV file is empty.")
        selected_encoding = _detect_encoding(sample)
        try:
            detected_delimiter = detect_csv_delimiter(source, selected_encoding)
        except ValueError:
            detected_delimiter = None

        parsed_delimiter = delimiter or detected_delimiter or ","
        chosen_encoding = selected_encoding
        chosen_delimiter = parsed_delimiter
        if (
            delimiter is not None
            and detected_delimiter is not None
            and delimiter != detected_delimiter
        ):
            raise ValueError(
                f"Selected delimiter {delimiter!r} does not match detected "
                f"delimiter {detected_delimiter!r}."
            )

        with _rewound_source(source) as file_obj:
            reader = pd.read_csv(
                file_obj,
                sep=parsed_delimiter,
                encoding=selected_encoding,
                chunksize=chunk_size,
                on_bad_lines="error",
            )
            try:
                chunks = list(reader)
            finally:
                reader.close()
        data = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame()
        )
        return {
            "data": data,
            "metadata": get_csv_metadata(
                data,
                source_name,
                parsed_delimiter,
                selected_encoding,
            ),
        }
    except (
        OSError,
        UnicodeError,
        csv.Error,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        return _failed_result(
            source_name, error, chosen_delimiter, chosen_encoding
        )


def extract_multiple_csv_files(
    sources: list[str | Path | BinaryIO | TextIO],
    delimiter: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Extract each CSV independently, returning one result per input source."""
    return [
        extract_csv_file(source, delimiter=delimiter, chunk_size=chunk_size)
        for source in sources
    ]
