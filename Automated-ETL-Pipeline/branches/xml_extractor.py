"""XML-only ingestion helpers for local paths and uploaded file objects."""

from __future__ import annotations

import hashlib
import io
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from xml.etree import ElementTree as element_tree

import pandas as pd


DEFAULT_RAW_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "xml"
)


def _source_name(source: str | Path | BinaryIO) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return Path(str(name)).name if name else "uploaded.xml"


def _read_source(source: str | Path | BinaryIO) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"XML file not found: {path}")
        return path.read_bytes()

    try:
        original_position = source.tell()
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise OSError("Uploaded XML file must support seeking.") from exc

    try:
        contents = source.read()
    finally:
        source.seek(original_position)

    if not isinstance(contents, bytes):
        raise OSError("Could not read uploaded XML file as bytes.")
    return contents


def _save_original(
    contents: bytes, filename: str, raw_directory: str | Path
) -> Path:
    """Archive the source bytes without replacing existing files."""
    directory = Path(raw_directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    original = Path(safe_name)
    stem = original.stem or "uploaded"
    suffix = original.suffix or ".xml"

    index = 0
    while True:
        name = f"{stem}{suffix}" if index == 0 else f"{stem}_{index}{suffix}"
        destination = directory / name
        try:
            with destination.open("xb") as output:
                output.write(contents)
            return destination
        except FileExistsError:
            index += 1


def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def parse_xml_document(xml_content: bytes | str) -> element_tree.Element:
    """Parse XML bytes or text and return its complete root element."""
    if not xml_content or not xml_content.strip():
        raise ValueError("XML file is empty.")
    try:
        root = element_tree.fromstring(xml_content)
    except element_tree.ParseError as exc:
        line, column = exc.position
        raise ValueError(
            f"Malformed XML at line {line}, column {column}: {exc}"
        ) from exc
    return root


def _node_value(element: element_tree.Element) -> Any:
    """Convert one element while retaining attributes and repeated children."""
    value: dict[str, Any] = {}
    if element.attrib:
        value["@attributes"] = dict(element.attrib)
    text = element.text
    if text is not None and text.strip():
        value["text"] = text

    grouped: dict[str, list[Any]] = {}
    for child in list(element):
        grouped.setdefault(str(child.tag), []).append(_node_value(child))
    local_counts: defaultdict[str, int] = defaultdict(int)
    for tag in grouped:
        local_counts[_local_name(tag)] += 1
    for tag, children in grouped.items():
        label = tag if local_counts[_local_name(tag)] > 1 else _local_name(tag)
        value[label] = children[0] if len(children) == 1 else children

    if not value:
        return text or ""
    if "text" in value and len(value) == 1:
        return value["text"]
    return value


def _element_record(
    element: element_tree.Element, element_id: str, parent_id: str | None
) -> dict[str, Any]:
    """Create one DataFrame row without flattening nested XML children."""
    record: dict[str, Any] = {
        "_element_id": element_id,
        "_parent_id": parent_id,
    }
    if element.attrib:
        record["@attributes"] = dict(element.attrib)
    if element.text is not None and element.text.strip():
        record["text"] = element.text

    child_groups: dict[str, list[element_tree.Element]] = {}
    for child in list(element):
        child_groups.setdefault(str(child.tag), []).append(child)
    local_counts: defaultdict[str, int] = defaultdict(int)
    for tag in child_groups:
        local_counts[_local_name(tag)] += 1
    for tag, children in child_groups.items():
        values = [_node_value(child) for child in children]
        label = tag if local_counts[_local_name(tag)] > 1 else _local_name(tag)
        record[label] = values[0] if len(values) == 1 else values
    return record


def _records_and_groups(
    root: element_tree.Element,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Build DataFrames for repeated same-tag siblings, retaining parent links."""
    groups: dict[str, list[tuple[element_tree.Element, str | None, str]]] = {}
    repeated_paths: set[str] = set()
    repeating_metadata: list[dict[str, Any]] = []

    def visit(
        parent: element_tree.Element, parent_id: str | None, parent_path: str
    ) -> None:
        children = list(parent)
        totals: defaultdict[str, int] = defaultdict(int)
        for child in children:
            totals[str(child.tag)] += 1
        seen: defaultdict[str, int] = defaultdict(int)
        for child in children:
            tag = str(child.tag)
            seen[tag] += 1
            element_id = f"{parent_id or 'root'}/{tag}[{seen[tag]}]"
            child_path = f"{parent_path}/{tag}"
            groups.setdefault(child_path, []).append((child, parent_id, element_id))
            if totals[tag] > 1:
                repeated_paths.add(child_path)
            visit(child, element_id, child_path)

    root_tag = _local_name(str(root.tag))
    visit(root, None, f"/{root_tag}")

    dataframes: dict[str, pd.DataFrame] = {}
    for path, records in groups.items():
        if path not in repeated_paths:
            continue
        table_name = _table_name(path)
        if table_name in dataframes:
            table_name = (
                f"{table_name}_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:8]}"
            )
        dataframes[table_name] = pd.DataFrame(
            [
                _element_record(element, element_id, parent_id)
                for element, parent_id, element_id in records
            ]
        )
        repeating_metadata.append(
            {
                "path": path,
                "element": path.rsplit("/", 1)[-1],
                "count": len(records),
                "dataframe": table_name,
            }
        )
    return dataframes, repeating_metadata


def _table_name(path: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", path.strip("/"))
    if "{" in path:
        name = f"{name}_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:8]}"
    return name


def _hierarchy_dataframe(root: element_tree.Element) -> pd.DataFrame:
    """Represent every XML node as a row with explicit parent-child links."""
    rows: list[dict[str, Any]] = []

    def visit(
        element: element_tree.Element,
        element_id: str,
        parent_id: str | None,
        path: str,
    ) -> None:
        children = list(element)
        rows.append(
            {
                "element_id": element_id,
                "parent_id": parent_id,
                "path": path,
                "tag": str(element.tag),
                "local_name": _local_name(str(element.tag)),
                "attributes": dict(element.attrib),
                "text": element.text,
                "tail": element.tail,
                "child_count": len(children),
            }
        )
        seen: defaultdict[str, int] = defaultdict(int)
        for child in children:
            tag = str(child.tag)
            seen[tag] += 1
            child_id = f"{element_id}/{tag}[{seen[tag]}]"
            visit(child, child_id, element_id, f"{path}/{tag}[{seen[tag]}]")

    root_name = _local_name(str(root.tag))
    visit(root, "root", None, f"/{root_name}")
    return pd.DataFrame(rows)


def convert_xml_to_dataframe(
    xml_document: element_tree.Element, record_path: str | None = None
) -> dict[str, pd.DataFrame]:
    """Create related DataFrames for repeated siblings and all hierarchy nodes.

    Repeated sibling elements receive a table per structural path. Nested child
    values remain nested dictionaries/lists in their parent record table, while
    ``xml_hierarchy`` provides a row for every node and explicit parent IDs.
    ``record_path`` optionally selects one repeated-element table as ``records``.
    """
    repeated_dataframes, repeating_metadata = _records_and_groups(xml_document)
    result = dict(repeated_dataframes)
    result["xml_hierarchy"] = _hierarchy_dataframe(xml_document)
    if record_path is not None:
        normalized_path = "/" + record_path.strip("/")
        matching = next(
            (
                item["dataframe"]
                for item in repeating_metadata
                if item["path"] == normalized_path
            ),
            "",
        )
        if not matching:
            raise ValueError(f"No repeated XML record table matches {record_path!r}.")
        result["records"] = repeated_dataframes[matching]
    elif repeated_dataframes:
        result["records"] = next(iter(repeated_dataframes.values()))
    else:
        result["records"] = result["xml_hierarchy"]
    return result


def get_xml_metadata(
    xml_document: element_tree.Element,
    dataframes: dict[str, pd.DataFrame],
    source: str,
    file_size: int,
    namespaces: dict[str, str] | None = None,
    status: str = "success",
    error: str | None = None,
    raw_file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build dynamic metadata for the parsed XML tree and related tables."""
    all_elements = list(xml_document.iter())
    _, repeating_elements = _records_and_groups(xml_document)
    metadata: dict[str, Any] = {
        "source": source,
        "source_type": "File",
        "format": "XML",
        "file_size": file_size,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "root_element": str(xml_document.tag),
        "namespaces": namespaces or {},
        "element_count": len(all_elements),
        "attribute_count": sum(len(element.attrib) for element in all_elements),
        "repeating_record_elements": repeating_elements,
        "dataframes": {
            name: {"rows": len(frame.index), "columns": len(frame.columns)}
            for name, frame in dataframes.items()
        },
        "status": status,
    }
    if raw_file_path is not None:
        metadata["raw_file_path"] = str(raw_file_path)
    if error is not None:
        metadata["error"] = error
    return metadata


def _find_namespaces(contents: bytes) -> dict[str, str]:
    namespaces: dict[str, str] = {}
    try:
        for _, declaration in element_tree.iterparse(
            io.BytesIO(contents), events=("start-ns",)
        ):
            prefix, uri = declaration
            namespaces[prefix or ""] = uri
    except element_tree.ParseError as exc:
        line, column = exc.position
        raise ValueError(
            f"Malformed XML at line {line}, column {column}: {exc}"
        ) from exc
    return namespaces


def _failed_result(
    source_name: str,
    file_size: int,
    error: Exception,
    raw_file_path: Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source_name,
        "source_type": "File",
        "format": "XML",
        "file_size": file_size,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "root_element": None,
        "namespaces": {},
        "element_count": 0,
        "attribute_count": 0,
        "repeating_record_elements": [],
        "dataframes": {},
        "status": "error",
        "error": str(error),
    }
    if raw_file_path is not None:
        metadata["raw_file_path"] = str(raw_file_path)
    return {
        "data": None,
        "dataframes": {},
        "xml_tree": None,
        "metadata": metadata,
    }


def extract_xml_file(
    source: str | Path | BinaryIO,
    record_path: str | None = None,
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> dict[str, Any]:
    """Parse one XML source, preserve its original bytes, and return related data.

    The result includes ``xml_tree`` (the complete ElementTree root), ``data``
    (the selected record DataFrame), ``dataframes`` (all record tables plus
    ``xml_hierarchy``), metadata, and the original source name.
    """
    source_name = _source_name(source)
    file_size = 0
    raw_file_path: Path | None = None
    try:
        if Path(source_name).suffix.lower() != ".xml":
            raise ValueError(
                f"Unsupported file format for '{source_name}'; expected .xml."
            )
        contents = _read_source(source)
        file_size = len(contents)
        raw_file_path = _save_original(contents, source_name, raw_directory)
        if not contents.strip():
            raise ValueError("XML file is empty.")

        namespaces = _find_namespaces(contents)
        xml_tree = parse_xml_document(contents)
        dataframes = convert_xml_to_dataframe(xml_tree, record_path=record_path)
        data = dataframes["records"]
        metadata = get_xml_metadata(
            xml_tree,
            dataframes,
            source_name,
            file_size,
            namespaces=namespaces,
            raw_file_path=raw_file_path,
        )
        return {
            "data": data,
            "dataframes": dataframes,
            "xml_tree": xml_tree,
            "metadata": metadata,
        }
    except (OSError, ValueError, TypeError) as error:
        return _failed_result(source_name, file_size, error, raw_file_path)


def extract_multiple_xml_files(
    sources: Sequence[str | Path | BinaryIO],
    record_path: str | None = None,
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> list[dict[str, Any]]:
    """Extract XML files independently, returning one result per input source."""
    return [
        extract_xml_file(
            source, record_path=record_path, raw_directory=raw_directory
        )
        for source in sources
    ]
