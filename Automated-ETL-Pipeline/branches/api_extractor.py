"""World Bank Data360 API ingestion helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

import pandas as pd
import requests


API_SPECIFICATION_URL = (
    "https://raw.githubusercontent.com/worldbank/open-api-specs/refs/heads/main/"
    "Data360%20Open_API.json"
)
API_PROVIDER = "World Bank"
API_NAME = "Data360 API"
API_BASE_URL = "https://data360api.worldbank.org"
API_ENDPOINT = "/data360/data"
DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_RECORDS = 10_000
DEFAULT_RAW_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "raw" / "api"


@lru_cache(maxsize=4)
def _cached_api_specification(
    specification_url: str, timeout: float
) -> dict[str, Any]:
    try:
        response = requests.get(specification_url, timeout=timeout)
        response.raise_for_status()
        specification = response.json()
    except requests.RequestException as exc:
        raise ValueError(f"Could not download API specification: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"API specification is not valid JSON: {exc}") from exc

    if not isinstance(specification, dict):
        raise ValueError("API specification must be a JSON object.")
    return specification


def load_api_specification(
    specification_url: str = API_SPECIFICATION_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Download and cache the OpenAPI JSON specification.

    A deep copy is returned so callers cannot mutate the cached document.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    return copy.deepcopy(_cached_api_specification(specification_url, timeout))


def get_api_parameters(
    specification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return required and optional query parameters declared for the data GET."""
    spec = (
        load_api_specification()
        if specification is None
        else specification
    )
    paths = spec.get("paths", {})
    operation = paths.get(API_ENDPOINT, {}).get("get")
    if not isinstance(operation, Mapping):
        raise ValueError(f"GET {API_ENDPOINT} was not found in the specification.")

    components = spec.get("components", {}).get("parameters", {})
    required: list[str] = []
    optional: list[str] = []
    details: list[dict[str, Any]] = []
    for parameter in operation.get("parameters", []):
        if not isinstance(parameter, Mapping):
            continue
        reference = parameter.get("$ref")
        if reference:
            component_name = str(reference).rsplit("/", 1)[-1]
            parameter = components.get(component_name, {})
        if parameter.get("in") != "query":
            continue
        name = parameter.get("name")
        if not name:
            continue
        is_required = bool(parameter.get("required", False))
        (required if is_required else optional).append(str(name))
        details.append(
            {
                "name": str(name),
                "required": is_required,
                "description": parameter.get("description", ""),
                "schema": copy.deepcopy(parameter.get("schema", {})),
            }
        )

    return {
        "required": required,
        "optional": optional,
        "parameters": details,
    }


def _clean_request_parameters(
    parameters: Mapping[str, Any],
    specification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request_parameters = {
        str(key): value
        for key, value in parameters.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }
    database_id = request_parameters.get("DATABASE_ID")
    if not database_id:
        raise ValueError("DATABASE_ID is required.")

    supported = get_api_parameters(specification)
    supported_names = set(supported["required"] + supported["optional"])
    supported_names.update({"format", "skip"})
    unknown = set(request_parameters).difference(supported_names)
    if unknown:
        raise ValueError(
            "Unsupported Data360 query parameter(s): "
            + ", ".join(sorted(unknown))
        )
    request_parameters.setdefault("format", "json")
    if str(request_parameters["format"]).lower() != "json":
        raise ValueError("This extractor supports only JSON response format.")
    try:
        start_skip = int(request_parameters.get("skip", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("skip must be an integer.") from exc
    request_parameters["skip"] = start_skip
    if start_skip < 0:
        raise ValueError("skip must be zero or greater.")
    return request_parameters


def fetch_api_data(
    parameters: Mapping[str, Any],
    *,
    base_url: str = API_BASE_URL,
    endpoint: str = API_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT,
    max_records: int = DEFAULT_MAX_RECORDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    specification: Mapping[str, Any] | None = None,
    session: Any = None,
) -> dict[str, Any]:
    """Fetch observation pages from Data360 and return raw responses together.

    The API maximum is 1,000 observations per page. ``max_records`` limits the
    total number retrieved starting at the requested ``skip`` offset.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    if max_records <= 0:
        raise ValueError("max_records must be greater than zero.")
    if page_size <= 0 or page_size > DEFAULT_PAGE_SIZE:
        raise ValueError("page_size must be between 1 and 1000.")

    request_parameters = _clean_request_parameters(parameters, specification)
    start_skip = request_parameters["skip"]
    get_request: Any = session.get if session is not None else requests.get
    pages: list[dict[str, Any]] = []
    statuses: list[int] = []
    next_skip = start_skip
    expected_count: int | None = None
    target_records: int | None = None
    seen_skips: set[int] = set()

    while target_records is None or sum(
        len(page["response"]["value"]) for page in pages
    ) < target_records:
        if next_skip in seen_skips:
            raise RuntimeError(f"Pagination repeated skip={next_skip}; stopping.")
        seen_skips.add(next_skip)

        page_parameters = dict(request_parameters)
        page_parameters["skip"] = next_skip
        try:
            response = get_request(
                urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/")),
                params=page_parameters,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Data360 returned invalid JSON at skip={next_skip}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(
            payload.get("value"), list
        ):
            raise ValueError(
                "Data360 response must contain a 'value' array of observations."
            )
        try:
            page_count = int(payload.get("count", len(payload["value"])))
        except (TypeError, ValueError) as exc:
            raise ValueError("Data360 response has an invalid 'count' value.") from exc
        if page_count < 0:
            raise ValueError("Data360 response count cannot be negative.")
        if expected_count is None:
            expected_count = page_count
            target_records = min(max(page_count - start_skip, 0), max_records)

        page_records = payload["value"]
        statuses.append(int(response.status_code))
        pages.append(
            {
                "skip": next_skip,
                "status_code": int(response.status_code),
                "response": payload,
            }
        )
        if not page_records:
            break
        next_skip += len(page_records)
        if len(page_records) < page_size:
            # A short page may still precede more data; use the API count to
            # decide whether a further page is needed.
            if expected_count is not None and next_skip >= expected_count:
                break

    observation_records = [
        record
        for page in pages
        for record in page["response"]["value"]
    ][: target_records if target_records is not None else max_records]
    combined_response = {
        "count": expected_count if expected_count is not None else 0,
        "value": observation_records,
    }
    retrieved_count = len(observation_records)
    return {
        "response": combined_response,
        "page_responses": [page["response"] for page in pages],
        "http_status_code": statuses[-1] if statuses else None,
        "request_parameters": request_parameters,
        "pagination": {
            "page_size": page_size,
            "pages_fetched": len(pages),
            "starting_skip": start_skip,
            "next_skip": start_skip + retrieved_count,
            "max_records": max_records,
            "api_record_count": expected_count or 0,
            "records_retrieved": retrieved_count,
            "complete": (
                expected_count is not None
                and start_skip + retrieved_count >= expected_count
            ),
            "page_skips": [page["skip"] for page in pages],
        },
    }


def convert_api_response_to_dataframe(
    api_response: Mapping[str, Any],
) -> pd.DataFrame:
    """Convert the response's observation ``value`` array without transforming it."""
    records = api_response.get("value", [])
    if not isinstance(records, list):
        raise ValueError("API response 'value' must be a list.")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("Every API observation in 'value' must be a JSON object.")
    return pd.DataFrame(records, dtype=object)


def get_api_metadata(
    *,
    request_parameters: Mapping[str, Any] | None = None,
    data: pd.DataFrame | None = None,
    api_response: Mapping[str, Any] | None = None,
    pagination: Mapping[str, Any] | None = None,
    http_status_code: int | None = None,
    status: str = "success",
    error: str | None = None,
    specification: Mapping[str, Any] | None = None,
    specification_url: str = API_SPECIFICATION_URL,
) -> dict[str, Any]:
    """Build dynamic metadata for the request and its extracted observations."""
    parameters = dict(request_parameters or {})
    info = (specification or {}).get("info", {})
    api_count = (api_response or {}).get("count", 0)
    metadata: dict[str, Any] = {
        "provider": API_PROVIDER,
        "api_name": info.get("title", API_NAME),
        "api_base_url": API_BASE_URL,
        "api_endpoint": API_ENDPOINT,
        "http_method": "GET",
        "api_specification_url": specification_url,
        "specification_version": info.get("version"),
        "database_id": parameters.get("DATABASE_ID"),
        "indicator_id": parameters.get("INDICATOR"),
        "country_reference_area": parameters.get("REF_AREA"),
        "request_parameters": parameters,
        "request_timestamp": datetime.now(timezone.utc).isoformat(),
        "http_status_code": http_status_code,
        "api_response_record_count": api_count,
        "records_retrieved": len(data.index) if data is not None else 0,
        "dataframe_columns": len(data.columns) if data is not None else 0,
        "column_names": list(data.columns) if data is not None else [],
        "pagination": dict(pagination or {}),
        "status": status,
    }
    if error is not None:
        metadata["error"] = error
    return metadata


def _save_raw_responses(
    api_response: Mapping[str, Any],
    page_responses: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    raw_directory: str | Path,
) -> Path:
    raw_root = Path(raw_directory)
    raw_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    while True:
        run_directory = raw_root / f"data360_{timestamp}_{uuid4().hex[:8]}"
        try:
            run_directory.mkdir()
            break
        except FileExistsError:
            continue

    (run_directory / "response.json").write_text(
        json.dumps(api_response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pages_directory = run_directory / "pages"
    pages_directory.mkdir()
    for index, page_response in enumerate(page_responses, start=1):
        (pages_directory / f"page_{index:04d}.json").write_text(
            json.dumps(page_response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return run_directory


def extract_api_data(
    parameters: Mapping[str, Any],
    *,
    specification_url: str = API_SPECIFICATION_URL,
    timeout: float = DEFAULT_TIMEOUT,
    max_records: int = DEFAULT_MAX_RECORDS,
    page_size: int = DEFAULT_PAGE_SIZE,
    raw_directory: str | Path = DEFAULT_RAW_DIRECTORY,
    session: Any = None,
) -> dict[str, Any]:
    """Fetch, convert, archive, and describe one Data360 observation request."""
    specification: dict[str, Any] = {}
    request_parameters = dict(parameters)
    response: dict[str, Any] | None = None
    page_responses: list[dict[str, Any]] = []
    pagination: dict[str, Any] = {}
    http_status_code: int | None = None
    data: pd.DataFrame | None = None
    try:
        specification = load_api_specification(specification_url, timeout)
        servers = specification.get("servers", [])
        if not servers or not isinstance(servers[0].get("url"), str):
            raise ValueError("API specification does not declare a server URL.")
        base_url = servers[0]["url"]
        request_parameters = _clean_request_parameters(parameters, specification)
        operation = specification.get("paths", {}).get(API_ENDPOINT, {}).get("get")
        if not isinstance(operation, Mapping):
            raise ValueError(f"GET {API_ENDPOINT} was not found in the specification.")
        fetched = fetch_api_data(
            request_parameters,
            base_url=base_url,
            endpoint=API_ENDPOINT,
            timeout=timeout,
            max_records=max_records,
            page_size=page_size,
            specification=specification,
            session=session,
        )
        response = fetched["response"]
        page_responses = fetched["page_responses"]
        pagination = fetched["pagination"]
        http_status_code = fetched["http_status_code"]
        if response is None:
            raise RuntimeError("Data360 returned no response data.")
        data = convert_api_response_to_dataframe(response)
        metadata = get_api_metadata(
            request_parameters=fetched["request_parameters"],
            data=data,
            api_response=response,
            pagination=pagination,
            http_status_code=http_status_code,
            specification=specification,
            specification_url=specification_url,
        )
        raw_path = _save_raw_responses(
            response, page_responses, metadata, raw_directory
        )
        metadata["raw_file_path"] = str(raw_path)
        (raw_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return {
            "data": data,
            "api_response": response,
            "page_responses": page_responses,
            "metadata": metadata,
        }
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        response_obj = getattr(exc, "response", None)
        if response_obj is not None:
            http_status_code = getattr(response_obj, "status_code", None)
        metadata = get_api_metadata(
            request_parameters=request_parameters,
            data=data,
            api_response=response,
            pagination=pagination,
            http_status_code=http_status_code,
            status="error",
            error=str(exc),
            specification=specification,
            specification_url=specification_url,
        )
        return {
            "data": data,
            "api_response": response,
            "page_responses": page_responses,
            "metadata": metadata,
        }
