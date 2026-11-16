"""Extract country information from the Scrapethissite simple page."""

import json
import logging

import requests
from bs4 import BeautifulSoup

URL = "https://www.scrapethissite.com/pages/simple/"
DEFAULT_TIMEOUT = 15

logger = logging.getLogger(__name__)


def _get_text(element, selector):
    """Return a selected element's text, or None when it is absent or empty."""
    field = element.select_one(selector)
    if field is None:
        return None

    text = field.get_text(" ", strip=True)
    return text or None


def parse_country_records(html):
    """Parse country cards from HTML into dictionaries of text values."""
    soup = BeautifulSoup(html, "html.parser")
    records = []
    fields = {
        "country": ".country-name",
        "capital": ".country-capital",
        "population": ".country-population",
        "area_sq_km": ".country-area",
    }

    for country in soup.select(".country"):
        record = {
            key: _get_text(country, selector)
            for key, selector in fields.items()
        }
        missing_fields = [key for key, value in record.items() if value is None]
        if missing_fields:
            logger.warning(
                "Country record is missing fields %s: %s",
                ", ".join(missing_fields),
                record.get("country"),
            )
        records.append(record)

    logger.info("Parsed %d country records", len(records))
    return records


def fetch_country_data(url=URL, timeout=DEFAULT_TIMEOUT):
    """Retrieve the page and return all parsed country records.

    Request failures are logged and return an empty list. Missing fields on
    individual records are represented as None by ``parse_country_records``.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Could not retrieve country data from %s", url)
        return []

    return parse_country_records(response.text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(json.dumps(fetch_country_data(), indent=2))
