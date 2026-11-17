"""Read country data from MySQL and generate statistical analysis outputs."""

import argparse
import json
import logging
import math
import numbers
import os
import re
import sys
from pathlib import Path

import mysql.connector
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "analysis"
DATABASE_NAME = "web_data_automation"
TABLE_NAME = "countries"
DATA_FIELDS = ("country_name", "capital", "population", "area_km2")
NUMERIC_FIELDS = ("population", "area_km2")
logger = logging.getLogger(__name__)


def get_mysql_connection_settings(overrides=None):
    """Load and validate MySQL access settings from a mapping or environment."""
    settings = overrides or {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", DATABASE_NAME),
    }
    username = settings.get("user")
    password = settings.get("password")
    if not isinstance(username, str) or not username.strip() or not password:
        raise RuntimeError(
            "Enter a MySQL username and password before running the analysis."
        )
    host = settings.get("host", "localhost")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("MYSQL_HOST is required.")

    try:
        port = int(settings.get("port", 3306))
    except (TypeError, ValueError) as error:
        raise ValueError("MYSQL_PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise ValueError("MYSQL_PORT must be between 1 and 65535.")

    database = settings.get("database", DATABASE_NAME)
    if not isinstance(database, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,63}", database
    ):
        raise ValueError("MYSQL_DATABASE must be a valid MySQL database name.")
    return {
        "host": host.strip(),
        "port": port,
        "user": username.strip(),
        "password": password,
        "database": database,
    }


def fetch_country_records(connection_settings=None):
    """Fetch country columns from MySQL without modifying stored records."""
    settings = get_mysql_connection_settings(connection_settings)
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            **settings,
            connection_timeout=10,
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            f"""
            SELECT country_name, capital, population, area_km2
            FROM `{TABLE_NAME}`
            ORDER BY country_name
            """
        )
        records = cursor.fetchall()
        logger.info("Retrieved %d country records from MySQL", len(records))
        return records
    except mysql.connector.Error as error:
        logger.error("Could not retrieve country records from MySQL (%s)", type(error).__name__)
        raise
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except mysql.connector.Error as error:
                logger.warning("Could not close MySQL cursor (%s)", type(error).__name__)
        if connection is not None:
            try:
                connection.close()
            except mysql.connector.Error as error:
                logger.warning("Could not close MySQL connection (%s)", type(error).__name__)


def prepare_analysis_frame(records):
    """Create an analysis DataFrame and safely derive population density."""
    frame = pd.DataFrame.from_records(records)
    for field in DATA_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    frame = frame.loc[:, list(DATA_FIELDS)].copy()

    for field in ("country_name", "capital"):
        frame[field] = frame[field].map(
            lambda value: value.strip() if isinstance(value, str) else pd.NA
        )
        frame.loc[frame[field] == "", field] = pd.NA

    for field in NUMERIC_FIELDS:
        original_values = frame[field]
        frame[field] = pd.to_numeric(original_values, errors="coerce")
        unparseable = original_values.notna() & frame[field].isna()
        if unparseable.any():
            logger.warning(
                "Ignoring %d invalid %s value(s) for analysis",
                int(unparseable.sum()),
                field,
            )
        finite = frame[field].map(
            lambda value: pd.isna(value) or math.isfinite(float(value))
        )
        non_positive = frame[field].notna() & (frame[field] <= 0)
        invalid = ~finite | non_positive
        if invalid.any():
            logger.warning(
                "Ignoring %d non-finite or non-positive %s value(s) for analysis",
                int(invalid.sum()),
                field,
            )
            frame.loc[invalid, field] = float("nan")

    missing_areas = frame["area_km2"].isna()
    if missing_areas.any():
        logger.warning(
            "Population density will be unavailable for %d record(s) with "
            "missing, invalid, or zero area",
            int(missing_areas.sum()),
        )
    frame["population_density"] = frame["population"] / frame["area_km2"]
    frame.loc[
        ~frame["population_density"].map(
            lambda value: pd.isna(value) or math.isfinite(float(value))
        ),
        "population_density",
    ] = float("nan")
    return frame


def descriptive_statistics(values):
    """Return count, mean, median, minimum, maximum, and sample standard deviation."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
        }

    stats = numeric.agg(["count", "mean", "median", "min", "max", "std"])
    return {
        "count": int(stats["count"]),
        "mean": _json_number(stats["mean"]),
        "median": _json_number(stats["median"]),
        "minimum": _json_number(stats["min"]),
        "maximum": _json_number(stats["max"]),
        "standard_deviation": _json_number(stats["std"]),
    }


def _json_number(value):
    """Convert Pandas/NumPy numeric values to JSON-safe Python numbers."""
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    number = float(value)
    return int(number) if number.is_integer() else number


def _json_safe_records(dataframe):
    """Convert ranked DataFrame rows into JSON-safe dictionaries."""
    records = []
    for row in dataframe.to_dict(orient="records"):
        records.append(
            {
                key: (
                    _json_number(value)
                    if isinstance(value, numbers.Real) or pd.isna(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    return records


def build_population_analysis(frame):
    """Return country population rankings, largest first."""
    columns = ["population_rank", *DATA_FIELDS]
    ranked = frame.dropna(subset=["country_name", "population"]).sort_values(
        ["population", "country_name"],
        ascending=[False, True],
        kind="stable",
    )
    ranked = ranked.loc[:, list(DATA_FIELDS)].reset_index(drop=True)
    ranked.insert(0, "population_rank", range(1, len(ranked) + 1))
    return ranked.reindex(columns=columns)


def build_area_analysis(frame):
    """Return country area rankings, largest first."""
    columns = ["area_rank", *DATA_FIELDS]
    ranked = frame.dropna(subset=["country_name", "area_km2"]).sort_values(
        ["area_km2", "country_name"],
        ascending=[False, True],
        kind="stable",
    )
    ranked = ranked.loc[:, list(DATA_FIELDS)].reset_index(drop=True)
    ranked.insert(0, "area_rank", range(1, len(ranked) + 1))
    return ranked.reindex(columns=columns)


def build_population_density_analysis(frame):
    """Return all countries with calculable density, ranked highest first."""
    ranked = frame.dropna(
        subset=["country_name", "population_density"]
    ).sort_values(
        ["population_density", "country_name"],
        ascending=[False, True],
        kind="stable",
    )
    ranked = ranked[
        ["country_name", "capital", "population", "area_km2", "population_density"]
    ].reset_index(drop=True)
    ranked.insert(0, "density_rank", range(1, len(ranked) + 1))
    return ranked


def compare_countries(frame, country_names):
    """Return selected country metrics and their calculated population density."""
    columns = [
        "country_name",
        "capital",
        "population",
        "area_km2",
        "population_density",
    ]
    if not country_names:
        return pd.DataFrame(columns=columns), []

    requested_names = [name.strip() for name in country_names if name.strip()]
    requested_keys = {name.casefold() for name in requested_names}
    selected = frame.loc[
        frame["country_name"].astype("string").str.casefold().isin(requested_keys),
        columns,
    ].copy()
    found_keys = set(selected["country_name"].astype("string").str.casefold())
    not_found = [
        name for name in requested_names if name.casefold() not in found_keys
    ]
    if not_found:
        logger.warning("Requested comparison country/countries not found: %s", not_found)
    return selected.reset_index(drop=True), not_found


def build_analysis_summary(frame, population_ranking, area_ranking, density_ranking,
                           comparison_not_found=None):
    """Build overall metrics, descriptive statistics, and top/bottom lists."""
    total_population = frame["population"].sum(min_count=1)
    total_area = frame["area_km2"].sum(min_count=1)
    population_values = frame["population"].dropna()
    area_values = frame["area_km2"].dropna()
    density_values = frame["population_density"].dropna()

    def country_value(ranking, value_field):
        if ranking.empty:
            return None
        row = ranking.iloc[0]
        return {"country_name": row["country_name"], value_field: _json_number(row[value_field])}

    bottom_population = population_ranking.sort_values(
        ["population", "country_name"], ascending=[True, True], kind="stable"
    ).head(10)
    least_density = density_ranking.sort_values(
        ["population_density", "country_name"],
        ascending=[True, True],
        kind="stable",
    ).head(10)

    return {
        "total_countries_analysed": int(frame["country_name"].notna().sum()),
        "total_population": (
            int(total_population) if not pd.isna(total_population) else None
        ),
        "average_population": _json_number(population_values.mean()),
        "largest_country_by_area": country_value(area_ranking, "area_km2"),
        "most_populated_country": country_value(population_ranking, "population"),
        "most_densely_populated_country": country_value(
            density_ranking, "population_density"
        ),
        "least_populated_country": country_value(bottom_population, "population"),
        "smallest_country_by_area": country_value(
            area_ranking.sort_values(
                ["area_km2", "country_name"], ascending=[True, True], kind="stable"
            ),
            "area_km2",
        ),
        "least_densely_populated_country": country_value(least_density, "population_density"),
        "total_geographical_area_km2": _json_number(total_area),
        "average_country_area_km2": _json_number(area_values.mean()),
        "total_countries_with_calculable_density": int(len(density_values)),
        "population_statistics": descriptive_statistics(frame["population"]),
        "area_km2_statistics": descriptive_statistics(frame["area_km2"]),
        "population_density_statistics": descriptive_statistics(density_values),
        "top_10_most_populated_countries": _json_safe_records(
            population_ranking.head(10)
        ),
        "bottom_10_least_populated_countries": _json_safe_records(bottom_population),
        "top_10_largest_countries_by_area": _json_safe_records(area_ranking.head(10)),
        "top_10_highest_population_density": _json_safe_records(
            density_ranking.head(10)
        ),
        "bottom_10_lowest_population_density": _json_safe_records(least_density),
        "comparison_countries_not_found": comparison_not_found or [],
    }


def _save_charts(frame, population_ranking, area_ranking, density_ranking, charts_dir):
    """Save the four requested charts as PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_data = [
        (
            population_ranking.head(10),
            "population",
            "Top 10 Countries by Population",
            "Population",
            "top_10_population.png",
        ),
        (
            area_ranking.head(10),
            "area_km2",
            "Top 10 Countries by Geographical Area",
            "Area (km²)",
            "top_10_area.png",
        ),
        (
            density_ranking.head(10),
            "population_density",
            "Top 10 Countries by Population Density",
            "People per km²",
            "top_10_population_density.png",
        ),
    ]
    for ranked, value_field, title, x_label, filename in chart_data:
        figure, axis = plt.subplots(figsize=(11, 6))
        if ranked.empty:
            axis.text(0.5, 0.5, "No valid data available", ha="center", va="center")
            axis.set_axis_off()
        else:
            plotted = ranked.sort_values(value_field, ascending=True)
            axis.barh(plotted["country_name"], plotted[value_field], color="#3977a8")
            axis.set_xlabel(x_label)
            axis.set_ylabel("Country")
            axis.grid(axis="x", linestyle=":", alpha=0.4)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(charts_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)

    scatter_data = frame.dropna(subset=["area_km2", "population"])
    figure, axis = plt.subplots(figsize=(9, 6))
    if scatter_data.empty:
        axis.text(0.5, 0.5, "No valid data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.scatter(
            scatter_data["area_km2"],
            scatter_data["population"],
            alpha=0.7,
            color="#3977a8",
        )
        axis.set_xlabel("Area (km²)")
        axis.set_ylabel("Population")
        axis.grid(linestyle=":", alpha=0.4)
    axis.set_title("Population versus Geographical Area")
    figure.tight_layout()
    figure.savefig(
        charts_dir / "population_vs_area.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_analysis_outputs(frame, comparison_names, output_directory):
    """Write all analytical CSV files, JSON summary, and chart images."""
    output_dir = Path(output_directory)
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    population_ranking = build_population_analysis(frame)
    area_ranking = build_area_analysis(frame)
    density_ranking = build_population_density_analysis(frame)
    comparisons, missing_comparisons = compare_countries(frame, comparison_names)
    summary = build_analysis_summary(
        frame,
        population_ranking,
        area_ranking,
        density_ranking,
        missing_comparisons,
    )

    outputs = (
        (population_ranking, output_dir / "population_analysis.csv"),
        (area_ranking, output_dir / "area_analysis.csv"),
        (density_ranking, output_dir / "population_density.csv"),
        (comparisons, output_dir / "country_comparison.csv"),
    )
    for dataframe, path in outputs:
        try:
            dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        except OSError:
            logger.exception("Could not write analysis file %s", path)
            raise

    try:
        with (output_dir / "analysis_summary.json").open(
            "w", encoding="utf-8"
        ) as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
            summary_file.write("\n")
    except OSError:
        logger.exception("Could not write analysis summary")
        raise

    _save_charts(frame, population_ranking, area_ranking, density_ranking, charts_dir)
    return summary


def run_analysis(
    comparison_names=None,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    connection_settings=None,
):
    """Fetch database records and execute the complete analysis workflow."""
    records = fetch_country_records(connection_settings)
    frame = prepare_analysis_frame(records)
    summary = write_analysis_outputs(
        frame,
        comparison_names or [],
        output_directory,
    )
    logger.info(
        "Analysis complete: %d countries, %d requested comparison(s)",
        summary["total_countries_analysed"],
        len(comparison_names or []),
    )
    return summary


def run_streamlit_app():
    """Render the interactive MySQL analysis page."""
    st.set_page_config(page_title="Country Data Analysis", page_icon="📊")
    st.title("Country Data Analysis")
    st.caption("Enter your MySQL connection details to generate the analysis.")

    with st.form("mysql_analysis_form"):
        host = st.text_input("MySQL Host", value="localhost")
        port = st.number_input(
            "MySQL Port",
            min_value=1,
            max_value=65535,
            value=3306,
            step=1,
        )
        username = st.text_input("MySQL Username")
        password = st.text_input("MySQL Password", type="password")
        database = st.text_input("Database Name", value=DATABASE_NAME)
        comparison_input = st.text_input(
            "Compare countries (optional; separate names with commas)"
        )
        submitted = st.form_submit_button("Run Analysis", type="primary")

    if not submitted:
        return

    try:
        settings = get_mysql_connection_settings(
            {
                "host": host,
                "port": port,
                "user": username,
                "password": password,
                "database": database,
            }
        )
        comparison_names = [
            name.strip() for name in comparison_input.split(",") if name.strip()
        ]
        with st.spinner("Connecting to MySQL and generating analysis..."):
            summary = run_analysis(
                comparison_names=comparison_names,
                connection_settings=settings,
            )
    except (mysql.connector.Error, OSError, ValueError, RuntimeError) as error:
        logger.error("Streamlit data analysis failed (%s)", type(error).__name__)
        safe_message = str(error)
        if password:
            safe_message = safe_message.replace(password, "[redacted]")
        st.error(f"Data analysis failed: {safe_message}")
        return

    st.success(
        f"Analysis complete: {summary['total_countries_analysed']} countries."
    )
    st.caption(f"Analysis files saved to `{DEFAULT_OUTPUT_DIRECTORY}`.")

    def format_number(value, decimals=0):
        if value is None:
            return "N/A"
        return f"{value:,.{decimals}f}"

    def insight(country_key, value_key, decimals=0, suffix=""):
        country = summary.get(country_key)
        if not country:
            return "N/A"
        return (
            f"{country['country_name']} "
            f"({format_number(country.get(value_key), decimals)}{suffix})"
        )

    first_row = st.columns(4)
    first_row[0].metric(
        "Countries analysed",
        format_number(summary["total_countries_analysed"]),
    )
    first_row[1].metric(
        "Total population",
        format_number(summary["total_population"]),
    )
    first_row[2].metric(
        "Average population",
        format_number(summary["average_population"]),
    )
    first_row[3].metric(
        "Total area",
        f"{format_number(summary['total_geographical_area_km2'])} km²",
    )

    st.subheader("Key findings")
    findings = st.columns(3)
    findings[0].metric(
        "Most populated",
        insight("most_populated_country", "population"),
    )
    findings[1].metric(
        "Largest by area",
        insight("largest_country_by_area", "area_km2"),
    )
    findings[2].metric(
        "Highest population density",
        insight("most_densely_populated_country", "population_density", 2, " people/km²"),
    )

    st.subheader("Analysis charts")
    chart_directory = DEFAULT_OUTPUT_DIRECTORY / "charts"
    chart_columns = st.columns(2)
    charts = (
        ("Top 10 countries by population", "top_10_population.png"),
        ("Top 10 countries by area", "top_10_area.png"),
        ("Top 10 countries by population density", "top_10_population_density.png"),
        ("Population versus area", "population_vs_area.png"),
    )
    for index, (title, filename) in enumerate(charts):
        with chart_columns[index % 2]:
            st.image(str(chart_directory / filename), caption=title, use_container_width=True)

    st.subheader("Country rankings")
    table_tabs = st.tabs(
        ["Population", "Area", "Population density", "Country comparison"]
    )
    table_files = (
        "population_analysis.csv",
        "area_analysis.csv",
        "population_density.csv",
        "country_comparison.csv",
    )
    for tab, filename in zip(table_tabs, table_files):
        with tab:
            ranking = pd.read_csv(DEFAULT_OUTPUT_DIRECTORY / filename)
            if ranking.empty:
                st.info("No country data to display.")
            else:
                st.dataframe(ranking, use_container_width=True, hide_index=True)

    with st.expander("Detailed statistics"):
        stat_labels = {
            "population_statistics": "Population",
            "area_km2_statistics": "Area (km²)",
            "population_density_statistics": "Population density (people/km²)",
        }
        statistics = []
        for key, label in stat_labels.items():
            values = summary[key]
            statistics.append(
                {
                    "Measure": label,
                    "Count": values["count"],
                    "Mean": values["mean"],
                    "Median": values["median"],
                    "Minimum": values["minimum"],
                    "Maximum": values["maximum"],
                    "Standard deviation": values["standard_deviation"],
                }
            )
        st.dataframe(
            pd.DataFrame(statistics),
            use_container_width=True,
            hide_index=True,
        )
        st.metric(
            "Least populated",
            insight("least_populated_country", "population"),
        )
        st.metric(
            "Smallest by area",
            insight("smallest_country_by_area", "area_km2"),
        )
        st.metric(
            "Lowest population density",
            insight(
                "least_densely_populated_country",
                "population_density",
                2,
                " people/km²",
            ),
        )

    missing = summary["comparison_countries_not_found"]
    if missing:
        st.warning("Countries not found for comparison: " + ", ".join(missing))


def main():
    """Run analysis, optionally comparing countries named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        nargs="+",
        default=[],
        metavar="COUNTRY",
        help="country name(s) to include in the comparison CSV",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for analysis CSV, JSON, and charts",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = run_analysis(args.compare, args.output_directory)
    except (mysql.connector.Error, OSError, ValueError, RuntimeError) as error:
        logger.error("Data analysis failed (%s): %s", type(error).__name__, error)
        raise SystemExit(1) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    if get_script_run_ctx(suppress_warning=True) is not None:
        run_streamlit_app()
    else:
        main()
