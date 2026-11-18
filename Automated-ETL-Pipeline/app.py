"""Streamlit interface for the ETL pipeline ingestion and processing stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import sys
from typing import Any, BinaryIO, cast
from xml.etree import ElementTree as element_tree

# Streamlit adds the app directory to sys.path when this file is run directly,
# but not always the project root needed for imports through the SRC package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ingestion.api_extractor import (
    API_SPECIFICATION_URL,
    extract_api_data,
)
from ingestion.csv_extractor import extract_multiple_csv_files
from ingestion.json_extractor import extract_multiple_json_files
from ingestion.xml_extractor import extract_multiple_xml_files
from integration.csv_integrator import (
    integrate_multiple_csv_datasets,
    match_common_columns,
)
from integration.api_integrator import (
    handle_inconsistent_schemas as handle_api_inconsistent_schemas,
    integrate_multiple_api_datasets,
    match_common_columns as match_api_common_columns,
)
from integration.json_integrator import (
    get_json_record_paths,
    handle_inconsistent_schemas as handle_json_inconsistent_schemas,
    integrate_multiple_json_datasets,
    match_common_fields,
)
from integration.xml_integrator import (
    get_xml_record_groups,
    handle_inconsistent_schemas as handle_xml_inconsistent_schemas,
    integrate_multiple_xml_datasets,
    match_common_elements,
)
from profiling.api_profiler import profile_api
from profiling.csv_profiler import profile_csv
from profiling.json_profiler import profile_json
from profiling.xml_profiler import profile_xml
from cleaning.csv_cleaner import (
    clean_csv_dataframe,
    identify_currency_columns,
    identify_date_columns,
    save_cleaned_csv,
)
from cleaning.api_cleaner import (
    clean_api_dataframe,
    identify_currency_columns as identify_api_currency_columns,
    identify_date_columns as identify_api_date_columns,
    save_cleaned_api_dataset,
)
from cleaning.json_cleaner import clean_json_data, save_cleaned_json
from cleaning.xml_cleaner import (
    clean_xml_data,
    get_record_groups,
    save_cleaned_xml,
)
from validation.csv_validator import (
    export_validation_report,
    validate_csv_dataframe,
)
from validation.json_validator import (
    discover_json_structure,
    export_json_validation_result,
    validate_json_data,
)
from validation.xml_validator import (
    export_xml_validation_result,
    get_xml_validation_paths,
    validate_xml_data,
)
from validation.api_validator import (
    export_api_validation_data,
    export_api_validation_report,
    validate_api_dataframe,
)
from database.json_loader import (
    get_connection_config_from_environment as get_json_db_environment_config,
    get_pipeline_execution_history as get_json_db_execution_history,
    load_validated_json_dataset,
    prepare_mysql_database as prepare_json_mysql_database,
    suggest_unique_key_fields,
)
from database.xml_loader import (
    get_xml_key_candidates,
    get_xml_structure_summary,
    load_validated_xml_dataset,
    prepare_mysql_database as prepare_xml_mysql_database,
)
from database.csv_loader import (
    get_connection_config_from_environment,
    get_pipeline_execution_history,
    load_validated_csv_dataset,
    prepare_mysql_database,
)
from SRC.automation.csv_automation import (
    get_csv_execution_history,
    rerun_failed_execution,
    run_csv_pipeline,
)
from SRC.automation.csv_monitor import get_csv_execution_log, get_csv_log_text
from SRC.automation.csv_scheduler import (
    DEFAULT_SCHEDULE_PATH,
    get_csv_schedule,
    get_next_run_time,
    save_csv_schedule,
    set_csv_schedule_enabled,
)
from SRC.automation.json_automation import (
    get_json_execution_history,
    rerun_failed_execution as rerun_failed_json_execution,
    run_json_pipeline,
)
from SRC.automation.json_monitor import (
    get_json_execution_log,
    get_json_log_text,
)
from SRC.automation.json_scheduler import (
    DEFAULT_SCHEDULE_PATH as DEFAULT_JSON_SCHEDULE_PATH,
    get_json_schedule,
    get_next_run_time as get_json_next_run_time,
    save_json_schedule,
    set_json_schedule_enabled,
)
from SRC.automation.xml_automation import (
    get_xml_execution_history,
    rerun_failed_execution as rerun_failed_xml_execution,
    run_xml_pipeline,
)
from SRC.automation.xml_monitor import get_xml_execution_log, get_xml_log_text
from SRC.automation.xml_scheduler import (
    DEFAULT_SCHEDULE_PATH as DEFAULT_XML_SCHEDULE_PATH,
    get_xml_schedule,
    get_next_run_time as get_xml_next_run_time,
    save_xml_schedule,
    set_xml_schedule_enabled,
)
from database.api_loader import (
    get_connection_config_from_environment as get_api_connection_config_from_environment,
    get_api_metadata_table_name,
    load_validated_api_dataset,
    suggest_api_unique_key_columns,
)
from SRC.automation.api_automation import (
    rerun_failed_api_execution,
    run_api_pipeline,
)
from SRC.automation.api_monitor import (
    get_api_execution_history,
    get_api_execution_log,
    get_api_log_text,
)
from SRC.automation.api_scheduler import (
    get_api_schedule,
    get_next_run_time as get_api_next_run_time,
    save_api_schedule,
    set_api_schedule_enabled,
)

st.set_page_config(page_title="Automated ETL Pipeline", layout="wide")
st.title("Automated ETL Pipeline")
st.subheader("Stage 1: Data Ingestion")


def show_results(session_key: str, *, show_metadata: bool = True) -> None:
    for ingestion_result in st.session_state.get(session_key, []):
        ingestion_metadata = ingestion_result["metadata"]
        if ingestion_metadata["status"] == "success":
            st.success(f"Extracted {ingestion_metadata['source']}")
            if show_metadata:
                st.json(ingestion_metadata)
            st.dataframe(ingestion_result["data"], width="stretch")
            if ingestion_result.get("json_data") is not None:
                with st.expander("Original parsed JSON"):
                    st.json(ingestion_result["json_data"])
        else:
            st.error(
                f"{ingestion_metadata['source']}: {ingestion_metadata['error']}"
            )
            if show_metadata:
                st.json(ingestion_metadata)


st.header("CSV ingestion")
csv_files = st.file_uploader(
    "Select one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="csv_upload",
)
if st.button("Extract CSV files", disabled=not csv_files):
    st.session_state["csv_results"] = extract_multiple_csv_files(
        cast(list[BinaryIO], csv_files)
    )
show_results("csv_results", show_metadata=False)

st.header("Stage 2: CSV Data Profiling")
successful_csv_results = [
    (index, item)
    for index, item in enumerate(st.session_state.get("csv_results", []))
    if item["metadata"]["status"] == "success" and item["data"] is not None
]
if successful_csv_results:
    csv_selection_labels = {
        index: (
            f"{item['metadata']['source']} (dataset {index + 1})"
        )
        for index, item in successful_csv_results
    }
    selected_csv_index = st.selectbox(
        "Select an ingested CSV dataset",
        [index for index, _ in successful_csv_results],
        format_func=csv_selection_labels.__getitem__,
        key="csv_profile_selection",
    )
    selected_csv = st.session_state["csv_results"][selected_csv_index]
    profile_key = (
        f"{selected_csv['metadata']['source']}::{selected_csv_index}"
    )
    if st.button("Profile selected CSV", key="profile_csv_button"):
        profiles = st.session_state.setdefault("csv_profile_results", {})
        profiles[profile_key] = profile_csv(
            selected_csv["data"], selected_csv["metadata"]["source"]
        )

    profile = st.session_state.get("csv_profile_results", {}).get(profile_key)
    if profile is not None:
        dimensions = profile["dimensions"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Rows", dimensions["total_rows"])
        metric_columns[1].metric("Columns", dimensions["total_columns"])
        metric_columns[2].metric(
            "Missing values",
            profile["missing_values"]["total_missing_values"],
        )
        metric_columns[3].metric(
            "Duplicate rows",
            profile["duplicates"]["total_duplicate_rows"],
        )

        st.session_state["stage3_csv_datasets"] = {
            key: value
            for key, value in st.session_state.get(
                "csv_profile_results", {}
            ).items()
        }
else:
    st.info("Ingest a CSV file in Stage 1 before profiling it.")

st.header("Stage 3: CSV Data Cleaning")
cleanable_csv_results = [
    (index, item)
    for index, item in enumerate(st.session_state.get("csv_results", []))
    if item["metadata"]["status"] == "success" and item["data"] is not None
]
if cleanable_csv_results:
    cleaning_labels = {
        index: f"{item['metadata']['source']} (dataset {index + 1})"
        for index, item in cleanable_csv_results
    }
    cleaning_index = st.selectbox(
        "Select an ingested CSV dataset to clean",
        [index for index, _ in cleanable_csv_results],
        format_func=cleaning_labels.__getitem__,
        key="csv_cleaning_selection",
    )
    selected_cleaning_result = st.session_state["csv_results"][cleaning_index]
    original_csv = selected_cleaning_result["data"]
    original_csv_name = selected_cleaning_result["metadata"]["source"]
    cleaning_key = f"{original_csv_name}::{cleaning_index}"

    st.subheader("Original CSV data")
    st.dataframe(original_csv, width="stretch")
    csv_stage2_profile = st.session_state.get("csv_profile_results", {}).get(
        cleaning_key
    )
    st.subheader("Stage 2 CSV profiling results")
    if csv_stage2_profile is None:
        st.info("Run Stage 2 CSV profiling for this dataset to see its profile here.")
    else:
        profile_metrics = st.columns(4)
        profile_metrics[0].metric(
            "Profiled rows", csv_stage2_profile["dimensions"]["total_rows"]
        )
        profile_metrics[1].metric(
            "Profiled columns", csv_stage2_profile["dimensions"]["total_columns"]
        )
        profile_metrics[2].metric(
            "Missing values",
            csv_stage2_profile["missing_values"]["total_missing_values"],
        )
        profile_metrics[3].metric(
            "Duplicate rows",
            csv_stage2_profile["duplicates"]["total_duplicate_rows"],
        )
    st.subheader("Cleaning operations")
    cleaning_flags = st.columns(3)
    clean_duplicates = cleaning_flags[0].checkbox(
        "Remove duplicate rows", key="clean_csv_duplicates"
    )
    clean_missing = cleaning_flags[1].checkbox(
        "Handle missing values", key="clean_csv_missing"
    )
    clean_names = cleaning_flags[2].checkbox(
        "Standardise column names", key="clean_csv_names"
    )
    clean_dates = cleaning_flags[0].checkbox(
        "Standardise dates", key="clean_csv_dates"
    )
    clean_currencies = cleaning_flags[1].checkbox(
        "Parse currency columns", key="clean_csv_currencies"
    )
    clean_types = cleaning_flags[2].checkbox(
        "Convert selected data types", key="clean_csv_types"
    )

    duplicate_subset = None
    if clean_duplicates:
        duplicate_subset_selection = st.multiselect(
            "Optional duplicate comparison columns (empty means full row)",
            list(original_csv.columns),
            key="csv_duplicate_subset",
        )
        duplicate_subset = duplicate_subset_selection or None

    missing_strategy = "leave"
    missing_fill_value = None
    if clean_missing:
        missing_strategy = st.selectbox(
            "Missing-value strategy",
            ["leave", "drop_rows", "mean", "median", "mode", "fill_value"],
            format_func=lambda value: {
                "leave": "Leave unchanged",
                "drop_rows": "Remove rows containing missing values",
                "mean": "Fill numeric columns with mean",
                "median": "Fill numeric columns with median",
                "mode": "Fill categorical columns with mode",
                "fill_value": "Fill with a specified value",
            }[value],
            key="csv_missing_strategy",
        )
        if missing_strategy == "fill_value":
            missing_fill_value = st.text_input(
                "Value to use for missing cells", key="csv_missing_fill_value"
            )

    selected_date_columns: list[str] = []
    date_format = None
    if clean_dates:
        date_candidates = identify_date_columns(original_csv)
        selected_date_columns = st.multiselect(
            "Date columns",
            list(original_csv.columns),
            default=date_candidates,
            key="csv_date_columns",
        )
        date_format = st.text_input(
            "Optional date format (for example %Y-%m-%d)",
            key="csv_date_format",
        ) or None

    selected_currency_columns: list[str] = []
    currency_format = "auto"
    if clean_currencies:
        selected_currency_columns = st.multiselect(
            "Currency columns (choose explicitly; year/percentage columns are not inferred)",
            list(original_csv.columns),
            default=identify_currency_columns(original_csv),
            key="csv_currency_columns",
        )
        currency_format = st.selectbox(
            "Currency code for values without a symbol",
            ["auto", "USD", "EUR", "GBP", "INR"],
            key="csv_currency_format",
        )

    type_conversions: dict[str, str] = {}
    if clean_types:
        type_columns = st.multiselect(
            "Columns to convert",
            list(original_csv.columns),
            key="csv_type_columns",
        )
        if type_columns:
            target_type = st.selectbox(
                "Target type for selected columns",
                ["integer", "float", "string", "boolean", "datetime"],
                key="csv_target_type",
            )
            type_conversions = {str(column): target_type for column in type_columns}

    if st.button("Run CSV Cleaning", key="run_csv_cleaning"):
        cleaning_result = clean_csv_dataframe(
            original_csv,
            dataset_name=original_csv_name,
            remove_duplicates=clean_duplicates,
            duplicate_subset=duplicate_subset,
            missing_strategy=missing_strategy,
            missing_fill_value=missing_fill_value,
            standardise_names=clean_names,
            standardise_dates=clean_dates,
            date_columns=selected_date_columns if clean_dates else None,
            date_format=date_format,
            standardise_currencies=clean_currencies,
            currency_columns=(
                selected_currency_columns if clean_currencies else None
            ),
            currency_format=currency_format,
            type_conversions=type_conversions,
        )
        processed_path = save_cleaned_csv(
            cleaning_result["data"], original_csv_name
        )
        cleaning_result["report"]["processed_file_path"] = str(processed_path)
        st.session_state.setdefault("csv_cleaning_results", {})[
            cleaning_key
        ] = cleaning_result

    cleaning_result = st.session_state.get("csv_cleaning_results", {}).get(
        cleaning_key
    )
    if cleaning_result is not None:
        cleaned_csv = cleaning_result["data"]
        cleaning_report = cleaning_result["report"]
        st.subheader("Cleaned CSV data")
        st.dataframe(cleaned_csv, width="stretch")
        before_after = st.columns(4)
        before_after[0].metric(
            "Rows before / after",
            f"{cleaning_report['original_row_count']} / "
            f"{cleaning_report['final_row_count']}",
        )
        before_after[1].metric(
            "Columns before / after",
            f"{cleaning_report['original_column_count']} / "
            f"{cleaning_report['final_column_count']}",
        )
        before_after[2].metric(
            "Duplicate rows removed",
            cleaning_report["duplicate_rows_removed"],
        )
        before_after[3].metric(
            "Missing values after",
            sum(cleaning_report["missing_values_after"].values()),
        )
        st.download_button(
            "Download cleaned CSV",
            data=cleaned_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(original_csv_name).stem}_cleaned.csv",
            mime="text/csv",
            key="download_cleaned_csv",
        )
        st.caption(
            f"Processed output saved separately to "
            f"`{cleaning_report['processed_file_path']}`. Original Stage 1 data "
            "remains unchanged."
        )
else:
    st.info("Ingest a CSV file in Stage 1 before cleaning it.")

st.header("Stage 4: CSV Data Validation")
cleaned_csv_datasets = st.session_state.get("csv_cleaning_results", {})
if cleaned_csv_datasets:
    validation_labels = {
        key: str(result.get("report", {}).get("dataset_name") or key)
        for key, result in cleaned_csv_datasets.items()
        if isinstance(result.get("data"), pd.DataFrame)
    }
    if validation_labels:
        selected_validation_key = st.selectbox(
            "Select a cleaned CSV dataset from Stage 3",
            list(validation_labels),
            format_func=validation_labels.__getitem__,
            key="csv_validation_selection",
        )
        selected_cleaning = cleaned_csv_datasets[selected_validation_key]
        validation_df = selected_cleaning["data"]
        validation_dataset_name = str(
            selected_cleaning.get("report", {}).get("dataset_name")
            or validation_labels[selected_validation_key]
        )
        st.subheader("Stage 3 cleaned CSV data")
        st.dataframe(validation_df, width="stretch")

        validation_flags = st.columns(3)
        run_required_validation = validation_flags[0].checkbox(
            "Validate required fields",
            value=True,
            key="csv_validation_required_enabled",
        )
        run_numeric_validation = validation_flags[1].checkbox(
            "Validate numerical values",
            value=True,
            key="csv_validation_numeric_enabled",
        )
        run_date_validation = validation_flags[2].checkbox(
            "Validate date formats",
            value=True,
            key="csv_validation_dates_enabled",
        )

        validation_columns = [str(column) for column in validation_df.columns]
        required_columns = []
        required_fields = []
        warning_fields = []
        if run_required_validation:
            required_columns = st.multiselect(
                "Columns that must exist",
                validation_columns,
                key="csv_validation_required_columns",
            )
            required_fields = st.multiselect(
                "Fields that must not be missing",
                validation_columns,
                key="csv_validation_required_fields",
            )
            warning_fields = st.multiselect(
                "Fields where missing values should be warnings",
                [
                    column
                    for column in validation_columns
                    if column not in required_fields
                ],
                key="csv_validation_warning_fields",
            )

        numerical_rules = {}
        if run_numeric_validation:
            numerical_columns = st.multiselect(
                "Columns expected to contain numbers",
                validation_columns,
                key="csv_validation_numerical_columns",
            )
            for column in numerical_columns:
                with st.expander(f"Numerical rules: {column}"):
                    minimum_text = st.text_input(
                        f"Minimum for {column} (optional)",
                        key=f"csv_validation_min_{column}",
                    )
                    maximum_text = st.text_input(
                        f"Maximum for {column} (optional)",
                        key=f"csv_validation_max_{column}",
                    )
                    allow_negative = st.checkbox(
                        f"Allow negative values in {column}",
                        value=True,
                        key=f"csv_validation_negative_{column}",
                    )
                    integer_only = st.checkbox(
                        f"Require integer values in {column}",
                        key=f"csv_validation_integer_{column}",
                    )
                numeric_rule: dict[str, Any] = {
                    "allow_negative": allow_negative,
                    "integer_only": integer_only,
                }
                if minimum_text and minimum_text.strip():
                    numeric_rule["min"] = minimum_text.strip()
                if maximum_text and maximum_text.strip():
                    numeric_rule["max"] = maximum_text.strip()
                numerical_rules[column] = numeric_rule

        date_rules = {}
        if run_date_validation:
            date_columns = st.multiselect(
                "Columns containing dates or year values",
                validation_columns,
                key="csv_validation_date_columns",
            )
            for column in date_columns:
                with st.expander(f"Date format: {column}"):
                    year_only = st.checkbox(
                        f"Accept four-digit years only for {column}",
                        key=f"csv_validation_year_only_{column}",
                    )
                    expected_format = st.text_input(
                        f"Expected Python date format for {column}",
                        value="" if year_only else "%Y-%m-%d",
                        key=f"csv_validation_format_{column}",
                        disabled=year_only,
                    )
                date_rules[column] = (
                    {"year_only": True}
                    if year_only
                    else {"format": expected_format or "%Y-%m-%d"}
                )

        if st.button("Run CSV Validation", key="run_csv_validation"):
            try:
                parsed_numeric_rules = {}
                for column, rule in numerical_rules.items():
                    parsed_rule = dict(rule)
                    for bound in ("min", "max"):
                        if bound in parsed_rule:
                            try:
                                parsed_rule[bound] = float(parsed_rule[bound])
                            except ValueError as exc:
                                raise ValueError(
                                    f"{bound.title()}imum for {column!r} must be numeric."
                                ) from exc
                    if (
                            parsed_rule.get("min") is not None
                            and parsed_rule.get("max") is not None
                            and parsed_rule["min"] > parsed_rule["max"]
                    ):
                        raise ValueError(
                            f"Minimum for {column!r} cannot exceed its maximum."
                        )
                    parsed_numeric_rules[column] = parsed_rule
                validation_result = validate_csv_dataframe(
                    validation_df,
                    validation_dataset_name,
                    required_columns=required_columns,
                    required_fields=required_fields,
                    warning_fields=warning_fields,
                    numerical_rules=parsed_numeric_rules,
                    date_rules=date_rules,
                    validate_required=run_required_validation,
                    validate_numerical=run_numeric_validation,
                    validate_dates=run_date_validation,
                )
                report_paths = export_validation_report(validation_result)
                validation_result["report_paths"] = report_paths
                st.session_state.setdefault("csv_validation_results", {})[
                    selected_validation_key
                ] = validation_result
            except (OSError, ValueError, TypeError) as error:
                st.error(f"CSV validation could not run: {error}")

        validation_result = st.session_state.get(
            "csv_validation_results", {}
        ).get(selected_validation_key)
        if validation_result is not None:
            validation_report = validation_result["validation_report"]
            status = validation_result["validation_status"]
            if status == "passed":
                st.success(f"Validation {status}.")
            elif status == "passed_with_warnings":
                st.warning(f"Validation {status}.")
            else:
                st.error(f"Validation {status}.")
            validation_metrics = st.columns(4)
            validation_metrics[0].metric(
                "Valid records", validation_report["valid_records"]
            )
            validation_metrics[1].metric(
                "Invalid records", validation_report["invalid_records"]
            )
            validation_metrics[2].metric(
                "Errors", validation_report["total_validation_errors"]
            )
            validation_metrics[3].metric(
                "Warnings", validation_report["total_validation_warnings"]
            )
            st.subheader("Invalid records")
            st.dataframe(
                validation_result["invalid_dataframe"],
                width="stretch",
            )
            st.subheader("Validation errors")
            st.dataframe(
                validation_result["validation_errors"],
                width="stretch",
                hide_index=True,
            )
            if not validation_result["validation_warnings"].empty:
                st.subheader("Validation warnings")
                st.dataframe(
                    validation_result["validation_warnings"],
                    width="stretch",
                    hide_index=True,
                )
            st.download_button(
                "Download valid records",
                data=validation_result["valid_dataframe"].to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"{Path(validation_dataset_name).stem}_valid.csv",
                mime="text/csv",
                key="download_csv_valid_records",
            )
            st.download_button(
                "Download invalid records",
                data=validation_result["invalid_dataframe"].to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"{Path(validation_dataset_name).stem}_invalid.csv",
                mime="text/csv",
                key="download_csv_invalid_records",
            )
            st.download_button(
                "Download validation error report (CSV)",
                data=Path(validation_result["report_paths"]["csv"]).read_bytes(),
                file_name=f"{Path(validation_dataset_name).stem}_validation_errors.csv",
                mime="text/csv",
                key="download_csv_validation_errors",
            )
            st.download_button(
                "Download validation report (JSON)",
                data=Path(validation_result["report_paths"]["json"]).read_bytes(),
                file_name=f"{Path(validation_dataset_name).stem}_validation_report.json",
                mime="application/json",
                key="download_csv_validation_json",
            )
    else:
        st.info("Run Stage 3 CSV cleaning before validating a dataset.")
else:
    st.info("Run Stage 3 CSV cleaning before validating a dataset.")

st.header("Stage 5: CSV Data Integration")
csv_validation_results = st.session_state.get("csv_validation_results", {})
integrable_csv_datasets = {}
for dataset_key, validation_result in csv_validation_results.items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and isinstance(validation_result.get("valid_dataframe"), pd.DataFrame)
    ):
        report = validation_result.get("validation_report", {})
        dataset_name = str(
            report.get("dataset_name")
            or cleaned_csv_datasets.get(dataset_key, {}).get("report", {}).get(
                "dataset_name"
            )
            or dataset_key
        )
        integrable_csv_datasets[dataset_key] = {
            "validation_result": validation_result,
            "source": dataset_name,
        }

if integrable_csv_datasets:
    integration_labels = {
        key: str(value["source"])
        for key, value in integrable_csv_datasets.items()
    }
    selected_integration_keys = st.multiselect(
        "Select validated CSV datasets",
        list(integration_labels),
        format_func=integration_labels.__getitem__,
        key="csv_integration_selection",
    )
    if selected_integration_keys:
        selected_integration_datasets = {
            key: integrable_csv_datasets[key]
            for key in selected_integration_keys
        }
        selected_frames = {
            key: value["validation_result"]["valid_dataframe"]
            .drop(
                columns=[
                    column
                    for column in value["validation_result"]["valid_dataframe"].columns
                    if str(column).startswith("_validation_status")
                    or str(column).startswith("_validation_errors")
                ],
                errors="ignore",
            )
            for key, value in selected_integration_datasets.items()
        }
        st.subheader("Selected validated datasets")
        for dataset_key, frame in selected_frames.items():
            with st.expander(integration_labels[dataset_key]):
                st.dataframe(frame, width="stretch")
                st.json(
                    selected_integration_datasets[dataset_key][
                        "validation_result"
                    ].get("validation_report", {})
                )

        integration_mode = st.radio(
            "Integration type",
            ["Append rows", "Merge datasets"],
            horizontal=True,
            key="csv_integration_type",
        )
        standardize_schema = st.checkbox(
            "Standardize column names before appending",
            value=False,
            disabled=integration_mode != "Append rows",
            key="csv_integration_standardize_names",
        )
        merge_join_type = "inner"
        merge_keys = []
        right_merge_keys = []
        merge_mapping = {}
        prevent_many_to_many = True
        merge_keys_valid = True
        if len(selected_frames) >= 2:
            frame_items = list(selected_frames.items())
            comparison = match_common_columns(
                {frame_items[0][0]: frame_items[0][1], frame_items[1][0]: frame_items[1][1]}
            )
            st.subheader("Column matching and schema differences")
            st.write(
                {
                    "common_columns": comparison["common_columns"],
                    "possible_matching_columns": comparison[
                        "possible_matching_columns"
                    ],
                    "unmatched_columns": {
                        frame_items[0][0]: comparison["unmatched_left_columns"],
                        frame_items[1][0]: comparison["unmatched_right_columns"],
                    },
                }
            )
            if integration_mode == "Merge datasets":
                merge_join_type = st.selectbox(
                    "Join type",
                    ["inner", "left", "right", "outer"],
                    key="csv_integration_join_type",
                )
                common_keys = comparison["common_columns"]
                merge_keys = st.multiselect(
                    "Merge key column(s)",
                    common_keys,
                    default=common_keys[:1],
                    key="csv_integration_merge_keys",
                )
                use_manual_mapping = st.checkbox(
                    "Map differently named merge keys",
                    value=not bool(common_keys),
                    key="csv_integration_use_manual_mapping",
                )
                if use_manual_mapping:
                    manual_left_keys = st.multiselect(
                        "Left dataset key column(s)",
                        [str(column) for column in frame_items[0][1].columns],
                        key="csv_integration_manual_left_keys",
                    )
                    manual_right_keys = st.multiselect(
                        "Right dataset key column(s), in matching order",
                        [str(column) for column in frame_items[1][1].columns],
                        key="csv_integration_manual_right_keys",
                    )
                    merge_keys = manual_left_keys
                    right_merge_keys = manual_right_keys
                    merge_mapping = dict(zip(manual_left_keys, manual_right_keys))
                    merge_keys_valid = (
                        bool(merge_keys)
                        and len(merge_keys) == len(right_merge_keys)
                    )
                else:
                    right_merge_keys = list(merge_keys)
                prevent_many_to_many = st.checkbox(
                    "Prevent many-to-many joins (recommended)",
                    value=True,
                    key="csv_integration_prevent_many_to_many",
                )
        can_run_integration = bool(selected_integration_keys) and (
            integration_mode != "Merge datasets"
            or (
                len(selected_integration_keys) >= 2
                and bool(merge_keys)
                and merge_keys_valid
            )
        )
        if st.button(
            "Run CSV Integration",
            disabled=not can_run_integration,
            key="run_csv_integration",
        ):
            try:
                integration_result = integrate_multiple_csv_datasets(
                    selected_integration_datasets,
                    integration_type=(
                        "append"
                        if integration_mode == "Append rows"
                        else "merge"
                    ),
                    join_type=merge_join_type,
                    key_columns=merge_keys or None,
                    right_key_columns=right_merge_keys or None,
                    column_mapping=merge_mapping,
                    standardize_column_names=standardize_schema,
                    many_to_many=(
                        "raise"
                        if integration_mode == "Merge datasets"
                        and prevent_many_to_many
                        else "allow"
                    ),
                )
                st.session_state["csv_integration_result"] = integration_result
                st.session_state["csv_integration_result_selection"] = list(
                    selected_integration_keys
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                st.error(f"CSV integration could not run: {error}")

        integration_result = st.session_state.get("csv_integration_result")
        if (
            integration_result is not None
            and st.session_state.get("csv_integration_result_selection")
            == list(selected_integration_keys)
        ):
            integration_report = integration_result["integration_report"]
            st.subheader("Integration statistics")
            if integration_report["integration_type"] == "append":
                st.dataframe(
                    pd.DataFrame(integration_report["source_record_counts"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                metrics = st.columns(4)
                metrics[0].metric("Matched records", integration_report["matched_records"])
                metrics[1].metric(
                    "Unmatched left records",
                    integration_report["unmatched_left_records"],
                )
                metrics[2].metric(
                    "Unmatched right records",
                    integration_report["unmatched_right_records"],
                )
                metrics[3].metric("Output records", integration_report["output_records"])
            st.subheader("Integrated CSV data")
            st.dataframe(integration_result["data"], width="stretch")
            st.subheader("Schema comparison report")
            st.json(integration_result["schema_report"])
            saved_integration = integration_result["saved_paths"]
            st.download_button(
                "Download integrated CSV",
                Path(saved_integration["data"]).read_bytes(),
                file_name=Path(saved_integration["data"]).name,
                mime="text/csv",
                key="download_integrated_csv",
            )
            st.download_button(
                "Download integration report (JSON)",
                Path(saved_integration["report_json"]).read_bytes(),
                file_name=Path(saved_integration["report_json"]).name,
                mime="application/json",
                key="download_csv_integration_report",
            )
            st.download_button(
                "Download schema differences (CSV)",
                Path(saved_integration["schema_csv"]).read_bytes(),
                file_name=Path(saved_integration["schema_csv"]).name,
                mime="text/csv",
                key="download_csv_integration_schema",
            )
            st.download_button(
                "Download matched and unmatched records (CSV)",
                Path(saved_integration["records_csv"]).read_bytes(),
                file_name=Path(saved_integration["records_csv"]).name,
                mime="text/csv",
                key="download_csv_integration_records",
            )
            st.caption(
                f"Integrated data saved under `{saved_integration['data_directory']}`; "
                f"reports saved under `{saved_integration['report_directory']}`."
            )
    else:
        st.info("Select at least one Stage 4 validated CSV dataset.")
else:
    st.info("Run Stage 4 CSV validation before integrating datasets.")

st.header("JSON ingestion")
json_files = st.file_uploader(
    "Select one or more JSON files",
    type=["json"],
    accept_multiple_files=True,
    key="json_upload",
)
if st.button("Extract JSON files", disabled=not json_files):
    st.session_state["json_results"] = extract_multiple_json_files(
        cast(list[BinaryIO], json_files)
    )
show_results("json_results")

st.header("Stage 2: JSON Data Profiling")
successful_json_results = [
    (index, item)
    for index, item in enumerate(st.session_state.get("json_results", []))
    if item["metadata"]["status"] == "success"
]
json_profile_choices = []
json_profile_sources = {}
for index, item in successful_json_results:
    source_name = item["metadata"]["source"]
    tables = item.get("dataframes")
    if isinstance(tables, dict) and tables:
        for table_name, dataframe in tables.items():
            if dataframe is not None:
                choice = f"{source_name}::{table_name}::{index}"
                json_profile_choices.append(choice)
                json_profile_sources[choice] = (
                    index,
                    source_name,
                    str(table_name),
                    dataframe,
                    item.get("json_data"),
                )
    elif item.get("data") is not None:
        choice = f"{source_name}::data::{index}"
        json_profile_choices.append(choice)
        json_profile_sources[choice] = (
            index,
            source_name,
            None,
            item["data"],
            item.get("json_data"),
        )

if json_profile_choices:
    selected_json_key = st.selectbox(
        "Select an ingested JSON dataset",
        json_profile_choices,
        format_func=lambda key: (
                f"{json_profile_sources[key][1]}"
                + (
                    f" — {json_profile_sources[key][2]}"
                    if json_profile_sources[key][2]
                    else ""
                )
        ),
        key="json_profile_selection",
    )
    (
        selected_json_index,
        selected_json_filename,
        selected_json_table,
        selected_json_dataframe,
        selected_parsed_json,
    ) = json_profile_sources[selected_json_key]
    if st.button("Profile selected JSON", key="profile_json_button"):
        json_profiles = st.session_state.setdefault("json_profile_results", {})
        json_profiles[selected_json_key] = profile_json(
            selected_json_dataframe,
            selected_json_filename,
            parsed_json=selected_parsed_json,
            table_name=selected_json_table,
        )

    json_profile = st.session_state.get("json_profile_results", {}).get(
        selected_json_key
    )
    if json_profile is not None:
        json_dimensions = json_profile["dimensions"]
        json_metrics = st.columns(4)
        json_metrics[0].metric("Rows", json_dimensions["total_rows"])
        json_metrics[1].metric("Columns", json_dimensions["total_columns"])
        json_metrics[2].metric(
            "Missing values", json_profile["missing_values"]["total_missing_values"]
        )
        json_metrics[3].metric(
            "Duplicate rows",
            json_profile["duplicates"]["total_duplicate_rows"],
        )

        json_missing_rows = [
            {"Column": column, **details}
            for column, details in json_profile["missing_values"]["columns"].items()
        ]
        st.subheader("JSON missing values by column")
        if json_missing_rows:
            st.dataframe(json_missing_rows, width="stretch", hide_index=True)
            st.bar_chart(
                {
                    row["Column"]: row["missing_count"]
                    for row in json_missing_rows
                }
            )
        else:
            st.info("This JSON table has no columns.")

        st.subheader("JSON column data types")
        st.dataframe(
            json_profile["data_types"],
            width="stretch",
            hide_index=True,
        )
        st.subheader("JSON duplicate records")
        st.write(
            f"Unique rows: {json_profile['duplicates']['unique_rows']} "
            f"({json_profile['duplicates']['duplicate_percentage']:.2f}% duplicates)"
        )
        st.dataframe(
            json_profile["duplicates"]["duplicate_records"],
            width="stretch",
        )
        st.subheader("Complete JSON data quality report")
        json_report = json_profile["quality_summary"]
        st.json(
            {
                key: value
                for key, value in json_report.items()
                if key != "columns"
            }
        )
        st.dataframe(json_report["columns"], width="stretch", hide_index=True)
        st.session_state["stage3_json_datasets"] = {
            key: value
            for key, value in st.session_state.get(
                "json_profile_results", {}
            ).items()
        }
        st.caption(
            "Stage 3 handoff: the unchanged DataFrame and parsed JSON are available "
            "at `st.session_state['stage3_json_datasets'][dataset_id]`."
        )
else:
    st.info("Ingest a JSON file in Stage 1 before profiling it.")

st.header("Stage 3: JSON Data Cleaning")
cleanable_json = [
    (index, item) for index, item in enumerate(st.session_state.get("json_results", []))
    if item["metadata"].get("status") == "success" and item.get("json_data") is not None
]
if cleanable_json:
    labels = {i: f"{item['metadata']['source']} (dataset {i + 1})" for i, item in cleanable_json}
    clean_index = st.selectbox("Select an ingested JSON dataset to clean", [i for i, _ in cleanable_json],
                               format_func=labels.__getitem__, key="json_clean_selection")
    source_result = st.session_state["json_results"][clean_index]
    source_json = source_result["json_data"]
    source_name = source_result["metadata"]["source"]
    clean_key = f"{source_name}::{clean_index}"
    st.subheader("Original JSON structure")
    st.json(source_json)
    matching_profile = next((p for p in st.session_state.get("json_profile_results", {}).values() if
                             p.get("filename") == source_name and p.get("parsed_json") is source_json), None)
    st.subheader("Stage 2 JSON profiling results")
    if matching_profile is None:
        st.info("Run Stage 2 JSON profiling for this dataset to see its profile here.")
    else:
        st.json(matching_profile["quality_summary"])
        st.dataframe(matching_profile["data_types"], width="stretch", hide_index=True)

    operation_cols = st.columns(3)
    do_duplicates = operation_cols[0].checkbox("Remove duplicate records", key="json_clean_duplicates")
    do_missing = operation_cols[1].checkbox("Handle missing values", key="json_clean_missing")
    do_names = operation_cols[2].checkbox("Standardise field names", key="json_clean_names")
    do_dates = operation_cols[0].checkbox("Standardise dates", key="json_clean_dates")
    do_currency = operation_cols[1].checkbox("Standardise currencies", key="json_clean_currency")
    do_types = operation_cols[2].checkbox("Correct selected data types", key="json_clean_types")

    fields = sorted({path.split(".")[-1].split("[")[0] for path, _, _ in
                     __import__("cleaning.json_cleaner", fromlist=["_walk"])._walk(source_json)})
    duplicate_fields = st.multiselect("Optional duplicate comparison fields (empty means exact object)", fields,
                                      key="json_duplicate_fields") if do_duplicates else []
    missing_strategy, fill_value, missing_fields, field_strategies = "leave", None, None, None
    if do_missing:
        missing_strategy = st.selectbox("Missing-value strategy", ["leave", "drop_records", "replace"],
                                        format_func=lambda x:
                                        {"leave": "Leave unchanged", "drop_records": "Remove containing records",
                                         "replace": "Replace with a value"}[x], key="json_missing_strategy")
        if missing_strategy == "replace":
            fill_value = st.text_input("Replacement value (entered as text)", key="json_fill_value")
        missing_fields = st.multiselect("Fields to handle (empty means all fields)", fields,
                                        key="json_missing_fields") or None
        custom_missing_fields = st.multiselect("Fields with their own strategy (optional)", fields,
                                               key="json_custom_missing_fields")
        if custom_missing_fields:
            field_strategies = {}
            for field in custom_missing_fields:
                selected_strategy = st.selectbox(f"Strategy for {field}", ["leave", "drop_records", "replace"],
                                                 key=f"json_missing_strategy_{field}")
                selected_value = st.text_input(f"Replacement for {field}",
                                               key=f"json_missing_value_{field}") if selected_strategy == "replace" else None
                field_strategies[field] = {"strategy": selected_strategy, "value": selected_value}
    recursive_names = st.checkbox("Standardise nested object keys recursively", value=False,
                                  key="json_recursive_names") if do_names else False
    date_fields = st.multiselect("Date fields", fields,
                                 default=[f for f in fields if re.search(r"date|time|timestamp", f, re.I)],
                                 key="json_date_fields") if do_dates else None
    date_format = st.text_input("Optional source date format (for example %d/%m/%Y)",
                                key="json_date_format") or None if do_dates else None
    currency_fields = st.multiselect("Currency fields (select explicitly)", fields,
                                     key="json_currency_fields") if do_currency else None
    conversions = {}
    if do_types:
        type_fields = st.multiselect("Fields to convert", fields, key="json_type_fields")
        for field in type_fields:
            conversions[field] = st.selectbox(f"Target type for {field}",
                                              ["integer", "float", "string", "boolean", "date", "list", "dictionary",
                                               "null"], key=f"json_type_{field}")
    if st.button("Run JSON Cleaning", key="run_json_cleaning"):
        result = clean_json_data(source_json, source_name, remove_duplicates=do_duplicates,
                                 duplicate_subset=duplicate_fields or None, handle_missing=do_missing,
                                 missing_strategy=missing_strategy, missing_fill_value=fill_value,
                                 field_strategies=field_strategies, missing_fields=missing_fields,
                                 standardise_names=do_names, recursive_names=recursive_names,
                                 standardise_date_values=do_dates, date_fields=date_fields or None,
                                 date_format=date_format, standardise_currency_values=do_currency,
                                 currency_fields=currency_fields or None,
                                 type_conversions=conversions if do_types else None)
        saved_path = save_cleaned_json(result["data"], source_name)
        result["report"]["processed_file_path"] = str(saved_path)
        st.session_state.setdefault("json_clean_results", {})[clean_key] = result
    cleaned_result = st.session_state.get("json_clean_results", {}).get(clean_key)
    if cleaned_result:
        st.subheader("Cleaned JSON structure")
        st.json(cleaned_result["data"])
        st.subheader("Before and after report")
        st.json(cleaned_result["report"])
        st.download_button("Download cleaned JSON", json.dumps(cleaned_result["data"], ensure_ascii=False, indent=2),
                           file_name=Path(source_name).stem + "_cleaned.json", mime="application/json",
                           key="download_json_cleaned")
        st.caption(f"Processed output saved to `{cleaned_result['report']['processed_file_path']}`.")
else:
    st.info("Ingest a JSON file in Stage 1 before cleaning it.")

st.header("Stage 4: JSON Data Validation")
cleaned_json_datasets = st.session_state.get("json_clean_results", {})
json_validation_labels = {
    key: str(result.get("report", {}).get("dataset_name") or key)
    for key, result in cleaned_json_datasets.items()
    if isinstance(result, dict) and "data" in result
}
if json_validation_labels:
    selected_json_validation_key = st.selectbox(
        "Select a cleaned JSON dataset from Stage 3",
        list(json_validation_labels),
        format_func=json_validation_labels.__getitem__,
        key="json_validation_selection",
    )
    selected_json_clean = cleaned_json_datasets[selected_json_validation_key]
    validation_json = selected_json_clean["data"]
    validation_json_name = str(
        selected_json_clean.get("report", {}).get("dataset_name")
        or json_validation_labels[selected_json_validation_key]
    )
    json_structure = discover_json_structure(validation_json)
    st.subheader("Stage 3 cleaned JSON structure")
    st.json(validation_json)

    json_validation_flags = st.columns(3)
    validate_json_required_enabled = json_validation_flags[0].checkbox(
        "Validate required fields and structure",
        value=True,
        key="json_validation_required_enabled",
    )
    validate_json_numeric_enabled = json_validation_flags[1].checkbox(
        "Validate numerical fields",
        value=True,
        key="json_validation_numeric_enabled",
    )
    validate_json_dates_enabled = json_validation_flags[2].checkbox(
        "Validate date formats",
        value=True,
        key="json_validation_dates_enabled",
    )

    required_fields = {}
    warning_fields = {}
    required_arrays = {}
    if validate_json_required_enabled:
        object_paths = list(json_structure["object_fields"])
        selected_object_paths = st.multiselect(
            "Object locations with required fields",
            object_paths,
            key="json_validation_object_paths",
        )
        for object_path in selected_object_paths:
            available_fields = json_structure["object_fields"][object_path]
            selected_fields = st.multiselect(
                f"Required keys at {object_path}",
                available_fields,
                key=f"json_required_fields_{object_path}",
            )
            if selected_fields:
                required_fields[object_path] = selected_fields
            optional_fields = st.multiselect(
                f"Missing keys to report as warnings at {object_path}",
                [
                    field
                    for field in available_fields
                    if field not in selected_fields
                ],
                key=f"json_warning_fields_{object_path}",
            )
            if optional_fields:
                warning_fields[object_path] = optional_fields

        selected_array_paths = st.multiselect(
            "Arrays to validate for required presence and structure",
            json_structure["array_paths"],
            key="json_validation_array_paths",
        )
        for array_path in selected_array_paths:
            with st.expander(f"Array rules: {array_path}"):
                array_required = st.checkbox(
                    f"{array_path} must exist",
                    value=True,
                    key=f"json_array_required_{array_path}",
                )
                allow_empty_array = st.checkbox(
                    f"Allow {array_path} to be empty",
                    value=True,
                    key=f"json_array_empty_{array_path}",
                )
                item_type = st.selectbox(
                    f"Expected array item type for {array_path}",
                    ["Any", "object", "array", "string", "integer", "number", "boolean"],
                    key=f"json_array_item_type_{array_path}",
                )
            array_rule = {
                "required": array_required,
                "allow_empty": allow_empty_array,
            }
            if item_type != "Any":
                array_rule["item_type"] = item_type
            required_arrays[array_path] = array_rule

    numerical_rules = {}
    if validate_json_numeric_enabled:
        selected_numeric_paths = st.multiselect(
            "Fields expected to contain numbers",
            json_structure["scalar_paths"],
            key="json_validation_numeric_paths",
        )
        for field_path in selected_numeric_paths:
            with st.expander(f"Numerical rules: {field_path}"):
                numeric_type = st.selectbox(
                    f"Numeric type for {field_path}",
                    ["number", "integer"],
                    key=f"json_numeric_type_{field_path}",
                )
                numeric_min = st.text_input(
                    f"Minimum for {field_path} (optional)",
                    key=f"json_numeric_min_{field_path}",
                )
                numeric_max = st.text_input(
                    f"Maximum for {field_path} (optional)",
                    key=f"json_numeric_max_{field_path}",
                )
                negative_values_allowed = st.checkbox(
                    f"Allow negative values for {field_path}",
                    value=True,
                    key=f"json_numeric_negative_{field_path}",
                )
            numeric_rule = {
                "type": numeric_type,
                "allow_negative": negative_values_allowed,
            }
            if numeric_min and numeric_min.strip():
                numeric_rule["min"] = numeric_min.strip()
            if numeric_max and numeric_max.strip():
                numeric_rule["max"] = numeric_max.strip()
            numerical_rules[field_path] = numeric_rule

    date_rules = {}
    if validate_json_dates_enabled:
        selected_date_paths = st.multiselect(
            "Date fields (only selected paths are checked)",
            json_structure["scalar_paths"],
            key="json_validation_date_paths",
        )
        for field_path in selected_date_paths:
            date_format_option = st.text_input(
                f"Expected format for {field_path} (%Y-%m-%d for ISO date)",
                value="%Y-%m-%d",
                key=f"json_validation_date_format_{field_path}",
            )
            date_required = st.checkbox(
                f"{field_path} is required",
                key=f"json_validation_date_required_{field_path}",
            )
            date_rules[field_path] = {
                "format": date_format_option or "%Y-%m-%d",
                "required": date_required,
            }

    if st.button("Run JSON Validation", key="run_json_validation"):
        try:
            validation_result = validate_json_data(
                validation_json,
                validation_json_name,
                required_fields=required_fields,
                warning_fields=warning_fields,
                required_arrays=required_arrays,
                numerical_rules=numerical_rules,
                date_rules=date_rules,
                validate_required=validate_json_required_enabled,
                validate_numerical=validate_json_numeric_enabled,
                validate_dates=validate_json_dates_enabled,
            )
            validation_result["report_paths"] = export_json_validation_result(
                validation_result
            )
            st.session_state.setdefault("json_validation_results", {})[
                selected_json_validation_key
            ] = validation_result
        except (OSError, ValueError, TypeError) as error:
            st.error(f"JSON validation could not run: {error}")

    validation_result = st.session_state.get(
        "json_validation_results", {}
    ).get(selected_json_validation_key)
    if validation_result is not None:
        validation_report = validation_result["validation_report"]
        validation_status = validation_result["validation_status"]
        if validation_status == "passed":
            st.success(f"Validation {validation_status}.")
        elif validation_status == "passed_with_warnings":
            st.warning(f"Validation {validation_status}.")
        else:
            st.error(f"Validation {validation_status}.")
        json_metrics = st.columns(4)
        json_metrics[0].metric("Valid records", validation_report["valid_records"])
        json_metrics[1].metric("Invalid records", validation_report["invalid_records"])
        json_metrics[2].metric("Errors", validation_report["total_validation_errors"])
        json_metrics[3].metric("Warnings", validation_report["total_validation_warnings"])
        st.subheader("Invalid JSON records")
        st.json(validation_result["invalid_data"])
        st.subheader("Validation errors")
        st.dataframe(
            pd.DataFrame(validation_result["validation_errors"]),
            width="stretch",
            hide_index=True,
        )
        if validation_result["validation_warnings"]:
            st.subheader("Validation warnings")
            st.dataframe(
                pd.DataFrame(validation_result["validation_warnings"]),
                width="stretch",
                hide_index=True,
            )
        st.subheader("JSON validation report")
        st.json(validation_report)
        paths = validation_result["report_paths"]
        st.download_button(
            "Download valid JSON data",
            data=Path(paths["valid_json"]).read_bytes(),
            file_name=f"{Path(validation_json_name).stem}_valid.json",
            mime="application/json",
            key="download_json_valid_data",
        )
        st.download_button(
            "Download invalid JSON data",
            data=Path(paths["invalid_json"]).read_bytes(),
            file_name=f"{Path(validation_json_name).stem}_invalid.json",
            mime="application/json",
            key="download_json_invalid_data",
        )
        st.download_button(
            "Download JSON validation errors (CSV)",
            data=Path(paths["errors_csv"]).read_bytes(),
            file_name=f"{Path(validation_json_name).stem}_validation_errors.csv",
            mime="text/csv",
            key="download_json_validation_errors",
        )
        st.download_button(
            "Download JSON validation report",
            data=Path(paths["report_json"]).read_bytes(),
            file_name=f"{Path(validation_json_name).stem}_validation_report.json",
            mime="application/json",
            key="download_json_validation_report",
        )
else:
    st.info("Run Stage 3 JSON cleaning before validating a dataset.")

st.header("Stage 5: JSON Data Integration")
json_validation_results = st.session_state.get("json_validation_results", {})
integrable_json_datasets = {}
for dataset_key, validation_result in json_validation_results.items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and validation_result.get("valid_data") is not None
    ):
        report = validation_result.get("validation_report", {})
        dataset_name = str(
            report.get("dataset_name")
            or cleaned_json_datasets.get(dataset_key, {}).get("report", {}).get(
                "dataset_name"
            )
            or dataset_key
        )
        integrable_json_datasets[dataset_key] = {
            "validation_result": validation_result,
            "source": dataset_name,
        }

if integrable_json_datasets:
    json_integration_labels = {
        key: str(value["source"])
        for key, value in integrable_json_datasets.items()
    }
    selected_json_integration_keys = st.multiselect(
        "Select validated JSON datasets",
        list(json_integration_labels),
        format_func=json_integration_labels.__getitem__,
        key="json_integration_selection",
    )
    if selected_json_integration_keys:
        selected_json_integration_datasets = {
            key: integrable_json_datasets[key]
            for key in selected_json_integration_keys
        }
        selected_json_values = {
            key: value["validation_result"]["valid_data"]
            for key, value in selected_json_integration_datasets.items()
        }
        st.subheader("Selected validated JSON datasets")
        for dataset_key, data in selected_json_values.items():
            with st.expander(json_integration_labels[dataset_key]):
                st.json(data)
                st.json(
                    selected_json_integration_datasets[dataset_key][
                        "validation_result"
                    ].get("validation_report", {})
                )

        json_record_paths = {
            key: get_json_record_paths(data)
            for key, data in selected_json_values.items()
        }
        selected_record_paths = {}
        for dataset_key, paths in json_record_paths.items():
            if len(paths) > 1:
                selected_record_paths[dataset_key] = st.selectbox(
                    f"Record array in {json_integration_labels[dataset_key]}",
                    paths,
                    key=f"json_integration_record_path_{dataset_key}",
                )
            elif paths:
                selected_record_paths[dataset_key] = paths[0]

        json_integration_mode = st.radio(
            "JSON integration operation",
            ["Combine records", "Merge datasets"],
            horizontal=True,
            key="json_integration_type",
        )
        json_join_type = "inner"
        json_merge_keys = []
        json_right_merge_keys = []
        json_field_mapping = {}
        json_many_to_many = "raise"
        json_comparison = None
        if len(selected_json_values) >= 2:
            comparison_keys = list(selected_json_values)[:2]
            comparison_datasets = {
                key: selected_json_integration_datasets[key]
                for key in comparison_keys
            }
            comparison_paths = {
                key: selected_record_paths[key]
                for key in comparison_keys
                if key in selected_record_paths
            }
            try:
                json_comparison = match_common_fields(
                    comparison_datasets, record_paths=comparison_paths
                )
                st.subheader("Common, suggested, and unmatched JSON fields")
                st.write(
                    {
                        "common_fields": json_comparison["common_fields"],
                        "possible_matching_fields": json_comparison[
                            "possible_matching_fields"
                        ],
                        "unmatched_fields": {
                            json_comparison["left_dataset"]: json_comparison[
                                "unmatched_left_fields"
                            ],
                            json_comparison["right_dataset"]: json_comparison[
                                "unmatched_right_fields"
                            ],
                        },
                    }
                )
                schema_comparison = handle_json_inconsistent_schemas(
                    comparison_datasets, record_paths=comparison_paths
                )
                st.subheader("JSON schema comparison")
                st.json(schema_comparison)
            except (TypeError, ValueError) as error:
                st.warning(f"Could not compare JSON fields automatically: {error}")
                json_comparison = None

            if json_integration_mode == "Merge datasets" and json_comparison:
                json_join_type = st.selectbox(
                    "JSON join type",
                    ["inner", "left", "right", "outer"],
                    key="json_integration_join_type",
                )
                common_fields = json_comparison["common_fields"]
                json_merge_keys = st.multiselect(
                    "Common JSON merge field(s)",
                    common_fields,
                    default=common_fields[:1],
                    key="json_integration_merge_keys",
                )
                use_json_field_mapping = st.checkbox(
                    "Map differently named nested fields",
                    value=not bool(common_fields),
                    key="json_integration_manual_mapping_enabled",
                )
                if use_json_field_mapping:
                    left_fields = list(json_comparison["left_field_types"])
                    right_fields = list(json_comparison["right_field_types"])
                    json_merge_keys = st.multiselect(
                        "Left dataset key field(s)",
                        left_fields,
                        key="json_integration_left_keys",
                    )
                    json_right_merge_keys = st.multiselect(
                        "Right dataset key field(s), in matching order",
                        right_fields,
                        key="json_integration_right_keys",
                    )
                    json_field_mapping = dict(
                        zip(json_merge_keys, json_right_merge_keys)
                    )
                else:
                    json_right_merge_keys = list(json_merge_keys)
                allow_json_many_to_many = st.checkbox(
                    "Allow many-to-many JSON joins",
                    value=False,
                    key="json_integration_allow_many_to_many",
                )
                json_many_to_many = "allow" if allow_json_many_to_many else "raise"

        json_keys_valid = (
            bool(json_merge_keys)
            and len(json_merge_keys) == len(json_right_merge_keys)
        )
        can_run_json_integration = (
            bool(selected_json_integration_keys)
            and (
                json_integration_mode != "Merge datasets"
                or (
                    len(selected_json_integration_keys) >= 2
                    and json_keys_valid
                    and json_comparison is not None
                )
            )
        )
        if st.button(
            "Run JSON Integration",
            disabled=not can_run_json_integration,
            key="run_json_integration",
        ):
            try:
                integration_result = integrate_multiple_json_datasets(
                    {
                        key: {
                            "validation_result": value["validation_result"],
                            "source": value["source"],
                        }
                        for key, value in selected_json_integration_datasets.items()
                    },
                    integration_type=(
                        "combine"
                        if json_integration_mode == "Combine records"
                        else "merge"
                    ),
                    record_path=selected_record_paths,
                    join_type=json_join_type,
                    key_fields=json_merge_keys or None,
                    right_key_fields=json_right_merge_keys or None,
                    field_mapping=json_field_mapping,
                    left_record_path=selected_record_paths.get(
                        selected_json_integration_keys[0]
                    ),
                    right_record_path=(
                        selected_record_paths.get(selected_json_integration_keys[1])
                        if len(selected_json_integration_keys) >= 2
                        else None
                    ),
                    many_to_many=json_many_to_many,
                )
                st.session_state["json_integration_result"] = integration_result
                st.session_state["json_integration_result_selection"] = list(
                    selected_json_integration_keys
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                st.error(f"JSON integration could not run: {error}")

        json_integration_result = st.session_state.get("json_integration_result")
        if (
            json_integration_result is not None
            and st.session_state.get("json_integration_result_selection")
            == list(selected_json_integration_keys)
        ):
            integration_report = json_integration_result["integration_report"]
            st.subheader("JSON integration statistics")
            if integration_report["integration_type"] == "combine":
                st.dataframe(
                    pd.DataFrame(integration_report["source_record_counts"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                integration_metrics = st.columns(4)
                integration_metrics[0].metric(
                    "Matched records", integration_report["matched_records"]
                )
                integration_metrics[1].metric(
                    "Unmatched left records",
                    integration_report["unmatched_left_records"],
                )
                integration_metrics[2].metric(
                    "Unmatched right records",
                    integration_report["unmatched_right_records"],
                )
                integration_metrics[3].metric(
                    "Output records", integration_report["output_records"]
                )
            st.subheader("Integrated JSON")
            st.json(json_integration_result["data"])
            st.subheader("JSON integration report")
            st.json(integration_report)
            st.subheader("Matched and unmatched records")
            st.json(
                {
                    "matched_records": json_integration_result["matched_records"],
                    "unmatched_records": json_integration_result["unmatched_records"],
                }
            )
            saved_json_integration = json_integration_result["saved_paths"]
            st.download_button(
                "Download integrated JSON",
                Path(saved_json_integration["data"]).read_bytes(),
                file_name=Path(saved_json_integration["data"]).name,
                mime="application/json",
                key="download_integrated_json",
            )
            st.download_button(
                "Download JSON integration report",
                Path(saved_json_integration["report_json"]).read_bytes(),
                file_name=Path(saved_json_integration["report_json"]).name,
                mime="application/json",
                key="download_json_integration_report",
            )
            st.download_button(
                "Download JSON integration report (CSV)",
                Path(saved_json_integration["report_csv"]).read_bytes(),
                file_name=Path(saved_json_integration["report_csv"]).name,
                mime="text/csv",
                key="download_json_integration_report_csv",
            )
            st.download_button(
                "Download JSON schema comparison (CSV)",
                Path(saved_json_integration["schema_csv"]).read_bytes(),
                file_name=Path(saved_json_integration["schema_csv"]).name,
                mime="text/csv",
                key="download_json_integration_schema",
            )
            st.download_button(
                "Download matched and unmatched JSON records (CSV)",
                Path(saved_json_integration["records_csv"]).read_bytes(),
                file_name=Path(saved_json_integration["records_csv"]).name,
                mime="text/csv",
                key="download_json_integration_records",
            )
            st.caption(
                f"Integrated JSON saved under `{saved_json_integration['data_directory']}`; "
                f"reports saved under `{saved_json_integration['report_directory']}`."
            )
    else:
        st.info("Select at least one Stage 4 validated JSON dataset.")
else:
    st.info("Run Stage 4 JSON validation before integrating datasets.")

st.header("XML ingestion")
xml_files = st.file_uploader(
    "Select one or more XML files",
    type=["xml"],
    accept_multiple_files=True,
    key="xml_upload",
)
if st.button("Extract XML files", disabled=not xml_files):
    st.session_state["xml_results"] = extract_multiple_xml_files(
        cast(list[BinaryIO], xml_files)
    )

for index, result in enumerate(st.session_state.get("xml_results", [])):
    metadata = result["metadata"]
    if metadata["status"] != "success":
        st.error(f"{metadata['source']}: {metadata['error']}")
        st.json(metadata)
        continue

    st.success(f"Extracted {metadata['source']}")
    st.json(metadata)
    st.write(f"Root element: `{metadata['root_element']}`")
    with st.expander(f"XML hierarchy: {metadata['source']}"):
        st.code(
            element_tree.tostring(result["xml_tree"], encoding="unicode"),
            language="xml",
        )

    table_names = list(result["dataframes"])
    selected_table = st.selectbox(
        f"Extracted table for {metadata['source']}",
        table_names,
        key=f"xml_table_{index}",
    )
    st.dataframe(
        result["dataframes"][selected_table],
        width="stretch",
    )

st.header("Stage 2: XML Data Profiling")
xml_profile_choices = []
xml_profile_sources = {}
for index, result in enumerate(st.session_state.get("xml_results", [])):
    metadata = result["metadata"]
    tables = result.get("dataframes")
    if metadata["status"] != "success" or not isinstance(tables, dict):
        continue
    for table_name, dataframe in tables.items():
        if dataframe is not None:
            choice = f"{metadata['source']}::{table_name}::{index}"
            xml_profile_choices.append(choice)
            xml_profile_sources[choice] = (
                index,
                metadata["source"],
                str(table_name),
                dataframe,
                metadata,
                result.get("xml_tree"),
            )

if xml_profile_choices:
    selected_xml_key = st.selectbox(
        "Select an ingested XML table",
        xml_profile_choices,
        format_func=lambda key: (
            f"{xml_profile_sources[key][1]} — {xml_profile_sources[key][2]}"
        ),
        key="xml_profile_selection",
    )
    (
        _,
        selected_xml_filename,
        selected_xml_table,
        selected_xml_dataframe,
        selected_xml_metadata,
        selected_xml_tree,
    ) = xml_profile_sources[selected_xml_key]
    if st.button("Profile selected XML", key="profile_xml_button"):
        table_metadata = dict(selected_xml_metadata)
        table_metadata["table_name"] = selected_xml_table
        xml_profile = profile_xml(
            selected_xml_dataframe,
            selected_xml_filename,
            xml_metadata=table_metadata,
        )
        xml_profile["xml_tree"] = selected_xml_tree
        st.session_state.setdefault("xml_profile_results", {})[
            selected_xml_key
        ] = xml_profile

    xml_profile = st.session_state.get("xml_profile_results", {}).get(
        selected_xml_key
    )
    if xml_profile is not None:
        xml_dimensions = xml_profile["dimensions"]
        xml_metrics = st.columns(4)
        xml_metrics[0].metric("Rows", xml_dimensions["total_rows"])
        xml_metrics[1].metric("Columns", xml_dimensions["total_columns"])
        xml_metrics[2].metric(
            "Missing values", xml_profile["missing_values"]["total_missing_values"]
        )
        xml_metrics[3].metric(
            "Duplicate rows",
            xml_profile["duplicates"]["total_duplicate_rows"],
        )

        xml_missing_rows = [
            {"Column": column, **details}
            for column, details in xml_profile["missing_values"]["columns"].items()
        ]
        st.subheader("XML missing values by column")
        if xml_missing_rows:
            st.dataframe(xml_missing_rows, width="stretch", hide_index=True)
            st.bar_chart(
                {
                    row["Column"]: row["missing_count"]
                    for row in xml_missing_rows
                }
            )
        else:
            st.info("This XML table has no columns.")

        st.subheader("XML column data types")
        st.dataframe(
            xml_profile["data_types"],
            width="stretch",
            hide_index=True,
        )
        st.subheader("XML duplicate records")
        st.write(
            f"Unique rows: {xml_profile['duplicates']['unique_rows']} "
            f"({xml_profile['duplicates']['duplicate_percentage']:.2f}% duplicates)"
        )
        st.dataframe(
            xml_profile["duplicates"]["duplicate_records"],
            width="stretch",
        )
        st.subheader("XML document metadata")
        st.json(
            {
                key: selected_xml_metadata.get(key)
                for key in (
                "source",
                "root_element",
                "namespaces",
                "element_count",
                "attribute_count",
                "repeating_record_elements",
                "dataframes",
            )
            }
        )
        st.subheader("Complete XML data quality report")
        xml_report = xml_profile["quality_summary"]
        st.json(
            {
                key: value
                for key, value in xml_report.items()
                if key != "columns"
            }
        )
        st.dataframe(xml_report["columns"], width="stretch", hide_index=True)
        st.session_state["stage3_xml_datasets"] = {
            key: value
            for key, value in st.session_state.get(
                "xml_profile_results", {}
            ).items()
        }
        st.caption(
            "Stage 3 handoff: the unchanged DataFrame, XML metadata, and parsed "
            "XML tree are available at "
            "`st.session_state['stage3_xml_datasets'][dataset_id]`."
        )
else:
    st.info("Ingest an XML file in Stage 1 before profiling it.")

st.header("Stage 3: XML Data Cleaning")
cleanable_xml = [
    (index, item) for index, item in enumerate(st.session_state.get("xml_results", []))
    if item["metadata"].get("status") == "success" and item.get("xml_tree") is not None
]
if cleanable_xml:
    xml_labels = {index: f"{item['metadata']['source']} (dataset {index + 1})" for index, item in cleanable_xml}
    xml_clean_index = st.selectbox("Select an ingested XML dataset to clean", [index for index, _ in cleanable_xml],
                                   format_func=xml_labels.__getitem__, key="xml_clean_selection")
    xml_extraction = st.session_state["xml_results"][xml_clean_index]
    xml_root = xml_extraction["xml_tree"]
    xml_name = xml_extraction["metadata"]["source"]
    xml_clean_key = f"{xml_name}::{xml_clean_index}"
    st.subheader("Original XML structure")
    st.code(element_tree.tostring(xml_root, encoding="unicode"), language="xml")
    st.subheader("Stage 2 XML profiling results")
    source_profiles = [profile for profile in st.session_state.get("xml_profile_results", {}).values() if
                       profile.get("filename") == xml_name and profile.get("xml_tree") is xml_root]
    if source_profiles:
        st.dataframe([{"Table": profile.get("table_name"), "Rows": profile["dimensions"]["total_rows"],
                       "Columns": profile["dimensions"]["total_columns"],
                       "Missing values": profile["missing_values"]["total_missing_values"],
                       "Duplicate rows": profile["duplicates"]["total_duplicate_rows"]} for profile in source_profiles],
                     width="stretch", hide_index=True)
        with st.expander("Full Stage 2 quality reports"):
            for profile in source_profiles:
                st.write(profile.get("table_name"))
                st.json({key: value for key, value in profile["quality_summary"].items() if key != "columns"})
    else:
        st.info("Run Stage 2 XML profiling for this dataset to see its profiles here.")

    groups = get_record_groups(xml_root)
    group_paths = [group["path"] for group in groups]
    element_names = sorted({str(node.tag).split("}", 1)[-1] for node in xml_root.iter()})
    attribute_names = sorted({str(name).split("}", 1)[-1] for node in xml_root.iter() for name in node.attrib})
    xml_flags = st.columns(3)
    xml_do_duplicates = xml_flags[0].checkbox("Remove duplicate XML records", key="xml_clean_duplicates")
    xml_do_missing = xml_flags[1].checkbox("Handle missing values", key="xml_clean_missing")
    xml_do_names = xml_flags[2].checkbox("Standardise element and attribute names", key="xml_clean_names")
    xml_do_dates = xml_flags[0].checkbox("Standardise dates", key="xml_clean_dates")
    xml_do_currency = xml_flags[1].checkbox("Standardise currencies", key="xml_clean_currencies")
    xml_do_types = xml_flags[2].checkbox("Convert selected value types", key="xml_clean_types")

    duplicate_group = None
    duplicate_fields = None
    if xml_do_duplicates:
        duplicate_group = st.selectbox("Repeating group for duplicate removal", group_paths,
                                       key="xml_duplicate_group") if group_paths else None
        duplicate_fields_selection = st.multiselect(
            "Optional duplicate comparison values (empty means exact nested record)",
            [*element_names, *[f"@{name}" for name in attribute_names]], key="xml_duplicate_fields")
        duplicate_fields = duplicate_fields_selection or None

    missing_strategy, missing_elements, missing_attributes, drop_fields = "leave", None, None, None
    missing_fill_value = ""
    if xml_do_missing:
        missing_strategy = st.selectbox("Missing-value strategy", ["leave", "replace", "drop_records"],
                                        key="xml_missing_strategy")
        missing_elements = st.multiselect("Selected element names to handle", element_names, key="xml_missing_elements")
        missing_attributes = st.multiselect("Selected attributes to populate/check", attribute_names,
                                            key="xml_missing_attributes")
        if missing_strategy == "replace":
            missing_fill_value = st.text_input("Replacement text for empty elements and attributes",
                                               key="xml_missing_fill")
        if missing_strategy == "drop_records":
            drop_fields = st.multiselect("Fields whose absence/emptiness removes a record",
                                         [*element_names, *[f"@{name}" for name in attribute_names]],
                                         key="xml_drop_fields")
            if duplicate_group is None and group_paths:
                duplicate_group = st.selectbox("Repeating group for missing-field record removal", group_paths,
                                               key="xml_missing_group")

    rename_elements = st.checkbox("Rename element tags", value=True,
                                  key="xml_rename_elements") if xml_do_names else False
    rename_attributes = st.checkbox("Rename attribute names", value=True,
                                    key="xml_rename_attributes") if xml_do_names else False
    recursive_names = st.checkbox("Apply name standardisation recursively", value=True,
                                  key="xml_recursive_names") if xml_do_names else False
    date_elements = st.multiselect("Date elements", element_names, default=[name for name in element_names if
                                                                            re.search(r"date|time|timestamp", name,
                                                                                      re.I)],
                                   key="xml_date_elements") if xml_do_dates else None
    date_attributes = st.multiselect("Date attributes", attribute_names, default=[name for name in attribute_names if
                                                                                  re.search(r"date|time|timestamp",
                                                                                            name, re.I)],
                                     key="xml_date_attributes") if xml_do_dates else None
    date_format = (st.text_input("Optional source date format (for example %d/%m/%Y)",
                                 key="xml_date_format") or None) if xml_do_dates else None
    currency_elements = st.multiselect("Currency elements (choose explicitly)", element_names,
                                       key="xml_currency_elements") if xml_do_currency else None
    currency_attributes = st.multiselect("Currency attributes (choose explicitly)", attribute_names,
                                         key="xml_currency_attributes") if xml_do_currency else None
    type_conversions = {}
    if xml_do_types:
        type_options = [*element_names, *[f"@{name}" for name in attribute_names]]
        selected_type_fields = st.multiselect("Elements or attributes to convert", type_options, key="xml_type_fields")
        for field in selected_type_fields:
            type_conversions[field] = st.selectbox(f"Target type for {field}",
                                                   ["integer", "float", "string", "boolean", "date", "datetime"],
                                                   key=f"xml_target_{field}")

    if st.button("Run XML Cleaning", key="run_xml_cleaning"):
        cleaned_result = clean_xml_data(xml_extraction, xml_name, remove_duplicates=xml_do_duplicates,
                                        duplicate_group=duplicate_group, duplicate_fields=duplicate_fields,
                                        handle_missing=xml_do_missing, missing_strategy=missing_strategy,
                                        missing_elements=missing_elements or None,
                                        missing_attributes=missing_attributes or None, drop_fields=drop_fields,
                                        missing_fill_value=missing_fill_value, standardise_names_enabled=xml_do_names,
                                        recursive_names=recursive_names, standardise_elements=rename_elements,
                                        standardise_attributes=rename_attributes, standardise_date_values=xml_do_dates,
                                        date_elements=date_elements or None, date_attributes=date_attributes or None,
                                        date_format=date_format, standardise_currency_values=xml_do_currency,
                                        currency_elements=currency_elements or None,
                                        currency_attributes=currency_attributes or None,
                                        type_conversions=type_conversions if xml_do_types else None)
        output_path = save_cleaned_xml(cleaned_result["xml_tree"], xml_name,
                                       xml_declaration=cleaned_result["xml_declaration"],
                                       namespaces=cleaned_result.get("namespaces"))
        cleaned_result["report"]["processed_file_path"] = str(output_path)
        st.session_state.setdefault("xml_clean_results", {})[xml_clean_key] = cleaned_result
    xml_clean_result = st.session_state.get("xml_clean_results", {}).get(xml_clean_key)
    if xml_clean_result:
        st.subheader("Cleaned XML structure")
        xml_text = element_tree.tostring(xml_clean_result["xml_tree"], encoding="unicode")
        st.code(xml_text, language="xml")
        stats = st.columns(4)
        stats[0].metric("Elements",
                        f"{xml_clean_result['report']['original_element_count']} → {xml_clean_result['report']['final_element_count']}")
        stats[1].metric("Attributes",
                        f"{xml_clean_result['report']['original_attribute_count']} → {xml_clean_result['report']['final_attribute_count']}")
        stats[2].metric("Duplicate records removed", xml_clean_result["report"]["duplicate_records_removed"])
        stats[3].metric("Cleaning status", xml_clean_result["report"]["cleaning_status"])
        st.subheader("XML cleaning report")
        st.json(xml_clean_result["report"])
        st.download_button("Download cleaned XML", Path(xml_clean_result["report"]["processed_file_path"]).read_bytes(),
                           file_name=Path(xml_name).stem + "_cleaned.xml", mime="application/xml",
                           key="download_xml_cleaned")
        st.caption(f"Processed XML saved to `{xml_clean_result['report']['processed_file_path']}`.")
else:
    st.info("Ingest an XML file in Stage 1 before cleaning it.")

st.header("Stage 4: XML Data Validation")
xml_clean_results = st.session_state.get("xml_clean_results", {})
if xml_clean_results:
    xml_validation_keys = list(xml_clean_results)
    xml_validation_labels = {
        key: str(
            xml_clean_results[key].get("report", {}).get("dataset_name", key)
        )
        for key in xml_validation_keys
    }
    xml_validation_key = st.selectbox(
        "Select a cleaned XML dataset to validate",
        xml_validation_keys,
        format_func=xml_validation_labels.__getitem__,
        key="xml_validation_selection",
    )
    xml_clean_result = xml_clean_results[xml_validation_key]
    validation_root = xml_clean_result.get("xml_tree")
    if isinstance(validation_root, element_tree.Element):
        validation_name = str(
            xml_clean_result.get("report", {}).get(
                "dataset_name", xml_validation_labels[xml_validation_key]
            )
        )
        st.subheader("Cleaned XML structure")
        st.code(
            element_tree.tostring(validation_root, encoding="unicode"),
            language="xml",
        )
        st.subheader("Stage 2 XML profiling results")
        validation_profiles = [
            profile
            for profile in st.session_state.get("xml_profile_results", {}).values()
            if profile.get("filename") == validation_name
        ]
        if validation_profiles:
            st.dataframe(
                [
                    {
                        "Record group": profile.get("table_name"),
                        "Rows": profile["dimensions"]["total_rows"],
                        "Columns": profile["dimensions"]["total_columns"],
                        "Missing values": profile["missing_values"][
                            "total_missing_values"
                        ],
                        "Duplicate rows": profile["duplicates"][
                            "total_duplicate_rows"
                        ],
                    }
                    for profile in validation_profiles
                ],
                width="stretch",
                hide_index=True,
            )
            with st.expander("Full Stage 2 XML quality reports"):
                for profile in validation_profiles:
                    st.write(profile.get("table_name"))
                    st.json(
                        {
                            key: value
                            for key, value in profile["quality_summary"].items()
                            if key != "columns"
                        }
                    )
        else:
            st.info("No Stage 2 XML profiling results are available for this dataset.")

        validation_groups = get_record_groups(validation_root)
        validation_group_paths = [group["path"] for group in validation_groups]
        validation_paths = get_xml_validation_paths(validation_root)
        if not validation_group_paths:
            validation_group_paths = [f"/{str(validation_root.tag).split('}', 1)[-1]}"]
        rule_group = st.selectbox(
            "Record group for required fields",
            validation_group_paths,
            key="xml_validation_required_group",
        )
        required_element_text = st.text_input(
            "Required element paths relative to each record (comma-separated)",
            key="xml_validation_required_elements",
        ) or ""
        required_attribute_text = st.text_input(
            "Required attribute names (comma-separated; omit @)",
            key="xml_validation_required_attributes",
        ) or ""
        required_elements = (
            {rule_group: [field.strip() for field in required_element_text.split(",") if field.strip()]}
            if required_element_text.strip()
            else {}
        )
        required_attributes = (
            {
                rule_group: [
                    f"@{field.strip().lstrip('@')}"
                    for field in required_attribute_text.split(",")
                    if field.strip()
                ]
            }
            if required_attribute_text.strip()
            else {}
        )

        numerical_choices = st.multiselect(
            "Numerical XML elements or attributes",
            [*validation_paths["elements"], *validation_paths["attributes"]],
            key="xml_validation_numeric_paths",
        )
        numerical_rules = {}
        for field_path in numerical_choices:
            numeric_columns = st.columns(4)
            numeric_type = numeric_columns[0].selectbox(
                f"Type: {field_path}",
                ["integer", "float"],
                key=f"xml_validation_numeric_type_{field_path}",
            )
            minimum = numeric_columns[1].text_input(
                "Minimum",
                key=f"xml_validation_numeric_min_{field_path}",
            )
            maximum = numeric_columns[2].text_input(
                "Maximum",
                key=f"xml_validation_numeric_max_{field_path}",
            )
            allow_negative = numeric_columns[3].checkbox(
                "Allow negative",
                value=True,
                key=f"xml_validation_numeric_negative_{field_path}",
            )
            rule = {"type": numeric_type, "allow_negative": allow_negative}
            if minimum.strip():
                rule["minimum"] = float(minimum)
            if maximum.strip():
                rule["maximum"] = float(maximum)
            numerical_rules[field_path] = rule

        date_choices = st.multiselect(
            "Date XML elements or attributes",
            [*validation_paths["elements"], *validation_paths["attributes"]],
            key="xml_validation_date_paths",
        )
        date_format = st.text_input(
            "Optional date format (blank validates ISO 8601)",
            key="xml_validation_date_format",
        ) or ""
        required_dates = st.checkbox(
            "Configured date fields are required",
            key="xml_validation_dates_required",
        )
        date_rules = {
            field_path: {
                "formats": [date_format] if date_format.strip() else [],
                "required": required_dates,
            }
            for field_path in date_choices
        }
        enabled = st.columns(3)
        validate_required = enabled[0].checkbox(
            "Validate required fields", value=True, key="xml_validate_required"
        )
        validate_numerical = enabled[1].checkbox(
            "Validate numerical values", value=True, key="xml_validate_numerical"
        )
        validate_dates = enabled[2].checkbox(
            "Validate dates", value=True, key="xml_validate_dates"
        )

        if st.button("Run XML Validation", key="run_xml_validation"):
            result = validate_xml_data(
                xml_clean_result,
                validation_name,
                required_elements=required_elements,
                required_attributes=required_attributes,
                numerical_rules=numerical_rules,
                date_rules=date_rules,
                validate_required=validate_required,
                validate_numerical=validate_numerical,
                validate_dates=validate_dates,
            )
            result["saved_paths"] = export_xml_validation_result(result)
            st.session_state.setdefault("xml_validation_results", {})[
                xml_validation_key
            ] = result

        xml_validation_result = st.session_state.get(
            "xml_validation_results", {}
        ).get(xml_validation_key)
        if xml_validation_result:
            report = xml_validation_result["validation_report"]
            st.metric("XML validation status", report["validation_status"])
            counts = st.columns(4)
            counts[0].metric("Records checked", report["total_records_checked"])
            counts[1].metric("Valid records", report["valid_records"])
            counts[2].metric("Invalid records", report["invalid_records"])
            counts[3].metric(
                "Errors / warnings",
                f"{report['total_validation_errors']} / "
                f"{report['total_validation_warnings']}",
            )
            st.subheader("XML validation errors")
            st.dataframe(
                xml_validation_result["validation_errors"],
                width="stretch",
                hide_index=True,
            )
            st.subheader("XML validation warnings")
            st.dataframe(
                xml_validation_result["validation_warnings"],
                width="stretch",
                hide_index=True,
            )
            st.subheader("Record-level validation report")
            st.json(report)
            saved = xml_validation_result["saved_paths"]
            if "valid_xml" in saved:
                st.download_button(
                    "Download valid XML",
                    Path(saved["valid_xml"]).read_bytes(),
                    file_name="valid.xml",
                    mime="application/xml",
                    key="download_xml_valid",
                )
            if "invalid_xml" in saved:
                st.download_button(
                    "Download invalid XML records",
                    Path(saved["invalid_xml"]).read_bytes(),
                    file_name="invalid.xml",
                    mime="application/xml",
                    key="download_xml_invalid",
                )
            st.download_button(
                "Download XML validation report (JSON)",
                Path(saved["report_json"]).read_bytes(),
                file_name="xml_validation_report.json",
                mime="application/json",
                key="download_xml_validation_json",
            )
            st.download_button(
                "Download XML validation errors (CSV)",
                Path(saved["errors_csv"]).read_bytes(),
                file_name="xml_validation_errors.csv",
                mime="text/csv",
                key="download_xml_validation_csv",
            )
            st.caption(f"Validation outputs saved to `{saved['directory']}`.")
    else:
        st.error("The selected Stage 3 result does not contain a cleaned XML tree.")
else:
    st.info("Run Stage 3 XML cleaning before validating an XML dataset.")

st.header("Stage 5: XML Data Integration")
xml_clean_results_for_integration = st.session_state.get("xml_clean_results", {})
xml_validation_results_for_integration = st.session_state.get(
    "xml_validation_results", {}
)
validated_xml_datasets = {}
xml_dataset_labels = {}
for dataset_key, validation_result in xml_validation_results_for_integration.items():
    status = validation_result.get("validation_status")
    valid_root = validation_result.get("valid_data")
    if status not in {"passed", "passed_with_warnings"} or not isinstance(
        valid_root, element_tree.Element
    ):
        continue
    cleaned = xml_clean_results_for_integration.get(dataset_key, {})
    dataset_name = str(
        cleaned.get("report", {}).get("dataset_name", dataset_key)
    )
    validated_xml_datasets[dataset_key] = {
        "validation_result": validation_result,
        "source": dataset_name,
        "namespaces": validation_result.get("namespaces")
        or cleaned.get("namespaces", {}),
        "xml_declaration": validation_result.get(
            "xml_declaration", cleaned.get("xml_declaration", True)
        ),
        "metadata": cleaned.get("metadata", {}),
    }
    xml_dataset_labels[dataset_key] = dataset_name

if len(validated_xml_datasets) < 1:
    st.info("Validate at least one XML dataset in Stage 4 before integration.")
else:
    selected_xml_keys = st.multiselect(
        "Select cleaned and validated XML datasets",
        list(validated_xml_datasets),
        format_func=xml_dataset_labels.__getitem__,
        key="xml_integration_datasets",
    )
    if selected_xml_keys:
        integration_roots = {
            key: validated_xml_datasets[key]["validation_result"]["valid_data"]
            for key in selected_xml_keys
        }
        xml_groups_by_key = {
            key: get_xml_record_groups(root)
            for key, root in integration_roots.items()
        }
        integration_record_paths = {}
        available_for_integration = True
        for key in selected_xml_keys:
            eligible_groups = [
                group for group in xml_groups_by_key[key] if group["parent_count"] == 1
            ]
            if not eligible_groups:
                available_for_integration = False
                st.warning(
                    f"{xml_dataset_labels[key]} has no repeated record group with a "
                    "single parent. Select a root-level or otherwise unambiguous "
                    "repeating group to preserve XML parent-child relationships."
                )
                continue
            group_paths = [group["path"] for group in eligible_groups]
            selected_path = st.selectbox(
                f"Record group: {xml_dataset_labels[key]}",
                group_paths,
                key=f"xml_integration_group_{key}",
            )
            integration_record_paths[key] = selected_path
            with st.expander(f"XML structure: {xml_dataset_labels[key]}"):
                st.json(
                    {
                        "root_element": str(integration_roots[key].tag),
                        "record_groups": xml_groups_by_key[key],
                        "validation_metadata": validated_xml_datasets[key][
                            "validation_result"
                        ]["validation_report"],
                    }
                )

        if len(selected_xml_keys) >= 2 and available_for_integration:
            selected_datasets = {
                key: validated_xml_datasets[key] for key in selected_xml_keys
            }
            try:
                xml_schema_preview = handle_xml_inconsistent_schemas(
                    selected_datasets, record_paths=integration_record_paths
                )
                st.subheader("XML schema comparison")
                st.dataframe(
                    xml_schema_preview["schema_report"],
                    width="stretch",
                    hide_index=True,
                )
                if len(selected_xml_keys) >= 2:
                    xml_matching_preview = match_common_elements(
                        {
                            selected_xml_keys[0]: selected_datasets[selected_xml_keys[0]],
                            selected_xml_keys[1]: selected_datasets[selected_xml_keys[1]],
                        },
                        record_paths=integration_record_paths,
                    )
                    st.write("Common XML fields")
                    st.json(
                        {
                            "common_elements": xml_matching_preview["common_elements"],
                            "common_attributes": xml_matching_preview["common_attributes"],
                            "possible_matching_fields": xml_matching_preview[
                                "possible_matching_fields"
                            ],
                            "unmatched_left_fields": xml_matching_preview[
                                "unmatched_left_fields"
                            ],
                            "unmatched_right_fields": xml_matching_preview[
                                "unmatched_right_fields"
                            ],
                            "incompatible_common_fields": xml_matching_preview[
                                "incompatible_common_fields"
                            ],
                        }
                    )
            except (TypeError, ValueError) as error:
                st.warning(f"Could not compare the selected XML record groups: {error}")

        xml_integration_type = st.selectbox(
            "XML integration operation",
            ["combine", "merge"],
            key="xml_integration_type",
        )
        xml_join_type = "inner"
        xml_left_keys = []
        xml_right_keys = []
        if xml_integration_type == "merge":
            if len(selected_xml_keys) != 2:
                st.info("XML merge currently accepts exactly two datasets at a time.")
                available_for_integration = False
            elif available_for_integration:
                try:
                    left_key = selected_xml_keys[0]
                    right_key = selected_xml_keys[1]
                    matching = match_common_elements(
                        {
                            left_key: validated_xml_datasets[left_key],
                            right_key: validated_xml_datasets[right_key],
                        },
                        record_paths=integration_record_paths,
                    )
                    left_fields = sorted(
                        set(matching["common_fields"])
                        | set(matching["unmatched_left_fields"])
                    )
                    right_fields = sorted(
                        set(matching["common_fields"])
                        | set(matching["unmatched_right_fields"])
                    )
                    mapping_columns = st.columns(2)
                    xml_left_keys = mapping_columns[0].multiselect(
                        "Left dataset merge key fields",
                        left_fields,
                        default=matching["common_fields"][:1],
                        key="xml_merge_left_keys",
                    )
                    xml_right_keys = mapping_columns[1].multiselect(
                        "Right dataset merge key fields (same order)",
                        right_fields,
                        default=matching["common_fields"][:1],
                        key="xml_merge_right_keys",
                    )
                    xml_join_type = st.selectbox(
                        "XML merge type",
                        ["inner", "left", "right", "outer"],
                        key="xml_merge_type",
                    )
                    st.caption(
                        "Map differently named element or attribute paths by selecting "
                        "corresponding fields in the same order. Many-to-many matches "
                        "are rejected by default."
                    )
                except (TypeError, ValueError) as error:
                    st.warning(f"Cannot configure an XML merge for these groups: {error}")
                    available_for_integration = False

        run_xml_integration = st.button(
            "Run XML Integration",
            key="run_xml_integration",
            disabled=not selected_xml_keys or not available_for_integration,
        )
        if run_xml_integration:
            try:
                field_mapping = {}
                if xml_integration_type == "merge":
                    if not xml_left_keys or not xml_right_keys:
                        raise ValueError("Select one or more matching merge keys.")
                    if len(xml_left_keys) != len(xml_right_keys):
                        raise ValueError(
                            "Select the same number of left and right merge key fields "
                            "in matching order."
                        )
                    field_mapping = dict(zip(xml_left_keys, xml_right_keys))
                result = integrate_multiple_xml_datasets(
                    selected_datasets,
                    integration_type=xml_integration_type,
                    record_paths=integration_record_paths,
                    join_type=xml_join_type,
                    key_fields=xml_left_keys or None,
                    right_key_fields=xml_right_keys or None,
                    field_mapping=field_mapping,
                )
                st.session_state["xml_integration_result"] = result
            except (OSError, TypeError, ValueError) as error:
                st.error(f"XML integration failed: {error}")

    xml_integration_result = st.session_state.get("xml_integration_result")
    if xml_integration_result:
        st.metric(
            "XML integration status",
            xml_integration_result.get("integration_status", "unknown"),
        )
        st.subheader("Integrated XML document")
        st.code(
            element_tree.tostring(
                xml_integration_result["data"], encoding="unicode"
            ),
            language="xml",
        )
        st.subheader("XML integration report")
        st.json(xml_integration_result["integration_report"])
        st.subheader("Matched and unmatched records")
        st.json(
            {
                "matched_records": len(xml_integration_result["matched_records"]),
                "unmatched_records": {
                    side: len(records)
                    for side, records in xml_integration_result[
                        "unmatched_records"
                    ].items()
                },
                "sources": xml_integration_result["source_datasets"],
                "schema_report": xml_integration_result["schema_report"],
            }
        )
        if "saved_paths" in xml_integration_result:
            saved_xml_paths = xml_integration_result["saved_paths"]
            st.download_button(
                "Download integrated XML",
                Path(saved_xml_paths["data"]).read_bytes(),
                file_name=Path(saved_xml_paths["data"]).name,
                mime="application/xml",
                key="download_integrated_xml",
            )
            st.download_button(
                "Download XML integration report (JSON)",
                Path(saved_xml_paths["report_json"]).read_bytes(),
                file_name=Path(saved_xml_paths["report_json"]).name,
                mime="application/json",
                key="download_xml_integration_json",
            )
            st.download_button(
                "Download XML integration report (CSV)",
                Path(saved_xml_paths["report_csv"]).read_bytes(),
                file_name=Path(saved_xml_paths["report_csv"]).name,
                mime="text/csv",
                key="download_xml_integration_csv",
            )
            st.caption(
                f"Integrated XML saved to `{saved_xml_paths['data_directory']}`; "
                f"reports saved to `{saved_xml_paths['report_directory']}`."
            )

st.header("Stage 1: API Data Ingestion")
api_provider = st.selectbox(
    "API provider",
    ["World Bank Data360"],
    key="api_provider",
)
api_specification_url = st.text_input(
    "API specification URL",
    value=API_SPECIFICATION_URL,
    key="api_specification_url",
)
api_database_id = st.text_input(
    "Database ID",
    value="WB_WDI",
    key="api_database_id",
)
api_indicator_id = st.text_input(
    "Indicator ID",
    value="WB_WDI_SP_POP_TOTL",
    key="api_indicator_id",
)
api_reference_area = st.text_input(
    "Country / reference area code",
    value="IND",
    key="api_reference_area",
)
api_year_columns = st.columns(2)
api_start_year = api_year_columns[0].text_input(
    "Start year",
    key="api_start_year",
)
api_end_year = api_year_columns[1].text_input(
    "End year",
    key="api_end_year",
)
api_max_records = st.number_input(
    "Maximum records",
    min_value=1,
    max_value=1_000_000,
    value=10_000,
    step=1_000,
    key="api_max_records",
)
api_skip = st.number_input(
    "Starting record offset (skip)",
    min_value=0,
    value=0,
    step=1000,
    key="api_skip",
)
with st.expander("Additional Data360 filters"):
    api_advanced_columns = st.columns(3)
    api_sex = api_advanced_columns[0].text_input("SEX", key="api_sex")
    api_age = api_advanced_columns[1].text_input("AGE", key="api_age")
    api_urbanisation = api_advanced_columns[2].text_input(
        "URBANISATION", key="api_urbanisation"
    )
    api_frequency = api_advanced_columns[0].text_input("FREQ", key="api_freq")
    api_unit_measure = api_advanced_columns[1].text_input(
        "UNIT_MEASURE", key="api_unit_measure"
    )
    api_time_period = api_advanced_columns[2].text_input(
        "TIME_PERIOD", key="api_time_period"
    )

if st.button(
        "Fetch API Data",
        disabled=not (api_database_id or "").strip()
                 or not (api_specification_url or "").strip(),
        key="fetch_api_data_button",
):
    api_parameters = {
        "DATABASE_ID": api_database_id,
        "INDICATOR": api_indicator_id,
        "REF_AREA": api_reference_area,
        "timePeriodFrom": api_start_year,
        "timePeriodTo": api_end_year,
        "SEX": api_sex,
        "AGE": api_age,
        "URBANISATION": api_urbanisation,
        "FREQ": api_frequency,
        "UNIT_MEASURE": api_unit_measure,
        "TIME_PERIOD": api_time_period,
        "skip": int(api_skip or 0),
        "format": "json",
    }
    st.session_state["api_result"] = extract_api_data(
        api_parameters,
        specification_url=(api_specification_url or "").strip(),
        max_records=int(api_max_records or 1),
    )

api_result = st.session_state.get("api_result")
if api_result is not None:
    api_metadata = api_result["metadata"]
    if api_metadata["status"] == "success":
        st.success(
            f"Retrieved {api_metadata['records_retrieved']} observation records "
            f"from {api_provider}."
        )
        st.metric("Retrieved records", api_metadata["records_retrieved"])
        st.write(
            "Pagination: "
            f"{api_metadata['pagination'].get('pages_fetched', 0)} page(s), "
            f"{api_metadata['pagination'].get('records_retrieved', 0)} record(s); "
            f"complete={api_metadata['pagination'].get('complete', False)}."
        )
        st.json(api_metadata)
        if api_result["data"] is not None:
            st.dataframe(api_result["data"], width="stretch")
        with st.expander("Original API response"):
            st.json(api_result["api_response"])
        api_datasets = st.session_state.setdefault("api_datasets", {})
        api_dataset_id = (
            f"{api_metadata.get('database_id') or 'api'}::"
            f"{api_metadata.get('indicator_id') or 'all'}::"
            f"{api_metadata['request_timestamp']}"
        )
        api_datasets[api_dataset_id] = api_result
    else:
        st.error(api_metadata.get("error", "API ingestion failed."))
        st.json(api_metadata)

st.header("Stage 2: API Data Profiling")
api_datasets = st.session_state.get("api_datasets", {})
if api_datasets:
    selected_api_dataset_id = st.selectbox(
        "Select an ingested API dataset",
        list(api_datasets),
        format_func=lambda dataset_id: (
            f"{api_datasets[dataset_id]['metadata'].get('database_id') or 'API'}"
            f" — "
            f"{api_datasets[dataset_id]['metadata'].get('indicator_id') or 'all indicators'}"
            f" ({api_datasets[dataset_id]['metadata'].get('records_retrieved', 0)} records)"
        ),
        key="api_profile_selection",
    )
    selected_api_dataset = api_datasets[selected_api_dataset_id]
    if st.button("Profile API Data", key="profile_api_button"):
        api_profiles = st.session_state.setdefault("api_profile_results", {})
        api_profiles[selected_api_dataset_id] = profile_api(
            selected_api_dataset["data"],
            api_metadata=selected_api_dataset["metadata"],
            api_response=selected_api_dataset["api_response"],
        )

    api_profile = st.session_state.get("api_profile_results", {}).get(
        selected_api_dataset_id
    )
    if api_profile is not None:
        api_profile_dimensions = api_profile["dimensions"]
        api_profile_metrics = st.columns(4)
        api_profile_metrics[0].metric("Rows", api_profile_dimensions["total_rows"])
        api_profile_metrics[1].metric(
            "Columns", api_profile_dimensions["total_columns"]
        )
        api_profile_metrics[2].metric(
            "Missing values",
            api_profile["missing_values"]["total_missing_values"],
        )
        api_profile_metrics[3].metric(
            "Duplicate rows",
            api_profile["duplicates"]["total_duplicate_rows"],
        )

        api_missing_rows = [
            {"Column": column, **details}
            for column, details in api_profile["missing_values"]["columns"].items()
        ]
        st.subheader("API missing values by column")
        if api_missing_rows:
            st.dataframe(api_missing_rows, width="stretch", hide_index=True)
            st.bar_chart(
                {
                    row["Column"]: row["missing_count"]
                    for row in api_missing_rows
                }
            )
        else:
            st.info("This API DataFrame has no columns.")

        st.subheader("API column data types")
        st.dataframe(
            api_profile["data_types"],
            width="stretch",
            hide_index=True,
        )
        st.subheader("Duplicate observations")
        st.write(
            f"Unique rows: {api_profile['duplicates']['unique_rows']} "
            f"({api_profile['duplicates']['duplicate_percentage']:.2f}% duplicates)"
        )
        st.dataframe(
            api_profile["duplicates"]["duplicate_records"],
            width="stretch",
        )
        st.subheader("Complete API data quality report")
        api_report = api_profile["quality_summary"]
        st.json({key: value for key, value in api_report.items() if key != "columns"})
        st.dataframe(api_report["columns"], width="stretch", hide_index=True)
        st.subheader("API request metadata")
        st.json(api_profile["api_metadata"])
        st.session_state["stage3_api_datasets"] = {
            key: value
            for key, value in st.session_state.get(
                "api_profile_results", {}
            ).items()
        }
        st.caption(
            "Stage 3 handoff: the unchanged DataFrame, API metadata, and original "
            "response are available at "
            "`st.session_state['stage3_api_datasets'][dataset_id]`."
        )
else:
    st.info("Retrieve API data in Stage 1 before profiling it.")

st.header("Stage 3: API Data Cleaning")
if api_datasets:
    selected_api_clean_id = st.selectbox(
        "Select an ingested API dataset to clean",
        list(api_datasets),
        format_func=lambda dataset_id: (
            f"{api_datasets[dataset_id]['metadata'].get('database_id') or 'API'}"
            f" — "
            f"{api_datasets[dataset_id]['metadata'].get('indicator_id') or 'all indicators'}"
            f" — "
            f"{api_datasets[dataset_id]['metadata'].get('country_reference_area') or 'all areas'}"
            f" ({api_datasets[dataset_id]['metadata'].get('records_retrieved', 0)} records)"
        ),
        key="api_clean_selection",
    )
    selected_api_clean_dataset = api_datasets[selected_api_clean_id]
    original_api_df = selected_api_clean_dataset["data"]
    original_api_metadata = selected_api_clean_dataset["metadata"]
    api_cleaning_key = selected_api_clean_id

    st.subheader("API request metadata")
    st.json(original_api_metadata)
    st.subheader("Original API observations")
    st.dataframe(original_api_df, width="stretch")

    st.subheader("Stage 2 API profiling results")
    selected_api_profile = st.session_state.get("api_profile_results", {}).get(
        selected_api_clean_id
    )
    if selected_api_profile is None:
        st.info("Run Stage 2 API profiling for this dataset to see its profile here.")
    else:
        api_stage2_metrics = st.columns(4)
        api_stage2_metrics[0].metric(
            "Profiled rows",
            selected_api_profile["dimensions"]["total_rows"],
        )
        api_stage2_metrics[1].metric(
            "Profiled columns",
            selected_api_profile["dimensions"]["total_columns"],
        )
        api_stage2_metrics[2].metric(
            "Missing values",
            selected_api_profile["missing_values"]["total_missing_values"],
        )
        api_stage2_metrics[3].metric(
            "Duplicate rows",
            selected_api_profile["duplicates"]["total_duplicate_rows"],
        )
        with st.expander("Full Stage 2 API quality report"):
            stage2_report = selected_api_profile["quality_summary"]
            st.json(
                {key: value for key, value in stage2_report.items() if key != "columns"}
            )
            st.dataframe(
                stage2_report["columns"],
                width="stretch",
                hide_index=True,
            )

    api_clean_flags = st.columns(3)
    api_clean_duplicates = api_clean_flags[0].checkbox(
        "Remove duplicate observations", key="api_clean_duplicates"
    )
    api_clean_missing = api_clean_flags[1].checkbox(
        "Handle missing values", key="api_clean_missing"
    )
    api_clean_names = api_clean_flags[2].checkbox(
        "Standardise column names", key="api_clean_names"
    )
    api_clean_dates = api_clean_flags[0].checkbox(
        "Standardise date columns", key="api_clean_dates"
    )
    api_clean_currency = api_clean_flags[1].checkbox(
        "Standardise currency values", key="api_clean_currency"
    )
    api_clean_types = api_clean_flags[2].checkbox(
        "Correct selected data types", key="api_clean_types"
    )

    api_columns = [str(column) for column in original_api_df.columns]
    duplicate_keys = None
    if api_clean_duplicates:
        duplicate_key_selection = st.multiselect(
            "Optional duplicate comparison columns (empty means full row)",
            api_columns,
            key="api_duplicate_keys",
        )
        duplicate_keys = duplicate_key_selection or None

    missing_strategy = "leave"
    missing_column_strategies = {}
    missing_fill_values = {}
    missing_fill = None
    if api_clean_missing:
        missing_strategy = st.selectbox(
            "Default missing-value strategy",
            ["leave", "drop_rows", "mean", "median", "mode", "fill_value"],
            format_func=lambda value: {
                "leave": "Leave unchanged",
                "drop_rows": "Remove rows containing missing values",
                "mean": "Fill numeric fields with mean",
                "median": "Fill numeric fields with median",
                "mode": "Fill categorical fields with mode",
                "fill_value": "Fill with a specified value",
            }[value],
            key="api_missing_strategy",
        )
        if missing_strategy == "fill_value":
            missing_fill = st.text_input(
                "Default missing-value fill", key="api_missing_fill_value"
            )
        override_columns = st.multiselect(
            "Columns with a different missing-value strategy",
            api_columns,
            key="api_missing_override_columns",
        )
        for column in override_columns:
            selected_strategy = st.selectbox(
                f"Missing-value strategy for {column}",
                ["leave", "drop_rows", "mean", "median", "mode", "fill_value"],
                key=f"api_missing_strategy_{column}",
            )
            missing_column_strategies[column] = selected_strategy
            if selected_strategy == "fill_value":
                missing_fill_values[column] = st.text_input(
                    f"Fill value for {column}",
                    key=f"api_missing_value_{column}",
                )

    selected_date_columns = []
    date_format = None
    if api_clean_dates:
        date_defaults = identify_api_date_columns(original_api_df)
        selected_date_columns = st.multiselect(
            "Date/time-period columns",
            api_columns,
            default=date_defaults,
            key="api_date_columns",
        )
        date_format = st.text_input(
            "Optional explicit date format (for example %Y-%m-%d)",
            key="api_date_format",
        ) or None

    selected_currency_columns = []
    currency_code = None
    if api_clean_currency:
        selected_currency_columns = st.multiselect(
            "Currency columns (choose explicitly when units are not monetary)",
            api_columns,
            default=identify_api_currency_columns(
                original_api_df, original_api_metadata
            ),
            key="api_currency_columns",
        )
        currency_choice = st.selectbox(
            "Currency code for values without a symbol",
            ["Auto-detect", "USD", "EUR", "GBP", "INR"],
            key="api_currency_code",
        )
        currency_code = None if currency_choice == "Auto-detect" else currency_choice

    api_type_conversions = {}
    if api_clean_types:
        api_type_columns = st.multiselect(
            "Columns to convert",
            api_columns,
            key="api_type_columns",
        )
        for column in api_type_columns:
            api_type_conversions[column] = st.selectbox(
                f"Target type for {column}",
                ["integer", "float", "string", "boolean", "datetime"],
                key=f"api_target_type_{column}",
            )

    if st.button("Run API Cleaning", key="run_api_cleaning"):
        api_clean_result = clean_api_dataframe(
            original_api_df,
            api_metadata=original_api_metadata,
            dataset_name=str(original_api_metadata.get("database_id") or "api_dataset"),
            remove_duplicates=api_clean_duplicates,
            duplicate_subset=duplicate_keys,
            handle_missing=api_clean_missing,
            missing_strategy=missing_strategy,
            column_missing_strategies=missing_column_strategies,
            missing_fill_value=missing_fill,
            column_fill_values=missing_fill_values,
            standardise_names=api_clean_names,
            standardise_dates=api_clean_dates,
            date_columns=selected_date_columns if api_clean_dates else None,
            date_format=date_format,
            standardise_currencies=api_clean_currency,
            currency_columns=(
                selected_currency_columns if api_clean_currency else None
            ),
            currency_code=currency_code,
            correct_data_types=api_clean_types,
            type_conversions=api_type_conversions,
            api_response=selected_api_clean_dataset.get("api_response"),
        )
        saved_paths = save_cleaned_api_dataset(api_clean_result)
        api_clean_result["saved_paths"] = saved_paths
        api_clean_result["report"]["processed_output_directory"] = saved_paths[
            "directory"
        ]
        st.session_state.setdefault("api_clean_results", {})[
            api_cleaning_key
        ] = api_clean_result

    api_clean_result = st.session_state.get("api_clean_results", {}).get(
        api_cleaning_key
    )
    if api_clean_result is not None:
        cleaned_api_df = api_clean_result["data"]
        api_clean_report = api_clean_result["report"]
        st.subheader("Cleaned API observations")
        st.dataframe(cleaned_api_df, width="stretch")
        api_before_after = st.columns(4)
        api_before_after[0].metric(
            "Rows before / after",
            f"{api_clean_report['original_row_count']} / "
            f"{api_clean_report['final_row_count']}",
        )
        api_before_after[1].metric(
            "Columns before / after",
            f"{api_clean_report['original_column_count']} / "
            f"{api_clean_report['final_column_count']}",
        )
        api_before_after[2].metric(
            "Duplicate records removed",
            api_clean_report["duplicate_records_removed"],
        )
        api_before_after[3].metric(
            "Missing values after",
            sum(api_clean_report["missing_values_after"].values()),
        )
        st.subheader("API cleaning report")
        st.json(api_clean_report)
        st.download_button(
            "Download cleaned API data as CSV",
            data=Path(api_clean_result["saved_paths"]["csv"]).read_bytes(),
            file_name="cleaned_api_data.csv",
            mime="text/csv",
            key="download_clean_api_csv",
        )
        st.download_button(
            "Download cleaned API data as JSON",
            data=Path(api_clean_result["saved_paths"]["json"]).read_bytes(),
            file_name="cleaned_api_data.json",
            mime="application/json",
            key="download_clean_api_json",
        )
        st.session_state["stage3_api_clean_datasets"] = {
            key: value
            for key, value in st.session_state.get("api_clean_results", {}).items()
        }
        st.caption(
            "Stage 3 output is available at "
            "`st.session_state['stage3_api_clean_datasets'][dataset_id]`; the "
            "Stage 1 API response and metadata remain unchanged."
        )
else:
    st.info("Retrieve API data in Stage 1 before cleaning it.")

st.header("Stage 4: API Data Validation")
stage3_api_results = st.session_state.get("api_clean_results", {})
if stage3_api_results:
    api_validation_ids = list(stage3_api_results)
    api_validation_id = st.selectbox(
        "Select a cleaned API dataset to validate",
        api_validation_ids,
        format_func=lambda dataset_id: (
            f"{stage3_api_results[dataset_id].get('api_metadata', {}).get('database_id') or 'API'}"
            f" — {stage3_api_results[dataset_id].get('api_metadata', {}).get('indicator_id') or 'all indicators'}"
            f" — {stage3_api_results[dataset_id].get('api_metadata', {}).get('country_reference_area') or 'all areas'}"
        ),
        key="api_validation_selection",
    )
    selected_api_validation = stage3_api_results[api_validation_id]
    validation_df = selected_api_validation.get("data")
    validation_metadata = selected_api_validation.get("api_metadata", {})
    if isinstance(validation_df, pd.DataFrame):
        validation_dataset_name = str(
            selected_api_validation.get("report", {}).get("dataset_name")
            or api_validation_id
        )
        st.subheader("API metadata")
        st.json(validation_metadata)
        st.subheader("Cleaned API observations")
        st.dataframe(validation_df, width="stretch")
        selected_profile = st.session_state.get("api_profile_results", {}).get(
            api_validation_id
        )
        if selected_profile is not None:
            with st.expander("Stage 2 API profiling results"):
                st.json(
                    {
                        key: value
                        for key, value in selected_profile["quality_summary"].items()
                        if key != "columns"
                    }
                )
                st.dataframe(
                    selected_profile["quality_summary"]["columns"],
                    width="stretch",
                    hide_index=True,
                )

        validation_columns = [str(column) for column in validation_df.columns]
        required_column_selection = st.multiselect(
            "Required API columns (schema presence)",
            validation_columns,
            key="api_validation_required_columns",
        )
        required_field_selection = st.multiselect(
            "Required API fields (non-null, non-empty values)",
            validation_columns,
            key="api_validation_required_fields",
        )
        warning_field_selection = st.multiselect(
            "Optional fields to report as warnings",
            validation_columns,
            key="api_validation_warning_fields",
        )

        numerical_columns = st.multiselect(
            "Numeric observation columns to validate",
            validation_columns,
            key="api_validation_numeric_columns",
        )
        numerical_rules = {}
        for column in numerical_columns:
            numeric_controls = st.columns(5)
            numeric_type = numeric_controls[0].selectbox(
                f"Numeric type: {column}",
                ["decimal", "integer"],
                key=f"api_validation_numeric_type_{column}",
            )
            minimum = numeric_controls[1].text_input(
                "Minimum",
                key=f"api_validation_min_{column}",
            )
            maximum = numeric_controls[2].text_input(
                "Maximum",
                key=f"api_validation_max_{column}",
            )
            allow_negative = numeric_controls[3].checkbox(
                "Allow negative",
                value=True,
                key=f"api_validation_negative_{column}",
            )
            numeric_severity = numeric_controls[4].selectbox(
                "Severity",
                ["error", "warning"],
                key=f"api_validation_numeric_severity_{column}",
            )
            numeric_rule = {
                "type": numeric_type,
                "allow_negative": allow_negative,
                "severity": numeric_severity,
            }
            if minimum.strip():
                numeric_rule["min"] = minimum
            if maximum.strip():
                numeric_rule["max"] = maximum
            numerical_rules[column] = numeric_rule

        date_columns = st.multiselect(
            "Date or time-period columns",
            validation_columns,
            key="api_validation_date_columns",
        )
        date_rules = {}
        for column in date_columns:
            date_controls = st.columns(3)
            date_mode = date_controls[0].selectbox(
                f"Date format: {column}",
                ["Auto (year/year-month/ISO)", "Year (%Y)", "Year-month (%Y-%m)", "ISO date (%Y-%m-%d)", "Custom"],
                key=f"api_validation_date_mode_{column}",
            )
            custom_date_format = ""
            if date_mode == "Custom":
                custom_date_format = date_controls[1].text_input(
                    "Python date format",
                    key=f"api_validation_date_custom_{column}",
                ) or ""
            required_date = date_controls[2].checkbox(
                "Required",
                key=f"api_validation_date_required_{column}",
            )
            expected_formats = {
                "Auto (year/year-month/ISO)": None,
                "Year (%Y)": "%Y",
                "Year-month (%Y-%m)": "%Y-%m",
                "ISO date (%Y-%m-%d)": "%Y-%m-%d",
                "Custom": custom_date_format or None,
            }[date_mode]
            date_rules[column] = {
                "format": expected_formats,
                "required": required_date,
            }

        validation_switches = st.columns(3)
        run_required_checks = validation_switches[0].checkbox(
            "Validate required fields",
            value=True,
            key="api_validate_required_enabled",
        )
        run_numeric_checks = validation_switches[1].checkbox(
            "Validate numerical values",
            value=True,
            key="api_validate_numeric_enabled",
        )
        run_date_checks = validation_switches[2].checkbox(
            "Validate dates",
            value=True,
            key="api_validate_dates_enabled",
        )
        if st.button("Run API Validation", key="run_api_validation"):
            result = validate_api_dataframe(
                selected_api_validation,
                dataset_name=validation_dataset_name,
                required_columns=required_column_selection,
                required_fields=required_field_selection,
                warning_fields=warning_field_selection,
                numerical_rules=numerical_rules,
                date_rules=date_rules,
                validate_required=run_required_checks,
                validate_numerical=run_numeric_checks,
                validate_dates=run_date_checks,
            )
            report_paths = export_api_validation_report(result)
            data_paths = export_api_validation_data(
                result, Path(report_paths["directory"])
            )
            result["saved_paths"] = {**report_paths, **data_paths}
            st.session_state.setdefault("api_validation_results", {})[
                api_validation_id
            ] = result

        validation_result = st.session_state.get(
            "api_validation_results", {}
        ).get(api_validation_id)
        if validation_result is not None:
            validation_report = validation_result["validation_report"]
            status_metrics = st.columns(4)
            status_metrics[0].metric(
                "Validation status", validation_report["validation_status"]
            )
            status_metrics[1].metric(
                "Valid records", validation_report["valid_records"]
            )
            status_metrics[2].metric(
                "Invalid records", validation_report["invalid_records"]
            )
            status_metrics[3].metric(
                "Errors / warnings",
                f"{validation_report['total_validation_errors']} / "
                f"{validation_report['total_validation_warnings']}",
            )
            st.subheader("API validation errors")
            st.dataframe(
                validation_result["validation_errors"],
                width="stretch",
                hide_index=True,
            )
            st.subheader("API validation warnings")
            st.dataframe(
                validation_result["validation_warnings"],
                width="stretch",
                hide_index=True,
            )
            st.subheader("API column validation summary")
            st.dataframe(
                validation_report["column_summary"],
                width="stretch",
                hide_index=True,
            )
            with st.expander("Complete API validation report"):
                st.json(validation_report)
            st.subheader("Valid API observations")
            st.dataframe(validation_result["valid_dataframe"], width="stretch")
            st.subheader("Invalid API observations")
            st.dataframe(validation_result["invalid_dataframe"], width="stretch")
            saved_api_validation = validation_result["saved_paths"]
            st.download_button(
                "Download valid API data",
                Path(saved_api_validation["valid_csv"]).read_bytes(),
                file_name=f"{validation_dataset_name}_valid.csv",
                mime="text/csv",
                key="download_api_valid_data",
            )
            st.download_button(
                "Download invalid API data",
                Path(saved_api_validation["invalid_csv"]).read_bytes(),
                file_name=f"{validation_dataset_name}_invalid.csv",
                mime="text/csv",
                key="download_api_invalid_data",
            )
            st.download_button(
                "Download API validation report (JSON)",
                Path(saved_api_validation["json"]).read_bytes(),
                file_name="api_validation_report.json",
                mime="application/json",
                key="download_api_validation_json",
            )
            st.download_button(
                "Download API validation errors (CSV)",
                Path(saved_api_validation["csv"]).read_bytes(),
                file_name="api_validation_errors.csv",
                mime="text/csv",
                key="download_api_validation_csv",
            )
            st.caption(
                f"API validation reports saved to `{saved_api_validation['directory']}`."
            )
    else:
        st.error("The selected Stage 3 API result does not contain a DataFrame.")
else:
    st.info("Run Stage 3 API cleaning before validating an API dataset.")

st.header("Stage 5: API Data Integration")
api_clean_results_for_integration = st.session_state.get("api_clean_results", {})
api_validation_results_for_integration = st.session_state.get(
    "api_validation_results", {}
)
validated_api_datasets = {}
api_integration_labels = {}
for dataset_id, validation_result in api_validation_results_for_integration.items():
    if validation_result.get("validation_status") not in {
        "passed",
        "passed_with_warnings",
    } or not isinstance(validation_result.get("valid_dataframe"), pd.DataFrame):
        continue
    clean_result = api_clean_results_for_integration.get(dataset_id, {})
    api_metadata = (
        validation_result.get("api_metadata")
        or clean_result.get("api_metadata")
        or {}
    )
    dataset_name = str(
        clean_result.get("report", {}).get("dataset_name")
        or dataset_id
    )
    validated_api_datasets[dataset_id] = {
        "validation_result": validation_result,
        "api_metadata": api_metadata,
        "report": clean_result.get("report", {}),
        "source": dataset_name,
    }
    api_integration_labels[dataset_id] = (
        f"{api_metadata.get('database_id') or api_metadata.get('provider') or 'API'}"
        f" — {api_metadata.get('indicator_id') or 'all indicators'}"
        f" — {api_metadata.get('country_reference_area') or 'all areas'}"
        f" ({len(validation_result['valid_dataframe'])} valid observations)"
    )

if not validated_api_datasets:
    st.info("Run Stage 4 API validation successfully before integrating datasets.")
else:
    selected_api_integration_ids = st.multiselect(
        "Select cleaned and validated API datasets",
        list(validated_api_datasets),
        format_func=api_integration_labels.__getitem__,
        key="api_integration_datasets",
    )
    if selected_api_integration_ids:
        selected_api_integration_datasets = {
            dataset_id: validated_api_datasets[dataset_id]
            for dataset_id in selected_api_integration_ids
        }
        for dataset_id, dataset in selected_api_integration_datasets.items():
            with st.expander(f"Dataset: {api_integration_labels[dataset_id]}"):
                st.subheader("API source metadata")
                st.json(dataset["api_metadata"])
                st.dataframe(
                    dataset["validation_result"]["valid_dataframe"],
                    width="stretch",
                )

        schema_preview = None
        if len(selected_api_integration_ids) >= 2:
            try:
                schema_preview = handle_api_inconsistent_schemas(
                    selected_api_integration_datasets
                )
                st.subheader("API schema comparison")
                st.dataframe(
                    schema_preview["schema_report"],
                    width="stretch",
                    hide_index=True,
                )
                matching_preview = match_api_common_columns(
                    {
                        selected_api_integration_ids[0]:
                            selected_api_integration_datasets[selected_api_integration_ids[0]],
                        selected_api_integration_ids[1]:
                            selected_api_integration_datasets[selected_api_integration_ids[1]],
                    }
                )
                st.subheader("Common, unmatched, and possible matching columns")
                st.json(
                    {
                        "common_columns": matching_preview["common_columns"],
                        "possible_matching_columns": matching_preview[
                            "possible_matching_columns"
                        ],
                        "unmatched_left_columns": matching_preview[
                            "unmatched_left_columns"
                        ],
                        "unmatched_right_columns": matching_preview[
                            "unmatched_right_columns"
                        ],
                        "incompatible_common_columns": matching_preview[
                            "incompatible_common_columns"
                        ],
                    }
                )
            except (TypeError, ValueError, KeyError) as error:
                st.warning(f"Could not compare the selected API schemas: {error}")

        api_integration_type = st.selectbox(
            "API integration operation",
            ["combine", "merge"],
            key="api_integration_type",
        )
        api_join_type = "inner"
        api_left_keys = []
        api_right_keys = []
        api_column_mappings = {}
        api_combine_mapping_ready = True
        if api_integration_type == "combine" and len(selected_api_integration_ids) >= 2:
            anchor_id = selected_api_integration_ids[0]
            anchor_columns = list(
                map(
                    str,
                    selected_api_integration_datasets[anchor_id]["validation_result"][
                        "valid_dataframe"
                    ].columns,
                )
            )
            with st.expander("Optional manual column mapping for row-wise combine"):
                st.caption(
                    "Map differently named source columns to columns in the first "
                    "selected dataset. Unmapped columns remain separate."
                )
                for dataset_id in selected_api_integration_ids[1:]:
                    source_columns = list(
                        map(
                            str,
                            selected_api_integration_datasets[dataset_id][
                                "validation_result"
                            ]["valid_dataframe"].columns,
                        )
                    )
                    mapping_controls = st.columns(2)
                    source_choices = mapping_controls[0].multiselect(
                        f"Source columns: {api_integration_labels[dataset_id]}",
                        source_columns,
                        key=f"api_combine_source_columns_{dataset_id}",
                    )
                    target_choices = mapping_controls[1].multiselect(
                        f"Map to first dataset columns: {api_integration_labels[dataset_id]}",
                        anchor_columns,
                        key=f"api_combine_target_columns_{dataset_id}",
                    )
                    if len(source_choices) != len(target_choices):
                        api_combine_mapping_ready = False
                        st.warning(
                            "Select the same number of source and target columns; "
                            "their selected order defines each mapping."
                        )
                    elif source_choices:
                        api_column_mappings[dataset_id] = dict(
                            zip(source_choices, target_choices)
                        )
        if api_integration_type == "merge":
            if len(selected_api_integration_ids) != 2:
                st.info("Select exactly two API datasets to configure a merge.")
            else:
                left_id, right_id = selected_api_integration_ids
                left_frame = selected_api_integration_datasets[left_id][
                    "validation_result"
                ]["valid_dataframe"]
                right_frame = selected_api_integration_datasets[right_id][
                    "validation_result"
                ]["valid_dataframe"]
                common_keys = [
                    str(column)
                    for column in left_frame.columns
                    if str(column) in set(map(str, right_frame.columns))
                ]
                left_fields = list(map(str, left_frame.columns))
                right_fields = list(map(str, right_frame.columns))
                key_controls = st.columns(2)
                api_left_keys = key_controls[0].multiselect(
                    "Left dataset merge key columns",
                    left_fields,
                    default=common_keys,
                    key="api_merge_left_keys",
                )
                api_right_keys = key_controls[1].multiselect(
                    "Right dataset merge key columns (matching order)",
                    right_fields,
                    default=common_keys,
                    key="api_merge_right_keys",
                )
                api_join_type = st.selectbox(
                    "API merge type",
                    ["inner", "left", "right", "outer"],
                    key="api_merge_type",
                )
                st.caption(
                    "Choose the same number of left and right keys in corresponding "
                    "order. Common World Bank observation identifiers include "
                    "INDICATOR, REF_AREA, and TIME_PERIOD. Many-to-many joins are "
                    "rejected unless explicitly enabled."
                )

        api_fill_values = {}
        api_fill_columns = []
        api_default_value = ""
        if len(selected_api_integration_ids) >= 2:
            fill_with_default = st.checkbox(
                "Fill missing schema values with a configured default",
                key="api_integration_fill_enabled",
            )
            if fill_with_default:
                all_api_columns = list(
                    dict.fromkeys(
                        str(column)
                        for dataset in selected_api_integration_datasets.values()
                        for column in dataset["validation_result"][
                            "valid_dataframe"
                        ].columns
                    )
                )
                fill_controls = st.columns(2)
                api_fill_columns = fill_controls[0].multiselect(
                    "Columns to fill",
                    all_api_columns,
                    key="api_integration_fill_columns",
                )
                api_default_value = fill_controls[1].text_input(
                    "Default value (stored as text)",
                    key="api_integration_fill_value",
                )
                api_fill_values = {
                    column: api_default_value for column in api_fill_columns
                }

        allow_mixed_indicators = False
        if len(selected_api_integration_ids) >= 2:
            indicator_ids = {
                str(
                    dataset["api_metadata"].get("indicator_id")
                    or dataset["api_metadata"].get("INDICATOR")
                )
                for dataset in selected_api_integration_datasets.values()
                if dataset["api_metadata"].get("indicator_id")
                or dataset["api_metadata"].get("INDICATOR")
            }
            if len(indicator_ids) > 1:
                allow_mixed_indicators = st.checkbox(
                    "Explicitly allow integrating different indicators",
                    value=False,
                    key="api_integration_mixed_indicators",
                )

        api_merge_ready = (
            api_integration_type != "merge"
            or (
                len(selected_api_integration_ids) == 2
                and bool(api_left_keys)
                and len(api_left_keys) == len(api_right_keys)
            )
        )
        if st.button(
            "Run API Integration",
            key="run_api_integration",
            disabled=(
                not selected_api_integration_ids
                or not api_merge_ready
                or not api_combine_mapping_ready
            ),
        ):
            try:
                mapping = (
                    dict(zip(api_left_keys, api_right_keys))
                    if api_integration_type == "merge"
                    else {}
                )
                result = integrate_multiple_api_datasets(
                    selected_api_integration_datasets,
                    integration_type=api_integration_type,
                    join_type=api_join_type,
                    key_columns=api_left_keys or None,
                    right_key_columns=api_right_keys or None,
                    column_mapping=mapping,
                    column_mappings=api_column_mappings,
                    fill_values=api_fill_values,
                    allow_mixed_indicators=allow_mixed_indicators,
                )
                st.session_state["api_integration_result"] = result
            except (OSError, TypeError, ValueError, KeyError) as error:
                st.error(f"API integration failed: {error}")

    api_integration_result = st.session_state.get("api_integration_result")
    if api_integration_result is not None:
        st.metric(
            "API integration status",
            api_integration_result.get("integration_status", "unknown"),
        )
        integration_report = api_integration_result["integration_report"]
        report_columns = st.columns(4)
        report_columns[0].metric(
            "Integrated observations",
            integration_report.get("output_record_count", integration_report.get("output_records", 0)),
        )
        report_columns[1].metric(
            "Matched observations",
            integration_report.get("matched_records", 0),
        )
        report_columns[2].metric(
            "Unmatched left",
            integration_report.get("unmatched_left_records", 0),
        )
        report_columns[3].metric(
            "Unmatched right",
            integration_report.get("unmatched_right_records", 0),
        )
        st.subheader("Integrated API observations")
        st.dataframe(api_integration_result["data"], width="stretch")
        st.subheader("API source metadata and integration report")
        st.json(
            {
                "integration_report": integration_report,
                "source_datasets": api_integration_result["source_datasets"],
                "schema_report": api_integration_result["schema_report"],
            }
        )
        if "saved_paths" in api_integration_result:
            saved_api_integration = api_integration_result["saved_paths"]
            st.download_button(
                "Download integrated API data (CSV)",
                Path(saved_api_integration["data_csv"]).read_bytes(),
                file_name=Path(saved_api_integration["data_csv"]).name,
                mime="text/csv",
                key="download_integrated_api_csv",
            )
            st.download_button(
                "Download integrated API data (JSON)",
                Path(saved_api_integration["data_json"]).read_bytes(),
                file_name=Path(saved_api_integration["data_json"]).name,
                mime="application/json",
                key="download_integrated_api_json",
            )
            st.download_button(
                "Download API metadata sidecar",
                Path(saved_api_integration["metadata_json"]).read_bytes(),
                file_name=Path(saved_api_integration["metadata_json"]).name,
                mime="application/json",
                key="download_api_integration_metadata",
            )
            st.download_button(
                "Download API integration report (JSON)",
                Path(saved_api_integration["report_json"]).read_bytes(),
                file_name=Path(saved_api_integration["report_json"]).name,
                mime="application/json",
                key="download_api_integration_report",
            )
            st.caption(
                f"Integrated data saved to `{saved_api_integration['data_directory']}`; "
                f"reports saved to `{saved_api_integration['report_directory']}`."
            )

st.header("Stage 6: CSV Database Loading")
csv_loading_candidates = {}
for dataset_key, validation_result in st.session_state.get(
    "csv_validation_results", {}
).items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and isinstance(validation_result.get("valid_dataframe"), pd.DataFrame)
    ):
        report = validation_result.get("validation_report", {})
        dataset_name = str(
            report.get("dataset_name")
            or str(dataset_key).split("::", maxsplit=1)[0]
        )
        csv_loading_candidates[dataset_key] = {
            "validation_result": validation_result,
            "source": str(dataset_key).split("::", maxsplit=1)[0],
            "dataset_name": dataset_name,
        }

if not csv_loading_candidates:
    st.info("Run Stage 4 CSV validation successfully before loading to MySQL.")
else:
    loading_defaults = get_connection_config_from_environment()
    st.caption(
        "Credentials are read from MYSQL_HOST, MYSQL_PORT, MYSQL_USER, "
        "MYSQL_PASSWORD, and MYSQL_DATABASE environment variables and can be "
        "overridden for this session. Passwords are masked and never displayed."
    )
    credential_columns = st.columns(2)
    mysql_host = credential_columns[0].text_input(
        "MySQL host",
        value=str(loading_defaults["host"]),
        key="csv_mysql_host",
    )
    mysql_port = credential_columns[1].number_input(
        "MySQL port",
        min_value=1,
        max_value=65535,
        value=int(loading_defaults["port"]),
        step=1,
        key="csv_mysql_port",
    )
    mysql_username = credential_columns[0].text_input(
        "MySQL username",
        value=str(loading_defaults["username"]),
        key="csv_mysql_username",
    )
    mysql_password = credential_columns[1].text_input(
        "MySQL password",
        value=str(loading_defaults["password"]),
        type="password",
        key="csv_mysql_password",
    )
    mysql_database = st.text_input(
        "MySQL database name",
        value=str(loading_defaults["database"]),
        key="csv_mysql_database",
    )
    mysql_config = {
        "host": mysql_host,
        "port": mysql_port,
        "username": mysql_username,
        "password": mysql_password,
        "database": mysql_database,
    }

    connection_status = st.session_state.get("csv_mysql_connection_status")
    if connection_status:
        if connection_status.get("status") == "success":
            st.success(
                f"MySQL connection verified; database "
                f"`{connection_status['database']}` is ready "
                f"({connection_status['database_status']})."
            )
        else:
            st.error(
                f"MySQL connection/database setup failed: "
                f"{connection_status['error']}"
            )

    if st.button("Connect and prepare database", key="prepare_csv_mysql_database"):
        try:
            prepared = prepare_mysql_database(mysql_config)
            prepared["connection"].close()
            st.session_state["csv_mysql_connection_status"] = {
                "status": "success",
                "database": prepared["database"],
                "database_status": prepared["database_status"],
            }
        except Exception as error:
            safe_error = (
                str(error).replace(mysql_password, "[redacted]")
                if mysql_password
                else str(error)
            )
            st.session_state["csv_mysql_connection_status"] = {
                "status": "error",
                "error": safe_error,
            }
        st.rerun()

    dataset_key = st.selectbox(
        "Select a successfully validated CSV dataset",
        list(csv_loading_candidates),
        format_func=lambda key: csv_loading_candidates[key]["dataset_name"],
        key="csv_database_dataset",
    )
    selected_loading_dataset = csv_loading_candidates[dataset_key]
    valid_csv = selected_loading_dataset["validation_result"]["valid_dataframe"]
    st.subheader("Validated records preview")
    st.dataframe(valid_csv.head(100), width="stretch")
    st.caption(f"{len(valid_csv)} validated rows, {len(valid_csv.columns)} columns.")

    loading_options = st.columns(2)
    suggested_table = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        Path(selected_loading_dataset["dataset_name"]).stem,
    ).strip("_")
    if not suggested_table:
        suggested_table = "csv_dataset"
    if suggested_table[0].isdigit():
        suggested_table = f"csv_{suggested_table}"
    suggested_table = suggested_table[:64]
    destination_table = loading_options[0].text_input(
        "Destination table",
        value=suggested_table,
        key=f"csv_database_table_{dataset_key}",
    )
    duplicate_strategy = loading_options[1].selectbox(
        "Duplicate strategy",
        ["skip", "update", "reject"],
        format_func={
            "skip": "Skip duplicates",
            "update": "Update existing records",
            "reject": "Reject duplicates",
        }.__getitem__,
        key="csv_database_duplicate_strategy",
    )
    unique_keys = st.multiselect(
        "Unique key columns (required to prevent accidental duplicate loads)",
        [str(column) for column in valid_csv.columns],
        key=f"csv_database_keys_{dataset_key}",
    )
    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        key="csv_database_batch_size",
    )
    if st.button(
        "Load validated CSV to MySQL",
        key="load_validated_csv_mysql",
        disabled=not unique_keys or not destination_table.strip(),
    ):
        with st.spinner("Loading validated CSV records into MySQL..."):
            load_result = load_validated_csv_dataset(
                selected_loading_dataset,
                connection_config=mysql_config,
                table_name=destination_table.strip(),
                unique_key_columns=unique_keys,
                duplicate_strategy=duplicate_strategy,
                batch_size=int(batch_size),
            )
        st.session_state["csv_database_load_result"] = load_result
        st.rerun()

    load_result = st.session_state.get("csv_database_load_result")
    if load_result is not None:
        if load_result.get("execution_status") == "success":
            st.success("CSV database load completed successfully.")
        else:
            st.error(
                "CSV database load failed: "
                f"{load_result.get('error_message') or 'Unknown database error'}"
            )
        load_metrics = st.columns(5)
        for column, label, key in zip(
            load_metrics,
            ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
            (
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
            ),
        ):
            column.metric(label, load_result.get(key, 0))
        st.json(
            {
                key: value
                for key, value in load_result.items()
                if key not in {"error_message"}
            }
        )

    if st.button("Refresh loading history", key="refresh_csv_loading_history"):
        try:
            st.session_state["csv_database_history"] = get_pipeline_execution_history(
                mysql_config
            )
        except Exception as error:
            safe_error = (
                str(error).replace(mysql_password, "[redacted]")
                if mysql_password
                else str(error)
            )
            st.session_state["csv_database_history_error"] = safe_error
        st.rerun()

    history_error = st.session_state.get("csv_database_history_error")
    if history_error:
        st.warning(f"Could not retrieve loading history: {history_error}")
    history = st.session_state.get("csv_database_history")
    if history:
        st.subheader("Previous CSV loading history")
        st.dataframe(pd.DataFrame(history), width="stretch")
    elif history is not None and not history_error:
        st.info("No loading history has been recorded yet.")

    st.caption(
        "The MySQL account needs server-level CREATE DATABASE permission and "
        "database-level CREATE TABLE, SELECT, INSERT, and UPDATE permissions."
    )

st.header("Stage 7: API Automation and Monitoring")
st.markdown(
    "Run the existing API ETL pipeline manually or configure a persistent daily "
    "or weekly schedule. Scheduled jobs run in a separate process; this page "
    "never starts a scheduler during a Streamlit rerun."
)
st.caption(
    "API request parameters are shared with the Stage 1 controls above. "
    "Scheduled execution reads MySQL and SMTP credentials from environment "
    "variables (or Streamlit secrets for manual runs); credentials are not "
    "written to the schedule file or displayed here."
)

api_automation_request = {
    "DATABASE_ID": api_database_id,
    "INDICATOR": api_indicator_id,
    "REF_AREA": api_reference_area,
    "timePeriodFrom": api_start_year,
    "timePeriodTo": api_end_year,
    "SEX": api_sex,
    "AGE": api_age,
    "URBANISATION": api_urbanisation,
    "FREQ": api_frequency,
    "UNIT_MEASURE": api_unit_measure,
    "TIME_PERIOD": api_time_period,
    "skip": int(api_skip or 0),
    "format": "json",
}
api_automation_extraction_options = {
    "specification_url": (api_specification_url or "").strip(),
    "max_records": int(api_max_records or 1),
}
api_auto_table = st.text_input(
    "API automation destination table",
    value="automated_api",
    key="api_automation_table",
)
api_auto_options_columns = st.columns(3)
api_auto_duplicate_strategy = api_auto_options_columns[0].selectbox(
    "API automation duplicate strategy",
    ["skip", "update", "reject"],
    format_func={
        "skip": "Skip duplicates",
        "update": "Update existing",
        "reject": "Reject duplicates",
    }.__getitem__,
    key="api_automation_duplicate_strategy",
)
api_auto_keys_text = api_auto_options_columns[1].text_input(
    "API observation key columns (comma-separated)",
    value="",
    key="api_automation_unique_keys",
    help="Leave blank to use verified API dimension keys or a stable row fingerprint.",
)
api_auto_retry_attempts = api_auto_options_columns[2].number_input(
    "Temporary failure retry attempts",
    min_value=0,
    max_value=5,
    value=2,
    step=1,
    key="api_automation_retry_attempts",
)
api_auto_notify = st.checkbox(
    "Email on API execution success or failure",
    value=False,
    key="api_automation_notify",
    help="Uses SMTP environment variables or Streamlit secrets; credentials are never shown.",
)
api_auto_unique_keys = [
    item.strip() for item in api_auto_keys_text.split(",") if item.strip()
]

api_auto_db_config = get_api_connection_config_from_environment()
for config_key, state_key in (
    ("host", "api_mysql_host"),
    ("port", "api_mysql_port"),
    ("username", "api_mysql_username"),
    ("password", "api_mysql_password"),
    ("database", "api_mysql_database"),
):
    if state_key in st.session_state:
        api_auto_db_config[config_key] = st.session_state[state_key]

if st.button(
    "Run API ETL now",
    key="run_api_automation_now",
    disabled=not api_auto_table.strip() or not api_automation_extraction_options["specification_url"],
):
    try:
        with st.spinner("Running API ingestion, processing, and database loading..."):
            api_automation_result = run_api_pipeline(
                api_automation_request,
                table_name=api_auto_table.strip(),
                unique_key_columns=api_auto_unique_keys or None,
                connection_config=api_auto_db_config,
                duplicate_strategy=api_auto_duplicate_strategy,
                extraction_options=api_automation_extraction_options,
                retry_attempts=int(api_auto_retry_attempts),
                notify=api_auto_notify,
            )
        st.session_state["api_automation_last_result"] = api_automation_result
    except Exception as error:
        from SRC.automation.api_monitor import sanitize_for_log

        st.session_state["api_automation_ui_error"] = sanitize_for_log(
            str(error),
            secrets=(str(api_auto_db_config.get("password") or ""),),
        )

api_automation_ui_error = st.session_state.get("api_automation_ui_error")
if api_automation_ui_error:
    st.error(f"API ETL execution could not start: {api_automation_ui_error}")
    st.session_state.pop("api_automation_ui_error", None)
api_automation_result = st.session_state.get("api_automation_last_result")
if api_automation_result:
    if api_automation_result.get("execution_status") == "success":
        st.success(
            "API ETL completed successfully. Execution "
            f"`{api_automation_result.get('execution_id')}`."
        )
    elif api_automation_result.get("execution_status") == "skipped":
        st.warning(api_automation_result.get("error_message", "API run was skipped."))
    else:
        st.error(
            f"API ETL {api_automation_result.get('execution_status', 'failed')}: "
            f"{api_automation_result.get('error_message') or 'See failure details.'}"
        )
    api_run_metrics = st.columns(6)
    for column, label, key in zip(
        api_run_metrics,
        ("Retrieved", "Processed", "Inserted", "Updated", "Skipped", "Failed"),
        (
            "records_retrieved",
            "processed_rows",
            "inserted_rows",
            "updated_rows",
            "skipped_rows",
            "failed_rows",
        ),
    ):
        column.metric(label, api_automation_result.get(key, 0))
    stage_records = api_automation_result.get("stage_records") or []
    if stage_records:
        st.subheader("API stage-by-stage progress")
        st.dataframe(pd.DataFrame(stage_records), width="stretch", hide_index=True)
    if api_automation_result.get("http_status_codes"):
        st.write("HTTP response status codes:", api_automation_result["http_status_codes"])
    if api_automation_result.get("pagination"):
        st.write("Pagination details:")
        st.json(api_automation_result["pagination"])
    if api_automation_result.get("failure_details"):
        st.error("Failure details")
        st.json(api_automation_result["failure_details"])

st.subheader("Persistent API schedule")
try:
    api_schedule = get_api_schedule()
except (OSError, ValueError, RuntimeError) as error:
    api_schedule = {"enabled": False}
    st.error(f"Could not read the saved API schedule: {error}")
api_schedule_columns = st.columns(3)
api_schedule_frequency = api_schedule_columns[0].selectbox(
    "API schedule frequency",
    ["daily", "weekly"],
    index=1 if api_schedule.get("frequency") == "weekly" else 0,
    key="api_schedule_frequency",
)
try:
    api_schedule_default_time = datetime.strptime(
        str(api_schedule.get("execution_time", "02:00")), "%H:%M"
    ).time()
except ValueError:
    api_schedule_default_time = datetime.strptime("02:00", "%H:%M").time()
api_schedule_time = api_schedule_columns[1].time_input(
    "API execution time (local)",
    value=api_schedule_default_time,
    key="api_schedule_time",
)
api_schedule_weekday = api_schedule_columns[2].selectbox(
    "API weekly day",
    list(range(7)),
    format_func=lambda day: (
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    )[day],
    index=max(0, min(6, int(api_schedule.get("weekday", 0)))),
    disabled=api_schedule_frequency != "weekly",
    key="api_schedule_weekday",
)
if st.button(
    "Create / update API schedule",
    key="save_api_schedule",
    disabled=not api_auto_table.strip()
    or not api_automation_extraction_options["specification_url"],
):
    try:
        api_schedule = save_api_schedule(
            api_requests=api_automation_request,
            frequency=api_schedule_frequency,
            execution_time=api_schedule_time.strftime("%H:%M"),
            weekday=int(api_schedule_weekday),
            table_name=api_auto_table.strip(),
            unique_key_columns=api_auto_unique_keys,
            enabled=(
                bool(api_schedule.get("enabled"))
                if api_schedule.get("api_requests")
                else True
            ),
            pipeline_options={
                "duplicate_strategy": api_auto_duplicate_strategy,
                "retry_attempts": int(api_auto_retry_attempts),
                "extraction_options": api_automation_extraction_options,
                "notify": api_auto_notify,
            },
        )
        st.success("API schedule saved. The standalone scheduler will use it.")
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not save API schedule: {error}")

api_schedule_toggle_col, api_schedule_next_col = st.columns(2)
if api_schedule.get("api_requests"):
    api_toggle_label = (
        "Disable saved API schedule"
        if api_schedule.get("enabled")
        else "Enable saved API schedule"
    )
    if api_schedule_toggle_col.button(api_toggle_label, key="toggle_api_schedule"):
        try:
            set_api_schedule_enabled(not bool(api_schedule.get("enabled")))
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not update API schedule status: {error}")
    try:
        api_next_run = get_api_next_run_time(api_schedule)
    except (KeyError, TypeError, ValueError) as error:
        api_next_run = None
        st.error(f"Could not calculate the next API execution: {error}")
    api_schedule_next_col.metric(
        "Next scheduled API execution",
        api_next_run.strftime("%Y-%m-%d %H:%M %Z") if api_next_run else "Disabled",
    )
else:
    api_schedule_next_col.metric("Next scheduled API execution", "Not configured")
st.caption(
    "To execute saved schedules independently of Streamlit, configure MySQL "
    "credentials in the scheduler process environment and run "
    "`python -m SRC.automation.api_scheduler`. This app does not start it."
)

st.subheader("API request and execution history")
try:
    api_automation_history = get_api_execution_history(limit=100)
except (OSError, ValueError) as error:
    api_automation_history = []
    st.error(f"Could not load API execution history: {error}")
if api_automation_history:
    api_history_rows = [
        {
            key: row.get(key)
            for key in (
                "execution_id",
                "api_provider",
                "api_endpoint",
                "dataset_name",
                "request_parameters",
                "execution_status",
                "start_time",
                "duration_seconds",
                "http_status_codes",
                "api_response_record_count",
                "records_retrieved",
                "pagination",
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
                "failure_stage",
                "error_message",
            )
        }
        for row in api_automation_history
    ]
    st.dataframe(pd.DataFrame(api_history_rows), width="stretch", hide_index=True)
    selected_api_log_id = st.selectbox(
        "Inspect API execution log",
        [row["execution_id"] for row in api_automation_history],
        format_func=lambda execution_id: next(
            (
                f"{row.get('dataset_name')} — {row.get('execution_status')} — "
                f"{row.get('start_time')}"
                for row in api_automation_history
                if row.get("execution_id") == execution_id
            ),
            execution_id,
        ),
        key="api_automation_log_selection",
    )
    try:
        selected_api_log = get_api_execution_log(selected_api_log_id)
        with st.expander("Selected API request history, pagination, and failure details"):
            st.json(selected_api_log)
        st.download_button(
            "Download API execution log",
            get_api_log_text(selected_api_log_id),
            file_name=f"{selected_api_log_id}.json",
            mime="application/json",
            key="download_api_execution_log",
        )
        failed_api_logs = [
            row
            for row in api_automation_history
            if row.get("execution_status") == "failed"
        ]
        if failed_api_logs:
            failed_api_ids = [row["execution_id"] for row in failed_api_logs]
            selected_failed_api_id = st.selectbox(
                "Failed API execution to rerun",
                failed_api_ids,
                format_func=lambda execution_id: next(
                    (
                        f"{row.get('dataset_name')} — {row.get('start_time')} "
                        f"— {row.get('error_message') or 'failed'}"
                        for row in failed_api_logs
                        if row.get("execution_id") == execution_id
                    ),
                    execution_id,
                ),
                key="failed_api_execution_selection",
            )
            st.caption(
                "Reruns repeat the recorded API query using the recorded destination "
                "and idempotent duplicate strategy. Runtime MySQL credentials come "
                "from the current environment or Stage 6 credential fields."
            )
            if st.button(
                "Rerun selected failed API execution",
                key="rerun_failed_api_execution",
            ):
                try:
                    with st.spinner("Safely rerunning the failed API execution..."):
                        rerun_result = rerun_failed_api_execution(
                            selected_failed_api_id,
                            connection_config=api_auto_db_config,
                            notify=api_auto_notify,
                        )
                    st.session_state["api_automation_last_result"] = rerun_result
                    st.rerun()
                except Exception as error:
                    from SRC.automation.api_monitor import sanitize_for_log

                    st.error(
                        "Could not rerun the selected API execution: "
                        + str(
                            sanitize_for_log(
                                str(error),
                                secrets=(
                                    str(api_auto_db_config.get("password") or ""),
                                ),
                            )
                        )
                    )
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load the selected API execution log: {error}")
else:
    st.info("No API ETL executions have been logged yet.")

st.header("Stage 7: XML Automation and Monitoring")
st.markdown(
    "Run the complete XML pipeline on local files. XML trees remain hierarchical "
    "through cleaning, validation, integration, and database loading."
)
xml_auto_defaults = get_json_db_environment_config()
xml_auto_col1, xml_auto_col2 = st.columns(2)
xml_auto_source_directory = xml_auto_col1.text_input(
    "XML source directory",
    value=str(Path("data") / "raw" / "xml"),
    key="xml_automation_source_directory",
    help="The pipeline reads *.xml files from this directory and does not modify them.",
)
xml_auto_table = xml_auto_col2.text_input(
    "Automation destination table",
    value="automated_xml",
    key="xml_automation_table",
)
xml_auto_key_field = st.text_input(
    "Optional XML duplicate key",
    value="",
    key="xml_automation_key_field",
    help="Use attr:ID for a root attribute or child:ID for a unique scalar root child. Leave blank for a full-document digest.",
)
xml_auto_columns = st.columns(3)
xml_auto_duplicate_strategy = xml_auto_columns[0].selectbox(
    "Database duplicate strategy",
    ["skip", "update", "reject"],
    format_func={
        "skip": "Skip duplicates",
        "update": "Update existing",
        "reject": "Reject duplicates",
    }.__getitem__,
    key="xml_automation_duplicate_strategy",
)
xml_auto_remove_duplicates = xml_auto_columns[1].checkbox(
    "Remove exact duplicate XML records",
    key="xml_automation_remove_duplicates",
)
xml_auto_notify = xml_auto_columns[2].checkbox(
    "Email on success or failure (SMTP environment settings required)",
    key="xml_automation_notify",
)
xml_auto_retry_count = st.number_input(
    "XML retry attempts for temporary errors",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
    key="xml_automation_retry_attempts",
)
xml_automation_connection_config = {
    **xml_auto_defaults,
    "host": st.text_input(
        "MySQL host (XML automation)",
        value=str(xml_auto_defaults["host"]),
        key="xml_automation_mysql_host",
    ),
    "port": st.number_input(
        "MySQL port (XML automation)",
        min_value=1,
        max_value=65535,
        value=int(xml_auto_defaults["port"]),
        key="xml_automation_mysql_port",
    ),
    "username": st.text_input(
        "MySQL username (XML automation)",
        value=str(xml_auto_defaults["username"]),
        key="xml_automation_mysql_username",
    ),
    "password": st.text_input(
        "MySQL password (XML automation)",
        value=str(xml_auto_defaults["password"]),
        type="password",
        key="xml_automation_mysql_password",
    ),
    "database": st.text_input(
        "MySQL database (XML automation)",
        value=str(xml_auto_defaults["database"]),
        key="xml_automation_mysql_database",
    ),
}
xml_auto_run_disabled = (
    not xml_auto_table.strip() or not xml_auto_source_directory.strip()
)
if st.button(
    "Run XML ETL now",
    key="run_xml_automation_now",
    disabled=xml_auto_run_disabled,
):
    xml_source_files = sorted(
        Path(xml_auto_source_directory).expanduser().glob("*.xml")
    )
    if not xml_source_files:
        st.error("No XML files were found in the selected source directory.")
    else:
        with st.spinner("Running XML ingestion through database loading..."):
            result = run_xml_pipeline(
                xml_source_files,
                table_name=xml_auto_table.strip(),
                unique_key_field=xml_auto_key_field.strip() or None,
                connection_config=xml_automation_connection_config,
                duplicate_strategy=xml_auto_duplicate_strategy,
                cleaning_options={
                    "remove_duplicates": xml_auto_remove_duplicates,
                },
                integration_options={"integration_type": "combine"},
                retry_attempts=int(xml_auto_retry_count),
                notify=xml_auto_notify,
            )
        st.session_state["xml_automation_last_result"] = result

xml_automation_result = st.session_state.get("xml_automation_last_result")
if xml_automation_result:
    if xml_automation_result.get("execution_status") == "success":
        st.success(
            "XML ETL completed successfully. Execution "
            f"`{xml_automation_result['execution_id']}`."
        )
    else:
        st.error(
            f"XML ETL {xml_automation_result.get('execution_status')}: "
            f"{xml_automation_result.get('error_message') or 'See stage details.'}"
        )
    if xml_automation_result.get("error_category"):
        st.caption(
            "Failure category: "
            + str(xml_automation_result["error_category"]).replace("_", " ")
        )
    xml_auto_metrics = st.columns(5)
    for column, label, key in zip(
        xml_auto_metrics,
        ("Processed elements", "Inserted", "Updated", "Skipped", "Failed"),
        (
            "processed_records",
            "inserted_rows",
            "updated_rows",
            "skipped_rows",
            "failed_rows",
        ),
    ):
        column.metric(label, xml_automation_result.get(key, 0))
    if xml_automation_result.get("stage_records"):
        st.subheader("XML stage-by-stage progress")
        st.dataframe(
            pd.DataFrame(xml_automation_result["stage_records"]),
            width="stretch",
            hide_index=True,
        )
    if xml_automation_result.get("integrated_xml") is not None:
        with st.expander("Integrated XML preview"):
            st.code(
                element_tree.tostring(
                    xml_automation_result["integrated_xml"],
                    encoding="unicode",
                ),
                language="xml",
            )

st.subheader("Persistent XML schedule")
try:
    xml_schedule = get_xml_schedule()
except (OSError, ValueError, RuntimeError) as error:
    xml_schedule = {"enabled": False}
    st.error(f"Could not read the saved XML schedule: {error}")
xml_schedule_columns = st.columns(3)
xml_schedule_frequency = xml_schedule_columns[0].selectbox(
    "XML schedule frequency",
    ["daily", "weekly"],
    index=1 if xml_schedule.get("frequency") == "weekly" else 0,
    key="xml_schedule_frequency",
)
xml_schedule_time = xml_schedule_columns[1].time_input(
    "XML execution time (local)",
    value=datetime.strptime(
        str(xml_schedule.get("execution_time", "02:00")), "%H:%M"
    ).time(),
    key="xml_schedule_time",
)
xml_schedule_weekday = xml_schedule_columns[2].selectbox(
    "XML weekly day",
    list(range(7)),
    format_func=lambda day: (
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    )[day],
    index=int(xml_schedule.get("weekday", 0)),
    disabled=xml_schedule_frequency != "weekly",
    key="xml_schedule_weekday",
)
if st.button(
    "Save / update XML schedule",
    key="save_xml_schedule",
    disabled=xml_auto_run_disabled,
):
    try:
        xml_schedule = save_xml_schedule(
            frequency=xml_schedule_frequency,
            execution_time=xml_schedule_time.strftime("%H:%M"),
            weekday=int(xml_schedule_weekday),
            source_directory=xml_auto_source_directory,
            table_name=xml_auto_table.strip(),
            unique_key_field=xml_auto_key_field.strip() or None,
            enabled=(
                bool(xml_schedule.get("enabled"))
                if xml_schedule.get("source_directory")
                else True
            ),
            pipeline_options={
                "duplicate_strategy": xml_auto_duplicate_strategy,
                "cleaning_options": {
                    "remove_duplicates": xml_auto_remove_duplicates,
                },
                "integration_options": {"integration_type": "combine"},
                "retry_attempts": int(xml_auto_retry_count),
                "notify": xml_auto_notify,
            },
        )
        st.session_state["xml_schedule_status"] = (
            "XML schedule saved. Run the standalone scheduler process to execute it."
        )
    except (OSError, ValueError) as error:
        st.error(f"Could not save XML schedule: {error}")
if st.session_state.get("xml_schedule_status"):
    st.success(st.session_state["xml_schedule_status"])
xml_toggle_col, xml_next_col = st.columns(2)
if xml_schedule.get("source_directory"):
    xml_toggle_label = (
        "Disable saved XML schedule"
        if xml_schedule.get("enabled")
        else "Enable saved XML schedule"
    )
    if xml_toggle_col.button(xml_toggle_label, key="toggle_xml_schedule"):
        try:
            xml_schedule = set_xml_schedule_enabled(not xml_schedule["enabled"])
            st.session_state["xml_schedule_status"] = (
                "XML schedule enabled."
                if xml_schedule["enabled"]
                else "XML schedule disabled."
            )
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not update XML schedule: {error}")
    xml_next_run = get_xml_next_run_time(xml_schedule)
    if xml_next_run:
        xml_next_col.metric(
            "Next XML scheduled execution",
            xml_next_run.strftime("%Y-%m-%d %H:%M %Z"),
        )
    else:
        xml_next_col.info("The XML schedule is disabled.")
else:
    st.info("Save a daily or weekly schedule to persist its configuration.")
st.caption(
    f"Schedule file: `{DEFAULT_XML_SCHEDULE_PATH}`. Run "
    "`python -m SRC.automation.xml_scheduler` in a separate long-running "
    "process (for example, Windows Task Scheduler). Scheduled database "
    "credentials come from MYSQL_* environment variables."
)
st.caption(
    "Email notifications use SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, "
    "SMTP_FROM, and SMTP_TO environment variables. Keep credentials outside "
    "the schedule file and source code."
)

st.subheader("XML execution history and logs")
try:
    xml_automation_history = get_xml_execution_history()
except (OSError, ValueError) as error:
    xml_automation_history = []
    st.error(f"Could not read XML execution history: {error}")
if xml_automation_history:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    key: row.get(key)
                    for key in (
                        "execution_id",
                        "dataset_name",
                        "execution_status",
                        "start_time",
                        "end_time",
                        "duration_seconds",
                        "processed_records",
                        "inserted_rows",
                        "updated_rows",
                        "skipped_rows",
                        "failed_rows",
                        "error_category",
                        "error_message",
                    )
                }
                for row in xml_automation_history
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    selected_xml_log_id = st.selectbox(
        "Inspect XML execution log",
        [row["execution_id"] for row in xml_automation_history],
        format_func=lambda execution_id: next(
            (
                f"{row.get('dataset_name')} — {row.get('execution_status')} — "
                f"{row.get('start_time')}"
                for row in xml_automation_history
                if row["execution_id"] == execution_id
            ),
            execution_id,
        ),
        key="xml_automation_log_selection",
    )
    try:
        selected_xml_log = get_xml_execution_log(selected_xml_log_id)
        st.json(selected_xml_log)
        st.download_button(
            "Download XML execution log",
            get_xml_log_text(selected_xml_log_id),
            file_name=f"{selected_xml_log_id}.json",
            mime="application/json",
            key="download_xml_execution_log",
        )
        if selected_xml_log.get("execution_status") == "failed" and st.button(
            "Rerun this failed XML execution",
            key="rerun_failed_xml_execution",
        ):
            rerun_result = rerun_failed_xml_execution(
                selected_xml_log_id,
                connection_config=xml_automation_connection_config,
            )
            st.session_state["xml_automation_last_result"] = rerun_result
            st.rerun()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load the selected XML execution log: {error}")
else:
    st.info("No XML automation executions have been logged yet.")

st.header("Stage 7: JSON Automation and Monitoring")
st.markdown(
    "Run the complete JSON pipeline on local files. Scheduled executions run in "
    "a separate process and preserve nested objects, arrays, and Unicode values."
)
json_auto_defaults = get_json_db_environment_config()
json_auto_col1, json_auto_col2 = st.columns(2)
json_auto_source_directory = json_auto_col1.text_input(
    "JSON source directory",
    value=str(Path("data") / "raw" / "json"),
    key="json_automation_source_directory",
    help="The pipeline reads *.json files from this directory and does not modify them.",
)
json_auto_table = json_auto_col2.text_input(
    "Automation destination table",
    value="automated_json",
    key="json_automation_table",
)
json_auto_keys_text = st.text_input(
    "Unique key fields (comma-separated)",
    value="",
    key="json_automation_keys",
    help="Optional fields used to identify duplicate records in the integrated JSON.",
)
json_auto_keys = [
    key.strip() for key in json_auto_keys_text.split(",") if key.strip()
]
json_auto_options_col1, json_auto_options_col2, json_auto_options_col3 = st.columns(3)
json_auto_duplicate_strategy = json_auto_options_col1.selectbox(
    "Database duplicate strategy",
    ["skip", "update", "reject"],
    format_func={
        "skip": "Skip duplicates",
        "update": "Update existing",
        "reject": "Reject duplicates",
    }.__getitem__,
    key="json_automation_duplicate_strategy",
)
json_auto_integration_type = json_auto_options_col2.selectbox(
    "Integration operation",
    ["combine", "merge"],
    key="json_automation_integration_type",
)
json_auto_join_type = json_auto_options_col2.selectbox(
    "Merge join type",
    ["inner", "left", "right", "outer"],
    key="json_automation_join_type",
    disabled=json_auto_integration_type != "merge",
)
json_auto_notify = st.checkbox(
    "Email me on success or failure (SMTP settings must be in environment variables)",
    key="json_automation_notify",
)
json_auto_retry_count = json_auto_options_col3.number_input(
    "Retry attempts for temporary errors",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
    key="json_automation_retry_attempts",
)
json_auto_required_fields_text = st.text_input(
    "Required JSON fields (optional; comma-separated root fields)",
    value="",
    key="json_automation_required_fields",
)
json_auto_required_fields = [
    field.strip()
    for field in json_auto_required_fields_text.split(",")
    if field.strip()
]
json_automation_connection_config = {
    **json_auto_defaults,
    "host": st.text_input(
        "MySQL host (JSON automation)",
        value=str(json_auto_defaults["host"]),
        key="json_automation_mysql_host",
    ),
    "port": st.number_input(
        "MySQL port (JSON automation)",
        min_value=1,
        max_value=65535,
        value=int(json_auto_defaults["port"]),
        key="json_automation_mysql_port",
    ),
    "username": st.text_input(
        "MySQL username (JSON automation)",
        value=str(json_auto_defaults["username"]),
        key="json_automation_mysql_username",
    ),
    "password": st.text_input(
        "MySQL password (JSON automation)",
        value=str(json_auto_defaults["password"]),
        type="password",
        key="json_automation_mysql_password",
    ),
    "database": st.text_input(
        "MySQL database (JSON automation)",
        value=str(json_auto_defaults["database"]),
        key="json_automation_mysql_database",
    ),
}
json_auto_cleaning_options = {
    "remove_duplicates": st.checkbox(
        "Remove exact JSON duplicates",
        key="json_automation_remove_duplicates",
    ),
}
json_auto_validation_options = {
    "required_fields": {"$": json_auto_required_fields}
    if json_auto_required_fields
    else {},
}
json_auto_integration_options = {
    "integration_type": json_auto_integration_type,
    "join_type": json_auto_join_type,
}
json_auto_run_disabled = (
    not json_auto_table.strip() or not json_auto_source_directory.strip()
)
if st.button(
    "Run JSON ETL now",
    key="run_json_automation_now",
    disabled=json_auto_run_disabled,
):
    source_files = sorted(
        Path(json_auto_source_directory).expanduser().glob("*.json")
    )
    if not source_files:
        st.error("No JSON files were found in the selected source directory.")
    else:
        with st.spinner("Running JSON ingestion through database loading..."):
            result = run_json_pipeline(
                source_files,
                table_name=json_auto_table.strip(),
                unique_key_fields=json_auto_keys,
                connection_config=json_automation_connection_config,
                duplicate_strategy=json_auto_duplicate_strategy,
                cleaning_options=json_auto_cleaning_options,
                validation_options=json_auto_validation_options,
                integration_options=json_auto_integration_options,
                retry_attempts=int(json_auto_retry_count),
                notify=json_auto_notify,
            )
        st.session_state["json_automation_last_result"] = result

json_automation_result = st.session_state.get("json_automation_last_result")
if json_automation_result:
    if json_automation_result.get("execution_status") == "success":
        st.success(
            "JSON ETL completed successfully. Execution "
            f"`{json_automation_result['execution_id']}`."
        )
    else:
        st.error(
            f"JSON ETL {json_automation_result.get('execution_status')}: "
            f"{json_automation_result.get('error_message') or 'See stage details.'}"
        )
    json_auto_metrics = st.columns(5)
    for column, label, key in zip(
        json_auto_metrics,
        ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
        (
            "processed_records",
            "inserted_rows",
            "updated_rows",
            "skipped_rows",
            "failed_rows",
        ),
    ):
        column.metric(label, json_automation_result.get(key, 0))
    if json_automation_result.get("stage_records"):
        st.subheader("JSON stage-by-stage progress")
        st.dataframe(
            pd.DataFrame(json_automation_result["stage_records"]),
            width="stretch",
            hide_index=True,
        )
    if json_automation_result.get("integrated_data") is not None:
        with st.expander("Integrated JSON preview"):
            st.json(json_automation_result["integrated_data"], expanded=False)

st.subheader("Persistent JSON schedule")
try:
    json_schedule = get_json_schedule()
except (OSError, ValueError, RuntimeError) as error:
    json_schedule = {"enabled": False}
    st.error(f"Could not read the saved JSON schedule: {error}")
json_schedule_columns = st.columns(3)
json_schedule_frequency = json_schedule_columns[0].selectbox(
    "JSON schedule frequency",
    ["daily", "weekly"],
    index=1 if json_schedule.get("frequency") == "weekly" else 0,
    key="json_schedule_frequency",
)
json_schedule_time = json_schedule_columns[1].time_input(
    "JSON execution time (local)",
    value=datetime.strptime(
        str(json_schedule.get("execution_time", "02:00")), "%H:%M"
    ).time(),
    key="json_schedule_time",
)
json_schedule_weekday = json_schedule_columns[2].selectbox(
    "JSON weekly day",
    list(range(7)),
    format_func=lambda day: (
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    )[day],
    index=int(json_schedule.get("weekday", 0)),
    disabled=json_schedule_frequency != "weekly",
    key="json_schedule_weekday",
)
if st.button(
    "Save / update JSON schedule",
    key="save_json_schedule",
    disabled=json_auto_run_disabled,
):
    try:
        scheduled_json_options = {
            "duplicate_strategy": json_auto_duplicate_strategy,
            "cleaning_options": json_auto_cleaning_options,
            "validation_options": json_auto_validation_options,
            "integration_options": json_auto_integration_options,
            "retry_attempts": int(json_auto_retry_count),
            "notify": json_auto_notify,
        }
        json_schedule = save_json_schedule(
            frequency=json_schedule_frequency,
            execution_time=json_schedule_time.strftime("%H:%M"),
            weekday=int(json_schedule_weekday),
            source_directory=json_auto_source_directory,
            table_name=json_auto_table.strip(),
            unique_key_fields=json_auto_keys,
            enabled=(
                bool(json_schedule.get("enabled"))
                if json_schedule.get("source_directory")
                else True
            ),
            pipeline_options=scheduled_json_options,
        )
        st.session_state["json_schedule_status"] = (
            "JSON schedule saved. Run the standalone scheduler process to execute it."
        )
    except (OSError, ValueError) as error:
        st.error(f"Could not save JSON schedule: {error}")
if st.session_state.get("json_schedule_status"):
    st.success(st.session_state["json_schedule_status"])
json_schedule_toggle_col, json_schedule_next_col = st.columns(2)
if json_schedule.get("source_directory"):
    json_toggle_label = (
        "Disable saved JSON schedule"
        if json_schedule.get("enabled")
        else "Enable saved JSON schedule"
    )
    if json_schedule_toggle_col.button(
        json_toggle_label,
        key="toggle_json_schedule",
    ):
        try:
            json_schedule = set_json_schedule_enabled(not json_schedule["enabled"])
            st.session_state["json_schedule_status"] = (
                "JSON schedule enabled."
                if json_schedule["enabled"]
                else "JSON schedule disabled."
            )
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not update JSON schedule: {error}")
    json_next_run = get_json_next_run_time(json_schedule)
    if json_next_run:
        json_schedule_next_col.metric(
            "Next JSON scheduled execution",
            json_next_run.strftime("%Y-%m-%d %H:%M %Z"),
        )
    else:
        json_schedule_next_col.info("The JSON schedule is disabled.")
else:
    st.info("Save a daily or weekly schedule to persist its configuration.")
st.caption(
    f"Schedule file: `{DEFAULT_JSON_SCHEDULE_PATH}`. Run "
    "`python -m SRC.automation.json_scheduler` in a separate long-running "
    "process (for example, a service or Windows Task Scheduler). Database "
    "credentials come from MYSQL_* environment variables in that process."
)
st.caption(
    "Email notifications use SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, "
    "SMTP_FROM, and SMTP_TO environment variables. Keep credentials outside "
    "the schedule file and source code."
)

st.subheader("JSON execution history and logs")
try:
    json_automation_history = get_json_execution_history()
except (OSError, ValueError) as error:
    json_automation_history = []
    st.error(f"Could not read JSON execution history: {error}")
if json_automation_history:
    json_history_display = [
        {
            key: row.get(key)
            for key in (
                "execution_id",
                "dataset_name",
                "execution_status",
                "start_time",
                "end_time",
                "duration_seconds",
                "processed_records",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
                "error_message",
            )
        }
        for row in json_automation_history
    ]
    st.dataframe(
        pd.DataFrame(json_history_display),
        width="stretch",
        hide_index=True,
    )
    selected_json_log_id = st.selectbox(
        "Inspect JSON execution log",
        [row["execution_id"] for row in json_automation_history],
        format_func=lambda execution_id: next(
            (
                f"{row.get('dataset_name')} — {row.get('execution_status')} — "
                f"{row.get('start_time')}"
                for row in json_automation_history
                if row["execution_id"] == execution_id
            ),
            execution_id,
        ),
        key="json_automation_log_selection",
    )
    try:
        selected_json_log = get_json_execution_log(selected_json_log_id)
        st.json(selected_json_log)
        st.download_button(
            "Download JSON execution log",
            get_json_log_text(selected_json_log_id),
            file_name=f"{selected_json_log_id}.json",
            mime="application/json",
            key="download_json_execution_log",
        )
        if selected_json_log.get("execution_status") == "failed" and st.button(
            "Rerun this failed JSON execution",
            key="rerun_failed_json_execution",
        ):
            rerun_result = rerun_failed_json_execution(
                selected_json_log_id,
                connection_config=json_automation_connection_config,
            )
            st.session_state["json_automation_last_result"] = rerun_result
            st.rerun()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load the selected JSON execution log: {error}")
else:
    st.info("No JSON automation executions have been logged yet.")

st.header("Stage 7: CSV Automation and Monitoring")
st.markdown(
    "Run the complete CSV pipeline on files in a local directory. Scheduled "
    "runs execute in a separate scheduler process, so Streamlit can be closed."
)
csv_automation_defaults = get_connection_config_from_environment()
csv_auto_col1, csv_auto_col2 = st.columns(2)
csv_auto_source_directory = csv_auto_col1.text_input(
    "CSV source directory",
    value=str(Path("data") / "raw" / "csv"),
    key="csv_automation_source_directory",
    help="The pipeline reads *.csv files from this directory and never changes them.",
)
csv_auto_table = csv_auto_col2.text_input(
    "Automation destination table",
    value="automated_csv",
    key="csv_automation_table",
)
csv_auto_keys_text = st.text_input(
    "Unique key columns (comma-separated)",
    value="",
    key="csv_automation_keys",
    help="Required for safe duplicate handling. Values must be non-null and unique in the integrated data.",
)
csv_auto_keys = [key.strip() for key in csv_auto_keys_text.split(",") if key.strip()]
csv_auto_cleaning_columns = st.columns(3)
csv_auto_remove_duplicates = csv_auto_cleaning_columns[0].checkbox(
    "Remove exact duplicate rows",
    key="csv_automation_remove_duplicates",
)
csv_auto_missing_strategy = csv_auto_cleaning_columns[1].selectbox(
    "Missing-value strategy",
    ["leave", "drop_rows", "mean", "median", "mode"],
    key="csv_automation_missing_strategy",
)
csv_auto_duplicate_strategy = csv_auto_cleaning_columns[2].selectbox(
    "Database duplicate strategy",
    ["skip", "update", "reject"],
    format_func={
        "skip": "Skip duplicates",
        "update": "Update existing",
        "reject": "Reject duplicates",
    }.__getitem__,
    key="csv_automation_duplicate_strategy",
)
csv_auto_integration_column, csv_auto_validation_column = st.columns(2)
csv_auto_integration_type = csv_auto_integration_column.selectbox(
    "Integration operation",
    ["append", "merge"],
    key="csv_automation_integration_type",
)
csv_auto_join_type = csv_auto_integration_column.selectbox(
    "Merge join type",
    ["inner", "left", "right", "outer"],
    key="csv_automation_join_type",
    disabled=csv_auto_integration_type != "merge",
)
csv_auto_required_text = csv_auto_validation_column.text_input(
    "Required fields (comma-separated; optional)",
    value="",
    key="csv_automation_required_fields",
)
csv_auto_required_fields = [
    field.strip()
    for field in csv_auto_required_text.split(",")
    if field.strip()
]
csv_auto_notify = st.checkbox(
    "Email me on success or failure (SMTP settings must be in environment variables)",
    key="csv_automation_notify",
)
csv_auto_retry_count = st.number_input(
    "Retry attempts for temporary errors",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
    key="csv_automation_retry_attempts",
)
csv_automation_connection_config = {
    **csv_automation_defaults,
    "host": st.text_input(
        "MySQL host (automation)",
        value=str(csv_automation_defaults["host"]),
        key="csv_automation_mysql_host",
    ),
    "port": st.number_input(
        "MySQL port (automation)",
        min_value=1,
        max_value=65535,
        value=int(csv_automation_defaults["port"]),
        key="csv_automation_mysql_port",
    ),
    "username": st.text_input(
        "MySQL username (automation)",
        value=str(csv_automation_defaults["username"]),
        key="csv_automation_mysql_username",
    ),
    "password": st.text_input(
        "MySQL password (automation)",
        value=str(csv_automation_defaults["password"]),
        type="password",
        key="csv_automation_mysql_password",
    ),
    "database": st.text_input(
        "MySQL database (automation)",
        value=str(csv_automation_defaults["database"]),
        key="csv_automation_mysql_database",
    ),
}
csv_auto_cleaning_options = {
    "remove_duplicates": csv_auto_remove_duplicates,
    "missing_strategy": csv_auto_missing_strategy,
}
csv_auto_validation_options = {
    "required_fields": csv_auto_required_fields,
}
csv_auto_integration_options = {
    "integration_type": csv_auto_integration_type,
    "join_type": csv_auto_join_type,
}
csv_auto_run_disabled = (
    not csv_auto_keys
    or not csv_auto_table.strip()
    or not csv_auto_source_directory.strip()
)
if st.button(
    "Run CSV ETL now",
    key="run_csv_automation_now",
    disabled=csv_auto_run_disabled,
):
    source_files = sorted(
        Path(csv_auto_source_directory).expanduser().glob("*.csv")
    )
    if not source_files:
        st.error("No CSV files were found in the selected source directory.")
    else:
        with st.spinner("Running CSV ingestion through database loading..."):
            result = run_csv_pipeline(
                source_files,
                table_name=csv_auto_table.strip(),
                unique_key_columns=csv_auto_keys,
                connection_config=csv_automation_connection_config,
                duplicate_strategy=csv_auto_duplicate_strategy,
                cleaning_options=csv_auto_cleaning_options,
                validation_options=csv_auto_validation_options,
                integration_options=csv_auto_integration_options,
                retry_attempts=int(csv_auto_retry_count),
                notify=csv_auto_notify,
            )
        st.session_state["csv_automation_last_result"] = result

csv_automation_result = st.session_state.get("csv_automation_last_result")
if csv_automation_result:
    if csv_automation_result.get("execution_status") == "success":
        st.success(
            f"CSV ETL completed successfully. Execution "
            f"`{csv_automation_result['execution_id']}`."
        )
    else:
        st.error(
            f"CSV ETL {csv_automation_result.get('execution_status')}: "
            f"{csv_automation_result.get('error_message') or 'See stage details.'}"
        )
    csv_auto_metrics = st.columns(5)
    for column, label, key in zip(
        csv_auto_metrics,
        ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
        (
            "processed_rows",
            "inserted_rows",
            "updated_rows",
            "skipped_rows",
            "failed_rows",
        ),
    ):
        column.metric(label, csv_automation_result.get(key, 0))
    stage_rows = csv_automation_result.get("stage_records", [])
    if stage_rows:
        st.subheader("Stage-by-stage progress")
        st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)
    if csv_automation_result.get("integrated_data") is not None:
        with st.expander("Integrated data preview"):
            st.dataframe(
                csv_automation_result["integrated_data"].head(100),
                width="stretch",
            )

st.subheader("Persistent CSV schedule")
try:
    csv_schedule = get_csv_schedule()
except (OSError, ValueError, RuntimeError) as error:
    csv_schedule = {"enabled": False}
    st.error(f"Could not read the saved CSV schedule: {error}")
schedule_columns = st.columns(3)
schedule_frequency = schedule_columns[0].selectbox(
    "Schedule frequency",
    ["daily", "weekly"],
    index=(
        1 if csv_schedule.get("frequency") == "weekly" else 0
    ),
    key="csv_schedule_frequency",
)
schedule_time = schedule_columns[1].time_input(
    "Execution time (local)",
    value=datetime.strptime(
        str(csv_schedule.get("execution_time", "02:00")), "%H:%M"
    ).time(),
    key="csv_schedule_time",
)
schedule_weekday = schedule_columns[2].selectbox(
    "Weekly day",
    list(range(7)),
    format_func=lambda day: (
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    )[day],
    index=int(csv_schedule.get("weekday", 0)),
    disabled=schedule_frequency != "weekly",
    key="csv_schedule_weekday",
)
if st.button(
    "Save / update CSV schedule",
    key="save_csv_schedule",
    disabled=csv_auto_run_disabled,
):
    try:
        scheduled_options = {
            "duplicate_strategy": csv_auto_duplicate_strategy,
            "cleaning_options": csv_auto_cleaning_options,
            "validation_options": csv_auto_validation_options,
            "integration_options": csv_auto_integration_options,
            "retry_attempts": int(csv_auto_retry_count),
            "notify": csv_auto_notify,
        }
        saved_schedule = save_csv_schedule(
            frequency=schedule_frequency,
            execution_time=schedule_time.strftime("%H:%M"),
            weekday=int(schedule_weekday),
            source_directory=csv_auto_source_directory,
            table_name=csv_auto_table.strip(),
            unique_key_columns=csv_auto_keys,
            enabled=(
                bool(csv_schedule.get("enabled"))
                if csv_schedule.get("source_directory")
                else True
            ),
            pipeline_options=scheduled_options,
        )
        st.session_state["csv_schedule_status"] = (
            "Schedule saved. Run the standalone scheduler process to execute it."
        )
        csv_schedule = saved_schedule
    except (OSError, ValueError) as error:
        st.error(f"Could not save CSV schedule: {error}")
schedule_status_message = st.session_state.get("csv_schedule_status")
if schedule_status_message:
    st.success(schedule_status_message)

schedule_toggle_col, schedule_next_col = st.columns(2)
if csv_schedule.get("source_directory"):
    toggle_label = (
        "Disable saved schedule"
        if csv_schedule.get("enabled")
        else "Enable saved schedule"
    )
    if schedule_toggle_col.button(
        toggle_label,
        key="toggle_csv_schedule",
    ):
        try:
            csv_schedule = set_csv_schedule_enabled(not csv_schedule["enabled"])
            st.session_state["csv_schedule_status"] = (
                "Schedule enabled." if csv_schedule["enabled"] else "Schedule disabled."
            )
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not update CSV schedule: {error}")
    next_run = get_next_run_time(csv_schedule)
    if next_run:
        schedule_next_col.metric("Next scheduled execution", next_run.strftime("%Y-%m-%d %H:%M %Z"))
    else:
        schedule_next_col.info("The CSV schedule is disabled.")
else:
    st.info("Save a daily or weekly schedule to persist its configuration.")
st.caption(
    f"Schedule file: `{DEFAULT_SCHEDULE_PATH}`. Run "
    "`python -m SRC.automation.csv_scheduler` from the project directory in a "
    "separate long-running process (for example, a service or Task Scheduler). "
    "Database credentials come from MYSQL_* environment variables in that process."
)
st.caption(
    "Email notifications use SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, "
    "SMTP_FROM, and SMTP_TO environment variables. Keep credentials outside "
    "the schedule file and source code."
)

st.subheader("CSV execution history and logs")
try:
    csv_automation_history = get_csv_execution_history()
except (OSError, ValueError) as error:
    csv_automation_history = []
    st.error(f"Could not read CSV execution history: {error}")
if csv_automation_history:
    history_display = [
        {
            key: row.get(key)
            for key in (
                "execution_id",
                "dataset_name",
                "execution_status",
                "start_time",
                "end_time",
                "duration_seconds",
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
                "error_message",
            )
        }
        for row in csv_automation_history
    ]
    st.dataframe(pd.DataFrame(history_display), width="stretch", hide_index=True)
    selected_log_id = st.selectbox(
        "Inspect execution log",
        [row["execution_id"] for row in csv_automation_history],
        format_func=lambda execution_id: next(
            (
                f"{row.get('dataset_name')} — {row.get('execution_status')} — "
                f"{row.get('start_time')}"
                for row in csv_automation_history
                if row["execution_id"] == execution_id
            ),
            execution_id,
        ),
        key="csv_automation_log_selection",
    )
    try:
        selected_log = get_csv_execution_log(selected_log_id)
        st.json(selected_log)
        st.download_button(
            "Download execution log",
            get_csv_log_text(selected_log_id),
            file_name=f"{selected_log_id}.json",
            mime="application/json",
            key="download_csv_execution_log",
        )
        if selected_log.get("execution_status") == "failed" and st.button(
            "Rerun this failed execution",
            key="rerun_failed_csv_execution",
        ):
            rerun_result = rerun_failed_execution(
                selected_log_id,
                connection_config=csv_automation_connection_config,
            )
            st.session_state["csv_automation_last_result"] = rerun_result
            st.rerun()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load the selected execution log: {error}")
else:
    st.info("No CSV automation executions have been logged yet.")

st.header("Stage 6: JSON Database Loading")
json_loading_candidates = {}
for dataset_key, validation_result in st.session_state.get(
    "json_validation_results", {}
).items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and "valid_data" in validation_result
    ):
        report = validation_result.get("validation_report", {})
        source_filename = str(dataset_key).split("::", maxsplit=1)[0]
        dataset_name = str(
            report.get("dataset_name")
            or Path(source_filename).name
            or dataset_key
        )
        json_loading_candidates[dataset_key] = {
            "validation_result": validation_result,
            "source": source_filename,
            "dataset_name": dataset_name,
        }

if not json_loading_candidates:
    st.info("Run Stage 4 JSON validation successfully before loading to MySQL.")
else:
    json_db_defaults = get_json_db_environment_config()
    st.caption(
        "Credentials default to MYSQL_HOST, MYSQL_PORT, MYSQL_USER, "
        "MYSQL_PASSWORD, and MYSQL_DATABASE environment variables. Password "
        "input is masked and the password is not included in reports."
    )
    json_credential_columns = st.columns(2)
    json_mysql_host = json_credential_columns[0].text_input(
        "MySQL host (JSON)",
        value=str(json_db_defaults["host"]),
        key="json_mysql_host",
    )
    json_mysql_port = json_credential_columns[1].number_input(
        "MySQL port (JSON)",
        min_value=1,
        max_value=65535,
        value=int(json_db_defaults["port"]),
        step=1,
        key="json_mysql_port",
    )
    json_mysql_username = json_credential_columns[0].text_input(
        "MySQL username (JSON)",
        value=str(json_db_defaults["username"]),
        key="json_mysql_username",
    )
    json_mysql_password = json_credential_columns[1].text_input(
        "MySQL password (JSON)",
        value=str(json_db_defaults["password"]),
        type="password",
        key="json_mysql_password",
    )
    json_mysql_database = st.text_input(
        "MySQL database name (JSON)",
        value=str(json_db_defaults["database"]),
        key="json_mysql_database",
    )
    json_mysql_config = {
        "host": json_mysql_host,
        "port": json_mysql_port,
        "username": json_mysql_username,
        "password": json_mysql_password,
        "database": json_mysql_database,
    }

    json_connection_status = st.session_state.get(
        "json_mysql_connection_status"
    )
    if json_connection_status:
        if json_connection_status.get("status") == "success":
            st.success(
                f"MySQL connection verified; database "
                f"`{json_connection_status['database']}` is ready "
                f"({json_connection_status['database_status']})."
            )
        else:
            st.error(
                "MySQL connection/database setup failed: "
                f"{json_connection_status['error']}"
            )
    if st.button(
        "Connect and prepare database for JSON",
        key="prepare_json_mysql_database",
    ):
        try:
            prepared = prepare_json_mysql_database(json_mysql_config)
            prepared["connection"].close()
            st.session_state["json_mysql_connection_status"] = {
                "status": "success",
                "database": prepared["database"],
                "database_status": prepared["database_status"],
            }
        except Exception as error:
            safe_error = (
                str(error).replace(json_mysql_password, "[redacted]")
                if json_mysql_password
                else str(error)
            )
            st.session_state["json_mysql_connection_status"] = {
                "status": "error",
                "error": safe_error,
            }
        st.rerun()

    json_dataset_key = st.selectbox(
        "Select a successfully validated JSON dataset",
        list(json_loading_candidates),
        format_func=lambda key: json_loading_candidates[key]["dataset_name"],
        key="json_database_dataset",
    )
    selected_json_loading = json_loading_candidates[json_dataset_key]
    valid_json = selected_json_loading["validation_result"]["valid_data"]
    st.subheader("Validated JSON preview")
    st.json(valid_json, expanded=False)
    if isinstance(valid_json, list):
        st.caption(
            f"{len(valid_json)} top-level records; previewing the first "
            f"{min(20, len(valid_json))} below."
        )
        st.json(valid_json[:20], expanded=False)
    elif isinstance(valid_json, dict):
        object_arrays = [
            (key, value)
            for key, value in valid_json.items()
            if isinstance(value, list)
        ]
        if object_arrays:
            st.caption(
                "Detected array fields: "
                + ", ".join(f"{key} ({len(value)} items)" for key, value in object_arrays)
            )
            first_array = object_arrays[0][1]
            st.json(first_array[:20], expanded=False)

    suggested_json_table = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        Path(selected_json_loading["dataset_name"]).stem,
    ).strip("_")
    if not suggested_json_table:
        suggested_json_table = "json_dataset"
    if suggested_json_table[0].isdigit():
        suggested_json_table = f"json_{suggested_json_table}"
    suggested_json_table = suggested_json_table[:64]
    json_table_column, json_strategy_column = st.columns(2)
    json_destination_table = json_table_column.text_input(
        "JSON destination table",
        value=suggested_json_table,
        key=f"json_database_table_{json_dataset_key}",
    )
    json_duplicate_strategy = json_strategy_column.selectbox(
        "JSON duplicate strategy",
        ["skip", "update", "reject"],
        format_func={
            "skip": "Skip duplicates",
            "update": "Update existing records",
            "reject": "Reject duplicates",
        }.__getitem__,
        key="json_database_duplicate_strategy",
    )
    json_suggested_keys = suggest_unique_key_fields(valid_json)
    if isinstance(valid_json, list):
        json_root_records = valid_json
    elif isinstance(valid_json, dict):
        object_arrays = [
            value
            for value in valid_json.values()
            if isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ]
        json_root_records = object_arrays[0] if len(object_arrays) == 1 else []
    else:
        json_root_records = []
    json_key_columns = sorted(
        {
            str(field)
            for row in json_root_records
            if isinstance(row, dict)
            for field in row
            if all(isinstance(item, dict) and field in item for item in json_root_records)
        }
    ) if json_root_records else []
    selected_json_keys = st.multiselect(
        "Unique key fields (optional; verified suggestions are preselected)",
        json_key_columns,
        default=(
            [json_suggested_keys[0]]
            if json_suggested_keys
            and json_suggested_keys[0] in json_key_columns
            else []
        ),
        key=f"json_database_keys_{json_dataset_key}",
    )
    st.caption(
        "When no key is selected, a stable content fingerprint is used to "
        "prevent loading an unchanged record twice. Nested arrays are stored "
        "in related child tables with parent keys; nested objects without "
        "arrays use MySQL JSON columns."
    )
    json_batch_size = st.number_input(
        "JSON batch size",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        key="json_database_batch_size",
    )

    if st.button(
        "Load validated JSON to MySQL",
        key="load_validated_json_mysql",
        disabled=not json_destination_table.strip(),
    ):
        with st.spinner("Loading validated JSON records into MySQL..."):
            try:
                json_load_result = load_validated_json_dataset(
                    {
                        "validation_result": selected_json_loading[
                            "validation_result"
                        ],
                        "source": selected_json_loading["source"],
                    },
                    connection_config=json_mysql_config,
                    table_name=json_destination_table.strip(),
                    unique_key_fields=selected_json_keys,
                    duplicate_strategy=json_duplicate_strategy,
                    batch_size=int(json_batch_size),
                )
            except Exception as error:
                safe_error = (
                    str(error).replace(json_mysql_password, "[redacted]")
                    if json_mysql_password
                    else str(error)
                )
                json_load_result = {
                    "execution_status": "failed",
                    "error_message": safe_error,
                    "processed_rows": 0,
                    "inserted_rows": 0,
                    "updated_rows": 0,
                    "skipped_rows": 0,
                    "failed_rows": 0,
                }
        st.session_state["json_database_load_result"] = json_load_result
        st.rerun()

    json_load_result = st.session_state.get("json_database_load_result")
    if json_load_result is not None:
        if json_load_result.get("execution_status") == "success":
            st.success("JSON database load completed successfully.")
        else:
            st.error(
                "JSON database load failed: "
                f"{json_load_result.get('error_message') or 'Unknown database error'}"
            )
        json_load_metrics = st.columns(5)
        for column, label, key in zip(
            json_load_metrics,
            ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
            (
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
            ),
        ):
            column.metric(label, json_load_result.get(key, 0))
        st.json(
            {
                key: value
                for key, value in json_load_result.items()
                if key != "error_message"
            }
        )

    if st.button(
        "Refresh JSON loading history",
        key="refresh_json_loading_history",
    ):
        try:
            history = get_json_db_execution_history(json_mysql_config)
            st.session_state["json_database_history"] = [
                row
                for row in history
                if str(row.get("destination_table", "")).startswith("json:")
            ]
            st.session_state.pop("json_database_history_error", None)
        except Exception as error:
            safe_error = (
                str(error).replace(json_mysql_password, "[redacted]")
                if json_mysql_password
                else str(error)
            )
            st.session_state["json_database_history_error"] = safe_error
        st.rerun()

    json_history_error = st.session_state.get("json_database_history_error")
    if json_history_error:
        st.warning(f"Could not retrieve JSON loading history: {json_history_error}")
    json_history = st.session_state.get("json_database_history")
    if json_history:
        st.subheader("Previous JSON loading history")
        st.dataframe(pd.DataFrame(json_history), width="stretch")
    elif json_history is not None and not json_history_error:
        st.info("No JSON loading history has been recorded yet.")
    st.caption(
        "The MySQL account needs server-level CREATE DATABASE permission and "
        "database-level CREATE TABLE, SELECT, INSERT, UPDATE, and DELETE "
        "permissions (DELETE is used to replace child rows during updates)."
    )

st.header("Stage 6: XML Database Loading")
xml_loading_candidates = {}
for dataset_key, validation_result in st.session_state.get(
    "xml_validation_results", {}
).items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and isinstance(validation_result.get("valid_data"), element_tree.Element)
    ):
        cleaned_result = st.session_state.get("xml_clean_results", {}).get(
            dataset_key, {}
        )
        report = validation_result.get("validation_report", {})
        source_filename = (
            report.get("dataset_name")
            or cleaned_result.get("report", {}).get("dataset_name")
            or str(dataset_key).split("::", maxsplit=1)[0]
        )
        xml_loading_candidates[dataset_key] = {
            "validation_result": validation_result,
            "source": str(source_filename),
            "namespaces": validation_result.get("namespaces")
            or cleaned_result.get("namespaces", {}),
            "dataset_name": Path(str(source_filename)).name,
        }

if not xml_loading_candidates:
    st.info("Run Stage 4 XML validation successfully before loading to MySQL.")
else:
    xml_db_defaults = get_json_db_environment_config()
    st.caption(
        "Credentials default to MYSQL_HOST, MYSQL_PORT, MYSQL_USER, "
        "MYSQL_PASSWORD, and MYSQL_DATABASE environment variables. Password "
        "input is masked and the password is not included in reports."
    )
    xml_credential_columns = st.columns(2)
    xml_mysql_host = xml_credential_columns[0].text_input(
        "MySQL host (XML)",
        value=str(xml_db_defaults["host"]),
        key="xml_mysql_host",
    )
    xml_mysql_port = xml_credential_columns[1].number_input(
        "MySQL port (XML)",
        min_value=1,
        max_value=65535,
        value=int(xml_db_defaults["port"]),
        step=1,
        key="xml_mysql_port",
    )
    xml_mysql_username = xml_credential_columns[0].text_input(
        "MySQL username (XML)",
        value=str(xml_db_defaults["username"]),
        key="xml_mysql_username",
    )
    xml_mysql_password = xml_credential_columns[1].text_input(
        "MySQL password (XML)",
        value=str(xml_db_defaults["password"]),
        type="password",
        key="xml_mysql_password",
    )
    xml_mysql_database = st.text_input(
        "MySQL database name (XML)",
        value=str(xml_db_defaults["database"]),
        key="xml_mysql_database",
    )
    xml_mysql_config = {
        "host": xml_mysql_host,
        "port": xml_mysql_port,
        "username": xml_mysql_username,
        "password": xml_mysql_password,
        "database": xml_mysql_database,
    }

    xml_connection_status = st.session_state.get("xml_mysql_connection_status")
    if xml_connection_status:
        if xml_connection_status.get("status") == "success":
            st.success(
                f"MySQL connection verified; database "
                f"`{xml_connection_status['database']}` is ready "
                f"({xml_connection_status['database_status']})."
            )
        else:
            st.error(
                "MySQL connection/database setup failed: "
                f"{xml_connection_status['error']}"
            )
    if st.button(
        "Connect and prepare database for XML",
        key="prepare_xml_mysql_database",
    ):
        try:
            prepared = prepare_xml_mysql_database(xml_mysql_config)
            prepared["connection"].close()
            st.session_state["xml_mysql_connection_status"] = {
                "status": "success",
                "database": prepared["database"],
                "database_status": prepared["database_status"],
            }
        except Exception as error:
            safe_error = (
                str(error).replace(xml_mysql_password, "[redacted]")
                if xml_mysql_password
                else str(error)
            )
            st.session_state["xml_mysql_connection_status"] = {
                "status": "error",
                "error": safe_error,
            }
        st.rerun()

    xml_dataset_key = st.selectbox(
        "Select a successfully validated XML dataset",
        list(xml_loading_candidates),
        format_func=lambda key: xml_loading_candidates[key]["dataset_name"],
        key="xml_database_dataset",
    )
    selected_xml_loading = xml_loading_candidates[xml_dataset_key]
    valid_xml = selected_xml_loading["validation_result"]["valid_data"]
    suggested_xml_table = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        Path(selected_xml_loading["dataset_name"]).stem,
    ).strip("_")
    if not suggested_xml_table:
        suggested_xml_table = "xml_dataset"
    if suggested_xml_table[0].isdigit():
        suggested_xml_table = f"xml_{suggested_xml_table}"
    suggested_xml_table = suggested_xml_table[:64]
    xml_destination_table = st.text_input(
        "XML root destination table",
        value=suggested_xml_table,
        key=f"xml_database_table_{xml_dataset_key}",
    )

    if xml_destination_table.strip():
        try:
            xml_structure = get_xml_structure_summary(
                valid_xml, xml_destination_table.strip()
            )
            st.subheader("Processed XML structure preview")
            st.code(
                element_tree.tostring(valid_xml, encoding="unicode"),
                language="xml",
            )
            xml_count_metrics = st.columns(2)
            xml_count_metrics[0].metric(
                "Elements to load", xml_structure["total_elements"]
            )
            xml_count_metrics[1].metric(
                "Tables", len(xml_structure["table_relationships"])
            )
            st.subheader("Proposed XML tables and relationships")
            st.dataframe(
                pd.DataFrame(xml_structure["table_relationships"]),
                width="stretch",
                hide_index=True,
            )
            xml_key_candidates = xml_structure["key_candidates"]
            xml_key_options = [None] + [
                candidate["field"] for candidate in xml_key_candidates
            ]
            xml_key_labels = {
                None: "Use stable fingerprint of the complete XML document",
                **{
                    candidate["field"]: candidate["label"]
                    for candidate in xml_key_candidates
                },
            }
            selected_xml_key = st.selectbox(
                "Optional stable XML document key",
                xml_key_options,
                format_func=xml_key_labels.__getitem__,
                key=f"xml_database_key_{xml_dataset_key}",
            )
        except (ValueError, TypeError) as error:
            st.error(f"Cannot map this XML structure to MySQL: {error}")
            xml_structure = None
            selected_xml_key = None
    else:
        st.error("Enter a valid XML destination table name.")
        xml_structure = None
        selected_xml_key = None

    xml_duplicate_strategy = st.selectbox(
        "XML duplicate strategy",
        ["skip", "update", "reject"],
        format_func={
            "skip": "Skip duplicates",
            "update": "Update existing records",
            "reject": "Reject duplicates",
        }.__getitem__,
        key="xml_database_duplicate_strategy",
    )
    xml_batch_size = st.number_input(
        "XML batch size",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        key="xml_database_batch_size",
    )

    if st.button(
        "Load validated XML to MySQL",
        key="load_validated_xml_mysql",
        disabled=xml_structure is None,
    ):
        with st.spinner("Loading validated XML records into MySQL..."):
            try:
                xml_load_result = load_validated_xml_dataset(
                    {
                        "validation_result": selected_xml_loading[
                            "validation_result"
                        ],
                        "source": selected_xml_loading["source"],
                        "namespaces": selected_xml_loading["namespaces"],
                    },
                    connection_config=xml_mysql_config,
                    table_name=xml_destination_table.strip(),
                    unique_key_field=selected_xml_key,
                    duplicate_strategy=xml_duplicate_strategy,
                    batch_size=int(xml_batch_size),
                )
            except Exception as error:
                safe_error = (
                    str(error).replace(xml_mysql_password, "[redacted]")
                    if xml_mysql_password
                    else str(error)
                )
                xml_load_result = {
                    "execution_status": "failed",
                    "error_message": safe_error,
                    "processed_rows": 0,
                    "inserted_rows": 0,
                    "updated_rows": 0,
                    "skipped_rows": 0,
                    "failed_rows": 0,
                }
        st.session_state["xml_database_load_result"] = xml_load_result
        st.rerun()

    xml_load_result = st.session_state.get("xml_database_load_result")
    if xml_load_result is not None:
        if xml_load_result.get("execution_status") == "success":
            st.success("XML database load completed successfully.")
        else:
            st.error(
                "XML database load failed: "
                f"{xml_load_result.get('error_message') or 'Unknown database error'}"
            )
        xml_load_metrics = st.columns(5)
        for column, label, key in zip(
            xml_load_metrics,
            ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
            (
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
            ),
        ):
            column.metric(label, xml_load_result.get(key, 0))
        st.json(
            {
                key: value
                for key, value in xml_load_result.items()
                if key != "error_message"
            }
        )

    if st.button(
        "Refresh XML loading history",
        key="refresh_xml_loading_history",
    ):
        try:
            xml_history = get_json_db_execution_history(xml_mysql_config)
            st.session_state["xml_database_history"] = [
                row
                for row in xml_history
                if str(row.get("destination_table", "")).startswith("xml:")
            ]
            st.session_state.pop("xml_database_history_error", None)
        except Exception as error:
            safe_error = (
                str(error).replace(xml_mysql_password, "[redacted]")
                if xml_mysql_password
                else str(error)
            )
            st.session_state["xml_database_history_error"] = safe_error
        st.rerun()

    xml_history_error = st.session_state.get("xml_database_history_error")
    if xml_history_error:
        st.warning(f"Could not retrieve XML loading history: {xml_history_error}")
    xml_history = st.session_state.get("xml_database_history")
    if xml_history:
        st.subheader("Previous XML loading history")
        st.dataframe(pd.DataFrame(xml_history), width="stretch")
    elif xml_history is not None and not xml_history_error:
        st.info("No XML loading history has been recorded yet.")
    st.caption(
        "The MySQL account needs server-level CREATE DATABASE permission and "
        "database-level CREATE TABLE, SELECT, INSERT, UPDATE, and DELETE "
        "permissions (DELETE is used when replacing child rows during updates)."
    )

st.header("Stage 6: API Database Loading")
api_loading_candidates = {}
for dataset_id, validation_result in st.session_state.get(
    "api_validation_results", {}
).items():
    if (
        isinstance(validation_result, dict)
        and validation_result.get("validation_status")
        in {"passed", "passed_with_warnings"}
        and isinstance(validation_result.get("valid_dataframe"), pd.DataFrame)
    ):
        cleaned_result = st.session_state.get("api_clean_results", {}).get(
            dataset_id, {}
        )
        api_metadata = (
            validation_result.get("api_metadata")
            or cleaned_result.get("api_metadata")
            or {}
        )
        dataset_name = str(
            cleaned_result.get("report", {}).get("dataset_name")
            or dataset_id
        )
        api_loading_candidates[dataset_id] = {
            "validation_result": validation_result,
            "api_metadata": api_metadata,
            "source": dataset_name,
            "dataset_name": dataset_name,
        }

if not api_loading_candidates:
    st.info("Run Stage 4 API validation successfully before loading to MySQL.")
else:
    api_db_defaults = get_connection_config_from_environment()
    st.caption(
        "Credentials default to MYSQL_HOST, MYSQL_PORT, MYSQL_USER, "
        "MYSQL_PASSWORD, and MYSQL_DATABASE environment variables. Password "
        "input is masked and is never included in the loading report."
    )
    api_credential_columns = st.columns(2)
    api_mysql_host = api_credential_columns[0].text_input(
        "MySQL host (API)",
        value=str(api_db_defaults["host"]),
        key="api_mysql_host",
    )
    api_mysql_port = api_credential_columns[1].number_input(
        "MySQL port (API)",
        min_value=1,
        max_value=65535,
        value=int(api_db_defaults["port"]),
        step=1,
        key="api_mysql_port",
    )
    api_mysql_username = api_credential_columns[0].text_input(
        "MySQL username (API)",
        value=str(api_db_defaults["username"]),
        key="api_mysql_username",
    )
    api_mysql_password = api_credential_columns[1].text_input(
        "MySQL password (API)",
        value=str(api_db_defaults["password"]),
        type="password",
        key="api_mysql_password",
    )
    api_mysql_database = st.text_input(
        "MySQL database name (API)",
        value=str(api_db_defaults["database"]),
        key="api_mysql_database",
    )
    api_mysql_config = {
        "host": api_mysql_host,
        "port": api_mysql_port,
        "username": api_mysql_username,
        "password": api_mysql_password,
        "database": api_mysql_database,
    }

    api_connection_status = st.session_state.get("api_mysql_connection_status")
    if api_connection_status:
        if api_connection_status.get("status") == "success":
            st.success(
                f"MySQL connection verified; database "
                f"`{api_connection_status['database']}` is ready "
                f"({api_connection_status['database_status']})."
            )
        else:
            st.error(
                "MySQL connection/database setup failed: "
                f"{api_connection_status['error']}"
            )
    if st.button(
        "Connect and prepare database for API",
        key="prepare_api_mysql_database",
    ):
        try:
            prepared = prepare_mysql_database(api_mysql_config)
            prepared["connection"].close()
            st.session_state["api_mysql_connection_status"] = {
                "status": "success",
                "database": prepared["database"],
                "database_status": prepared["database_status"],
            }
        except Exception as error:
            safe_error = (
                str(error).replace(api_mysql_password, "[redacted]")
                if api_mysql_password
                else str(error)
            )
            st.session_state["api_mysql_connection_status"] = {
                "status": "error",
                "error": safe_error,
            }
        st.rerun()

    api_dataset_id = st.selectbox(
        "Select a successfully validated API dataset",
        list(api_loading_candidates),
        format_func=lambda key: (
            f"{api_loading_candidates[key]['api_metadata'].get('database_id') or 'API'}"
            f" — {api_loading_candidates[key]['api_metadata'].get('indicator_id') or 'all indicators'}"
            f" — {api_loading_candidates[key]['api_metadata'].get('country_reference_area') or 'all areas'}"
            f" ({len(api_loading_candidates[key]['validation_result']['valid_dataframe'])} observations)"
        ),
        key="api_database_dataset",
    )
    selected_api_loading = api_loading_candidates[api_dataset_id]
    valid_api_dataframe = selected_api_loading["validation_result"][
        "valid_dataframe"
    ]
    api_metadata = selected_api_loading["api_metadata"]
    st.subheader("Validated API observation preview")
    st.dataframe(valid_api_dataframe.head(100), width="stretch")
    st.caption(
        f"{len(valid_api_dataframe)} validated observations, "
        f"{len(valid_api_dataframe.columns)} columns."
    )
    with st.expander("API request metadata"):
        st.json(api_metadata)

    api_table_seed = "_".join(
        str(value)
        for value in (
            api_metadata.get("database_id"),
            api_metadata.get("indicator_id"),
        )
        if value
    )
    suggested_api_table = re.sub(
        r"[^A-Za-z0-9_]+", "_", api_table_seed or "api_observations"
    ).strip("_")
    if not suggested_api_table:
        suggested_api_table = "api_observations"
    if suggested_api_table[0].isdigit():
        suggested_api_table = f"api_{suggested_api_table}"
    suggested_api_table = suggested_api_table[:64]
    api_table_column, api_strategy_column = st.columns(2)
    api_destination_table = api_table_column.text_input(
        "API observation destination table",
        value=suggested_api_table,
        key=f"api_database_table_{api_dataset_id}",
    )
    api_duplicate_strategy = api_strategy_column.selectbox(
        "API duplicate strategy",
        ["skip", "update", "reject"],
        format_func={
            "skip": "Skip duplicates",
            "update": "Update existing observations",
            "reject": "Reject duplicates",
        }.__getitem__,
        key="api_database_duplicate_strategy",
    )
    suggested_api_keys = suggest_api_unique_key_columns(valid_api_dataframe)
    selected_api_keys = st.multiselect(
        "Observation key columns (verified unique suggestions are preselected)",
        [str(column) for column in valid_api_dataframe.columns],
        default=suggested_api_keys,
        key=f"api_database_keys_{api_dataset_id}",
    )
    st.caption(
        "When no key columns are selected, a stable fingerprint of all "
        "observation values prevents unchanged records from being loaded twice. "
        "Request metadata is stored separately in a related ingestion table."
    )
    api_batch_size = st.number_input(
        "API batch size",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        key="api_database_batch_size",
    )

    if st.button(
        "Load validated API observations to MySQL",
        key="load_validated_api_mysql",
        disabled=not api_destination_table.strip(),
    ):
        with st.spinner("Loading validated API observations into MySQL..."):
            api_load_result = load_validated_api_dataset(
                {
                    "validation_result": selected_api_loading[
                        "validation_result"
                    ],
                    "api_metadata": api_metadata,
                    "source": selected_api_loading["source"],
                },
                connection_config=api_mysql_config,
                table_name=api_destination_table.strip(),
                unique_key_columns=selected_api_keys,
                duplicate_strategy=api_duplicate_strategy,
                batch_size=int(api_batch_size),
            )
        st.session_state["api_database_load_result"] = api_load_result
        st.rerun()

    api_load_result = st.session_state.get("api_database_load_result")
    if api_load_result is not None:
        if api_load_result.get("execution_status") == "success":
            st.success("API database load completed successfully.")
        else:
            st.error(
                "API database load failed: "
                f"{api_load_result.get('error_message') or 'Unknown database error'}"
            )
        api_load_metrics = st.columns(5)
        for column, label, key in zip(
            api_load_metrics,
            ("Processed", "Inserted", "Updated", "Skipped", "Failed"),
            (
                "processed_rows",
                "inserted_rows",
                "updated_rows",
                "skipped_rows",
                "failed_rows",
            ),
        ):
            column.metric(label, api_load_result.get(key, 0))
        st.json(
            {
                key: value
                for key, value in api_load_result.items()
                if key not in {"error_message", "api_metadata"}
            }
        )

    if st.button(
        "Refresh API loading history",
        key="refresh_api_loading_history",
    ):
        try:
            api_history = get_pipeline_execution_history(api_mysql_config)
            st.session_state["api_database_history"] = [
                row
                for row in api_history
                if str(row.get("destination_table", "")).startswith("api:")
            ]
            st.session_state.pop("api_database_history_error", None)
        except Exception as error:
            safe_error = (
                str(error).replace(api_mysql_password, "[redacted]")
                if api_mysql_password
                else str(error)
            )
            st.session_state["api_database_history_error"] = safe_error
        st.rerun()

    api_history_error = st.session_state.get("api_database_history_error")
    if api_history_error:
        st.warning(f"Could not retrieve API loading history: {api_history_error}")
    api_history = st.session_state.get("api_database_history")
    if api_history:
        st.subheader("Previous API loading history")
        st.dataframe(pd.DataFrame(api_history), width="stretch")
    elif api_history is not None and not api_history_error:
        st.info("No API loading history has been recorded yet.")
    if api_destination_table.strip():
        try:
            api_metadata_table = get_api_metadata_table_name(
                api_destination_table.strip()
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.caption(f"API metadata table: `{api_metadata_table}`.")
    st.caption(
        "The MySQL account needs server-level CREATE DATABASE permission and "
        "database-level CREATE TABLE, SELECT, INSERT, and UPDATE permissions."
    )
