"""Streamlit interface for testing and saving validated records to MySQL."""

import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import mysql.connector
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_storage

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "validated" / "countries_validated.json"
PASSWORD_KEY = "mysql_password_input"
SAVED_SETTINGS_KEY = "saved_mysql_settings"
STORAGE_LOCK = threading.RLock()
ENVIRONMENT_KEYS = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
)

logger = logging.getLogger(__name__)


def _connection_settings(host, port, username, password, database_name):
    """Validate form values and return settings without logging credentials."""
    host = host.strip()
    username = username.strip()
    database_name = database_name.strip()
    if not host or not username or not password:
        raise ValueError("Host, username, and password are required.")
    if not 1 <= port <= 65535:
        raise ValueError("MySQL port must be between 1 and 65535.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", database_name):
        raise ValueError(
            "Database name must be 1–64 characters and contain only "
            "letters, digits, and underscores; it cannot start with a digit."
        )
    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database_name,
    }


def test_mysql_connection(settings):
    """Open and close a MySQL connection, returning a user-safe result."""
    try:
        connection = mysql.connector.connect(
            host=settings["host"],
            port=settings["port"],
            user=settings["username"],
            password=settings["password"],
            database=settings["database"],
            connection_timeout=10,
        )
    except mysql.connector.Error as error:
        logger.warning("MySQL connection test failed (%s)", type(error).__name__)
        message = str(error)
        if settings["password"]:
            message = message.replace(settings["password"], "[redacted]")
        return False, f"MySQL connection failed: {message}"

    try:
        connection.close()
    except mysql.connector.Error as error:
        logger.warning("Could not close tested MySQL connection (%s)", type(error).__name__)
        return False, "The connection opened, but could not be closed cleanly."
    return True, "MySQL connection successful."


@contextmanager
def _provide_storage_credentials(settings):
    """Temporarily provide credentials to the existing storage module.

    The storage module reads host/user/password from environment variables and
    takes the database name from its module constant. Restore all process state
    even if MySQL insertion raises an exception.
    """
    with STORAGE_LOCK:
        environment_values = {
            key: os.environ.get(key)
            for key in ENVIRONMENT_KEYS
        }
        original_database = data_storage.DATABASE_NAME
        try:
            os.environ["MYSQL_HOST"] = settings["host"]
            os.environ["MYSQL_PORT"] = str(settings["port"])
            os.environ["MYSQL_USER"] = settings["username"]
            os.environ["MYSQL_PASSWORD"] = settings["password"]
            data_storage.DATABASE_NAME = settings["database"]
            yield
        finally:
            data_storage.DATABASE_NAME = original_database
            for key, value in environment_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def save_records_to_mysql(settings, input_path=DEFAULT_INPUT_PATH):
    """Use existing storage functions to load validated data and upsert it."""
    records = data_storage.load_validated_records(input_path)
    with _provide_storage_credentials(settings):
        count = data_storage.export_to_mysql(records)
    return {
        "records_upserted_to_mysql": count,
        "database": settings["database"],
        "table": data_storage.TABLE_NAME,
    }


def _clear_connection_state():
    """Clear saved credentials and the password widget when disconnecting."""
    st.session_state.pop(SAVED_SETTINGS_KEY, None)
    st.session_state[PASSWORD_KEY] = ""
    st.session_state["mysql_username"] = ""


def _get_form_settings():
    """Read and validate the currently entered connection settings."""
    return _connection_settings(
        host=st.session_state["mysql_host"],
        port=st.session_state["mysql_port"],
        username=st.session_state["mysql_username"],
        password=st.session_state.get(PASSWORD_KEY, ""),
        database_name=st.session_state["mysql_database"],
    )


def render_connection_form(input_path=DEFAULT_INPUT_PATH):
    """Render the reusable MySQL settings, test, save, and upsert interface."""
    st.title("MySQL Database Connection")
    st.caption("Credentials are kept in this Streamlit session and are never saved to files.")

    st.text_input("MySQL Host", value="localhost", key="mysql_host")
    st.number_input(
        "MySQL Port",
        min_value=1,
        max_value=65535,
        value=3306,
        step=1,
        key="mysql_port",
    )
    st.text_input("MySQL Username", key="mysql_username")
    st.text_input("MySQL Password", type="password", key=PASSWORD_KEY)
    st.text_input(
        "Database Name",
        value=data_storage.DATABASE_NAME,
        key="mysql_database",
    )

    test_column, save_column, disconnect_column = st.columns(3)
    if test_column.button("Test MySQL Connection", use_container_width=True):
        try:
            settings = _get_form_settings()
            connected, message = test_mysql_connection(settings)
            if connected:
                st.success(message)
            else:
                st.error(message)
        except ValueError as error:
            st.error(str(error))

    if save_column.button("Save Database Settings", use_container_width=True):
        try:
            st.session_state[SAVED_SETTINGS_KEY] = _get_form_settings()
            st.success("Database settings saved for this session.")
        except ValueError as error:
            st.error(str(error))

    disconnect_column.button(
        "Disconnect / Clear Credentials",
        use_container_width=True,
        on_click=_clear_connection_state,
    )

    if SAVED_SETTINGS_KEY in st.session_state:
        st.info("Database settings are saved for this session.")
        if st.button("Save to MySQL", type="primary"):
            try:
                summary = save_records_to_mysql(
                    st.session_state[SAVED_SETTINGS_KEY],
                    input_path=input_path,
                )
                st.success(
                    f"Stored {summary['records_upserted_to_mysql']} records in "
                    f"{summary['database']}.{summary['table']}."
                )
                st.json(summary)
            except Exception as error:
                logger.error("Save to MySQL failed (%s)", type(error).__name__)
                safe_message = str(error)
                password = st.session_state[SAVED_SETTINGS_KEY]["password"]
                if password:
                    safe_message = safe_message.replace(password, "[redacted]")
                st.error(f"Could not save records to MySQL: {safe_message}")
    else:
        st.caption("Save database settings before using Save to MySQL.")


def main():
    """Configure the Streamlit page and render the connection interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    st.set_page_config(page_title="MySQL Connection", page_icon="🔌")
    render_connection_form()


if __name__ == "__main__":
    main()
