"""DuckDB-backed feature store — read/write API for all pipeline stages."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
DB_PATH = _REPO_ROOT / "feature_store" / "store.db"
SCHEMA_PATH = _REPO_ROOT / "feature_store" / "schema.sql"

FEATURE_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
]
TARGET_COL = "income"


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Open a persistent DuckDB connection, initialising schema on first use."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Execute schema.sql to create tables if they don't exist."""
    sql = SCHEMA_PATH.read_text()
    conn.executemany("", [])  # no-op to ensure connection is live
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------

def write_features(df: pd.DataFrame, table: str, overwrite: bool = True) -> None:
    """Write a DataFrame to a feature table.

    Args:
        df: DataFrame with the feature columns (and 'income' for train/test).
        table: One of 'features_train' or 'features_test'.
        overwrite: If True, truncate the table before inserting.
    """
    allowed = {"features_train", "features_test"}
    if table not in allowed:
        raise ValueError(f"table must be one of {allowed}")

    conn = _get_connection()
    if overwrite:
        conn.execute(f"DELETE FROM {table}")
        logger.info(f"Cleared existing rows from {table}")

    # Add integer primary key
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    conn.execute(f"INSERT INTO {table} SELECT * FROM df")
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    logger.info(f"Wrote {count} rows to {table}")
    conn.close()


def log_inference(
    features: dict,
    prediction: int,
    prediction_proba: float,
    model_version: str,
    timestamp: Optional[datetime] = None,
) -> None:
    """Append a single inference record to features_inference.

    Args:
        features: Dict of feature name → value (must include all FEATURE_COLS).
        prediction: Model output (0 or 1).
        prediction_proba: Probability of positive class.
        model_version: MLflow model version string.
        timestamp: Inference time; defaults to now.
    """
    ts = timestamp or datetime.utcnow()
    row = {col: features.get(col) for col in FEATURE_COLS if col != "fnlwgt"}
    row["prediction"] = prediction
    row["prediction_proba"] = prediction_proba
    row["model_version"] = model_version
    row["timestamp"] = ts

    df_row = pd.DataFrame([row])
    conn = _get_connection()
    conn.execute(
        "INSERT INTO features_inference "
        "(timestamp, age, workclass, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country, prediction, prediction_proba, model_version) "
        "SELECT timestamp, age, workclass, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country, prediction, prediction_proba, model_version "
        "FROM df_row"
    )
    conn.close()


def log_inference_batch(df: pd.DataFrame, model_version: str) -> None:
    """Append multiple inference records to features_inference.

    Args:
        df: DataFrame with feature columns + 'prediction' + 'prediction_proba'.
        model_version: MLflow model version string.
    """
    df = df.copy()
    df["model_version"] = model_version
    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.utcnow()

    inference_cols = [
        "timestamp", "age", "workclass", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "prediction", "prediction_proba", "model_version",
    ]
    df = df[[c for c in inference_cols if c in df.columns]]

    conn = _get_connection()
    conn.execute(
        "INSERT INTO features_inference "
        "(timestamp, age, workclass, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country, prediction, prediction_proba, model_version) "
        "SELECT timestamp, age, workclass, education, education_num, marital_status, "
        "occupation, relationship, race, sex, capital_gain, capital_loss, "
        "hours_per_week, native_country, prediction, prediction_proba, model_version "
        "FROM df"
    )
    count = conn.execute("SELECT COUNT(*) FROM features_inference").fetchone()[0]
    logger.info(f"features_inference now has {count} rows")
    conn.close()


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def read_features(table: str) -> pd.DataFrame:
    """Read all rows from a feature table.

    Args:
        table: One of 'features_train', 'features_test', 'features_inference'.
    """
    conn = _get_connection()
    df = conn.execute(f"SELECT * FROM {table}").df()
    conn.close()
    return df


def read_inference_window(days: int) -> pd.DataFrame:
    """Read inference records from the last N days.

    Args:
        days: Number of days to look back from now.
    """
    since = datetime.utcnow() - timedelta(days=days)
    conn = _get_connection()
    df = conn.execute(
        "SELECT * FROM features_inference WHERE timestamp >= ?", [since]
    ).df()
    conn.close()
    logger.info(f"Read {len(df)} inference rows from last {days} days")
    return df


def get_inference_count() -> int:
    """Return total number of rows in features_inference."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM features_inference").fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = _get_connection()
    tables = conn.execute("SHOW TABLES").fetchall()
    logger.info(f"Feature store initialised. Tables: {[t[0] for t in tables]}")
    conn.close()
