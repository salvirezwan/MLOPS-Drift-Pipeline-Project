"""Feature engineering: raw Parquet → processed Parquet + DuckDB feature store."""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_store.feature_store import write_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "adult.parquet"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
NUMERIC_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]


def load_params() -> dict:
    """Load pipeline parameters."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transformations to the raw DataFrame.

    Transformations:
    - Fill nulls in categorical columns with 'Unknown'
    - Cast categoricals to string (consistent dtype for DuckDB)
    - Cast numerics to int (drop any remaining nulls)
    """
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object).fillna("Unknown").astype(str).str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["income"] = df["income"].astype(int)
    return df


def featurize() -> None:
    """Full featurize pipeline: load raw → engineer → split → save Parquet + DuckDB."""
    params = load_params()
    test_size: float = params["data"]["test_size"]
    random_state: int = params["data"]["random_state"]

    logger.info(f"Loading raw data from {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    logger.info(f"Raw shape: {df.shape}")

    df = engineer_features(df)
    logger.info(f"After feature engineering: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["income"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # Save processed Parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved {train_path} and {test_path}")

    # Write to DuckDB feature store
    write_features(train_df, "features_train", overwrite=True)
    write_features(test_df, "features_test", overwrite=True)
    logger.info("Feature store updated.")


if __name__ == "__main__":
    featurize()
    logger.info("Featurize complete.")
