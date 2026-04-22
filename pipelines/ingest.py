"""Download UCI Adult Income dataset and save as Parquet to data/raw/."""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def download_adult_dataset() -> pd.DataFrame:
    """Fetch UCI Adult Income dataset via sklearn and return as DataFrame."""
    logger.info("Fetching UCI Adult Income dataset from OpenML...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df: pd.DataFrame = dataset.frame.copy()

    # Normalize column names: lowercase, replace spaces/hyphens with underscores
    df.columns = [c.lower().replace("-", "_").replace(" ", "_") for c in df.columns]

    # Encode target: '>50K' → 1, '<=50K' → 0
    df["income"] = (df["class"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
    df = df.drop(columns=["class"])

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['income'].value_counts(normalize=True).round(3)}")
    return df


def save_raw(df: pd.DataFrame) -> Path:
    """Save DataFrame to data/raw/adult.parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "adult.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved raw data to {out_path}")
    return out_path


def ingest() -> Path:
    """Full ingest pipeline: download → save raw Parquet."""
    load_params()  # validate params file is readable
    df = download_adult_dataset()
    return save_raw(df)


if __name__ == "__main__":
    path = ingest()
    logger.info(f"Ingest complete: {path}")
