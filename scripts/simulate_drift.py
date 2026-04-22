"""Inject synthetic drift into the inference feature store for demo purposes.

Simulates a population shift by perturbing numeric features and swapping
categorical distributions, then logs the records as if they came from live
inference. Run drift_detector.py afterwards to see the drift score rise.

Usage:
    python scripts/simulate_drift.py --noise-factor 0.3 --n-records 500
    python scripts/simulate_drift.py --noise-factor 0.3 --n-records 500 --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_store.feature_store import log_inference_batch, read_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Drifted distributions for categoricals (opposite of typical training data)
DRIFTED_CATEGORICALS = {
    "workclass": ["Self-emp-inc", "Federal-gov", "Self-emp-not-inc"],
    "education": ["Doctorate", "Prof-school", "Masters"],
    "marital_status": ["Divorced", "Separated", "Widowed"],
    "occupation": ["Farming-fishing", "Handlers-cleaners", "Other-service"],
    "relationship": ["Wife", "Other-relative", "Unmarried"],
    "race": ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo"],
    "sex": ["Female"],
    "native_country": ["Mexico", "Philippines", "Germany"],
}


def perturb_numerics(df: pd.DataFrame, noise_factor: float, rng: np.random.Generator) -> pd.DataFrame:
    """Add multiplicative noise to numeric columns to shift their distributions.

    Args:
        df: DataFrame with numeric feature columns.
        noise_factor: Scale of perturbation (0.0 = no change, 1.0 = extreme shift).
        rng: NumPy random generator for reproducibility.
    """
    numeric_cols = ["age", "capital_gain", "capital_loss", "hours_per_week", "education_num"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        # Shift mean upward and add noise proportional to noise_factor
        shift = df[col].mean() * noise_factor
        noise = rng.normal(loc=shift, scale=df[col].std() * noise_factor, size=len(df))
        df[col] = (df[col] + noise).clip(lower=0).astype(int)

    # Clamp to valid ranges
    if "age" in df.columns:
        df["age"] = df["age"].clip(17, 90)
    if "hours_per_week" in df.columns:
        df["hours_per_week"] = df["hours_per_week"].clip(1, 99)
    if "education_num" in df.columns:
        df["education_num"] = df["education_num"].clip(1, 16)

    return df


def perturb_categoricals(df: pd.DataFrame, noise_factor: float, rng: np.random.Generator) -> pd.DataFrame:
    """Replace a fraction of categorical values with drifted alternatives.

    Args:
        df: DataFrame with categorical feature columns.
        noise_factor: Fraction of rows to replace (0.0 = none, 1.0 = all).
        rng: NumPy random generator for reproducibility.
    """
    n = len(df)
    n_replace = int(n * noise_factor)
    replace_idx = rng.choice(n, size=n_replace, replace=False)

    for col, drifted_values in DRIFTED_CATEGORICALS.items():
        if col not in df.columns:
            continue
        df.loc[replace_idx, col] = rng.choice(drifted_values, size=n_replace)

    return df


def simulate_drift(noise_factor: float, n_records: int, seed: int, model_version: str) -> None:
    """Generate drifted inference records and write them to the feature store.

    Args:
        noise_factor: How much to perturb the data (0.0–1.0).
        n_records: Number of synthetic inference records to generate.
        seed: Random seed for reproducibility.
        model_version: Model version string to tag the records with.
    """
    rng = np.random.default_rng(seed)

    logger.info(f"Loading test features as base population...")
    base_df = read_features("features_test")
    if base_df.empty:
        logger.error("features_test is empty. Run featurize.py first.")
        sys.exit(1)

    # Sample with replacement to get n_records rows
    sample = base_df.sample(n=n_records, replace=True, random_state=seed).reset_index(drop=True)

    logger.info(f"Applying noise_factor={noise_factor} to {n_records} records...")
    sample = perturb_numerics(sample, noise_factor, rng)
    sample = perturb_categoricals(sample, noise_factor, rng)

    # Synthesise predictions (random, since no model call needed for drift demo)
    sample["prediction"] = rng.integers(0, 2, size=n_records)
    sample["prediction_proba"] = rng.uniform(0.0, 1.0, size=n_records).round(4)

    log_inference_batch(sample, model_version=model_version)
    logger.info(
        f"Injected {n_records} drifted inference records "
        f"(noise_factor={noise_factor}, seed={seed}). "
        f"Run: python monitoring/drift_detector.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject synthetic drift into the inference feature store."
    )
    parser.add_argument(
        "--noise-factor", type=float, default=0.3,
        help="Perturbation scale 0.0–1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--n-records", type=int, default=500,
        help="Number of synthetic records to inject (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model-version", type=str, default="simulated",
        help="Model version tag for injected records (default: simulated)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.noise_factor <= 1.0:
        logger.error("--noise-factor must be between 0.0 and 1.0")
        sys.exit(1)

    simulate_drift(
        noise_factor=args.noise_factor,
        n_records=args.n_records,
        seed=args.seed,
        model_version=args.model_version,
    )
