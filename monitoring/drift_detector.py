"""Evidently drift detection — compare reference data vs recent inference window.

Outputs:
  - monitoring/reports/drift_YYYY-MM-DD.html
  - monitoring/reports/drift_YYYY-MM-DD.json
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.report import Report

_METRICS = [DatasetDriftMetric(), DataDriftTable()]

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_store.feature_store import read_inference_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
REFERENCE_PATH = Path(__file__).parent.parent / "data" / "reference" / "reference.parquet"
REPORTS_DIR = Path(__file__).parent / "reports"

FEATURE_COLS = [
    "age", "workclass", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
]


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_reference() -> pd.DataFrame:
    """Load the reference dataset saved at training time."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference snapshot not found at {REFERENCE_PATH}. Run train.py first."
        )
    df = pd.read_parquet(REFERENCE_PATH)
    # Keep only feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available]


def load_current(days: int) -> pd.DataFrame:
    """Load recent inference data from the feature store.

    Args:
        days: Number of days to look back.
    """
    df = read_inference_window(days)
    if df.empty:
        logger.warning(f"No inference data found in the last {days} days.")
        return df
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available]


def extract_drift_score(report: Report) -> float:
    """Extract the share of drifted columns from an Evidently report.

    Returns:
        Float between 0 and 1 representing the fraction of drifted features.
    """
    report_dict = report.as_dict()
    for metric in report_dict.get("metrics", []):
        if "DatasetDriftMetric" in metric.get("metric", ""):
            result = metric.get("result", {})
            return float(result.get("share_of_drifted_columns", 0.0))
    return 0.0


def run_drift_detection() -> dict:
    """Run full drift detection pipeline.

    Returns:
        Dict with keys: drift_score, drifted_columns, total_columns,
        drift_detected, report_html_path, report_json_path, current_rows.
    """
    params = load_params()
    check_window_days: int = params["monitoring"]["check_window_days"]
    drift_threshold: float = params["monitoring"]["drift_threshold"]

    logger.info("Loading reference data...")
    reference_df = load_reference()
    logger.info(f"Reference: {reference_df.shape}")

    logger.info(f"Loading inference data (last {check_window_days} days)...")
    current_df = load_current(check_window_days)

    if current_df.empty:
        logger.warning("No current data — skipping drift detection.")
        return {
            "drift_score": 0.0,
            "drifted_columns": 0,
            "total_columns": 0,
            "drift_detected": False,
            "current_rows": 0,
            "report_html_path": None,
            "report_json_path": None,
        }

    logger.info(f"Current: {current_df.shape}")

    # Align columns between reference and current
    shared_cols = [c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns]
    reference_df = reference_df[shared_cols]
    current_df = current_df[shared_cols]

    column_mapping = ColumnMapping(
        numerical_features=[
            c for c in shared_cols
            if reference_df[c].dtype in ("int32", "int64", "float32", "float64")
        ],
        categorical_features=[
            c for c in shared_cols
            if reference_df[c].dtype == object
        ],
    )

    report = Report(metrics=_METRICS)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    drift_score = extract_drift_score(report)
    drift_detected = drift_score > drift_threshold

    logger.info(f"Drift score: {drift_score:.4f} (threshold: {drift_threshold})")
    logger.info(f"Drift detected: {drift_detected}")

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    html_path = REPORTS_DIR / f"drift_{today}.html"
    json_path = REPORTS_DIR / f"drift_{today}.json"

    report.save_html(str(html_path))
    logger.info(f"HTML report saved to {html_path}")

    report_dict = report.as_dict()
    json_path.write_text(json.dumps(report_dict, indent=2, default=str))
    logger.info(f"JSON report saved to {json_path}")

    result = {
        "drift_score": round(drift_score, 6),
        "drift_detected": drift_detected,
        "drift_threshold": drift_threshold,
        "current_rows": len(current_df),
        "reference_rows": len(reference_df),
        "check_window_days": check_window_days,
        "report_html_path": str(html_path),
        "report_json_path": str(json_path),
    }

    return result


if __name__ == "__main__":
    result = run_drift_detection()
    logger.info(f"Drift detection complete: {result}")
    sys.exit(0)
