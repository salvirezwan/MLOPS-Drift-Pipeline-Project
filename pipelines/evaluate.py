"""Compare Staging vs Production model AUC on hold-out test set.

Writes evaluation results to evaluate_result.json for promote.py to consume.
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
TEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "test.parquet"
RESULT_PATH = Path(__file__).parent.parent / "evaluate_result.json"
MLRUNS_PATH = Path(__file__).parent.parent / "mlruns"

FEATURE_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week", "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country",
]
TARGET_COL = "income"
MODEL_NAME = "adult_income_classifier"


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categoricals and return (X, y)."""
    df = df.copy()
    cat_cols = [c for c in FEATURE_COLS if df[c].dtype == object]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df[FEATURE_COLS], df[TARGET_COL]


def load_model_by_stage(client: MlflowClient, stage: str):
    """Load the latest model version for a given registry stage."""
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        return None, None
    v = versions[-1]
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.xgboost.load_model(model_uri)
    logger.info(f"Loaded {stage} model v{v.version} from run {v.run_id}")
    return model, v.version


def score_model(model, X: pd.DataFrame, y: pd.Series) -> float:
    """Return ROC-AUC for a model on (X, y)."""
    y_proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, y_proba))


def evaluate() -> dict:
    """Run evaluation and return result dict.

    Returns a dict with keys: staging_auc, production_auc, improvement, should_promote.
    """
    params = load_params()
    threshold: float = params["promotion"]["min_auc_improvement"]

    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient(tracking_uri="mlruns")

    logger.info(f"Loading test data from {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)
    X_test, y_test = prepare_features(test_df)

    staging_model, staging_version = load_model_by_stage(client, "Staging")
    if staging_model is None:
        logger.error("No Staging model found. Run train.py first.")
        sys.exit(1)

    staging_auc = score_model(staging_model, X_test, y_test)
    logger.info(f"Staging model v{staging_version} AUC: {staging_auc:.4f}")

    prod_model, prod_version = load_model_by_stage(client, "Production")
    if prod_model is None:
        logger.warning("No Production model found. Staging will be promoted automatically.")
        production_auc = 0.0
        prod_version = "none"
    else:
        production_auc = score_model(prod_model, X_test, y_test)
        logger.info(f"Production model v{prod_version} AUC: {production_auc:.4f}")

    improvement = staging_auc - production_auc
    should_promote = improvement >= threshold

    result = {
        "staging_version": str(staging_version),
        "production_version": str(prod_version),
        "staging_auc": round(staging_auc, 6),
        "production_auc": round(production_auc, 6),
        "improvement": round(improvement, 6),
        "threshold": threshold,
        "should_promote": should_promote,
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Evaluation result: {result}")
    logger.info(f"Saved result to {RESULT_PATH}")

    if should_promote:
        logger.info(f"Staging beats Production by {improvement:.4f} >= {threshold} → PROMOTE")
    else:
        logger.info(f"Staging improvement {improvement:.4f} < {threshold} → DO NOT promote")

    return result


if __name__ == "__main__":
    evaluate()
