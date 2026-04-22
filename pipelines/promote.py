"""Promote Staging model to Production if evaluate.py says it should be promoted."""

import json
import logging
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
RESULT_PATH = Path(__file__).parent.parent / "evaluate_result.json"
MLRUNS_PATH = Path(__file__).parent.parent / "mlruns"
MODEL_NAME = "adult_income_classifier"


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def promote() -> bool:
    """Read evaluate_result.json and promote Staging → Production if warranted.

    Returns:
        True if promotion occurred, False otherwise.
    """
    if not RESULT_PATH.exists():
        logger.error(f"Evaluation result not found at {RESULT_PATH}. Run evaluate.py first.")
        sys.exit(1)

    result = json.loads(RESULT_PATH.read_text())
    logger.info(f"Evaluation result: {result}")

    if not result["should_promote"]:
        logger.info(
            f"Promotion skipped: improvement {result['improvement']:.4f} "
            f"< threshold {result['threshold']}"
        )
        return False

    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient(tracking_uri="mlruns")

    staging_version = result["staging_version"]

    # Archive current Production model (if any)
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=pv.version,
            stage="Archived",
            archive_existing_versions=False,
        )
        logger.info(f"Archived previous Production model v{pv.version}")

    # Promote Staging → Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_version,
        stage="Production",
        archive_existing_versions=False,
    )
    logger.info(
        f"Promoted model v{staging_version} to Production "
        f"(AUC {result['staging_auc']:.4f}, "
        f"+{result['improvement']:.4f} vs previous Production)"
    )
    return True


if __name__ == "__main__":
    promoted = promote()
    sys.exit(0 if promoted else 0)  # never block pipeline on promotion decision
