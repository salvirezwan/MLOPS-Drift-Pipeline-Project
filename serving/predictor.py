"""Model loading and inference logic for the FastAPI serving layer."""

import logging
from pathlib import Path
from typing import Optional

import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

MLRUNS_PATH = Path(__file__).parent.parent / "mlruns"
MODEL_NAME = "adult_income_classifier"

FEATURE_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week", "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country",
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]


class ModelPredictor:
    """Loads and serves a registered MLflow model for a given stage."""

    def __init__(self, stage: str = "Production") -> None:
        """Initialise predictor for the given registry stage.

        Args:
            stage: MLflow model registry stage ('Production' or 'Staging').
        """
        self.stage = stage
        self.model = None
        self.version: Optional[str] = None
        self.run_id: Optional[str] = None
        mlflow.set_tracking_uri("mlruns")

    def load(self) -> bool:
        """Load the latest model for this stage from the MLflow registry.

        Returns:
            True if a model was successfully loaded, False if none exists.
        """
        client = MlflowClient(tracking_uri="mlruns")
        versions = client.get_latest_versions(MODEL_NAME, stages=[self.stage])
        if not versions:
            logger.warning(f"No {self.stage} model found in registry")
            return False

        v = versions[-1]
        model_uri = f"models:/{MODEL_NAME}/{self.stage}"
        self.model = mlflow.xgboost.load_model(model_uri)
        self.version = v.version
        self.run_id = v.run_id
        logger.info(f"Loaded {self.stage} model v{self.version} (run {self.run_id})")
        return True

    def predict(self, features: dict) -> tuple[int, float]:
        """Run inference on a single feature dict.

        Args:
            features: Dict mapping feature name to value.

        Returns:
            Tuple of (predicted_class, probability_of_positive_class).
        """
        if self.model is None:
            raise RuntimeError(f"{self.stage} model is not loaded")

        df = pd.DataFrame([features])

        # Ensure all feature columns present
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0 if col not in CATEGORICAL_COLS else "Unknown"

        # Encode categoricals the same way training did
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category").cat.codes

        df = df[FEATURE_COLS]
        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0][1])
        return pred, proba

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return self.model is not None

    def info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_name": MODEL_NAME,
            "stage": self.stage,
            "version": str(self.version) if self.version else "none",
            "run_id": self.run_id or "none",
        }
