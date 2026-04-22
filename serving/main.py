"""FastAPI serving app — /predict, /shadow, /health, /model-info, /metrics."""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from serving.metrics import (
    MODEL_VERSION_INFO,
    PREDICTION_DISTRIBUTION,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from serving.predictor import ModelPredictor
from serving.shadow import ShadowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHADOW_MODE = os.getenv("SHADOW_MODE", "true").lower() == "true"

# Module-level singletons — initialised in lifespan
prod_predictor: ModelPredictor = ModelPredictor(stage="Production")
shadow_predictor: ModelPredictor = ModelPredictor(stage="Staging")
shadow_tracker: ShadowTracker = ShadowTracker()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load models on startup."""
    loaded = prod_predictor.load()
    if not loaded:
        logger.warning("No Production model loaded — /predict will return 503")
    else:
        info = prod_predictor.info()
        MODEL_VERSION_INFO.info({
            "model_name": info["model_name"],
            "version": info["version"],
            "run_id": info["run_id"],
        })

    if SHADOW_MODE:
        shadow_predictor.load()

    yield


app = FastAPI(
    title="Adult Income MLOps API",
    description="Production serving with shadow deployment and Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input features for the Adult Income classifier."""

    age: int = Field(..., ge=17, le=90)
    workclass: str = "Private"
    fnlwgt: int = Field(default=0, ge=0)
    education: str = "HS-grad"
    education_num: int = Field(default=9, ge=1, le=16)
    marital_status: str = "Never-married"
    occupation: str = "Other-service"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(default=0, ge=0)
    capital_loss: int = Field(default=0, ge=0)
    hours_per_week: int = Field(default=40, ge=1, le=99)
    native_country: str = "United-States"


class PredictResponse(BaseModel):
    """Prediction output."""

    model_config = {"protected_namespaces": ()}

    prediction: int
    prediction_proba: float
    model_version: str
    income_label: str


class ShadowResponse(BaseModel):
    """Shadow deployment response — returns Production result only."""

    model_config = {"protected_namespaces": ()}

    prediction: int
    prediction_proba: float
    model_version: str
    income_label: str
    shadow_divergence_rate: float
    shadow_total_requests: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_prod_model() -> None:
    if not prod_predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Production model not available")


def _log_inference(features: dict, prediction: int, proba: float, version: str) -> None:
    """Fire-and-forget: log to feature store without blocking the response."""
    try:
        from feature_store.feature_store import log_inference
        from datetime import datetime
        log_inference(
            features=features,
            prediction=prediction,
            prediction_proba=proba,
            model_version=version,
        )
    except Exception as e:
        logger.warning(f"Failed to log inference to feature store: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {
        "status": "ok",
        "production_model_loaded": prod_predictor.is_loaded,
        "shadow_model_loaded": shadow_predictor.is_loaded,
        "shadow_mode": SHADOW_MODE,
    }


@app.get("/model-info")
def model_info() -> dict:
    """Return current Production model version and metadata."""
    _require_prod_model()
    return prod_predictor.info()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run Production model inference."""
    _require_prod_model()
    start = time.perf_counter()

    features = request.model_dump()
    prediction, proba = prod_predictor.predict(features)
    version = str(prod_predictor.version)
    label = ">50K" if prediction == 1 else "<=50K"

    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    REQUEST_COUNT.labels(endpoint="/predict", status_code="200").inc()
    PREDICTION_DISTRIBUTION.labels(prediction=str(prediction)).inc()

    _log_inference(features, prediction, proba, version)

    return PredictResponse(
        prediction=prediction,
        prediction_proba=round(proba, 4),
        model_version=version,
        income_label=label,
    )


@app.post("/shadow", response_model=ShadowResponse)
def shadow(request: PredictRequest) -> ShadowResponse:
    """Run both Production and Staging models; return Production result.

    Staging prediction is logged silently and used for divergence tracking only.
    """
    _require_prod_model()
    start = time.perf_counter()

    features = request.model_dump()

    # Production prediction
    prod_pred, prod_proba = prod_predictor.predict(features)
    prod_version = str(prod_predictor.version)

    # Shadow (Staging) prediction — silent, never returned to caller
    shadow_pred: Optional[int] = None
    if SHADOW_MODE and shadow_predictor.is_loaded:
        try:
            shadow_pred, shadow_proba = shadow_predictor.predict(features)
            shadow_tracker.record(prod_pred, shadow_pred)

            # Log shadow inference separately
            _log_inference(features, shadow_pred, shadow_proba, f"shadow-v{shadow_predictor.version}")
        except Exception as e:
            logger.warning(f"Shadow model inference failed: {e}")

    latency = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint="/shadow").observe(latency)
    REQUEST_COUNT.labels(endpoint="/shadow", status_code="200").inc()
    PREDICTION_DISTRIBUTION.labels(prediction=str(prod_pred)).inc()

    _log_inference(features, prod_pred, prod_proba, prod_version)

    label = ">50K" if prod_pred == 1 else "<=50K"
    return ShadowResponse(
        prediction=prod_pred,
        prediction_proba=round(prod_proba, 4),
        model_version=prod_version,
        income_label=label,
        shadow_divergence_rate=round(shadow_tracker.divergence_rate, 4),
        shadow_total_requests=shadow_tracker.total_requests,
    )


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
