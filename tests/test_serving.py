"""Tests for the FastAPI serving layer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


SAMPLE_PAYLOAD = {
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States",
}


def _make_mock_predictor(prediction: int = 1, proba: float = 0.82, version: str = "1") -> MagicMock:
    """Build a fully-configured mock ModelPredictor."""
    mock = MagicMock()
    mock.load.return_value = True
    mock.is_loaded = True
    mock.version = version
    mock.predict.return_value = (prediction, proba)
    mock.info.return_value = {
        "model_name": "adult_income_classifier",
        "stage": "Production",
        "version": version,
        "run_id": "abc123",
    }
    return mock


@pytest.fixture()
def client():
    """TestClient with both predictors mocked — no MLflow or disk access needed."""
    prod_mock = _make_mock_predictor(prediction=1, proba=0.82)
    shadow_mock = _make_mock_predictor(prediction=1, proba=0.79, version="2")

    # Patch the module-level singletons in serving.main directly
    with patch("serving.main.prod_predictor", prod_mock), \
         patch("serving.main.shadow_predictor", shadow_mock):
        from serving.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


class TestModelInfoEndpoint:
    def test_returns_200_when_model_loaded(self, client):
        assert client.get("/model-info").status_code == 200

    def test_contains_required_keys(self, client):
        data = client.get("/model-info").json()
        assert "model_name" in data
        assert "stage" in data
        assert "version" in data


class TestPredictEndpoint:
    def test_returns_200(self, client):
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert "prediction" in data
        assert "prediction_proba" in data
        assert "model_version" in data
        assert "income_label" in data

    def test_prediction_is_binary(self, client):
        assert client.post("/predict", json=SAMPLE_PAYLOAD).json()["prediction"] in (0, 1)

    def test_income_label_matches_prediction(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        expected = ">50K" if data["prediction"] == 1 else "<=50K"
        assert data["income_label"] == expected

    def test_rejects_invalid_age(self, client):
        assert client.post("/predict", json={**SAMPLE_PAYLOAD, "age": 5}).status_code == 422

    def test_rejects_missing_required_field(self, client):
        payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "age"}
        assert client.post("/predict", json=payload).status_code == 422


class TestShadowEndpoint:
    def test_returns_200(self, client):
        assert client.post("/shadow", json=SAMPLE_PAYLOAD).status_code == 200

    def test_response_schema(self, client):
        data = client.post("/shadow", json=SAMPLE_PAYLOAD).json()
        assert "prediction" in data
        assert "shadow_divergence_rate" in data
        assert "shadow_total_requests" in data

    def test_divergence_rate_is_float(self, client):
        rate = client.post("/shadow", json=SAMPLE_PAYLOAD).json()["shadow_divergence_rate"]
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


class TestMetricsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_content_type_is_prometheus(self, client):
        assert "text/plain" in client.get("/metrics").headers["content-type"]

    def test_contains_custom_metrics(self, client):
        client.post("/predict", json=SAMPLE_PAYLOAD)
        body = client.get("/metrics").text
        assert "mlops_request_count_total" in body
        assert "mlops_request_latency_seconds" in body
        assert "mlops_prediction_total" in body


class TestPredictor:
    def test_predict_returns_tuple(self):
        from serving.predictor import ModelPredictor
        predictor = ModelPredictor(stage="Production")
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.18, 0.82]]
        predictor.model = mock_model
        predictor.version = "1"
        pred, proba = predictor.predict(SAMPLE_PAYLOAD)
        assert pred == 1
        assert abs(proba - 0.82) < 0.001

    def test_predict_raises_when_model_not_loaded(self):
        from serving.predictor import ModelPredictor
        predictor = ModelPredictor(stage="Production")
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.predict(SAMPLE_PAYLOAD)


class TestShadowTracker:
    def test_divergence_rate_zero_when_predictions_match(self):
        from serving.shadow import ShadowTracker
        tracker = ShadowTracker()
        for _ in range(10):
            tracker.record(0, 0)
        assert tracker.divergence_rate == 0.0

    def test_divergence_rate_one_when_all_differ(self):
        from serving.shadow import ShadowTracker
        tracker = ShadowTracker()
        for _ in range(10):
            tracker.record(0, 1)
        assert tracker.divergence_rate == 1.0

    def test_divergence_rate_partial(self):
        from serving.shadow import ShadowTracker
        tracker = ShadowTracker()
        tracker.record(0, 1)
        tracker.record(0, 0)
        assert abs(tracker.divergence_rate - 0.5) < 0.001

    def test_total_requests_increments(self):
        from serving.shadow import ShadowTracker
        tracker = ShadowTracker()
        tracker.record(0, 0)
        tracker.record(1, 1)
        assert tracker.total_requests == 2
