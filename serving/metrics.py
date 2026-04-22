"""Prometheus metrics definitions for the FastAPI serving layer."""

from prometheus_client import Counter, Gauge, Histogram, Info

REQUEST_COUNT = Counter(
    "mlops_request_count_total",
    "Total number of requests",
    ["endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "mlops_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_DISTRIBUTION = Counter(
    "mlops_prediction_total",
    "Count of predictions by class",
    ["prediction"],
)

SHADOW_DIVERGENCE_RATE = Gauge(
    "mlops_shadow_divergence_rate",
    "Rolling percentage of requests where production and shadow predictions differ",
)

DRIFT_SCORE = Gauge(
    "mlops_drift_score",
    "Latest Evidently data drift score",
)

MODEL_VERSION_INFO = Info(
    "mlops_model_version",
    "Current production model version metadata",
)
