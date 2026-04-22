# CLAUDE.md — MLOps Pipeline with Drift Detection & CI/CD

## Project Overview

End-to-end MLOps portfolio project demonstrating production engineering maturity.
The ML model is intentionally simple (XGBoost binary classifier on UCI Adult Income dataset).
All complexity lives in the infrastructure. This is the point.

**Goal:** Automated retraining pipeline with data validation, experiment tracking, model versioning,
containerized serving, drift monitoring, and a self-healing CI/CD loop — using 100% free tools.

---

## Repository Structure

```
mlops-pipeline/
├── CLAUDE.md                   ← you are here
├── README.md
├── .env.example                ← env vars template (never commit .env)
├── .github/
│   └── workflows/
│       ├── pr-validation.yml   ← runs on every PR
│       ├── retrain.yml         ← triggered by dispatch or weekly cron
│       └── drift-monitor.yml   ← daily cron drift check
├── data/
│   ├── raw/                    ← raw CSVs (tracked by DVC)
│   ├── processed/              ← feature-engineered Parquet (tracked by DVC)
│   └── reference/              ← training reference snapshot for Evidently
├── feature_store/
│   ├── store.db                ← DuckDB database file
│   ├── schema.sql              ← table definitions
│   └── feature_store.py        ← read/write API
├── pipelines/
│   ├── ingest.py               ← download + save raw data
│   ├── validate.py             ← Great Expectations validation
│   ├── featurize.py            ← feature engineering → DuckDB + Parquet
│   ├── train.py                ← XGBoost training + MLflow logging
│   ├── evaluate.py             ← compare new vs Production model
│   └── promote.py              ← promote model in MLflow registry
├── serving/
│   ├── main.py                 ← FastAPI app
│   ├── predictor.py            ← model loading + inference logic
│   ├── shadow.py               ← shadow deployment logic
│   ├── metrics.py              ← Prometheus metrics definitions
│   └── Dockerfile
├── monitoring/
│   ├── drift_detector.py       ← Evidently drift reports
│   ├── alerting.py             ← GitHub dispatch trigger on drift
│   └── reports/                ← generated HTML Evidently reports
├── tests/
│   ├── test_validate.py
│   ├── test_featurize.py
│   ├── test_serving.py
│   └── test_drift.py
├── great_expectations/
│   ├── great_expectations.yml
│   └── expectations/
│       └── adult_income_suite.json
├── docker-compose.yml          ← FastAPI + MLflow + Prometheus + Grafana
├── dvc.yaml                    ← DVC pipeline DAG
├── dvc.lock
├── MLproject                   ← MLflow project entry points
├── params.yaml                 ← all hyperparams and thresholds (DVC params)
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Model | XGBoost (sklearn API) | Simple — infra is the focus |
| Data validation | Great Expectations | Schema + quality checks |
| Feature store | DuckDB + Parquet | Lightweight, no server needed |
| Experiment tracking | MLflow (local) | Runs via `mlflow ui` |
| Data/model versioning | DVC | Local remote at `./dvc-storage` |
| Orchestration | GitHub Actions | Free CI/CD runner |
| Serving | FastAPI | With shadow deployment endpoint |
| Metrics export | Prometheus (via prometheus-fastapi-instrumentator) | |
| Monitoring | Evidently AI | Data + target drift |
| Dashboards | Grafana (self-hosted via Docker Compose) | No cloud needed |
| Containerization | Docker + Docker Compose | No Kubernetes needed |

---

## Dataset

**UCI Adult Income** — binary classification (income >50K or not)

```python
# Canonical load:
from sklearn.datasets import fetch_openml
data = fetch_openml(name='adult', version=2, as_frame=True)
```

Features: age, workclass, education, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

Target: `income` → binary (0 = <=50K, 1 = >50K)

---

## params.yaml — Single Source of Truth

All configurable values must live here. Never hardcode thresholds.

```yaml
data:
  test_size: 0.2
  random_state: 42
  reference_sample_size: 5000   # rows used as Evidently reference

model:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  use_label_encoder: false
  eval_metric: logloss

promotion:
  min_auc_improvement: 0.005    # new model must beat production by this much

monitoring:
  drift_threshold: 0.15         # Evidently drift score that triggers retraining
  check_window_days: 7          # how many days of inference data to check
```

---

## DVC Pipeline (dvc.yaml)

Stages run in order. Each stage has explicit deps and outs for caching.

```
ingest → validate → featurize → train → evaluate → promote
```

- `ingest`: downloads raw data, saves to `data/raw/adult.parquet`
- `validate`: runs Great Expectations suite, exits 1 on failure
- `featurize`: feature engineering, writes to `data/processed/` and DuckDB
- `train`: trains XGBoost, logs to MLflow, registers model as Staging
- `evaluate`: compares Staging vs Production model AUC on hold-out set
- `promote`: if improvement >= threshold, promotes Staging → Production

Run the full pipeline:
```bash
dvc repro
```

---

## MLflow Setup

Run locally — no remote server needed for dev:

```bash
mlflow ui --port 5000
# visit http://localhost:5000
```

Model registry stages used:
- `None` → freshly logged run
- `Staging` → auto-registered after training
- `Production` → promoted after evaluation passes
- `Archived` → old Production model after promotion

The `MLFLOW_TRACKING_URI` env var controls where runs are stored.
Default: `./mlruns` (local filesystem).

---

## Great Expectations Suite

Suite name: `adult_income_suite`

Expectations to implement:
- `expect_table_columns_to_match_ordered_list` — exact schema check
- `expect_column_values_to_not_be_null` — for all critical columns
- `expect_column_values_to_be_in_set` — for categoricals (workclass, education, etc.)
- `expect_column_values_to_be_between` — age (17–90), hours-per-week (1–99)
- `expect_column_proportion_of_unique_values_to_be_between` — for ID-like cols
- Custom: class imbalance check — positive class must be between 20%–40%

Validation checkpoint must:
1. Run against `data/raw/adult.parquet`
2. Generate an HTML report to `great_expectations/uncommitted/data_docs/`
3. Return exit code 1 if any expectation fails (blocks DVC pipeline)

---

## Feature Store (DuckDB)

Tables:
- `features_train` — training split features
- `features_test` — test split features
- `features_inference` — inference logs with timestamp (append-only)

Schema for inference log (used by Evidently for drift detection):
```sql
CREATE TABLE features_inference (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age INTEGER,
    workclass VARCHAR,
    education VARCHAR,
    education_num INTEGER,
    marital_status VARCHAR,
    occupation VARCHAR,
    relationship VARCHAR,
    race VARCHAR,
    sex VARCHAR,
    capital_gain INTEGER,
    capital_loss INTEGER,
    hours_per_week INTEGER,
    native_country VARCHAR,
    prediction INTEGER,
    prediction_proba FLOAT,
    model_version VARCHAR
);
```

---

## FastAPI Serving (serving/main.py)

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Production model inference |
| POST | `/shadow` | Both models, returns Production result, logs both |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Liveness check |
| GET | `/model-info` | Current Production model version + metadata |

### Shadow Deployment Logic

`/shadow` must:
1. Run Production model → get prediction P
2. Run Staging model → get prediction S
3. Log both to `features_inference` with `model_version` field
4. Return P to caller (Staging result is never shown to user)
5. Track divergence rate: count(P != S) / count(total) in Prometheus gauge

### Prometheus Metrics to Export

```python
# All of these must be in serving/metrics.py
REQUEST_COUNT         # Counter, labels: endpoint, status_code
REQUEST_LATENCY       # Histogram, labels: endpoint
PREDICTION_DISTRIBUTION  # Counter, labels: prediction (0 or 1)
SHADOW_DIVERGENCE_RATE   # Gauge: rolling % where prod != shadow
DRIFT_SCORE           # Gauge: latest Evidently drift score
MODEL_VERSION_INFO    # Info metric: current production model version
```

---

## Drift Detection & Self-Healing (monitoring/drift_detector.py)

### What to check
Use Evidently `DataDriftPreset` comparing:
- **Reference:** `data/reference/` snapshot (saved at training time, tracked by DVC)
- **Current:** last N days from `features_inference` table in DuckDB

### Report output
- HTML report saved to `monitoring/reports/drift_YYYY-MM-DD.html`
- JSON summary saved to `monitoring/reports/drift_YYYY-MM-DD.json`

### Self-healing trigger
```python
# In monitoring/alerting.py
if drift_score > params['monitoring']['drift_threshold']:
    # 1. Write drift_flag.txt with score + timestamp
    # 2. Send GitHub repository_dispatch event:
    #    event_type: "drift-detected"
    #    payload: { drift_score, report_path, triggered_at }
    # This kicks off retrain.yml workflow automatically
```

GitHub token for dispatch: `GITHUB_TOKEN` env var (set in Actions secrets).

---

## GitHub Actions Workflows

### 1. pr-validation.yml
**Trigger:** `pull_request` to `main`
**Steps:**
1. Checkout + setup Python
2. Install dependencies
3. Run `pytest tests/` 
4. Run `dvc repro validate` (Great Expectations only)
5. Build Docker image (no push)

### 2. retrain.yml
**Trigger:** `workflow_dispatch`, `repository_dispatch` (drift event), weekly cron (`0 2 * * 1`)
**Steps:**
1. Checkout + DVC pull
2. `dvc repro` (full pipeline)
3. If promotion succeeded: build + push Docker image to `ghcr.io`
4. Post summary as GitHub Actions job summary (AUC before/after, drift score, promotion decision)
5. If promotion failed: open GitHub Issue with metrics diff

### 3. drift-monitor.yml
**Trigger:** Daily cron (`0 6 * * *`)
**Steps:**
1. Checkout
2. Run `python monitoring/drift_detector.py`
3. Upload HTML report as Actions artifact
4. If drift detected: post GitHub Issue comment with score + link to report

---

## Docker Compose Services

```yaml
services:
  api:          # FastAPI on :8000
  mlflow:       # MLflow UI on :5000
  prometheus:   # Scrapes :8000/metrics, stores locally
  grafana:      # Reads from Prometheus, dashboards on :3000
```

Grafana provisioning:
- Auto-provision Prometheus datasource via `grafana/provisioning/datasources/`
- Auto-load dashboard JSON from `grafana/provisioning/dashboards/`
- Dashboard must show: request rate, latency p95, drift score, shadow divergence rate, prediction distribution

---

## Environment Variables

```bash
# .env (copy from .env.example, never commit)
MLFLOW_TRACKING_URI=./mlruns
DRIFT_THRESHOLD=0.15
GITHUB_TOKEN=your_token_here
GITHUB_REPO=your-username/mlops-pipeline
MODEL_NAME=adult_income_classifier
SHADOW_MODE=true               # set false to disable shadow deployment
```

---

## Development Commands

```bash
# Setup
pip install -r requirements.txt
dvc init

# Run full pipeline
dvc repro

# Run individual stages
python pipelines/ingest.py
python pipelines/validate.py
python pipelines/featurize.py
python pipelines/train.py
python pipelines/evaluate.py
python pipelines/promote.py

# Serve
docker-compose up

# Manual drift check
python monitoring/drift_detector.py

# Tests
pytest tests/ -v

# MLflow UI
mlflow ui --port 5000

# Simulate drift (for demo)
python scripts/simulate_drift.py --noise-factor 0.3
```

---

## Implementation Order

Build in this sequence. Do not skip ahead.

1. `requirements.txt` + `params.yaml` + `.env.example`
2. `pipelines/ingest.py` + `pipelines/validate.py` (Great Expectations suite)
3. `feature_store/` (DuckDB schema + read/write API)
4. `pipelines/featurize.py` + `dvc.yaml`
5. `pipelines/train.py` (MLflow logging + model registry)
6. `pipelines/evaluate.py` + `pipelines/promote.py`
7. `serving/` (FastAPI + shadow + Prometheus metrics)
8. `serving/Dockerfile` + `docker-compose.yml` (with Prometheus + Grafana)
9. `monitoring/drift_detector.py` + `monitoring/alerting.py`
10. `.github/workflows/` (all three workflows)
11. `tests/` (pytest for each layer)
12. `scripts/simulate_drift.py` (demo helper)
13. `README.md` with Mermaid architecture diagram

---

## Code Quality Rules

- All functions must have type hints and docstrings
- No hardcoded values — everything goes in `params.yaml` or `.env`
- Every script must be runnable standalone (`if __name__ == "__main__"`)
- Log with Python `logging` module, not `print()`
- Use `pathlib.Path` for all file paths, never string concatenation
- DVC stages must explicitly declare all `deps` and `outs`
- MLflow runs must log: all params, all metrics, confusion matrix artifact, feature importance plot

---

## Key Design Decisions

- **No Kubernetes** — Docker Compose is sufficient and easier to demo locally
- **No cloud deployment** — Render free tier optional; local is fine for portfolio
- **DuckDB over Feast** — same concept, zero infra overhead
- **Local MLflow** — no remote tracking server needed; `./mlruns` is fine
- **Self-hosted Grafana** — avoids Grafana Cloud 14-day retention concern
- **XGBoost kept simple** — resist the urge to add complexity to the model itself

---

## Demo Script (for interviews/README)

The README must show this loop end-to-end:

```
1. docker-compose up            ← bring up entire stack
2. dvc repro                    ← train initial model, promote to Production  
3. curl /predict                ← show live inference
4. curl /shadow                 ← show shadow deployment divergence
5. python simulate_drift.py     ← inject synthetic drift into inference data
6. python drift_detector.py     ← Evidently detects drift, fires GitHub dispatch
7. retrain.yml runs             ← automatic retraining triggered
8. new model promoted           ← AUC improves, Production updated
9. Grafana dashboard            ← drift score drops back to baseline
```

Screenshot or record step 5–9. This is the money shot for the portfolio.
