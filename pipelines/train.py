"""Train XGBoost model, log to MLflow, register as Staging in model registry."""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
TRAIN_PATH = Path(__file__).parent.parent / "data" / "processed" / "train.parquet"
TEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "test.parquet"
REFERENCE_DIR = Path(__file__).parent.parent / "data" / "reference"

FEATURE_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week", "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country",
]
TARGET_COL = "income"


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y, encoding categoricals."""
    cat_cols = [c for c in FEATURE_COLS if df[c].dtype == object]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """Generate a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_feature_importance(model: XGBClassifier, feature_names: list[str]) -> plt.Figure:
    """Generate a horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importances")
    plt.tight_layout()
    return fig


def save_reference_snapshot(df: pd.DataFrame, params: dict) -> None:
    """Save a random sample of training data as the Evidently reference set."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    n = params["data"]["reference_sample_size"]
    sample = df.sample(n=min(n, len(df)), random_state=params["data"]["random_state"])
    out = REFERENCE_DIR / "reference.parquet"
    sample.to_parquet(out, index=False)
    logger.info(f"Saved reference snapshot ({len(sample)} rows) to {out}")


def train() -> str:
    """Train XGBoost, log everything to MLflow, register model as Staging.

    Returns:
        The MLflow run ID.
    """
    params = load_params()
    model_params = params["model"]
    model_name = "adult_income_classifier"

    logger.info("Loading training and test data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train = prepare_data(train_df.copy())
    X_test, y_test = prepare_data(test_df.copy())

    # Align test columns to training encoding (handle unseen categories)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("adult_income_classifier")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Log all params
        mlflow.log_params({
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "learning_rate": model_params["learning_rate"],
            "subsample": model_params["subsample"],
            "colsample_bytree": model_params["colsample_bytree"],
            "eval_metric": model_params["eval_metric"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        model = XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            eval_metric=model_params["eval_metric"],
            use_label_encoder=False,
            random_state=params["data"]["random_state"],
            enable_categorical=False,
        )

        logger.info("Training XGBoost model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        accuracy = (y_pred == y_test).mean()

        logger.info(f"Test AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")
        mlflow.log_metrics({"test_auc": auc, "test_accuracy": accuracy})

        # Confusion matrix artifact
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close(cm_fig)

        # Feature importance artifact
        fi_fig = plot_feature_importance(model, FEATURE_COLS)
        mlflow.log_figure(fi_fig, "feature_importance.png")
        plt.close(fi_fig)

        # Log and register model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        logger.info(f"Model logged and registered as '{model_name}'")

    # Transition latest version to Staging
    client = MlflowClient(tracking_uri="mlruns")
    versions = client.get_latest_versions(model_name, stages=["None"])
    if versions:
        version = versions[-1].version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(f"Model v{version} transitioned to Staging")

    # Save reference snapshot for Evidently drift detection
    save_reference_snapshot(train_df, params)

    return run_id


if __name__ == "__main__":
    run_id = train()
    logger.info(f"Training complete. Run ID: {run_id}")
