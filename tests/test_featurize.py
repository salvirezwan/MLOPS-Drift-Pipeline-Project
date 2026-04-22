"""Tests for pipelines/featurize.py and feature_store/feature_store.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.featurize import CATEGORICAL_COLS, NUMERIC_COLS, engineer_features


def _make_raw_df(n: int = 50) -> pd.DataFrame:
    """Return a minimal raw DataFrame as it comes out of ingest.py."""
    return pd.DataFrame({
        "age": [25] * n,
        "workclass": ["Private"] * n,
        "fnlwgt": [200000] * n,
        "education": ["Bachelors"] * n,
        "education_num": [13] * n,
        "marital_status": ["Never-married"] * n,
        "occupation": ["Exec-managerial"] * n,
        "relationship": ["Not-in-family"] * n,
        "race": ["White"] * n,
        "sex": ["Male"] * n,
        "capital_gain": [0] * n,
        "capital_loss": [0] * n,
        "hours_per_week": [40] * n,
        "native_country": ["United-States"] * n,
        "income": [0] * n,
    })


class TestEngineerFeatures:
    def test_output_shape_unchanged(self):
        df = _make_raw_df(50)
        out = engineer_features(df)
        assert out.shape == df.shape

    def test_categoricals_are_strings(self):
        df = _make_raw_df(50)
        out = engineer_features(df)
        for col in CATEGORICAL_COLS:
            if col in out.columns:
                assert out[col].dtype == object, f"{col} should be object dtype"

    def test_numerics_are_int(self):
        df = _make_raw_df(50)
        out = engineer_features(df)
        for col in NUMERIC_COLS:
            if col in out.columns:
                assert out[col].dtype in ("int32", "int64"), f"{col} should be int dtype"

    def test_nulls_filled_in_categoricals(self):
        df = _make_raw_df(50)
        df.loc[:5, "workclass"] = None
        out = engineer_features(df)
        assert out["workclass"].isnull().sum() == 0

    def test_nulls_filled_in_numerics(self):
        df = _make_raw_df(50)
        df["capital_gain"] = df["capital_gain"].astype(float)
        df.loc[:5, "capital_gain"] = None
        out = engineer_features(df)
        assert out["capital_gain"].isnull().sum() == 0

    def test_does_not_mutate_input(self):
        df = _make_raw_df(50)
        original_dtypes = df.dtypes.copy()
        engineer_features(df)
        assert df.dtypes.equals(original_dtypes)

    def test_categorical_dtype_columns_handled(self):
        """Pandas Categorical dtype columns should not raise on fillna."""
        df = _make_raw_df(50)
        df["workclass"] = pd.Categorical(df["workclass"])
        out = engineer_features(df)
        assert out["workclass"].dtype == object


class TestFeatureStore:
    def test_write_and_read_roundtrip(self, tmp_path, monkeypatch):
        """write_features then read_features returns the same data."""
        import feature_store.feature_store as fs
        monkeypatch.setattr(fs, "DB_PATH", tmp_path / "test.db")

        df = _make_raw_df(30)
        df.insert(0, "id", range(30))
        fs.write_features(df.drop(columns=["id"]), "features_train", overwrite=True)
        result = fs.read_features("features_train")
        assert len(result) == 30
        assert "age" in result.columns

    def test_overwrite_clears_existing(self, tmp_path, monkeypatch):
        import feature_store.feature_store as fs
        monkeypatch.setattr(fs, "DB_PATH", tmp_path / "test.db")

        df = _make_raw_df(20)
        fs.write_features(df, "features_train", overwrite=True)
        fs.write_features(df, "features_train", overwrite=True)
        result = fs.read_features("features_train")
        assert len(result) == 20  # not 40

    def test_log_inference_batch(self, tmp_path, monkeypatch):
        import feature_store.feature_store as fs
        monkeypatch.setattr(fs, "DB_PATH", tmp_path / "test.db")

        df = _make_raw_df(10)
        df["prediction"] = 0
        df["prediction_proba"] = 0.2
        fs.log_inference_batch(df, model_version="1")
        assert fs.get_inference_count() == 10

    def test_read_inference_window_empty_on_fresh_db(self, tmp_path, monkeypatch):
        import feature_store.feature_store as fs
        monkeypatch.setattr(fs, "DB_PATH", tmp_path / "test.db")

        result = fs.read_inference_window(days=7)
        assert result.empty
