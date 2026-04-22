"""Tests for pipelines/validate.py — Great Expectations suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.validate import build_suite, validate


def _make_valid_df(n: int = 100) -> pd.DataFrame:
    """Return a minimal valid DataFrame matching the UCI Adult schema."""
    df = pd.DataFrame({
        "age": [30] * n,
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
    # Set 30% positive class to pass imbalance check
    df.loc[: int(n * 0.3) - 1, "income"] = 1
    return df


def _get_success(result) -> bool:
    """Extract success bool from either a GE result object or a plain dict."""
    if isinstance(result, dict):
        return bool(result.get("success"))
    return bool(result.success)


def _get_expectation_type(result) -> str:
    """Extract expectation_type from either a GE result object or a plain dict."""
    if isinstance(result, dict):
        return result.get("expectation_type", "")
    try:
        return result.expectation_config.expectation_type
    except AttributeError:
        return ""


def _get_column(result) -> str:
    """Extract the column kwarg from a GE result object."""
    if isinstance(result, dict):
        return ""
    try:
        return result.expectation_config.kwargs.get("column", "")
    except AttributeError:
        return ""


class TestBuildSuite:
    def test_all_pass_on_valid_data(self):
        import great_expectations as ge
        df = _make_valid_df(200)
        gdf = ge.from_pandas(df)
        results = build_suite(gdf)
        failed = [r for r in results if not _get_success(r)]
        assert failed == [], f"Unexpected failures: {[_get_expectation_type(r) for r in failed]}"

    def test_fails_on_wrong_column_order(self):
        import great_expectations as ge
        df = _make_valid_df(100)
        df = df[list(reversed(df.columns))]
        gdf = ge.from_pandas(df)
        results = build_suite(gdf)
        schema_result = next(
            r for r in results
            if _get_expectation_type(r) == "expect_table_columns_to_match_ordered_list"
        )
        assert not _get_success(schema_result)

    def test_fails_on_age_out_of_range(self):
        import great_expectations as ge
        df = _make_valid_df(100)
        df["age"] = 5  # below minimum of 17
        gdf = ge.from_pandas(df)
        results = build_suite(gdf)
        age_result = next(
            r for r in results
            if _get_expectation_type(r) == "expect_column_values_to_be_between"
            and _get_column(r) == "age"
        )
        assert not _get_success(age_result)

    def test_fails_on_class_imbalance(self):
        import great_expectations as ge
        df = _make_valid_df(100)
        df["income"] = 0  # 0% positive class — fails 20%-40% check
        gdf = ge.from_pandas(df)
        results = build_suite(gdf)
        imbalance_result = next(
            r for r in results
            if _get_expectation_type(r) == "custom_class_imbalance_check"
        )
        assert not _get_success(imbalance_result)

    def test_fails_on_null_income(self):
        import great_expectations as ge
        df = _make_valid_df(100)
        df.loc[:10, "income"] = None
        gdf = ge.from_pandas(df)
        results = build_suite(gdf)
        null_results = [
            r for r in results
            if _get_expectation_type(r) == "expect_column_values_to_not_be_null"
            and _get_column(r) == "income"
        ]
        assert any(not _get_success(r) for r in null_results)


class TestValidateIntegration:
    def test_validate_passes_on_real_data(self, tmp_path, monkeypatch):
        raw_path = Path(__file__).parent.parent / "data" / "raw" / "adult.parquet"
        if not raw_path.exists():
            pytest.skip("Raw data not present — run ingest.py first")
        import pipelines.validate as val_module
        monkeypatch.setattr(val_module, "DOCS_DIR", tmp_path)
        assert validate() is True

    def test_validate_returns_false_when_no_data(self, tmp_path, monkeypatch):
        import pipelines.validate as val_module
        monkeypatch.setattr(val_module, "RAW_PATH", tmp_path / "nonexistent.parquet")
        assert validate() is False
