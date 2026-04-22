"""Tests for monitoring/drift_detector.py and monitoring/alerting.py."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_reference_df(n: int = 200) -> pd.DataFrame:
    """Minimal reference dataset matching feature store schema."""
    return pd.DataFrame({
        "age": [30] * n,
        "workclass": ["Private"] * n,
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
    })


def _make_drifted_df(n: int = 200) -> pd.DataFrame:
    """Dataset with obvious distribution shift."""
    import numpy as np
    rng = pd.np.random if hasattr(pd, "np") else __import__("numpy").random
    return pd.DataFrame({
        "age": rng.randint(60, 90, n).tolist(),           # very different from ref (30)
        "workclass": ["Self-emp-inc"] * n,                # different from ref
        "education": ["Doctorate"] * n,
        "education_num": [16] * n,
        "marital_status": ["Divorced"] * n,
        "occupation": ["Farming-fishing"] * n,
        "relationship": ["Wife"] * n,
        "race": ["Black"] * n,
        "sex": ["Female"] * n,
        "capital_gain": [99999] * n,
        "capital_loss": [4000] * n,
        "hours_per_week": [1] * n,
        "native_country": ["Mexico"] * n,
    })


class TestExtractDriftScore:
    def test_returns_zero_on_empty_metrics(self):
        from monitoring.drift_detector import extract_drift_score
        mock_report = type("R", (), {"as_dict": lambda self: {"metrics": []}})()
        assert extract_drift_score(mock_report) == 0.0

    def test_extracts_score_from_dataset_drift_metric(self):
        from monitoring.drift_detector import extract_drift_score
        mock_report = type("R", (), {
            "as_dict": lambda self: {
                "metrics": [
                    {
                        "metric": "DatasetDriftMetric",
                        "result": {"share_of_drifted_columns": 0.42},
                    }
                ]
            }
        })()
        assert abs(extract_drift_score(mock_report) - 0.42) < 0.001


class TestLoadReference:
    def test_raises_when_reference_missing(self, tmp_path, monkeypatch):
        import monitoring.drift_detector as dd
        monkeypatch.setattr(dd, "REFERENCE_PATH", tmp_path / "nonexistent.parquet")
        with pytest.raises(FileNotFoundError):
            dd.load_reference()

    def test_loads_parquet_and_filters_feature_cols(self, tmp_path, monkeypatch):
        import monitoring.drift_detector as dd
        ref = _make_reference_df(100)
        ref["extra_col"] = 99  # should be dropped
        path = tmp_path / "reference.parquet"
        ref.to_parquet(path)
        monkeypatch.setattr(dd, "REFERENCE_PATH", path)
        result = dd.load_reference()
        assert "extra_col" not in result.columns
        assert len(result) == 100


class TestRunDriftDetection:
    def test_returns_no_drift_when_no_inference_data(self, tmp_path, monkeypatch):
        import monitoring.drift_detector as dd

        ref = _make_reference_df(200)
        ref_path = tmp_path / "reference.parquet"
        ref.to_parquet(ref_path)
        monkeypatch.setattr(dd, "REFERENCE_PATH", ref_path)
        monkeypatch.setattr(dd, "REPORTS_DIR", tmp_path / "reports")

        with patch("monitoring.drift_detector.read_inference_window", return_value=pd.DataFrame()):
            result = dd.run_drift_detection()

        assert result["drift_detected"] is False
        assert result["current_rows"] == 0

    def test_detects_drift_on_shifted_data(self, tmp_path, monkeypatch):
        import numpy as np
        import monitoring.drift_detector as dd

        ref = _make_reference_df(300)
        ref_path = tmp_path / "reference.parquet"
        ref.to_parquet(ref_path)
        monkeypatch.setattr(dd, "REFERENCE_PATH", ref_path)
        monkeypatch.setattr(dd, "REPORTS_DIR", tmp_path / "reports")

        current = pd.DataFrame({
            "age": np.random.randint(60, 90, 300).tolist(),
            "workclass": ["Self-emp-inc"] * 300,
            "education": ["Doctorate"] * 300,
            "education_num": [16] * 300,
            "marital_status": ["Divorced"] * 300,
            "occupation": ["Farming-fishing"] * 300,
            "relationship": ["Wife"] * 300,
            "race": ["Black"] * 300,
            "sex": ["Female"] * 300,
            "capital_gain": [99999] * 300,
            "capital_loss": [4000] * 300,
            "hours_per_week": [1] * 300,
            "native_country": ["Mexico"] * 300,
        })

        with patch("monitoring.drift_detector.read_inference_window", return_value=current):
            result = dd.run_drift_detection()

        assert result["drift_score"] > 0.15
        assert result["drift_detected"] is True
        assert Path(result["report_html_path"]).exists()
        assert Path(result["report_json_path"]).exists()

    def test_no_drift_on_identical_data(self, tmp_path, monkeypatch):
        import monitoring.drift_detector as dd

        ref = _make_reference_df(300)
        ref_path = tmp_path / "reference.parquet"
        ref.to_parquet(ref_path)
        monkeypatch.setattr(dd, "REFERENCE_PATH", ref_path)
        monkeypatch.setattr(dd, "REPORTS_DIR", tmp_path / "reports")

        with patch("monitoring.drift_detector.read_inference_window", return_value=ref.copy()):
            result = dd.run_drift_detection()

        assert result["drift_score"] < 0.15


class TestAlerting:
    def test_no_alert_when_below_threshold(self, tmp_path, monkeypatch, caplog):
        import monitoring.alerting as al
        monkeypatch.setattr(al, "FLAG_PATH", tmp_path / "drift_flag.txt")
        al.alert({"drift_score": 0.05, "drift_detected": False, "report_html_path": ""})
        assert not (tmp_path / "drift_flag.txt").exists()

    def test_writes_flag_file_on_drift(self, tmp_path, monkeypatch):
        import monitoring.alerting as al
        flag_path = tmp_path / "drift_flag.txt"
        monkeypatch.setattr(al, "FLAG_PATH", flag_path)
        with patch("monitoring.alerting.send_github_dispatch", return_value=False):
            al.alert({"drift_score": 0.3, "drift_detected": True, "report_html_path": "report.html"})
        assert flag_path.exists()
        content = flag_path.read_text()
        assert "drift_score=0.3" in content

    def test_flag_file_contains_timestamp(self, tmp_path, monkeypatch):
        import monitoring.alerting as al
        flag_path = tmp_path / "drift_flag.txt"
        monkeypatch.setattr(al, "FLAG_PATH", flag_path)
        with patch("monitoring.alerting.send_github_dispatch", return_value=False):
            al.alert({"drift_score": 0.3, "drift_detected": True, "report_html_path": ""})
        content = flag_path.read_text()
        assert "triggered_at=" in content

    def test_github_dispatch_skipped_without_token(self, tmp_path, monkeypatch):
        import monitoring.alerting as al
        monkeypatch.setattr(al, "FLAG_PATH", tmp_path / "flag.txt")
        monkeypatch.setenv("GITHUB_TOKEN", "")
        result = al.send_github_dispatch(0.3, "report.html")
        assert result is False
