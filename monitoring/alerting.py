"""Drift alerting — write drift_flag.txt and fire GitHub repository_dispatch.

Called by drift_detector.py when drift_score exceeds the threshold.
Triggers retrain.yml workflow automatically via GitHub Actions.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
FLAG_PATH = Path(__file__).parent.parent / "drift_flag.txt"
GITHUB_API_URL = "https://api.github.com"


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def write_drift_flag(drift_score: float, report_path: str) -> None:
    """Write drift_flag.txt with score and timestamp so CI can detect it.

    Args:
        drift_score: The Evidently drift score that triggered the alert.
        report_path: Path to the generated drift report.
    """
    triggered_at = datetime.now(timezone.utc).isoformat()
    content = (
        f"drift_score={drift_score}\n"
        f"triggered_at={triggered_at}\n"
        f"report_path={report_path}\n"
    )
    FLAG_PATH.write_text(content)
    logger.info(f"Drift flag written to {FLAG_PATH}")


def send_github_dispatch(drift_score: float, report_path: str) -> bool:
    """Fire a GitHub repository_dispatch event to trigger retrain.yml.

    Args:
        drift_score: The Evidently drift score that triggered the alert.
        report_path: Path to the generated drift report.

    Returns:
        True if the dispatch was accepted (HTTP 204), False otherwise.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GH_REPO")

    if not token or token == "your_token_here":
        logger.warning("GITHUB_TOKEN not set — skipping GitHub dispatch")
        return False
    if not repo or repo == "your-username/mlops-pipeline":
        logger.warning("GH_REPO not set — skipping GitHub dispatch")
        return False

    triggered_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "event_type": "drift-detected",
        "client_payload": {
            "drift_score": drift_score,
            "report_path": report_path,
            "triggered_at": triggered_at,
        },
    }

    url = f"{GITHUB_API_URL}/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    logger.info(f"Sending repository_dispatch to {repo} (event: drift-detected)...")
    response = requests.post(url, headers=headers, json=payload, timeout=10)

    if response.status_code == 204:
        logger.info("GitHub dispatch accepted — retrain.yml will be triggered")
        return True
    else:
        logger.error(
            f"GitHub dispatch failed: HTTP {response.status_code} — {response.text}"
        )
        return False


def alert(drift_result: dict) -> None:
    """Entry point: write flag file and optionally fire GitHub dispatch.

    Args:
        drift_result: Output dict from drift_detector.run_drift_detection().
    """
    params = load_params()
    threshold: float = params["monitoring"]["drift_threshold"]
    drift_score: float = drift_result["drift_score"]
    report_path: str = drift_result.get("report_html_path") or ""

    if not drift_result.get("drift_detected"):
        logger.info(
            f"Drift score {drift_score:.4f} below threshold {threshold} — no alert needed"
        )
        return

    logger.warning(
        f"DRIFT ALERT: score {drift_score:.4f} exceeds threshold {threshold}"
    )

    write_drift_flag(drift_score, report_path)
    sent = send_github_dispatch(drift_score, report_path)

    if not sent:
        logger.info(
            "GitHub dispatch skipped (no token/repo configured). "
            "Drift flag written — trigger retrain manually: "
            "python pipelines/train.py && python pipelines/evaluate.py && python pipelines/promote.py"
        )


if __name__ == "__main__":
    # Allow running standalone with a manual drift score for testing
    import argparse

    parser = argparse.ArgumentParser(description="Send drift alert")
    parser.add_argument("--drift-score", type=float, required=True)
    parser.add_argument("--report-path", type=str, default="")
    args = parser.parse_args()

    params = load_params()
    threshold = params["monitoring"]["drift_threshold"]
    alert({
        "drift_score": args.drift_score,
        "drift_detected": args.drift_score > threshold,
        "report_html_path": args.report_path,
    })
