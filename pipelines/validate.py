"""Run Great Expectations validation against data/raw/adult.parquet.

Exits with code 1 if any expectation fails, blocking the DVC pipeline.
Generates an HTML data docs report.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "adult.parquet"
DOCS_DIR = Path(__file__).parent.parent / "great_expectations" / "uncommitted" / "data_docs"

EXPECTED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

WORKCLASS_VALUES = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
]

EDUCATION_VALUES = [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
    "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
]

MARITAL_STATUS_VALUES = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]

RELATIONSHIP_VALUES = [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried",
]

RACE_VALUES = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]

SEX_VALUES = ["Male", "Female"]


def build_suite(gdf: PandasDataset) -> list[dict]:
    """Apply all expectations to the GE dataset and return results."""
    results = []

    # Schema check
    results.append(gdf.expect_table_columns_to_match_ordered_list(EXPECTED_COLUMNS))

    # Null checks — strict for columns that must never be null
    strict_null_cols = ["age", "education", "income", "hours_per_week", "sex", "education_num"]
    for col in strict_null_cols:
        results.append(gdf.expect_column_values_to_not_be_null(col))

    # Columns with known missingness in UCI dataset (coded as '?') — allow up to 10% null
    nullable_cols = ["workclass", "occupation", "native_country"]
    for col in nullable_cols:
        results.append(gdf.expect_column_values_to_not_be_null(col, mostly=0.90))

    # Categorical value sets
    results.append(gdf.expect_column_values_to_be_in_set("workclass", WORKCLASS_VALUES, mostly=0.98))
    results.append(gdf.expect_column_values_to_be_in_set("education", EDUCATION_VALUES, mostly=0.98))
    results.append(gdf.expect_column_values_to_be_in_set("marital_status", MARITAL_STATUS_VALUES, mostly=0.98))
    results.append(gdf.expect_column_values_to_be_in_set("relationship", RELATIONSHIP_VALUES, mostly=0.98))
    results.append(gdf.expect_column_values_to_be_in_set("race", RACE_VALUES, mostly=0.98))
    results.append(gdf.expect_column_values_to_be_in_set("sex", SEX_VALUES))

    # Numeric range checks
    results.append(gdf.expect_column_values_to_be_between("age", min_value=17, max_value=90))
    results.append(gdf.expect_column_values_to_be_between("hours_per_week", min_value=1, max_value=99))
    results.append(gdf.expect_column_values_to_be_between("education_num", min_value=1, max_value=16))

    # Target column: binary values only
    results.append(gdf.expect_column_values_to_be_in_set("income", [0, 1]))

    # Class imbalance: positive class (income=1) must be 20%–40%
    positive_rate = gdf["income"].mean()
    imbalance_ok = 0.20 <= positive_rate <= 0.40
    results.append({
        "expectation_type": "custom_class_imbalance_check",
        "success": imbalance_ok,
        "result": {"positive_rate": round(float(positive_rate), 4)},
    })
    if imbalance_ok:
        logger.info(f"Class imbalance check passed: positive rate = {positive_rate:.2%}")
    else:
        logger.error(f"Class imbalance check FAILED: positive rate = {positive_rate:.2%} (expected 20%-40%)")

    return results


def save_html_report(gdf: PandasDataset) -> None:
    """Generate and save an HTML validation report to the data_docs directory."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "validation_report.html"

    validation_result = gdf.validate()
    html = ge.render.view.view.DefaultJinjaPageView().render(
        ge.render.renderer.ValidationResultsPageRenderer().render(validation_result)
    )
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved to {report_path}")


def validate() -> bool:
    """Load raw data, run all expectations, return True if all pass."""
    if not RAW_PATH.exists():
        logger.error(f"Raw data not found at {RAW_PATH}. Run ingest.py first.")
        return False

    logger.info(f"Loading data from {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    gdf = ge.from_pandas(df)

    results = build_suite(gdf)

    def _success(r: dict) -> bool:
        return bool(r.get("success"))

    passed = sum(1 for r in results if _success(r))
    total = len(results)
    failed = [r for r in results if not _success(r)]

    logger.info(f"Validation: {passed}/{total} expectations passed")

    for r in failed:
        exp_type = r.get("expectation_type", r.get("expectation_config", {}).get("expectation_type", "unknown"))
        logger.error(f"FAILED: {exp_type} → {r.get('result', {})}")

    # Attempt HTML report (non-blocking)
    try:
        save_html_report(gdf)
    except Exception as e:
        logger.warning(f"Could not save HTML report: {e}")

    return len(failed) == 0


if __name__ == "__main__":
    success = validate()
    if not success:
        logger.error("Validation failed. Blocking pipeline.")
        sys.exit(1)
    logger.info("Validation passed.")
