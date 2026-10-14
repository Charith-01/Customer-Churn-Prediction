"""Tests for Phase 6 metrics, calibration, thresholding, and persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation import (
    calibration_diagnostics,
    choose_calibration_method,
    compute_classification_metrics,
    compute_threshold_metrics,
    generate_nested_calibration_oof,
    positive_class_probabilities,
    predictions_at_threshold,
    robustness_summary,
    select_operating_threshold,
)
from src.modeling import build_model_pipelines

EXPECTED_SPLIT_ARTIFACT_SHA256 = (
    "4325cbfeb6523b8594174ef19926790c08114ca360a78dbe2701378e4fd0a235"
)


@pytest.fixture
def known_binary_case() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9])


def test_precision_calculation(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p)["precision"] == pytest.approx(2 / 3)


def test_recall_calculation(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p)["recall"] == pytest.approx(1.0)


def test_f1_calculation(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p)["f1"] == pytest.approx(0.8)


def test_roc_auc_calculation(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p)["roc_auc"] == pytest.approx(
        roc_auc_score(y, p)
    )


def test_average_precision_calculation(
    known_binary_case: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p)["average_precision"] == pytest.approx(
        average_precision_score(y, p)
    )


def test_confusion_matrix_values(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    metrics = compute_classification_metrics(y, p)
    assert (
        metrics["true_negative"],
        metrics["false_positive"],
        metrics["false_negative"],
        metrics["true_positive"],
    ) == (1, 1, 0, 2)


def test_threshold_behavior_includes_equal_scores() -> None:
    probabilities = np.array([0.49, 0.50, 0.51])
    np.testing.assert_array_equal(
        predictions_at_threshold(probabilities, 0.50), np.array([0, 1, 1])
    )


def test_contact_constraint_keeps_ties_together() -> None:
    y = np.array([1, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.8, 0.2, 0.1])
    selected = select_operating_threshold(y, probabilities, capacity=0.40)
    assert selected["threshold"] == pytest.approx(0.9)
    assert selected["predicted_positive_count"] == 1
    assert selected["predicted_positive_rate"] <= 0.40


def test_calibration_diagnostics_match_known_brier_score() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    summary, bins = calibration_diagnostics(y, probabilities, n_bins=2)
    assert summary["brier_score"] == pytest.approx(0.025)
    assert bins["sample_count"].sum() == 4
    assert bins["bin_number"].is_unique


@pytest.mark.parametrize(
    "bad_probabilities",
    [[0.1, np.nan, 0.8, 0.9], [0.1, -0.1, 0.8, 0.9], [0.1, 1.1, 0.8, 0.9]],
)
def test_missing_or_invalid_probabilities_raise(bad_probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        compute_classification_metrics([0, 0, 1, 1], bad_probabilities)


class _ReverseClassEstimator(BaseEstimator):
    classes_ = np.array([1, 0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = np.asarray(X["score"], dtype=float)
        return np.column_stack([positive, 1 - positive])


def test_positive_class_probability_uses_class_order() -> None:
    X = pd.DataFrame({"score": [0.2, 0.8]})
    probabilities = positive_class_probabilities(_ReverseClassEstimator(), X)
    np.testing.assert_allclose(probabilities, [0.2, 0.8])


def test_model_serialization_and_prediction_consistency(tmp_path: Path) -> None:
    X = pd.DataFrame({"x": [-2, -1, 0, 1, 2, 3]})
    y = np.array([0, 0, 0, 1, 1, 1])
    fitted = LogisticRegression(random_state=42).fit(X, y)
    path = tmp_path / "evaluation_model.joblib"
    before = positive_class_probabilities(fitted, X)
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(before, positive_class_probabilities(loaded, X))


@pytest.mark.parametrize("forbidden", ["customer_id", "churn"])
def test_target_and_identifier_leakage_are_rejected(forbidden: str) -> None:
    X = pd.DataFrame(
        {
            "age": [20, 30],
            "tenure_months": [1, 2],
            "monthly_charges": [20.0, 30.0],
            "total_charges": [20.0, 60.0],
            "complaints_last_12m": [0, 1],
            "late_payments_last_12m": [0, 1],
            "satisfaction_score": [8.0, 5.0],
            "monthly_usage_gb": [10.0, 20.0],
            "discount_pct": [0.0, 5.0],
            "contract_type": ["One year", "Month-to-month"],
            "internet_service": ["DSL", "Fiber optic"],
            "region": ["Western", "Central"],
            "payment_method": ["Credit card", "Electronic check"],
            "paperless_billing": ["No", "Yes"],
            "dependents": ["Yes", "No"],
            "support_plan": ["Yes", "No"],
            "tech_support": ["Yes", "No"],
            "senior_citizen": [0, 0],
            forbidden: [1, 2],
        }
    )
    with pytest.raises(ValueError, match="cannot be model predictors|removed before"):
        build_model_pipelines(include_dummy=False)["random_forest"].fit(X, [0, 1])


def test_fixed_train_test_split_is_unchanged() -> None:
    path = Path("data/processed/02_split_indices.json")
    if not path.exists():
        pytest.skip("Local split artifact is intentionally not committed.")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SPLIT_ARTIFACT_SHA256


def test_metrics_are_reproducible(known_binary_case: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = known_binary_case
    assert compute_classification_metrics(y, p) == compute_classification_metrics(y, p)


def test_selected_threshold_is_used_during_inference() -> None:
    probabilities = np.array([0.72, 0.73, 0.74])
    np.testing.assert_array_equal(
        predictions_at_threshold(probabilities, threshold=0.73), [0, 1, 1]
    )


def test_threshold_table_reports_contact_volume() -> None:
    table = compute_threshold_metrics(
        [0, 1, 1, 0], [0.1, 0.4, 0.8, 0.9], thresholds=[0.5, 0.8]
    )
    assert table.loc[0, "predicted_positive_count"] == 2
    assert table.loc[0, "predicted_positive_rate"] == pytest.approx(0.5)


def test_nested_calibration_oof_has_one_score_per_row_and_method() -> None:
    X = pd.DataFrame({"x": np.linspace(-3, 3, 60)})
    y = pd.Series(np.resize([0, 0, 1, 1, 1], 60), index=X.index)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    result = generate_nested_calibration_oof(
        LogisticRegression(random_state=42),
        X,
        y,
        splits,
        methods=("uncalibrated", "sigmoid"),
        inner_splits=2,
    )
    assert len(result) == 120
    assert not result.duplicated(["dataset_row_index", "method"]).any()
    assert result["probability"].between(0, 1).all()


def test_calibration_choice_requires_material_improvement() -> None:
    comparison = pd.DataFrame(
        {
            "method": ["uncalibrated", "sigmoid"],
            "brier_score": [0.10, 0.099],
            "log_loss": [0.35, 0.34],
            "average_precision": [0.90, 0.90],
            "roc_auc": [0.85, 0.85],
            "expected_calibration_error": [0.08, 0.07],
        }
    )
    assert choose_calibration_method(comparison)["method"] == "uncalibrated"


def test_robustness_summary_reports_fold_range() -> None:
    folds = pd.DataFrame(
        {
            "precision": [0.8, 0.9],
            "recall": [0.7, 0.8],
            "roc_auc": [0.75, 0.85],
            "pr_auc": [0.80, 0.90],
        }
    )
    summary = robustness_summary(folds).set_index("metric")
    assert summary.loc["roc_auc", "mean"] == pytest.approx(0.80)
    assert summary.loc["roc_auc", "minimum"] == pytest.approx(0.75)
    assert summary.loc["roc_auc", "maximum"] == pytest.approx(0.85)
