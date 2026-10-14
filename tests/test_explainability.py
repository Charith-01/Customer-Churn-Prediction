"""Tests for Phase 7 global and customer-level explanations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.evaluation import positive_class_probabilities
from src.explainability import (
    EXPECTED_PREDICTORS,
    anonymized_case_summary,
    assign_confusion_categories,
    calculate_global_permutation_importance,
    file_sha256,
    load_trusted_locked_model,
    local_probability_sensitivity,
    plot_global_importance,
    select_representative_cases,
    summarize_prediction_errors,
    training_reference_values,
    validate_feature_schema,
)
from src.modeling import build_model_pipelines


@pytest.fixture
def explanation_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    rows = 80
    usage = np.linspace(5, 180, rows)
    satisfaction = np.linspace(9.5, 2.0, rows)
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 75, rows),
            "tenure_months": rng.integers(1, 73, rows),
            "monthly_charges": rng.uniform(25, 130, rows),
            "total_charges": rng.uniform(50, 8_000, rows),
            "complaints_last_12m": rng.integers(0, 5, rows),
            "late_payments_last_12m": rng.integers(0, 4, rows),
            "satisfaction_score": satisfaction,
            "monthly_usage_gb": usage,
            "discount_pct": rng.uniform(0, 25, rows),
            "contract_type": np.resize(
                ["Month-to-month", "One year", "Two year"], rows
            ),
            "internet_service": np.resize(["DSL", "Fiber optic", "No"], rows),
            "region": np.resize(["Western", "Central", "Eastern"], rows),
            "payment_method": np.resize(
                ["Credit card", "Electronic check", "Bank transfer"], rows
            ),
            "paperless_billing": np.resize(["Yes", "No"], rows),
            "dependents": np.resize(["No", "Yes"], rows),
            "support_plan": np.resize(["No", "Yes"], rows),
            "tech_support": np.resize(["No", "Yes"], rows),
            "senior_citizen": np.resize([0, 1], rows),
        },
        index=np.arange(1_000, 1_000 + rows),
    )
    X.loc[X.index[[4, 21]], "monthly_charges"] = np.nan
    X.loc[X.index[[7, 33]], "region"] = pd.NA
    y = pd.Series((usage + rng.normal(0, 28, rows) > 75).astype(int), index=X.index)
    return X, y


@pytest.fixture
def fitted_pipeline(explanation_data: tuple[pd.DataFrame, pd.Series]):
    X, y = explanation_data
    pipeline = build_model_pipelines(include_dummy=False)["logistic_regression"]
    return pipeline.fit(X, y)


def test_global_importance_calculation(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    importance = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=2
    )
    assert len(importance) == len(EXPECTED_PREDICTORS)
    assert np.isfinite(importance["importance_mean"]).all()


def test_global_importance_feature_name_alignment(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    importance = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=2
    )
    assert set(importance["feature"]) == set(X.columns) == set(EXPECTED_PREDICTORS)


def test_global_importance_is_deterministic(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    first = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=2, random_state=42
    )
    second = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=2, random_state=42
    )
    pd.testing.assert_frame_equal(first, second)


def test_global_importance_handles_missing_inputs(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    assert X.isna().any().any()
    result = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=1
    )
    assert len(result) == 18


def test_global_importance_handles_categorical_inputs(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    result = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=1
    )
    assert "contract_type" in set(result["feature"])


def test_individual_sensitivity_generation(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, _ = explanation_data
    references = training_reference_values(X)
    explanation = local_probability_sensitivity(
        fitted_pipeline, X.iloc[[0]], references
    )
    assert len(explanation) == 18
    assert set(explanation["feature"]) == set(EXPECTED_PREDICTORS)
    assert (explanation["absolute_effect"] >= 0).all()


class _ReverseClassEstimator(BaseEstimator):
    classes_ = np.array([1, 0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = np.asarray(X["score"], dtype=float)
        return np.column_stack([positive, 1 - positive])


def test_correct_churn_positive_probability_selection() -> None:
    X = pd.DataFrame({"score": [0.15, 0.85]})
    np.testing.assert_allclose(
        positive_class_probabilities(_ReverseClassEstimator(), X), [0.15, 0.85]
    )


def test_explanation_probabilities_remain_valid(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, _ = explanation_data
    explanation = local_probability_sensitivity(
        fitted_pipeline, X.iloc[[2]], training_reference_values(X)
    )
    assert explanation["original_probability"].between(0, 1).all()
    assert explanation["reference_probability"].between(0, 1).all()


def test_selected_customers_belong_to_permitted_partition(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    probabilities = positive_class_probabilities(fitted_pipeline, X)
    selected = select_representative_cases(
        X, y, probabilities, 0.5, permitted_indices=X.index
    )
    assert set(selected["_row_index"]).issubset(set(X.index))
    with pytest.raises(ValueError, match="outside the permitted partition"):
        select_representative_cases(
            X, y, probabilities, 0.5, permitted_indices=X.index[:-1]
        )


@pytest.mark.parametrize("forbidden", ["customer_id", "churn"])
def test_identifier_and_target_are_excluded(
    explanation_data: tuple[pd.DataFrame, pd.Series], forbidden: str
) -> None:
    X, _ = explanation_data
    invalid = X.copy()
    invalid[forbidden] = np.arange(len(invalid))
    with pytest.raises(ValueError, match="cannot be explanation inputs"):
        validate_feature_schema(invalid)


def test_input_dataframes_are_not_modified(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, _ = explanation_data
    original = X.copy(deep=True)
    local_probability_sensitivity(
        fitted_pipeline, X.iloc[[0]], training_reference_values(X)
    )
    pd.testing.assert_frame_equal(X, original)


def test_predictions_are_unchanged_after_explanations(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    before = positive_class_probabilities(fitted_pipeline, X)
    calculate_global_permutation_importance(fitted_pipeline, X, y, n_repeats=1)
    local_probability_sensitivity(
        fitted_pipeline, X.iloc[[0]], training_reference_values(X)
    )
    after = positive_class_probabilities(fitted_pipeline, X)
    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_confusion_categories_match_locked_threshold_logic() -> None:
    categories = assign_confusion_categories(
        [0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], threshold=0.5
    )
    assert categories.tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]


def test_error_summary_counts_match_confusion_categories(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    probabilities = positive_class_probabilities(fitted_pipeline, X)
    counts, numeric, categorical = summarize_prediction_errors(
        X,
        y,
        probabilities,
        0.5,
        numerical_features=["tenure_months"],
        categorical_features=["contract_type"],
    )
    assert counts["customer_count"].sum() == len(X)
    assert set(counts["confusion_category"]) == {
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    }
    assert set(numeric["feature"]) == {"tenure_months"}
    assert set(categorical["feature"]) == {"contract_type"}


def test_anonymized_outputs_do_not_expose_identifiers(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline
) -> None:
    X, y = explanation_data
    probabilities = positive_class_probabilities(fitted_pipeline, X)
    selected = select_representative_cases(
        X, y, probabilities, 0.5, permitted_indices=X.index
    )
    public = anonymized_case_summary(
        selected, X, ["tenure_months", "contract_type"]
    )
    assert "customer_id" not in public.columns
    assert "churn" not in public.columns
    assert "_row_index" not in public.columns


def test_trusted_model_loader_checks_hash_and_explicit_trust(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline, tmp_path: Path
) -> None:
    path = tmp_path / "trusted.joblib"
    joblib.dump(fitted_pipeline, path)
    digest = file_sha256(path)
    with pytest.raises(ValueError, match="trusted=True"):
        load_trusted_locked_model(path, expected_sha256=digest)
    loaded = load_trusted_locked_model(
        path, expected_sha256=digest, trusted=True
    )
    X, _ = explanation_data
    np.testing.assert_allclose(
        positive_class_probabilities(fitted_pipeline, X.iloc[:5]),
        positive_class_probabilities(loaded, X.iloc[:5]),
    )


def test_global_importance_plot_is_created(
    explanation_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline, tmp_path: Path
) -> None:
    X, y = explanation_data
    importance = calculate_global_permutation_importance(
        fitted_pipeline, X, y, n_repeats=1
    )
    output = tmp_path / "importance.png"
    plot_global_importance(importance, output, top_n=10)
    assert output.is_file() and output.stat().st_size > 0
