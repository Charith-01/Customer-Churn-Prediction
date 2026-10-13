"""Tests for Phase 5 model pipelines and ranking utilities."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import CustomerFeatureEngineer

from src.modeling import (
    CONTACT_FRACTION,
    N_SPLITS,
    RANDOM_STATE,
    build_model_pipelines,
    calculate_top_fraction_metrics,
    fit_and_save_candidate,
    generate_oof_probabilities,
    get_search_spaces,
    make_cv_splits,
    make_stratified_cv,
    positive_class_probabilities,
    run_hyperparameter_tuning,
)

EXPECTED_SPLIT_ARTIFACT_SHA256 = (
    "4325cbfeb6523b8594174ef19926790c08114ca360a78dbe2701378e4fd0a235"
)


@pytest.fixture
def modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    rows = 60
    tenure = rng.integers(1, 73, size=rows)
    monthly_charges = rng.uniform(25, 130, size=rows)
    usage = rng.uniform(5, 180, size=rows)
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 76, size=rows),
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": monthly_charges * tenure * rng.uniform(0.8, 1.1, size=rows),
            "complaints_last_12m": rng.integers(0, 5, size=rows),
            "late_payments_last_12m": rng.integers(0, 4, size=rows),
            "satisfaction_score": rng.uniform(1, 10, size=rows),
            "monthly_usage_gb": usage,
            "discount_pct": rng.uniform(0, 25, size=rows),
            "contract_type": np.resize(
                ["Month-to-month", "One year", "Two year"], rows
            ),
            "internet_service": np.resize(["DSL", "Fiber optic", "No"], rows),
            "region": np.resize(["Western", "Central", "Eastern"], rows),
            "payment_method": np.resize(
                ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
                rows,
            ),
            "paperless_billing": np.resize(["Yes", "No"], rows),
            "dependents": np.resize(["No", "Yes"], rows),
            "support_plan": np.resize(["No", "Yes"], rows),
            "tech_support": np.resize(["No", "Yes"], rows),
            "senior_citizen": rng.integers(0, 2, size=rows),
        },
        index=np.arange(500, 500 + rows),
    )
    data.loc[data.index[[2, 17]], "monthly_charges"] = np.nan
    data.loc[data.index[[5, 23]], "satisfaction_score"] = np.nan
    data.loc[data.index[[7, 31]], "monthly_usage_gb"] = np.nan
    target = pd.Series(
        np.resize([0, 1, 1, 1, 1], rows), index=data.index, name="churn"
    )
    return data, target


def test_all_three_required_models_can_be_constructed() -> None:
    pipelines = build_model_pipelines(include_dummy=False)
    assert set(pipelines) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert isinstance(pipelines["logistic_regression"].named_steps["model"], LogisticRegression)
    assert isinstance(pipelines["random_forest"].named_steps["model"], RandomForestClassifier)
    assert isinstance(
        pipelines["hist_gradient_boosting"].named_steps["model"],
        HistGradientBoostingClassifier,
    )


def test_every_model_uses_complete_required_pipeline() -> None:
    for pipeline in build_model_pipelines().values():
        assert list(pipeline.named_steps) == [
            "feature_engineering",
            "preprocessing",
            "model",
        ]
        assert isinstance(
            pipeline.named_steps["feature_engineering"], CustomerFeatureEngineer
        )
        assert isinstance(pipeline.named_steps["preprocessing"], ColumnTransformer)


@pytest.mark.parametrize("forbidden", ["customer_id", "churn"])
def test_identifier_and_target_cannot_enter_model_predictors(
    modeling_data: tuple[pd.DataFrame, pd.Series], forbidden: str
) -> None:
    X, y = modeling_data
    invalid = X.copy()
    invalid[forbidden] = np.arange(len(invalid))
    with pytest.raises(ValueError, match="cannot be model predictors|removed before"):
        build_model_pipelines(include_dummy=False)["logistic_regression"].fit(invalid, y)


@pytest.mark.parametrize(
    "model_key",
    ["logistic_regression", "random_forest", "hist_gradient_boosting"],
)
def test_models_fit_missing_values_and_produce_valid_predictions(
    modeling_data: tuple[pd.DataFrame, pd.Series], model_key: str
) -> None:
    X, y = modeling_data
    train, held_out = np.arange(48), np.arange(48, 60)
    pipeline = build_model_pipelines(include_dummy=False)[model_key]
    pipeline.fit(X.iloc[train], y.iloc[train])
    predictions = pipeline.predict(X.iloc[held_out])
    probabilities = positive_class_probabilities(pipeline, X.iloc[held_out])

    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(held_out),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert list(pipeline.named_steps["model"].classes_) == [0, 1]


def test_unknown_categorical_values_are_handled(
    modeling_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = modeling_data
    train = X.iloc[:48]
    unseen = X.iloc[48:].copy()
    unseen.loc[unseen.index[0], "region"] = "Future region"
    unseen.loc[unseen.index[1], "contract_type"] = "Future contract"
    pipeline = build_model_pipelines(include_dummy=False)["logistic_regression"].fit(
        train, y.iloc[:48]
    )
    probabilities = positive_class_probabilities(pipeline, unseen)
    assert np.isfinite(probabilities).all()


def test_cross_validation_is_stratified_and_reproducible(
    modeling_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = modeling_data
    first = make_cv_splits(X, y)
    second = make_cv_splits(X, y)
    assert len(first) == N_SPLITS == 5
    for (train_a, valid_a), (train_b, valid_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(valid_a, valid_b)
        assert set(y.iloc[valid_a].unique()) == {0, 1}
        assert abs(y.iloc[valid_a].mean() - y.mean()) <= 0.051
    cv = make_stratified_cv()
    assert cv.shuffle is True
    assert cv.random_state == RANDOM_STATE


def test_top_twenty_percent_metrics_use_stable_tie_handling() -> None:
    y = np.array([1, 0, 1, 1, 1])
    probabilities = np.full(5, 0.8)
    result = calculate_top_fraction_metrics(y, probabilities, fraction=CONTACT_FRACTION)
    assert result["contact_count"] == 1
    assert result["true_positives_contacted"] == 1
    assert result["precision_at_20"] == pytest.approx(1.0)
    assert result["recall_at_20"] == pytest.approx(0.25)
    assert result["prevalence"] == pytest.approx(0.8)
    assert result["lift_at_20"] == pytest.approx(1.25)


def test_oof_probabilities_cover_each_row_once(
    modeling_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = modeling_data
    splits = make_cv_splits(X, y)
    pipeline = build_model_pipelines(include_dummy=False)["logistic_regression"]
    oof = generate_oof_probabilities(pipeline, X, y, splits)
    assert len(oof) == len(X)
    assert oof["dataset_row_index"].is_unique
    assert set(oof["cv_fold"]) == set(range(1, 6))
    assert oof["oof_churn_probability"].between(0, 1).all()


def test_saved_pipeline_loads_with_consistent_predictions(
    modeling_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = modeling_data
    output_path = tmp_path / "candidate.joblib"
    pipeline = build_model_pipelines(include_dummy=False)["logistic_regression"]
    fitted = fit_and_save_candidate(pipeline, X, y, output_path)
    loaded = joblib.load(output_path)
    np.testing.assert_allclose(
        positive_class_probabilities(fitted, X.iloc[:8]),
        positive_class_probabilities(loaded, X.iloc[:8]),
    )


def test_search_spaces_cover_all_models_and_class_weight_options() -> None:
    spaces = get_search_spaces()
    assert set(spaces) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    for space in spaces.values():
        assert space["model__class_weight"] == [None, "balanced"]


def test_saved_split_artifact_is_not_modified() -> None:
    path = Path("data/processed/02_split_indices.json")
    if not path.exists():
        pytest.skip("Local split artifact is intentionally not committed.")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SPLIT_ARTIFACT_SHA256


def test_tuning_api_has_no_final_test_inputs() -> None:
    parameters = inspect.signature(run_hyperparameter_tuning).parameters
    assert "X_test" not in parameters
    assert "y_test" not in parameters
    assert "test_indices" not in parameters
