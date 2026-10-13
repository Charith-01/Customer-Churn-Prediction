"""Tests for deterministic and leakage-safe feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ENGINEERED_NUMERICAL_FEATURES,
    CustomerFeatureEngineer,
    build_feature_preprocessing_pipeline,
)


@pytest.fixture
def predictor_data() -> pd.DataFrame:
    """Small predictor-only fixture using the audited feature schema."""
    return pd.DataFrame(
        {
            "age": [25, 40, 61, 35, 52, 29],
            "tenure_months": [0, 6, 12, 24, 36, 60],
            "monthly_charges": [0.0, 50.0, 80.0, np.nan, 100.0, 65.0],
            "total_charges": [0.0, 300.0, 960.0, 1500.0, 3600.0, 3900.0],
            "complaints_last_12m": [0, 1, 2, 0, 3, 1],
            "late_payments_last_12m": [0, 1, 2, 1, 0, 3],
            "satisfaction_score": [10.0, 8.0, 6.0, np.nan, 4.0, 7.0],
            "monthly_usage_gb": [0.0, 25.0, 80.0, 40.0, np.nan, 130.0],
            "discount_pct": [0.0, 5.0, 10.0, 0.0, 15.0, 8.0],
            "contract_type": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "One year",
            ],
            "internet_service": ["DSL", "Fiber optic", "DSL", "No", "DSL", "Fiber optic"],
            "region": ["Western", "Central", "Eastern", "Western", "Central", "Eastern"],
            "payment_method": [
                "Credit card",
                "Bank transfer",
                "Electronic check",
                "Mailed check",
                "Credit card",
                "Bank transfer",
            ],
            "paperless_billing": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "dependents": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "support_plan": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "tech_support": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "senior_citizen": [0, 0, 1, 0, 1, 0],
        },
        index=[101, 103, 107, 109, 113, 127],
    )


def test_all_intended_features_are_generated(predictor_data: pd.DataFrame) -> None:
    transformed = CustomerFeatureEngineer().fit_transform(predictor_data)
    assert list(transformed.columns[-len(ENGINEERED_NUMERICAL_FEATURES) :]) == list(
        ENGINEERED_NUMERICAL_FEATURES
    )
    assert set(ENGINEERED_NUMERICAL_FEATURES).issubset(transformed.columns)


def test_calculations_match_documented_formulas(predictor_data: pd.DataFrame) -> None:
    transformed = CustomerFeatureEngineer().fit_transform(predictor_data)
    row = transformed.loc[107]
    assert row["tenure_years"] == pytest.approx(1.0)
    assert row["complaints_per_active_month"] == pytest.approx(2 / 12)
    assert row["payment_risk_score"] == pytest.approx(4.0)
    assert row["complaint_satisfaction_interaction"] == pytest.approx(8.0)
    assert row["monthly_contract_tenure"] == pytest.approx(12.0)
    assert row["late_payment_charge_interaction"] == pytest.approx(160.0)
    assert row["usage_per_charge"] == pytest.approx(1.0)


def test_rows_indices_and_input_are_preserved(predictor_data: pd.DataFrame) -> None:
    original = predictor_data.copy(deep=True)
    transformed = CustomerFeatureEngineer().fit_transform(predictor_data)
    pd.testing.assert_frame_equal(predictor_data, original)
    assert len(transformed) == len(predictor_data)
    assert transformed.index.equals(predictor_data.index)
    assert list(transformed.columns[: predictor_data.shape[1]]) == list(
        predictor_data.columns
    )


def test_missing_and_zero_denominators_are_safe(predictor_data: pd.DataFrame) -> None:
    data = predictor_data.copy()
    data.loc[101, "tenure_months"] = np.nan
    data.loc[101, "contract_type"] = pd.NA
    transformed = CustomerFeatureEngineer().fit_transform(data)

    assert np.isnan(transformed.loc[101, "complaints_per_active_month"])
    assert np.isnan(transformed.loc[101, "monthly_contract_tenure"])
    assert np.isnan(transformed.loc[101, "usage_per_charge"])
    engineered = transformed.loc[:, ENGINEERED_NUMERICAL_FEATURES]
    assert not np.isinf(engineered.to_numpy(dtype="float64")).any()


def test_zero_tenure_uses_one_month_exposure_floor(
    predictor_data: pd.DataFrame,
) -> None:
    transformed = CustomerFeatureEngineer().fit_transform(predictor_data)
    assert transformed.loc[101, "complaints_per_active_month"] == 0.0


@pytest.mark.parametrize("forbidden", ["customer_id", "churn"])
def test_identifier_and_target_are_rejected(
    predictor_data: pd.DataFrame, forbidden: str
) -> None:
    invalid = predictor_data.copy()
    invalid[forbidden] = np.arange(len(invalid))
    with pytest.raises(ValueError, match="removed before feature engineering"):
        CustomerFeatureEngineer().fit(invalid)


def test_output_is_deterministic_and_feature_names_are_stable(
    predictor_data: pd.DataFrame,
) -> None:
    transformer = CustomerFeatureEngineer().fit(predictor_data)
    first = transformer.transform(predictor_data)
    second = transformer.transform(predictor_data)
    pd.testing.assert_frame_equal(first, second)
    assert list(transformer.get_feature_names_out()) == list(first.columns)


def test_transformer_handles_unseen_contract_value(
    predictor_data: pd.DataFrame,
) -> None:
    train = predictor_data.iloc[:4]
    unseen = predictor_data.iloc[4:].copy()
    unseen.loc[unseen.index[0], "contract_type"] = "Future contract"
    transformer = CustomerFeatureEngineer().fit(train)
    transformed = transformer.transform(unseen)
    assert transformed.loc[unseen.index[0], "monthly_contract_tenure"] == 0.0


def test_integrated_pipeline_fits_train_and_transforms_unseen_data(
    predictor_data: pd.DataFrame,
) -> None:
    train = predictor_data.iloc[:4].copy()
    held_out = predictor_data.iloc[4:].copy()
    held_out.loc[held_out.index[0], "contract_type"] = "Future contract"
    pipeline = build_feature_preprocessing_pipeline()

    train_transformed = pipeline.fit_transform(train)
    held_out_transformed = pipeline.transform(held_out)

    assert train_transformed.shape[0] == len(train)
    assert held_out_transformed.shape[0] == len(held_out)
    assert train_transformed.shape[1] == held_out_transformed.shape[1]
    assert np.isfinite(train_transformed).all()
    assert np.isfinite(held_out_transformed).all()


def test_imputation_statistics_are_learned_from_training_only(
    predictor_data: pd.DataFrame,
) -> None:
    train = predictor_data.iloc[:4].copy()
    held_out = predictor_data.iloc[4:].copy()
    held_out.loc[:, "monthly_charges"] = [10_000.0, np.nan]
    pipeline = build_feature_preprocessing_pipeline().fit(train)
    imputer = pipeline.named_steps["preprocessing"].named_transformers_[
        "numeric"
    ].named_steps["imputer"]
    expected = train["monthly_charges"].median()
    monthly_charge_index = list(
        pipeline.named_steps["preprocessing"].transformers_[0][2]
    ).index("monthly_charges")

    assert imputer.statistics_[monthly_charge_index] == pytest.approx(expected)
    pipeline.transform(held_out)
    assert imputer.statistics_[monthly_charge_index] == pytest.approx(expected)


def test_missing_required_source_column_raises(predictor_data: pd.DataFrame) -> None:
    invalid = predictor_data.drop(columns="monthly_usage_gb")
    with pytest.raises(ValueError, match="Missing feature-engineering source columns"):
        CustomerFeatureEngineer().fit(invalid)
