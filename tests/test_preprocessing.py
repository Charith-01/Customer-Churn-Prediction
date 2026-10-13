"""Tests for deterministic cleaning and leakage-safe preprocessing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    apply_deterministic_cleaning,
    build_preprocessing_transformer,
    create_or_load_fixed_split,
    get_feature_groups,
    load_and_validate_data,
    separate_features_target_ids,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Small balanced fixture using the actual audited schema."""
    rows = []
    for index in range(10):
        rows.append(
            {
                "customer_id": f" CUST9{index:05d} ",
                "age": 25 + index,
                "tenure_months": index + 1,
                "contract_type": ["Month-to-month", "One year"][index % 2],
                "internet_service": ["DSL", "Fiber optic"][index % 2],
                "region": ["Western", "Central"][index % 2],
                "monthly_charges": np.nan if index == 0 else 50.0 + index,
                "total_charges": 100.0 + index * 50,
                "payment_method": ["Credit card", "Bank transfer"][index % 2],
                "paperless_billing": ["Yes", "No"][index % 2],
                "dependents": ["No", "Yes"][index % 2],
                "support_plan": ["No", "Yes"][index % 2],
                "tech_support": ["No", "Yes"][index % 2],
                "complaints_last_12m": index % 3,
                "late_payments_last_12m": index % 2,
                "satisfaction_score": np.nan if index == 1 else 5.0 + index / 10,
                "monthly_usage_gb": 20.0 + index,
                "discount_pct": float(index),
                "senior_citizen": index % 2,
                "churn": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_original_dataset_is_not_modified() -> None:
    raw_path = Path("data/raw/customer_churn_advanced_dataset.csv")
    if not raw_path.exists():
        pytest.skip("Local raw dataset is intentionally not committed.")

    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw = load_and_validate_data(raw_path)
    _ = apply_deterministic_cleaning(raw)
    after = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert after == before


def test_cleaning_is_deterministic_and_row_preserving(sample_data: pd.DataFrame) -> None:
    original = sample_data.copy(deep=True)
    first = apply_deterministic_cleaning(sample_data)
    second = apply_deterministic_cleaning(sample_data)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(sample_data, original)
    assert first.shape == sample_data.shape
    assert first[IDENTIFIER_COLUMN].str.startswith(" ").sum() == 0
    assert first.isna().sum().sum() == sample_data.isna().sum().sum()


def test_feature_groups_are_explicit_and_disjoint(sample_data: pd.DataFrame) -> None:
    numerical, categorical = get_feature_groups(sample_data)
    assert numerical == list(NUMERICAL_FEATURES)
    assert categorical == list(CATEGORICAL_FEATURES)
    assert not set(numerical) & set(categorical)
    assert IDENTIFIER_COLUMN not in numerical + categorical
    assert TARGET_COLUMN not in numerical + categorical


def test_identifier_and_target_cannot_enter_feature_matrix(sample_data: pd.DataFrame) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    predictors, target, identifiers = separate_features_target_ids(cleaned)
    assert IDENTIFIER_COLUMN not in predictors.columns
    assert TARGET_COLUMN not in predictors.columns
    assert target.name == TARGET_COLUMN
    assert identifiers.name == IDENTIFIER_COLUMN

    with pytest.raises(ValueError, match="cannot be predictors"):
        build_preprocessing_transformer(
            numerical_features=[*NUMERICAL_FEATURES, TARGET_COLUMN]
        )


def test_pipeline_imputes_and_handles_unknown_categories(sample_data: pd.DataFrame) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    predictors, _, _ = separate_features_target_ids(cleaned)
    train = predictors.iloc[:8].copy()
    held_out = predictors.iloc[8:].copy()
    held_out.loc[held_out.index[0], "contract_type"] = "Unseen contract"
    held_out.loc[held_out.index[0], "monthly_usage_gb"] = np.nan

    transformer = build_preprocessing_transformer()
    with pytest.raises(NotFittedError):
        transformer.transform(held_out)
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(held_out)

    assert train_transformed.shape[0] == len(train)
    assert test_transformed.shape[0] == len(held_out)
    assert train_transformed.shape[1] == test_transformed.shape[1]
    assert not np.isnan(train_transformed).any()
    assert not np.isnan(test_transformed).any()


def test_numerical_imputation_is_learned_from_training_only(sample_data: pd.DataFrame) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    predictors, _, _ = separate_features_target_ids(cleaned)
    train = predictors.iloc[:8]
    held_out = predictors.iloc[8:].copy()
    held_out.loc[:, "monthly_charges"] = [10_000.0, np.nan]

    transformer = build_preprocessing_transformer()
    transformer.fit(train)
    learned_statistics = transformer.named_transformers_["numeric"].named_steps[
        "imputer"
    ].statistics_
    monthly_index = list(NUMERICAL_FEATURES).index("monthly_charges")
    expected_train_median = train["monthly_charges"].median()

    assert learned_statistics[monthly_index] == pytest.approx(expected_train_median)
    transformer.transform(held_out)
    assert learned_statistics[monthly_index] == pytest.approx(expected_train_median)


def test_entirely_missing_training_features_are_retained(sample_data: pd.DataFrame) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    predictors, _, _ = separate_features_target_ids(cleaned)
    train = predictors.iloc[:8].copy()
    held_out = predictors.iloc[8:].copy()
    train.loc[:, "monthly_usage_gb"] = np.nan
    train.loc[:, "region"] = pd.NA

    transformer = build_preprocessing_transformer()
    train_transformed = transformer.fit_transform(train)
    held_out_transformed = transformer.transform(held_out)

    assert train_transformed.shape[1] == held_out_transformed.shape[1]
    assert not np.isnan(train_transformed).any()
    assert not np.isnan(held_out_transformed).any()


def test_fixed_split_is_disjoint_and_reproducible(
    sample_data: pd.DataFrame, tmp_path: Path
) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    split_path = tmp_path / "split.json"
    first = create_or_load_fixed_split(cleaned, split_path)
    second = create_or_load_fixed_split(cleaned, split_path)

    assert set(first["train_indices"]).isdisjoint(first["test_indices"])
    assert set(first["train_indices"]) | set(first["test_indices"]) == set(range(10))
    np.testing.assert_array_equal(first["train_indices"], second["train_indices"])
    np.testing.assert_array_equal(first["test_indices"], second["test_indices"])
    assert second["loaded_existing"] is True

    train_ids = set(cleaned.iloc[first["train_indices"]][IDENTIFIER_COLUMN])
    test_ids = set(cleaned.iloc[first["test_indices"]][IDENTIFIER_COLUMN])
    assert train_ids.isdisjoint(test_ids)


def test_saved_split_rejects_changed_dataset(
    sample_data: pd.DataFrame, tmp_path: Path
) -> None:
    cleaned = apply_deterministic_cleaning(sample_data)
    split_path = tmp_path / "split.json"
    create_or_load_fixed_split(cleaned, split_path)
    changed = cleaned.copy()
    changed.loc[0, "age"] += 1

    with pytest.raises(ValueError, match="does not match"):
        create_or_load_fixed_split(changed, split_path)


@pytest.mark.parametrize("bad_label", [2, -1, "churned", np.nan])
def test_invalid_target_labels_raise_informative_error(
    sample_data: pd.DataFrame, bad_label: object
) -> None:
    invalid = sample_data.copy()
    # Object dtype allows this test to exercise the validator rather than
    # pandas' stricter assignment rules for an integer Series.
    invalid[TARGET_COLUMN] = invalid[TARGET_COLUMN].astype("object")
    invalid.loc[0, TARGET_COLUMN] = bad_label
    with pytest.raises(ValueError, match="Target column 'churn'"):
        apply_deterministic_cleaning(invalid)
