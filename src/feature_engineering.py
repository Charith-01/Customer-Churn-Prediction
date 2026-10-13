"""Deterministic, leakage-aware feature engineering for customer churn.

The transformer in this module uses predictor values only. It deliberately
contains no target-derived thresholds, encodings, aggregates, or learned
statistics. Imputation, scaling, and category encoding remain downstream so
they can be fitted independently within every future cross-validation fold.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_transformer,
)

SATISFACTION_SCALE_MAX = 10.0

ENGINEERED_NUMERICAL_FEATURES = (
    "tenure_years",
    "complaints_per_active_month",
    "payment_risk_score",
    "complaint_satisfaction_interaction",
    "monthly_contract_tenure",
    "late_payment_charge_interaction",
    "usage_per_charge",
)

REQUIRED_SOURCE_COLUMNS = (
    "tenure_months",
    "complaints_last_12m",
    "late_payments_last_12m",
    "satisfaction_score",
    "contract_type",
    "monthly_charges",
    "monthly_usage_gb",
)

NUMERIC_SOURCE_COLUMNS = tuple(
    column for column in REQUIRED_SOURCE_COLUMNS if column != "contract_type"
)


def _as_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Feature engineering input must be a pandas DataFrame.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not allowed: {duplicates}")
    forbidden = {IDENTIFIER_COLUMN, TARGET_COLUMN} & set(data.columns)
    if forbidden:
        raise ValueError(
            "Identifier and target must be removed before feature engineering: "
            f"{sorted(forbidden)}"
        )
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing feature-engineering source columns: {missing}")
    return data


def _numeric_sources(data: pd.DataFrame) -> dict[str, pd.Series]:
    numeric: dict[str, pd.Series] = {}
    for column in NUMERIC_SOURCE_COLUMNS:
        source = data[column]
        values = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & values.isna()
        if invalid.any():
            examples = source[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values, for example: {examples}"
            )
        numeric[column] = values.astype("float64")

    non_negative = (
        "tenure_months",
        "complaints_last_12m",
        "late_payments_last_12m",
        "monthly_charges",
        "monthly_usage_gb",
    )
    for column in non_negative:
        if numeric[column].dropna().lt(0).any():
            raise ValueError(f"Column '{column}' cannot contain negative values.")

    satisfaction = numeric["satisfaction_score"].dropna()
    if not satisfaction.between(0, SATISFACTION_SCALE_MAX).all():
        raise ValueError(
            "Column 'satisfaction_score' must be within the validated 0–10 scale."
        )
    return numeric


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide valid values and return missing for absent or non-positive divisors."""
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & denominator.gt(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.replace([np.inf, -np.inf], np.nan)


class CustomerFeatureEngineer(TransformerMixin, BaseEstimator):
    """Append seven deterministic numerical features to predictor data.

    The transformation is stateless. ``fit`` records only the input schema so
    scikit-learn can validate inference data; it learns no values or cutoffs.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "CustomerFeatureEngineer":
        data = _as_dataframe(X)
        _numeric_sources(data)
        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_in_", "n_features_in_"])
        data = _as_dataframe(X)
        if list(data.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Inference columns and ordering must match the data supplied to fit."
            )

        numeric = _numeric_sources(data)
        output = data.copy(deep=True)

        tenure = numeric["tenure_months"]
        complaints = numeric["complaints_last_12m"]
        late_payments = numeric["late_payments_last_12m"]
        satisfaction = numeric["satisfaction_score"]
        monthly_charges = numeric["monthly_charges"]
        usage = numeric["monthly_usage_gb"]

        output["tenure_years"] = tenure / 12.0

        observed_months = tenure.clip(lower=1.0, upper=12.0)
        output["complaints_per_active_month"] = _safe_ratio(
            complaints, observed_months
        )

        output["payment_risk_score"] = complaints + late_payments
        output["complaint_satisfaction_interaction"] = complaints * (
            SATISFACTION_SCALE_MAX - satisfaction
        )

        contract = data["contract_type"].astype("string")
        month_to_month = pd.Series(np.nan, index=data.index, dtype="float64")
        known_contract = contract.notna()
        month_to_month.loc[known_contract] = (
            contract.loc[known_contract].eq("Month-to-month").astype("float64")
        )
        output["monthly_contract_tenure"] = tenure * month_to_month

        output["late_payment_charge_interaction"] = (
            late_payments * monthly_charges
        )
        output["usage_per_charge"] = _safe_ratio(usage, monthly_charges)

        engineered = output.loc[:, ENGINEERED_NUMERICAL_FEATURES]
        if np.isinf(engineered.to_numpy(dtype="float64")).any():
            raise AssertionError("Feature engineering introduced an infinite value.")
        return output

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])
        if input_features is None:
            original = list(self.feature_names_in_)
        else:
            original = list(input_features)
            if original != list(self.feature_names_in_):
                raise ValueError("input_features must match the fitted input columns.")
        return np.asarray([*original, *ENGINEERED_NUMERICAL_FEATURES], dtype=object)


def build_feature_preprocessing_pipeline() -> Pipeline:
    """Build an unfitted feature-engineering and preprocessing pipeline."""
    numerical = [*NUMERICAL_FEATURES, *ENGINEERED_NUMERICAL_FEATURES]
    preprocessing = build_preprocessing_transformer(
        numerical_features=numerical,
        categorical_features=CATEGORICAL_FEATURES,
        allowed_additional_numerical_features=ENGINEERED_NUMERICAL_FEATURES,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", CustomerFeatureEngineer()),
            ("preprocessing", preprocessing),
        ]
    )
