"""Leakage-safe data preparation utilities for the churn project.

This module performs only deterministic, row-preserving cleaning. Learned
operations such as imputation, scaling, and encoding are returned in an
unfitted scikit-learn transformer so they can be fitted within a training fold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IDENTIFIER_COLUMN = "customer_id"
TARGET_COLUMN = "churn"

NUMERICAL_FEATURES = (
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "complaints_last_12m",
    "late_payments_last_12m",
    "satisfaction_score",
    "monthly_usage_gb",
    "discount_pct",
)

CATEGORICAL_FEATURES = (
    "contract_type",
    "internet_service",
    "region",
    "payment_method",
    "paperless_billing",
    "dependents",
    "support_plan",
    "tech_support",
    "senior_citizen",
)

EXPECTED_COLUMNS = (
    IDENTIFIER_COLUMN,
    "age",
    "tenure_months",
    "contract_type",
    "internet_service",
    "region",
    "monthly_charges",
    "total_charges",
    "payment_method",
    "paperless_billing",
    "dependents",
    "support_plan",
    "tech_support",
    "complaints_last_12m",
    "late_payments_last_12m",
    "satisfaction_score",
    "monthly_usage_gb",
    "discount_pct",
    "senior_citizen",
    TARGET_COLUMN,
)

TEXT_COLUMNS = (
    IDENTIFIER_COLUMN,
    "contract_type",
    "internet_service",
    "region",
    "payment_method",
    "paperless_billing",
    "dependents",
    "support_plan",
    "tech_support",
)

NUMERIC_INPUT_COLUMNS = (*NUMERICAL_FEATURES, "senior_citizen")


def _validate_schema(data: pd.DataFrame) -> None:
    """Validate the fixed, audited Phase 1 schema."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not allowed: {duplicates}")

    missing = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "Dataset schema does not match the audited schema. "
            f"Missing columns: {missing or 'none'}; "
            f"unexpected columns: {unexpected or 'none'}."
        )
    if data.empty:
        raise ValueError("Dataset contains no customer records.")


def _encode_target(target: pd.Series) -> pd.Series:
    """Map unambiguous 0/1 labels to integer values or raise an error."""
    target_text = target.astype("string").str.strip()
    if target_text.isna().any() or target_text.eq("").any():
        raise ValueError("Target column 'churn' contains missing or blank labels.")

    numeric_target = pd.to_numeric(target_text, errors="coerce")
    invalid = numeric_target.isna() | ~numeric_target.isin([0, 1])
    if invalid.any():
        labels = sorted(target_text[invalid].unique().tolist())
        raise ValueError(
            "Target column 'churn' must contain only binary labels 0 and 1. "
            f"Unexpected labels: {labels}"
        )
    return numeric_target.astype("int64").rename(TARGET_COLUMN)


def _numeric_view(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature values, rejecting non-numeric non-missing text."""
    converted: dict[str, pd.Series] = {}
    for column in NUMERIC_INPUT_COLUMNS:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            examples = original[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values, for example: {examples}"
            )
        converted[column] = numeric
    return pd.DataFrame(converted, index=data.index)


def _validate_domain_ranges(numeric: pd.DataFrame) -> None:
    """Apply broad, documented validity screens without changing values."""
    bounded_rules = {
        "age": (0, None),
        "tenure_months": (0, None),
        "monthly_charges": (0, None),
        "total_charges": (0, None),
        "complaints_last_12m": (0, None),
        "late_payments_last_12m": (0, None),
        "satisfaction_score": (0, 10),
        "monthly_usage_gb": (0, None),
        "discount_pct": (0, 100),
    }
    for column, (lower, upper) in bounded_rules.items():
        values = numeric[column].dropna()
        invalid = values.lt(lower)
        if upper is not None:
            invalid |= values.gt(upper)
        if invalid.any():
            sample = values[invalid].head(5).tolist()
            bounds = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise ValueError(
                f"Column '{column}' has values outside the broad valid range {bounds}: {sample}"
            )

    integer_columns = (
        "age",
        "tenure_months",
        "complaints_last_12m",
        "late_payments_last_12m",
        "senior_citizen",
    )
    for column in integer_columns:
        values = numeric[column].dropna()
        non_integer = ~np.isclose(values % 1, 0)
        if non_integer.any():
            raise ValueError(f"Column '{column}' must contain whole-number values.")

    senior = numeric["senior_citizen"].dropna()
    if not senior.isin([0, 1]).all():
        invalid_values = sorted(senior[~senior.isin([0, 1])].unique().tolist())
        raise ValueError(
            "Column 'senior_citizen' must contain only 0 and 1. "
            f"Unexpected values: {invalid_values}"
        )


def validate_input_data(data: pd.DataFrame) -> None:
    """Validate schema, identifiers, target labels, and broad value constraints."""
    _validate_schema(data)

    normalized_ids = data[IDENTIFIER_COLUMN].astype("string").str.strip()
    missing_ids = normalized_ids.isna() | normalized_ids.eq("")
    if missing_ids.any():
        raise ValueError("Customer identifiers cannot be missing or blank.")
    duplicate_ids = normalized_ids.duplicated(keep=False)
    if duplicate_ids.any():
        examples = normalized_ids[duplicate_ids].unique()[:5].tolist()
        raise ValueError(
            "Repeated customer IDs create a train/test split-integrity risk. "
            f"Resolve them before splitting. Examples: {examples}"
        )

    _encode_target(data[TARGET_COLUMN])
    numeric = _numeric_view(data)
    _validate_domain_ranges(numeric)


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate it without changing or overwriting the file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    data = pd.read_csv(csv_path)
    validate_input_data(data)
    return data


def apply_deterministic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize representation while preserving rows, missingness, and outliers.

    Rules are deliberately conservative: trim text whitespace, represent blank
    text as missing, parse audited numeric columns, and encode an unambiguous
    0/1 target. No record, missing value, or outlier is removed or imputed.
    """
    _validate_schema(data)
    cleaned = data.copy(deep=True)

    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].mask(cleaned[column].eq(""), pd.NA)

    for column in NUMERIC_INPUT_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    cleaned[TARGET_COLUMN] = _encode_target(cleaned[TARGET_COLUMN])

    validate_input_data(cleaned)
    return cleaned.loc[:, EXPECTED_COLUMNS]


def get_feature_groups(data: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Return the audited numerical and categorical predictor lists."""
    if data is not None:
        _validate_schema(data)
    return list(NUMERICAL_FEATURES), list(CATEGORICAL_FEATURES)


def separate_features_target_ids(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate predictors, binary target, and identifiers for traceability."""
    validate_input_data(data)
    numerical, categorical = get_feature_groups(data)
    predictors = data.loc[:, [*numerical, *categorical]].copy()
    target = _encode_target(data[TARGET_COLUMN]).copy()
    identifiers = data[IDENTIFIER_COLUMN].astype("string").copy()

    if IDENTIFIER_COLUMN in predictors or TARGET_COLUMN in predictors:
        raise AssertionError("Identifier or target leaked into the predictor matrix.")
    return predictors, target, identifiers


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    """Create an order-sensitive SHA-256 fingerprint of dataframe contents."""
    canonical_csv = data.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<MISSING>",
        float_format="%.17g",
    )
    return hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()


def _validate_saved_split(
    payload: dict[str, Any],
    data: pd.DataFrame,
    fingerprint: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that a persisted split belongs to this exact dataset."""
    expected_metadata = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "row_count": len(data),
        "test_size": test_size,
        "random_state": random_state,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_metadata.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Saved split metadata does not match the current dataset or settings: "
            f"{mismatches}. Delete the local split artifact only after reviewing the change."
        )

    train_indices = np.asarray(payload.get("train_indices", []), dtype=int)
    test_indices = np.asarray(payload.get("test_indices", []), dtype=int)
    train_set, test_set = set(train_indices.tolist()), set(test_indices.tolist())
    expected_indices = set(range(len(data)))
    if train_set & test_set:
        raise ValueError("Saved train and test indices overlap.")
    if train_set | test_set != expected_indices:
        raise ValueError("Saved split does not cover every dataset row exactly once.")
    if len(train_indices) != len(train_set) or len(test_indices) != len(test_set):
        raise ValueError("Saved split contains repeated row indices.")
    return train_indices, test_indices


def create_or_load_fixed_split(
    data: pd.DataFrame,
    split_path: str | Path,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Create or reload a fingerprint-bound, stratified row-index split."""
    validate_input_data(data)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    path = Path(split_path)
    fingerprint = dataframe_fingerprint(data)
    target = _encode_target(data[TARGET_COLUMN])

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_indices, test_indices = _validate_saved_split(
            payload, data, fingerprint, test_size, random_state
        )
        created = False
    else:
        all_indices = np.arange(len(data))
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        payload = {
            "schema_version": 1,
            "dataset_fingerprint": fingerprint,
            "fingerprint_method": "SHA-256 of canonical row-ordered CSV content",
            "row_count": len(data),
            "test_size": test_size,
            "random_state": random_state,
            "target_mapping": {"0": 0, "1": 1},
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_target_counts": {
                str(key): int(value)
                for key, value in target.iloc[train_indices].value_counts().sort_index().items()
            },
            "test_target_counts": {
                str(key): int(value)
                for key, value in target.iloc[test_indices].value_counts().sort_index().items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        created = True

    train_ids = set(data.iloc[train_indices][IDENTIFIER_COLUMN].astype(str))
    test_ids = set(data.iloc[test_indices][IDENTIFIER_COLUMN].astype(str))
    if train_ids & test_ids:
        raise ValueError("Customer IDs overlap between the training and test sets.")

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "dataset_fingerprint": fingerprint,
        "random_state": random_state,
        "test_size": test_size,
        "loaded_existing": not created,
    }


def build_preprocessing_transformer(
    numerical_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    *,
    allowed_additional_numerical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Construct an unfitted transformer for use inside later model pipelines."""
    numerical = list(numerical_features or NUMERICAL_FEATURES)
    categorical = list(categorical_features or CATEGORICAL_FEATURES)
    additional_numerical = set(allowed_additional_numerical_features or ())
    selected = numerical + categorical

    if len(selected) != len(set(selected)):
        raise ValueError("A feature cannot be both numerical and categorical.")
    forbidden = {IDENTIFIER_COLUMN, TARGET_COLUMN} & set(selected)
    if forbidden:
        raise ValueError(f"Identifier and target columns cannot be predictors: {forbidden}")
    unknown = (
        set(selected)
        - set(NUMERICAL_FEATURES)
        - set(CATEGORICAL_FEATURES)
        - additional_numerical
    )
    if unknown:
        raise ValueError(f"Unrecognized predictor columns: {sorted(unknown)}")
    categorical_additions = additional_numerical & set(categorical)
    if categorical_additions:
        raise ValueError(
            "Additional numerical features cannot be configured as categorical: "
            f"{sorted(categorical_additions)}"
        )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                    missing_values=pd.NA,
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
