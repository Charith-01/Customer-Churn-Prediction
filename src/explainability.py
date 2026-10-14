"""Model-preserving explainability utilities for the locked Phase 6 artifact.

Global explanations use permutation importance on original input columns.
Local explanations use one-at-a-time probability sensitivity relative to
training-derived reference values. Neither method establishes causality.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from src.evaluation import positive_class_probabilities, predictions_at_threshold
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

EXPECTED_PREDICTORS = (*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES)
CONFUSION_LABELS = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)
BLUE = "#3569D4"
PURPLE = "#8051D6"


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest for an artifact without loading it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trusted_locked_model(
    path: str | Path,
    *,
    expected_sha256: str,
    trusted: bool = False,
) -> BaseEstimator:
    """Load a locally trusted joblib model only after verifying its digest.

    Joblib uses pickle semantics and can execute code while loading. Callers
    must explicitly assert that the artifact comes from a trusted source.
    """
    if not trusted:
        raise ValueError("Refusing to load joblib until trusted=True is explicit.")
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    actual_sha256 = file_sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Model artifact digest does not match locked metadata: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    model = joblib.load(artifact_path)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Locked model must expose fitted classes_ and predict_proba.")
    return model


def validate_feature_schema(X: pd.DataFrame) -> pd.DataFrame:
    """Validate predictor-only raw schema and return an isolated ordered copy."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Explanation input must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("Explanation input cannot be empty.")
    forbidden = {IDENTIFIER_COLUMN, TARGET_COLUMN} & set(X.columns)
    if forbidden:
        raise ValueError(f"Identifier and target cannot be explanation inputs: {forbidden}")
    missing = sorted(set(EXPECTED_PREDICTORS) - set(X.columns))
    unexpected = sorted(set(X.columns) - set(EXPECTED_PREDICTORS))
    if missing or unexpected:
        raise ValueError(
            f"Predictor schema mismatch. Missing: {missing or 'none'}; "
            f"unexpected: {unexpected or 'none'}."
        )
    return X.loc[:, EXPECTED_PREDICTORS].copy(deep=True)


def calculate_global_permutation_importance(
    fitted_model: BaseEstimator,
    X: pd.DataFrame,
    y: Sequence[int] | pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "average_precision",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Calculate deterministic raw-input permutation importance.

    Because permutation occurs before the complete pipeline, each result maps
    to one original input column, not an engineered or one-hot feature.
    Negative importance values are retained.
    """
    predictors = validate_feature_schema(X)
    target = np.asarray(y, dtype=int)
    if target.ndim != 1 or len(target) != len(predictors):
        raise ValueError("Target must be one-dimensional and align with predictors.")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Permutation importance requires both binary target classes.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    result = permutation_importance(
        fitted_model,
        predictors,
        target,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    importance = pd.DataFrame(
        {
            "feature": predictors.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, kind="stable")
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance["scoring"] = scoring
    return importance.reset_index(drop=True)


def plot_global_importance(
    importance: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 15,
) -> None:
    """Save a horizontal raw-feature permutation-importance chart."""
    required = {"feature", "importance_mean", "importance_std"}
    if required - set(importance.columns):
        raise ValueError("Importance table is missing required columns.")
    shown = importance.head(min(top_n, len(importance))).sort_values(
        "importance_mean", ascending=True, kind="stable"
    )
    colors = [PURPLE if value >= 0 else "#A7A7A7" for value in shown["importance_mean"]]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(
        shown["feature"],
        shown["importance_mean"],
        xerr=shown["importance_std"],
        color=colors,
        alpha=0.9,
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        title="Global Original-Feature Permutation Importance",
        xlabel="Decrease in held-out Average Precision after permutation",
        ylabel="Original input feature",
    )
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def training_reference_values(X_train: pd.DataFrame) -> dict[str, Any]:
    """Create observed near-median/mode references from training data.

    Numerical references use the observed non-missing value nearest the median.
    This keeps integer fields integral and avoids synthetic out-of-domain values.
    """
    predictors = validate_feature_schema(X_train)
    references: dict[str, Any] = {}
    for feature in NUMERICAL_FEATURES:
        values = pd.to_numeric(predictors[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Cannot derive a numerical reference for '{feature}'.")
        median = values.median()
        nearest_index = (values - median).abs().sort_values(kind="stable").index[0]
        references[feature] = predictors.loc[nearest_index, feature]
    for feature in CATEGORICAL_FEATURES:
        modes = predictors[feature].dropna().mode()
        if modes.empty:
            raise ValueError(f"Cannot derive a categorical reference for '{feature}'.")
        references[feature] = modes.iloc[0]
    return references


def local_probability_sensitivity(
    fitted_model: BaseEstimator,
    customer: pd.DataFrame,
    reference_values: Mapping[str, Any],
) -> pd.DataFrame:
    """Measure one-feature-at-a-time probability sensitivity for one row.

    ``probability_effect`` is original probability minus the probability after
    replacing one raw input with its training reference. Effects are neither
    additive attributions nor causal counterfactual effects.
    """
    predictors = validate_feature_schema(customer)
    if len(predictors) != 1:
        raise ValueError("Local explanation requires exactly one customer row.")
    missing_references = set(EXPECTED_PREDICTORS) - set(reference_values)
    if missing_references:
        raise ValueError(f"Missing reference values: {sorted(missing_references)}")
    original_probability = float(positive_class_probabilities(fitted_model, predictors)[0])
    rows: list[dict[str, Any]] = []
    for feature in EXPECTED_PREDICTORS:
        perturbed = predictors.copy(deep=True)
        customer_value = perturbed.iloc[0][feature]
        reference_value = reference_values[feature]
        perturbed.loc[perturbed.index[0], feature] = reference_value
        reference_probability = float(
            positive_class_probabilities(fitted_model, perturbed)[0]
        )
        effect = original_probability - reference_probability
        if np.isclose(effect, 0.0, atol=1e-12):
            direction = "no measurable change"
        elif effect > 0:
            direction = "actual value raises predicted risk vs reference"
        else:
            direction = "actual value lowers predicted risk vs reference"
        rows.append(
            {
                "feature": feature,
                "customer_value": _display_value(customer_value),
                "training_reference": _display_value(reference_value),
                "original_probability": original_probability,
                "reference_probability": reference_probability,
                "probability_effect": effect,
                "absolute_effect": abs(effect),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_effect", ascending=False, kind="stable"
    ).reset_index(drop=True)


def plot_local_sensitivity(
    explanation: pd.DataFrame,
    output_path: str | Path,
    *,
    customer_label: str,
    predicted_probability: float,
    actual_outcome: int,
    top_n: int = 10,
) -> None:
    """Save a signed local probability-sensitivity chart."""
    required = {"feature", "probability_effect", "absolute_effect"}
    if required - set(explanation.columns):
        raise ValueError("Local explanation is missing required columns.")
    shown = explanation.head(min(top_n, len(explanation))).sort_values(
        "probability_effect", ascending=True, kind="stable"
    )
    colors = [PURPLE if value >= 0 else BLUE for value in shown["probability_effect"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(shown["feature"], shown["probability_effect"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        title=(
            f"{customer_label}: Local Probability Sensitivity\n"
            f"Predicted churn probability={predicted_probability:.3f}; "
            f"actual churn={actual_outcome}"
        ),
        xlabel=(
            "Probability effect: original risk minus risk after replacing one "
            "feature with its training reference"
        ),
        ylabel="Original input feature",
    )
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def assign_confusion_categories(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    index: Iterable[Any] | None = None,
) -> pd.Series:
    """Assign true/false positive/negative labels at the locked threshold."""
    target = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or len(target) != len(scores):
        raise ValueError("Target and probabilities must be aligned one-dimensional data.")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")
    predicted = predictions_at_threshold(scores, threshold)
    categories = np.select(
        [
            (target == 1) & (predicted == 1),
            (target == 0) & (predicted == 0),
            (target == 0) & (predicted == 1),
            (target == 1) & (predicted == 0),
        ],
        list(CONFUSION_LABELS),
        default="unknown",
    )
    return pd.Series(categories, index=index, name="confusion_category")


def select_representative_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    permitted_indices: Iterable[Any],
) -> pd.DataFrame:
    """Select anonymized TP, TN, and error cases from the permitted partition."""
    predictors = validate_feature_schema(X_test)
    permitted = set(permitted_indices)
    if not set(predictors.index).issubset(permitted):
        raise ValueError("Explanation candidates include rows outside the permitted partition.")
    target = pd.Series(np.asarray(y_test, dtype=int), index=predictors.index)
    scores = pd.Series(np.asarray(probabilities, dtype=float), index=predictors.index)
    if len(target) != len(predictors) or len(scores) != len(predictors):
        raise ValueError("Case-selection inputs are not aligned.")
    categories = assign_confusion_categories(
        target, scores, threshold, index=predictors.index
    )
    frame = pd.DataFrame(
        {
            "actual_churn": target,
            "predicted_probability": scores,
            "predicted_class": predictions_at_threshold(scores, threshold),
            "confusion_category": categories,
        },
        index=predictors.index,
    )

    requested = ["true_positive", "true_negative", "false_negative"]
    if not (categories == "false_negative").any():
        requested[-1] = "false_positive"
    selections: list[dict[str, Any]] = []
    used: set[Any] = set()
    for label, category in zip(("Customer A", "Customer B", "Customer C"), requested):
        candidates = frame.loc[(frame["confusion_category"] == category) & ~frame.index.isin(used)]
        if candidates.empty:
            candidates = frame.loc[~frame.index.isin(used)]
        if category in {"false_negative", "false_positive"}:
            distance = (candidates["predicted_probability"] - threshold).abs()
        else:
            category_median = candidates["predicted_probability"].median()
            distance = (candidates["predicted_probability"] - category_median).abs()
        selected_index = distance.sort_values(kind="stable").index[0]
        used.add(selected_index)
        row = frame.loc[selected_index]
        selections.append(
            {
                "customer_label": label,
                "_row_index": selected_index,
                "actual_churn": int(row["actual_churn"]),
                "predicted_probability": float(row["predicted_probability"]),
                "threshold": float(threshold),
                "predicted_class": int(row["predicted_class"]),
                "correct_prediction": bool(
                    row["actual_churn"] == row["predicted_class"]
                ),
                "confusion_category": str(row["confusion_category"]),
            }
        )
    return pd.DataFrame(selections)


def anonymized_case_summary(
    selected_cases: pd.DataFrame,
    X_test: pd.DataFrame,
    relevant_features: Sequence[str],
) -> pd.DataFrame:
    """Return selected case metadata and values without IDs or dataset indices."""
    predictors = validate_feature_schema(X_test)
    unknown = set(relevant_features) - set(EXPECTED_PREDICTORS)
    if unknown:
        raise ValueError(f"Unknown relevant features: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for _, case in selected_cases.iterrows():
        values = predictors.loc[case["_row_index"]]
        row = {
            key: case[key]
            for key in (
                "customer_label",
                "actual_churn",
                "predicted_probability",
                "threshold",
                "predicted_class",
                "correct_prediction",
                "confusion_category",
            )
        }
        row.update({feature: _display_value(values[feature]) for feature in relevant_features})
        rows.append(row)
    result = pd.DataFrame(rows)
    if IDENTIFIER_COLUMN in result or TARGET_COLUMN in result or "_row_index" in result:
        raise AssertionError("Anonymized output exposed a forbidden field.")
    return result


def summarize_prediction_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize confusion counts and observed attributes by error category."""
    predictors = validate_feature_schema(X_test)
    categories = assign_confusion_categories(
        y_test, probabilities, threshold, index=predictors.index
    )
    counts = (
        categories.value_counts()
        .reindex(CONFUSION_LABELS, fill_value=0)
        .rename_axis("confusion_category")
        .reset_index(name="customer_count")
    )
    numeric_rows: list[dict[str, Any]] = []
    for category in CONFUSION_LABELS:
        subset = predictors.loc[categories == category]
        for feature in numerical_features:
            values = pd.to_numeric(subset[feature], errors="coerce")
            numeric_rows.append(
                {
                    "confusion_category": category,
                    "feature": feature,
                    "non_missing_count": int(values.notna().sum()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "median": float(values.median()) if values.notna().any() else np.nan,
                }
            )
    categorical_rows: list[dict[str, Any]] = []
    for category in CONFUSION_LABELS:
        subset = predictors.loc[categories == category]
        for feature in categorical_features:
            values = subset[feature].astype("string").fillna("Missing")
            counts_by_value = values.value_counts(dropna=False)
            for value, count in counts_by_value.items():
                categorical_rows.append(
                    {
                        "confusion_category": category,
                        "feature": feature,
                        "value": str(value),
                        "customer_count": int(count),
                        "within_category_share": float(count / len(subset))
                        if len(subset)
                        else np.nan,
                    }
                )
    return counts, pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def summarize_observed_feature_effect(
    X: pd.DataFrame,
    probabilities: Sequence[float] | np.ndarray,
    feature: str,
    *,
    numerical_bins: int = 5,
) -> pd.DataFrame:
    """Summarize observed predictions across one raw feature's groups."""
    predictors = validate_feature_schema(X)
    if feature not in EXPECTED_PREDICTORS:
        raise ValueError(f"Unknown feature: {feature}")
    scores = np.asarray(probabilities, dtype=float)
    if len(scores) != len(predictors):
        raise ValueError("Probability count does not match predictor rows.")
    if feature in NUMERICAL_FEATURES:
        values = pd.to_numeric(predictors[feature], errors="coerce")
        groups = pd.qcut(values, q=numerical_bins, duplicates="drop")
    else:
        groups = predictors[feature].astype("string").fillna("Missing")
    summary = (
        pd.DataFrame({"group": groups, "predicted_probability": scores})
        .groupby("group", observed=True)
        .agg(
            customer_count=("predicted_probability", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            median_predicted_probability=("predicted_probability", "median"),
            probability_std=("predicted_probability", "std"),
        )
        .reset_index()
    )
    summary.insert(0, "feature", feature)
    summary["group"] = summary["group"].astype("string")
    return summary


def _display_value(value: Any) -> Any:
    if pd.isna(value):
        return "Missing"
    if isinstance(value, np.generic):
        return value.item()
    return value
