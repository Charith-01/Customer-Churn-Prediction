"""Leakage-safe model development utilities for Phase 5.

All estimators are wrapped around the Phase 4 feature-engineering and
preprocessing steps. Functions in this module operate on a predictor-only
DataFrame and a binary target supplied by the caller; they never load or use
the final test partition.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    ENGINEERED_NUMERICAL_FEATURES,
    build_feature_preprocessing_pipeline,
)
from src.preprocessing import IDENTIFIER_COLUMN, TARGET_COLUMN

RANDOM_STATE = 42
N_SPLITS = 5
CONTACT_FRACTION = 0.20
PRIMARY_SCORING = "average_precision"

MODEL_DISPLAY_NAMES = {
    "dummy": "Naive Prior Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
}

SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
}


def make_stratified_cv() -> StratifiedKFold:
    """Return the single cross-validation definition used throughout Phase 5."""
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_cv_splits(X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize reproducible folds so every model uses identical row splits."""
    _validate_modeling_inputs(X, y)
    return [(train.copy(), valid.copy()) for train, valid in make_stratified_cv().split(X, y)]


def _validate_modeling_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Model predictors must be supplied as a pandas DataFrame.")
    forbidden = {IDENTIFIER_COLUMN, TARGET_COLUMN} & set(X.columns)
    if forbidden:
        raise ValueError(f"Identifier and target cannot be model predictors: {sorted(forbidden)}")
    if len(X) != len(y):
        raise ValueError("Predictor and target row counts do not match.")
    labels = set(pd.Series(y).dropna().astype(int).unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Model development requires both binary labels 0 and 1; found {labels}.")


def get_baseline_estimators() -> dict[str, BaseEstimator]:
    """Create the naive reference and three required baseline classifiers."""
    return {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            l1_ratio=0.0,
            C=1.0,
            class_weight=None,
            max_iter=2_000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.10,
            max_iter=100,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_depth=None,
            class_weight=None,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Append a cloned classifier to the complete Phase 4 transformation chain."""
    transformations = build_feature_preprocessing_pipeline()
    return Pipeline(
        steps=[
            *transformations.steps,
            ("model", clone(estimator)),
        ]
    )


def build_model_pipelines(*, include_dummy: bool = True) -> dict[str, Pipeline]:
    """Build unfitted end-to-end pipelines for the requested estimators."""
    estimators = get_baseline_estimators()
    if not include_dummy:
        estimators.pop("dummy")
    return {name: build_model_pipeline(estimator) for name, estimator in estimators.items()}


def get_search_spaces() -> dict[str, dict[str, Sequence[Any]]]:
    """Return bounded, compatible randomized-search spaces for each ML model."""
    return {
        "logistic_regression": {
            "model__C": [0.03, 0.10, 0.30, 1.0, 3.0, 10.0, 30.0],
            "model__l1_ratio": [0.0, 1.0],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [120, 240, 400],
            "model__max_depth": [None, 5, 10, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", 0.5, None],
            "model__class_weight": [None, "balanced"],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.03, 0.06, 0.10, 0.16],
            "model__max_iter": [80, 150, 250],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
            "model__max_depth": [None, 3, 6],
            "model__class_weight": [None, "balanced"],
        },
    }


def get_search_iterations() -> dict[str, int]:
    """Return the documented number of sampled configurations per estimator."""
    return {
        "logistic_regression": 16,
        "random_forest": 16,
        "hist_gradient_boosting": 16,
    }


def positive_class_probabilities(fitted_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities for label 1 using the fitted classifier's class order."""
    classifier = fitted_pipeline.named_steps["model"]
    classes = np.asarray(classifier.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one positive class label 1; found {classes.tolist()}.")
    probabilities = fitted_pipeline.predict_proba(X)[:, int(matches[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be finite values between 0 and 1.")
    return probabilities


def calculate_top_fraction_metrics(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    *,
    fraction: float = CONTACT_FRACTION,
) -> dict[str, float | int]:
    """Calculate deterministic ranking metrics for the highest-scored fraction.

    The contact count is ``ceil(fraction * n)``. Scores are ordered descending;
    equal scores keep their original validation-fold order via stable sorting.
    """
    y_array = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if y_array.ndim != 1 or scores.ndim != 1 or len(y_array) != len(scores):
        raise ValueError("Targets and probabilities must be equal-length one-dimensional arrays.")
    if len(y_array) == 0:
        raise ValueError("Ranking metrics require at least one row.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")
    if not set(np.unique(y_array)).issubset({0, 1}):
        raise ValueError("Ranking targets must contain only 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Ranking probabilities must be finite values between 0 and 1.")

    contact_count = max(1, math.ceil(fraction * len(y_array)))
    ranked_positions = np.argsort(-scores, kind="stable")
    contacted = y_array[ranked_positions[:contact_count]]
    true_positives = int(contacted.sum())
    positives = int(y_array.sum())
    precision_at_fraction = true_positives / contact_count
    recall_at_fraction = true_positives / positives if positives else 0.0
    prevalence = positives / len(y_array)
    lift = precision_at_fraction / prevalence if prevalence else 0.0
    return {
        "contact_count": contact_count,
        "true_positives_contacted": true_positives,
        "precision_at_20": precision_at_fraction,
        "recall_at_20": recall_at_fraction,
        "prevalence": prevalence,
        "random_precision_baseline": prevalence,
        "random_recall_baseline": contact_count / len(y_array),
        "lift_at_20": lift,
    }


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evaluate one pipeline on fixed folds, including fold-local top-20% ranking."""
    _validate_modeling_inputs(X, y)
    rows: list[dict[str, float | int]] = []
    for fold, (train_positions, valid_positions) in enumerate(cv_splits, start=1):
        fitted = clone(pipeline).fit(X.iloc[train_positions], y.iloc[train_positions])
        X_valid = X.iloc[valid_positions]
        y_valid = y.iloc[valid_positions]
        predictions = fitted.predict(X_valid)
        probabilities = positive_class_probabilities(fitted, X_valid)
        ranking = calculate_top_fraction_metrics(y_valid, probabilities)
        rows.append(
            {
                "fold": fold,
                "validation_rows": len(valid_positions),
                "precision": precision_score(y_valid, predictions, pos_label=1, zero_division=0),
                "recall": recall_score(y_valid, predictions, pos_label=1, zero_division=0),
                "f1": f1_score(y_valid, predictions, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "pr_auc": average_precision_score(y_valid, probabilities),
                "accuracy": accuracy_score(y_valid, predictions),
                **ranking,
            }
        )
    return pd.DataFrame(rows)


def summarize_cv_results(
    fold_results: pd.DataFrame,
    *,
    model_key: str,
    stage: str,
    selected_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize fold metrics into one comparison-table row."""
    return {
        "model_key": model_key,
        "model": MODEL_DISPLAY_NAMES[model_key],
        "stage": stage,
        "mean_precision": fold_results["precision"].mean(),
        "std_precision": fold_results["precision"].std(ddof=1),
        "mean_recall": fold_results["recall"].mean(),
        "std_recall": fold_results["recall"].std(ddof=1),
        "mean_f1": fold_results["f1"].mean(),
        "std_f1": fold_results["f1"].std(ddof=1),
        "mean_roc_auc": fold_results["roc_auc"].mean(),
        "std_roc_auc": fold_results["roc_auc"].std(ddof=1),
        "mean_pr_auc": fold_results["pr_auc"].mean(),
        "std_pr_auc": fold_results["pr_auc"].std(ddof=1),
        "mean_accuracy": fold_results["accuracy"].mean(),
        "std_accuracy": fold_results["accuracy"].std(ddof=1),
        "mean_precision_at_20": fold_results["precision_at_20"].mean(),
        "std_precision_at_20": fold_results["precision_at_20"].std(ddof=1),
        "mean_recall_at_20": fold_results["recall_at_20"].mean(),
        "std_recall_at_20": fold_results["recall_at_20"].std(ddof=1),
        "mean_lift_at_20": fold_results["lift_at_20"].mean(),
        "std_lift_at_20": fold_results["lift_at_20"].std(ddof=1),
        "selected_parameters": json.dumps(
            _json_ready(dict(selected_parameters or {})), sort_keys=True
        ),
    }


def run_hyperparameter_tuning(
    model_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_jobs: int = -1,
) -> tuple[RandomizedSearchCV, pd.DataFrame, float]:
    """Tune one required model on the fixed Phase 5 folds using PR-AUC."""
    if model_key not in get_search_spaces():
        raise ValueError(f"No Phase 5 search space is defined for '{model_key}'.")
    _validate_modeling_inputs(X, y)
    pipeline = build_model_pipelines(include_dummy=False)[model_key]
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_search_spaces()[model_key],
        n_iter=get_search_iterations()[model_key],
        scoring=SCORING,
        refit=PRIMARY_SCORING,
        cv=cv_splits,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X, y)
    elapsed_seconds = time.perf_counter() - started
    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_average_precision", kind="stable"
    )
    return search, results, elapsed_seconds


def generate_oof_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Generate one probability per training row from a model that excluded it."""
    _validate_modeling_inputs(X, y)
    probabilities = np.full(len(X), np.nan, dtype=float)
    fold_numbers = np.full(len(X), -1, dtype=int)
    for fold, (train_positions, valid_positions) in enumerate(cv_splits, start=1):
        fitted = clone(pipeline).fit(X.iloc[train_positions], y.iloc[train_positions])
        probabilities[valid_positions] = positive_class_probabilities(
            fitted, X.iloc[valid_positions]
        )
        fold_numbers[valid_positions] = fold
    if np.isnan(probabilities).any() or (fold_numbers < 1).any():
        raise AssertionError("Every training row must receive exactly one OOF prediction.")
    return pd.DataFrame(
        {
            "dataset_row_index": X.index.to_numpy(),
            "cv_fold": fold_numbers,
            "oof_churn_probability": probabilities,
        }
    )


def fit_and_save_candidate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str | Path,
) -> Pipeline:
    """Fit the provisional pipeline on training data only and serialize it."""
    _validate_modeling_inputs(X, y)
    fitted = clone(pipeline).fit(X, y)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    return fitted


def save_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Save aggregate, non-sensitive development metadata as readable JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_model_metadata(
    *,
    model_key: str,
    best_parameters: Mapping[str, Any],
    dataset_fingerprint: str,
    training_rows: int,
) -> dict[str, Any]:
    """Create non-sensitive metadata for the local provisional artifact."""
    return {
        "phase": 5,
        "artifact_role": "provisional candidate; not final test validated",
        "model_key": model_key,
        "model_name": MODEL_DISPLAY_NAMES[model_key],
        "best_parameters": dict(best_parameters),
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "feature_engineering": list(ENGINEERED_NUMERICAL_FEATURES),
        "training_dataset_fingerprint": dataset_fingerprint,
        "training_rows": training_rows,
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "primary_scoring": PRIMARY_SCORING,
        "contact_fraction": CONTACT_FRACTION,
        "final_test_used": False,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
