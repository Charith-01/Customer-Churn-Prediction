"""Reusable, leakage-safe evaluation utilities for Phase 6.

The functions in this module do not load project data. Callers must provide
the fixed training or test partitions explicitly. Calibration comparison uses
outer-fold predictions, while each calibrated candidate is fitted only inside
the corresponding outer training fold.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.modeling import RANDOM_STATE

DEFAULT_CONTACT_CAPACITY = 0.20
DEFAULT_THRESHOLD = 0.50
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def _as_binary_target(y_true: Sequence[int] | pd.Series) -> np.ndarray:
    target = np.asarray(y_true)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("The target must be a non-empty one-dimensional sequence.")
    if pd.isna(target).any():
        raise ValueError("The target cannot contain missing values.")
    try:
        target = target.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("The target must contain binary labels 0 and 1.") from exc
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("The target must contain only binary labels 0 and 1.")
    return target


def _as_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Probabilities must be a non-empty one-dimensional sequence.")
    if expected_length is not None and len(scores) != expected_length:
        raise ValueError("Target and probability lengths do not match.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities cannot contain missing or infinite values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1 inclusive.")
    return scores


def positive_class_probabilities(
    fitted_estimator: BaseEstimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return the probability for churn label 1 using estimator class order."""
    if not hasattr(fitted_estimator, "predict_proba"):
        raise TypeError("The fitted estimator must implement predict_proba.")
    classes = np.asarray(getattr(fitted_estimator, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one positive class label 1; found {classes.tolist()}."
        )
    probabilities = fitted_estimator.predict_proba(X)[:, int(matches[0])]
    return _as_probabilities(probabilities, expected_length=len(X))


def predictions_at_threshold(
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert probabilities to labels; scores equal to the threshold are positive."""
    scores = _as_probabilities(probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1.")
    return (scores >= float(threshold)).astype(int)


def compute_classification_metrics(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | int]:
    """Compute discrimination, probability, and threshold-dependent metrics."""
    target = _as_binary_target(y_true)
    scores = _as_probabilities(probabilities, expected_length=len(target))
    if len(np.unique(target)) != 2:
        raise ValueError("ROC-AUC and Average Precision require both target classes.")
    predictions = predictions_at_threshold(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "observations": int(len(target)),
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, scores, labels=[0, 1])),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def compute_threshold_metrics(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Calculate classification and contact-volume behavior by threshold."""
    target = _as_binary_target(y_true)
    scores = _as_probabilities(probabilities, expected_length=len(target))
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Every threshold must be between 0 and 1.")
        predictions = predictions_at_threshold(scores, float(threshold))
        tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive_count": int(predictions.sum()),
                "predicted_positive_rate": float(predictions.mean()),
                "precision": float(
                    precision_score(target, predictions, zero_division=0)
                ),
                "recall": float(recall_score(target, predictions, zero_division=0)),
                "f1": float(f1_score(target, predictions, zero_division=0)),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "true_negative": int(tn),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold", kind="stable").reset_index(drop=True)


def select_operating_threshold(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    *,
    capacity: float = DEFAULT_CONTACT_CAPACITY,
) -> dict[str, float | int | str]:
    """Select a fixed threshold using OOF labels under a contact-rate cap.

    All unique score values are candidates. A threshold includes every score
    equal to it, so tied customers are never split. Eligible candidates must
    contact no more than ``capacity``. Selection maximizes recall, then
    precision, then uses fewer contacts, then the higher threshold.
    """
    target = _as_binary_target(y_true)
    scores = _as_probabilities(probabilities, expected_length=len(target))
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in the interval (0, 1].")
    candidates = np.unique(scores)
    metrics = compute_threshold_metrics(target, scores, candidates)
    eligible = metrics.loc[metrics["predicted_positive_rate"] <= capacity + 1e-12].copy()
    if eligible.empty:
        raise ValueError("No observed-score threshold satisfies the contact capacity.")
    selected = eligible.sort_values(
        ["recall", "precision", "predicted_positive_count", "threshold"],
        ascending=[False, False, True, False],
        kind="stable",
    ).iloc[0]
    result = selected.to_dict()
    result.update(
        {
            "capacity": float(capacity),
            "selection_method": (
                "maximize OOF recall subject to contact rate <= capacity; "
                "break ties by precision, fewer contacts, then higher threshold"
            ),
            "tie_handling": (
                "probability >= threshold; all exact-score ties are included together"
            ),
        }
    )
    return result


def calibration_diagnostics(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Return probability-quality metrics and an equal-frequency reliability table."""
    target = _as_binary_target(y_true)
    scores = _as_probabilities(probabilities, expected_length=len(target))
    if len(np.unique(target)) != 2:
        raise ValueError("Calibration diagnostics require both target classes.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    quantiles = np.linspace(0, 1, min(n_bins, len(scores)) + 1)
    edges = np.unique(np.quantile(scores, quantiles))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0] = min(0.0, edges[0])
    edges[-1] = max(1.0, edges[-1])
    bin_ids = pd.cut(scores, bins=edges, include_lowest=True, duplicates="drop")
    reliability = (
        pd.DataFrame({"target": target, "probability": scores, "bin": bin_ids})
        .groupby("bin", observed=True)
        .agg(
            sample_count=("target", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_churn_rate=("target", "mean"),
            probability_min=("probability", "min"),
            probability_max=("probability", "max"),
        )
        .reset_index(drop=True)
    )
    reliability.insert(0, "bin_number", np.arange(1, len(reliability) + 1))
    reliability["absolute_gap"] = (
        reliability["mean_predicted_probability"]
        - reliability["observed_churn_rate"]
    ).abs()
    ece = float(
        np.average(reliability["absolute_gap"], weights=reliability["sample_count"])
    )
    summary = {
        "observations": int(len(target)),
        "bins": int(len(reliability)),
        "prevalence": float(target.mean()),
        "mean_predicted_probability": float(scores.mean()),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, scores, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "expected_calibration_error": ece,
    }
    return summary, reliability


def generate_nested_calibration_oof(
    base_estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    methods: Sequence[str] = CALIBRATION_METHODS,
    inner_splits: int = 3,
) -> pd.DataFrame:
    """Generate leakage-safe outer-fold probabilities for calibration candidates."""
    target = _as_binary_target(y)
    if len(X) != len(target):
        raise ValueError("Predictor and target row counts do not match.")
    unsupported = set(methods) - set(CALIBRATION_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported calibration methods: {sorted(unsupported)}")
    rows: list[pd.DataFrame] = []
    for fold, (train_positions, valid_positions) in enumerate(outer_splits, start=1):
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        X_valid = X.iloc[valid_positions]
        for method in methods:
            if method == "uncalibrated":
                estimator = clone(base_estimator)
            else:
                inner_cv = StratifiedKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=RANDOM_STATE + fold,
                )
                estimator = CalibratedClassifierCV(
                    estimator=clone(base_estimator),
                    method=method,
                    cv=inner_cv,
                    ensemble=True,
                    n_jobs=1,
                )
            estimator.fit(X_train, y_train)
            probabilities = positive_class_probabilities(estimator, X_valid)
            rows.append(
                pd.DataFrame(
                    {
                        "dataset_row_index": X.index.to_numpy()[valid_positions],
                        "cv_fold": fold,
                        "method": method,
                        "probability": probabilities,
                    }
                )
            )
    result = pd.concat(rows, ignore_index=True)
    expected = len(X) * len(methods)
    if len(result) != expected:
        raise AssertionError("Calibration OOF output has incomplete row coverage.")
    duplicates = result.duplicated(["dataset_row_index", "method"]).any()
    if duplicates:
        raise AssertionError("Each row must have one OOF probability per method.")
    return result


def summarize_calibration_methods(
    y: pd.Series,
    calibration_oof: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Summarize nested OOF calibration candidates against row-indexed targets."""
    target_by_index = pd.Series(np.asarray(y, dtype=int), index=y.index)
    summaries: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}
    for method, group in calibration_oof.groupby("method", sort=False):
        ordered = group.sort_values("dataset_row_index", kind="stable")
        aligned_target = target_by_index.loc[ordered["dataset_row_index"]].to_numpy()
        summary, reliability = calibration_diagnostics(
            aligned_target, ordered["probability"].to_numpy(), n_bins=n_bins
        )
        summaries.append({"method": method, **summary})
        tables[method] = reliability
    return pd.DataFrame(summaries), tables


def choose_calibration_method(
    comparison: pd.DataFrame,
    *,
    min_brier_improvement: float = 0.002,
    max_ranking_drop: float = 0.002,
) -> dict[str, Any]:
    """Choose calibration by a prespecified probability/ranking safeguard rule."""
    required = {
        "method",
        "brier_score",
        "log_loss",
        "average_precision",
        "roc_auc",
        "expected_calibration_error",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Calibration comparison is missing columns: {sorted(missing)}")
    if "uncalibrated" not in set(comparison["method"]):
        raise ValueError("Calibration comparison must include an uncalibrated reference.")
    baseline = comparison.set_index("method").loc["uncalibrated"]
    candidates = comparison.loc[comparison["method"] != "uncalibrated"].copy()
    candidates["brier_improvement"] = baseline["brier_score"] - candidates["brier_score"]
    candidates["ap_drop"] = baseline["average_precision"] - candidates["average_precision"]
    candidates["roc_drop"] = baseline["roc_auc"] - candidates["roc_auc"]
    eligible = candidates.loc[
        (candidates["brier_improvement"] >= min_brier_improvement)
        & (candidates["log_loss"] < baseline["log_loss"])
        & (candidates["expected_calibration_error"] < baseline["expected_calibration_error"])
        & (candidates["ap_drop"] <= max_ranking_drop)
        & (candidates["roc_drop"] <= max_ranking_drop)
    ]
    if eligible.empty:
        return {
            "method": "uncalibrated",
            "decision": "retain uncalibrated probabilities",
            "rule": (
                "require Brier improvement >= 0.002, lower log loss and ECE, "
                "and AP/ROC-AUC degradation <= 0.002"
            ),
        }
    winner = eligible.sort_values(
        ["brier_score", "log_loss", "expected_calibration_error"], kind="stable"
    ).iloc[0]
    return {
        "method": str(winner["method"]),
        "decision": f"adopt {winner['method']} calibration",
        "rule": (
            "require Brier improvement >= 0.002, lower log loss and ECE, "
            "and AP/ROC-AUC degradation <= 0.002; choose lowest Brier score"
        ),
        "brier_improvement": float(winner["brier_improvement"]),
    }


def build_probability_estimator(
    base_estimator: BaseEstimator,
    calibration_method: str,
    *,
    calibration_splits: int = 5,
) -> BaseEstimator:
    """Build the locked probability-producing estimator for full-training fit."""
    if calibration_method == "uncalibrated":
        return clone(base_estimator)
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be uncalibrated, sigmoid, or isotonic.")
    cv = StratifiedKFold(
        n_splits=calibration_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(
        estimator=clone(base_estimator),
        method=calibration_method,
        cv=cv,
        ensemble=True,
        n_jobs=1,
    )


def curve_data(
    y_true: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return ROC and precision-recall curve coordinates and scalar summaries."""
    target = _as_binary_target(y_true)
    scores = _as_probabilities(probabilities, expected_length=len(target))
    if len(np.unique(target)) != 2:
        raise ValueError("Curve calculation requires both target classes.")
    fpr, tpr, roc_thresholds = roc_curve(target, scores)
    precision, recall, pr_thresholds = precision_recall_curve(target, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "prevalence": float(target.mean()),
    }


def robustness_summary(
    fold_results: pd.DataFrame,
    metrics: Sequence[str] = ("precision", "recall", "roc_auc", "pr_auc"),
) -> pd.DataFrame:
    """Summarize Phase 5 fold metrics with mean, SD, minimum, and maximum."""
    missing = set(metrics) - set(fold_results.columns)
    if missing:
        raise ValueError(f"Fold results are missing metrics: {sorted(missing)}")
    rows = []
    for metric in metrics:
        values = pd.to_numeric(fold_results[metric], errors="raise")
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "folds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def save_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Save JSON-compatible evaluation metadata with NumPy values normalized."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
