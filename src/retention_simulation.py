"""Capacity-constrained retention simulation for the locked churn model.

This module ranks already-generated churn probabilities and evaluates a
hypothetical retention campaign. It never fits, calibrates, or modifies a
model. Actual outcomes are used only after ranking for retrospective analysis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#3569D4"
PURPLE = "#8051D6"
DEFAULT_CONTACT_FRACTION = 0.20
DEFAULT_CONTACT_COST = 5.0
DEFAULT_RETAINED_VALUE = 150.0
DEFAULT_SUCCESS_RATE = 0.10


def validate_churn_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return a validated copy of one-dimensional churn probabilities."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Probabilities must be a non-empty one-dimensional sequence.")
    if expected_length is not None and len(scores) != expected_length:
        raise ValueError("Probability count does not match the expected row count.")
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities cannot contain missing or infinite values.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1 inclusive.")
    return scores.copy()


def validate_binary_outcomes(
    outcomes: Sequence[int] | pd.Series,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return a validated copy of binary retrospective outcomes."""
    target = np.asarray(outcomes)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("Outcomes must be a non-empty one-dimensional sequence.")
    if expected_length is not None and len(target) != expected_length:
        raise ValueError("Outcome count does not match the expected row count.")
    if pd.isna(target).any():
        raise ValueError("Outcomes cannot contain missing values.")
    try:
        target = target.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Outcomes must contain binary labels 0 and 1.") from exc
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Outcomes must contain only binary labels 0 and 1.")
    return target.copy()


def rank_customer_risks(
    probabilities: Sequence[float] | np.ndarray,
    *,
    internal_references: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rank churn risks descending with original order as a stable tie-breaker.

    No outcome or customer identifier is accepted, so selection cannot depend
    on the retrospective target. Generated references encode only held-out row
    order and are not source-data indices or customer IDs.
    """
    scores = validate_churn_probabilities(probabilities)
    row_count = len(scores)
    if internal_references is None:
        width = max(4, len(str(row_count)))
        references = [f"heldout_row_{position + 1:0{width}d}" for position in range(row_count)]
    else:
        references = [str(value) for value in internal_references]
        if len(references) != row_count:
            raise ValueError("Internal reference count does not match probabilities.")
        if len(set(references)) != row_count:
            raise ValueError("Internal references must be unique.")

    ranked = pd.DataFrame(
        {
            "internal_row_reference": references,
            "heldout_order": np.arange(row_count, dtype=int),
            "predicted_churn_probability": scores,
        }
    ).sort_values(
        ["predicted_churn_probability", "heldout_order"],
        ascending=[False, True],
        kind="stable",
    )
    ranked = ranked.reset_index(drop=True)
    ranked["risk_rank"] = np.arange(1, row_count + 1, dtype=int)
    ranked["score_tie_size"] = ranked.groupby(
        "predicted_churn_probability", sort=False
    )["predicted_churn_probability"].transform("size")
    return ranked


def determine_contact_capacity(
    total_customers: int,
    contact_fraction: float = DEFAULT_CONTACT_FRACTION,
) -> int:
    """Return floor(N * fraction), never exceeding the available population."""
    if isinstance(total_customers, bool) or int(total_customers) != total_customers:
        raise ValueError("total_customers must be a non-negative integer.")
    total = int(total_customers)
    if total < 0:
        raise ValueError("total_customers must be a non-negative integer.")
    if not np.isfinite(contact_fraction) or not 0 <= contact_fraction <= 1:
        raise ValueError("contact_fraction must be between 0 and 1 inclusive.")
    return min(total, math.floor(total * float(contact_fraction)))


def select_highest_risk_customers(
    ranked_customers: pd.DataFrame,
    *,
    contact_fraction: float = DEFAULT_CONTACT_FRACTION,
) -> pd.DataFrame:
    """Flag exactly the capacity-safe highest-ranked customers."""
    required = {
        "internal_row_reference",
        "heldout_order",
        "predicted_churn_probability",
        "risk_rank",
    }
    if required - set(ranked_customers.columns):
        raise ValueError("Ranked customer table is missing required columns.")
    selected = ranked_customers.copy(deep=True)
    expected_ranks = np.arange(1, len(selected) + 1)
    if not np.array_equal(selected["risk_rank"].to_numpy(), expected_ranks):
        raise ValueError("Ranked customer table must be in consecutive risk-rank order.")
    contact_count = determine_contact_capacity(len(selected), contact_fraction)
    selected["selected_top_20pct"] = selected["risk_rank"] <= contact_count
    return selected


def build_internal_analysis_table(
    probabilities: Sequence[float] | np.ndarray,
    actual_outcomes: Sequence[int] | pd.Series,
    locked_threshold_predictions: Sequence[int] | np.ndarray,
    *,
    contact_fraction: float = DEFAULT_CONTACT_FRACTION,
) -> pd.DataFrame:
    """Build the protected row-level retrospective analysis table.

    Ranking and selection occur before outcome columns are attached. This
    prevents the observed churn label from influencing campaign priority.
    """
    scores = validate_churn_probabilities(probabilities)
    outcomes = validate_binary_outcomes(actual_outcomes, expected_length=len(scores))
    predictions = validate_binary_outcomes(
        locked_threshold_predictions, expected_length=len(scores)
    )
    ranked = select_highest_risk_customers(
        rank_customer_risks(scores), contact_fraction=contact_fraction
    )
    source_positions = ranked["heldout_order"].to_numpy(dtype=int)
    ranked["actual_churn"] = outcomes[source_positions]
    ranked["locked_threshold_prediction"] = predictions[source_positions]
    return ranked


def probability_ranking_summary(
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Summarize score distribution and exact ties without altering scores."""
    scores = validate_churn_probabilities(probabilities)
    values, counts = np.unique(scores, return_counts=True)
    tied = counts > 1
    return {
        "total_customers": int(len(scores)),
        "minimum_probability": float(scores.min()),
        "maximum_probability": float(scores.max()),
        "mean_probability": float(scores.mean()),
        "median_probability": float(np.median(scores)),
        "unique_probabilities": int(len(values)),
        "tied_score_groups": int(tied.sum()),
        "customers_with_tied_scores": int(counts[tied].sum()),
    }


def calculate_selection_metrics(
    actual_outcomes: Sequence[int] | pd.Series,
    selected_flags: Sequence[bool] | np.ndarray,
) -> dict[str, float | int]:
    """Calculate capacity, capture, precision, recall, prevalence, and lift."""
    outcomes = validate_binary_outcomes(actual_outcomes)
    selected = np.asarray(selected_flags)
    if selected.ndim != 1 or len(selected) != len(outcomes):
        raise ValueError("Selection flags must align with outcomes.")
    if not set(np.unique(selected)).issubset({False, True, 0, 1}):
        raise ValueError("Selection flags must be boolean.")
    selected = selected.astype(bool)

    total = len(outcomes)
    selected_count = int(selected.sum())
    total_churners = int(outcomes.sum())
    selected_churners = int(outcomes[selected].sum())
    selected_non_churners = selected_count - selected_churners
    outside_churners = total_churners - selected_churners
    prevalence = total_churners / total
    precision = selected_churners / selected_count if selected_count else 0.0
    recall = selected_churners / total_churners if total_churners else 0.0
    lift = precision / prevalence if prevalence else 0.0
    return {
        "total_customers": total,
        "selected_customers": selected_count,
        "contact_rate": selected_count / total,
        "actual_selected_percentage": selected_count / total * 100,
        "total_actual_churners": total_churners,
        "selected_churners": selected_churners,
        "selected_non_churners": selected_non_churners,
        "outside_churners": outside_churners,
        "full_sample_churn_prevalence": prevalence,
        "selected_group_churn_prevalence": precision,
        "precision_at_20": precision,
        "recall_at_20": recall,
        "capture_rate_at_20": recall,
        "lift_at_20": lift,
    }


def exact_random_selection_expectations(
    total_customers: int,
    contact_count: int,
    total_churners: int,
) -> dict[str, float | int]:
    """Calculate exact expectations for uniform sampling without replacement."""
    values = (total_customers, contact_count, total_churners)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("Population, contact, and churner counts must be integers.")
    total, contacts, churners = map(int, values)
    if total < 0 or contacts < 0 or churners < 0:
        raise ValueError("Population, contact, and churner counts cannot be negative.")
    if contacts > total or churners > total:
        raise ValueError("Contacts and churners cannot exceed the population.")
    if total == 0:
        return {
            "total_customers": 0,
            "contact_count": 0,
            "expected_churners_captured": 0.0,
            "expected_random_precision": 0.0,
            "expected_random_recall": 0.0,
            "expected_random_lift": 0.0,
        }
    prevalence = churners / total
    expected_captured = contacts * prevalence
    precision = prevalence if contacts else 0.0
    recall = contacts / total if churners else 0.0
    return {
        "total_customers": total,
        "contact_count": contacts,
        "expected_churners_captured": expected_captured,
        "expected_random_precision": precision,
        "expected_random_recall": recall,
        "expected_random_lift": 1.0 if contacts and churners else 0.0,
    }


def simulate_random_selection(
    actual_outcomes: Sequence[int] | pd.Series,
    contact_count: int,
    *,
    n_simulations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run reproducible equal-budget random selections without replacement."""
    outcomes = validate_binary_outcomes(actual_outcomes)
    if isinstance(contact_count, bool) or int(contact_count) != contact_count:
        raise ValueError("contact_count must be an integer.")
    contacts = int(contact_count)
    if not 0 <= contacts <= len(outcomes):
        raise ValueError("contact_count must be between zero and population size.")
    if isinstance(n_simulations, bool) or int(n_simulations) != n_simulations:
        raise ValueError("n_simulations must be a positive integer.")
    simulations = int(n_simulations)
    if simulations < 1:
        raise ValueError("n_simulations must be a positive integer.")

    rng = np.random.default_rng(random_state)
    captured = np.zeros(simulations, dtype=int)
    if contacts:
        for simulation in range(simulations):
            positions = rng.choice(len(outcomes), size=contacts, replace=False)
            captured[simulation] = int(outcomes[positions].sum())
    total_churners = int(outcomes.sum())
    return pd.DataFrame(
        {
            "simulation": np.arange(1, simulations + 1, dtype=int),
            "churners_captured": captured,
            "precision": captured / contacts if contacts else np.zeros(simulations),
            "recall": captured / total_churners
            if total_churners
            else np.zeros(simulations),
        }
    )


def calculate_contact_cost(contact_count: int, cost_per_contact: float) -> float:
    """Calculate campaign contact cost."""
    if contact_count < 0 or int(contact_count) != contact_count:
        raise ValueError("contact_count must be a non-negative integer.")
    if not np.isfinite(cost_per_contact) or cost_per_contact < 0:
        raise ValueError("cost_per_contact must be a finite non-negative value.")
    return float(contact_count * cost_per_contact)


def calculate_expected_retained_customers(
    captured_churners: float,
    intervention_success_rate: float,
) -> float:
    """Calculate fractional expected retained customers under an assumption."""
    if not np.isfinite(captured_churners) or captured_churners < 0:
        raise ValueError("captured_churners must be finite and non-negative.")
    if not np.isfinite(intervention_success_rate) or not 0 <= intervention_success_rate <= 1:
        raise ValueError("intervention_success_rate must be between 0 and 1.")
    return float(captured_churners * intervention_success_rate)


def calculate_expected_gross_value(
    expected_retained_customers: float,
    value_per_retained_customer: float,
) -> float:
    """Calculate hypothetical gross retained-customer value."""
    if not np.isfinite(expected_retained_customers) or expected_retained_customers < 0:
        raise ValueError("expected_retained_customers must be finite and non-negative.")
    if not np.isfinite(value_per_retained_customer) or value_per_retained_customer < 0:
        raise ValueError("value_per_retained_customer must be finite and non-negative.")
    return float(expected_retained_customers * value_per_retained_customer)


def calculate_expected_net_benefit(gross_value: float, contact_cost: float) -> float:
    """Calculate hypothetical net benefit after contact cost."""
    if not np.isfinite(gross_value) or not np.isfinite(contact_cost):
        raise ValueError("Gross value and contact cost must be finite.")
    if gross_value < 0 or contact_cost < 0:
        raise ValueError("Gross value and contact cost cannot be negative.")
    return float(gross_value - contact_cost)


def calculate_roi(net_benefit: float, contact_cost: float) -> float:
    """Return percentage ROI, using zero when no campaign cost exists."""
    if not np.isfinite(net_benefit) or not np.isfinite(contact_cost):
        raise ValueError("Net benefit and contact cost must be finite.")
    if contact_cost < 0:
        raise ValueError("contact_cost cannot be negative.")
    return float(net_benefit / contact_cost * 100) if contact_cost else 0.0


def calculate_retention_economics(
    captured_churners: float,
    contact_count: int,
    *,
    cost_per_contact: float = DEFAULT_CONTACT_COST,
    value_per_retained_customer: float = DEFAULT_RETAINED_VALUE,
    intervention_success_rate: float = DEFAULT_SUCCESS_RATE,
) -> dict[str, float | bool]:
    """Calculate one strategy's hypothetical retention economics."""
    contact_cost = calculate_contact_cost(contact_count, cost_per_contact)
    retained = calculate_expected_retained_customers(
        captured_churners, intervention_success_rate
    )
    gross_value = calculate_expected_gross_value(retained, value_per_retained_customer)
    net_benefit = calculate_expected_net_benefit(gross_value, contact_cost)
    return {
        "captured_churners": float(captured_churners),
        "contact_count": int(contact_count),
        "cost_per_contact": float(cost_per_contact),
        "value_per_retained_customer": float(value_per_retained_customer),
        "intervention_success_rate": float(intervention_success_rate),
        "total_contact_cost": contact_cost,
        "expected_retained_customers": retained,
        "expected_gross_value": gross_value,
        "expected_net_benefit": net_benefit,
        "roi_percent": calculate_roi(net_benefit, contact_cost),
        "roi_defined": bool(contact_cost > 0),
    }


def calculate_break_even_success_rate(
    captured_churners: float,
    contact_count: int,
    *,
    cost_per_contact: float = DEFAULT_CONTACT_COST,
    value_per_retained_customer: float = DEFAULT_RETAINED_VALUE,
) -> float:
    """Return the assumed success rate needed for zero expected net benefit."""
    cost = calculate_contact_cost(contact_count, cost_per_contact)
    if captured_churners < 0 or not np.isfinite(captured_churners):
        raise ValueError("captured_churners must be finite and non-negative.")
    if value_per_retained_customer < 0 or not np.isfinite(value_per_retained_customer):
        raise ValueError("value_per_retained_customer must be finite and non-negative.")
    denominator = float(captured_churners * value_per_retained_customer)
    return float(cost / denominator) if denominator else float("nan")


def run_sensitivity_scenarios(
    model_captured_churners: float,
    random_expected_churners: float,
    contact_count: int,
    *,
    success_rates: Sequence[float] = (0.05, 0.10, 0.20),
    contact_costs: Sequence[float] = (2.0, 5.0, 10.0),
    retained_values: Sequence[float] = (75.0, 150.0, 300.0),
) -> pd.DataFrame:
    """Evaluate a full grid of hypothetical effectiveness, cost, and value."""
    rows: list[dict[str, Any]] = []
    scenario_number = 0
    for success_rate in success_rates:
        for cost_per_contact in contact_costs:
            for retained_value in retained_values:
                scenario_number += 1
                model = calculate_retention_economics(
                    model_captured_churners,
                    contact_count,
                    cost_per_contact=cost_per_contact,
                    value_per_retained_customer=retained_value,
                    intervention_success_rate=success_rate,
                )
                random = calculate_retention_economics(
                    random_expected_churners,
                    contact_count,
                    cost_per_contact=cost_per_contact,
                    value_per_retained_customer=retained_value,
                    intervention_success_rate=success_rate,
                )
                rows.append(
                    {
                        "scenario_id": scenario_number,
                        "intervention_success_rate": float(success_rate),
                        "cost_per_contact": float(cost_per_contact),
                        "value_per_retained_customer": float(retained_value),
                        "model_expected_retained_customers": model[
                            "expected_retained_customers"
                        ],
                        "random_expected_retained_customers": random[
                            "expected_retained_customers"
                        ],
                        "model_expected_net_benefit": model["expected_net_benefit"],
                        "random_expected_net_benefit": random["expected_net_benefit"],
                        "incremental_expected_net_benefit": (
                            model["expected_net_benefit"]
                            - random["expected_net_benefit"]
                        ),
                        "model_roi_percent": model["roi_percent"],
                        "random_roi_percent": random["roi_percent"],
                        "model_financially_attractive": model["expected_net_benefit"] > 0,
                        "random_financially_attractive": random["expected_net_benefit"] > 0,
                    }
                )
    return pd.DataFrame(rows)


def cumulative_gains_data(
    actual_outcomes: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return model and random cumulative churner-capture curves."""
    scores = validate_churn_probabilities(probabilities)
    outcomes = validate_binary_outcomes(actual_outcomes, expected_length=len(scores))
    order = np.argsort(-scores, kind="stable")
    ranked_outcomes = outcomes[order]
    total_churners = int(outcomes.sum())
    model_capture = (
        np.cumsum(ranked_outcomes) / total_churners
        if total_churners
        else np.zeros(len(outcomes))
    )
    selected_fraction = np.arange(1, len(outcomes) + 1) / len(outcomes)
    return pd.DataFrame(
        {
            "selected_fraction": np.concatenate(([0.0], selected_fraction)),
            "model_capture_rate": np.concatenate(([0.0], model_capture)),
            "random_capture_rate": np.concatenate(([0.0], selected_fraction)),
        }
    )


def create_strategy_comparison_table(
    model_metrics: Mapping[str, float | int],
    random_expectations: Mapping[str, float | int],
) -> pd.DataFrame:
    """Create an equal-budget model-versus-random comparison table."""
    contacts = int(model_metrics["selected_customers"])
    return pd.DataFrame(
        [
            {
                "strategy": "Model-ranked top 20%",
                "contact_count": contacts,
                "churners_captured": float(model_metrics["selected_churners"]),
                "precision_at_20": float(model_metrics["precision_at_20"]),
                "recall_at_20": float(model_metrics["recall_at_20"]),
                "lift_vs_random": float(model_metrics["lift_at_20"]),
            },
            {
                "strategy": "Uniform random expectation",
                "contact_count": int(random_expectations["contact_count"]),
                "churners_captured": float(
                    random_expectations["expected_churners_captured"]
                ),
                "precision_at_20": float(
                    random_expectations["expected_random_precision"]
                ),
                "recall_at_20": float(random_expectations["expected_random_recall"]),
                "lift_vs_random": float(random_expectations["expected_random_lift"]),
            },
        ]
    )


def plot_strategy_comparison(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Plot captured churners, precision, and recall for both strategies."""
    required = {
        "strategy",
        "churners_captured",
        "precision_at_20",
        "recall_at_20",
    }
    if required - set(comparison.columns):
        raise ValueError("Strategy comparison is missing required columns.")
    colors = [PURPLE, BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specifications = (
        ("churners_captured", "Churners captured", "Customers"),
        ("precision_at_20", "Precision@20%", "Precision"),
        ("recall_at_20", "Recall / capture@20%", "Recall"),
    )
    labels = ["Model-ranked", "Random expectation"]
    for ax, (column, title, ylabel) in zip(axes, specifications):
        bars = ax.bar(labels, comparison[column], color=colors, alpha=0.9)
        ax.set(title=title, ylabel=ylabel)
        if column != "churners_captured":
            ax.set_ylim(0, 1.05)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.suptitle("Equal-Budget Retention Targeting Comparison", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_cumulative_gains(gains: pd.DataFrame, output_path: str | Path) -> None:
    """Plot cumulative model capture, random reference, and 20% capacity."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        gains["selected_fraction"],
        gains["model_capture_rate"],
        color=PURPLE,
        linewidth=2.5,
        label="Model-ranked capture",
    )
    ax.plot(
        gains["selected_fraction"],
        gains["random_capture_rate"],
        color=BLUE,
        linestyle="--",
        linewidth=2,
        label="Random expectation",
    )
    ax.axvline(0.20, color="black", linestyle=":", label="20% contact capacity")
    ax.set(
        title="Cumulative Churner Capture by Contact Fraction",
        xlabel="Fraction of customers selected",
        ylabel="Fraction of actual churners captured",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_lift_comparison(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Plot model lift against the random-selection reference of one."""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["Model-ranked", "Random expectation"],
        comparison["lift_vs_random"],
        color=[PURPLE, BLUE],
        alpha=0.9,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.set(title="Lift at the 20% Contact Limit", ylabel="Lift versus random")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_financial_comparison(economics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot hypothetical base-case gross value, cost, and net benefit."""
    required = {"strategy", "expected_gross_value", "total_contact_cost", "expected_net_benefit"}
    if required - set(economics.columns):
        raise ValueError("Economics table is missing required columns.")
    plotted = economics.set_index("strategy")
    ax = plotted[["expected_gross_value", "total_contact_cost", "expected_net_benefit"]].plot(
        kind="bar",
        color=[PURPLE, BLUE, "#5B3AA8"],
        figsize=(10, 6),
        rot=0,
    )
    ax.set(title="Hypothetical Base-Case Retention Economics", ylabel="USD (illustrative)")
    ax.legend(["Gross value", "Contact cost", "Net benefit"])
    ax.figure.tight_layout()
    _save_figure(ax.figure, output_path)


def plot_sensitivity_analysis(
    sensitivity: pd.DataFrame,
    output_path: str | Path,
    *,
    base_cost_per_contact: float = DEFAULT_CONTACT_COST,
) -> None:
    """Plot net benefit versus effectiveness at the base contact cost."""
    subset = sensitivity.loc[
        np.isclose(sensitivity["cost_per_contact"], base_cost_per_contact)
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    blue_purple_palette = (BLUE, PURPLE, "#5B3AA8")
    grouped = subset.groupby("value_per_retained_customer", sort=True)
    for color, (retained_value, group) in zip(blue_purple_palette, grouped):
        group = group.sort_values("intervention_success_rate")
        ax.plot(
            group["intervention_success_rate"] * 100,
            group["model_expected_net_benefit"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"Model, value=${retained_value:,.0f}",
        )
        ax.plot(
            group["intervention_success_rate"] * 100,
            group["random_expected_net_benefit"],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            color=color,
            alpha=0.75,
            label=f"Random, value=${retained_value:,.0f}",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title=f"Retention Economics Sensitivity (Contact Cost ${base_cost_per_contact:,.0f})",
        xlabel="Hypothetical intervention success rate (%)",
        ylabel="Expected net benefit (illustrative USD)",
    )
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_path)


def save_aggregate_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save a non-customer-level aggregate result table."""
    forbidden = {
        "customer_id",
        "internal_row_reference",
        "heldout_order",
        "actual_churn",
        "predicted_churn_probability",
    }
    exposed = forbidden & set(results.columns)
    if exposed:
        raise ValueError(f"Aggregate output contains customer-level fields: {sorted(exposed)}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

