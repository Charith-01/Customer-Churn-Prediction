"""Tests for capacity-safe, label-independent retention simulation."""

from __future__ import annotations

import numpy as np

from src.retention_simulation import (
    build_internal_analysis_table,
    calculate_contact_cost,
    calculate_expected_gross_value,
    calculate_expected_net_benefit,
    calculate_expected_retained_customers,
    calculate_roi,
    calculate_selection_metrics,
    determine_contact_capacity,
    exact_random_selection_expectations,
    rank_customer_risks,
    run_sensitivity_scenarios,
    select_highest_risk_customers,
    simulate_random_selection,
)


def test_correct_churn_probability_ranking() -> None:
    ranked = rank_customer_risks([0.2, 0.9, 0.5])
    assert ranked["heldout_order"].tolist() == [1, 2, 0]
    assert ranked["risk_rank"].tolist() == [1, 2, 3]


def test_risk_order_is_descending() -> None:
    ranked = rank_customer_risks([0.1, 0.7, 0.3, 0.8])
    assert ranked["predicted_churn_probability"].is_monotonic_decreasing


def test_probability_ties_use_original_order_deterministically() -> None:
    first = rank_customer_risks([0.8, 0.8, 0.4, 0.8])
    second = rank_customer_risks([0.8, 0.8, 0.4, 0.8])
    assert first["heldout_order"].tolist() == [0, 1, 3, 2]
    assert first.equals(second)


def test_top_twenty_percent_capacity_uses_floor() -> None:
    assert determine_contact_capacity(11, 0.20) == 2
    assert determine_contact_capacity(240, 0.20) == 48


def test_selected_count_never_exceeds_capacity() -> None:
    selected = select_highest_risk_customers(rank_customer_risks(np.linspace(0, 1, 17)))
    assert int(selected["selected_top_20pct"].sum()) == 3
    assert selected["selected_top_20pct"].mean() <= 0.20


def test_small_population_can_have_zero_contacts() -> None:
    selected = select_highest_risk_customers(rank_customer_risks([0.9, 0.8, 0.7, 0.6]))
    assert determine_contact_capacity(4, 0.20) == 0
    assert not selected["selected_top_20pct"].any()


def test_precision_at_twenty_percent_calculation() -> None:
    metrics = calculate_selection_metrics([1, 0, 1, 1, 0], [True, True, False, False, False])
    assert metrics["precision_at_20"] == 0.5


def test_recall_at_twenty_percent_calculation() -> None:
    metrics = calculate_selection_metrics([1, 0, 1, 1, 0], [True, True, False, False, False])
    assert np.isclose(metrics["recall_at_20"], 1 / 3)


def test_lift_calculation() -> None:
    metrics = calculate_selection_metrics([1, 0, 1, 1, 0], [True, True, False, False, False])
    assert np.isclose(metrics["lift_at_20"], 0.5 / 0.6)


def test_exact_random_selection_expectation() -> None:
    expected = exact_random_selection_expectations(100, 20, 30)
    assert expected["expected_churners_captured"] == 6
    assert expected["expected_random_precision"] == 0.30
    assert expected["expected_random_recall"] == 0.20


def test_random_simulation_is_deterministic_and_without_replacement() -> None:
    outcomes = [1, 1, 1, 0, 0, 0]
    first = simulate_random_selection(outcomes, 3, n_simulations=50, random_state=42)
    second = simulate_random_selection(outcomes, 3, n_simulations=50, random_state=42)
    assert first.equals(second)
    assert first["churners_captured"].between(0, 3).all()


def test_contact_cost_calculation() -> None:
    assert calculate_contact_cost(48, 5.0) == 240.0


def test_expected_retention_value_calculation() -> None:
    retained = calculate_expected_retained_customers(48, 0.10)
    assert np.isclose(retained, 4.8)
    assert np.isclose(calculate_expected_gross_value(retained, 150.0), 720.0)


def test_expected_net_benefit_calculation() -> None:
    assert calculate_expected_net_benefit(720.0, 240.0) == 480.0


def test_roi_calculation() -> None:
    assert calculate_roi(480.0, 240.0) == 200.0


def test_zero_denominators_are_handled_safely() -> None:
    metrics = calculate_selection_metrics([0, 0, 0], [False, False, False])
    random = exact_random_selection_expectations(0, 0, 0)
    assert metrics["precision_at_20"] == 0
    assert metrics["recall_at_20"] == 0
    assert metrics["lift_at_20"] == 0
    assert random["expected_churners_captured"] == 0
    assert calculate_roi(0, 0) == 0


def test_sensitivity_scenario_results() -> None:
    result = run_sensitivity_scenarios(
        4,
        2,
        2,
        success_rates=[0.10],
        contact_costs=[5.0],
        retained_values=[100.0],
    ).iloc[0]
    assert np.isclose(result["model_expected_retained_customers"], 0.4)
    assert np.isclose(result["random_expected_retained_customers"], 0.2)
    assert np.isclose(result["model_expected_net_benefit"], 30.0)
    assert np.isclose(result["random_expected_net_benefit"], 10.0)
    assert np.isclose(result["incremental_expected_net_benefit"], 20.0)


def test_target_labels_do_not_affect_selection() -> None:
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.5]
    first = build_internal_analysis_table(probabilities, [1, 0, 1, 0, 1], [1, 1, 1, 0, 0])
    second = build_internal_analysis_table(probabilities, [0, 1, 0, 1, 0], [1, 1, 1, 0, 0])
    columns = ["internal_row_reference", "risk_rank", "selected_top_20pct"]
    assert first[columns].equals(second[columns])


def test_customer_references_are_not_predictive_features() -> None:
    scores = [0.8, 0.8, 0.2]
    first = rank_customer_risks(scores, internal_references=["A", "B", "C"])
    second = rank_customer_risks(scores, internal_references=["Z", "Y", "X"])
    assert first["heldout_order"].tolist() == second["heldout_order"].tolist()
    assert "customer_id" not in first.columns


def test_model_probabilities_remain_unchanged_after_simulation() -> None:
    probabilities = np.array([0.91, 0.72, 0.44, 0.11, 0.83])
    before = probabilities.copy()
    _ = build_internal_analysis_table(
        probabilities,
        [1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(probabilities, before)

