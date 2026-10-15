# Customer Retention Intervention Simulation and Business Impact Analysis

## 1. Executive Summary

Phase 8 applies a strict 20% campaign capacity to the unchanged Phase 6 churn probabilities. Of 240 held-out customers, the predeclared policy contacts the 48 highest-risk customers. All 48 historically churned, producing precision@20% of **1.0000**, recall/capture@20% of **0.2319**, and lift@20% of **1.1594** against the sample's 86.25% churn prevalence.

Uniform random selection with the same 48-contact budget would capture **41.4 churners in expectation**. Model ranking therefore concentrates **6.6 additional historical churners** within the same budget. Under illustrative assumptions of $5 per contact, $150 value per successfully retained customer, and 10% intervention effectiveness, expected net benefit is **$480** for model ranking and **$381** for random selection, an incremental **$99**. These are retrospective scenario estimates, not realized financial results or evidence that outreach prevents churn.

## 2. Business Problem

The retention team cannot contact every customer. The practical decision is whom to prioritize when no more than 20% of the population can be contacted. This analysis measures whether locked-model ranking concentrates observed churners better than random selection and explores the hypothetical economics of acting on that ranking.

Prediction and intervention are distinct. A high churn probability does not reveal whether a customer is persuadable or which treatment would work.

## 3. Final Model Used

The simulation reuses the trusted `models/03_model_artifact.joblib` artifact: a sigmoid-calibrated ensemble of five complete tuned Random Forest pipelines. The positive class is `churn = 1`. No model fitting, feature change, recalibration, hyperparameter change, threshold adjustment, or new split occurred.

The verified model SHA-256 remains `b3635a3e65e850bca895a8c7506de6bcd3add2953a3829759544eddede0ed496`. Its 240 test probabilities reproduce the Phase 7 probability hash `4e95bae1a6bff60cfdb7a8b2161716e31555cdbcc7ca73d003109a926219e1ae` and the Phase 6 metrics.

## 4. Contact Capacity Assumption

The campaign capacity is **20%**. Contact count is calculated as `floor(240 × 0.20) = 48`, so the limit cannot be exceeded. The resulting contact rate is exactly **20.00%**.

This is a separate top-k decision policy. The locked Phase 6 threshold remains **0.9707090004321576** and is not moved to make its classifications fit the capacity.

## 5. Customer Risk Ranking Method

Customers are sorted by predicted churn probability descending. The existing held-out row order is the predeclared secondary key for exact ties; it does not use customer IDs, actual churn outcomes, or any other target-derived information. Outcomes are attached only after selection for retrospective measurement.

The 240 probabilities range from **0.245218** to **0.980746**, with mean **0.871795** and median **0.962813**. All 240 scores are unique, so no observed tie required the secondary rule. Probabilities express fitted risk estimates, not guaranteed individual outcomes.

The protected row-level analysis table is stored only at the Git-ignored `data/processed/internal_retention_ranking.csv` path. It uses synthetic internal references and contains no raw customer ID. No customer-level contact list is published.

## 6. Top-20% Customer Selection

| Measure | Result |
|---|---:|
| Evaluated customers | 240 |
| Maximum/selected contacts | 48 |
| Actual selected percentage | 20.00% |
| Historical churners selected | 48 |
| Historical non-churners selected | 0 |
| Historical churners outside selection | 159 |
| Total historical churners | 207 |

Selection uses scores only. The historical label does not influence who enters the top 48.

## 7. Churner Capture Results

The top-risk group contains **48 of 207** observed churners. The other **159** observed churners remain outside the campaign because the capacity covers only one-fifth of customers. “Captured” means present in the retrospective contact group; it does not mean successfully retained.

## 8. Precision@20% and Recall@20%

- **Precision@20% = 48 / 48 = 1.0000.** Every selected customer historically churned.
- **Recall@20% / capture rate@20% = 48 / 207 = 0.2319.** The campaign includes 23.19% of all historical churners.
- **Contact rate = 48 / 240 = 0.2000.**

These are ranking-policy metrics. They must not be confused with Phase 6 precision and recall at the fixed classification threshold, which selected 99 customers rather than respecting a strict 48-contact capacity.

## 9. Ranking Lift

Overall held-out churn prevalence is **207 / 240 = 0.8625**. Lift@20% is therefore `1.0000 / 0.8625 = 1.1594`. The selected group is about **15.94% more concentrated** with historical churners than uniform random selection.

The lift is useful but bounded by the already high prevalence. A perfect selected-group precision of 1.0 cannot exceed a lift of `1 / 0.8625 = 1.1594` in this sample.

## 10. Random Selection Comparison

With `N = 240`, `C = 207`, and `k = 48`, exact uniform-random expectations are:

- churners captured: `48 × 207 / 240 = 41.4`;
- precision@20%: **0.8625**;
- recall@20%: **0.2000**;
- lift: **1.0000**.

| Strategy | Contacts | Churners captured | Precision@20% | Recall@20% | Lift |
|---|---:|---:|---:|---:|---:|
| Model-ranked top 20% | 48 | 48.0 observed | 1.0000 | 0.2319 | 1.1594 |
| Uniform random selection | 48 | 41.4 expected | 0.8625 | 0.2000 | 1.0000 |

The measured model advantage is **6.6 additional historical churners** and **3.19 percentage points of churner capture** at equal contact volume. A 10,000-run random simulation using seed 42 produced mean capture **41.4133**, standard deviation **2.1305**, and a central 95% empirical range of **37–45**, consistent with the exact expectation.

## 11. Hypothetical Cost Assumptions

The dataset supplies no campaign cost, retained-customer value, or treatment-response data. Phase 8 therefore uses the required illustrative assumptions:

- contact cost: **$5 per customer**;
- value of successfully retaining one would-be churner: **$150**;
- intervention success rate: **10%** of contacted customers who otherwise would churn;
- currency: **hypothetical USD accounting units**;
- contact capacity: **20%**.

The calculation assumes the historical churn label approximates the no-intervention outcome. That assumption is not directly verifiable from this dataset. The success rate is a scenario input, not a measured causal effect.

## 12. Model-Based Financial Simulation

The model-ranked strategy contacts 48 customers at a total hypothetical cost of **$240**. Applying the 10% scenario rate to 48 observed churners gives **4.8 expected retained customers**, a fractional expectation rather than confirmed individual outcomes. Gross expected retained-customer value is **$720**, expected net benefit is **$480**, and scenario ROI is **200.00%**.

## 13. Random-Selection Financial Simulation

Random selection uses the same 48 contacts and **$240** contact cost. Applying the 10% scenario rate to the exact expected capture of 41.4 gives **4.14 expected retained customers**, **$621** gross value, **$381** net benefit, and scenario ROI of **158.75%**.

## 14. Incremental Expected Benefit

The base scenario's incremental expected net benefit is:

`$480 model net benefit - $381 random net benefit = $99`

The difference arises entirely from the model ranking's 6.6 additional historical churners under equal contact cost and equal hypothetical treatment assumptions. No actual financial improvement has occurred.

## 15. Sensitivity Analysis

The fixed ranking was evaluated across 27 combinations of success rate (5%, 10%, 20%), contact cost ($2, $5, $10), and retained-customer value ($75, $150, $300). It was not re-ranked using profitability or test outcomes.

At the base $5 contact cost and $150 retained value:

| Assumed success rate | Model net benefit | Random net benefit | Incremental net benefit | Model ROI | Random ROI |
|---:|---:|---:|---:|---:|---:|
| 5% | $120.00 | $70.50 | $49.50 | 50.00% | 29.38% |
| 10% | $480.00 | $381.00 | $99.00 | 200.00% | 158.75% |
| 20% | $1,200.00 | $1,002.00 | $198.00 | 500.00% | 417.50% |

The model strategy is financially positive in **23 of 27** illustrative scenarios. It is negative in low-value/high-cost combinations, demonstrating that positive economics are not automatic. Under base cost and value, break-even effectiveness is **3.33%** for model ranking and **3.86%** for random selection. These thresholds are scenario arithmetic, not estimates of achievable response.

## 16. Proposed Retention Interventions

| Evidence | Candidate intervention | Intended group | Proposed mechanism | Risks and success metrics |
|---|---|---|---|---|
| Usage is the dominant model signal | Service-quality and plan-fit review | Top-ranked eligible high-usage customers | Identify reliability, congestion, or plan mismatch | Technical cost and unnecessary contact; measure randomized retention uplift and issue resolution |
| Satisfaction contributes to ranking; lower satisfaction is associated with risk | Proactive support and complaint resolution | Top-ranked eligible customers with current service issues | Resolve friction and restore confidence | Agent time and poor-contact risk; measure resolution time, satisfaction change, and uplift |
| Recent late payments add model information | Optional payment-support outreach | Top-ranked eligible customers with recent payment difficulty | Reduce involuntary loss through approved reminders or arrangements | Privacy, compliance, and fairness risks; measure payment completion, complaints, and uplift |
| Contract type is the second-ranked global input | Contract and renewal assistance | Top-ranked eligible month-to-month customers | Clarify suitable voluntary plan options | Discount leakage and unequal treatment; measure margin-adjusted uplift, acceptance, and opt-outs |

These interventions are hypotheses. A randomized pilot with an eligible no-contact control group is required before broader rollout. Sensitive or protected attributes should not drive unsupported adverse treatment.

## 17. Operational Workflow

1. Collect current consented customer data and validate the model schema.
2. Generate churn probabilities using the unchanged trusted artifact.
3. Rank scores descending with the predeclared stable tie-breaker.
4. Select `floor(N × 0.20)` customers.
5. Apply documented eligibility, privacy, fairness, and business constraints.
6. Assign approved interventions independently of protected characteristics.
7. Record treatment assignment, contact, cost, response, churn, and opt-out outcomes.
8. Use a randomized holdout to estimate incremental retention uplift.
9. Monitor input quality, population drift, calibration, ranking capture, capacity, costs, and subgroup outcomes.

This is a conceptual workflow only. Phase 8 builds no API, application, database, or production deployment.

## 18. Business Risks and Limitations

- The analysis is retrospective on one 240-row held-out sample with unusually high 86.25% churn prevalence.
- Historical churn is assumed to approximate the no-treatment outcome, but no campaign treatment or response data exist.
- Targeting likely churners is not the same as identifying customers whose behavior can be changed.
- Cost, value, and success-rate inputs are hypothetical and omit offer cost, channel cost, margin, cannibalization, and long-term effects.
- Predictor timing and production availability require confirmation; leakage could occur operationally if fields are not available at scoring time.
- Calibration and ranking can drift, so current-period monitoring and temporal validation are required.
- Fairness, consent, privacy, security, eligibility, and opt-out requirements need formal review.
- The internal ranking contains retrospective outcomes for analysis and must never become a production-facing contact list.

## 19. Recommendations

Use deterministic top-20% ranking—not the Phase 6 fixed threshold—when the campaign truly has a hard capacity. Treat the present economics as a decision aid for designing a pilot, not a business case based on realized uplift. Start with a small randomized campaign, define treatment eligibility and success metrics before launch, capture complete costs and outcomes, and evaluate incremental churn reduction and margin rather than contacted-customer churn alone.

## 20. Phase 9 Preparation

Phase 9 may assemble the final business report using the audited results from Phases 1–8. It should preserve the distinction between observed data, measured predictive performance, hypothetical intervention assumptions, scenario estimates, and proposed future actions. Before any final PDF is generated, confirm stakeholder audience, branding, required executive summary length, and whether financial scenarios should remain in USD illustrative units or be replaced with verified business inputs.

## Reproducible Outputs

- Executed notebook: `notebooks/08_retention_simulation.ipynb`
- Reusable functions: `src/retention_simulation.py`
- Aggregate results: `reports/retention_simulation_results.csv`
- Sensitivity grid: `reports/retention_sensitivity_analysis.csv`
- Integrity audit: `reports/retention_simulation_integrity.json`
- Figures: `reports/figures/phase8_*.png`
- Automated tests: `tests/test_retention_simulation.py`

