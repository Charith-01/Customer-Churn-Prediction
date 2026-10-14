# Model Explainability and Customer-Level Churn Risk Analysis

## 1. Executive summary

Phase 7 explains the locked Phase 6 calibrated Random Forest without retraining, tuning, recalibrating, or changing its operating threshold. On the fixed 240-row test partition, original-feature permutation importance shows that `monthly_usage_gb` is by far the model's most important input for ranking churners. Contract type, satisfaction, and late-payment history provide smaller additional signals. Three anonymized customer cases illustrate how individual predictions respond when one input at a time is replaced by a training-derived reference value. The locked threshold continues to favor very high precision (98 true positives and one false positive) at the cost of 109 false negatives.

These explanations describe model behavior and observed associations. They do not prove why a customer churned and must not be interpreted as causal intervention effects.

## 2. Objectives and scope

This phase was designed to:

- explain global behavior using original, business-readable input fields;
- inspect observed prediction patterns for influential attributes and an interaction;
- explain at least three anonymized test customers, including correct and incorrect predictions;
- characterize errors at the unchanged Phase 6 threshold;
- preserve the final model, calibration, threshold, split, data, and predictions exactly;
- prepare evidence for Phase 8 retention simulation without performing that simulation.

No training, model selection, hyperparameter search, calibration choice, threshold optimization, intervention simulation, or cost-benefit analysis was performed.

## 3. Final model used

The explained artifact is the complete trusted `CalibratedClassifierCV` saved at `models/03_model_artifact.joblib`. Its five calibrated members each contain the Phase 4 feature engineer, training-fitted preprocessing, and the Phase 5 tuned balanced Random Forest. Sigmoid calibration and the operating threshold of **0.9707090004321576** remain locked.

The artifact SHA-256 was verified before deserialization and again after analysis: `b3635a3e65e850bca895a8c7506de6bcd3add2953a3829759544eddede0ed496`. The before/after prediction hash also matched. Only the trusted local joblib artifact was loaded.

## 4. Global explanation method

Global importance uses scikit-learn permutation importance on the complete calibrated pipeline and the unchanged held-out test set. Each of the 18 original predictors is shuffled independently for **10 deterministic repeats** with random seed 42. Importance is the mean decrease in held-out Average Precision (AP); standard deviation across repeats quantifies instability.

This design preserves direct mappings to original fields and captures the fitted feature-engineering and preprocessing path. Correlated or redundant inputs can share or mask importance, and negative values are retained rather than clipped. The test set is used only for post-lock diagnostic explanation, not model or policy selection.

## 5. Global importance results

| Rank | Original feature | Mean AP decrease | Repeat SD |
|---:|---|---:|---:|
| 1 | `monthly_usage_gb` | 0.107152 | 0.016357 |
| 2 | `contract_type` | 0.004396 | 0.002636 |
| 3 | `satisfaction_score` | 0.003581 | 0.001007 |
| 4 | `late_payments_last_12m` | 0.001687 | 0.000923 |
| 5 | `region` | 0.000634 | 0.000664 |
| 6 | `monthly_charges` | 0.000242 | 0.001121 |
| 7 | `discount_pct` | 0.000172 | 0.000197 |
| 8 | `internet_service` | 0.000169 | 0.000218 |
| 9 | `dependents` | 0.000166 | 0.000161 |
| 10 | `paperless_billing` | 0.000111 | 0.000250 |

`monthly_usage_gb` has a much larger and more stable importance than every other raw field. Features below the first four have effects close to zero relative to repeat variation, so their precise ranking should not be over-interpreted. The small negative values for `tech_support`, `support_plan`, `total_charges`, `age`, and `tenure_months` indicate that shuffling did not reduce AP in this finite sample; they are not evidence that those fields are inherently harmful.

## 6. Feature effects and interaction patterns

Observed test-set grouping provides direction context but is not a partial-dependence or causal analysis:

- Mean predicted churn probability rises from **0.6300** in the lowest monthly-usage quintile to **0.9757** in the fourth quintile and remains high at **0.9740** in the highest. The relationship is strong, nonlinear, and saturates at high usage.
- Mean predicted risk generally falls as satisfaction rises: **0.9239** in the lowest satisfaction quintile versus **0.8446** in the highest, although the middle bins are not perfectly monotonic.
- Month-to-month customers have the highest observed mean prediction (**0.9106**), compared with one-year (**0.7898**) and two-year (**0.8465**) customers. The two-year estimate exceeding one-year cautions against assuming a simple causal duration effect.
- In the observed usage-by-satisfaction table, low-usage groups range from **0.6564** to **0.7719**, middle-usage groups from **0.9090** to **0.9688**, and high-usage groups from **0.9734** to **0.9755**. Satisfaction differentiates predictions more at low and middle usage; very high usage dominates the fitted score.

## 7. Individual explanation method and case selection

SHAP was not used because it was not installed in the reproducible project environment and the final estimator is a calibrated five-member ensemble rather than a single native tree model. The chosen model-agnostic method is one-feature-at-a-time probability sensitivity.

For each customer, one original field is replaced with a training-derived reference while all other fields remain fixed. Numerical references are observed training values nearest the median; categorical references are training modes. The reported effect is:

`original churn probability - probability after reference replacement`

A positive value means the customer's actual value raises fitted risk relative to that reference; a negative value means it lowers fitted risk. Effects are local sensitivity checks, are not additive, can depend on feature correlations and reference choice, and are not causal counterfactuals.

Cases were selected only from the permitted fixed test partition: a representative true positive, a representative true negative, and the false negative closest to the locked threshold. Dataset indices and customer identifiers are excluded from all exported case summaries.

## 8. Customer A — representative true positive

Customer A actually churned and was correctly classified as churn at probability **0.978279**, just above the **0.970709** threshold. The customer has monthly usage of **75.8 GB**, a month-to-month contract, satisfaction **7.2**, and one late payment.

The dominant local signal is usage: replacing 75.8 GB with the 65.0 GB training reference reduces predicted risk by **0.047154**. Monthly charges add only **0.001243** relative to their reference. Tenure, satisfaction, paperless billing, dependents, support, and internet service have small locally risk-lowering effects. This case is a confident positive whose score is primarily sensitive to usage.

## 9. Customer B — representative true negative

Customer B did not churn and was correctly classified as non-churn at probability **0.642320**, well below the locked threshold. A one-year contract lowers fitted risk by **0.269059** relative to the month-to-month training reference, and usage of **35.4 GB** lowers risk by **0.230734** relative to 65.0 GB. The observed age raises risk by **0.196094** relative to its training reference, while the missing monthly-charge value lowers the fitted score by **0.113714** under the pipeline's learned missing-value handling.

This example shows why explanations must report opposing signals: lower usage and the one-year contract dominate age and late-payment contributions. The missing-charge result describes the fitted pipeline response and must not be treated as a recommendation to omit data.

## 10. Customer C — near-threshold false negative

Customer C actually churned but was classified as non-churn at probability **0.970636**, only **0.000073** below the locked threshold. Usage of **76.3 GB** raises fitted risk by **0.154426** relative to 65.0 GB. A one-year contract lowers it by **0.007853**, and monthly charges lower it by **0.008825**, relative to their respective references. Other one-at-a-time effects are small.

This is a policy-boundary miss: a negligible score difference changes the class at a very high cutoff. It is evidence that binary labels near the threshold are fragile and that Phase 8 should use ranked probabilities when enforcing a strict contact capacity.

## 11. Prediction error analysis

At the unchanged threshold, outcomes remain **98 true positives, 32 true negatives, 1 false positive, and 109 false negatives**. Precision is **0.9899** and recall is **0.4734**.

False negatives have lower median usage (**55.95 GB**) than true positives (**90.40 GB**), while median satisfaction is similar (**6.6** versus **6.3**). Month-to-month contracts account for **67.0%** of false negatives and **62.2%** of true positives; fiber service accounts for **54.1%** and **53.1%**, respectively. These overlaps show that simple business categories do not cleanly separate captured and missed churners.

The single false positive has unusually high usage (**131.8 GB**) and therefore should not support broad conclusions about false-positive profiles. Error slices are descriptive and small groups require caution.

## 12. Business interpretation

Usage is the strongest prioritization signal in this fitted model, particularly once consumption exceeds roughly 53 GB per month. Contract type, satisfaction, and recent payment behavior provide useful context for outreach messaging, but they should not be treated as proven intervention levers.

For retention operations, use the score to prioritize review, then combine it with permissible operational context. High-usage churn-risk customers may warrant service-quality, capacity, or plan-fit investigation; month-to-month customers may be offered optional longer-term plans only after testing; dissatisfied customers may warrant service recovery. All actions should be evaluated through controlled experiments and fairness monitoring.

## 13. Association versus causation

Permutation importance answers how much held-out ranking performance changes when a field's information is disrupted. Local sensitivity answers how the fitted prediction changes under a one-field reference substitution. Observed group plots summarize association in this test sample. None identifies a causal treatment effect.

The model may learn proxies, measurement processes, and correlations. Changing a field in reality can alter other fields and customer behavior, violating the one-feature-at-a-time assumption. Retention offers must therefore be validated with randomized or credible quasi-experimental evaluation.

## 14. Limitations and safeguards

- The 240-row test set is small, highly churn-prevalent, and already used for final evaluation; explanations may vary in new cohorts.
- Correlated predictors can dilute permutation importance, while one-at-a-time perturbations can create unlikely combinations.
- Calibration compresses and aggregates behavior across five fitted pipelines, limiting direct tree-path interpretation.
- Local effects depend on the selected training reference and do not sum to the full prediction.
- Only three representative cases are discussed; they are not population segments.
- `region`, age, and other potentially sensitive or proxy attributes require fairness review before operational use.
- The fixed threshold selected 41.25% of test customers despite a 20% OOF capacity design; it must not be assumed to enforce campaign capacity.
- Raw identifiers are excluded from model inputs, explanation exports, and report examples.

## 15. Recommendations for Phase 8

Phase 8 should simulate a deterministic top-20% ranked outreach policy, compare it with the unchanged locked-threshold policy, and report captured churners, precision, recall, lift, expected value, and sensitivity to retention success and contact/offer costs. It should define tie handling before results are inspected, avoid using protected/proxy fields for adverse treatment, and retain customer-level data only in access-controlled operational outputs. Intervention impact should be framed as scenario assumptions until a controlled campaign establishes causal uplift.

## Reproducible outputs

- Executed notebook: `notebooks/07_model_explainability.ipynb`
- Reusable code: `src/explainability.py`
- Aggregate and anonymized tables: `reports/explainability/`
- Figures: `reports/figures/phase7_*.png`
- Tests: `tests/test_explainability.py`

