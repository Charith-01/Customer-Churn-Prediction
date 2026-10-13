# Exploratory Data Analysis Report

## 1. Executive Summary

This Phase 3 analysis uses only the 960-row training portion of the Phase 2 cleaned dataset. The saved 240-row final test partition remained sealed. In the training data, 829 customers (86.35%) are labelled as churned and 131 (13.65%) as retained. Because the majority class is churn, a trivial all-churn prediction would be 86.35% accurate; accuracy alone would therefore be misleading in later modeling.

The strongest descriptive separation is in `monthly_usage_gb`: median usage is 70.75 GB for churned customers and 28.00 GB for retained customers, and churn rises from 57.45% in the lowest usage quintile to 99.47% in the highest. Month-to-month customers also have a higher observed churn rate (91.10%) than one-year (80.00%) and two-year customers (79.08%). Repeated late payments, three or more complaints, lower satisfaction, and higher charge bands show additional associations, although some extreme groups are small. Tenure and payment method do not show strong monotonic separation.

These are observational associations, not evidence that any characteristic causes churn. Usage timing and several unusual service combinations require business clarification before they are used for prediction.

## 2. Dataset Overview

- Source: `data/processed/02_cleaned_dataset.csv`
- Split artifact: `data/processed/02_split_indices.json`
- Saved split: 960 training rows (80%) and 240 final-test rows (20%), stratified by `churn`, random state 42
- Split fingerprint: `df788deba598b73abdc3a41b8a1947af63a90c5f369c2a498386a2c0eefeb769`
- EDA scope: the 960 training rows only; test outcomes and test distributions were not inspected
- Training data shape: 960 rows and 20 columns, including the identifier and target
- Identifier handling: `customer_id` was not analyzed or displayed at customer level

The training portion contains 109 missing values across 103 rows. They occur in `monthly_charges` (28), `monthly_usage_gb` (23), `total_charges` (20), and `satisfaction_score` (38). EDA summaries use available values and report the effective group sizes. Missing values were not imputed or otherwise changed in this phase.

## 3. Churn Distribution

The training target contains 829 churned customers (86.35%) and 131 retained customers (13.65%). This is a substantial imbalance toward the positive class. Later model evaluation should therefore include class-specific recall and precision, PR-AUC, ROC-AUC, calibration, and a metric aligned with selecting the highest-risk 20%, rather than accuracy alone.

[Figure 1 — Training churn distribution](figures/01_churn_distribution.png)

## 4. Contract Analysis

| Contract type | Customers | Churned | Churn rate |
|---|---:|---:|---:|
| Month-to-month | 562 | 512 | 91.10% |
| One year | 245 | 196 | 80.00% |
| Two year | 153 | 121 | 79.08% |

Month-to-month customers have an observed churn rate about 11.1 percentage points above one-year customers and 12.0 points above two-year customers. The similar one- and two-year rates suggest that the clearest descriptive distinction is flexible versus fixed-term contracts, rather than a smooth improvement with every additional contract year. Contract choice may also reflect unmeasured customer preferences, so the result is not causal.

[Figure 2 — Churn rate by contract type](figures/02_contract_churn.png)

## 5. Tenure Analysis

Tenure alone provides limited separation. Churned customers have a mean tenure of 35.73 months and a median of 34 months; retained customers have a mean of 36.69 months and a median of 38 months. Across fixed 12-month bands, churn ranges from 83.55% to 90.48%:

| Tenure band | Customers | Churn rate |
|---|---:|---:|
| 1–12 months | 175 | 87.43% |
| 13–24 months | 172 | 85.47% |
| 25–36 months | 147 | 90.48% |
| 37–48 months | 152 | 83.55% |
| 49–60 months | 149 | 85.91% |
| 61–72 months | 165 | 85.45% |

The peak in months 25–36 and the lower rate in months 37–48 are non-monotonic. Each band has 147–175 customers, so the pattern is not driven by a single tiny group, but it should still be validated in future cross-validation rather than converted directly into fixed rules.

[Figure 3 — Tenure distribution and churn by tenure band](figures/03_tenure_churn.png)

## 6. Monthly Charges Analysis

Churned customers have somewhat higher monthly charges: mean 71.41 and median 72.27, compared with mean 67.76 and median 65.79 for retained customers. Training-only charge quintiles show rates of 81.82%, 86.02%, 84.41%, 90.32%, and 88.77% from lowest to highest. The two highest bands are elevated, but the middle reversal means the relationship is not simply linear.

There are eight training values outside the IQR fences for monthly charges; the maximum is 235.01. These values were retained because an IQR flag is not proof of an error.

[Figure 4 — Monthly charges by outcome and charge quintile](figures/04_monthly_charges_churn.png)

## 7. Satisfaction Analysis

Satisfaction is lower among churned customers: median 6.7 versus 7.3 for retained customers. Churn generally falls as satisfaction increases: 92.06% for scores above 2 through 4, 88.79% for above 4 through 6, 86.17% for above 6 through 8, and 81.22% for above 8 through 10. The 0–2 band has only two customers and a 100% rate, so it is too small for a stable conclusion. There are also 38 missing satisfaction values.

[Figure 5 — Satisfaction distribution and churn bands](figures/05_satisfaction_churn.png)

## 8. Complaints Analysis

Churned customers average 0.73 complaints in the last 12 months, compared with 0.57 for retained customers. At the individual count level, rates fluctuate: 84.07% for zero complaints, 88.99% for one, 84.82% for two, 100% for three, 87.50% for four, and 100% for five. The groups at four and five complaints contain only eight and two customers.

Aggregating sparse counts gives more stable context: customers with three or more complaints have a 97.14% churn rate (34 of 35), compared with 85.95% among customers with fewer than three. This threshold is a candidate for later validation, not an established rule.

[Figure 6 — Churn rate by complaint count](figures/06_complaints_churn.png)

## 9. Late Payments Analysis

Churned customers average 0.84 late payments in the last 12 months, compared with 0.69 for retained customers. Churn rates are 84.77%, 85.64%, 91.72%, 89.66%, and 90.00% for zero through four late payments. When grouped, customers with two or more late payments have a 91.30% rate (168 of 184), versus 85.18% for customers with fewer than two. The increase appears threshold-like rather than linear, and the group with four late payments has only ten customers.

[Figure 7 — Churn rate by late-payment count](figures/07_late_payments_churn.png)

## 10. Usage Analysis

Usage is the clearest measured association with churn. Median `monthly_usage_gb` is 70.75 GB among churned customers and 28.00 GB among retained customers. Across training-derived usage quintiles, churn increases from 57.45% in the lowest band to 82.45%, 93.62%, 99.46%, and 99.47% in the highest band. Each band contains roughly 186–188 non-missing observations.

The increase is nonlinear: the largest change occurs between the first and second quintiles, followed by saturation near 100% in the upper two bands. Before using this field, the business must confirm that usage is measured before the prediction cutoff. If it includes behavior after a cancellation decision or during the outcome window, it could leak the target.

[Figure 8 — Monthly usage by outcome and usage quintile](figures/08_usage_churn.png)

## 11. Payment Method Analysis

| Payment method | Customers | Churn rate |
|---|---:|---:|
| Bank transfer | 243 | 88.48% |
| Electronic check | 318 | 86.79% |
| Credit card | 270 | 84.81% |
| Mailed check | 129 | 84.50% |

The total spread is only 3.98 percentage points. Bank transfer is highest and mailed check lowest in this training sample, but the differences are modest relative to usage or contract type and may not generalize.

[Figure 9 — Churn rate by payment method](figures/09_payment_method_churn.png)

## 12. Nonlinear Relationships

The grouped analyses identify several relationships that averages or simple correlations would miss:

1. **Usage:** churn rises sharply from 57.45% in the lowest quintile to 82.45% in the second, then saturates at approximately 99.5% in the upper two quintiles. The groups are well populated, but timing must be verified.
2. **Tenure:** churn peaks at 90.48% in months 25–36 rather than steadily decreasing with tenure. All six bands have at least 147 customers.
3. **Monthly charges:** churn rises overall from 81.82% in the lowest quintile to 88.77% in the highest but dips in the middle and peaks in the fourth quintile at 90.32%.
4. **Repeated events:** two or more late payments and three or more complaints show threshold-like changes. The late-payment group is reasonably sized (184), while the 3+ complaint group is small (35) and less stable.

The numerical correlation heatmap supports using grouped views: Pearson correlations with `churn` are 0.364 for usage, -0.098 for satisfaction, and only about ±0.06 or less for the other numerical fields. A small linear correlation does not exclude a useful nonlinear pattern.

[Figure 10 — Numerical-feature correlation heatmap](figures/10_numerical_correlation_heatmap.png)

## 13. Feature Interactions

### Contract type × tenure

Month-to-month churn remains high in all broad tenure groups: 90.31% for months 1–24, 93.60% for months 25–48, and 89.69% for months 49–72. One-year rates are 80.22%, 78.48%, and 81.33%; two-year rates are 83.33%, 77.08%, and 75.56%. The contract difference therefore persists across tenure, while the longest-tenure two-year group has the lowest rate. Cell sizes range from 45 to 196, so the two-year estimates warrant more uncertainty than the month-to-month estimates.

[Figure 11 — Contract and tenure interaction](figures/11_contract_tenure_interaction.png)

### Satisfaction × complaints

The combination reveals non-additive patterns. Among customers with no complaints, churn falls from 87.92% at satisfaction ≤6 to 76.07% above 8. With one complaint, the corresponding rates are 95.60%, 84.21%, and 88.46%. With two or more complaints, rates are 83.33%, 93.55%, and 82.35%. Counts range from 34 to 191; the smallest combinations are uncertain, and the reversals argue against a simplistic monotonic score.

[Figure 12 — Satisfaction and complaints interaction](figures/12_satisfaction_complaints_interaction.png)

## 14. Unusual Observations

IQR screening flags 97 training rows with at least one numerical outlier (107 field-level flags). Counts include 39 for late payments, 35 for complaints, 11 for total charges, eight for monthly charges, nine for satisfaction, four for usage, and one for discount rate. These were retained.

Several combinations need business clarification:

- Of 123 customers recorded as having no internet service, 122 have positive recorded monthly usage; 48 also have `tech_support = Yes`.
- Four rows have a relative discrepancy above 50% when total charges are compared with a simple monthly-charges × tenure expectation. Promotions, partial periods, charge changes, or field-definition differences could explain this.
- Extreme training values include monthly charges of 235.01, total charges of 8,853.76, and monthly usage of 451.3 GB.

These observations may be legitimate behavior or definition mismatches rather than errors. No row was removed or corrected in EDA.

## 15. Main Churn-Risk Patterns

- High monthly usage has the strongest descriptive relationship with churn, with upper-quintile rates near 99.5% versus 57.45% in the lowest quintile.
- Month-to-month contracts have a 91.10% churn rate, about 11–12 percentage points above fixed-term contracts.
- Lower satisfaction is associated with higher churn, subject to 38 missing values and one very small low-score group.
- Repeated late payments (2+) and repeated complaints (3+) coincide with higher churn, although the complaint threshold group is small.
- Higher charge bands are somewhat elevated but not monotonic.
- Tenure and payment method have relatively weak standalone separation.

## 16. Potential Feature Engineering Opportunities

These are proposals for Phase 4 only; none were implemented during EDA.

| Proposed name | Source columns | Business rationale and expected usefulness | Leakage concern | Available at prediction time? |
|---|---|---|---|---|
| `tenure_stage` | `tenure_months` | Capture the observed non-monotonic lifecycle pattern with business-defined or fold-derived bands. | If learned from outcomes, cut points must be selected inside training folds. | Yes, if current tenure is known at scoring. |
| `contract_tenure_interaction` | `contract_type`, `tenure_months` | Represent different tenure patterns within flexible and fixed-term contracts. | Low, provided both fields are pre-cutoff; do not target-encode on all data. | Yes. |
| `usage_band` or `high_usage_flag` | `monthly_usage_gb` | Capture the sharp nonlinear rise and upper-band saturation. | High priority to verify the usage measurement window; fold-derived cut points are required if data-driven. | Unresolved until timing is confirmed. |
| `satisfaction_complaint_interaction` | `satisfaction_score`, `complaints_last_12m` | Combine service sentiment and recent friction, reflecting the interaction heatmap. | Both must be recorded before the outcome window; missingness must be handled within the pipeline. | Likely, subject to timestamp confirmation. |
| `repeated_payment_difficulty` | `late_payments_last_12m` | Encode the observed threshold at two or more late payments. | The 12-month window must end at or before scoring; validate the threshold within folds. | Yes, if historical events are frozen at scoring. |
| `charge_tenure_consistency` | `monthly_charges`, `total_charges`, `tenure_months` | Flag accounts whose accumulated charges differ from a simple expected total, potentially reflecting plan changes or promotions. | Field definitions and billing timing are unresolved; avoid using it until clarified. | Unresolved. |

## 17. Business Implications

The descriptive results suggest that future retention analysis should prioritize validation of usage and contract signals, while treating tenure or payment method alone as weak segmentation tools. Fixed-term contracts are associated with lower churn, but an intervention cannot be recommended from this observational analysis because contract selection may reflect customer self-selection. Service-friction measures—satisfaction, complaints, and late payments—may help identify operationally interpretable risk groups once their timestamps are confirmed.

The business constraint of contacting only the highest-risk 20% should guide later evaluation. Phase 5 should assess ranking quality and recall/precision within that top fifth, after Phase 4 features are fit without leakage.

## 18. Limitations and Next Steps

- Results are descriptive and do not establish causality.
- The positive churn prevalence is unusually high; target semantics and observation-window construction should be reconfirmed with the data owner.
- Several fields lack explicit event timestamps, especially usage, satisfaction, complaints, and payment history. Prediction-time availability must be documented before modeling.
- Missing values were retained, and grouped results use the available observations.
- Sparse extreme categories produce unstable rates; counts are shown to prevent overinterpretation.
- No statistical model, feature engineering, hyperparameter tuning, final-test evaluation, or retention simulation was performed.
- Phase 4 should implement candidate features inside leakage-safe pipelines, using training folds only. The 240-row final test set must remain sealed until final model evaluation.
