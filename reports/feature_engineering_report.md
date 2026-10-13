# Feature Engineering Report

## 1. Executive Summary

Phase 4 adds seven deterministic numerical features to the existing 18 predictors, producing 25 pre-encoding predictors while preserving all 960 training rows, their order, and their indices. A reusable scikit-learn transformer and an integrated feature-engineering/preprocessing pipeline are implemented in `src/feature_engineering.py`.

All feature decisions use only the saved Phase 2 training partition. No feature formula uses `churn`, no customer identifier enters the predictor matrix, and no classifier is trained. `charges_per_tenure` is deferred because the cumulative-charge definition and its relationship with current charges, discounts, and tenure remain unresolved.

## 2. EDA Findings Used

Phase 3 provided the following training-only motivation:

- monthly usage had the strongest observed association with churn, but its timing and service scope remain unresolved;
- month-to-month customers had 91.10% churn, compared with 80.00% for one-year and 79.08% for two-year contracts;
- contract-by-tenure and satisfaction-by-complaint views revealed interactions not captured by standalone averages;
- customers with two or more late payments had 91.30% churn, compared with 85.18% below that count;
- tenure, charge, complaint, and usage patterns were not uniformly linear.

These observations motivate candidate representations; they do not establish causality or performance improvement.

## 3. Feature Engineering Strategy

The selected transformations are simple, deterministic, row-level calculations. They use predictor values available in the input row and do not learn cut points, averages, encodings, or target statistics. The transformer:

- accepts and returns pandas DataFrames;
- copies the input rather than mutating it;
- validates source columns and numerical domains;
- rejects `customer_id` and `churn`;
- preserves original columns, row order, row count, and indices;
- propagates missingness for downstream fold-fitted imputation;
- converts invalid divisions to missing instead of infinity;
- exposes stable names through `get_feature_names_out`.

## 4. Original Features

The Phase 2 predictor matrix contains 18 columns:

- Numerical: `age`, `tenure_months`, `monthly_charges`, `total_charges`, `complaints_last_12m`, `late_payments_last_12m`, `satisfaction_score`, `monthly_usage_gb`, and `discount_pct`.
- Categorical/binary: `contract_type`, `internet_service`, `region`, `payment_method`, `paperless_billing`, `dependents`, `support_plan`, `tech_support`, and `senior_citizen`.

`customer_id` is an identifier and `churn` is the target. Neither is an original model predictor.

## 5. Engineered Features

Seven candidates are accepted for technical implementation:

| Feature name | Source columns | Calculation | Business rationale | Leakage risk |
|---|---|---|---|---|
| `tenure_years` | `tenure_months` | tenure ÷ 12 | Express relationship duration in an interpretable scale | Low; exact rescaling and fully redundant with tenure months |
| `complaints_per_active_month` | complaints, tenure | complaints ÷ min(max(tenure, 1), 12) | Approximate complaint intensity over observed service exposure | Medium; denominator is an approximation because only the numerator's 12-month window is documented |
| `payment_risk_score` | complaints, late payments | complaints + late payments | Combine two recent service/payment-friction counts | Medium; equal-weight heuristic, not a validated financial-risk score |
| `complaint_satisfaction_interaction` | complaints, satisfaction | complaints × (10 − satisfaction) | Represent complaints occurring alongside lower satisfaction | Medium; 0–10 bound is validated, but source timestamps are absent |
| `monthly_contract_tenure` | tenure, contract | tenure × I(contract is Month-to-month) | Represent the contract-by-tenure pattern found in EDA | Low to medium; prediction-time contract status must be frozen consistently |
| `late_payment_charge_interaction` | late payments, monthly charges | late payments × monthly charges | Represent repeated payment difficulty alongside a higher current charge | Medium; interaction score is not an overdue balance |
| `usage_per_charge` | usage, monthly charges | usage ÷ monthly charges | Represent usage intensity relative to current charge | High until timing is confirmed; not a monetary return-on-investment measure |

The validated transformer output contains 25 columns before one-hot encoding: the 18 originals plus these seven engineered features.

[Figure FE-1 — Engineered feature distributions](figures/fe_01_engineered_feature_distributions.png)

## 6. Business Rationale

The features address three analytical goals:

1. **Interpretability:** `tenure_years` presents an existing duration in familiar units.
2. **Exposure and intensity:** complaint intensity and usage per charge distinguish equal raw counts observed under different exposure or price conditions.
3. **Interactions:** contract-tenure, complaint-satisfaction, and late-payment-charge features allow a future model to represent combinations observed in Phase 3 without using target-derived encodings.

`payment_risk_score` is deliberately described as a heuristic friction count. Its name does not imply a credit score or validated financial-risk measure.

## 7. Transformation Formulas

- `tenure_years = tenure_months / 12`
- `complaints_per_active_month = complaints_last_12m / clip(tenure_months, 1, 12)`
- `payment_risk_score = complaints_last_12m + late_payments_last_12m`
- `complaint_satisfaction_interaction = complaints_last_12m × (10 − satisfaction_score)`
- `monthly_contract_tenure = tenure_months × I(contract_type == "Month-to-month")`
- `late_payment_charge_interaction = late_payments_last_12m × monthly_charges`
- `usage_per_charge = monthly_usage_gb / monthly_charges`, only when the denominator is positive

The 10-point satisfaction maximum comes from the Phase 2 validated 0–10 range. Although the observed training maximum is also 10, the formula does not estimate the scale maximum from the data.

## 8. Missing-Value Handling

Feature engineering does not impute values. If a required source is missing, the derived result remains missing so the downstream numerical imputer can learn a median from the current training fold.

Training-only missing counts are:

| Engineered feature | Missing rows |
|---|---:|
| `tenure_years` | 0 |
| `complaints_per_active_month` | 0 |
| `payment_risk_score` | 0 |
| `complaint_satisfaction_interaction` | 38 |
| `monthly_contract_tenure` | 0 |
| `late_payment_charge_interaction` | 28 |
| `usage_per_charge` | 50 |

Zero tenure uses the documented one-month floor in complaint intensity. A zero or missing monthly charge produces missing `usage_per_charge`. No engineered field contains positive or negative infinity.

[Figure FE-4 — Engineered feature missingness](figures/fe_04_engineered_feature_missingness.png)

## 9. Feature Redundancy Analysis

The strongest absolute training-only correlations involving engineered fields are:

- `tenure_years` with `tenure_months`: 1.0000;
- `late_payment_charge_interaction` with `late_payments_last_12m`: 0.9019;
- `complaint_satisfaction_interaction` with complaints: 0.8209;
- `usage_per_charge` with monthly usage: 0.7421;
- `payment_risk_score` with late payments: 0.7326;
- `payment_risk_score` with complaints: 0.7250.

These correlations are expected from the formulas. Interactions may still be useful to a model that cannot easily construct them, but Phase 5 should compare pipelines with and without redundant fields, especially for regularized linear models. No feature is retained on a claim of predictive improvement.

Training-only descriptive ranges include 0.0833–6.0000 years for tenure, 0–8 for the heuristic payment-risk score, 0–23.7 for complaint-satisfaction interaction, and 0.0166–7.7568 for non-missing usage per charge.

[Figure FE-2 — Source and engineered relationships](figures/fe_02_source_engineered_relationships.png)

[Figure FE-3 — Source and engineered feature correlations](figures/fe_03_feature_correlation_heatmap.png)

## 10. Leakage Risk Assessment

Implemented controls:

- The feature transformer rejects any dataframe containing `customer_id` or `churn`.
- Formulas contain no target values, outcome rates, target-derived bins, or target encoding.
- The transformer is stateless and learns no numerical parameters or thresholds.
- Feature engineering precedes the `ColumnTransformer` inside one scikit-learn `Pipeline`.
- Imputation, scaling, and one-hot encoding therefore fit only on the data supplied to `Pipeline.fit` in each future fold.
- Unknown categorical values are handled by the existing one-hot encoder with `handle_unknown="ignore"`.
- The final-test partition was not used to select formulas or calculate report statistics.

Technical target leakage is prevented by construction. Temporal leakage cannot be fully ruled out without event timestamps and field-window definitions.

## 11. Prediction-Time Data Availability

| Source family | Expected availability | Unresolved risk |
|---|---|---|
| Tenure and contract | Likely available | Must represent status at the scoring timestamp |
| Complaints and late payments | Likely available as history | Their “last 12 months” windows must end before scoring |
| Satisfaction | Possibly available | Survey timing relative to churn is undocumented |
| Monthly charges | Likely available | Must be the charge known at scoring, not a later revision |
| Monthly usage | Unresolved | Strong EDA association makes cutoff confirmation especially important |

Until the data owner confirms these timestamps, Phase 5 should treat usage-derived and behavioral-history results as provisional.

## 12. Train/Test Integrity

- Existing split loaded rather than recreated: yes
- Training rows used for analysis: 960
- Final-test rows kept sealed: 240
- Split strategy: stratified 80/20, random state 42
- Dataset fingerprint: `df788deba598b73abdc3a41b8a1947af63a90c5f369c2a498386a2c0eefeb769`
- Training/test index overlap: zero
- Cleaned dataset and split artifact hashes before/after notebook execution: unchanged

The notebook does not construct a held-out dataframe or inspect held-out labels.

## 13. Validation Results

The feature notebook executes from top to bottom with no error outputs and verifies four generated figures. Its integrated demonstration fits feature engineering and preprocessing on 800 training rows, transforms the remaining 160 training rows, produces a stable transformed width, and contains only finite values after imputation and scaling.

Automated tests cover all intended features, numerical formulas, row/index preservation, input immutability, missing values, zero denominators, infinity prevention, stable names, unknown categories, unseen data, identifier/target rejection, deterministic output, train-only imputation, and integrated pipeline transformation. The complete Phase 1–4 suite passes: **25 tests passed**.

## 14. Limitations

- `tenure_years` is exactly redundant with tenure months.
- Complaint exposure is approximate because tenure and the complaint lookback window are not fully aligned by documentation.
- Equal weighting in `payment_risk_score` is a heuristic assumption.
- Interaction features can amplify source outliers and correlations.
- Usage/service inconsistencies from earlier audits remain unresolved.
- No timestamps are available to prove prediction-time availability.
- No model has tested whether these representations improve validation or business-ranking metrics.

`charges_per_tenure` is deferred rather than implemented. Earlier audits found unexplained cumulative-charge discrepancies, and `total_charges` may reflect price changes, discounts, or partial periods. Target-derived tenure stages and usage thresholds are also deferred because any learned cut points must be selected within training folds and justified during modeling.

## 15. Recommendations for Modeling

1. Confirm field timestamps, target semantics, satisfaction scale ownership, and charge/usage definitions with the data owner.
2. Place the Phase 5 classifier after `build_feature_preprocessing_pipeline()` so every cross-validation fold refits preprocessing independently.
3. Compare a baseline with original predictors against the seven-feature representation; do not assume improvement.
4. Use regularization or feature-ablation checks for highly correlated engineered fields.
5. Keep the 240-row final test partition sealed until final model selection is complete.
6. Evaluate ranking quality in the highest-risk 20% in addition to imbalance-aware classification metrics.
