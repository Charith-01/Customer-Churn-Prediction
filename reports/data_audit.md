# Customer Churn Dataset — Initial Data Audit

**Audit date:** 2026-10-13  
**Source:** `data/raw/customer_churn_advanced_dataset.csv`  
**Source SHA-256:** `B9F3981F7D32BA3BFD2256F69B6E12F730BE79139B22DBA6CED563F53DB9462C`

## Scope and safeguards

This Phase 1 audit profiles the original CSV without modifying customer records. It does not impute missing data, remove outliers, scale values, create predictive features, or train a model. `customer_id` is treated strictly as an identifier and `churn` as the target.

## Dataset overview

The dataset contains **1,200 rows and 20 columns**. There are 11 numeric columns at load time, including the binary target and the `senior_citizen` indicator, and 9 text columns, including `customer_id`.

| Role | Columns |
|---|---|
| Identifier | `customer_id` |
| Target | `churn` |
| Continuous/count predictors | `age`, `tenure_months`, `monthly_charges`, `total_charges`, `complaints_last_12m`, `late_payments_last_12m`, `satisfaction_score`, `monthly_usage_gb`, `discount_pct` |
| Categorical/binary predictors | `contract_type`, `internet_service`, `region`, `payment_method`, `paperless_billing`, `dependents`, `support_plan`, `tech_support`, `senior_citizen` |

All 1,200 customer IDs are non-missing, unique, and match the observed `CUST` plus six-digit format. This format check describes the current file; it is not a substitute for a documented source-system rule.

## Target distribution

| `churn` | Count | Percentage |
|---:|---:|---:|
| 0 | 164 | 13.67% |
| 1 | 1,036 | 86.33% |

The target is complete and binary, but highly concentrated in class 1 (about 6.3 class-1 records for each class-0 record). The business owner should confirm that `1` means churn and that the unusually high churn prevalence reflects the intended population and observation window. Later evaluation should use a stratified split and imbalance-aware metrics rather than accuracy alone.

## Missing-value summary

| Column | Missing | Percentage |
|---|---:|---:|
| `satisfaction_score` | 48 | 4.00% |
| `monthly_usage_gb` | 36 | 3.00% |
| `monthly_charges` | 30 | 2.50% |
| `total_charges` | 24 | 2.00% |
| All other columns | 0 | 0.00% |

There are **138 missing cells across 132 rows**; therefore, **11.00% of rows** have at least one missing value. Most affected rows have only one missing field. One row is missing both charge fields. Missingness must be investigated and handled only after the train/test split so fitted imputation cannot leak information from evaluation data.

## Duplicate checks

- Completely duplicated rows: **0**
- Missing customer IDs: **0**
- Customer IDs repeated beyond their first occurrence: **0**
- Distinct customer IDs: **1,200**

No deletion is required from these checks. The identifier should still be excluded from all model inputs and retained only for traceability and final customer-level outputs.

## Data types and category checks

The inferred types are suitable for an initial audit: counts and measurements load as numeric values, and descriptive categories load as text. The `churn` and `senior_citizen` fields contain only 0 and 1. Categorical fields have no blank strings, leading/trailing whitespace, or case-only duplicate labels.

Observed category cardinalities are plausible for encoding later: `contract_type` and `internet_service` each have 3 values, `region` has 9, `payment_method` has 4, and each Yes/No field has 2. Category meanings and allowed-value lists should nevertheless be confirmed against a data dictionary.

## Numerical data quality

| Field | Non-missing | Minimum | Median | Maximum | Initial assessment |
|---|---:|---:|---:|---:|---|
| `age` | 1,200 | 18 | 47 | 75 | Within the audit screening range 18–100 |
| `tenure_months` | 1,200 | 1 | 35 | 72 | Positive; unit and 72-month ceiling need documentation |
| `monthly_charges` | 1,170 | 20.00 | 71.82 | 235.01 | No zero/negative values; 30 missing |
| `total_charges` | 1,176 | 22.73 | 2,244.46 | 8,853.76 | No negative values; 24 missing |
| `complaints_last_12m` | 1,200 | 0 | 1 | 5 | Non-negative integers |
| `late_payments_last_12m` | 1,200 | 0 | 1 | 4 | Non-negative integers |
| `satisfaction_score` | 1,152 | 1.0 | 6.75 | 10.0 | Within 0–10 screening range; scale needs confirmation |
| `monthly_usage_gb` | 1,164 | 2.0 | 66.4 | 466.9 | Non-negative; 36 missing and a high extreme |
| `discount_pct` | 1,200 | 0.0 | 8.715 | 32.89 | Within 0–100 screening range |

No values failed the notebook's broad audit screens for age, non-negative tenure/charges/counts/usage, satisfaction from 0 to 10, discount from 0% to 100%, or binary indicator/target fields. These screens are provisional and must be replaced or confirmed with documented business constraints.

## Potential outliers

The notebook applies the standard **1.5 × IQR rule** separately to each numeric predictor. This is a statistical flag, not proof that a record is wrong.

| Field | Lower bound | Upper bound | Flagged | Flagged observed range |
|---|---:|---:|---:|---:|
| `age` | -11.50 | 104.50 | 0 | — |
| `tenure_months` | -36.00 | 108.00 | 0 | — |
| `monthly_charges` | 5.62 | 135.82 | 12 | 136.90–235.01 |
| `total_charges` | -2,829.97 | 7,701.22 | 14 | 7,704.76–8,853.76 |
| `complaints_last_12m` | -1.50 | 2.50 | 41 | 3–5 |
| `late_payments_last_12m` | -1.50 | 2.50 | 57 | 3–4 |
| `satisfaction_score` | 2.00 | 11.60 | 4 | 1.0–1.9 |
| `monthly_usage_gb` | -31.05 | 161.55 | 5 | 169.8–466.9 |
| `discount_pct` | -11.60 | 28.87 | 2 | 29.76–32.89 |

Counts such as complaints and late payments are discrete and right-skewed, so IQR can label valid uncommon behavior. All flagged records should remain in the data until their source values and business meaning are verified.

## Potential cross-field inconsistencies

These checks identify questions for the data owner; they do not impose undocumented business rules.

- **Charge relationship:** among the 1,147 rows where both charge fields are present, five have `total_charges` more than 50% below `monthly_charges × tenure_months`. The affected IDs are `CUST100023`, `CUST100361`, `CUST100605`, `CUST100636`, and `CUST100651`. Discounts, price changes, partial billing periods, or field definitions may explain the difference, so no correction is justified yet.
- **Service and usage meaning:** 147 of 149 customers labeled `internet_service = No` have positive `monthly_usage_gb`; 56 also have `tech_support = Yes`. Confirm whether usage and support cover products beyond the listed internet service, or whether these are inconsistent records.
- **Support fields:** 271 customers have `tech_support = Yes` with `support_plan = No`, while 306 show the reverse. These fields may represent independent products, so their relationship requires documentation before validation rules are written.
- **Senior indicator:** age and `senior_citizen` coexist, but no age threshold is documented. Do not recalculate or overwrite the indicator until its source definition is confirmed.

## Data quality concerns and churn relevance

1. The high churn rate may represent a preselected high-risk population, reversed label semantics, or genuine prevalence. This directly affects evaluation design and the interpretation of a top-20% targeting simulation.
2. Missing values affect four potentially important behavioral/financial fields. Complete-case deletion would discard 11% of customers and could introduce bias.
3. A small number of extreme charges and usage values can influence linear models and should be verified before deciding on transformations.
4. The relationship between total charges, monthly charges, tenure, and discounts needs a documented definition before any derived financial feature is considered.
5. `customer_id` is unique but carries no intended predictive meaning and poses an overfitting risk if encoded.

## Proposed Phase 2 cleaning strategy

Subject to approval and business clarification:

1. Preserve the source CSV as immutable and write any cleaned output to `data/processed/`.
2. Confirm target semantics, field definitions, units, category dictionaries, and valid ranges with the data owner.
3. Keep `customer_id` only as a row identifier; exclude it from the predictive matrix.
4. Create a stratified train/test split before learning any preprocessing parameter.
5. Investigate why values are missing. Fit numerical imputers on training data only, consider explicit missingness indicators where operationally meaningful, and apply the fitted pipeline to validation/test data.
6. Review the five charge-relationship records and service/usage combinations against source records. Correct values only with evidence; otherwise retain and document them.
7. Retain IQR-flagged observations initially. If verified extremes materially destabilize a model, compare robust transformations or capping thresholds learned from training data and document the effect.
8. Encode categorical variables and perform any scaling inside a training-only pipeline.
9. Use stratified cross-validation and report precision, recall, F1, PR-AUC, ROC-AUC, confusion matrices, and business targeting metrics; do not rely on accuracy alone.

## Phase 1 conclusion

The file is structurally usable: identifiers are complete and unique, the target is binary and complete, categories are consistently written, and broad numerical range checks find no definite invalid values. Phase 2 should focus on documented validation, leakage-safe missing-value handling, investigation of cross-field questions, and an evaluation plan appropriate for the target imbalance. No data has been cleaned or removed in this phase.
