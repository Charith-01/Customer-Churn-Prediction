# Customer Churn Data Cleaning and Preprocessing Report

**Phase:** 2 — Data Cleaning and Preprocessing  
**Date:** 2026-10-13  
**Raw source:** `data/raw/customer_churn_advanced_dataset.csv`  
**Cleaned output:** `data/processed/02_cleaned_dataset.csv`  
**Fixed split metadata:** `data/processed/02_split_indices.json`

## Scope and safeguards

This phase applies only deterministic representation cleaning, creates a fixed stratified split, and defines reusable preprocessing. It does not perform feature engineering, advanced EDA, model training, tuning, or final-test evaluation. The raw and processed customer-level CSV files are excluded from Git.

The raw source remained byte-for-byte unchanged. Its SHA-256 before and after Phase 2 was:

`B9F3981F7D32BA3BFD2256F69B6E12F730BE79139B22DBA6CED563F53DB9462C`

## Original dataset summary

- Rows: **1,200**
- Columns: **20**
- Identifier: `customer_id`
- Target: `churn`
- Numerical predictors: **9**
- Categorical/binary predictors: **9**
- Completely duplicated rows: **0**
- Repeated customer IDs beyond the first occurrence: **0**
- Distinct customer IDs: **1,200**

The audited schema and Phase 1 files were present and complete. No Phase 1 artifact was replaced.

## Missing-value findings

| Column | Missing | Percentage | Phase 2 decision |
|---|---:|---:|---|
| `satisfaction_score` | 48 | 4.00% | Preserve in CSV; median-impute inside training pipeline |
| `monthly_usage_gb` | 36 | 3.00% | Preserve in CSV; median-impute inside training pipeline |
| `monthly_charges` | 30 | 2.50% | Preserve in CSV; median-impute inside training pipeline |
| `total_charges` | 24 | 2.00% | Preserve in CSV; median-impute inside training pipeline |
| All other columns | 0 | 0.00% | No action |

The dataset has **138 missing cells across 132 rows (11.00%)**. Complete-case deletion would discard customer information and may bias the data. No missing value was replaced in the cleaned CSV. Numerical imputation statistics are learned only from training data and will be refitted inside each later cross-validation fold.

No categorical predictor is missing in the current file. The reusable categorical pipeline nevertheless represents future missing categories explicitly as `Missing`, which also handles a feature that is entirely missing within a training fold.

## Numerical validation

The numerical columns loaded as numeric values. The reusable validator rejects non-numeric non-missing input and checks broad constraints that do not impose undocumented business rules:

- age, tenure, charges, counts, and usage cannot be negative;
- `satisfaction_score` must remain within the observed/documented audit scale of 0–10;
- `discount_pct` must remain within 0–100;
- count fields and the `senior_citizen` indicator must contain whole numbers;
- `senior_citizen` must contain only 0 and 1.

The actual dataset produced **zero failures** for these screens. Phase 2 does not derive `senior_citizen` from age because no defining threshold is documented.

## Categorical validation

The eight text predictors contain:

- no blank strings;
- no leading or trailing whitespace;
- no case-only duplicate levels identified in Phase 1;
- no missing values.

Deterministic cleaning trims surrounding whitespace and converts truly blank text to missing. These defensive rules changed **zero actual values**. No category labels were standardized because no clearly equivalent conflicting labels were found and no authoritative data dictionary was available.

## Duplicate findings

- Completely duplicated rows: **0**
- Duplicate customer IDs: **0**
- Rows removed as duplicates: **0**

The reusable validator stops if IDs repeat after whitespace normalization. This prevents one customer from silently entering both training and test data. No grouping intervention was required for the current dataset.

## Outlier findings

Potential outliers were rechecked with the 1.5 × IQR rule.

| Field | Flagged records |
|---|---:|
| `age` | 0 |
| `tenure_months` | 0 |
| `monthly_charges` | 12 |
| `total_charges` | 14 |
| `complaints_last_12m` | 41 |
| `late_payments_last_12m` | 57 |
| `satisfaction_score` | 4 |
| `monthly_usage_gb` | 5 |
| `discount_pct` | 2 |

There are **135 field-level flags across 125 unique rows**. IQR flags are not proof of bad data, especially for discrete count fields. No outlier was removed, capped, or overwritten.

## Deterministic cleaning decisions

| Identified risk | Selected method | Rationale | Actual effect |
|---|---|---|---|
| Accidental whitespace in text | Strip surrounding whitespace | Prevent false categories and normalized-ID collisions | 0 values changed |
| Blank text values | Represent as missing | A blank is not a reliable business category | 0 values changed |
| Numeric text in audited numeric columns | Strict numeric parsing | Preserve valid numbers and reject ambiguous text | Types validated; 0 values changed |
| Binary target representation | Map unambiguous `0` to 0 and `1` to 1 | Makes label handling explicit | 0 values changed |
| Missing numeric values | Preserve in cleaned CSV | Avoid whole-dataset leakage | 138 missing cells preserved |
| Duplicates | Validate and report; do not silently drop | Removal requires evidence | None found |
| Outliers | Retain and document | Unusual behavior can be legitimate and predictive | 125 flagged rows retained |
| Cross-field questions | Retain and document | No authoritative relationship rules exist | No records altered |

## Records changed or excluded

- Original dimensions: **1,200 × 20**
- Cleaned dimensions: **1,200 × 20**
- Rows excluded: **0**
- Cells with changed values: **0**
- Missing cells before and after: **138**

Because the actual source already satisfied the justified deterministic rules, the generated cleaned CSV is byte-identical to the source CSV and has the same SHA-256. It remains a separate artifact so later phases use an explicitly validated input and never write to the source location.

## Remaining unresolved issues

The following findings cannot be corrected without business documentation or source-system verification:

1. `churn = 1` represents 86.33% of customers. Confirm label semantics, observation window, and whether the population was preselected.
2. Five customers have `total_charges` more than 50% below `monthly_charges × tenure_months`. Price changes, discounts, or partial billing may explain this.
3. Of 149 customers with `internet_service = No`, 147 have positive `monthly_usage_gb`, and 56 have `tech_support = Yes`. Confirm field scope.
4. `support_plan` and `tech_support` may be independent products; no consistency rule was imposed.
5. The definition of `senior_citizen` is unknown, so it was not recalculated from age.
6. IQR extremes require verification rather than automatic treatment.

## Target encoding

Observed target labels were already binary:

| Original label | Encoded label | Count | Percentage |
|---:|---:|---:|---:|
| 0 | 0 | 164 | 13.67% |
| 1 | 1 | 1,036 | 86.33% |

Any missing label or value other than unambiguous 0/1 raises an informative error. The target is separated before preprocessing.

## Train/test split strategy

An **80/20 stratified split** was created with `random_state=42`.

| Partition | Rows | Class 0 | Class 0 % | Class 1 | Class 1 % |
|---|---:|---:|---:|---:|---:|
| Training | 960 | 131 | 13.65% | 829 | 86.35% |
| Final test | 240 | 33 | 13.75% | 207 | 86.25% |

- Row-index overlap: **0**
- Customer-ID overlap: **0**
- All 1,200 rows occur in exactly one partition.
- Dataset fingerprint: `df788deba598b73abdc3a41b8a1947af63a90c5f369c2a498386a2c0eefeb769`

`02_split_indices.json` stores row indices, the fingerprint, target counts, test size, random seed, and target mapping. Later calls reload the exact indices only when the data fingerprint and split settings match. A changed dataset raises an error instead of silently producing a different split.

## Preprocessing pipeline

`src/preprocessing.py` constructs an unfitted scikit-learn `ColumnTransformer` suitable for placement inside Logistic Regression, Random Forest, or Gradient Boosting pipelines.

### Numerical branch

1. `SimpleImputer(strategy="median", keep_empty_features=True)`
2. `StandardScaler()`

Median values are learned from training rows only. If a feature is entirely missing in a training fold, `keep_empty_features=True` retains the column using scikit-learn's documented fallback because no median exists.

### Categorical branch

1. `SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)`
2. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

The explicit missing category is stable even for an entirely missing categorical feature. Unknown categories in held-out data do not cause transformation failures. Dense output is practical for this low-cardinality dataset and compatible with the three planned model families.

## Leakage-prevention measures

- The fixed split is created before fitting the transformer.
- `customer_id` and `churn` are excluded from the predictor matrix by an explicit schema.
- The transformer is returned unfitted and is fitted on `X_train` only during notebook validation.
- The final test partition is transformed using training-fitted parameters and is not used for fitting or predictive evaluation.
- The transformed arrays are not persisted; later cross-validation must fit the transformer independently inside each fold.
- The split fingerprint prevents accidental reuse with changed or reordered data.
- Duplicate IDs are rejected before splitting.

## Before-and-after summary

| Measure | Raw | Cleaned CSV | Decision |
|---|---:|---:|---|
| Rows | 1,200 | 1,200 | Preserve all customers |
| Columns | 20 | 20 | Preserve audited schema |
| Missing cells | 138 | 138 | Handle in training pipeline |
| Complete duplicates | 0 | 0 | No removal |
| Duplicate IDs | 0 | 0 | No grouping action required |
| Changed cells | — | 0 | No supported correction required |
| IQR-flagged rows | 125 | 125 | Retain unchanged |

## Phase 2 conclusion

The dataset is prepared for Phase 3 without changing customer evidence. A fixed, non-overlapping stratified split and reusable leakage-safe preprocessing implementation are available. Business clarification remains necessary for the target prevalence and cross-field questions, but those issues do not justify invented corrections. No modeling or final-test performance inspection occurred.
