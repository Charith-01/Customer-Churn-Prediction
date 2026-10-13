# Customer Churn Prediction and Retention Analysis

An end-to-end data science project for predicting subscription-customer churn and supporting evidence-based retention decisions. This repository is being developed as an internship portfolio project.

## Business problem

Customer churn reduces recurring revenue and increases replacement costs. The business needs a reproducible way to identify customers who may churn during the next retention cycle, understand the factors associated with that risk, and prioritize retention activity.

## Objectives

- Audit and clean the source data without changing the original file.
- Explore customer and churn patterns.
- Build and compare Logistic Regression, Random Forest, and Gradient Boosting models.
- Handle class imbalance and tune models without data leakage.
- Explain model behavior and simulate targeting the highest-risk 20% of customers.
- Translate the results into a professional business report.

`customer_id` is an identifier only and will not be used as a predictive feature. The target is `churn`.

## Technology stack

- Python 3.11+
- pandas and NumPy for data work
- Matplotlib and seaborn for visual analysis
- scikit-learn for preprocessing, modeling, and evaluation
- JupyterLab for reproducible notebooks

## Repository structure

```text
customer-churn-prediction/
|-- data/
|   |-- raw/                         # Original local data (ignored by Git)
|   `-- processed/                   # Cleaned data created in a later phase
|-- notebooks/
|   |-- 01_data_audit.ipynb          # Phase 1 audit
|   |-- 02_data_cleaning_preprocessing.ipynb
|   `-- 03_exploratory_data_analysis.ipynb
|-- src/
|   |-- __init__.py
|   `-- preprocessing.py             # Reusable cleaning, split, and pipeline code
|-- models/                          # Saved model artifacts
|-- reports/
|   |-- figures/                     # Phase 3 presentation-ready PNG charts
|   |-- data_audit.md                # Phase 1 findings
|   |-- data_cleaning_report.md      # Phase 2 decisions
|   `-- eda_analysis.md              # Phase 3 findings
|-- tests/
|   `-- test_preprocessing.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

Empty directories contain `.gitkeep` placeholders so Git can preserve the structure.

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Dataset

Place the original dataset at:

```text
data/raw/customer_churn_advanced_dataset.csv
```

Raw and processed customer-level CSV files are intentionally excluded from Git. Do not rename, edit, or commit the source CSV.

Phase 2 creates these local artifacts:

```text
data/processed/02_cleaned_dataset.csv
data/processed/02_split_indices.json
```

The cleaned CSV preserves all customer records and genuine missing values. The JSON file stores the fixed split indices and dataset fingerprint so later phases reuse the same partition. Both are local generated artifacts and are ignored by Git.

## Dataset preparation approach

Phase 2 uses conservative, deterministic cleaning:

- text whitespace is normalized and truly blank text is represented as missing;
- audited numerical fields and binary target labels are validated;
- duplicates, invalid values, and IQR outliers are reported rather than silently removed;
- no missing value is filled in the persisted cleaned CSV;
- `customer_id` is retained separately for traceability and excluded from predictors.

The actual dataset required no value correction and no row exclusion. Missing numerical values are handled only inside a training-fitted preprocessing pipeline.

## Preprocessing architecture

`src/preprocessing.py` provides reusable functions for loading and validation, deterministic cleaning, role separation, fixed-split creation/loading, feature grouping, and construction of an unfitted scikit-learn `ColumnTransformer`.

- Numerical predictors: median imputation followed by standard scaling
- Categorical predictors: explicit `Missing` category followed by one-hot encoding
- Unknown categories: ignored safely during transformation
- Empty features: retained where necessary
- Identifier and target: explicitly blocked from preprocessing

The transformer is designed to be placed inside a complete model `Pipeline`, ensuring preprocessing is refitted within each training fold in later phases.

## Fixed train/test split

- Training/test ratio: **80/20**
- Strategy: stratified by `churn`
- Random seed: **42**
- Training rows: **960**
- Final test rows: **240**
- Row and customer overlap: **0**

The final test partition must remain untouched until final model evaluation. A dataset fingerprint prevents the saved indices from being reused with changed or reordered data.

## Exploratory data analysis

Phase 3 uses only the saved 960-row training partition for target-aware analysis; the 240-row final test partition remains sealed. The EDA notebook covers target imbalance, contract, tenure, charges, satisfaction, complaints, late payments, usage, payment method, nonlinear patterns, feature interactions, and unusual observations.

The main training-sample findings are:

- churn is the majority class at **86.35%**, so accuracy alone will not be an adequate model metric;
- month-to-month customers have a **91.10%** churn rate, compared with **80.00%** for one-year and **79.08%** for two-year contracts;
- usage shows the strongest descriptive separation: churn rises from **57.45%** in the lowest training-derived usage quintile to **99.47%** in the highest;
- repeated late payments, three or more complaints, and lower satisfaction are associated with elevated churn, with small-group uncertainty noted in the report;
- tenure, monthly charges, complaints, and usage show nonlinear or threshold-like patterns.

These are associations rather than causal conclusions. The complete analysis is in `reports/eda_analysis.md`, and the 12 exported charts are under `reports/figures/`.

## Run the notebooks

Start JupyterLab from the repository root:

```powershell
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_data_cleaning_preprocessing.ipynb`
3. `notebooks/03_exploratory_data_analysis.ipynb`

All notebooks resolve project paths safely when launched from the repository root or the `notebooks` directory.

For a command-line execution check:

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_data_audit.ipynb --output 01_data_audit.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning_preprocessing.ipynb --output 02_data_cleaning_preprocessing.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/03_exploratory_data_analysis.ipynb --output 03_exploratory_data_analysis.ipynb --output-dir notebooks
```

## Run the tests

From the repository root:

```powershell
python -m pytest tests -q
```

The tests cover deterministic cleaning, raw-file preservation, missing-value handling, unknown categories, feature-role leakage, fixed-split reproducibility, train/test separation, training-only imputation, and invalid target labels.

## Project status

**Phase 3 complete:** the saved training split has been analyzed, 12 presentation-ready figures have been generated, and the findings and future feature proposals are documented. The final test set remained sealed. No feature engineering, model training, tuning, final evaluation, or retention simulation has been performed.

**Next phase:** Phase 4 — Feature Engineering, after approval of the EDA findings.
