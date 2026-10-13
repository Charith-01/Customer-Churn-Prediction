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
|   |-- 03_exploratory_data_analysis.ipynb
|   |-- 04_feature_engineering.ipynb
|   `-- 05_model_development.ipynb    # Phase 5 training-only model comparison
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py             # Reusable cleaning, split, and preprocessing code
|   |-- feature_engineering.py       # Predictor-only feature transformer and pipeline
|   `-- modeling.py                  # CV, tuning, ranking metrics, and model persistence
|-- models/                          # Local saved model artifacts (ignored by Git)
|-- reports/
|   |-- figures/                     # Presentation-ready PNG charts
|   |-- modeling/                    # Aggregate Phase 5 CSV and JSON results
|   |-- data_audit.md                # Phase 1 findings
|   |-- data_cleaning_report.md      # Phase 2 decisions
|   |-- eda_analysis.md              # Phase 3 findings
|   |-- feature_engineering_report.md
|   `-- model_development_report.md
|-- tests/
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   `-- test_modeling.py
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

## Feature engineering

Phase 4 adds seven deterministic numerical features without using target values or identifiers:

- `tenure_years`
- `complaints_per_active_month`
- `payment_risk_score`
- `complaint_satisfaction_interaction`
- `monthly_contract_tenure`
- `late_payment_charge_interaction`
- `usage_per_charge`

`src/feature_engineering.py` provides a scikit-learn-compatible `CustomerFeatureEngineer` and `build_feature_preprocessing_pipeline()`. The integrated sequence is predictor input → deterministic feature engineering → training-fitted numerical imputation/scaling and categorical imputation/one-hot encoding. A classifier will be appended only in Phase 5, keeping all fitted transformations inside each future cross-validation fold.

The original 18 predictors become 25 pre-encoding predictors with no row loss. `charges_per_tenure` remains deferred until cumulative-charge definitions are clarified. Temporal availability of usage, satisfaction, complaints, and payment history also requires business confirmation. Details and training-only statistics are in `reports/feature_engineering_report.md`; four additional figures use the `fe_` filename prefix under `reports/figures/`.

## Model development

Phase 5 compares a naive prior reference with Logistic Regression, Random Forest, and Histogram Gradient Boosting. Each classifier uses one complete leakage-safe pipeline:

`predictors → Phase 4 feature engineering → numerical/categorical preprocessing → classifier`

Model development uses only the fixed 960-row training partition. Every baseline, class-weight experiment, and hyperparameter candidate uses the same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` folds. `RandomizedSearchCV` tests 16 configurations for each required model (48 configurations and 240 fits total), with Average Precision as the primary refit metric. Precision, recall, F1, ROC-AUC, PR-AUC, accuracy, precision@20%, recall@20%, and lift are retained for comparison.

The provisional winner is the tuned Random Forest: mean CV PR-AUC **0.9782 ± 0.0060**, ROC-AUC **0.8822**, and precision@20% **0.9949**. It uses 400 trees, `max_features=0.5`, `min_samples_split=10`, `min_samples_leaf=2`, unrestricted depth, and balanced class weights. This is a training-CV result, not final held-out performance.

Phase 5 outputs are located at:

- `reports/model_development_report.md` — interpreted results and limitations
- `reports/modeling/` — aggregate fold, comparison, search, and metadata files
- `reports/figures/m5_*.png` — five model-comparison charts
- `models/phase5_candidate_model.joblib` — local provisional fitted pipeline, ignored by Git
- `data/processed/05_selected_model_oof_probabilities.csv` — local training-only out-of-fold probabilities, ignored by Git

The serialized model and row-level out-of-fold file are intentionally excluded from version control. The final 240-row test partition remains sealed.

## Run the notebooks

Start JupyterLab from the repository root:

```powershell
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_data_cleaning_preprocessing.ipynb`
3. `notebooks/03_exploratory_data_analysis.ipynb`
4. `notebooks/04_feature_engineering.ipynb`
5. `notebooks/05_model_development.ipynb`

All notebooks resolve project paths safely when launched from the repository root or the `notebooks` directory.

For a command-line execution check:

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_data_audit.ipynb --output 01_data_audit.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning_preprocessing.ipynb --output 02_data_cleaning_preprocessing.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/03_exploratory_data_analysis.ipynb --output 03_exploratory_data_analysis.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering.ipynb --output 04_feature_engineering.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/05_model_development.ipynb --output 05_model_development.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=900
```

## Run the tests

From the repository root:

```powershell
python -m pytest tests -q
```

To run the Phase 5 modeling tests only:

```powershell
python -m pytest tests/test_modeling.py -q
```

The tests cover deterministic cleaning and feature engineering, raw-file preservation, missing and zero-denominator handling, unknown categories, target/identifier leakage, fixed-split reproducibility, train/test separation, training-only imputation, model construction, valid probabilities and class ordering, stratified CV, top-20% tie handling, candidate persistence, and final-test isolation.

The executed Phase 5 environment used Python 3.13.3, pandas 3.0.6, NumPy 2.5.4, Matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, and joblib 1.6.0. Install the repository's declared dependency ranges from `requirements.txt`; exact results can vary across compatible library versions.

## Project status

**Phase 5 complete:** three required classifiers and a naive reference were trained and compared with leakage-safe five-fold CV, bounded hyperparameter tuning, class-weight experiments, and training-only top-20% ranking analysis. A tuned Random Forest is saved locally as the provisional candidate. The fixed split was preserved and no held-out test metric or prediction was calculated.

**Next phase:** Class Imbalance, Threshold Selection, Calibration and Final Evaluation, after approval. Final-test evaluation, final threshold selection, explainability, and retention simulation are outside Phase 5 and have not started.
