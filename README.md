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
|   |-- 05_model_development.ipynb    # Phase 5 training-only model comparison
|   |-- 06_model_evaluation.ipynb     # Locked policy and final evaluation
|   `-- 07_model_explainability.ipynb # Global, local, and error explanations
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py             # Reusable cleaning, split, and preprocessing code
|   |-- feature_engineering.py       # Predictor-only feature transformer and pipeline
|   |-- modeling.py                  # CV, tuning, ranking metrics, and model persistence
|   |-- evaluation.py                # Calibration, thresholds, metrics, and robustness
|   `-- explainability.py            # Model-preserving global/local explanation utilities
|-- models/                          # Local saved model artifacts (ignored by Git)
|-- reports/
|   |-- figures/                     # Presentation-ready PNG charts
|   |-- modeling/                    # Aggregate Phase 5 CSV and JSON results
|   |-- evaluation/                  # Aggregate Phase 6 evaluation results
|   |-- explainability/              # Aggregate/anonymized Phase 7 explanation tables
|   |-- data_audit.md                # Phase 1 findings
|   |-- data_cleaning_report.md      # Phase 2 decisions
|   |-- eda_analysis.md              # Phase 3 findings
|   |-- feature_engineering_report.md
|   |-- model_development_report.md
|   |-- model_evaluation_report.md
|   `-- model_explainability_report.md
|-- tests/
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   |-- test_modeling.py
|   |-- test_evaluation.py
|   `-- test_explainability.py
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

The serialized model and row-level out-of-fold file are intentionally excluded from version control. The final 240-row test partition remained sealed throughout Phase 5.

## Final model evaluation

Phase 6 retains the tuned Random Forest and balanced class weighting selected in Phase 5. Sigmoid calibration was selected through nested training-only cross-validation because it improved OOF Brier score from **0.1061** to **0.0840** and log loss from **0.3328** to **0.2752** without reducing ranking performance.

The fixed operating threshold is **0.970709**, selected from sigmoid-calibrated OOF probabilities to maximize churner recall subject to a 20% training contact-rate cap. At that OOF threshold, precision was **0.9948**, recall was **0.2304**, and 192 of 960 training rows were selected.

After the policy was persisted, the 240-row test partition was evaluated once. Final probability and ranking results were:

- ROC-AUC: **0.8994**
- Average Precision: **0.9819**, versus 86.25% test churn prevalence
- Brier score: **0.0825**
- Log loss: **0.2627**

At the locked threshold, accuracy was **0.5417**, precision **0.9899**, recall **0.4734**, and F1 **0.6405**. The confusion matrix contained 32 true negatives, 1 false positive, 109 false negatives, and 98 true positives.

The fixed threshold selected **41.25%** of test rows, despite selecting 20% in training OOF data. It was not revised after test evaluation. This is an important operational limitation: a fixed probability cutoff cannot guarantee the contact cap when score distributions shift. A strict campaign limit requires deterministic top-k ranking, which is deferred to Phase 8.

Phase 6 outputs are located at:

- `reports/model_evaluation_report.md` — final metrics, calibration, limitations, and interpretation
- `reports/evaluation/` — locked policy, aggregate metrics, calibration bins, and robustness tables
- `reports/figures/phase6_*.png` — eight evaluation charts
- `models/03_model_artifact.joblib` — local complete calibrated model, ignored by Git
- `models/final_model_metadata.json` — locked model and policy metadata

Only trusted joblib artifacts should be loaded. No explainability or retention simulation was performed in Phase 6.

## Model explainability

Phase 7 explains the unchanged calibrated Random Forest and locked threshold. Global explanation uses 10-repeat original-feature permutation importance on the held-out test partition, scored by Average Precision. `monthly_usage_gb` is the dominant input (mean AP decrease **0.1072 +/- 0.0164**), followed by contract type (**0.0044**), satisfaction (**0.0036**), and late payments (**0.0017**). Observed feature-effect and usage-by-satisfaction summaries add direction context without making causal claims.

Three anonymized test cases—a true positive, true negative, and near-threshold false negative—use model-agnostic one-feature-at-a-time probability sensitivity against training-derived references. The error review preserves the Phase 6 counts of 98 true positives, 32 true negatives, 1 false positive, and 109 false negatives. No customer identifier is included in explanation outputs.

Phase 7 outputs are located at:

- `reports/model_explainability_report.md` — methods, findings, limitations, and Phase 8 recommendations
- `reports/explainability/` — global importance, feature effects, error summaries, anonymized cases, and integrity checks
- `reports/figures/phase7_*.png` — seven explanation charts
- `src/explainability.py` — validated reusable explanation utilities

The model artifact SHA-256, probability predictions, calibration, and operating threshold **0.970709** were verified unchanged. Explanations characterize fitted-model behavior and associations, not causal effects. Retention simulation remains deferred to Phase 8.

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
6. `notebooks/06_model_evaluation.ipynb`
7. `notebooks/07_model_explainability.ipynb`

All notebooks resolve project paths safely when launched from the repository root or the `notebooks` directory.

For a command-line execution check:

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_data_audit.ipynb --output 01_data_audit.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning_preprocessing.ipynb --output 02_data_cleaning_preprocessing.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/03_exploratory_data_analysis.ipynb --output 03_exploratory_data_analysis.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering.ipynb --output 04_feature_engineering.ipynb --output-dir notebooks
jupyter nbconvert --to notebook --execute notebooks/05_model_development.ipynb --output 05_model_development.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute notebooks/06_model_evaluation.ipynb --output 06_model_evaluation.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1200
jupyter nbconvert --to notebook --execute notebooks/07_model_explainability.ipynb --output 07_model_explainability.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1200
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

To run the Phase 6 evaluation tests only:

```powershell
python -m pytest tests/test_evaluation.py -q
```

To run the Phase 7 explainability tests only:

```powershell
python -m pytest tests/test_explainability.py -q
```

The tests cover deterministic cleaning and feature engineering, raw-file preservation, missing and zero-denominator handling, unknown categories, target/identifier leakage, fixed-split reproducibility, train/test separation, training-only imputation, model construction, valid probabilities and class ordering, stratified CV, top-20% tie handling, threshold metrics, capacity and probability-tie behavior, calibration diagnostics, robustness summaries, artifact persistence, inference consistency, deterministic permutation importance, local sensitivity, anonymization, confusion categories, and before/after prediction preservation.

The executed Phase 6–7 environment used Python 3.13.3, pandas 3.0.6, NumPy 2.5.4, Matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, and joblib 1.6.0. Install the repository's declared dependency ranges from `requirements.txt`; exact results can vary across compatible library versions.

## Project status

**Phase 7 complete:** the locked calibrated Random Forest was explained globally and for three anonymized customer cases. Original-feature permutation importance identifies monthly usage as the dominant ranking signal, and error analysis documents the threshold's high-precision/low-recall behavior. Artifact, prediction, data, split, and threshold integrity checks all pass.

**Next phase:** Retention Intervention Simulation, after approval. No intervention simulation, uplift claim, or cost-benefit analysis has started.
