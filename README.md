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
- ReportLab, PyMuPDF, and Pillow for deterministic PDF generation, rendering, and visual-review artifacts

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
|   |-- 07_model_explainability.ipynb # Global, local, and error explanations
|   `-- 08_retention_simulation.ipynb # Capacity and business-impact simulation
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py             # Reusable cleaning, split, and preprocessing code
|   |-- feature_engineering.py       # Predictor-only feature transformer and pipeline
|   |-- modeling.py                  # CV, tuning, ranking metrics, and model persistence
|   |-- evaluation.py                # Calibration, thresholds, metrics, and robustness
|   |-- explainability.py            # Model-preserving global/local explanation utilities
|   |-- retention_simulation.py      # Capacity, baseline, and economics utilities
|   `-- reporting.py                 # Deterministic final-report generation and validation
|-- models/                          # Local saved model artifacts (ignored by Git)
|-- reports/
|   |-- figures/                     # Presentation-ready PNG charts
|   |-- business_report_preview/     # High-resolution pages and seven-page contact sheet
|   |-- modeling/                    # Aggregate Phase 5 CSV and JSON results
|   |-- evaluation/                  # Aggregate Phase 6 evaluation results
|   |-- explainability/              # Aggregate/anonymized Phase 7 explanation tables
|   |-- data_audit.md                # Phase 1 findings
|   |-- data_cleaning_report.md      # Phase 2 decisions
|   |-- eda_analysis.md              # Phase 3 findings
|   |-- feature_engineering_report.md
|   |-- model_development_report.md
|   |-- model_evaluation_report.md
|   |-- model_explainability_report.md
|   |-- retention_simulation_report.md
|   |-- retention_simulation_results.csv
|   |-- retention_sensitivity_analysis.csv
|   |-- business_report_content.md   # Auditable narrative source for the PDF
|   `-- report_source_mapping.md     # Report-to-evidence traceability map
|-- tests/
|   |-- test_preprocessing.py
|   |-- test_feature_engineering.py
|   |-- test_modeling.py
|   |-- test_evaluation.py
|   |-- test_explainability.py
|   |-- test_retention_simulation.py
|   `-- test_reporting.py
|-- 04_business_report.pdf           # Canonical seven-page business report
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

## Retention intervention simulation

Phase 8 applies a strict **20% contact capacity** to the unchanged held-out churn probabilities. Customers are ranked by probability descending; exact ties use original held-out order as a stable, target-independent secondary key. Capacity is calculated with `floor(N × 0.20)`, so 48 of 240 customers are selected. This top-k policy is separate from—and does not alter—the locked Phase 6 threshold.

All 48 selected customers historically churned. Measured ranking results are precision@20% **1.0000**, recall/capture@20% **0.2319**, and lift@20% **1.1594**. Uniform random selection with the same 48-contact budget would capture **41.4 churners in expectation**, compared with 48 observed in the model-ranked group. The advantage is meaningful but bounded by the test sample's high **86.25%** churn prevalence.

The base financial scenario uses illustrative assumptions only: **$5** per contact, **$150** value per successfully retained customer, and **10%** intervention effectiveness. It estimates model-ranked net benefit of **$480**, random-selection net benefit of **$381**, and incremental expected benefit of **$99**. These are not company financials, realized savings, or evidence of causal retention impact. A 27-scenario sensitivity grid varies effectiveness, cost, and value; a randomized pilot is recommended before rollout.

Phase 8 outputs are located at:

- `reports/retention_simulation_report.md` — business interpretation, recommendations, risks, and operational workflow
- `reports/retention_simulation_results.csv` — aggregate measured results, assumptions, and base estimates
- `reports/retention_sensitivity_analysis.csv` — 27 hypothetical financial scenarios
- `reports/retention_simulation_integrity.json` — model, prediction, split, and data preservation checks
- `reports/figures/phase8_*.png` — five ranking and financial charts
- `data/processed/internal_retention_ranking.csv` — protected, Git-ignored retrospective analysis table with no raw customer IDs

No API, application, database, or live campaign is produced in this phase.

## Final business report

Phase 9 consolidates the verified Phase 1-8 evidence into the canonical seven-page A4 report at `04_business_report.pdf`. It is written for business stakeholders while preserving technical traceability: the report distinguishes measured results, fitted-model interpretations, hypothetical financial scenarios, and proposed actions. It covers the business problem, dataset and preparation, EDA, feature engineering, model comparison, locked held-out evaluation, calibration, global and customer-level explainability, top-20% retention simulation, recommendations, roadmap, limitations, and conclusion.

The report is generated deterministically from `reports/business_report_content.md` and the validated analytical outputs. `reports/report_source_mapping.md` maps every report section to its source evidence, while `reports/business_report_preview/` contains seven high-resolution rendered pages and `contact_sheet.png` for visual inspection. Report-specific EDA and model-comparison charts are regenerated directly from the locked training split and model-comparison output. Regenerate and validate the PDF from the repository root with:

```powershell
python -m src.reporting
```

The generator checks the locked model hash, split fingerprint, Phase 7-8 integrity evidence, report metrics, page count, portrait layout, page text, required business interpretations, unresolved placeholders, and customer-ID privacy. It reads but never fits or modifies the model. ReportLab creates the PDF, PyMuPDF validates and renders it, and Pillow assembles the visual-review contact sheet.

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
8. `notebooks/08_retention_simulation.ipynb`

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
jupyter nbconvert --to notebook --execute notebooks/08_retention_simulation.ipynb --output 08_retention_simulation.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1200
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

To run the Phase 8 retention-simulation tests only:

```powershell
python -m pytest tests/test_retention_simulation.py -q
```

To run the Phase 9 reporting tests only:

```powershell
python -m pytest tests/test_reporting.py -q
```

The tests cover deterministic cleaning and feature engineering, raw-file preservation, missing and zero-denominator handling, unknown categories, target/identifier leakage, fixed-split reproducibility, train/test separation, training-only imputation, model construction, valid probabilities and class ordering, stratified CV, top-20% tie handling, threshold metrics, capacity and probability-tie behavior, calibration diagnostics, robustness summaries, artifact persistence, inference consistency, deterministic permutation importance, local sensitivity, anonymization, confusion categories, capacity-safe ranking, random baselines, retention economics, sensitivity scenarios, before/after prediction preservation, deterministic PDF generation, evidence consistency, page completeness, visual-source availability, and report privacy.

The executed Phase 6-9 environment used Python 3.13.3, pandas 3.0.6, NumPy 2.5.4, Matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, joblib 1.6.0, ReportLab 4.4.10, PyMuPDF 1.28.2, and Pillow 12.3.0. Install the repository's declared dependency ranges from `requirements.txt`; exact results can vary across compatible library versions.

## Project status

**Phase 9.1 complete:** the verified Phase 1-8 evidence is consolidated into a corrected, submission-ready `04_business_report.pdf`. The seven-page report uses larger report-specific charts, readable tables and captions, balanced page layouts, professional headers and footers, explicit low-accuracy interpretation, calibrated-probability context, and a prominent distinction between locked-threshold classification and capacity-safe top-20% ranking. All pages were programmatically validated and visually inspected; the source mapping, privacy checks, required metrics, data and model integrity checks, and byte-level reproducibility checks pass.

**Next phase:** Final Project Integration and Submission, after approval. No submission ZIP or Phase 10 packaging has been created.
