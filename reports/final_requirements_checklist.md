# Final Assignment Requirements Checklist

Validated on 2026-10-15 against the repository, executed notebooks, saved aggregate
outputs, locked model metadata, generated PDF, automated tests, and private submission
package. `PASS` is used only where the cited evidence was inspected or executed.

| Requirement | Status | Supporting file | Evidence | Remaining issue |
|---|---|---|---|---|
| **A. Data Audit and Preparation** |  |  |  |  |
| Data types inspected | PASS | `notebooks/01_data_audit.ipynb`; `reports/data_audit.md` | Per-column types are audited; final notebook renders a type table. | None |
| Missing values inspected | PASS | `reports/data_audit.md`; `reports/data_cleaning_report.md` | Counts and percentages are reported by field. | None |
| Duplicate records checked | PASS | `reports/data_audit.md` | Full-row and identifier duplication checks report zero duplicates. | None |
| Invalid values reviewed | PASS | `src/preprocessing.py`; `reports/data_cleaning_report.md` | Schema, binary target, and domain ranges are validated. | None |
| Outliers identified | PASS | `reports/data_audit.md`; `notebooks/09_final_data_science_notebook.ipynb` | IQR candidates are calculated and reported without silent deletion. | None |
| Target distribution measured | PASS | `reports/eda_analysis.md` | Training churn prevalence is 86.35%; test prevalence is 86.25%. | None |
| Cleaning decisions documented | PASS | `reports/data_cleaning_report.md` | Conservative, row-preserving rules and no-value-correction result are documented. | None |
| Reproducible preprocessing pipeline | PASS | `src/preprocessing.py` | ColumnTransformer performs training-fitted imputation, scaling, and encoding. | None |
| `customer_id` excluded from predictive features | PASS | `src/preprocessing.py`; `tests/test_preprocessing.py` | Role separation and explicit leakage assertions block identifier and target inputs. | None |
| **B. Exploratory Data Analysis** |  |  |  |  |
| Contract type analysis | PASS | `reports/eda_analysis.md`; `reports/figures/02_contract_churn.png` | Training-only churn comparison is interpreted by contract type. | None |
| Tenure analysis | PASS | `reports/figures/03_tenure_churn.png` | Tenure distributions and nonlinear pattern are reviewed. | None |
| Charges analysis | PASS | `reports/figures/04_monthly_charges_churn.png` | Monthly-charge relationship is visualized and interpreted. | None |
| Satisfaction analysis | PASS | `reports/figures/05_satisfaction_churn.png` | Satisfaction relationship is visualized and interpreted. | None |
| Complaints analysis | PASS | `reports/figures/06_complaints_churn.png` | Complaint counts and small-group uncertainty are discussed. | None |
| Late-payment analysis | PASS | `reports/figures/07_late_payments_churn.png` | Late-payment relationship is visualized and interpreted. | None |
| Usage analysis | PASS | `reports/figures/08_usage_churn.png` | Usage quintiles show the strongest descriptive separation. | None |
| Payment-method analysis | PASS | `reports/figures/09_payment_method_churn.png` | Payment method is compared by churn outcome. | None |
| Nonlinear relationships | PASS | `reports/eda_analysis.md` | Threshold-like usage, tenure, charges, and complaint patterns are documented. | None |
| Feature interactions | PASS | `reports/figures/11_contract_tenure_interaction.png`; `12_satisfaction_complaints_interaction.png` | Two interaction views are included. | None |
| Unusual observations | PASS | `reports/eda_analysis.md` | Missingness, IQR candidates, and sparse groups are reviewed. | None |
| At least six informative visualizations | PASS | `reports/figures/01_churn_distribution.png` through `12_satisfaction_complaints_interaction.png` | Twelve Phase 3 charts exist; final notebook renders nine EDA panels. | None |
| Concise interpretations | PASS | `reports/eda_analysis.md`; `notebooks/09_final_data_science_notebook.ipynb` | Findings and causal limitations accompany the visuals. | None |
| **C. Feature Engineering** |  |  |  |  |
| At least five engineered features | PASS | `src/feature_engineering.py` | Seven deterministic features are implemented. | None |
| Business rationale for each feature | PASS | `reports/feature_engineering_report.md`; `notebooks/09_final_data_science_notebook.ipynb` | Each transformation has an explicit behavioral or value rationale. | None |
| Appropriate missing-value handling | PASS | `src/feature_engineering.py`; `src/preprocessing.py` | Safe ratios preserve undefined values for pipeline imputation. | None |
| Leakage checks | PASS | `tests/test_feature_engineering.py` | Target and identifier inputs are rejected; transformations do not use outcomes. | None |
| Integration into modeling pipeline | PASS | `src/modeling.py` | Feature engineering precedes preprocessing and classifiers inside one Pipeline. | None |
| **D. Modeling** |  |  |  |  |
| Logistic Regression implemented | PASS | `src/modeling.py`; `reports/modeling/phase5_model_comparison.csv` | Baseline and tuned results are saved. | None |
| Random Forest implemented | PASS | `src/modeling.py`; `reports/modeling/phase5_model_comparison.csv` | Baseline and tuned results are saved. | None |
| Histogram Gradient Boosting implemented | PASS | `src/modeling.py`; `reports/modeling/phase5_model_comparison.csv` | Baseline and tuned results are saved. | None |
| Stratified train/test split | PASS | `data/processed/02_split_indices.json` | Fixed 80/20 split uses random state 42 and target stratification. | None |
| Final test held out during development | PASS | `reports/model_development_report.md`; `models/final_model_metadata.json` | Selection and calibration decisions use training CV/OOF results only. | None |
| Cross-validation performed | PASS | `reports/modeling/phase5_fold_results.csv` | Five fixed stratified folds are recorded. | None |
| Hyperparameter tuning performed | PASS | `reports/modeling/phase5_tuning_candidates.csv` | Randomized searches cover 48 configurations and 240 fits. | None |
| Baseline and tuned models compared | PASS | `reports/modeling/phase5_model_comparison.csv` | Naive, baseline, class-weight, and tuned stages are comparable. | None |
| **E. Class Imbalance and Evaluation** |  |  |  |  |
| Class imbalance measured | PASS | `reports/eda_analysis.md` | Churn prevalence is explicitly reported. | None |
| Imbalance treatment justified | PASS | `reports/model_development_report.md`; `reports/modeling/phase5_class_weight_comparison.csv` | Balanced random-forest weights were selected from training evidence. | None |
| Precision reported | PASS | `reports/evaluation/phase6_final_test_metrics.json` | Locked-threshold precision is 0.989899. | None |
| Recall reported | PASS | `reports/evaluation/phase6_final_test_metrics.json` | Locked-threshold recall is 0.473430. | None |
| F1-score reported | PASS | `reports/evaluation/phase6_final_test_metrics.json` | Locked-threshold F1 is 0.640523. | None |
| ROC-AUC reported | PASS | `reports/evaluation/phase6_final_test_metrics.json` | Held-out ROC-AUC is 0.899429. | None |
| Average Precision / PR-AUC reported | PASS | `reports/evaluation/phase6_final_test_metrics.json` | Held-out Average Precision is 0.981852. | None |
| Confusion matrix generated | PASS | `reports/figures/phase6_confusion_matrix.png` | Counts are TN 32, FP 1, FN 109, TP 98. | None |
| Multiple thresholds analyzed | PASS | `reports/evaluation/phase6_threshold_grid.csv` | Precision, recall, F1, and contact rates are tabulated across thresholds. | None |
| Business operating threshold justified | PASS | `reports/evaluation/phase6_selected_threshold.json` | Threshold 0.970709 was selected on training OOF predictions under the capacity rule. | None |
| Accuracy limitations explained | PASS | `reports/model_evaluation_report.md`; `reports/04_business_report.pdf` | Low accuracy is contextualized as a consequence of the strict high-precision policy. | None |
| **F. Explainability** |  |  |  |  |
| Defensible explainability method | PASS | `src/explainability.py`; `reports/model_explainability_report.md` | Original-feature permutation importance and one-at-a-time sensitivity are documented. | None |
| Global feature importance | PASS | `reports/explainability/phase7_global_permutation_importance.csv` | Ten-repeat held-out permutation importance uses Average Precision. | None |
| At least three individual explanations | PASS | `reports/explainability/phase7_selected_customers.csv`; `reports/explainability/phase7_customer_A_sensitivity.csv` | Three anonymized cases cover TP, TN, and near-threshold FN behavior. | None |
| Association distinguished from causation | PASS | `reports/model_explainability_report.md`; final notebook | Explicit non-causal language is present. | None |
| **G. Retention Simulation** |  |  |  |  |
| Customers ranked by churn probability | PASS | `src/retention_simulation.py` | Descending scores use a stable, target-independent tie break. | None |
| Top-20% contact capacity enforced | PASS | `src/retention_simulation.py`; `tests/test_retention_simulation.py` | Exactly floor(240 × 0.20) = 48 are selected. | None |
| Actual churners captured calculated | PASS | `reports/retention_simulation_results.csv` | 48 historical churners are captured. | None |
| Random-selection baseline calculated | PASS | `reports/retention_simulation_results.csv` | Equal-budget expectation is 41.4 churners. | None |
| Precision@20% calculated | PASS | `reports/retention_simulation_results.csv` | Precision@20% is 1.0000. | None |
| Recall@20% calculated | PASS | `reports/retention_simulation_results.csv` | Recall/capture@20% is 0.231884. | None |
| Ranking lift calculated | PASS | `reports/retention_simulation_results.csv` | Lift is 1.15942. | None |
| Hypothetical financial simulation | PASS | `reports/retention_sensitivity_analysis.csv` | Base scenario and 27-scenario grid are saved. | None |
| Business assumptions documented | PASS | `reports/retention_simulation_report.md` | $5 contact, $150 retained value, and 10% effectiveness are labeled hypothetical. | None |
| **H. Final Analysis** |  |  |  |  |
| Final model-selection rationale | PASS | `reports/model_development_report.md`; `models/final_model_metadata.json` | Ranking quality, stability, calibration, and operational policy are documented. | None |
| Major observed churn-risk patterns | PASS | `reports/eda_analysis.md`; `reports/04_business_report.pdf` | Contract, usage, satisfaction, complaints, payments, and tenure patterns are summarized. | None |
| Three to five evidence-linked recommendations | PASS | `reports/04_business_report.pdf`; final notebook | Five actionable, evidence-linked recommendations are included. | None |
| Relevant business limitations | PASS | `reports/04_business_report.pdf`; `README.md` | Retrospective, non-causal, sample, drift, capacity, and financial caveats are explicit. | None |
| **Advanced Requirements** |  |  |  |  |
| Pipeline and ColumnTransformer | PASS | `src/modeling.py`; `src/preprocessing.py` | Complete pipeline encapsulates transformations and estimator. | None |
| Cross-validation model selection | PASS | `reports/modeling/phase5_model_comparison.csv` | Five-fold stratified results drive selection. | None |
| Documented hyperparameter search | PASS | `reports/model_development_report.md` | Search spaces, fit counts, scoring, and selected parameters are recorded. | None |
| Precision/recall across thresholds | PASS | `reports/evaluation/phase6_threshold_grid.csv`; final notebook | Threshold behavior is tabulated and interpreted. | None |
| Calibration analysis | PASS | `reports/evaluation/phase6_calibration_comparison.csv`; final notebook | Uncalibrated, sigmoid, and isotonic methods are compared on training OOF data. | None |
| At least one robustness check | PASS | `reports/evaluation/phase6_cv_robustness.csv`; final notebook | Five-fold means, dispersion, and ranges are displayed. | None |
| Reproducibility and random seeds | PASS | `data/processed/02_split_indices.json`; `README.md` | Random state 42 and fingerprint-bound fixed split are documented. | None |
| Package versions | PASS | `models/final_model_metadata.json`; `notebooks/09_final_data_science_notebook.ipynb` | Training and execution versions are recorded. | None |
| Execution instructions | PASS | `README.md` | Setup, notebook, tests, model, and report commands are documented. | None |

## Summary

- PASS: **72**
- FAIL: **0**
- PARTIAL: **0**
- NOT VERIFIED: **0**

Every critical assignment requirement has directly inspected or executed evidence. The
private data/model deliverables remain excluded from public Git and are available only in
the local authorized submission package.
