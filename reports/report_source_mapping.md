# Final Business Report Source Mapping

This table links each final-report section to the authoritative Phase 1–8 evidence used. The PDF generator reads the listed machine-readable outputs where available and validates the locked model metadata before generation.

| Report section | Supporting project file | Verified content used |
|---|---|---|
| Executive summary | `reports/model_development_report.md`; `reports/evaluation/phase6_final_test_metrics.json`; `reports/retention_simulation_results.csv` | Selected model, final ROC-AUC/AP, top-20% capture, lift, and scenario economics |
| Dataset overview | `reports/data_audit.md`; `reports/data_cleaning_report.md`; `data/processed/02_split_indices.json` | 1,200 × 20 dataset, class counts, missingness, duplicates, fixed split, fingerprint |
| Data preparation | `reports/data_cleaning_report.md`; `src/preprocessing.py` | Row preservation, train-fitted imputation/scaling/encoding, identifier and target exclusion |
| Exploratory analysis | `reports/eda_analysis.md`; `reports/figures/02_contract_churn.png`; `reports/figures/08_usage_churn.png` | Usage, contract, satisfaction, complaint, payment, tenure, and charge associations |
| Feature engineering | `reports/feature_engineering_report.md`; `src/feature_engineering.py` | Seven deterministic features, formulas, rationale, leakage and timing limitations |
| Model comparison | `reports/modeling/phase5_model_comparison.csv`; `reports/modeling/phase5_tuning_summary.json`; `reports/figures/m5_02_pr_auc_comparison.png` | Tuned Logistic Regression, Random Forest, and Histogram Gradient Boosting cross-validation metrics |
| Final evaluation | `reports/evaluation/phase6_final_test_metrics.json`; `reports/evaluation/phase6_calibration_comparison.csv`; `reports/figures/phase6_confusion_matrix.png` | Locked threshold, final metrics, confusion counts, calibration decision, contact-rate limitation |
| Global explainability | `reports/explainability/phase7_global_permutation_importance.csv`; `reports/figures/phase7_global_feature_importance.png` | Permutation method and measured original-feature importance |
| Customer examples | `reports/explainability/phase7_selected_customers.csv`; `reports/explainability/phase7_customer_*_sensitivity.csv` | Anonymized Customer A, B, and C outcomes, probabilities, classifications, and local sensitivity |
| Retention simulation | `reports/retention_simulation_results.csv`; `reports/figures/phase8_cumulative_gains.png` | 48-customer selection, precision@20%, recall@20%, lift, and equal-budget random baseline |
| Financial scenarios | `reports/retention_simulation_results.csv`; `reports/retention_sensitivity_analysis.csv`; `reports/figures/phase8_sensitivity_analysis.png` | Hypothetical cost/value/effectiveness inputs, model/random economics, sensitivity, and break-even rate |
| Recommendations and limitations | `reports/model_explainability_report.md`; `reports/retention_simulation_report.md` | Evidence-linked intervention hypotheses, operational workflow, uncertainty, fairness, privacy, and causal limitations |
| Integrity validation | `models/final_model_metadata.json`; `reports/explainability/phase7_integrity.json`; `reports/retention_simulation_integrity.json` | Artifact SHA-256, probability hash, split and dataset preservation, unchanged threshold and model |

## Interpretation Labels

- **Observed result:** a direct summary of dataset records, such as churn prevalence or missing-value counts.
- **Measured model result:** a cross-validation, held-out evaluation, explainability, or ranking statistic.
- **Hypothetical assumption or estimate:** an illustrative campaign cost, retained value, success rate, or resulting financial scenario.
- **Proposed action:** an intervention hypothesis requiring controlled causal evaluation.

