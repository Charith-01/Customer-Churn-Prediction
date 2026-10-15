# Final Quality Assurance Report

**Project:** Customer Churn Prediction and Retention Analysis  
**Validation date:** 2026-10-15  
**Gate result:** **SUBMISSION READY**

## 1. Project overview

The repository implements an end-to-end, leakage-safe churn-risk workflow: audit and
cleaning, training-only EDA, deterministic feature engineering, cross-validated model
selection and tuning, calibration and threshold selection, one-time held-out evaluation,
model explainability, capacity-constrained retention simulation, and stakeholder reporting.
The final model and evaluation results were preserved during Phase 10.

## 2. Final repository structure

The repository retains the eight phase notebooks, seven original reusable modules, all
reports/figures, and 118 automated tests. Phase 10 adds:

- `notebooks/09_final_data_science_notebook.ipynb`
- `src/final_integration.py`
- `reports/final_requirements_checklist.md`
- `reports/final_quality_assurance.md`

Canonical customer data, model, notebook, and report artifacts are organized under
`data/processed/`, `models/`, `notebooks/`, and `reports/`, respectively.

## 3. Assignment compliance summary

The itemized checklist records 72 PASS, 0 FAIL, 0 PARTIAL, and 0 NOT VERIFIED results.
All final-gate items are satisfied. Evidence is mapped in
`reports/final_requirements_checklist.md`.

## 4. Dataset validation results

- Shape: 1,200 rows and 20 columns.
- Identifier: `customer_id`, preserved for protected traceability and excluded from predictors.
- Target: binary `churn`.
- Split: 960 training and 240 held-out test rows; stratified; random state 42; disjoint and exhaustive.
- Cleaning: no record removal and no silent value correction; genuine missing values remain for pipeline imputation.
- No prediction, evaluation-label, or learned full-dataset columns were added.
- Authoritative, root-copy, and source CSV SHA-256:
  `b9f3981f7d32ba3bfd2256f69b6e12f730be79139b22dba6ced563f53db9462c`.

The raw and cleaned digests match because the audited source required no value correction;
the cleaning phase primarily validated schema/types and established a controlled artifact.

## 5. Notebook execution status

`notebooks/09_final_data_science_notebook.ipynb` contains all 14 required sections and 12 executable code
cells. Clean-kernel execution succeeded from both supported working directories:

- Repository root: 14.52 seconds.
- Private `submission/` directory: 9.90 seconds.
- Failed cells: 0; unexecuted code cells: 0; output error objects: 0.
- Figures rendered: EDA, confusion matrix, ROC, precision–recall, calibration, importance,
  and cumulative gains.
- Numerical consistency assertions passed against saved Phase 6 metrics.
- Python: 3.13.3.

Execution packages: NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, Matplotlib 3.11.2,
seaborn 0.13.2, joblib 1.6.0, nbformat 5.11.1, and nbconvert 7.17.2.

The managed Windows sandbox required a project-local Jupyter runtime directory and
`JUPYTER_ALLOW_INSECURE_WRITES=true` because it denies ACL changes in user runtime paths.
This produced environment-specific permission/TCP warnings but no cell warnings or errors;
the temporary runtime directory was removed. Normal local execution uses the documented
command without that sandbox workaround.

## 6. Model artifact validation

- Trusted local artifact loaded successfully and exposes `predict_proba` and fitted classes.
- Pipeline includes deterministic feature engineering, ColumnTransformer preprocessing,
  tuned Random Forest, balanced class weighting, and sigmoid calibration.
- Model SHA-256 (canonical and root copy):
  `b3635a3e65e850bca895a8c7506de6bcd3add2953a3829759544eddede0ed496`.
- Operating threshold is present in metadata: 0.9707090004321576.
- The artifact accepted held-out raw-schema predictors and generated 240 valid probabilities.
- Notebook metric assertions and serialization tests verified reproducible predictions.

No retraining, retuning, recalibration, or model replacement occurred.

## 7. Model evaluation consistency

Recomputed held-out results match `phase6_final_test_metrics.json` within 1e-12:

| Metric | Value |
|---|---:|
| Accuracy | 0.541667 |
| Precision | 0.989899 |
| Recall | 0.473430 |
| F1 | 0.640523 |
| ROC-AUC | 0.899429 |
| Average Precision | 0.981852 |
| Brier score | 0.082468 |
| Log loss | 0.262660 |

Confusion counts remain TN 32, FP 1, FN 109, and TP 98. Evaluation JSON SHA-256 remains
`56977290d8dac7114efb35df57dbcc71df193a7f35703a258e757b1dd1297b3a`.

## 8. PDF validation

- File opens and contains seven nonblank portrait pages.
- Required sections, metrics, interpretations, three anonymized customer cases, and
  hypothetical-finance labels are present.
- Customer identifiers and unresolved placeholders are absent.
- Page 6 was rendered and visually inspected after correction.
- Figure 7 now states that the dashed blue diagonal is random selection and the vertical
  dotted line is the 20% contact limit; chart data were unchanged.
- Current PDF SHA-256:
  `fad4a1778ba18d325864f9981a7a422c1757400eb44064ce66fd5016697157d8`.

## 9. Retention simulation consistency

The unchanged held-out probabilities are ranked descending with a stable, outcome-free tie
break. A strict 20% cap selects 48/240 customers. All 48 historically churned, yielding
precision@20% 1.0000, recall@20% 0.231884, and lift 1.15942. Uniform random selection at
the same budget captures 41.4 churners in expectation.

The base hypothetical assumptions are $5/contact, $150/value per successfully retained
customer, and 10% intervention effectiveness. Expected net benefit is $480 model-ranked
versus $381 random, or $99 incremental. These are scenario estimates, not measured causal
effects or realized profit. The 27-scenario sensitivity grid remains unchanged.

## 10. Unit test results

Command: `python -m pytest tests -q -p no:cacheprovider --basetemp=tests\\.tmp`

- Collected: 118
- Passed: 118
- Failed: 0
- Errors: 0
- Skipped: 0
- Test warnings: 0 in the successful run
- Duration: 14.16 seconds

An initial restricted-sandbox run could not create system temp files or Windows worker
pipes and therefore was not treated as a project test result. The same suite passed when
run with the approved workspace temp configuration.

## 11. Reproducibility checks

- No machine-specific paths occur in executable project code or the final notebook.
- Notebook execution does not depend on prior interactive state.
- Random state 42 and fixed split metadata are recorded.
- Split metadata are bound to the exact order-sensitive dataset fingerprint.
- Imputation, encoding, scaling, feature engineering, calibration, and classification are
  fit inside training-safe pipelines.
- Saved development/evaluation results are explicitly loaded rather than presented as newly tuned.
- Final report figures map to source analyses and validated aggregate outputs.
- The private package includes all source modules and aggregate support files used by the notebook.

The public repository alone cannot reproduce customer-level results without the authorized
private dataset and serialized model. This privacy constraint is explicit. The private
assessment package is self-contained for the documented notebook run.

## 12. Dependency checks

`python -m pip check` reports **No broken requirements found**. Requirements declare data,
visualization, scikit-learn, Jupyter/nbconvert/nbformat, pytest, and PDF dependencies. No
external service or network API is required. The installed PDF packages were ReportLab
5.0.1, PyMuPDF 1.28.2, and Pillow 12.3.0. The scikit-learn version matches the locked-model
metadata.

## 13. Data privacy checks

- Raw/processed CSVs, joblib files, private package, and final ZIP are Git-ignored.
- Outputs from three legacy notebook cells that exposed raw customer IDs were cleared.
- All nine tracked notebooks were scanned against all source customer IDs: zero output matches.
- Final report privacy validation found no customer identifiers.
- Package validation excludes Git internals, environments, caches, keys, PEM files, `.env`
  files, and protected internal ranking data.
- No privacy exclusions were weakened.

## 14. Git repository status

The worktree intentionally contains Phase 10 changes and no commit or push was made.
Tracked changes include the corrected report/caption and Page 6 preview, privacy-sanitized
legacy notebook outputs, README/requirements updates, and Phase 10 source/docs. Private
customer data, serialized models, and `submission/` do not appear in normal Git status
because explicit ignore rules protect them. `git diff --check` reports no whitespace errors.

## 15. Final submission package validation

- Directory: `submission/`
- Archive validated as `Customer_Churn_Prediction_Final_Submission.zip`; the user later
  moved the archive to a personal location and removed the redundant root copy.
- ZIP size: 3,781,943 bytes.
- ZIP SHA-256: `ea684405fbd45be3468a028ef857009f31fc46201c99e96d95e2ebf35655b437`.
- Archive integrity: `ZipFile.testzip()` returned no bad member.
- Files: 28, including all five numbered deliverables, `requirements.txt`, `src/`, and
  13 aggregate/fingerprint support results.
- Unsafe files found: 0.
- Submitted notebook executed successfully in-place before the final ZIP was created.

## 16. Known limitations

- Customer data and the joblib model are private artifacts; public Git users need authorized
  copies to reproduce final results.
- The dataset is small and has unusually high churn prevalence; external and temporal
  generalization are not established.
- The fixed probability threshold does not enforce campaign capacity under score drift;
  the separate top-k rule must be used for strict capacity.
- Explanations are model associations, not causal intervention effects.
- Financial results depend on hypothetical costs, values, and intervention effectiveness.
- A randomized pilot is required to estimate incremental retention impact.

## 17. Remaining actions before submission

No critical technical, documentation, privacy, or packaging issue remains. The user should:

1. Review the tracked diff and visually open the organized notebook and report.
2. Transfer the externally archived private ZIP only through the authorized assessment channel.
3. Commit only the reviewed tracked files; do not force-add ignored data/model/package files.
4. Push only after explicit approval and verification of the intended remote/branch.
