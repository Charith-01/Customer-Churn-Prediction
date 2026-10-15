"""Build and validate the private Phase 10 assessment deliverables.

The final notebook is generated from explicit cells so it can run from either
the repository root or the self-contained ``submission`` directory. Customer
identifiers are never rendered, and the locked model is loaded only after its
SHA-256 digest is checked against metadata.
"""

from __future__ import annotations

import hashlib
import shutil
import textwrap
import zipfile
from pathlib import Path

import nbformat


REQUIRED_SUBMISSION_FILES = (
    "01_data_science_notebook.ipynb",
    "02_cleaned_dataset.csv",
    "03_model_artifact.joblib",
    "04_business_report.pdf",
    "05_README.md",
)

SUPPORT_FILES = (
    "data/processed/02_split_indices.json",
    "models/final_model_metadata.json",
    "reports/modeling/phase5_model_comparison.csv",
    "reports/modeling/phase5_class_weight_comparison.csv",
    "reports/modeling/phase5_tuning_summary.json",
    "reports/evaluation/phase6_final_test_metrics.json",
    "reports/evaluation/phase6_threshold_key_points.csv",
    "reports/evaluation/phase6_calibration_comparison.csv",
    "reports/evaluation/phase6_cv_robustness.csv",
    "reports/explainability/phase7_global_permutation_importance.csv",
    "reports/explainability/phase7_selected_customers.csv",
    "reports/retention_simulation_results.csv",
    "reports/retention_sensitivity_analysis.csv",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def _code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def build_final_notebook(output_path: str | Path) -> Path:
    """Create the submission-facing end-to-end notebook."""
    cells = [
        _markdown(
            """
            # Customer Churn Prediction — Final Data Science Notebook

            ## 1. Introduction and objectives

            Customer churn is the loss of active subscribers and can reduce recurring
            revenue while increasing replacement costs. The business problem is to identify
            customers at elevated risk early enough to prioritize limited retention capacity.
            Machine learning is appropriate because risk can depend on nonlinear combinations
            of service, usage, payment, contract, and experience signals.

            This notebook integrates the validated outputs of Phases 1–8 into one
            reproducible, leakage-safe workflow. Its objectives are to audit the cleaned data,
            explain preprocessing and engineered features, compare candidate models, verify
            the checksum-locked final model on the untouched test split, explain its behavior,
            and translate score rankings into a capacity-constrained retention scenario.

            **Governance boundary:** this is retrospective decision support, not a causal
            impact study. Customer identifiers are excluded from predictors and outputs.
            """
        ),
        _markdown("## 2. Imports, configuration, and package versions"),
        _code(
            """
            from pathlib import Path
            import json, platform, sys

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").is_dir() and (PROJECT_ROOT.parent / "src").is_dir():
                PROJECT_ROOT = PROJECT_ROOT.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import sklearn

            from src.evaluation import compute_classification_metrics, positive_class_probabilities, predictions_at_threshold
            from src.explainability import load_trusted_locked_model
            from src.feature_engineering import CustomerFeatureEngineer
            from src.preprocessing import dataframe_fingerprint, separate_features_target_ids
            from src.retention_simulation import (
                build_internal_analysis_table, calculate_selection_metrics,
                cumulative_gains_data, exact_random_selection_expectations,
            )

            RANDOM_STATE = 42
            pd.set_option("display.max_columns", 30)
            sns.set_theme(style="whitegrid", context="notebook")
            plt.rcParams.update({"figure.figsize": (8, 4.5), "figure.dpi": 110})

            print({
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            })
            """
        ),
        _markdown("## 3. Data and artifact loading"),
        _code(
            """
            ROOT = PROJECT_ROOT

            def first_existing(*relative_paths):
                for relative in relative_paths:
                    candidate = ROOT / relative
                    if candidate.is_file():
                        return candidate
                raise FileNotFoundError(f"Required artifact not found; checked: {relative_paths}")

            DATA_PATH = first_existing("data/processed/02_cleaned_dataset.csv", "02_cleaned_dataset.csv")
            MODEL_PATH = first_existing("models/03_model_artifact.joblib", "03_model_artifact.joblib")
            SPLIT_PATH = first_existing("data/processed/02_split_indices.json", "support/results/02_split_indices.json")
            METADATA_PATH = first_existing("models/final_model_metadata.json", "support/results/final_model_metadata.json")

            data = pd.read_csv(DATA_PATH)
            split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            print(f"Loaded {data.shape[0]:,} rows and {data.shape[1]} columns.")
            print("Dataset fingerprint:", dataframe_fingerprint(data))
            """
        ),
        _markdown(
            """
            ## 4. Initial data audit

            The audit checks shape, types, missingness, duplicates, numeric ranges,
            outlier candidates, and class balance. Identifier values are intentionally
            never displayed.
            """
        ),
        _code(
            """
            public_data = data.drop(columns=["customer_id"], errors="ignore")
            audit = pd.DataFrame({
                "dtype": public_data.dtypes.astype(str),
                "missing_count": public_data.isna().sum(),
                "missing_percent": public_data.isna().mean().mul(100).round(2),
                "unique_values": public_data.nunique(dropna=True),
            })
            display(audit)
            print("Duplicate full rows:", int(data.duplicated().sum()))
            display(public_data.describe(include="all").transpose().round(3))

            numeric = public_data.select_dtypes(include="number").drop(columns=["churn"], errors="ignore")
            q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
            iqr = q3 - q1
            outlier_counts = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).sum()
            display(outlier_counts.rename("IQR outlier candidates").sort_values(ascending=False).to_frame())

            ax = public_data["churn"].value_counts().sort_index().plot.bar(color=["#42A5A5", "#8051D6"])
            ax.set(title="Observed churn distribution", xlabel="Churn class", ylabel="Customers")
            plt.tight_layout(); plt.show()
            """
        ),
        _markdown(
            """
            ## 5. Cleaning, preprocessing, and leakage-safe split

            Phase 2 standardized data types and validated domain rules without dropping
            rows. A fingerprint-bound, stratified 80/20 split (random state 42) was saved
            before modeling. Missing-value imputation, scaling, one-hot encoding, and all
            learned transformations live inside the model pipeline and are fitted only on
            training folds. `customer_id` and `churn` are explicitly excluded from inputs.
            """
        ),
        _code(
            """
            X, y, _identifiers = separate_features_target_ids(data)
            train_idx = np.asarray(split_payload["train_indices"], dtype=int)
            test_idx = np.asarray(split_payload["test_indices"], dtype=int)
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

            split_summary = pd.DataFrame({
                "rows": [len(X_train), len(X_test)],
                "churn_rate": [y_train.mean(), y_test.mean()],
                "predictor_columns": [X_train.shape[1], X_test.shape[1]],
            }, index=["training", "held-out test"])
            display(split_summary.round(4))
            assert not ({"customer_id", "churn"} & set(X.columns))
            assert not (set(train_idx) & set(test_idx))
            assert set(train_idx) | set(test_idx) == set(range(len(data)))
            print("Leakage checks passed; EDA below uses training rows only.")
            """
        ),
        _markdown(
            """
            ## 6. Exploratory data analysis

            The following nine panels use only the training partition and cover the main
            behavioral, service, financial, and demographic relationships assessed in
            the project. Associations are descriptive and should not be read as causal.
            """
        ),
        _code(
            """
            eda = X_train.copy(); eda["churn"] = y_train.to_numpy()
            fig, axes = plt.subplots(3, 3, figsize=(16, 13))
            sns.countplot(data=eda, x="contract_type", hue="churn", ax=axes[0,0], palette="Set2")
            sns.boxplot(data=eda, x="churn", y="tenure_months", ax=axes[0,1], color="#8EC5FC")
            sns.boxplot(data=eda, x="churn", y="monthly_charges", ax=axes[0,2], color="#D3A4F9")
            sns.boxplot(data=eda, x="churn", y="satisfaction_score", ax=axes[1,0], color="#F6C177")
            sns.boxplot(data=eda, x="churn", y="complaints_last_12m", ax=axes[1,1], color="#E98973")
            sns.boxplot(data=eda, x="churn", y="late_payments_last_12m", ax=axes[1,2], color="#9CCB86")
            sns.boxplot(data=eda, x="churn", y="monthly_usage_gb", ax=axes[2,0], color="#7FB3D5")
            sns.countplot(data=eda, y="payment_method", hue="churn", ax=axes[2,1], palette="Set2")
            sns.scatterplot(data=eda, x="tenure_months", y="monthly_charges", hue="churn",
                            alpha=.55, ax=axes[2,2], palette={0:"#42A5A5", 1:"#8051D6"})
            titles = ["Contract type", "Tenure", "Monthly charges", "Satisfaction",
                      "Complaints", "Late payments", "Usage", "Payment method",
                      "Tenure–charge interaction"]
            for ax, title in zip(axes.flat, titles): ax.set_title(title)
            fig.suptitle("Training-only exploratory analysis", fontsize=16, y=1.01)
            plt.tight_layout(); plt.show()
            """
        ),
        _markdown(
            """
            ## 7. Feature engineering

            Seven deterministic, interpretable features are created inside every fit/resample
            operation:

            - `tenure_years`: converts tenure into a business-friendly time scale.
            - `complaints_per_active_month`: normalizes complaints for customer exposure time.
            - `payment_risk_score`: combines late-payment frequency and payment friction.
            - `complaint_satisfaction_interaction`: captures compounding experience risk.
            - `monthly_contract_tenure`: represents recurring value accumulated over tenure.
            - `late_payment_charge_interaction`: represents financial exposure with lateness.
            - `usage_per_charge`: approximates perceived usage value per monetary unit.

            All use predictor-time information only; neither the target nor identifier is used.
            """
        ),
        _code(
            """
            engineer = CustomerFeatureEngineer()
            engineered_train = engineer.fit_transform(X_train)
            engineered_names = [name for name in engineered_train.columns if name not in X_train.columns]
            display(engineered_train[engineered_names].describe().transpose().round(3))
            print("Engineered features:", ", ".join(engineered_names))
            """
        ),
        _markdown(
            """
            ## 8. Model development and hyperparameter tuning

            Phase 5 compared a naive prior baseline, logistic regression, random forest,
            and histogram gradient boosting with five-fold stratified cross-validation.
            Average precision was primary because churn is prevalent and the business use
            case ranks customers. Tuning and model selection used training data only; this
            final notebook reuses the saved results instead of rerunning expensive searches.
            """
        ),
        _code(
            """
            comparison_path = first_existing("reports/modeling/phase5_model_comparison.csv", "support/results/phase5_model_comparison.csv")
            comparison = pd.read_csv(comparison_path)
            display(comparison.loc[comparison["stage"].isin(["Naive reference", "Tuned"]),
                                   ["model", "stage", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_precision_at_20"]]
                    .sort_values("mean_pr_auc", ascending=False).round(4))
            print("Selected model: tuned random forest with class_weight='balanced'.")
            """
        ),
        _markdown("## 9. Class imbalance, calibration, and operating policy"),
        _code(
            """
            weight_path = first_existing("reports/modeling/phase5_class_weight_comparison.csv", "support/results/phase5_class_weight_comparison.csv")
            calibration_path = first_existing("reports/evaluation/phase6_calibration_comparison.csv", "support/results/phase6_calibration_comparison.csv")
            display(pd.read_csv(weight_path)[["model", "class_weight", "mean_pr_auc", "mean_roc_auc", "mean_precision_at_20"]].round(4))
            display(pd.read_csv(calibration_path)[["method", "brier_score", "log_loss", "average_precision", "expected_calibration_error"]].round(4))
            print(f"Locked calibration: {metadata['calibration']['method']}; operating threshold: {metadata['operating_threshold']:.12f}")
            print("The threshold was selected from training-only out-of-fold probabilities, before the test set was evaluated.")

            threshold_path = first_existing("reports/evaluation/phase6_threshold_key_points.csv", "support/results/phase6_threshold_key_points.csv")
            display(pd.read_csv(threshold_path).round(4))
            """
        ),
        _markdown("## 10. Locked final-model verification and evaluation"),
        _code(
            """
            locked_model = load_trusted_locked_model(
                MODEL_PATH, expected_sha256=metadata["model_artifact_sha256"], trusted=True
            )
            probabilities = positive_class_probabilities(locked_model, X_test)
            threshold = float(metadata["operating_threshold"])
            predictions = predictions_at_threshold(probabilities, threshold)
            measured = compute_classification_metrics(y_test, probabilities, threshold=threshold)
            display(pd.Series(measured, name="held-out value").to_frame().round(6))

            stored_path = first_existing("reports/evaluation/phase6_final_test_metrics.json", "support/results/phase6_final_test_metrics.json")
            stored = json.loads(stored_path.read_text(encoding="utf-8"))["selected_business_threshold"]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score", "log_loss"):
                assert np.isclose(measured[key], stored[key], atol=1e-12)
            print("Locked held-out metrics reproduce exactly within numerical tolerance.")

            robustness_path = first_existing("reports/evaluation/phase6_cv_robustness.csv", "support/results/phase6_cv_robustness.csv")
            print("Training-CV robustness summary:")
            display(pd.read_csv(robustness_path).round(4))
            """
        ),
        _code(
            """
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Purples", ax=axes[0], colorbar=False)
            RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1], curve_kwargs={"color": "#8051D6"})
            PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[2], curve_kwargs={"color": "#3569D4"})
            axes[0].set_title("Confusion matrix"); axes[1].set_title("ROC curve"); axes[2].set_title("Precision–recall curve")
            plt.tight_layout(); plt.show()

            observed, predicted = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
            plt.plot([0,1], [0,1], "--", color="#3569D4", label="ideal calibration")
            plt.plot(predicted, observed, marker="o", color="#8051D6", label="locked model")
            plt.xlabel("Mean predicted probability"); plt.ylabel("Observed churn rate")
            plt.title("Held-out calibration diagnostic"); plt.legend(); plt.tight_layout(); plt.show()
            """
        ),
        _markdown(
            """
            ## 11. Model explainability and customer-level risk analysis

            Permutation importance measures held-out predictive dependence—not causality.
            Local cases are anonymized. Probability sensitivity helps explain model behavior
            but should not be interpreted as an intervention effect.
            """
        ),
        _code(
            """
            importance_path = first_existing("reports/explainability/phase7_global_permutation_importance.csv", "support/results/phase7_global_permutation_importance.csv")
            cases_path = first_existing("reports/explainability/phase7_selected_customers.csv", "support/results/phase7_selected_customers.csv")
            importance = pd.read_csv(importance_path).head(10).sort_values("importance_mean")
            importance.plot.barh(x="feature", y="importance_mean", xerr="importance_std", legend=False, color="#8051D6")
            plt.xlabel("Decrease in average precision"); plt.title("Global permutation importance")
            plt.tight_layout(); plt.show()
            cases = pd.read_csv(cases_path)
            safe_columns = [c for c in cases.columns if c not in {"customer_id", "internal_row_reference"}]
            display(cases[safe_columns].round(4))
            """
        ),
        _markdown(
            """
            ## 12. Retention intervention simulation and business impact

            A strict top-20% capacity policy selects exactly 48 of 240 held-out customers
            by score alone. Outcomes are attached only after ranking for retrospective
            measurement. Financial values below are scenarios, not realized savings.
            """
        ),
        _code(
            """
            analysis = build_internal_analysis_table(probabilities, y_test, predictions, contact_fraction=.20)
            selection = calculate_selection_metrics(analysis["actual_churn"], analysis["selected_top_20pct"])
            random_expected = exact_random_selection_expectations(
                selection["total_customers"], selection["selected_customers"], selection["total_actual_churners"]
            )
            display(pd.DataFrame({
                "model ranked": [selection["selected_churners"], selection["precision_at_20"], selection["recall_at_20"], selection["lift_at_20"]],
                "random expectation": [random_expected["expected_churners_captured"], random_expected["expected_random_precision"], random_expected["expected_random_recall"], random_expected["expected_random_lift"]],
            }, index=["churners captured", "precision", "recall", "lift"]).round(4))

            gains = cumulative_gains_data(y_test, probabilities)
            plt.plot(gains["selected_fraction"], gains["model_capture_rate"], color="#8051D6", linewidth=2.5, label="model ranked")
            plt.plot([0,1], [0,1], "--", color="#3569D4", label="random selection")
            plt.axvline(.20, color="#555555", linestyle=":", label="20% capacity")
            plt.xlabel("Population contacted"); plt.ylabel("Cumulative churners captured")
            plt.title("Cumulative gains on locked held-out data"); plt.legend(); plt.tight_layout(); plt.show()

            retained_model = selection["selected_churners"] * .10
            retained_random = random_expected["expected_churners_captured"] * .10
            economics = pd.DataFrame({
                "model ranked": [retained_model, retained_model * 150 - 48 * 5],
                "random expectation": [retained_random, retained_random * 150 - 48 * 5],
            }, index=["expected retained customers", "expected net benefit ($)"])
            display(economics.round(2))

            sensitivity_path = first_existing("reports/retention_sensitivity_analysis.csv", "support/results/retention_sensitivity_analysis.csv")
            sensitivity = pd.read_csv(sensitivity_path)
            print("Sensitivity range across 27 hypothetical cost, value, and effectiveness scenarios:")
            display(sensitivity[["incremental_expected_net_benefit", "model_expected_net_benefit", "random_expected_net_benefit"]]
                    .agg(["min", "median", "max"]).round(2))
            """
        ),
        _markdown(
            """
            ## 13. Business recommendations

            The main evidence is strong ranking performance, usage-dominant model dependence,
            service/experience associations, and a retrospective top-20% lift of about 1.16.

            1. Pilot the score-ranked top-20% contact policy with a randomized holdout group.
            2. Prioritize service/usage diagnosis and customer experience rather than treating
               model importance as proof that a specific action will work.
            3. Track contact cost, realized retention, incremental value, capacity, calibration,
               and subgroup outcomes before scaling.
            4. Revalidate data schema, missingness, score drift, and performance on a regular cadence.
            5. Require human review, lawful use, access controls, and an audit trail for outreach.
            """
        ),
        _markdown(
            """
            ## 14. Conclusion

            The locked calibrated random-forest pipeline is reproducible and strong at ranking
            churn risk (held-out ROC-AUC ≈ 0.899; average precision ≈ 0.982). Its high operating
            threshold favors precision (≈ 0.990) over recall (≈ 0.473). Under the separate strict
            top-20% campaign rule, all 48 contacted held-out customers were churners, compared with
            41.4 expected under random selection. These are retrospective results. A controlled
            pilot is required to estimate causal retention lift and realized financial return.
            Future work should add temporal/out-of-time validation, drift monitoring, subgroup
            review, and experiment-based uplift or treatment-effect modeling.
            """
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return destination


def sanitize_legacy_notebook_outputs(root: str | Path) -> list[str]:
    """Remove only output cells known to contain raw identifier values."""
    project_root = Path(root)
    targets = {
        "notebooks/01_data_audit.ipynb": (7, 25),
        "notebooks/02_data_cleaning_preprocessing.ipynb": (5,),
    }
    changed: list[str] = []
    for relative, indexes in targets.items():
        path = project_root / relative
        notebook = nbformat.read(path, as_version=4)
        for index in indexes:
            cell = notebook.cells[index]
            cell["outputs"] = []
            cell["execution_count"] = None
        nbformat.write(notebook, path)
        changed.append(relative)
    return changed


def assemble_submission(root: str | Path) -> Path:
    """Create a private, self-contained assessment directory."""
    project_root = Path(root)
    destination = project_root / "submission"
    destination.mkdir(parents=True, exist_ok=True)

    root_files = {
        "01_data_science_notebook.ipynb": project_root / "notebooks" / "09_final_data_science_notebook.ipynb",
        "02_cleaned_dataset.csv": project_root / "data" / "processed" / "02_cleaned_dataset.csv",
        "03_model_artifact.joblib": project_root / "models" / "03_model_artifact.joblib",
        "04_business_report.pdf": project_root / "reports" / "04_business_report.pdf",
        "05_README.md": project_root / "README.md",
        "requirements.txt": project_root / "requirements.txt",
    }
    for name, source in root_files.items():
        shutil.copy2(source, destination / name)

    shutil.copytree(project_root / "src", destination / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    results = destination / "support" / "results"
    results.mkdir(parents=True, exist_ok=True)
    for relative in SUPPORT_FILES:
        source = project_root / relative
        shutil.copy2(source, results / source.name)
    return destination


def create_submission_zip(root: str | Path) -> Path:
    """Zip the private package with required deliverables at archive root."""
    project_root = Path(root)
    submission = project_root / "submission"
    archive = project_root / "Customer_Churn_Prediction_Final_Submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(submission.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(submission).as_posix())
    return archive


def validate_submission(root: str | Path) -> dict[str, object]:
    """Validate package structure, privacy exclusions, and artifact identity."""
    project_root = Path(root)
    submission = project_root / "submission"
    archive = project_root / "Customer_Churn_Prediction_Final_Submission.zip"
    missing = [name for name in REQUIRED_SUBMISSION_FILES if not (submission / name).is_file()]
    forbidden_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    forbidden_suffixes = {".env", ".key", ".pem", ".pyc"}
    unsafe = []
    for path in submission.rglob("*"):
        relative = path.relative_to(submission)
        if any(part in forbidden_parts for part in relative.parts) or path.suffix.lower() in forbidden_suffixes:
            unsafe.append(relative.as_posix())
    if missing or unsafe or not archive.is_file():
        raise ValueError(f"Submission validation failed: missing={missing}, unsafe={unsafe}, zip={archive.is_file()}")
    if sha256_file(project_root / "data" / "processed" / "02_cleaned_dataset.csv") != sha256_file(submission / "02_cleaned_dataset.csv"):
        raise ValueError("Packaged dataset is not checksum-identical to the root copy.")
    if sha256_file(project_root / "models" / "03_model_artifact.joblib") != sha256_file(submission / "03_model_artifact.joblib"):
        raise ValueError("Packaged model is not checksum-identical to the root copy.")
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        names = set(handle.namelist())
    if bad_member or not set(REQUIRED_SUBMISSION_FILES).issubset(names):
        raise ValueError(f"ZIP validation failed: bad_member={bad_member}")
    return {
        "required_files": len(REQUIRED_SUBMISSION_FILES),
        "package_files": sum(path.is_file() for path in submission.rglob("*")),
        "zip_bytes": archive.stat().st_size,
        "zip_sha256": sha256_file(archive),
        "unsafe_files": unsafe,
    }
