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
|   `-- 01_data_audit.ipynb          # Phase 1 audit
|-- src/
|   `-- __init__.py
|-- models/                          # Saved model artifacts
|-- reports/
|   |-- figures/
|   `-- data_audit.md                # Phase 1 findings
|-- tests/
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

Raw customer data is intentionally excluded from Git. Do not rename, edit, or commit the source CSV.

## Run the Phase 1 notebook

Start JupyterLab from the repository root:

```powershell
jupyter lab
```

Open `notebooks/01_data_audit.ipynb` and select **Run All Cells**. The notebook also works when launched from the `notebooks` directory because it resolves the project path safely.

For a command-line execution check:

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_data_audit.ipynb --output 01_data_audit.ipynb --output-dir notebooks
```

## Project status

**Phase 1 complete:** repository setup and initial data audit. No cleaning, feature engineering, model training, or advanced EDA has been performed. Phase 2 should begin only after the audit findings and proposed cleaning decisions are approved.
