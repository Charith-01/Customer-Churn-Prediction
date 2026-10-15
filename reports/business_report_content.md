# Customer Churn Prediction and Retention Analysis

*End-to-End Machine Learning and Business Retention Insights*

## Executive Summary

This project examined whether customer information can support earlier, capacity-aware retention decisions. The analysis covered 1,200 subscription customers and followed a controlled workflow from data audit through cleaning, exploratory analysis, feature engineering, model development, final evaluation, explainability, and a retrospective top-20% retention simulation.

Logistic Regression, Random Forest, and Histogram Gradient Boosting were compared using leakage-safe five-fold stratified cross-validation. The tuned Random Forest achieved the strongest training-validation Average Precision at 0.9782 and was retained. Sigmoid calibration was selected using training-only evidence, and the final model achieved held-out ROC-AUC 0.8994 and Average Precision 0.9819.

At the locked probability threshold, precision was 0.9899 and recall was 0.4734. For a campaign with a hard 20% contact limit, deterministic ranking selected 48 of 240 held-out customers; all 48 historically churned. This produced precision@20% 1.0000, recall@20% 0.2319, and lift 1.1594 versus random selection. The result supports risk prioritization, but it does not show that contacting a customer prevents churn.

The principal business takeaway is to use the model as a prioritization aid rather than an automated retention decision. Current customers should be scored, ranked, and filtered through a fixed contact capacity, eligibility rules, privacy controls, and a controlled intervention pilot.

## Business Problem and Objectives

Customer churn reduces recurring value and creates replacement costs, while retention teams have finite capacity. The project objectives were to build a reproducible churn-risk model, identify the strongest measured risk signals, explain individual predictions, and assess whether ranking can concentrate historical churners within a 20% contact budget. The project deliberately separates prediction from intervention effectiveness.

## Dataset Overview

The source contains 1,200 customers and 20 columns: one identifier, one binary churn target, and 18 predictors. Churn is unusually prevalent: 1,036 customers (86.33%) have label 1 and 164 (13.67%) have label 0. The high prevalence means accuracy alone is not an adequate measure of usefulness.

Four numerical fields contain missing values: satisfaction score 48, monthly usage 36, monthly charges 30, and total charges 24. In total, 138 cells across 132 customers are missing. There are no complete duplicate rows and no duplicate customer identifiers. Statistical outliers and unusual service combinations were retained because the available evidence did not show that they were erroneous.

## Data Preparation and Leakage Prevention

Cleaning standardized representation only; it changed no customer values and excluded no rows. Missing numerical values were preserved in the cleaned file and median-imputed inside training-fitted pipelines. Categorical values use an explicit missing category and one-hot encoding with safe handling for unseen levels. Numerical predictors are scaled after imputation.

The data was split once into 960 training rows and 240 final-test rows using an 80/20 stratified split with random state 42. Customer ID and churn were explicitly excluded from predictors. Feature engineering, imputation, scaling, encoding, calibration, and model fitting were all contained inside leakage-safe scikit-learn pipelines. The held-out test partition remained protected until final evaluation.

## Exploratory Churn Patterns

Training-only EDA found the clearest separation in monthly usage. Median usage was 70.75 GB among churned customers and 28.00 GB among retained customers. Churn rose from 57.45% in the lowest usage quintile to approximately 99.5% in the upper two quintiles, a nonlinear relationship that requires confirmation that usage is measured before the prediction cutoff.

Month-to-month customers had 91.10% observed churn, compared with 80.00% for one-year and 79.08% for two-year contracts. Satisfaction was lower among churned customers, while two or more late payments and three or more complaints coincided with elevated churn. Tenure and monthly charges showed non-monotonic patterns, and payment method provided relatively weak separation. These are predictive associations, not causal findings.

## Feature Engineering

Seven deterministic features were added without using target values: tenure in years, complaints per active month, a payment-friction score, complaint-satisfaction interaction, month-to-month contract by tenure, late-payment by monthly-charge interaction, and usage per charge. These features express duration, intensity, and combined customer-friction signals. They preserve missingness for training-fold imputation and do not establish that any representation improved outcomes by itself.

## Model Development

Logistic Regression, Random Forest, and Histogram Gradient Boosting were evaluated alongside a naive prior reference. All models used the same five stratified folds and complete feature/preprocessing pipeline. Sixteen hyperparameter configurations were evaluated for each main model using Average Precision as the primary selection metric.

The tuned Random Forest produced mean cross-validation Average Precision 0.9782 +/- 0.0060 and ROC-AUC 0.8822. Tuned Logistic Regression achieved Average Precision 0.9752, while tuned Histogram Gradient Boosting achieved 0.9741. These differences are small: Random Forest exceeded Logistic Regression by 0.0030 and Histogram Gradient Boosting by 0.0041 in mean Average Precision.

Random Forest was selected before final-test access because it combined the highest mean Average Precision and ROC-AUC, joint-best precision@20% of 0.9949, and relatively low fold-to-fold Average Precision variation. Its 400-tree ensemble is more complex than Logistic Regression, so the simpler model remains a reasonable fallback where transparency or operating cost outweighs the modest ranking gain.

## Final Evaluation and Calibration

Sigmoid calibration was selected through nested training-only cross-validation because it improved Brier score from 0.1061 to 0.0840, log loss from 0.3328 to 0.2752, and expected calibration error from 0.1181 to 0.0219 without material ranking degradation. Isotonic calibration slightly improved expected calibration error but reduced ranking metrics and carried greater overfitting risk on the limited sample. Calibration improves probability reliability; it does not make probabilities perfectly calibrated.

On the 240-row held-out test set, the final model achieved accuracy 0.5417, precision 0.9899, recall 0.4734, F1 0.6405, ROC-AUC 0.8994, and Average Precision 0.9819. The confusion matrix contained 98 true positives, 32 true negatives, one false positive, and 109 false negatives.

Accuracy of 0.5417 is not strong and is below the 0.8625 held-out majority-class baseline. This occurs because the deliberately high locked threshold predicts churn for only 99 customers: it produces just one false positive and therefore precision of 0.9899, but it misses 109 of 207 churners, limiting recall to 0.4734. High precision answers whether flagged customers churned; it does not imply that most churners were found.

ROC-AUC 0.8994 measures discrimination across thresholds, while Average Precision 0.9819 summarizes positive-class ranking and must be judged against the already-high 0.8625 churn prevalence. Both support risk ranking even though the locked binary rule has weak accuracy and incomplete recall. The final ROC-AUC and Average Precision remain inside the Phase 5 fold ranges; held-out Brier score 0.0825 and log loss 0.2627 are also consistent with nested training estimates. Uncertainty remains because the test set contains only 33 retained customers.

The threshold and campaign rules serve different purposes. The locked classification threshold of 0.970709 was preserved and selected 99 of 240 held-out customers, or 41.25%. The operational policy instead ranks every probability from highest to lowest and selects exactly the top 48 customers, with original held-out order as the deterministic tie-breaker. No threshold was retuned from test outcomes.

## Global Explainability

Original-feature permutation importance measured the decrease in held-out Average Precision after each input was shuffled. Monthly usage was the dominant feature with mean importance 0.1072 +/- 0.0164. Contract type ranked second at 0.0044, followed by satisfaction score at 0.0036 and late payments at 0.0017. The remaining features were close to zero relative to repeated-permutation variability.

Observed predictions rose sharply with usage and generally fell as satisfaction increased. Month-to-month customers had higher average fitted risk than fixed-term customers. Importance and grouped predictions describe model behavior and association; they do not prove that changing a feature will change churn.

## Anonymized Customer Explanations

Local explanations used one-feature-at-a-time probability sensitivity against training-derived reference values. Customer A was a correct churn prediction at probability 0.9783; usage of 75.8 GB was the dominant risk-raising local signal. Customer B was a correct non-churn prediction at 0.6423; a one-year contract and usage of 35.4 GB strongly lowered fitted risk relative to their references, despite opposing age and payment signals.

Customer C was a false negative at probability 0.9706, only 0.000073 below the locked threshold. Usage of 76.3 GB raised fitted risk, while the one-year contract and charge effect lowered it slightly. The case illustrates the fragility of binary decisions near a high threshold. Local effects are non-additive and non-causal.

## Top-20% Retention Simulation

The campaign policy is separate from threshold-based classification. It ranks the unchanged held-out probabilities descending and selects exactly floor(N x 0.20) customers. Original held-out order is the predeclared secondary key for ties and does not use customer identity or actual churn. For 240 evaluated customers, 48 were selected; the locked threshold itself would have selected 99 and therefore cannot enforce the contact limit.

All 48 selected customers historically churned, giving precision@20% 1.0000 and recall/capture@20% 0.2319. With overall churn prevalence 0.8625, lift was 1.1594. Uniform random selection with the same contact count would capture 41.4 churners in expectation, so model ranking concentrated 6.6 additional historical churners. "Captured" means present in the retrospective group, not successfully retained.

## Hypothetical Business Impact

The financial illustration assumes a $5 contact cost, $150 value per successfully retained would-be churner, and 10% intervention effectiveness. These are hypothetical USD accounting inputs, not company figures or measured treatment effects. Under these assumptions, model ranking costs $240, yields 4.8 expected retained customers, $720 gross value, $480 net benefit, and 200% scenario ROI.

Random selection under the same budget produces $381 expected net benefit and 158.75% scenario ROI. The incremental expected model benefit is $99. At the base cost and value, model net benefit is $120, $480, and $1,200 at assumed success rates of 5%, 10%, and 20%. The model strategy is positive in 23 of 27 tested cost-value-effectiveness scenarios, with a base break-even success rate of 3.33%.

## Evidence-Linked Retention Recommendations

First, use service-quality and plan-fit reviews for eligible high-risk, high-usage customers because usage is the strongest model signal. Second, prioritize proactive support and complaint resolution for eligible high-risk customers with low satisfaction or unresolved issues. Third, offer compliant, voluntary payment-support outreach where recent late payments indicate possible friction. Fourth, provide contract and renewal assistance to eligible month-to-month customers when a suitable plan exists.

These actions are intervention hypotheses. They can create agent cost, discount leakage, privacy concerns, or a poor customer experience. A randomized holdout is necessary to estimate incremental retention uplift and margin before broader rollout.

## Implementation Roadmap

The practical sequence is: validate current and consented input data; generate probabilities using the saved model; rank customers by risk; apply the top-20% limit; review eligibility, privacy, and fairness constraints; assign approved interventions; run a controlled pilot and record treatment and outcome data; then monitor data quality, drift, calibration, capture, costs, and subgroup outcomes.

## Limitations

The dataset is modest and has unusually high churn prevalence. Missing values and unresolved service/usage relationships remain. Predictor timing is not documented well enough to eliminate all temporal-leakage risk. The final evaluation uses one held-out sample rather than external or temporal validation, and only 33 retained customers appear in the test set. Calibration and the very high threshold may shift in a future population.

Model importance and local sensitivity are associative rather than causal. No treatment-response data or verified company cost and value inputs exist, so the financial analysis is hypothetical. Customer scoring also requires purpose limitation, access control, fairness review, human oversight, and monitoring for subgroup harm. The project is not production-ready without those controls and a causal pilot evaluation.

## Conclusion

The project demonstrates that a calibrated Random Forest can rank churn risk effectively and concentrate historical churners within a limited contact budget. The strongest result is operational prioritization, not proof of churn prevention. The appropriate next step is a governed randomized retention pilot that measures incremental outcomes and replaces illustrative economics with verified business inputs.
