
# Model Card — PaySim Fraud Detection

## 1. Model overview

**Project:** PaySim Fraud Detection  
**Model:** LightGBM binary classifier  
**Selected feature set:** A0  
**Alert-prioritization policy:** Fraud score × transaction amount  
**Primary use case:** Prioritizing transactions for human fraud review  
**Status:** Research prototype; not approved for production deployment

The model estimates transaction fraud scores and ranks transactions using:

    priority_score = fraud_score × transaction_amount

The resulting priority score is a ranking heuristic. It is not a calibrated estimate of expected financial loss.

The system is designed to operate under a limited daily manual-review capacity.

## 2. Intended use

The model is intended for:

- Fraud detection research using the PaySim synthetic dataset.
- Comparing machine-learning alert prioritization against rule-based policies.
- Demonstrating chronological model evaluation.
- Investigating the trade-off between fraud cases detected and fraudulent transaction value captured.
- Demonstrating feature engineering, leakage prevention, explainability and operational evaluation.

The model is not approved for:

- Automatic transaction blocking.
- Automatic transaction approval.
- Direct deployment in a real banking or payment environment.
- Estimating actual recoverable financial losses.
- Making customer-level decisions without human review.

## 3. Dataset

**Dataset:** PaySim synthetic mobile-money transaction data.

Observed dataset characteristics:

- Total transactions: 6,362,620
- Observed step range: 1–743
- Target: `isFraud`
- Transaction categories: CASH_IN, CASH_OUT, DEBIT, PAYMENT and TRANSFER.
- Fraud cases: 8,213 in the complete dataset.

The dataset is synthetic. Its fraud-generation process and transaction behavior may not represent real-world financial institutions.

Transaction amount is used as a proxy for fraud value, not as a verified measure of financial loss.

## 4. Feature design

The deployable candidate uses the A0 feature set.

Features include:

- Hour of day.
- Transaction type.
- Log transaction amount.
- Destination merchant indicator.
- Sender transaction count over the previous 24 steps.
- Sender transaction amount over the previous 24 steps.
- Destination transaction count over the previous 24 steps.
- Destination transaction amount over the previous 24 steps.
- First-time destination indicator.
- Historical sender amount ratio.
- Historical sender-ratio missingness indicator.

Historical account activity uses only information from earlier steps.

Raw account identifiers and target variables are not model inputs.

### Excluded features

All balance-derived features are excluded from the deployable candidate:

- `oldbalanceOrg`
- `oldbalanceDest`
- `newbalanceOrig`
- `newbalanceDest`
- `drain_ratio`
- `amount_to_oldbalanceDest_ratio`

The A1 experiment demonstrated exceptionally high validation performance, but PaySim balance fields are subject to simulator-specific limitations. These features were therefore rejected as unsuitable deployment signals.

## 5. Model training and evaluation protocol

### Validation model

- Training steps: 1–421.
- Purged label-delay gap: steps 422–445.
- Validation steps: 446–594.
- Training rows: 5,994,350.
- Training fraud cases: 4,755.

### Final test model

- Training label cutoff: step 570.
- Purged label-delay gap: steps 571–594.
- Locked test steps: 595–743.
- Test rows: 123,580.

The final test evaluation was completed once.

Earlier full-dataset exploratory summaries included statistics from the eventual test period. Therefore, the test set was kept out of model fitting and model selection, but was not completely unseen during the entire project analysis.

## 6. Selected LightGBM configuration

The selected configuration was Trial 10 from the corrected A0 tuning experiment.

| Parameter | Value |
|---|---:|
| Objective | Binary |
| Estimators | 200 |
| Learning rate | 0.05 |
| Number of leaves | 31 |
| Maximum depth | Unlimited (-1) |
| Minimum child samples | 100 |
| Maximum bins | 127 |
| Feature sampling | 0.6 |
| Row sampling | 0.8 |
| Row sampling frequency | 1 |
| L2 regularization | 1.0 |
| Positive class weight | 1.0 |
| Random seed | 42 |

## 7. Operating policy

The selected policy ranks transactions by:

    fraud_score × transaction_amount

The primary operating scenario is:

- Alert budget: 500 alerts per 24 simulation steps.
- Comparator: TRANSFER/CASH_OUT transactions ranked by amount.
- Success criterion: At least 5% more fraudulent transaction value captured than the rule at the same capacity.

The policy does not use a fixed probability threshold. It selects the highest-priority transactions within the available daily review capacity.

## 8. Validation performance

At the 500-alert-per-24-step budget:

| Metric | Best rule | A0 score × amount |
|---|---:|---:|
| Alerts | 3,104 | 3,104 |
| True positives | 397 | 756 |
| False positives | 2,707 | 2,348 |
| Precision | 12.79% | 24.36% |
| Recall | 25.58% | 48.71% |
| Fraud value caught | 1.568B | 1.769B |
| Dollar-weighted recall | 77.13% | 87.03% |

Validation fraud-value uplift: approximately 12.84%.

The model substantially improved case detection and fraud-value capture on validation.

## 9. Locked test performance

The final locked evaluation used 500 alerts per 24 simulation steps.

| Metric | Best rule | A0 score × amount |
|---|---:|---:|
| Available capacity | 3,104 | 3,104 |
| Alerts used | 2,897 | 2,897 |
| True positives | 818 | 1,043 |
| False positives | 2,079 | 1,854 |
| False negatives | 836 | 611 |
| Precision | 28.24% | 36.00% |
| Recall | 49.46% | 63.06% |
| Fraud value caught | 2.510B | 2.579B |
| Dollar-weighted recall | 92.20% | 94.74% |

Fraud-value improvement over the rule: approximately 2.76%.

Required improvement: at least 5%.

**The model did not meet the predeclared business acceptance criterion.**

The model did outperform the rule on fraud cases caught, precision, recall and fraud-value capture. However, the improvement in fraud value was insufficient to satisfy the predefined 5% hurdle.

A paired daily-block bootstrap produced an exploratory 95% interval of approximately 1.18% to 4.86% for fraud-value uplift. Only five complete daily blocks were available, so this interval should be interpreted cautiously.

## 10. Explainability

The validation SHAP analysis identified the following dominant features:

1. Transaction type.
2. Destination merchant indicator.
3. Destination transaction count over the previous 24 steps.
4. Hour of day.
5. Log transaction amount.
6. Destination transaction amount over the previous 24 steps.

Sender-history features contributed very little to the selected model's predictions.

SHAP values explain the model's behavior. They do not prove causation or identify universal fraud mechanisms.

## 11. Error analysis

On validation, the frozen policy selected:

- True positives: 756.
- False positives: 2,348.
- False negatives: 796.
- True negatives: 224,203.

Major findings:

- False positives were heavily concentrated among large legitimate transfers.
- TRANSFER accounted for 1,974 false positives.
- CASH_OUT accounted for 374 false positives.
- Of the 796 missed fraud cases, 529 belonged to the amount band of 100,000 or higher.
- Some fraud cases narrowly missed the daily alert cutoff.
- Other missed cases received very low fraud scores despite large transaction amounts.

These findings illustrate the operational trade-off between prioritizing large potential fraud value and maintaining case detection across smaller fraudulent transactions.

## 12. Data and model drift

Training fraud prevalence:

0.0793%

Validation fraud prevalence:

0.6804%

Validation prevalence was approximately 8.58 times the training prevalence.

Feature PSI analysis identified:

| Feature | PSI |
|---|---:|
| Destination transaction count | 2.093 |
| Destination amount sum | 1.044 |
| Hour of day | 0.338 |

These features showed substantial distribution drift.

Transaction type and log transaction amount had low PSI.

Daily validation precision ranged from approximately 18.8% to 29.9%, illustrating operational variability across simulation days.

PSI thresholds are monitoring guidelines, not proof of model failure or statistical significance.

## 13. Limitations and risks

1. **Synthetic data:** PaySim may not reflect real transaction behavior or real fraud mechanisms.
2. **Transaction value proxy:** Fraudulent transaction value is not necessarily equivalent to financial loss.
3. **Temporal distribution shift:** Fraud prevalence changes substantially between training and validation.
4. **Feature drift:** Destination activity and hour distributions shift across periods.
5. **Simulator-specific balance behavior:** Balance-derived features were excluded from deployable candidates.
6. **Ranking heuristic:** Fraud score × amount is not a calibrated expected-loss estimate.
7. **Operational costs:** The simplified manual-review proxy does not fully model false blocking, verification, recovery, investigation outcomes or customer friction.
8. **Limited daily blocks:** The final test confidence interval has low effective sample size.
9. **Exploratory test-period analysis:** Some full-dataset exploratory summaries included the eventual test period.
10. **External generalization:** No evidence yet establishes that the model will perform similarly on real payment data.

## 14. Deployment recommendation

**Recommendation: Do not approve the model for production deployment based on this project alone.**

The A0 LightGBM policy demonstrates a promising improvement over the baseline rule, but it did not meet the predefined 5% fraud-value improvement hurdle on the locked test set.

The rule remains the selected baseline under the agreed decision protocol.

The A0 model should remain a research challenger pending:

- Validation on a representative real payment dataset.
- Better estimates of actual fraud losses and recovery rates.
- More complete review, false-blocking and customer-friction cost models.
- Further investigation of temporal and feature drift.
- Evaluation across more independent time periods.
- A new prospective evaluation protocol before any deployment decision.

## 15. Reproducibility artifacts

Key project artifacts include:

- `data/processed/paysim_a0_features.parquet`
- `models/lightgbm_a0_tuned_best.joblib`
- `models/lightgbm_a0_final_test_model.joblib`
- `reports/lightgbm_a0_tuning_results.csv`
- `reports/lightgbm_a0_vs_a1_metrics.csv`
- `reports/a0_final_test_evaluation.json`
- `reports/a0_final_test_evaluation.csv`
- `reports/a0_error_analysis_report.json`
- `reports/a0_rule_overlap_report.json`
- `reports/a0_drift_analysis_report.json`
- `reports/shap_a0_validation_summary.json`

The final test lock is stored in:

`reports/test_evaluation_lock.json`

The locked test must not be rescored or used for further model selection.
