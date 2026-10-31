
# Evaluation Protocol — PaySim Fraud Detection

## 1. Business decision and scoring moment

The model is designed to score transactions **before approval**.

The operational objective is to prioritize transactions for a limited manual-review or verification queue.

The evaluation compares machine-learning policies against rule-based alternatives under the same review capacity.

The primary business question is:

> Does the model capture materially more fraudulent transaction value than the best feasible rule, at the same alert capacity and with a better modeled net-benefit outcome?

A model is not automatically preferred simply because it has a higher predictive metric.

## 2. Dataset and temporal evaluation design

The dataset contains simulation steps 1–743.

All evaluation splits are chronological. Transactions are not randomly assigned to training, validation, or test sets.

### Nominal temporal boundaries

| Period | Simulation steps | Purpose |
|---|---:|---|
| Training window | 1–445 | Development training window |
| Validation | 446–594 | Model and policy selection |
| Locked test | 595–743 | One-time final evaluation |

The boundaries are based on simulation steps and do not guarantee equal transaction volumes.

### Label-delay purge

A 24-step label delay is applied when determining which training labels are available at a given scoring boundary.

For the validation model:

- Nominal training window: steps 1–445.
- Effective training steps: 1–421.
- Purged steps: 422–445.
- Validation steps: 446–594.

The final 24 steps of the nominal training window are excluded so that the training labels respect the declared availability delay.

### Final test refit

Before the final locked evaluation, the frozen model configuration was refitted using the expanded historical training window:

- Training labels available through step 570.
- Purged steps: 571–594.
- Locked test steps: 595–743.

The final test model therefore has a different training cutoff from the earlier validation model.

The locked test period was not used for model fitting or policy selection.

### Test-period exposure caveat

Initial exploratory analysis included aggregated fraud statistics from the complete dataset, including the eventual test period.

Therefore, the final test was kept out of model fitting and model selection, but it was not completely unseen during all project analysis.

This limitation must be disclosed when interpreting the final test result.

## 3. Label and information availability

The target variable is `isFraud`.

It is used for supervised training and evaluation only. It must never enter the model feature matrix.

The 24-step label delay is respected when determining the eligible training labels at temporal boundaries.

Historical features must use only information available before the transaction being scored.

For rolling account-activity features, the historical window is:

`[step - 24, step - 1]`

Transactions from the current simulation step are excluded because their relative ordering is not known.

Future transactions and future labels must not influence earlier feature values.

## 4. Feature sets and deployment eligibility

Three feature sets were investigated.

| Feature set | Description | Purpose |
|---|---|---|
| A0 | Decision-time features without direct balance-column usage | Approved deployable candidate |
| A1 | A0 plus eligible pre-transaction balance features | Diagnostic experiment |
| B | A1 plus post-transaction balance information | Leakage demonstration only |

### A0 feature set

The A0 model uses:

1. `hour_of_day`
2. `type`
3. `log_amount`
4. `destination_is_merchant`
5. `sender_txn_count_prev_24_steps`
6. `sender_amount_sum_prev_24_steps`
7. `destination_txn_count_prev_24_steps`
8. `destination_amount_sum_prev_24_steps`
9. `is_first_time_destination_for_sender`
10. `amount_to_sender_past_mean_ratio`

The fitted model also uses:

`amount_to_sender_past_mean_ratio_is_missing`

This missingness indicator is derived from the historical sender-ratio feature.

The final model therefore expects 11 ordered model features.

### Feature restrictions

The following information must not enter the pre-approval model feature matrix:

- `isFraud` or other target-derived information.
- Raw sender and destination identifiers.
- Post-transaction balance columns.
- Features derived from future transactions.
- Any information unavailable at the decision point.

A1 remains diagnostic because of simulator-specific balance-data reliability concerns.

B is not suitable for pre-approval scoring because it includes post-transaction information.

The approved A0 feature contract is maintained in `src/feature_contract.py`.

## 5. Model and policy baselines

The project compares rule-based policies with statistical and machine-learning models.

The evaluated approaches include:

- Existing `isFlaggedFraud` indicator.
- Rule-based transaction-type and amount policies.
- Logistic regression.
- LightGBM A0 models.
- Diagnostic A1 experiments, kept separate from the deployable candidate.

All comparable alert policies must use the same evaluation population and alert-capacity assumptions.

The final locked comparison uses:

**Rule comparator:** TRANSFER/CASH_OUT transactions ranked by transaction amount.

**Machine-learning policy:** A0 fraud score multiplied by transaction amount.

## 6. Evaluation metrics

### Predictive metric

**Average Precision (PR-AUC)** is the primary predictive metric.

It is preferred over accuracy because fraud is highly imbalanced and identifying the minority fraud class is the objective.

### Operational classification metrics

Evaluate:

- True positives.
- False positives.
- False negatives.
- Precision.
- Recall.
- False-positive rate.

These metrics must be calculated under the same alert-selection policy and capacity.

### Fraud-value metrics

Fraudulent transaction value is used as a proxy for value exposed to fraud.

Evaluate:

- Total fraudulent transaction value.
- Fraudulent transaction value captured by selected alerts.
- Dollar-weighted recall.

Transaction amount is not verified financial loss, recovered money, or guaranteed financial benefit.

### Modeled net benefit

The final evaluation uses a simplified manual-review cost proxy of 100 units per alert.

The net-benefit proxy is:

`fraud value caught - manual review cost proxy`

This is a project-specific modeling assumption, not an estimate of actual institutional profit or realized savings.

The simplified proxy does not fully account for false-blocking costs, customer friction, investigation outcomes, recovery rates, or verification costs.

## 7. Alert capacity and ranking policy

The primary operating scenario is:

- 500 alerts per 24 simulation steps.
- 24 simulation steps per day.
- Partial simulation days receive a proportionate alert capacity.

Sensitivity analysis also examines:

- 100 alerts per 24 steps.
- 1,000 alerts per 24 steps.

The selected machine-learning policy ranks transactions using:

`priority_score = fraud_score * amount`

This is a ranking heuristic, not a calibrated expected-loss prediction.

The policy selects the highest-priority transactions within each simulation day's available alert capacity.

Ties are resolved deterministically by:

1. Priority score descending.
2. Transaction amount descending.
3. Original row position.

The rule comparator ranks eligible TRANSFER/CASH_OUT transactions by amount under the same capacity constraints.

## 8. Model and policy selection

Model and policy selection must use validation data only.

The LightGBM A0 tuning experiment used a bounded search of 12 configurations.

The selection criteria were:

1. Fraudulent transaction value captured at the fixed alert budget.
2. Average Precision as the secondary criterion.
3. True positives as an additional tie-breaker.

The selected configuration was Trial 10:

`lower_feature_sampling`

Its key parameters were:

| Parameter | Value |
|---|---:|
| Estimators | 200 |
| Learning rate | 0.05 |
| Number of leaves | 31 |
| Maximum depth | -1 |
| Minimum child samples | 100 |
| Maximum bins | 127 |
| Feature sampling | 0.6 |
| Row sampling | 0.8 |
| Row sampling frequency | 1 |
| L2 regularization | 1.0 |
| Positive class weight | 1.0 |
| Random seed | 42 |

The final test model uses this frozen configuration.

No test-period performance may be used to choose another model, alter the alert policy, or tune model parameters.

## 9. Predeclared business acceptance criteria

A machine-learning policy must:

1. Respect the alert capacity.
2. Produce a positive modeled net-benefit outcome under the declared cost assumptions.
3. Capture at least 5% more fraudulent transaction value than the best feasible rule at the same alert capacity.
4. Produce higher modeled net benefit than the best feasible rule.

The 5% improvement hurdle is a project-specific business requirement, not a universal industry threshold.

The model should not replace the rule unless the declared acceptance criteria are met.

## 10. Locked final test procedure

The final test uses:

- Training label cutoff: step 570.
- Purged steps: 571–594.
- Test steps: 595–743.
- Test population: 123,580 transactions.
- Alert budget: 500 alerts per 24 simulation steps.

The final model and comparison policy are frozen before test evaluation.

The one-time evaluation is recorded in:

- `reports/a0_final_test_evaluation.json`
- `reports/a0_final_test_evaluation.csv`
- `reports/test_evaluation_lock.json`

The lock status is:

`completed_once`

**The locked test must not be rescored.**

Do not delete the lock, rerun the final test evaluation, or use its results for additional model selection.

### Recorded final decision

The final locked evaluation recorded approximately 2.76% more fraudulent transaction value captured by the model than by the rule.

This did not meet the predeclared 5% hurdle.

Therefore, the rule remains the selected policy, while the A0 LightGBM model remains a research challenger.

The final outcome and detailed metrics are documented in `reports/a0_final_test_evaluation.json` and `reports/model_card.md`.

## 11. Leakage and implementation checks

The automated tests and feature-generation checks should verify that:

- Training, validation, and test periods are chronologically separated.
- The 24-step label-delay purge is respected.
- Historical activity features use past-only transactions.
- Same-step and future transactions do not influence earlier features.
- Target, identifier, and post-transaction fields are rejected as model inputs.
- The inference feature order matches the saved model.
- Categorical levels match the training metadata.
- Missing historical ratio values retain the same behavior as during training.
- Alert selection respects the configured capacity.
- Priority scores follow the frozen formula.
- The final test lock prevents accidental repeated scoring.

Run the automated test suite with:

```bash
python -m pytest -q
```

## 12. Interpretation and limitations

The evaluation must be interpreted in light of the following limitations:

1. PaySim is synthetic and may not represent real payment behavior or fraud mechanisms.
2. Transaction amount is not verified financial loss.
3. Fraud prevalence changes substantially across temporal periods.
4. Destination activity and hour-of-day distributions show evidence of drift.
5. Balance-derived features have simulator-specific reliability concerns.
6. Fraud score multiplied by amount is a ranking heuristic, not calibrated expected loss.
7. The simplified net-benefit proxy excludes several operational costs and benefits.
8. Only five complete daily blocks were available for the paired bootstrap analysis, making its interval exploratory.
9. Earlier full-dataset exploratory analysis included some eventual test-period aggregates.
10. Performance on synthetic data does not establish generalization to real payment systems.

The A0 model is a research challenger, not a production-approved payment decision system.

The rule remains selected under the predeclared business acceptance criteria.
