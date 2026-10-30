# Evaluation Protocol — PaySim Fraud Detection

## 1. Scoring decision

Score transactions before approval. Deployable features must contain only information available at the time of scoring.

## 2. Temporal split

- Training: steps 1–445.
- Validation: steps 446–594.
- Test: steps 595–743.

Splits are chronological, with no random transaction split. The boundaries are based on simulation steps and do not guarantee equal transaction volumes.

**Test integrity caveat:** Initial EDA included aggregated fraud statistics across the complete dataset, including later periods. Therefore, the test block is an exposed temporal holdout rather than a completely blind test. This limitation must be retained in the final case study.

## 3. Feature sets

- A0: non-balance, decision-time features. This is the initial deployable candidate.
- A1: A0 plus eligible pre-transaction balance features and ratios, evaluated separately because of data-reliability concerns.
- B: A1 plus post-transaction balances. Diagnostic only; not deployable.

The target label `isFraud` is not a feature. Future transactions, future labels, and post-transaction fields must not influence pre-approval features.

## 4. Primary and secondary metrics

- Primary model metric: PR-AUC (average precision).
- Operational metrics: precision and recall under a fixed alert budget.
- Value-based metric: dollar-weighted recall, using fraudulent transaction value as a proxy rather than verified losses.
- Business metric: modeled net benefit under explicitly stated cost assumptions.

Accuracy and ROC-AUC will not be primary selection criteria because the dataset has extreme class imbalance and the operational decision is constrained by alert capacity.

## 5. Baselines

Compare:
- The existing `isFlaggedFraud` indicator.
- A rule flagging TRANSFER transactions above 200,000.
- A hand-written behavioral rule.
- Logistic regression.
- A gradient-boosting model and other declared candidates.

All comparable policies must be evaluated on the same data blocks and under the same alert budget.

## 6. Operating constraints and success criteria

The initial review-capacity scenario is 500 alerts per day, with sensitivity scenarios of 100 and 1,000 alerts per day.

A model must:
1. Stay within the chosen review capacity.
2. Produce positive modeled net benefit under the base-case cost assumptions.
3. Identify at least 5% more attempted fraud transaction value than the best feasible rule under the same alert budget.
4. Produce higher modeled net benefit than the best feasible rule to justify preferring the model.

These values are project assumptions, not universal industry standards.

## 7. Leakage tests

Tests must verify that:
- Train, validation, and test periods are disjoint in time.
- Historical features use only data available before the transaction.
- Label-derived features respect a predefined delay.
- Perturbing future records cannot change an earlier feature.
- The test-evaluation guard prevents accidental repeated final scoring.

## 8. Limitations

PaySim is synthetic. Transaction value is not the same as verified financial loss. Temporal patterns and balance behaviors may reflect the simulator's design and may not generalize to real payments.
