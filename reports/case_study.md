# PaySim Fraud Detection: From Exploratory Analysis to a Business Decision

## Executive summary

I developed a fraud-detection and transaction-prioritization project using 6.36 million synthetic mobile-money transactions from the PaySim dataset.

The objective was to identify potentially fraudulent transactions before approval and prioritize them for a limited manual-review queue.

I compared rule-based policies, logistic regression, and LightGBM models. The final A0 LightGBM policy caught more fraud cases than the strongest rule in the locked test evaluation. However, it captured only 2.76% more fraudulent transaction value, below the predeclared business requirement of at least 5%.

**Final recommendation:** Retain the existing rule as the selected policy and keep the A0 LightGBM model as a research challenger.

The project demonstrates that a model with better predictive performance is not automatically the better business decision.

## 1. Business problem

Payment fraud teams have limited resources to investigate suspicious transactions. Reviewing every transaction is impractical, so the business needs a method for prioritizing transactions before approval.

The project focused on this question:

*Can a machine-learning policy identify more fraudulent transaction value than a rule-based approach while operating under the same review capacity?*

The scoring moment was defined as before transaction approval. This decision shaped the feature design: information that would only become available after a transaction could not be used by the deployable candidate.

The primary operating assumption was a capacity of 500 alerts per 24 simulation steps.

A model would only replace the best feasible rule if it captured at least 5% more fraudulent transaction value and produced a better modeled net-benefit outcome under the same capacity.

## 2. Data and exploratory analysis

The project used the PaySim synthetic mobile-money transaction dataset.

Key dataset characteristics:

- 6,362,620 transactions.
- 8,213 observed fraud cases.
- Overall fraud rate of approximately 0.129%.
- Simulation steps 1–743.

The initial analysis examined fraud rates by transaction type, transaction amount, simulation hour, destination category, and historical account behavior.

Some important findings were:

- TRANSFER had 4,097 observed fraud cases among 532,909 transactions, a fraud rate of approximately 0.769%.
- CASH_OUT had 4,116 observed fraud cases among 2,237,500 transactions, a rate of approximately 0.184%.
- PAYMENT, DEBIT, and CASH_IN had no observed fraud cases in this dataset.
- Fraudulent transaction value was concentrated in larger transactions, but large legitimate transactions were also common.
- The existing `isFlaggedFraud` indicator captured very little of the observed fraud.

An important analytical distinction was separating transaction counts from fraud rates. A transaction type can produce many fraud cases because it has high transaction volume, without necessarily having the highest underlying fraud risk.

The analysis also investigated transaction sequences and balance-derived signals. Simulator-specific balance behavior raised reliability concerns, so balance-derived features were not included in the deployable A0 candidate.

## 3. Temporal evaluation and leakage prevention

A random train/test split would not reflect the intended chronological decision process. I therefore used time-based evaluation periods.

For the validation model:

- Effective training steps: 1–421.
- Purged label-delay gap: 422–445.
- Validation steps: 446–594.

A 24-step label delay was applied when determining which labels were available for training.

For the final test model:

- Training labels available through step 570.
- Purged steps: 571–594.
- Locked test steps: 595–743.
- Test population: 123,580 transactions.

The final test evaluation was completed once and protected by a test lock.

There is an important limitation: some early exploratory summaries included aggregates from the complete dataset, including the eventual test period. The test was excluded from model fitting and model selection, but was not completely unseen during all project analysis.

## 4. Feature engineering and model development

I defined three feature sets:

- A0: decision-time features without direct use of PaySim balance columns.
- A1: A0 plus eligible pre-transaction balance features, investigated diagnostically.
- B: includes post-transaction balance information and is not suitable for pre-approval scoring.

A0 was selected as the deployable candidate.

Its features included transaction type, hour, log transaction amount, destination merchant status, past-24-step sender and destination activity, first-time destination status, and a historical sender-amount ratio.

The model also used a missingness indicator for the historical sender ratio, giving the final model 11 ordered inputs.

Historical activity features used earlier simulation steps only. Same-step transactions were excluded because their relative ordering was unknown.

### Baselines and model selection

I compared rule-based policies with logistic regression and LightGBM.

The LightGBM A0 tuning experiment evaluated 12 configurations using validation data. The selected configuration was Trial 10, `lower_feature_sampling`.

The final configuration used:

- 200 estimators.
- Learning rate of 0.05.
- 31 leaves.
- Maximum feature sampling of 0.6.
- Row sampling of 0.8.
- L2 regularization of 1.0.
- Positive-class weight of 1.0.
- Random seed of 42.

The selected operating policy ranked transactions using:

`priority_score = fraud_score × transaction_amount`

This was a prioritization heuristic, not a calibrated expected-loss estimate.

## 5. Validation results

On validation, the A0 score-times-amount policy was compared with the TRANSFER/CASH_OUT-by-amount rule at the same alert capacity.

| Metric | Best rule | A0 model |
|---|---:|---:|
| Alerts | 3,104 | 3,104 |
| True positives | 397 | 756 |
| False positives | 2,707 | 2,348 |
| Precision | 12.79% | 24.36% |
| Recall | 25.58% | 48.71% |
| Fraud value caught | 1.568B | 1.769B |
| Dollar-weighted recall | 77.13% | 87.03% |

The model captured approximately 12.84% more fraudulent transaction value than the rule on validation, exceeding the 5% hurdle at that stage.

This supported carrying the model forward for the final locked evaluation. It did not establish that the model had passed the business acceptance criterion.

## 6. Locked test results

The final test compared the frozen A0 score-times-amount policy against the TRANSFER/CASH_OUT-by-amount rule.

| Metric | Best rule | A0 model |
|---|---:|---:|
| Alerts used | 2,897 | 2,897 |
| True positives | 818 | 1,043 |
| False positives | 2,079 | 1,854 |
| False negatives | 836 | 611 |
| Precision | 28.24% | 36.00% |
| Recall | 49.46% | 63.06% |
| Fraud value caught | 2.510B | 2.579B |
| Dollar-weighted recall | 92.20% | 94.74% |

The model identified 225 additional fraud cases and improved precision and recall.

However, the increase in fraudulent transaction value was approximately 69.19 million simulated units, equivalent to a 2.76% improvement over the rule.

The predeclared hurdle required at least 5% more fraudulent transaction value. The model therefore fell short by approximately 56.31 million simulated units of the required captured value.

A paired simulation-day block bootstrap produced an exploratory 95% interval of approximately 1.18% to 4.86%. Only five complete daily blocks were available, with two partial blocks held fixed, so this interval has a low effective sample size and should be interpreted cautiously.

**The model did not meet the predeclared business acceptance criterion. The rule remained selected.**

## 7. Explainability and error analysis

The validation SHAP analysis identified transaction type as the strongest global contributor to model predictions.

Other influential features included:

- Destination merchant status.
- Destination transaction count over the previous 24 steps.
- Hour of day.
- Log transaction amount.
- Destination transaction amount over the previous 24 steps.

Sender-history features contributed very little to the selected model's global SHAP importance.

The error analysis identified a concentration of false positives among large legitimate transfers.

Of the 2,348 validation false positives, 1,974 were TRANSFER transactions and 374 were CASH_OUT transactions.

The amount-band analysis showed that 2,310 false positives and 529 missed fraud cases occurred in the transaction amount band of 100,000 or higher.

This highlighted a practical trade-off: prioritizing large potential fraud value can increase the review of legitimate high-value transactions, while smaller fraudulent transactions may still be missed.

SHAP values describe model behavior; they do not establish causation.

## 8. Drift and operational stability

Fraud prevalence changed substantially between periods:

- Effective training prevalence: approximately 0.0793%.
- Validation prevalence: approximately 0.6804%.
- Test prevalence: approximately 1.3384%.

The validation drift analysis identified substantial distribution differences in:

- Destination transaction count: PSI approximately 2.093.
- Destination transaction amount sum: PSI approximately 1.044.
- Hour of day: PSI approximately 0.338.

Transaction type and log amount had comparatively low PSI.

Daily validation precision varied from approximately 18.8% to 29.87%. Recall ranged from approximately 40.23% to 60.22%.

These results indicate that average model performance does not fully describe day-to-day operational behavior. PSI measures distribution differences; it does not prove model degradation or establish that drift caused specific errors.

## 9. Business recommendation

I would retain the TRANSFER/CASH_OUT-by-amount rule as the selected policy under the project's predeclared acceptance criteria.

The A0 LightGBM model remains a promising research challenger because it identified more fraud cases and achieved better precision and recall on the locked test.

However, its 2.76% improvement in fraudulent transaction value did not meet the 5% hurdle.

Before any real deployment decision, the work would need representative real payment data, better estimates of actual financial losses and recovery, more complete operational cost assumptions, further drift investigation, and prospective evaluation across additional time periods.

The model should not be used to automatically approve or block real transactions based on this synthetic-data evaluation.

## 10. Limitations

The main limitations are:

1. PaySim is synthetic and may not represent real payment behavior or fraud mechanisms.
2. Transaction value is a proxy, not verified financial loss.
3. Fraud prevalence and feature distributions shift across time periods.
4. The ranking heuristic is not calibrated expected loss.
5. The simplified net-benefit proxy excludes several operational costs and benefits.
6. The final test bootstrap interval is based on only five complete daily blocks.
7. Early full-dataset exploratory analysis included some eventual test-period aggregates.
8. Performance on synthetic data does not establish real-world generalization.

## 11. Reproducibility artifacts

The project includes:

- `src/feature_contract.py` — approved feature definitions and restrictions.
- `src/features.py` — A0 feature generation.
- `src/inference.py` — inference and alert-prioritization interface.
- `src/evaluation.py` — temporal evaluation utilities.
- `src/tune_lightgbm_a0.py` — bounded LightGBM tuning.
- `src/evaluate_a0_final_test.py` — guarded one-time final evaluation.
- `reports/model_card.md` — model card and deployment recommendation.
- `reports/evaluation_protocol.md` — evaluation and leakage protocol.
- `reports/model_manifest.json` — model identity and reproducibility metadata.
- `reports/test_evaluation_lock.json` — final test lock.

Automated tests are run using:

`python -m pytest -q`

The model and processed data binaries are intentionally excluded from Git and must be provided separately for local inference.

---

**Final takeaway:** The project was not simply about building a model with better metrics. It was about defining a decision, evaluating competing policies under realistic constraints, understanding failure modes, and selecting the option that met the business requirement. The final evidence supported retaining the rule while treating LightGBM as a research challenger.