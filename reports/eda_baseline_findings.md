# Fraud Baseline Findings

## Our main findings are:

| Finding           | Conclusion                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Overall fraud     | 8,213 cases among 6,362,620 transactions; a 0.129082% fraud rate                                                            |
| Transaction types | TRANSFER has a substantially higher fraud rate than CASH_OUT                                                                |
| Fraud value       | Both TRANSFER and CASH_OUT contribute similar total fraudulent transaction values                                           |
| Amount bands      | The 100,000-and-above band contains approximately 79.2% of recorded fraud cases and 99.39% of fraudulent transaction value  |
| Temporal pattern  | Extreme rates are associated with sparse steps containing only fraud-labelled transactions; interpretation requires caution |

## 1. Overall Fraud Baseline

### Observed results

The dataset contains 6,362,620 transactions, of which 8,213 are labelled fraudulent and 6,354,407 are labelled legitimate.

The observed fraud rate is 0.129082%. The total transaction value associated with fraud is 12,056,415,427.84 PaySim amount units, and the average fraudulent transaction amount is approximately 1,467,967.30 units.

### Interpretation

Fraud is rare relative to legitimate transaction volume. This extreme class imbalance means that accuracy alone would be an inadequate measure of fraud-detection effectiveness. A classifier that predicts every transaction as legitimate would achieve approximately 99.8709% accuracy while detecting no fraud.

### Limitation

The reported fraudulent transaction value is not a verified measure of money stolen or financial loss. PaySim is synthetic, and its fraud cancellation mechanics affect how financial outcomes should be interpreted.

## 2. Fraud by Transaction Type

### Observed results

TRANSFER accounts for 4,097 fraud cases from 532,909 transactions, producing a fraud rate of 0.768799%.

CASH_OUT accounts for 4,116 fraud cases from 2,237,500 transactions, producing a fraud rate of 0.183955%.

PAYMENT, DEBIT and CASH_IN have zero recorded fraud cases in this dataset.

TRANSFER and CASH_OUT together account for all 8,213 recorded fraud cases and approximately 100% of fraudulent transaction value.

### Interpretation

CASH_OUT has 19 more fraud cases than TRANSFER, but TRANSFER has approximately 4.18 times the observed fraud rate.

The difference is explained in part by transaction volume: CASH_OUT occurs much more frequently. Therefore, fraud counts and fraud rates provide different information.

TRANSFER contributes approximately 6.067 billion units of fraudulent transaction value, compared with approximately 5.989 billion for CASH_OUT. The total values are similar despite the difference in transaction volumes and rates.

### Business implication

TRANSFER and CASH_OUT warrant particular investigation in subsequent EDA and modeling. However, the observed rates do not by themselves justify blocking either transaction type. Any intervention must consider legitimate transactions, false positives, alert capacity and modeled business costs.

### Limitations

The concentration of fraud labels in these two types reflects this synthetic dataset and must not be assumed to represent every real-world payments system.

## 3. Decisions for Modeling

- Use PR-AUC and precision/recall under a fixed alert budget rather than relying on accuracy alone.
- Evaluate fraud counts and fraud rates separately.
- Track attempted fraudulent transaction value identified alongside case-based metrics.
- Compare predictive models against simpler transaction-type and behavioral rules.
- Investigate false positives and the consequences of blocking legitimate transactions.
- Do not interpret fraudulent transaction value as confirmed financial loss.

## 4. Fraud by Simulated Hour and Simulation Day

### Observed results

The simulated-hour query shows substantial differences in transaction volume across time-of-day buckets. The highest observed fraud rates occur in simulated hours 3 and 4, at 22.079% and 22.303%, respectively. These buckets contain 1,241 and 1,641 transactions.

The daily aggregation shows similarly uneven volumes. Simulation day 3 contains 1,070 transactions and 310 fraud cases, while simulation day 31 contains 272 transactions, all labelled fraudulent.

### Interpretation

Fraud rates vary sharply across time groups, but the groups also differ significantly in transaction volume. The extreme rates in small groups require further investigation before we conclude that time-of-day or simulation day provides a robust predictive signal.

### Limitation and follow-up

The day numbers are derived from groups of 24 sequential time steps; the final group is incomplete. The data is synthetic, and the observed time patterns may reflect simulator behavior rather than a generalizable real-world fraud pattern.

Inspect individual time steps, transaction counts, legitimate counts, fraud counts, and transaction-type composition before finalizing the temporal findings.

## 5. Fraud by Transaction Amount

### Observed results

The 100,000-and-above band contains 2,837,364 transactions and 6,506 fraud cases, with an observed fraud rate of 0.229297%.

The band accounts for approximately 79.2% of recorded fraud cases and 99.39% of total fraudulent transaction value, despite containing approximately 44.6% of all transactions.

The below-100 band has a fraud rate of 0.127632%, while the two middle bands from 100 to below 10,000 have lower observed rates.

### Interpretation

Fraudulent transaction value is highly concentrated in the highest amount band. However, the band still contains 2,830,858 legitimate transactions, meaning that automatically flagging every transaction in this band would generate a very large number of false positives.

### Business implication

Transaction amount warrants further investigation and may be useful as a model feature. However, an amount threshold alone should not be adopted as the fraud policy without evaluating precision, review capacity, false-positive consequences, and modeled net benefit.

### Limitation

The amount boundaries are exploratory choices. The value associated with fraudulent transactions is not a verified measure of actual financial loss in the synthetic dataset.

## 6. Fraudulent Transfer-to-Cash-Out Sequence

### Observed results

The dataset contains 4,097 fraudulent TRANSFER transactions. Matching their destination account IDs against CASH_OUT origin account IDs produced three account-level matches across all time periods.

Two matching cash-outs occurred before the corresponding transfers. The remaining cash-out occurred 481 steps after its matched transfer but was labelled legitimate.

No fraudulent transfer destination was matched to a later fraudulent cash-out origin under the defined join conditions.

### Interpretation

The expected account-level sequence was not observed using this matching rule. Account-ID overlap alone does not establish that a later cash-out is part of the same fraud event.

### Limitations

The analysis does not prove that no account-takeover activity occurs in the simulator. It establishes only that the specified account IDs, temporal ordering, and fraud labels did not identify a qualifying sequence.

Any future sequence-based feature must be based on relationships that can actually be reconstructed from the data and must use only information available at scoring time.

## 7. Existing Fraud Flag vs. Simple Transfer Threshold

### Observed results

The existing `isFlaggedFraud` indicator generated 16 alerts, all of which corresponded to fraud-labelled transactions. Its precision was 100%, its case-based recall was 0.1948%, and its dollar-weighted recall was 0.6452%.

Flagging every TRANSFER above 200,000 generated 409,110 alerts, including 2,740 true positives and 406,370 false positives. Precision was 0.6697%, case-based recall was 33.3617%, and dollar-weighted recall was 49.4038%.

### Interpretation

The existing indicator is highly precise but captures very little of the recorded fraud. The broader threshold catches substantially more fraudulent transaction value, but more than 99% of its alerts are false positives.

### Business implication

The simple threshold is a useful benchmark, but its alert volume makes it infeasible under the project's assumed review budget of 500 alerts per day. Any alternative must balance detection value, precision, review workload, and modeled net benefit.

### Limitations

These are descriptive results on the full dataset, not a final out-of-time evaluation. The threshold is not a selected deployment policy. The findings describe synthetic data and transaction value associated with fraud, not verified financial losses.

## 8. Account Reuse Across Temporal Blocks

### Observed results

Across the dataset, 9,073,900 distinct account identifiers appeared in the combined origin and destination roles.

Under the provisional temporal boundaries, 8,566,556 accounts appeared in training, 438,632 in validation, and 242,051 in test. A total of 158,793 distinct accounts appeared in more than one block, representing approximately 1.75% of distinct accounts.

A total of 14,546 accounts appeared in all three blocks. Pairwise overlap counts include these accounts.

### Interpretation

Account recurrence across temporal blocks is not inherently data leakage. Existing accounts may naturally participate in transactions during multiple periods.

The critical requirement is that features for each transaction use only information available before that transaction is scored. Future transactions and future fraud labels must not affect historical features.

### Modeling decision

Retain chronological evaluation to approximate deployment on future transactions. Construct account-history features with strictly preceding windows, and test that perturbing future records cannot change earlier features.

### Limitation

The provisional time boundaries are not yet the final evaluation protocol. The account overlap results describe identifiers across periods; they do not independently establish whether a feature leaks future information.

## 9. Historical Sender Mean Ratio — Feature Coverage

### Observed results

Within the training period, 8,304 of 6,010,937 transactions (0.1381%) have prior outgoing transaction history. This group contains 9 fraud cases and has a fraud rate of 0.108382%.

The remaining 6,002,633 transactions have no prior outgoing transaction history. This group contains 4,998 fraud cases and has a fraud rate of 0.083263%.

### Interpretation

The current amount-to-sender-past-mean ratio is undefined for approximately 99.86% of training transactions. Although the fraud rate is higher among transactions with prior outgoing history, the group contains only nine fraud cases, making the observed difference insufficient to establish reliable predictive value.

### Modeling decision

Retain the original A0 specification for the initial baseline, but predefine a validation ablation that removes the historical-mean ratio. Compare both versions under the same evaluation protocol and alert budget. Handle undefined ratios explicitly.

### Limitation

The observed association is descriptive and does not establish that the ratio improves future fraud detection. Feature utility must be assessed on validation data before the final model decision.


## 10. Rule Baselines on the Validation Period

### Observed results

The validation period contains 228,103 transactions, including 1,552 fraud-labelled transactions.

The existing `isFlaggedFraud` indicator generated two alerts, both correctly labelled as fraud. It achieved 100% precision, 0.1289% recall, and 0.1933% dollar-weighted recall.

The rule flagging every TRANSFER and CASH_OUT transaction generated 94,512 alerts, including 1,552 true positives and 92,960 false positives. It achieved 100% recall and 100% dollar-weighted recall, but its precision was only 1.6421%.

The high-value TRANSFER/CASH_OUT rule generated 65,370 alerts, including 1,264 true positives and 64,106 false positives. Its case-based recall was 81.4433%, while its dollar-weighted recall was 99.4576%.

The rule flagging TRANSFER transactions above 200,000 generated 15,672 alerts, including 518 true positives and 15,154 false positives. It achieved 33.3763% recall and 48.9973% dollar-weighted recall.

### Interpretation

The transaction-type rule detects all recorded validation fraud because the dataset's fraud labels are concentrated in TRANSFER and CASH_OUT. However, it generates an impractically large alert volume.

The high-value composite rule identifies nearly all recorded fraudulent transaction value but misses a larger share of fraud cases. This illustrates why case-based recall and value-weighted recall must be evaluated separately.

The existing fraud flag is highly precise but detects only a tiny fraction of the recorded fraud.

### Operational implication

Under the assumed limit of 500 manual-review alerts per day, all three broad rules generate alert volumes far above capacity on average. The existing flag satisfies the volume constraint but is insufficient as a standalone detection policy.

The next stage should compare predictive models and rules under a common alert budget and modeled cost assumptions.

### Limitations

These are validation-period results on synthetic data. Fraudulent transaction value is not verified money saved or actual financial loss. The historical EDA included later-period aggregate statistics, so the formal test block is not a completely blind holdout.


## 11. Class-Weighted Logistic-Loss Baseline

### Training setup

The model used A0 features and was trained on 6,010,937 transactions from steps 1–445, including 5,007 fraud cases. Class weights were calculated from training labels only. Numeric preprocessing was fitted on training data, and the model was evaluated on validation steps 446–594.

### Validation results

- Average precision: 0.126056.
- Validation alerts: 3,104 under the defined capacity convention.
- True positives: 618.
- False positives: 2,486.
- False negatives: 934.
- Precision: 19.9098%.
- Recall: 39.8196%.
- False-positive rate: 1.0973%.
- Dollar-weighted recall: 51.3660%.

### Interpretation

At the selected alert budget, the model identifies approximately 39.8% of validation fraud cases and 51.4% of recorded fraudulent transaction value. Its precision is substantially higher than the underlying validation fraud prevalence, indicating that the model concentrates fraud-labelled transactions in the alert queue.

### Limitations

The validation fraud rate is approximately 8.2 times the training fraud rate, indicating a substantial temporal distribution difference. The reported transaction value is not verified financial loss, and average precision does not establish a calibrated probability.

### Next evaluation step

Compare the model with budget-constrained rule variants on the same validation period. Also evaluate an A0 ablation that removes the historical-mean ratio, whose coverage is extremely low.
