# PaySim Fraud Detection — Interview Preparation

## 1. Project summary

**Project:** Mobile-money fraud detection using the PaySim synthetic transaction dataset.

**Business problem:** Identify potentially fraudulent transactions before approval so that a limited review team can prioritize the transactions most worth investigating.

**Dataset:** 6,362,620 transactions, including 8,213 fraudulent transactions. The overall fraud rate is approximately 0.129%.

**Modeling approach:** A temporally validated LightGBM model using the A0 feature set, compared against a feasible transaction-type rule under a fixed alert budget.

**Final decision:** The model identified more fraud cases and achieved higher precision and recall than the rule on the locked test set. However, its fraud-value uplift was only 2.76%, below the project's 5% business hurdle. The rule therefore remains the selected policy, while the model remains a research challenger.

**Important limitation:** PaySim is synthetic. Transaction amount is used as a proxy for fraud value, not as verified financial loss or money that would necessarily have been prevented.

---

## 2. Business understanding and decision-making

### Q1. Why did you retain the rule when the model detected more fraud?

The model detected 225 more fraud cases than the rule on the locked test set. It also achieved higher precision and recall. However, I did not define success using detection count alone.

The business objective was to capture more fraudulent transaction value under the same review capacity. I set a hurdle requiring at least 5% more fraud value captured than the best feasible rule.

The model achieved a 2.76% uplift, which did not meet that threshold. Therefore, I retained the rule as the selected policy and treated the model as a challenger.

This demonstrates that a model can perform better statistically without necessarily meeting the business requirements for replacing an existing policy.

### Q2. How would you improve the model's fraud-value capture?

I would investigate the transactions where the model made costly mistakes.

On validation data, the model had 2,310 false positives and 529 false negatives in the transaction amount band of at least 100,000. Those groups are important because they involve large transaction amounts, but I would not assume every large false positive represents an avoidable mistake.

I would examine the legitimate transactions incorrectly prioritized, the fraudulent transactions missed, and whether the errors share patterns involving transaction type, destination history, amount, or time.

I would then test alternative features, ranking strategies, and policies using validation data under the same alert budget. Any promising approach would need evaluation on a new, untouched future period. I would not reuse the locked test set for tuning.

### Q3. What if a new model captures 6% more fraud value but has only 20% precision, compared with the current model's 36%?

I would not automatically select it.

At the same capacity of 2,897 alerts, 20% precision would mean approximately 579 alerts were true positives and 2,318 were false positives. The model might capture more transaction value, but the additional false positives could increase investigation workload, customer friction, or unnecessary verification.

I would compare the policies using the same data and review budget, then assess fraud value captured, precision, false-positive burden, and a more complete operational cost model.

If it still met the business requirements, I would consider a controlled human-review pilot on new future data rather than immediately replacing the existing policy.

---

## 3. Feature engineering and data reliability

### Q4. Why did you choose A0 even though A1 achieved much higher average precision?

A1 achieved much higher validation average precision, approximately 0.9696, compared with roughly 0.39 for the tuned A0 experiment.

However, A1 included balance-derived features. The simulator's balance behavior raised data-reliability concerns, so I did not consider those features sufficiently trustworthy for the primary deployable candidate.

A0 used transaction information, historical activity, and other features that were more defensible at the intended scoring moment.

My decision was based on more than predictive performance. Feature availability, reliability, and the likelihood that a relationship would remain valid outside the synthetic dataset also mattered.

I would describe A1 as a diagnostic experiment, not proof that A1 had formal target leakage.

### Q5. Why did you use `fraud_score × amount` to rank transactions?

The business objective emphasized fraudulent transaction value, not simply the number of fraudulent transactions detected.

A score-only ranking prioritizes transactions with the highest estimated fraud scores. Multiplying the score by transaction amount gives higher priority to transactions that combine a high fraud score with a larger amount.

I compared both approaches on validation data under the same alert budgets. The amount-weighted ranking captured more fraud value than score-only ranking in those experiments.

However, this is a ranking heuristic. It is not a calibrated expected-loss calculation because the fraud score was not established as a calibrated probability and transaction amount is only a proxy for value.

### Q6. Can the model's score of 0.90 be interpreted as a 90% probability of fraud?

Not automatically.

A classification score is not necessarily a calibrated probability. A score of 0.90 may indicate a high ranking, but it does not prove that 90% of transactions receiving that score are fraudulent.

To assess calibration, I would use a reliability diagram and metrics such as Brier score or log loss. I could then evaluate calibration methods such as Platt scaling or isotonic regression using an appropriate held-out temporal validation period.

I would not use the locked test set to choose or tune a calibration method.

### Q7. Is the model just learning transaction type?

Transaction type is an important feature. In the validation SHAP analysis, `type` had the highest population-weighted mean absolute SHAP value, approximately 2.3122.

This is understandable because observed fraud in PaySim is concentrated in TRANSFER and CASH_OUT transactions. Other transaction types had no observed fraud in this dataset.

I would not remove transaction type automatically. It is available at the scoring moment and contains useful predictive information.

However, I would investigate how dependent the model is on it by testing a type-only baseline, performing a type-ablation experiment, and evaluating model performance within each transaction type. The key question is whether the model learns meaningful transaction-level patterns beyond the synthetic dataset's transaction-type structure.

If removing type caused average precision to fall substantially, I would retain it provisionally while documenting the generalization risk and seeking evidence from more realistic data.

---

## 4. Evaluation and class imbalance

### Q8. Why did you use average precision instead of accuracy?

The dataset is extremely imbalanced. Only approximately 0.129% of transactions are fraudulent.

A model predicting every transaction as legitimate would achieve approximately 99.87% accuracy while detecting no fraud. That makes accuracy misleading as the primary model-selection metric.

Average precision summarizes the precision-recall trade-off and is more informative for evaluating a rare positive class.

The tuned A0 validation experiment achieved average precision of approximately 0.3902, compared with a validation fraud prevalence of approximately 0.6804%. This is a substantial improvement over the prevalence-level reference, but it does not by itself establish business value.

I therefore also evaluated precision, recall, fraud value captured, and performance at fixed alert budgets.

### Q9. Is an average precision score of approximately 0.39 good?

It is a strong result relative to the validation dataset's fraud prevalence of approximately 0.0068, but I would avoid describing it as universally good without context.

A random-ranking baseline would have expected average precision close to the positive-class prevalence. The tuned A0 result of approximately 0.3902 is far above that reference and also improved on the logistic baseline's average precision of approximately 0.1261.

However, average precision does not tell the business how much fraud value will be captured at a particular review capacity.

That is why I also compared the model and rule at fixed alert budgets and ultimately used the locked test business hurdle to make the final policy decision.

### Q10. Why did you compare class weighting and undersampling?

Both approaches address the imbalance between legitimate and fraudulent transactions, but they do so differently.

Class weighting keeps the training observations and increases the importance assigned to fraudulent cases. This can help the model focus on the minority class, but it can also change the model's behavior and score distribution.

Undersampling reduces the number of legitimate training examples. It can make training more manageable, but it removes information about legitimate transaction diversity and changes the class distribution seen during training.

In my validation experiments, the unweighted LightGBM baseline achieved average precision of approximately 0.3751, while the class-weighted experiment achieved approximately 0.0209 and the undersampled experiment approximately 0.3290.

The weighted experiment captured a higher share of fraud value than the other two in that particular comparison, but its precision and recall were much lower. The selected tuned A0 model used unweighted training.

I chose based on the full operating metrics rather than assuming that a more aggressive imbalance treatment must produce a better model.

---

## 5. Temporal validation and leakage

### Q11. Why did you purge steps 422–445 between training and validation?

The project used a 24-step label-delay assumption.

At the start of validation, step 446, labels were assumed to be available only through step 421. Therefore, steps 422–445 could not be included in training because their outcomes would not yet have matured at that scoring point.

I excluded those 24 steps to avoid training on labels that would not realistically have been available when the validation period began.

For the final test evaluation, I applied the same logic: the final refit cutoff was step 570, steps 571–594 were purged, and testing began at step 595.

This makes the evaluation more representative of a real chronological deployment process.

### Q12. Does having the same accounts in training and validation mean there is data leakage?

Not necessarily.

Account overlap is expected in a chronological transaction system. The same customer or destination can appear across different time periods in real operations.

The important requirement is that features for a transaction use only information available before that transaction. In this project, historical features used the previous 24 steps and excluded same-step activity because the within-step ordering was unknown.

A group-based split would answer a different question: how well does the model generalize to entirely unseen accounts?

That could be a useful additional experiment, but account overlap alone does not prove temporal leakage.

---

## 6. Drift, explainability, and model limitations

### Q13. Your PSI for destination transaction count is approximately 2.09. What would you do?

I would treat that as a major distribution-shift warning that requires investigation.

First, I would check whether the data pipeline, feature definitions, or transaction-generating process changed. Then I would examine the change by transaction type, time period, and other useful segments.

I would also monitor model performance once labels mature, including precision, recall, false-positive rates, fraud value captured, and alert volume.

PSI measures distribution change in a feature. It does not prove that model performance has degraded, and it does not by itself establish concept drift.

I would not retrain automatically based on PSI alone. I would investigate the cause and look for evidence that the change affects model behavior or business outcomes.

### Q14. What are the main limitations of this project?

There are several important limitations.

1. **Synthetic data:** PaySim is a simulator, so patterns learned from it may not represent real-world mobile-money fraud.
2. **Transaction-type dependence:** Fraud is concentrated in TRANSFER and CASH_OUT in this dataset, and transaction type is the model's most influential feature.
3. **Balance-field reliability:** Balance-derived features showed suspicious simulator behavior, so they were excluded from the primary A0 candidate.
4. **Value proxy:** Transaction amount is not verified financial loss or money actually prevented.
5. **Temporal distribution shift:** Fraud prevalence increased from approximately 0.0793% in effective training to 0.6804% in validation and 1.3384% in test.
6. **Limited test period:** The final test period contained only 123,580 transactions, and the exploratory day-block bootstrap had only five complete days, so its uncertainty estimate should be interpreted cautiously.
7. **Test exposure during exploration:** Some exploratory analysis used the full dataset, including the eventual test period. The test set was not used for model training or policy selection, but this exposure should be disclosed as a limitation.

The project demonstrates a reproducible analytical workflow and a disciplined evaluation process. It does not establish that the model is production-ready.

---

## 7. Final test result and recommendation

The locked test set covered steps 595–743 and contained 123,580 transactions. Both policies selected 2,897 alerts under the same capacity.

| Metric | Best rule | A0 model |
|---|---:|---:|
| Alerts selected | 2,897 | 2,897 |
| True positives | 818 | 1,043 |
| False positives | 2,079 | 1,854 |
| False negatives | 836 | 611 |
| Precision | 28.24% | 36.00% |
| Recall | 49.46% | 63.06% |
| Fraud value captured | 2.510B | 2.579B |
| Dollar-weighted recall | 92.20% | 94.74% |

The model captured approximately 69.19 million more transaction-value units than the rule, corresponding to a 2.76% uplift.

The required 5% hurdle was approximately 2.635B in fraud value captured. The model fell short by approximately 56.31 million.

**Recommendation:** Retain the TRANSFER/CASH_OUT amount-based rule as the selected policy. Keep the A0 LightGBM model as a research challenger and evaluate future improvements on new, untouched data. Do not use the locked test set for further model tuning or policy selection.

---

## 8. A concise project explanation for interviews

I built a fraud-detection analytics project using the PaySim mobile-money transaction dataset, which contains more than 6.3 million transactions and a highly imbalanced fraud class.

I started with exploratory analysis to understand fraud distribution, transaction types, transaction amounts, temporal patterns, and the limitations of the simulator. I then engineered past-only features and built a chronological modeling pipeline with a 24-step label-delay purge to reduce temporal leakage risk.

I compared a logistic baseline, LightGBM experiments, and rule-based policies. Rather than selecting a model based on average precision alone, I evaluated policies under a fixed alert capacity and used fraudulent transaction value captured as the main business-oriented metric.

On the locked test set, the LightGBM model detected 225 more fraud cases than the best rule and improved precision from 28.24% to 36.00%. However, its fraud-value uplift was only 2.76%, below the predefined 5% hurdle.

I therefore retained the rule as the selected policy and kept the model as a challenger. The main lesson was that predictive performance and business value are related, but they are not the same thing.

---

## 9. Interview reminders

- Explain the business objective before discussing algorithms.
- State the alert budget whenever comparing policies.
- Distinguish average precision from precision at the operating threshold.
- Do not call transaction amount actual financial loss.
- Do not describe A1's high score as proof of leakage; explain the balance-feature reliability concern.
- Do not claim that PSI proves performance degradation.
- Do not claim that the model is production-ready.
- Do not use the locked test set for another round of tuning.
- Be prepared to explain why the final decision favored the rule even though the model detected more fraud cases.

**The central message:** I did not choose the model simply because it was more sophisticated or detected more cases. I evaluated it against a predefined business requirement and kept the existing rule when the model did not meet that requirement.
