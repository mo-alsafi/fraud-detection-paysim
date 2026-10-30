
# PaySim Data Dictionary

## Purpose

This dictionary documents the 11 source columns, their meanings,
stored types, availability at the pre-approval scoring moment,
and their proposed treatment in the fraud-detection project.

The project scores transactions before approval. A feature is only
eligible for deployment if it is available at that moment and
passes the project's reliability and leakage audits.

| Column | DuckDB type | Meaning | Available before approval? | Modeling decision |
|---|---|---|---|---|
| step | INTEGER | Simulation time step; one step represents one simulated hour. | Yes, as the current transaction time. | Eligible for time-derived features, subject to the declared time mapping. |
| type | VARCHAR | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER. | Yes. | Eligible as a categorical feature. |
| amount | DOUBLE | Amount of the transaction. | Yes. | Eligible; amount transformations may be evaluated. |
| nameOrig | VARCHAR | Identifier of the customer initiating the transaction. | Yes. | Use as an account key for past-only history features; do not treat the raw identifier as a numeric quantity. |
| oldbalanceOrg | DOUBLE | Sender balance before the transaction. | Potentially yes in a real system, but its reliability in PaySim requires investigation. | Exclude from the deployable baseline until the balance audit; examine separately in the declared feature-set experiments. |
| newbalanceOrig | DOUBLE | Sender balance after the transaction. | No. It describes a post-transaction state. | Exclude from pre-approval prediction; retain only for auditing and leakage demonstrations. |
| nameDest | VARCHAR | Identifier of the receiving account. | Yes. | Use as an account key for past-only history features. Merchant/customer classification must follow the documented account convention and be validated. |
| oldbalanceDest | DOUBLE | Recipient balance before the transaction. | Potentially yes in a real system, but its reliability in PaySim requires investigation. | Exclude from the deployable baseline until the balance audit; examine separately in the declared feature-set experiments. |
| newbalanceDest | DOUBLE | Recipient balance after the transaction. | No. It describes a post-transaction state. | Exclude from pre-approval prediction; retain only for auditing and leakage demonstrations. |
| isFraud | UTINYINT | Ground-truth fraud label: 1 indicates fraud and 0 indicates non-fraud. | No. It is the outcome being predicted, not a decision-time feature. | Target label only. Never use as an input feature. |
| isFlaggedFraud | UTINYINT | Existing fraud-rule indicator supplied with the dataset. The documented rule flags qualifying large transfers. | Conditional: it is only available if the corresponding rule has run before the decision. | Evaluate as a standalone baseline. Do not automatically include it as a model feature; verify when it is generated and how it relates to the existing rule. |

## Important interpretation notes   

- Dataset availability does not prove that a field is reliable.
- The dataset publisher warns that fraudulent transactions are cancelled in the simulator and advises against using the four balance columns blindly.
- `newbalanceOrig` and `newbalanceDest` are post-transaction values and are not appropriate inputs for the pre-approval model.
- `isFraud` is the prediction target, not a model feature.
- `isFlaggedFraud` must be evaluated as an existing-rule baseline; its inclusion in a predictive model is a separate decision that requires justification.
- Account identifiers should be treated as keys for history construction, not as ordinary numerical measurements.
- Historical features must use only information available before the current transaction, subject to the project's label-delay protocol.
- All deployable features must pass the future-perturbation leakage tests defined by the evaluation protocol.
