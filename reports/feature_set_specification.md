
# Feature Set Specification — PaySim Fraud Detection

## Scoring moment

The system scores transactions before approval.

## A0 — Primary deployable candidate

Includes:
- Simulated hour of day
- Transaction type
- Log-transformed transaction amount
- Destination-is-merchant flag
- Sender and destination transaction counts over the preceding 24 steps
- Sender and destination transaction amount sums over the preceding 24 steps
- First-time destination-for-sender flag
- Current amount relative to the sender's past mean transaction amount

Account IDs are used as history keys, not direct numeric model features.

Historical transaction features must use steps strictly earlier than
the current transaction step. Transactions from the same step are
excluded from the current row's history because their within-step
ordering is unavailable.

## A1 — Diagnostic pre-transaction balance experiment

A1 includes every A0 feature, plus:
- oldbalanceOrg
- oldbalanceDest
- drain_ratio
- amount_to_oldbalanceDest_ratio

Ratios are defined only when the relevant denominator is positive.
Missing and undefined ratios must be handled explicitly.

A1 is diagnostic, not currently approved for deployment, because
PaySim's recorded balance behavior requires additional scrutiny.

## B — Leakage demonstration only

B includes all A1 features, plus:
- newbalanceOrig
- newbalanceDest

These columns describe post-transaction balances and are not allowed
for pre-approval deployment.

## Labels and existing rules

- isFraud is the target, never an input feature.
- isFlaggedFraud is a standalone benchmark, not automatically
  included as a feature.

## Label-delay rule

Any feature derived from historical fraud labels must use only
labels from steps at least 24 steps older than the step being scored.

For a transaction at step s, a label event at step t is eligible
only if t <= s - 24.

The 24-step delay is a project assumption representing one simulated
day. It is not a measured real-world fraud-investigation delay.

## Leakage tests

- Changing future rows must not change earlier transaction features.
- Changing future labels must not change earlier label-history features.
- Generated model columns must match the declared feature schema.
- Post-transaction balances and the target label are prohibited
  from deployable model inputs.

## Status

A0 is the initial deployable candidate. A1 is diagnostic and B is
strictly for demonstrating leakage effects.
