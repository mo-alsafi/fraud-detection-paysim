
# PaySim Fraud Detection

A time-aware fraud-detection and alert-prioritization project using the PaySim synthetic mobile-money transaction dataset.

The project compares rule-based policies with machine-learning models under a fixed manual-review capacity. It focuses not only on predictive performance, but also on information leakage, temporal validation, operational trade-offs, explainability, and business acceptance criteria.

## 1. Business problem

The business decision is made **before transaction approval**.

The objective is to identify transactions that should receive additional attention, such as manual review or verification, before they are approved.

The project evaluates policies using:

- Fraud cases identified.
- False positives and missed fraud.
- Fraudulent transaction value captured.
- Alert capacity.
- Simplified modeled net benefit.

The primary operating scenario is 500 alerts per 24 simulation steps, with additional capacity sensitivity analysis.

## 2. Dataset

**Dataset:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset contains 6,362,620 transactions across simulation steps 1–743, with 8,213 observed fraud cases.

PaySim is synthetic. Its patterns do not establish how a model will perform on real payment transactions.

Transaction amount is used as a proxy for fraudulent value, not as verified financial loss or money recoverable by a financial institution.

Raw and processed data files are excluded from Git.

## 3. Project approach

The project follows a chronological, phase-based workflow:

| Phase | Focus | Status |
|---|---|---|
| 0 | Business framing and success criteria | Complete |
| 1 | Project setup and data engineering | Complete |
| 2 | Exploratory data analysis | Complete |
| 3 | Temporal evaluation protocol and leakage controls | Complete |
| 4 | Baseline rules and logistic regression | Complete |
| 5 | LightGBM training and model comparison | Complete |
| 6 | Model selection and locked test evaluation | Complete |
| 7 | Explainability, error analysis, and drift diagnostics | Complete |
| 8 | Inference, testing, and reproducibility packaging | In progress |

## 4. Key modeling decisions

### Scoring time

Transactions are scored before approval. The deployable candidate must rely only on information available at that decision point.

### Feature sets

- **A0:** Decision-time features that do not directly use PaySim balance columns. This is the approved deployable candidate.
- **A1:** A0 plus eligible pre-transaction balance features. Evaluated diagnostically because of data-reliability concerns.
- **B:** Includes post-transaction balance information. Diagnostic only and not suitable for pre-approval scoring.

The A0 feature contract is defined in `src/feature_contract.py`.

The final model uses 11 columns: the ten A0 features plus a missingness indicator for the historical sender-amount ratio.

### Temporal evaluation

The project uses chronological periods rather than random transaction splits.

**Validation model:**

- Effective training steps: 1–421.
- Purged label-delay gap: 422–445.
- Validation steps: 446–594.

**Final test model:**

- Training labels available through step 570.
- Purged steps: 571–594.
- Locked test steps: 595–743.

The final test was evaluated once. Its lock is recorded in `reports/test_evaluation_lock.json`.

An important limitation is that some early full-dataset exploratory summaries included statistics from the eventual test period. The test was not fully unseen during all project analysis.

## 5. Final model and operating policy

The final research challenger is a LightGBM binary classifier trained on A0 features.

The frozen model is:

`models/lightgbm_a0_final_test_model.joblib`

The alert-prioritization policy is:

`priority_score = fraud_score × transaction_amount`

This is a ranking heuristic, not a calibrated expected-loss estimate.

The selected model configuration came from Trial 10 of the bounded validation search:

| Parameter | Value |
|---|---:|
| Objective | Binary |
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

The model is intended to prioritize transactions for human review. It is not designed or approved to automatically approve or block payments.

## 6. Locked test results

The final test compared the frozen A0 score × amount policy with the TRANSFER/CASH_OUT-by-amount rule at the same alert capacity.

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

The model captured approximately **2.76% more fraudulent transaction value** than the rule.

The predeclared business hurdle was at least **5% more fraud value** at the same alert capacity.

**The model did not meet the hurdle. The rule remains the selected policy.**

The model is a promising research challenger: it identified more fraud cases and achieved higher precision and recall, but the increase in fraud value was not sufficient to satisfy the agreed acceptance criterion.

The paired simulation-day bootstrap interval was approximately 1.18%–4.86%. It is exploratory because only five complete daily blocks were available.

See `reports/a0_final_test_evaluation.json` and `reports/model_card.md` for the full results and limitations.

## 7. Explainability and diagnostics

The project includes:

- SHAP global and local explanation work.
- False-positive and false-negative analysis.
- Model-versus-rule overlap analysis.
- Alert-budget sensitivity analysis.
- Feature and prevalence drift diagnostics.

The validation analysis identified transaction type, destination activity, hour of day, and transaction amount among the most influential model features.

The diagnostics are documented in the `reports/` directory. SHAP explanations describe model behavior; they do not establish causation.

## 8. Repository structure

```text
fraud-detection-paysim/
│
├── data/
│   ├── raw/                 # Local source CSV; excluded from Git
│   └── processed/           # Local Parquet files; excluded from Git
│
├── models/                  # Local model binaries; excluded from Git
│
├── reports/                 # Protocols, metrics, diagnostics, model card
│
├── sql/                     # DuckDB analysis queries
│
├── src/
│   ├── feature_contract.py
│   ├── features.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── rules.py
│   ├── train_*.py
│   ├── tune_lightgbm_a0.py
│   ├── evaluate_a0_final_test.py
│   └── create_model_manifest.py
│
├── tests/
│   ├── test_evaluation.py
│   ├── test_feature_contract.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_logistic_baseline.py
│   ├── test_rules.py
│   └── test_validation.py
│
├── requirements.in
├── requirements.txt
└── README.md
```

## 9. Environment setup

The project environment was initially configured for Python 3.12.

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 10. Data preparation

The raw PaySim CSV is expected at:

`data/raw/PS_20174392719_1491204439457_log.csv`

The retrieval manifest records the dataset version, retrieval details, and source-file validation information.

After obtaining the CSV, record its provenance:

```powershell
python -m src.data_manifest data/raw/PS_20174392719_1491204439457_log.csv --retrieved-at-utc "<actual UTC retrieval timestamp>" --retrieval-method manual
```

Convert the CSV to Parquet:

```powershell
python -m src.convert_to_parquet
```

Validate the converted data:

```powershell
python -m src.validate_data
```

Build the A0 feature dataset:

```powershell
python -m src.features
```

The default feature output is:

`data/processed/paysim_a0_features.parquet`

These steps rebuild the data inputs. They do not reproduce the locked test evaluation.

## 11. Tests

Run the complete automated test suite from the project root:

```powershell
python -m pytest -q
```

The latest reported local result was **66 passed**.

The tests cover feature contracts, feature generation, evaluation logic, rules, and the inference interface.

## 12. Inference interface

`src/inference.py` provides the `PaySimA0Scorer` interface.

It validates the approved A0 feature schema, preserves the training-time missing-value behavior, loads the frozen model, produces fraud and priority scores, and applies the alert budget.

Example:

```python
import pandas as pd

from src.inference import PaySimA0Scorer

# The input must already contain past-only A0 features.
# Remove the target label before scoring.
feature_batch = pd.read_parquet(
    "data/processed/paysim_a0_features.parquet"
)

feature_batch = feature_batch.drop(
    columns=["isFraud"],
    errors="ignore",
)

scorer = PaySimA0Scorer()

scored_alerts = scorer.score_and_prioritize(
    feature_batch,
    daily_budget=500,
    steps_per_day=24,
)
```

The final model binary must be present locally at:

`models/lightgbm_a0_final_test_model.joblib`

The inference interface does not build raw transaction history features itself; the supplied A0 features must already have been engineered using past-only information.

## 13. Model and data artifacts

The repository intentionally excludes raw data, processed data, and saved model binaries through `.gitignore`.

A fresh clone therefore does not contain everything required to run inference.

The exact final model artifact must be supplied separately. Its checksum and feature contract are recorded in:

`reports/model_manifest.json`

The manifest can be regenerated locally after the required model and evaluation artifacts are available:

```powershell
python -m src.create_model_manifest
```

Do not substitute a different model binary while claiming it is the final tested model.

## 14. Important evaluation safeguard

The final test evaluation is a one-time procedure.

`src/evaluate_a0_final_test.py` contains a lock check that prevents ordinary repeated evaluation when the test lock exists.

**Do not delete `reports/test_evaluation_lock.json` or rerun the final test script.**

The locked test result must not be used for additional model selection or tuning.

## 15. Limitations and recommendation

This project is a research prototype using synthetic data.

Important limitations include:

1. PaySim may not represent real payment behavior or fraud mechanisms.
2. Transaction amount is not verified financial loss.
3. Fraud prevalence and some feature distributions shift across temporal periods.
4. The priority score is a heuristic rather than calibrated expected loss.
5. The simplified net-benefit proxy does not capture all review, false-blocking, verification, recovery, and customer-friction costs.
6. The locked test contains only a small number of complete simulation-day blocks.
7. Early exploratory summaries included some test-period aggregates.
8. No evidence yet establishes generalization to real payment data.

**Recommendation:** Keep the rule as the selected baseline and retain the A0 LightGBM model as a research challenger. Do not approve production deployment based on this synthetic-data evaluation alone.
