
# Fraud Detection (PaySim)

## Project Overview

This project investigates mobile-money fraud detection using the
PaySim synthetic financial transaction dataset.

The objective is to evaluate fraud-detection policies in terms of
fraudulent transaction value identified, operational review capacity,
false positives, and modeled business costs.

## Business Decision

Determine which transactions should be approved, blocked, subjected
to additional verification, or sent for manual review before approval.

## Scoring Moment

Before transaction approval.

The deployable model must use only information available when the
decision is made. Features that depend on future events or
post-transaction information must not be used for pre-approval scoring.

## Dataset and Limitations

PaySim is synthetic data. The findings describe performance under
the simulator and do not establish real-world fraud-detection
performance or actual money saved.

Raw and processed datasets are excluded from Git.

## Repository Structure

- `src/`: reusable Python modules
- `sql/`: DuckDB queries
- `tests/`: automated tests
- `notebooks/`: analysis and experiments
- `reports/`: protocols, findings, and case study
- `data/raw/`: original downloaded data
- `data/processed/`: transformed data
- `models/`: saved model artifacts

## Environment Setup

Use Python 3.12 for the initial project environment.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Install the locked dependencies after setting up the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproducibility

- Dependency versions are recorded in `requirements.txt`.
- Random seeds and evaluation protocols will be documented.
- Data retrieval metadata and validation results will be recorded.
- The final workflow must be reproducible from a fresh clone.

## Evaluation Principles

- Use time-based splits rather than random transaction splits.
- Prevent information leakage.
- Compare predictive models against simpler rules.
- Select the operating policy on validation data.
- Evaluate the locked test period only once.

## Project Status

- Phase 0: business framing and evaluation criteria drafted.
- Phase 1: project setup and data engineering in progress.

## Results

Results and recommendations will be added as the project progresses.
