
"""Validation error analysis for the frozen A0 fraud-value policy.

Uses:
- Selected A0 LightGBM model
- Validation steps 446-594
- Ranking score = fraud_score * transaction_amount
- 500 alerts per 24 simulation steps

The locked test period is not loaded or scored.

Outputs:
- Confusion summary
- Errors by transaction type
- Errors by amount band
- Errors by hour
- Highest-priority false positives
- Closest-to-cutoff false negatives
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_lightgbm_baseline import load_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_a0_tuned_best.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALERTS_PER_DAY = 500
STEPS_PER_DAY = 24

SEED = 42

OUTPUT_FILES = {
    "confusion": REPORTS_DIR / "a0_error_confusion.csv",
    "by_type": REPORTS_DIR / "a0_error_by_type.csv",
    "by_amount": REPORTS_DIR / "a0_error_by_amount_band.csv",
    "by_hour": REPORTS_DIR / "a0_error_by_hour.csv",
    "false_positives": REPORTS_DIR / "a0_false_positive_examples.csv",
    "false_negatives": REPORTS_DIR / "a0_false_negative_nearmisses.csv",
    "report": REPORTS_DIR / "a0_error_analysis_report.json",
}


def select_alerts(
    priority: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, float], int]:
    """Apply the frozen daily budget and return each day's cutoff."""
    steps = metadata["step"].to_numpy(dtype=np.int32)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)

    day_ids = (steps - 1) // STEPS_PER_DAY

    selected = np.zeros(len(metadata), dtype=bool)
    daily_cutoffs = {}
    total_capacity = 0

    for day_id in np.unique(day_ids):
        positions = np.flatnonzero(day_ids == day_id)
        covered_steps = np.unique(steps[positions]).size

        daily_capacity = int(
            np.floor(
                ALERTS_PER_DAY
                * covered_steps
                / STEPS_PER_DAY
            )
        )
        total_capacity += daily_capacity

        if daily_capacity == 0 or len(positions) == 0:
            continue

        # Same ranking and deterministic tie-breaks as the frozen policy.
        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority[positions],
            )
        )

        chosen = positions[order[:daily_capacity]]
        selected[chosen] = True

        if len(chosen):
            daily_cutoffs[int(day_id)] = float(
                priority[chosen].min()
            )

    return selected, daily_cutoffs, total_capacity


def summarize_outcomes(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize confusion outcomes within each supplied group."""
    result = (
        frame.groupby(
            group_columns + ["decision"],
            observed=True,
            dropna=False,
        )
        .agg(
            transactions=("isFraud", "size"),
            average_amount=("amount", "mean"),
            total_transaction_value=("amount", "sum"),
            average_fraud_score=("fraud_score", "mean"),
            fraud_value=("fraud_value", "sum"),
            selected_fraud_value=("selected_fraud_value", "sum"),
        )
        .reset_index()
    )

    return result


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected A0 model not found: {MODEL_PATH}"
        )

    # Reuse the established A0 validation features.
    _, X_val, metadata, _, _ = load_data()

    model = joblib.load(MODEL_PATH)
    fraud_scores = model.predict_proba(X_val)[:, 1]

    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)

    value_priority = fraud_scores * amounts

    selected, daily_cutoffs, capacity = select_alerts(
        priority=value_priority,
        metadata=metadata,
    )

    # Build an anonymized analysis frame.
    analysis = pd.DataFrame(
        {
            "step": metadata["step"].to_numpy(),
            "simulation_day": (
                (metadata["step"].to_numpy(dtype=np.int32) - 1)
                // STEPS_PER_DAY
            ),
            "type": X_val["type"].astype(str).to_numpy(),
            "hour_of_day": (
                X_val["hour_of_day"].astype(str).to_numpy()
            ),
            "amount": amounts,
            "isFraud": y,
            "fraud_score": fraud_scores,
            "value_priority": value_priority,
            "selected": selected,
        }
    )

    # Keep amount bands consistent with Phase 2 EDA.
    analysis["amount_band"] = pd.cut(
        analysis["amount"],
        bins=[-np.inf, 100, 1_000, 10_000, 100_000, np.inf],
        labels=[
            "<100",
            "100-<1,000",
            "1,000-<10,000",
            "10,000-<100,000",
            ">=100,000",
        ],
        right=False,
    )

    # Define the four confusion outcomes.
    conditions = [
        analysis["selected"] & (analysis["isFraud"] == 1),
        analysis["selected"] & (analysis["isFraud"] == 0),
        (~analysis["selected"]) & (analysis["isFraud"] == 1),
    ]

    analysis["decision"] = np.select(
        conditions,
        ["TP", "FP", "FN"],
        default="TN",
    )

    # Value fields make false-positive and fraud-value summaries clearer.
    analysis["fraud_value"] = np.where(
        analysis["isFraud"] == 1,
        analysis["amount"],
        0.0,
    )

    analysis["selected_fraud_value"] = np.where(
        analysis["selected"] & (analysis["isFraud"] == 1),
        analysis["amount"],
        0.0,
    )

    # Attach each transaction's daily alert cutoff.
    analysis["daily_priority_cutoff"] = (
        analysis["simulation_day"].map(daily_cutoffs)
    )

    # How far below its day's cutoff was each missed fraud transaction?
    analysis["priority_gap_pct"] = np.where(
        analysis["daily_priority_cutoff"] > 0,
        100.0
        * (
            analysis["daily_priority_cutoff"]
            - analysis["value_priority"]
        )
        / analysis["daily_priority_cutoff"],
        np.nan,
    )

    # -----------------------------------------------------------------
    # 1. Overall confusion analysis
    # -----------------------------------------------------------------

    confusion = (
        analysis.groupby("decision", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            average_amount=("amount", "mean"),
            average_fraud_score=("fraud_score", "mean"),
            total_transaction_value=("amount", "sum"),
            fraud_value=("fraud_value", "sum"),
            selected_fraud_value=("selected_fraud_value", "sum"),
        )
        .reset_index()
    )

    counts = (
        analysis["decision"]
        .value_counts()
        .to_dict()
    )

    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    tn = int(counts.get("TN", 0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # -----------------------------------------------------------------
    # 2. Break down errors by key dimensions
    # -----------------------------------------------------------------

    by_type = summarize_outcomes(
        analysis,
        ["type"],
    )

    by_amount = summarize_outcomes(
        analysis,
        ["amount_band"],
    )

    by_hour = summarize_outcomes(
        analysis,
        ["hour_of_day"],
    )

    # -----------------------------------------------------------------
    # 3. Highest-priority false positives
    # -----------------------------------------------------------------

    fp_examples = (
        analysis.loc[analysis["decision"] == "FP"]
        .sort_values(
            ["value_priority", "amount"],
            ascending=[False, False],
        )
        [
            [
                "step",
                "simulation_day",
                "type",
                "hour_of_day",
                "amount",
                "fraud_score",
                "value_priority",
                "amount_band",
            ]
        ]
        .head(25)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # 4. Fraud cases closest to the daily alert cutoff
    # -----------------------------------------------------------------

    fn_nearmisses = (
        analysis.loc[analysis["decision"] == "FN"]
        .sort_values(
            ["priority_gap_pct", "value_priority"],
            ascending=[True, False],
        )
        [
            [
                "step",
                "simulation_day",
                "type",
                "hour_of_day",
                "amount",
                "fraud_score",
                "value_priority",
                "daily_priority_cutoff",
                "priority_gap_pct",
                "amount_band",
            ]
        ]
        .head(25)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # 5. Persist the analysis
    # -----------------------------------------------------------------

    confusion.to_csv(OUTPUT_FILES["confusion"], index=False)
    by_type.to_csv(OUTPUT_FILES["by_type"], index=False)
    by_amount.to_csv(OUTPUT_FILES["by_amount"], index=False)
    by_hour.to_csv(OUTPUT_FILES["by_hour"], index=False)
    fp_examples.to_csv(
        OUTPUT_FILES["false_positives"],
        index=False,
    )
    fn_nearmisses.to_csv(
        OUTPUT_FILES["false_negatives"],
        index=False,
    )

    report = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "feature_set": "A0 only",
        "validation_steps": [446, 594],
        "validation_rows": int(len(analysis)),
        "alert_budget_per_24_steps": ALERTS_PER_DAY,
        "available_alert_capacity": int(capacity),
        "alerts_used": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "error_definition": {
            "FP": "Legitimate transaction selected for review",
            "FN": "Fraudulent transaction not selected",
        },
        "notes": [
            "Uses validation only, not the locked test period.",
            "False-negative near-misses are ranked by distance below "
            "their own simulation-day alert cutoff.",
            "False-positive examples are anonymized; account names are "
            "not included in the A0 feature table.",
            "The validation period was used in model/policy selection, "
            "so error patterns are exploratory.",
        ],
        "outputs": {
            key: str(path.relative_to(ROOT))
            for key, path in OUTPUT_FILES.items()
        },
    }

    with OUTPUT_FILES["report"].open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("A0 validation error analysis")
    print("----------------------------")
    print(f"Validation transactions: {len(analysis):,}")
    print(f"Available alert capacity: {capacity:,}")
    print(f"Alerts used: {int(selected.sum()):,}")
    print(f"True positives: {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True negatives: {tn:,}")
    print(f"Precision: {100.0 * precision:.4f}%")
    print(f"Recall: {100.0 * recall:.4f}%")

    print("\nErrors by transaction type")
    print("--------------------------")
    print(by_type.to_string(index=False))

    print("\nErrors by amount band")
    print("---------------------")
    print(by_amount.to_string(index=False))

    print("\nErrors by hour")
    print("--------------")
    print(by_hour.to_string(index=False))

    print("\nTop 25 false positives by value priority")
    print("----------------------------------------")
    print(fp_examples.to_string(index=False))

    print("\nTop 25 missed fraud cases nearest to daily cutoff")
    print("--------------------------------------------------")
    print(fn_nearmisses.to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    for key, path in OUTPUT_FILES.items():
        print(f"{key}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
