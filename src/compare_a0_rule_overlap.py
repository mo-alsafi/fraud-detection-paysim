
"""Compare the frozen A0 value-priority queue against the best rule.

Validation only:
- Model: fraud score * amount
- Rule: TRANSFER/CASH_OUT ranked by amount
- Capacity: 500 alerts per 24 steps

No locked-test data is loaded or scored.
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

OUTPUT_CSV = REPORTS_DIR / "a0_rule_overlap_groups.csv"
METRICS_CSV = REPORTS_DIR / "a0_rule_overlap_metrics.csv"
REPORT_JSON = REPORTS_DIR / "a0_rule_overlap_report.json"


def select_alerts(
    priority: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Apply the same prorated daily alert budget to a policy."""
    steps = metadata["step"].to_numpy(dtype=np.int32)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)
    day_ids = (steps - 1) // STEPS_PER_DAY

    selected = np.zeros(len(metadata), dtype=bool)
    capacity_total = 0

    for day_id in np.unique(day_ids):
        day_mask = day_ids == day_id
        positions = np.flatnonzero(day_mask & eligible)

        covered_steps = np.unique(steps[day_mask]).size

        daily_capacity = int(
            np.floor(
                ALERTS_PER_DAY
                * covered_steps
                / STEPS_PER_DAY
            )
        )

        capacity_total += daily_capacity

        if not len(positions) or daily_capacity == 0:
            continue

        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority[positions],
            )
        )

        selected[positions[order[:daily_capacity]]] = True

    return selected, capacity_total


def evaluate_selected(
    name: str,
    selected: np.ndarray,
    y: np.ndarray,
    amounts: np.ndarray,
    capacity: int,
) -> dict:
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))

    fraud_count = int(np.sum(y == 1))
    legitimate_count = int(np.sum(y == 0))

    alert_count = int(selected.sum())

    precision = tp / alert_count if alert_count else 0.0
    recall = tp / fraud_count if fraud_count else 0.0

    total_fraud_value = float(amounts[y == 1].sum())
    caught_fraud_value = float(amounts[selected & (y == 1)].sum())

    return {
        "policy": name,
        "capacity": capacity,
        "alerts": alert_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "fraud_value_caught": caught_fraud_value,
        "dollar_weighted_recall_pct": (
            100.0 * caught_fraud_value / total_fraud_value
            if total_fraud_value > 0 else 0.0
        ),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"A0 model not found: {MODEL_PATH}"
        )

    _, X_val, metadata, _, _ = load_data()

    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(X_val)[:, 1]

    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)

    types = X_val["type"].astype(str).to_numpy()

    eligible_all = np.ones(len(y), dtype=bool)
    eligible_rule = np.isin(types, ["TRANSFER", "CASH_OUT"])

    # Frozen model policy.
    model_priority = scores * amounts

    model_selected, model_capacity = select_alerts(
        priority=model_priority,
        eligible=eligible_all,
        metadata=metadata,
    )

    # Best rule: eligible movement transactions ranked by amount.
    rule_selected, rule_capacity = select_alerts(
        priority=amounts,
        eligible=eligible_rule,
        metadata=metadata,
    )

    if model_capacity != rule_capacity:
        raise RuntimeError(
            "Model and rule capacities do not match."
        )

    # Four mutually exclusive outcome groups.
    groups = np.select(
        [
            model_selected & rule_selected,
            model_selected & (~rule_selected),
            (~model_selected) & rule_selected,
        ],
        [
            "Both selected",
            "Model only",
            "Rule only",
        ],
        default="Neither selected",
    )

    comparison = pd.DataFrame(
        {
            "group": groups,
            "isFraud": y,
            "amount": amounts,
            "type": types,
            "fraud_score": scores,
        }
    )

    # Group-level case and value analysis.
    group_summary = (
        comparison.groupby("group", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            legitimate_transactions=(
                "isFraud",
                lambda s: int((s == 0).sum()),
            ),
            fraud_value=(
                "amount",
                lambda s: float(
                    s[comparison.loc[s.index, "isFraud"] == 1].sum()
                ),
            ),
            average_amount=("amount", "mean"),
            total_transaction_value=("amount", "sum"),
            average_fraud_score=("fraud_score", "mean"),
        )
        .reset_index()
    )

    total_fraud_value = float(amounts[y == 1].sum())

    group_summary["fraud_value_share_pct"] = (
        100.0 * group_summary["fraud_value"] / total_fraud_value
        if total_fraud_value > 0 else 0.0
    )

    # Also examine only fraudulent transactions and show which
    # policy catches them.
    fraud_group_summary = (
        comparison.loc[comparison["isFraud"] == 1]
        .groupby("group", observed=True)
        .agg(
            fraud_cases=("isFraud", "size"),
            fraud_value=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    model_metrics = evaluate_selected(
        "A0 score x amount",
        model_selected,
        y,
        amounts,
        model_capacity,
    )

    rule_metrics = evaluate_selected(
        "TRANSFER/CASH_OUT by amount",
        rule_selected,
        y,
        amounts,
        rule_capacity,
    )

    metrics_df = pd.DataFrame([rule_metrics, model_metrics])

    group_summary.to_csv(OUTPUT_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)

    report = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "validation_steps": [446, 594],
        "training_label_cutoff_step": 421,
        "alerts_per_24_steps": ALERTS_PER_DAY,
        "model_policy": "fraud score * amount",
        "rule_policy": "TRANSFER/CASH_OUT ranked by amount",
        "total_fraud_cases": int(y.sum()),
        "total_fraud_value": total_fraud_value,
        "group_summary": group_summary.to_dict(orient="records"),
        "fraud_group_summary": (
            fraud_group_summary.to_dict(orient="records")
        ),
        "policy_metrics": [rule_metrics, model_metrics],
        "notes": [
            "Validation only; the locked test remains closed.",
            "Groups are mutually exclusive by policy selection.",
            "A fraud case selected by both policies is not counted as unique value from either alone.",
            "The group analysis is descriptive, not a new policy-selection experiment.",
        ],
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Model versus rule overlap")
    print("-------------------------")
    print(f"Validation transactions: {len(y):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Alert capacity per policy: {model_capacity:,}")

    print("\nPolicy performance")
    print("------------------")
    print(metrics_df.to_string(index=False))

    print("\nMutually exclusive alert groups")
    print("--------------------------------")
    print(group_summary.to_string(index=False))

    print("\nFraud cases by exclusive group")
    print("------------------------------")
    print(fraud_group_summary.to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    print(f"Groups: {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Metrics: {METRICS_CSV.relative_to(ROOT)}")
    print(f"Report: {REPORT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
