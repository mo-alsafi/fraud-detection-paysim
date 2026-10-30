
"""Compare alert queue policies using the approved A0 LightGBM model.

Policies:
1. Best rule: TRANSFER/CASH_OUT, ranked by amount.
2. A0: fraud score, all transaction types.
3. A0: fraud score * amount, all transaction types.
4. A0: fraud score, restricted to TRANSFER/CASH_OUT.
5. A0: fraud score * amount, restricted to TRANSFER/CASH_OUT.

The locked test period is not scored.
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

BEST_RULE_VALUE_FROM_SQL = 1_567_743_893.0


def select_alerts(
    priority_scores: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Rank eligible rows within each simulation day under the alert budget."""
    steps = metadata["step"].to_numpy(dtype=np.int32)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)
    day_ids = (steps - 1) // STEPS_PER_DAY

    selected = np.zeros(len(metadata), dtype=bool)
    total_capacity = 0

    for day_id in np.unique(day_ids):
        day_mask = day_ids == day_id
        positions = np.flatnonzero(day_mask & eligible)

        # Capacity depends on the validation steps covered by the day,
        # not on how many rows a given policy considers eligible.
        covered_steps = np.unique(steps[day_mask]).size
        daily_capacity = int(
            np.floor(ALERTS_PER_DAY * covered_steps / STEPS_PER_DAY)
        )
        total_capacity += daily_capacity

        if len(positions) == 0 or daily_capacity == 0:
            continue

        # Priority descending; amount descending; original row index
        # provides a deterministic final tie-breaker.
        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority_scores[positions],
            )
        )
        chosen = positions[order[:daily_capacity]]
        selected[chosen] = True

    return selected, total_capacity


def evaluate_policy(
    name: str,
    priority_scores: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)

    selected, capacity = select_alerts(
        priority_scores=priority_scores,
        eligible=eligible,
        metadata=metadata,
    )

    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))

    fraud_cases = int(np.sum(y == 1))
    legitimate_cases = int(np.sum(y == 0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / fraud_cases if fraud_cases else 0.0
    fpr = fp / legitimate_cases if legitimate_cases else 0.0

    total_fraud_value = float(amounts[y == 1].sum())
    value_caught = float(amounts[selected & (y == 1)].sum())

    weighted_recall = (
        value_caught / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    return {
        "policy": name,
        "available_alert_capacity": int(capacity),
        "alerts_used": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "false_positive_rate_pct": 100.0 * fpr,
        "fraud_value_caught": value_caught,
        "total_fraud_value": total_fraud_value,
        "dollar_weighted_recall_pct": 100.0 * weighted_recall,
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained A0 model not found: {MODEL_PATH}\n"
            "Run the corrected A0 tuning experiment first."
        )

    # Load exactly the model's approved A0 feature contract and
    # the corresponding validation rows.
    _, X_val, validation_meta, _, _ = load_data()

    model = joblib.load(MODEL_PATH)
    fraud_scores = model.predict_proba(X_val)[:, 1]

    amounts = validation_meta["amount"].to_numpy(dtype=np.float64)
    all_rows = np.ones(len(validation_meta), dtype=bool)

    # The same transaction types used by the best rule baseline.
    transaction_types = X_val["type"].astype(str).to_numpy()
    transfer_cashout = np.isin(
        transaction_types, ["TRANSFER", "CASH_OUT"]
    )

    # Risk-value priority score: useful as a ranking heuristic,
    # but not a calibrated dollar-loss estimate.
    fraud_value_priority = fraud_scores * amounts

    policies = [
        (
            "Best rule: TRANSFER/CASH_OUT by amount",
            amounts,
            transfer_cashout,
        ),
        (
            "A0 fraud score: all transaction types",
            fraud_scores,
            all_rows,
        ),
        (
            "A0 fraud score x amount: all transaction types",
            fraud_value_priority,
            all_rows,
        ),
        (
            "A0 fraud score: TRANSFER/CASH_OUT only",
            fraud_scores,
            transfer_cashout,
        ),
        (
            "A0 fraud score x amount: TRANSFER/CASH_OUT only",
            fraud_value_priority,
            transfer_cashout,
        ),
    ]

    results = []

    print("A0 alert-policy comparison")
    print("--------------------------")
    print(f"Validation rows: {len(validation_meta):,}")
    print(
        "Validation fraud cases: "
        f"{int(validation_meta['isFraud'].sum()):,}"
    )
    print(f"Alert budget: {ALERTS_PER_DAY} per 24 steps")
    print(f"Model: {MODEL_PATH.relative_to(ROOT)}")

    for name, priority, eligible in policies:
        result = evaluate_policy(
            name=name,
            priority_scores=priority,
            eligible=eligible,
            metadata=validation_meta,
        )
        results.append(result)

    rule_result = results[0]
    measured_rule_value = rule_result["fraud_value_caught"]
    required_value = measured_rule_value * 1.05

    for result in results:
        result["value_uplift_vs_rule_pct"] = (
            100.0
            * (
                result["fraud_value_caught"] / measured_rule_value
                - 1.0
            )
        )
        result["meets_5pct_value_hurdle"] = (
            result["fraud_value_caught"] >= required_value
        )

    result_df = pd.DataFrame(results)

    metrics_csv = REPORTS_DIR / "a0_alert_policy_comparison.csv"
    metrics_json = REPORTS_DIR / "a0_alert_policy_comparison.json"

    result_df.to_csv(metrics_csv, index=False)

    report = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "validation_steps": [446, 594],
        "training_label_cutoff_step": 421,
        "alerts_per_24_steps": ALERTS_PER_DAY,
        "best_rule_value_from_sql": BEST_RULE_VALUE_FROM_SQL,
        "measured_rule_value": measured_rule_value,
        "required_value_for_5pct_uplift": required_value,
        "results": results,
        "notes": [
            "Only approved A0 features are used for model scoring.",
            "The locked test period is not scored.",
            "Fraud score multiplied by amount is a ranking heuristic.",
            "The heuristic is not a calibrated expected-loss estimate.",
            "Validation results must not be presented as final test performance.",
        ],
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    display_columns = [
        "policy",
        "alerts_used",
        "true_positives",
        "false_positives",
        "precision_pct",
        "recall_pct",
        "fraud_value_caught",
        "dollar_weighted_recall_pct",
        "value_uplift_vs_rule_pct",
        "meets_5pct_value_hurdle",
    ]

    print("\nFinal policy comparison")
    print(result_df[display_columns].to_string(index=False))
    print(f"\nSaved metrics: {metrics_csv.relative_to(ROOT)}")
    print(f"Saved report: {metrics_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
