
"""Evaluate A0 alert policies at 100, 500, and 1,000 alerts per day.

Uses the locked temporal protocol's validation period only.
The locked test period is not scored.

Modeled net benefit proxy:
    fraud value caught - (manual review cost * number of alerts)

The value caught is a synthetic PaySim transaction-value proxy, not
verified real-world avoided loss. Alerts are assumed to receive manual
review; this script does not model automatic blocking or step-up costs.
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

BUDGETS_PER_DAY = [100, 500, 1000]
STEPS_PER_DAY = 24
MANUAL_REVIEW_COST = 100.0

BEST_RULE_FRAUD_VALUE_AT_500 = 1_567_743_893.0


def select_alerts(
    priority: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
    alerts_per_day: int,
) -> tuple[np.ndarray, int]:
    """Select the highest-priority eligible rows within each day budget."""
    steps = metadata["step"].to_numpy(dtype=np.int32)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)
    day_ids = (steps - 1) // STEPS_PER_DAY

    selected = np.zeros(len(metadata), dtype=bool)
    total_capacity = 0

    for day_id in np.unique(day_ids):
        day_mask = day_ids == day_id
        positions = np.flatnonzero(day_mask & eligible)

        covered_steps = np.unique(steps[day_mask]).size
        daily_capacity = int(
            np.floor(alerts_per_day * covered_steps / STEPS_PER_DAY)
        )
        total_capacity += daily_capacity

        if not len(positions) or daily_capacity <= 0:
            continue

        # Highest priority first, then amount, then stable row position.
        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority[positions],
            )
        )

        selected[positions[order[:daily_capacity]]] = True

    return selected, total_capacity


def evaluate_policy(
    name: str,
    priority: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
    transaction_types: np.ndarray,
    alerts_per_day: int,
) -> tuple[dict, np.ndarray]:
    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amounts = metadata["amount"].to_numpy(dtype=np.float64)

    selected, capacity = select_alerts(
        priority=priority,
        eligible=eligible,
        metadata=metadata,
        alerts_per_day=alerts_per_day,
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

    dollar_weighted_recall = (
        value_caught / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    alert_count = int(selected.sum())
    review_cost = alert_count * MANUAL_REVIEW_COST

    result = {
        "budget_per_day": alerts_per_day,
        "policy": name,
        "available_alert_capacity": capacity,
        "alerts_used": alert_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "false_positive_rate_pct": 100.0 * fpr,
        "fraud_value_caught": value_caught,
        "total_fraud_value": total_fraud_value,
        "dollar_weighted_recall_pct": 100.0 * dollar_weighted_recall,
        "manual_review_cost": review_cost,
        "modeled_net_benefit_proxy": value_caught - review_cost,
        "selected_transaction_types": json.dumps(
            pd.Series(transaction_types[selected])
            .value_counts()
            .to_dict(),
            sort_keys=True,
        ),
    }

    return result, selected


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run the corrected A0 tuning experiment first."
        )

    # This returns training features, validation features, validation
    # metadata, training labels, and the training summary.
    _, X_val, metadata, _, _ = load_data()

    model = joblib.load(MODEL_PATH)
    fraud_scores = model.predict_proba(X_val)[:, 1]

    amounts = metadata["amount"].to_numpy(dtype=np.float64)
    transaction_types = X_val["type"].astype(str).to_numpy()

    all_rows = np.ones(len(metadata), dtype=bool)
    transfer_cashout = np.isin(
        transaction_types,
        ["TRANSFER", "CASH_OUT"],
    )

    priorities = {
        "Best rule: TRANSFER/CASH_OUT by amount": amounts,
        "A0: fraud score": fraud_scores,
        "A0: fraud score x amount": fraud_scores * amounts,
    }

    eligibility = {
        "Best rule: TRANSFER/CASH_OUT by amount": transfer_cashout,
        "A0: fraud score": all_rows,
        "A0: fraud score x amount": all_rows,
        "A0: fraud score x amount, TRANSFER/CASH_OUT only": (
            transfer_cashout
        ),
        "A0: fraud score, TRANSFER/CASH_OUT only": transfer_cashout,
    }

    # Restricted versions are included to validate whether the type
    # restriction actually changes the selected alert set.
    priorities["A0: fraud score x amount, TRANSFER/CASH_OUT only"] = (
        fraud_scores * amounts
    )
    priorities["A0: fraud score, TRANSFER/CASH_OUT only"] = fraud_scores

    results = []
    overlap_diagnostics = []

    for budget in BUDGETS_PER_DAY:
        selected_masks = {}

        print(f"\n{'=' * 75}")
        print(f"ALERT BUDGET: {budget:,} per 24 steps")
        print(f"{'=' * 75}")

        budget_results = []

        for policy_name, priority in priorities.items():
            result, selected = evaluate_policy(
                name=policy_name,
                priority=priority,
                eligible=eligibility[policy_name],
                metadata=metadata,
                transaction_types=transaction_types,
                alerts_per_day=budget,
            )

            selected_masks[policy_name] = selected
            budget_results.append(result)
            results.append(result)

        # Check whether eligibility restrictions really leave the
        # selected transactions unchanged.
        comparison_pairs = [
            (
                "A0: fraud score",
                "A0: fraud score, TRANSFER/CASH_OUT only",
            ),
            (
                "A0: fraud score x amount",
                "A0: fraud score x amount, TRANSFER/CASH_OUT only",
            ),
        ]

        for unrestricted, restricted in comparison_pairs:
            left = selected_masks[unrestricted]
            right = selected_masks[restricted]

            union_count = int(np.sum(left | right))
            intersection_count = int(np.sum(left & right))
            overlap_pct = (
                100.0 * intersection_count / union_count
                if union_count else 100.0
            )

            overlap_diagnostics.append(
                {
                    "budget_per_day": budget,
                    "unrestricted_policy": unrestricted,
                    "restricted_policy": restricted,
                    "unrestricted_alerts": int(left.sum()),
                    "restricted_alerts": int(right.sum()),
                    "selected_alert_jaccard_pct": overlap_pct,
                    "selected_sets_identical": bool(
                        np.array_equal(left, right)
                    ),
                }
            )

        budget_df = pd.DataFrame(budget_results)

        # Compare every policy to the amount-prioritized rule at the
        # same daily alert budget.
        rule_value = float(
            budget_df.loc[
                budget_df["policy"]
                == "Best rule: TRANSFER/CASH_OUT by amount",
                "fraud_value_caught",
            ].iloc[0]
        )
        rule_net = float(
            budget_df.loc[
                budget_df["policy"]
                == "Best rule: TRANSFER/CASH_OUT by amount",
                "modeled_net_benefit_proxy",
            ].iloc[0]
        )

        for result in budget_results:
            result["fraud_value_uplift_vs_rule_pct"] = (
                100.0 * (
                    result["fraud_value_caught"] / rule_value - 1.0
                )
                if rule_value else 0.0
            )

            result["net_benefit_uplift_vs_rule"] = (
                result["modeled_net_benefit_proxy"] - rule_net
            )

            result["meets_5pct_value_hurdle"] = (
                result["fraud_value_caught"] >= rule_value * 1.05
            )

        print(
            budget_df[
                [
                    "policy",
                    "available_alert_capacity",
                    "alerts_used",
                    "true_positives",
                    "false_positives",
                    "fraud_value_caught",
                    "dollar_weighted_recall_pct",
                    "manual_review_cost",
                    "modeled_net_benefit_proxy",
                ]
            ].to_string(index=False)
        )

    results_df = pd.DataFrame(results)
    overlap_df = pd.DataFrame(overlap_diagnostics)

    results_csv = REPORTS_DIR / "a0_budget_sensitivity.csv"
    results_json = REPORTS_DIR / "a0_budget_sensitivity.json"
    overlap_csv = REPORTS_DIR / "a0_policy_overlap_diagnostics.csv"

    results_df.to_csv(results_csv, index=False)
    overlap_df.to_csv(overlap_csv, index=False)

    with results_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(MODEL_PATH.relative_to(ROOT)),
                "validation_steps": [446, 594],
                "training_label_cutoff_step": 421,
                "budgets_per_day": BUDGETS_PER_DAY,
                "manual_review_cost_per_alert": MANUAL_REVIEW_COST,
                "best_rule_fraud_value_at_500": (
                    BEST_RULE_FRAUD_VALUE_AT_500
                ),
                "results": results,
                "overlap_diagnostics": overlap_diagnostics,
                "notes": [
                    "All reported model results use A0 features only.",
                    "No post-transaction balance features are used.",
                    "Net benefit is a proxy under the stated cost assumptions.",
                    "Captured synthetic transaction value is not verified loss avoided.",
                    "No automatic blocking or extra-verification costs are modeled.",
                    "The locked test period is not scored.",
                ],
            },
            f,
            indent=2,
        )

    print("\nPolicy-overlap diagnostics")
    print(
        overlap_df.to_string(index=False)
    )

    print(f"\nSaved results: {results_csv.relative_to(ROOT)}")
    print(f"Saved JSON report: {results_json.relative_to(ROOT)}")
    print(f"Saved overlap diagnostics: {overlap_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
