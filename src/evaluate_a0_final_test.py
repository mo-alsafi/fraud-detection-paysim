
"""One-time final test evaluation of the frozen A0 alert policy.

Frozen model:
- LightGBM Trial 10 configuration from the corrected A0 tuning run.

Frozen policy:
- A0 fraud score * transaction amount, top 500 alerts per 24 steps.

Comparator:
- TRANSFER/CASH_OUT, ranked by transaction amount, same alert budget.

Final refit:
- Train using labels available through step 570.
- Purge steps 571-594 before scoring test steps 595-743.

Important:
- Do not run if reports/test_evaluation_lock.json already exists.
- The locked test must not be used for additional model/policy selection.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LOCK_PATH = REPORTS_DIR / "test_evaluation_lock.json"
MODEL_PATH = MODELS_DIR / "lightgbm_a0_final_test_model.joblib"
REPORT_PATH = REPORTS_DIR / "a0_final_test_evaluation.json"
METRICS_CSV = REPORTS_DIR / "a0_final_test_evaluation.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen temporal protocol
TEST_START_STEP = 595
TEST_END_STEP = 743
LABEL_DELAY_STEPS = 24
TRAIN_CUTOFF_STEP = TEST_START_STEP - 1 - LABEL_DELAY_STEPS
PURGE_START_STEP = TRAIN_CUTOFF_STEP + 1
PURGE_END_STEP = TEST_START_STEP - 1

# Frozen operating policy
ALERTS_PER_DAY = 500
STEPS_PER_DAY = 24
MANUAL_REVIEW_COST = 100.0

# Frozen Trial 10 configuration from corrected tuning.
MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "max_bin": 127,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}

CATEGORICAL_FEATURES = ["hour_of_day", "type"]

BASE_FEATURES = [
    "hour_of_day",
    "type",
    "log_amount",
    "destination_is_merchant",
    "sender_txn_count_prev_24_steps",
    "sender_amount_sum_prev_24_steps",
    "destination_txn_count_prev_24_steps",
    "destination_amount_sum_prev_24_steps",
    "is_first_time_destination_for_sender",
    "amount_to_sender_past_mean_ratio",
]

MISSINGNESS_FEATURE = "amount_to_sender_past_mean_ratio_is_missing"
FEATURES = BASE_FEATURES + [MISSINGNESS_FEATURE]

REQUIRED_COLUMNS = [
    "step",
    "isFraud",
    "amount",
    *BASE_FEATURES,
]


def load_train_and_test():
    """Load A0 data and create consistent training/test feature matrices."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"A0 feature table not found: {FEATURES_PATH}"
        )

    frame = pd.read_parquet(FEATURES_PATH, columns=REQUIRED_COLUMNS)

    frame["step"] = pd.to_numeric(
        frame["step"], errors="raise"
    ).astype("int16")
    frame["isFraud"] = pd.to_numeric(
        frame["isFraud"], errors="raise"
    ).astype("int8")
    frame["amount"] = pd.to_numeric(
        frame["amount"], errors="raise"
    ).astype("float64")

    train_mask = frame["step"] <= TRAIN_CUTOFF_STEP
    test_mask = (
        (frame["step"] >= TEST_START_STEP)
        & (frame["step"] <= TEST_END_STEP)
    )

    if not train_mask.any():
        raise ValueError("No training rows available.")
    if not test_mask.any():
        raise ValueError("No locked test rows available.")

    # Create category vocabularies from eligible training data only.
    for column in CATEGORICAL_FEATURES:
        train_categories = sorted(
            frame.loc[train_mask, column].dropna().unique().tolist()
        )

        unknown = (
            frame.loc[test_mask, column].notna()
            & ~frame.loc[test_mask, column].isin(train_categories)
        )

        if unknown.any():
            unseen = (
                frame.loc[test_mask & unknown, column]
                .dropna()
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Unexpected categories in test column {column}: {unseen}"
            )

        frame[column] = pd.Categorical(
            frame[column],
            categories=train_categories,
        )

    ratio_column = "amount_to_sender_past_mean_ratio"
    frame[ratio_column] = pd.to_numeric(
        frame[ratio_column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    frame[MISSINGNESS_FEATURE] = (
        frame[ratio_column].isna().astype("int8")
    )

    for column in FEATURES:
        if column in CATEGORICAL_FEATURES:
            continue

        if column == MISSINGNESS_FEATURE:
            frame[column] = frame[column].astype("int8")
            continue

        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        frame[column] = frame[column].astype("float32")

    X_train = frame.loc[train_mask, FEATURES].reset_index(drop=True)
    y_train = frame.loc[train_mask, "isFraud"].to_numpy(
        dtype=np.int8, copy=True
    )

    X_test = frame.loc[test_mask, FEATURES].reset_index(drop=True)

    test_meta = frame.loc[
        test_mask, ["step", "isFraud", "amount", "type"]
    ].reset_index(drop=True)

    del frame, train_mask, test_mask
    gc.collect()

    if np.unique(y_train).size != 2:
        raise ValueError("Training data does not contain both classes.")

    return X_train, y_train, X_test, test_meta


def select_alerts(
    priority: np.ndarray,
    eligible: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Apply the fixed daily budget, prorated for partial simulation days."""
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
            np.floor(ALERTS_PER_DAY * covered_steps / STEPS_PER_DAY)
        )
        total_capacity += daily_capacity

        if len(positions) == 0 or daily_capacity == 0:
            continue

        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority[positions],
            )
        )
        selected[positions[order[:daily_capacity]]] = True

    return selected, total_capacity


def score_policy(
    policy_name: str,
    selected: np.ndarray,
    metadata: pd.DataFrame,
    capacity: int,
) -> dict:
    """Compute fixed-test metrics for an already selected alert set."""
    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amount = metadata["amount"].to_numpy(dtype=np.float64)

    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))

    fraud_count = int(np.sum(y == 1))
    legitimate_count = int(np.sum(y == 0))
    alert_count = int(selected.sum())

    precision = tp / alert_count if alert_count else 0.0
    recall = tp / fraud_count if fraud_count else 0.0
    fpr = fp / legitimate_count if legitimate_count else 0.0

    total_fraud_value = float(amount[y == 1].sum())
    fraud_value_caught = float(amount[selected & (y == 1)].sum())

    weighted_recall = (
        fraud_value_caught / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    review_cost = alert_count * MANUAL_REVIEW_COST

    return {
        "policy": policy_name,
        "available_alert_capacity": int(capacity),
        "alerts_used": alert_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "false_positive_rate_pct": 100.0 * fpr,
        "fraud_value_caught": fraud_value_caught,
        "total_fraud_value": total_fraud_value,
        "dollar_weighted_recall_pct": 100.0 * weighted_recall,
        "manual_review_cost_proxy": review_cost,
        "net_benefit_proxy": fraud_value_caught - review_cost,
    }


def bootstrap_uplift_ci(
    metadata: pd.DataFrame,
    model_selected: np.ndarray,
    rule_selected: np.ndarray,
    replicates: int = 5000,
    seed: int = 20261029,
) -> dict:
    """Paired bootstrap over simulation-day groups.

    Partial first/last test-day groups are retained. Complete 24-step
    groups are resampled with replacement. Because the test period has
    only five complete days, treat these intervals as exploratory.
    """
    steps = metadata["step"].to_numpy(dtype=np.int32)
    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    amount = metadata["amount"].to_numpy(dtype=np.float64)
    day_ids = (steps - 1) // STEPS_PER_DAY

    groups = sorted(np.unique(day_ids).tolist())
    full_groups = []
    partial_groups = []

    for group in groups:
        group_steps = np.unique(steps[day_ids == group]).size
        if group_steps == STEPS_PER_DAY:
            full_groups.append(group)
        else:
            partial_groups.append(group)

    if len(full_groups) < 3:
        return {
            "method": "paired simulation-day block bootstrap",
            "status": "not_computed",
            "reason": "Fewer than three complete day blocks.",
            "complete_day_blocks": len(full_groups),
            "partial_day_blocks": len(partial_groups),
        }

    daily = {}

    for group in groups:
        group_mask = day_ids == group
        fraud_mask = y == 1

        daily[group] = {
            "model_caught": float(
                amount[group_mask & model_selected & fraud_mask].sum()
            ),
            "rule_caught": float(
                amount[group_mask & rule_selected & fraud_mask].sum()
            ),
            "fraud_value_total": float(
                amount[group_mask & fraud_mask].sum()
            ),
        }

    rng = np.random.default_rng(seed)
    uplift_samples = []
    difference_samples = []
    weighted_recall_difference_samples = []

    for _ in range(replicates):
        sampled_full = rng.choice(
            full_groups,
            size=len(full_groups),
            replace=True,
        )
        sampled_groups = partial_groups + sampled_full.tolist()

        model_caught = sum(
            daily[group]["model_caught"] for group in sampled_groups
        )
        rule_caught = sum(
            daily[group]["rule_caught"] for group in sampled_groups
        )
        total_fraud_value = sum(
            daily[group]["fraud_value_total"] for group in sampled_groups
        )

        if rule_caught > 0:
            uplift_samples.append(
                100.0 * (model_caught / rule_caught - 1.0)
            )

        difference_samples.append(model_caught - rule_caught)

        if total_fraud_value > 0:
            weighted_recall_difference_samples.append(
                100.0 * (model_caught - rule_caught)
                / total_fraud_value
            )

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "method": "paired simulation-day block bootstrap",
        "replicates": replicates,
        "seed": seed,
        "complete_day_blocks_resampled": len(full_groups),
        "partial_day_blocks_held_fixed": len(partial_groups),
        "uplift_vs_rule_pct_95pct_ci": interval(uplift_samples),
        "difference_in_value_caught_95pct_ci": interval(
            difference_samples
        ),
        "difference_in_dollar_weighted_recall_percentage_points_95pct_ci":
            interval(weighted_recall_difference_samples),
        "interpretation_note": (
            "Only five complete simulation-day groups are available. "
            "Intervals are exploratory, have low effective sample size, "
            "and do not include uncertainty from refitting the model."
        ),
    }


def main() -> None:
    # Refuse to rescore if the existing Phase 3 test lock is present.
    if LOCK_PATH.exists():
        raise RuntimeError(
            f"Test lock already exists: {LOCK_PATH.relative_to(ROOT)}. "
            "Do not delete the lock or rerun test evaluation. Inspect "
            "the existing lock/report first."
        )

    X_train, y_train, X_test, test_meta = load_train_and_test()

    print("Frozen final test protocol")
    print("--------------------------")
    print(f"Training labels available through: step {TRAIN_CUTOFF_STEP}")
    print(f"Purged steps: {PURGE_START_STEP}-{PURGE_END_STEP}")
    print(f"Locked test steps: {TEST_START_STEP}-{TEST_END_STEP}")
    print(f"Training rows: {len(y_train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Test rows: {len(test_meta):,}")
    print(
        "Scoring the locked test once. No further policy selection "
        "will be performed using these results."
    )

    model = lgb.LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    # Save the frozen final model before scoring.
    joblib.dump(model, MODEL_PATH, compress=3)

    scores = model.predict_proba(X_test)[:, 1]
    amount = test_meta["amount"].to_numpy(dtype=np.float64)
    transaction_types = test_meta["type"].astype(str).to_numpy()

    all_rows = np.ones(len(test_meta), dtype=bool)
    transfer_cashout = np.isin(
        transaction_types,
        ["TRANSFER", "CASH_OUT"],
    )

    # Frozen candidate policy: rank by score * amount.
    model_priority = scores * amount
    model_selected, model_capacity = select_alerts(
        priority=model_priority,
        eligible=all_rows,
        metadata=test_meta,
    )

    # Frozen rule comparator: rank eligible movement transactions by amount.
    rule_priority = amount
    rule_selected, rule_capacity = select_alerts(
        priority=rule_priority,
        eligible=transfer_cashout,
        metadata=test_meta,
    )

    if model_capacity != rule_capacity:
        raise RuntimeError("Model and rule alert capacities differ.")

    model_result = score_policy(
        policy_name="A0 fraud score x amount",
        selected=model_selected,
        metadata=test_meta,
        capacity=model_capacity,
    )
    rule_result = score_policy(
        policy_name="TRANSFER/CASH_OUT by amount",
        selected=rule_selected,
        metadata=test_meta,
        capacity=rule_capacity,
    )

    rule_value = rule_result["fraud_value_caught"]
    model_value = model_result["fraud_value_caught"]

    uplift_pct = (
        100.0 * (model_value / rule_value - 1.0)
        if rule_value > 0
        else None
    )

    required_value = rule_value * 1.05
    value_difference = model_value - rule_value

    # Alert counts should equal the capacity under both policies. Since
    # the review cost is identical, net-benefit difference equals value
    # difference under this deliberately simplified proxy.
    net_benefit_difference = (
        model_result["net_benefit_proxy"]
        - rule_result["net_benefit_proxy"]
    )

    ci = bootstrap_uplift_ci(
        metadata=test_meta,
        model_selected=model_selected,
        rule_selected=rule_selected,
    )

    # Persist metrics and lock only after a complete successful evaluation.
    results = [rule_result, model_result]
    metrics_df = pd.DataFrame(results)

    model_result["value_uplift_vs_rule_pct"] = uplift_pct
    model_result["required_value_for_5pct_uplift"] = required_value
    model_result["meets_5pct_value_hurdle"] = (
        model_value >= required_value
    )

    report = {
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_once",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "input_features": str(FEATURES_PATH.relative_to(ROOT)),
        "feature_set": "A0 only",
        "model_parameters": MODEL_PARAMS,
        "training_label_cutoff_step": TRAIN_CUTOFF_STEP,
        "purged_steps": [PURGE_START_STEP, PURGE_END_STEP],
        "test_steps": [TEST_START_STEP, TEST_END_STEP],
        "test_rows": int(len(test_meta)),
        "alerts_per_24_steps": ALERTS_PER_DAY,
        "manual_review_cost_per_alert_proxy": MANUAL_REVIEW_COST,
        "policy_frozen_before_test": {
            "model_policy": "A0 fraud score x amount",
            "rule_comparator": "TRANSFER/CASH_OUT by amount",
            "budget_per_24_steps": ALERTS_PER_DAY,
            "success_hurdle": "At least 5% more fraud value than rule",
        },
        "best_rule_fraud_value_caught": rule_value,
        "model_fraud_value_caught": model_value,
        "value_difference_vs_rule": value_difference,
        "value_uplift_vs_rule_pct": uplift_pct,
        "required_value_for_5pct_uplift": required_value,
        "meets_5pct_value_hurdle": model_value >= required_value,
        "net_benefit_proxy_difference": net_benefit_difference,
        "confidence_interval": ci,
        "results": results,
        "limitations": [
            "PaySim is synthetic; transaction amount is not verified loss avoided.",
            "Fraud score multiplied by amount is a ranking heuristic, not calibrated expected loss.",
            "The net-benefit proxy deducts only manual review costs.",
            "Five complete daily blocks make the bootstrap interval exploratory.",
            "Earlier full-dataset exploratory summaries included test-period aggregates.",
            "Do not choose another policy based on these test results.",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics_df.to_csv(METRICS_CSV, index=False)

    # Lock after the report and metrics have been written.
    lock = {
        "status": "completed_once",
        "scored_at_utc": report["scored_at_utc"],
        "test_steps": [TEST_START_STEP, TEST_END_STEP],
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "metrics_csv": str(METRICS_CSV.relative_to(ROOT)),
        "policy": "A0 fraud score x amount",
        "budget_per_24_steps": ALERTS_PER_DAY,
        "note": "Do not rerun or tune based on the locked test.",
    }
    with LOCK_PATH.open("w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)

    print("\nFinal locked test comparison")
    print("----------------------------")
    print(metrics_df.to_string(index=False))
    print(f"\nModel fraud value caught: {model_value:,.2f}")
    print(f"Rule fraud value caught: {rule_value:,.2f}")
    print(f"Value difference: {value_difference:,.2f}")
    print(f"Value uplift vs rule: {uplift_pct:+.3f}%")
    print(f"Required uplift target: {required_value:,.2f}")
    print(
        "Meets 5% value hurdle: "
        f"{model_value >= required_value}"
    )
    print(
        "Net-benefit proxy difference: "
        f"{net_benefit_difference:,.2f}"
    )
    print("\nPaired day-block bootstrap (exploratory)")
    print(json.dumps(ci, indent=2))
    print(f"\nSaved report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_CSV.relative_to(ROOT)}")
    print(f"Test lock: {LOCK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
