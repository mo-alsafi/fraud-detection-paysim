
"""Ablation study for unweighted LightGBM A0 fraud detection.

All variants use:
- Training steps 1-421 (24-step label-delay purge)
- Validation steps 446-594
- Identical LightGBM parameters and random seed
- The same 500-alerts-per-24-steps evaluation budget

This script never trains on the locked test period.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_STEP = 445
LABEL_DELAY_STEPS = 24
EFFECTIVE_TRAIN_END_STEP = TRAIN_END_STEP - LABEL_DELAY_STEPS

VALIDATION_START_STEP = 446
VALIDATION_END_STEP = 594

STEPS_PER_DAY = 24
ALERTS_PER_DAY = 500
RANDOM_SEED = 42

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

FULL_FEATURES = BASE_FEATURES + [MISSINGNESS_FEATURE]

FEATURE_SETS = {
    "full_a0": FULL_FEATURES,
    "no_sparse_ratio": [
        col for col in FULL_FEATURES
        if col not in {
            "amount_to_sender_past_mean_ratio",
            MISSINGNESS_FEATURE,
        }
    ],
    "no_sender_history": [
        col for col in FULL_FEATURES
        if col not in {
            "sender_txn_count_prev_24_steps",
            "sender_amount_sum_prev_24_steps",
            "is_first_time_destination_for_sender",
            "amount_to_sender_past_mean_ratio",
            MISSINGNESS_FEATURE,
        }
    ],
}

REQUIRED_COLUMNS = [
    "step",
    "isFraud",
    "amount",
    *BASE_FEATURES,
]


def load_data():
    """Load features, create the missingness flag, and split chronologically."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature Parquet not found: {FEATURES_PATH}"
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

    # Define the category vocabulary consistently for both periods.
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("category")

    # Normalize non-finite values to missing values.
    ratio_col = "amount_to_sender_past_mean_ratio"
    frame[ratio_col] = pd.to_numeric(
        frame[ratio_col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    # Missingness is calculated before fitting any model.
    frame[MISSINGNESS_FEATURE] = (
        frame[ratio_col].isna().astype("int8")
    )

    # Keep numeric feature dtypes consistent across experiments.
    for col in BASE_FEATURES:
        if col not in CATEGORICAL_FEATURES:
            frame[col] = pd.to_numeric(
                frame[col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            frame[col] = frame[col].astype("float32")

    train_mask = frame["step"] <= EFFECTIVE_TRAIN_END_STEP

    validation_mask = (
        (frame["step"] >= VALIDATION_START_STEP)
        & (frame["step"] <= VALIDATION_END_STEP)
    )

    if not train_mask.any():
        raise ValueError("No eligible training rows found.")
    if not validation_mask.any():
        raise ValueError("No validation rows found.")

    X_train = frame.loc[train_mask, FULL_FEATURES].reset_index(drop=True)
    y_train = frame.loc[train_mask, "isFraud"].to_numpy(
        dtype=np.int8, copy=True
    )

    X_validation = frame.loc[
        validation_mask, FULL_FEATURES
    ].reset_index(drop=True)

    validation_meta = frame.loc[
        validation_mask, ["step", "isFraud", "amount"]
    ].reset_index(drop=True)

    del frame, train_mask, validation_mask
    gc.collect()

    if y_train.sum() == 0:
        raise ValueError("Training data contains no fraud cases.")

    return X_train, y_train, X_validation, validation_meta


def select_alerts(
    scores: np.ndarray,
    validation_meta: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Select high-score alerts within a prorated daily alert budget."""
    steps = validation_meta["step"].to_numpy(dtype=np.int32)
    amounts = validation_meta["amount"].to_numpy(dtype=np.float64)

    day_ids = (steps - 1) // STEPS_PER_DAY
    selected = np.zeros(len(validation_meta), dtype=bool)
    total_capacity = 0

    for day_id in np.unique(day_ids):
        positions = np.flatnonzero(day_ids == day_id)
        covered_steps = np.unique(steps[positions]).size

        daily_capacity = int(
            np.floor(ALERTS_PER_DAY * covered_steps / STEPS_PER_DAY)
        )
        total_capacity += daily_capacity

        if daily_capacity == 0:
            continue

        # Highest score first; amount breaks any score ties.
        order = np.lexsort(
            (-amounts[positions], -scores[positions])
        )
        chosen = positions[order[:daily_capacity]]
        selected[chosen] = True

    return selected, total_capacity


def calculate_metrics(
    name: str,
    features: list[str],
    scores: np.ndarray,
    validation_meta: pd.DataFrame,
    training_rows: int,
    training_fraud_cases: int,
) -> dict:
    y_val = validation_meta["isFraud"].to_numpy(dtype=np.int8)
    amounts = validation_meta["amount"].to_numpy(dtype=np.float64)

    selected, capacity = select_alerts(scores, validation_meta)

    tp = int(np.sum(selected & (y_val == 1)))
    fp = int(np.sum(selected & (y_val == 0)))
    fn = int(np.sum((~selected) & (y_val == 1)))

    total_fraud_cases = int(np.sum(y_val == 1))
    total_legitimate_cases = int(np.sum(y_val == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_fraud_cases if total_fraud_cases else 0.0
    fpr = (
        fp / total_legitimate_cases
        if total_legitimate_cases
        else 0.0
    )

    total_fraud_value = float(amounts[y_val == 1].sum())
    fraud_value_caught = float(amounts[selected & (y_val == 1)].sum())
    weighted_recall = (
        fraud_value_caught / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    return {
        "experiment": name,
        "feature_count": len(features),
        "features": ";".join(features),
        "training_rows": training_rows,
        "training_fraud_cases": training_fraud_cases,
        "validation_rows": len(y_val),
        "validation_fraud_cases": total_fraud_cases,
        "average_precision": float(
            average_precision_score(y_val, scores)
        ),
        "available_alert_capacity": capacity,
        "alerts_used": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "false_positive_rate_pct": 100.0 * fpr,
        "fraud_value_caught": fraud_value_caught,
        "total_fraud_value": total_fraud_value,
        "dollar_weighted_recall_pct": 100.0 * weighted_recall,
    }


def main() -> None:
    X_train, y_train, X_val, validation_meta = load_data()

    print("Feature ablation experiment")
    print("---------------------------")
    print(f"Training cutoff: step {EFFECTIVE_TRAIN_END_STEP}")
    print(
        f"Purged label-delay steps: "
        f"{EFFECTIVE_TRAIN_END_STEP + 1}-{TRAIN_END_STEP}"
    )
    print(
        f"Validation steps: "
        f"{VALIDATION_START_STEP}-{VALIDATION_END_STEP}"
    )
    print(f"Training rows: {len(y_train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation_meta):,}")
    print(
        "Validation fraud cases: "
        f"{int(validation_meta['isFraud'].sum()):,}"
    )

    results = []
    importance_rows = []

    for experiment, features in FEATURE_SETS.items():
        print(f"\n{'=' * 65}")
        print(f"Experiment: {experiment}")
        print(f"Features ({len(features)}): {', '.join(features)}")
        print(f"{'=' * 65}")

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            max_bin=127,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            importance_type="gain",
        )

        categorical_features = [
            col for col in CATEGORICAL_FEATURES if col in features
        ]

        model.fit(
            X_train[features],
            y_train,
            categorical_feature=categorical_features,
        )

        scores = model.predict_proba(X_val[features])[:, 1]

        metrics = calculate_metrics(
            name=experiment,
            features=features,
            scores=scores,
            validation_meta=validation_meta,
            training_rows=len(y_train),
            training_fraud_cases=int(y_train.sum()),
        )

        model_path = MODELS_DIR / f"lightgbm_ablation_{experiment}.joblib"
        joblib.dump(model, model_path, compress=3)
        metrics["model_path"] = str(model_path.relative_to(ROOT))
        results.append(metrics)

        gain_importances = model.booster_.feature_importance(
            importance_type="gain"
        )

        for feature, gain in zip(features, gain_importances):
            importance_rows.append(
                {
                    "experiment": experiment,
                    "feature": feature,
                    "importance_gain": float(gain),
                }
            )

        print(f"Average precision: {metrics['average_precision']:.6f}")
        print(
            f"True positives: {metrics['true_positives']:,}; "
            f"false positives: {metrics['false_positives']:,}"
        )
        print(f"Precision: {metrics['precision_pct']:.4f}%")
        print(f"Recall: {metrics['recall_pct']:.4f}%")
        print(
            "Dollar-weighted recall: "
            f"{metrics['dollar_weighted_recall_pct']:.4f}%"
        )
        print(
            f"Fraud value caught: "
            f"{metrics['fraud_value_caught']:,.2f}"
        )
        print(f"Saved model: {model_path.relative_to(ROOT)}")

        del model, scores
        gc.collect()

    metrics_df = pd.DataFrame(results)
    importance_df = pd.DataFrame(importance_rows)

    metrics_csv = REPORTS_DIR / "lightgbm_a0_ablation_metrics.csv"
    metrics_json = REPORTS_DIR / "lightgbm_a0_ablation_metrics.json"
    importance_csv = REPORTS_DIR / "lightgbm_a0_ablation_importance.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    display_columns = [
        "experiment",
        "feature_count",
        "average_precision",
        "true_positives",
        "false_positives",
        "precision_pct",
        "recall_pct",
        "fraud_value_caught",
        "dollar_weighted_recall_pct",
    ]

    print("\nFinal ablation comparison")
    print(metrics_df[display_columns].to_string(index=False))
    print(f"\nSaved metrics: {metrics_csv.relative_to(ROOT)}")
    print(f"Saved metrics JSON: {metrics_json.relative_to(ROOT)}")
    print(f"Saved importances: {importance_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
