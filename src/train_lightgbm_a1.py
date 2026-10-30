
"""Controlled A0 vs A1 LightGBM comparison for PaySim.

A0 = original A0 features plus the existing ratio-missingness indicator.
A1 = A0 plus four pre-transaction balance features.

Both models use the same training cutoff, validation rows, parameters,
random seed, and alert budget. Post-transaction balances are excluded.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.train_lightgbm_baseline import evaluate_predictions


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "paysim_a1_features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Preserve the established temporal evaluation protocol.
TRAIN_END_STEP = 445
LABEL_DELAY_STEPS = 24
EFFECTIVE_TRAIN_END_STEP = TRAIN_END_STEP - LABEL_DELAY_STEPS

VALIDATION_START_STEP = 446
VALIDATION_END_STEP = 594

RANDOM_SEED = 42

# Best configuration from corrected A0 tuning trial 10.
MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "max_bin": 127,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}

CATEGORICAL_FEATURES = ["hour_of_day", "type"]

A0_BASE_FEATURES = [
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

A0_RATIO_MISSINGNESS = (
    "amount_to_sender_past_mean_ratio_is_missing"
)

A0_FEATURES = A0_BASE_FEATURES + [A0_RATIO_MISSINGNESS]

A1_BALANCE_FEATURES = [
    "oldbalanceOrg",
    "oldbalanceDest",
    "drain_ratio",
    "amount_to_oldbalanceDest_ratio",
]

A1_FEATURES = A0_FEATURES + A1_BALANCE_FEATURES

REQUIRED_COLUMNS = [
    "transaction_id",
    "step",
    "isFraud",
    "amount",
    *A0_BASE_FEATURES,
    *A1_BALANCE_FEATURES,
]

BEST_RULE_FRAUD_VALUE = 1_567_743_893.0
REQUIRED_VALUE_FOR_5PCT_UPLIFT = BEST_RULE_FRAUD_VALUE * 1.05


def load_data():
    """Load A1, derive A0/A1 feature matrices, and create fixed time splits."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"A1 Parquet file not found: {INPUT_PATH}"
        )

    frame = pd.read_parquet(INPUT_PATH, columns=REQUIRED_COLUMNS)

    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"A1 input is missing columns: {sorted(missing_columns)}"
        )

    # Guard against row identity problems before fitting.
    if frame["transaction_id"].isna().any():
        raise ValueError("Missing transaction IDs in A1 table.")

    if frame["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in A1 table.")

    frame["step"] = pd.to_numeric(
        frame["step"], errors="raise"
    ).astype("int16")

    frame["isFraud"] = pd.to_numeric(
        frame["isFraud"], errors="raise"
    ).astype("int8")

    frame["amount"] = pd.to_numeric(
        frame["amount"], errors="raise"
    ).astype("float64")

    # Keep the categorical data types consistent for both experiments.
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("category")

    # Preserve missing values and normalize infinities.
    ratio_columns = [
        "amount_to_sender_past_mean_ratio",
        "drain_ratio",
        "amount_to_oldbalanceDest_ratio",
    ]

    for col in ratio_columns:
        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    # Derive the same missingness feature used by the A0 baseline.
    frame[A0_RATIO_MISSINGNESS] = (
        frame["amount_to_sender_past_mean_ratio"]
        .isna()
        .astype("int8")
    )

    # Convert numeric model inputs consistently. NaNs are intentional;
    # LightGBM can handle them without imputing zero.
    all_features = A1_FEATURES

    for col in all_features:
        if col in CATEGORICAL_FEATURES:
            continue

        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        if col != A0_RATIO_MISSINGNESS:
            frame[col] = frame[col].astype("float32")

    train_mask = frame["step"] <= EFFECTIVE_TRAIN_END_STEP

    validation_mask = (
        (frame["step"] >= VALIDATION_START_STEP)
        & (frame["step"] <= VALIDATION_END_STEP)
    )

    if not train_mask.any():
        raise ValueError("No eligible training rows.")
    if not validation_mask.any():
        raise ValueError("No validation rows.")

    # Create both feature sets from exactly the same transactions.
    X_train_a0 = frame.loc[
        train_mask, A0_FEATURES
    ].reset_index(drop=True)

    X_train_a1 = frame.loc[
        train_mask, A1_FEATURES
    ].reset_index(drop=True)

    y_train = frame.loc[
        train_mask, "isFraud"
    ].to_numpy(dtype=np.int8, copy=True)

    X_val_a0 = frame.loc[
        validation_mask, A0_FEATURES
    ].reset_index(drop=True)

    X_val_a1 = frame.loc[
        validation_mask, A1_FEATURES
    ].reset_index(drop=True)

    validation_meta = frame.loc[
        validation_mask, ["step", "isFraud", "amount"]
    ].reset_index(drop=True)

    # Guard against accidental differences between the A0/A1 samples.
    if len(X_train_a0) != len(X_train_a1):
        raise RuntimeError("A0/A1 training row counts differ.")

    if len(X_val_a0) != len(X_val_a1):
        raise RuntimeError("A0/A1 validation row counts differ.")

    del frame, train_mask, validation_mask
    gc.collect()

    if int(y_train.sum()) == 0:
        raise ValueError("No fraud cases in eligible training data.")

    return (
        X_train_a0,
        X_train_a1,
        y_train,
        X_val_a0,
        X_val_a1,
        validation_meta,
    )


def train_and_evaluate(
    name: str,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    validation_meta: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Fit one model and evaluate it with the shared alert-budget logic."""
    categorical = [
        col for col in CATEGORICAL_FEATURES if col in features
    ]

    model = lgb.LGBMClassifier(**MODEL_PARAMS)

    model.fit(
        X_train[features],
        y_train,
        categorical_feature=categorical,
    )

    scores = model.predict_proba(X_val[features])[:, 1]

    metrics = evaluate_predictions(
        strategy=name,
        scores=scores,
        y_train=y_train,
        fit_rows=len(y_train),
        fit_fraud=int(y_train.sum()),
        validation_meta=validation_meta,
    )

    metrics["feature_count"] = len(features)
    metrics["features"] = ";".join(features)
    metrics["training_end_step"] = EFFECTIVE_TRAIN_END_STEP
    metrics["validation_start_step"] = VALIDATION_START_STEP
    metrics["validation_end_step"] = VALIDATION_END_STEP

    value_uplift_pct = (
        100.0
        * (
            metrics["fraud_value_caught"]
            / BEST_RULE_FRAUD_VALUE
            - 1.0
        )
    )

    metrics["value_uplift_vs_best_rule_pct"] = value_uplift_pct
    metrics["required_value_for_5pct_uplift"] = (
        REQUIRED_VALUE_FOR_5PCT_UPLIFT
    )
    metrics["meets_5pct_value_hurdle"] = (
        metrics["fraud_value_caught"]
        >= REQUIRED_VALUE_FOR_5PCT_UPLIFT
    )

    model_path = MODELS_DIR / f"lightgbm_{name}_comparison.joblib"
    joblib.dump(model, model_path, compress=3)
    metrics["model_path"] = str(model_path.relative_to(ROOT))

    importance = pd.DataFrame({
        "model": name,
        "feature": features,
        "importance_gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
    }).sort_values("importance_gain", ascending=False)

    del model, scores
    gc.collect()

    return metrics, importance


def main() -> None:
    (
        X_train_a0,
        X_train_a1,
        y_train,
        X_val_a0,
        X_val_a1,
        validation_meta,
    ) = load_data()

    print("A0 vs A1 controlled comparison")
    print("------------------------------")
    print(f"Training cutoff: step {EFFECTIVE_TRAIN_END_STEP}")
    print(
        f"Purged label-delay gap: steps "
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
        f"Validation fraud cases: "
        f"{int(validation_meta['isFraud'].sum()):,}"
    )
    print(f"A0 feature count: {len(A0_FEATURES)}")
    print(f"A1 feature count: {len(A1_FEATURES)}")
    print(f"Best rule fraud value: {BEST_RULE_FRAUD_VALUE:,.2f}")
    print(
        "Value required for 5% improvement: "
        f"{REQUIRED_VALUE_FOR_5PCT_UPLIFT:,.2f}"
    )

    all_results = []
    all_importances = []

    experiments = [
        (
            "a0",
            A0_FEATURES,
            X_train_a0,
            X_val_a0,
        ),
        (
            "a1",
            A1_FEATURES,
            X_train_a1,
            X_val_a1,
        ),
    ]

    for name, features, X_train, X_val in experiments:
        print(f"\n{'=' * 65}")
        print(f"Training model: {name.upper()}")
        print(f"Feature count: {len(features)}")
        print(f"{'=' * 65}")

        metrics, importance = train_and_evaluate(
            name=name,
            features=features,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            validation_meta=validation_meta,
        )

        all_results.append(metrics)
        all_importances.append(importance)

        print(f"Average precision: {metrics['average_precision']:.6f}")
        print(f"True positives: {metrics['true_positives']:,}")
        print(f"False positives: {metrics['false_positives']:,}")
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
        print(
            "Value uplift vs best rule: "
            f"{metrics['value_uplift_vs_best_rule_pct']:+.3f}%"
        )

    result_df = pd.DataFrame(all_results)
    importance_df = pd.concat(all_importances, ignore_index=True)

    metrics_csv = REPORTS_DIR / "lightgbm_a0_vs_a1_metrics.csv"
    metrics_json = REPORTS_DIR / "lightgbm_a0_vs_a1_metrics.json"
    importance_csv = REPORTS_DIR / "lightgbm_a0_vs_a1_importance.csv"

    result_df.to_csv(metrics_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)

    report = {
        "comparison": "A0 vs A1",
        "input": str(INPUT_PATH.relative_to(ROOT)),
        "effective_training_end_step": EFFECTIVE_TRAIN_END_STEP,
        "label_delay_steps": LABEL_DELAY_STEPS,
        "validation_steps": [
            VALIDATION_START_STEP,
            VALIDATION_END_STEP,
        ],
        "model_parameters": MODEL_PARAMS,
        "a0_features": A0_FEATURES,
        "a1_features": A1_FEATURES,
        "best_rule_fraud_value": BEST_RULE_FRAUD_VALUE,
        "required_value_for_5pct_uplift": REQUIRED_VALUE_FOR_5PCT_UPLIFT,
        "results": all_results,
        "notes": [
            "A0 and A1 are trained on identical training rows.",
            "A0 and A1 are evaluated on identical validation rows.",
            "Post-transaction balance features are excluded.",
            "LightGBM handles undefined ratio values as missing.",
            "Validation is used for comparison; the locked test is not scored.",
            "The 5% hurdle is a business criterion, not a statistical significance test.",
        ],
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    display_columns = [
        "strategy",
        "feature_count",
        "average_precision",
        "true_positives",
        "false_positives",
        "precision_pct",
        "recall_pct",
        "fraud_value_caught",
        "dollar_weighted_recall_pct",
        "value_uplift_vs_best_rule_pct",
        "meets_5pct_value_hurdle",
    ]

    print("\nFinal A0 vs A1 comparison")
    print(result_df[display_columns].to_string(index=False))

    print(f"\nSaved metrics: {metrics_csv.relative_to(ROOT)}")
    print(f"Saved metrics JSON: {metrics_json.relative_to(ROOT)}")
    print(f"Saved feature importance: {importance_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
