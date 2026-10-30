
"""Train LightGBM A0 baselines under a chronological split and alert budget.

Strategies:
1. Unweighted training on all eligible training rows.
2. Class-weighted training on all eligible training rows.
3. Negative-class undersampling at up to 20 legitimate rows per fraud row.

Validation is used for model evaluation and is not used to fit preprocessing.
The locked test period is not loaded into model fitting or evaluation.
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


# ---------------------------------------------------------------------
# 1. Paths and the predeclared evaluation protocol
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END_STEP = 445
VALIDATION_START_STEP = 446
VALIDATION_END_STEP = 594
LABEL_DELAY_STEPS = 24

# Purge the most recent 24 steps of the nominal training window.
EFFECTIVE_TRAIN_END_STEP = TRAIN_END_STEP - LABEL_DELAY_STEPS

STEPS_PER_DAY = 24
ALERTS_PER_DAY = 500
RANDOM_SEED = 42
UNDERSAMPLE_NEGATIVES_PER_POSITIVE = 20

CATEGORICAL_FEATURES = ["hour_of_day", "type"]

# The ten A0 feature-contract columns, plus a missingness indicator
# derived from amount_to_sender_past_mean_ratio.
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
FEATURE_COLUMNS = BASE_FEATURES + [MISSINGNESS_FEATURE]

REQUIRED_COLUMNS = [
    "step",
    "isFraud",
    "amount",
    *BASE_FEATURES,
]


# ---------------------------------------------------------------------
# 2. Load only the columns needed for this experiment
# ---------------------------------------------------------------------

def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {FEATURES_PATH}\n"
            "Build the A0 feature table before running this script."
        )

    frame = pd.read_parquet(FEATURES_PATH, columns=REQUIRED_COLUMNS)

    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype("int16")
    frame["isFraud"] = pd.to_numeric(
        frame["isFraud"], errors="raise"
    ).astype("int8")
    frame["amount"] = pd.to_numeric(
        frame["amount"], errors="raise"
    ).astype("float64")

    # Keep the categories consistent across training and validation.
    frame["hour_of_day"] = frame["hour_of_day"].astype("category")
    frame["type"] = frame["type"].astype("category")

    # Preserve missing values for LightGBM's native missing-value handling.
    ratio_col = "amount_to_sender_past_mean_ratio"
    frame[ratio_col] = pd.to_numeric(frame[ratio_col], errors="coerce")
    frame[ratio_col] = frame[ratio_col].replace([np.inf, -np.inf], np.nan)

    # A missingness indicator lets the model distinguish "no history"
    # from a numerical ratio value.
    frame[MISSINGNESS_FEATURE] = (
        frame[ratio_col].isna().astype("int8")
    )

    numeric_features = [
        col for col in FEATURE_COLUMNS
        if col not in CATEGORICAL_FEATURES
    ]
    for col in numeric_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
        frame[col] = frame[col].astype("float32")

    train_mask = frame["step"] <= EFFECTIVE_TRAIN_END_STEP
    val_mask = (
        (frame["step"] >= VALIDATION_START_STEP)
        & (frame["step"] <= VALIDATION_END_STEP)
    )

    if not train_mask.any():
        raise ValueError("No training rows found under the delayed-label cutoff.")
    if not val_mask.any():
        raise ValueError("No validation rows found in the configured step range.")

    X_train = frame.loc[train_mask, FEATURE_COLUMNS].reset_index(drop=True)
    y_train = frame.loc[train_mask, "isFraud"].to_numpy(
        dtype=np.int8, copy=True
    )

    validation_meta = frame.loc[
        val_mask, ["step", "isFraud", "amount"]
    ].reset_index(drop=True)

    # No rows from validation or test are used to fit the model.
    train_rows = len(y_train)
    train_fraud = int(y_train.sum())
    train_legitimate = int(train_rows - train_fraud)

    X_val = frame.loc[val_mask, FEATURE_COLUMNS].reset_index(drop=True)

    del frame, train_mask, val_mask
    gc.collect()

    if train_fraud == 0 or train_legitimate == 0:
        raise ValueError("Training data must contain both classes.")

    return X_train, X_val, validation_meta, y_train, pd.DataFrame(
        [
            {
                "training_rows": train_rows,
                "training_fraud_cases": train_fraud,
                "training_legitimate_cases": train_legitimate,
                "training_fraud_rate_pct": 100.0 * train_fraud / train_rows,
                "effective_training_end_step": EFFECTIVE_TRAIN_END_STEP,
            }
        ]
    )


# ---------------------------------------------------------------------
# 3. Select no more than the per-day alert capacity
# ---------------------------------------------------------------------

def select_alerts_at_daily_budget(
    scores: np.ndarray,
    validation_meta: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Select the highest-scoring transactions within each simulation day.

    Partial days receive a proportionate budget based on the number of
    validation steps represented in that day. Ties are broken by amount.
    """
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

        if daily_capacity <= 0:
            continue

        # Primary sort: score descending. Secondary sort: amount descending.
        order = np.lexsort(
            (-amounts[positions], -scores[positions])
        )
        chosen = positions[order[:daily_capacity]]
        selected[chosen] = True

    return selected, total_capacity


# ---------------------------------------------------------------------
# 4. Evaluate on the same validation population and capacity
# ---------------------------------------------------------------------

def evaluate_predictions(
    strategy: str,
    scores: np.ndarray,
    y_train: np.ndarray,
    fit_rows: int,
    fit_fraud: int,
    validation_meta: pd.DataFrame,
) -> dict:
    y_val = validation_meta["isFraud"].to_numpy(dtype=np.int8)
    amounts = validation_meta["amount"].to_numpy(dtype=np.float64)

    selected, capacity = select_alerts_at_daily_budget(
        scores, validation_meta
    )

    tp = int(np.sum(selected & (y_val == 1)))
    fp = int(np.sum(selected & (y_val == 0)))
    fn = int(np.sum((~selected) & (y_val == 1)))

    fraud_count = int(np.sum(y_val == 1))
    legitimate_count = int(np.sum(y_val == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / fraud_count if fraud_count else 0.0
    false_positive_rate = fp / legitimate_count if legitimate_count else 0.0

    total_fraud_value = float(amounts[y_val == 1].sum())
    fraud_value_caught = float(amounts[selected & (y_val == 1)].sum())
    dollar_weighted_recall = (
        fraud_value_caught / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    return {
        "strategy": strategy,
        "training_rows_used": int(fit_rows),
        "training_fraud_cases_used": int(fit_fraud),
        "validation_rows": int(len(y_val)),
        "validation_fraud_cases": fraud_count,
        "validation_fraud_rate_pct": float(100.0 * y_val.mean()),
        "average_precision": float(average_precision_score(y_val, scores)),
        "available_alert_capacity": int(capacity),
        "alerts_used": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": float(100.0 * precision),
        "recall_pct": float(100.0 * recall),
        "false_positive_rate_pct": float(100.0 * false_positive_rate),
        "fraud_value_caught": fraud_value_caught,
        "total_fraud_value": total_fraud_value,
        "dollar_weighted_recall_pct": float(100.0 * dollar_weighted_recall),
    }


# ---------------------------------------------------------------------
# 5. Train the three controlled LightGBM experiments
# ---------------------------------------------------------------------

def main() -> None:
    X_train, X_val, validation_meta, y_train, train_summary = load_data()
    y_val = validation_meta["isFraud"].to_numpy(dtype=np.int8)
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count

    print("\nTraining protocol")
    print("-----------------")
    print(f"Nominal train range: steps 1-{TRAIN_END_STEP}")
    print(f"Effective train cutoff: step {EFFECTIVE_TRAIN_END_STEP}")
    print(
        f"Purged label-delay gap: steps "
        f"{EFFECTIVE_TRAIN_END_STEP + 1}-{TRAIN_END_STEP}"
    )
    print(
        f"Validation range: steps "
        f"{VALIDATION_START_STEP}-{VALIDATION_END_STEP}"
    )
    print(f"Training rows eligible: {len(y_train):,}")
    print(f"Training fraud cases: {positive_count:,}")
    print(f"Validation rows: {len(y_val):,}")
    print(f"Validation fraud cases: {int(y_val.sum()):,}")
    print(f"Class-weight multiplier: {scale_pos_weight:,.2f}")

    strategies = [
        "unweighted",
        "class_weighted",
        "undersample_20_to_1",
    ]

    results: list[dict] = []
    importance_rows: list[dict] = []

    for strategy in strategies:
        print(f"\n{'=' * 70}\nTraining strategy: {strategy}\n{'=' * 70}")

        if strategy == "undersample_20_to_1":
            positive_positions = np.flatnonzero(y_train == 1)
            negative_positions = np.flatnonzero(y_train == 0)

            target_negative_count = min(
                len(negative_positions),
                len(positive_positions) * UNDERSAMPLE_NEGATIVES_PER_POSITIVE,
            )

            rng = np.random.default_rng(RANDOM_SEED)
            sampled_negative_positions = rng.choice(
                negative_positions,
                size=target_negative_count,
                replace=False,
            )

            fit_positions = np.concatenate(
                [positive_positions, sampled_negative_positions]
            )
            rng.shuffle(fit_positions)

            X_fit = X_train.iloc[fit_positions]
            y_fit = y_train[fit_positions]
            model_scale_pos_weight = 1.0

        else:
            X_fit = X_train
            y_fit = y_train
            model_scale_pos_weight = (
                scale_pos_weight if strategy == "class_weighted" else 1.0
            )

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
            scale_pos_weight=model_scale_pos_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            importance_type="gain",
        )

        print(f"Rows used to fit: {len(y_fit):,}")
        print(f"Fraud cases used to fit: {int(y_fit.sum()):,}")
        print(f"Fraud prevalence in fitted sample: {100.0 * y_fit.mean():.4f}%")

        model.fit(
            X_fit,
            y_fit,
            categorical_feature=CATEGORICAL_FEATURES,
        )

        scores = model.predict_proba(X_val)[:, 1]

        # Save the fitted model for reproducibility.
        model_path = MODELS_DIR / f"lightgbm_a0_{strategy}.joblib"
        joblib.dump(model, model_path, compress=3)

        result = evaluate_predictions(
            strategy=strategy,
            scores=scores,
            y_train=y_train,
            fit_rows=len(y_fit),
            fit_fraud=int(y_fit.sum()),
            validation_meta=validation_meta,
        )
        result["model_path"] = str(model_path.relative_to(ROOT))
        result["lightgbm_version"] = lgb.__version__
        result["training_label_cutoff_step"] = EFFECTIVE_TRAIN_END_STEP
        results.append(result)

        for feature_name, importance in zip(
            FEATURE_COLUMNS, model.feature_importances_
        ):
            importance_rows.append(
                {
                    "strategy": strategy,
                    "feature": feature_name,
                    "importance_gain": float(importance),
                }
            )

        print(
            f"Average precision: {result['average_precision']:.6f}\n"
            f"Alerts used: {result['alerts_used']:,} / "
            f"{result['available_alert_capacity']:,}\n"
            f"Precision: {result['precision_pct']:.4f}%\n"
            f"Recall: {result['recall_pct']:.4f}%\n"
            f"False-positive rate: {result['false_positive_rate_pct']:.4f}%\n"
            f"Dollar-weighted recall: "
            f"{result['dollar_weighted_recall_pct']:.4f}%\n"
            f"Fraud value caught: {result['fraud_value_caught']:,.2f}"
        )

        del model, X_fit, y_fit, scores
        gc.collect()

    metrics_df = pd.DataFrame(results)
    metrics_csv = REPORTS_DIR / "lightgbm_a0_validation_metrics.csv"
    metrics_json = REPORTS_DIR / "lightgbm_a0_validation_metrics.json"
    importance_csv = REPORTS_DIR / "lightgbm_a0_feature_importance.csv"
    train_report_path = REPORTS_DIR / "lightgbm_a0_training_report.json"

    metrics_df.to_csv(metrics_csv, index=False)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(importance_csv, index=False)

    report = {
        "feature_set": "A0 plus ratio missingness indicator",
        "input": str(FEATURES_PATH.relative_to(ROOT)),
        "nominal_training_steps": [1, TRAIN_END_STEP],
        "effective_training_end_step": EFFECTIVE_TRAIN_END_STEP,
        "label_delay_steps": LABEL_DELAY_STEPS,
        "purged_steps": [
            EFFECTIVE_TRAIN_END_STEP + 1,
            TRAIN_END_STEP,
        ],
        "validation_steps": [
            VALIDATION_START_STEP,
            VALIDATION_END_STEP,
        ],
        "alerts_per_24_steps": ALERTS_PER_DAY,
        "undersampling_negatives_per_positive": (
            UNDERSAMPLE_NEGATIVES_PER_POSITIVE
        ),
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURE_COLUMNS,
        "lightgbm_version": lgb.__version__,
        "training_summary": train_summary.to_dict(orient="records"),
        "outputs": {
            "metrics_csv": str(metrics_csv.relative_to(ROOT)),
            "metrics_json": str(metrics_json.relative_to(ROOT)),
            "feature_importance_csv": str(importance_csv.relative_to(ROOT)),
        },
        "notes": [
            "Validation is used for comparison, not for fitting preprocessing.",
            "Locked test steps are not used by this script.",
            "Undersampling changes the training class prevalence; scores may not be calibrated.",
            "A0 excludes post-transaction balance features.",
        ],
    }
    with train_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    display_columns = [
        "strategy",
        "average_precision",
        "alerts_used",
        "true_positives",
        "false_positives",
        "precision_pct",
        "recall_pct",
        "dollar_weighted_recall_pct",
    ]
    print("\nFinal validation comparison")
    print(metrics_df[display_columns].to_string(index=False))
    print(f"\nSaved metrics: {metrics_csv.relative_to(ROOT)}")
    print(f"Saved metrics JSON: {metrics_json.relative_to(ROOT)}")
    print(f"Saved feature importances: {importance_csv.relative_to(ROOT)}")
    print(f"Saved training report: {train_report_path.relative_to(ROOT)}")


def _validation_features(
    X_train: pd.DataFrame,
    validation_meta: pd.DataFrame,
    X_fit: pd.DataFrame,
    model: lgb.LGBMClassifier,
) -> pd.DataFrame:
    """Return validation features using category levels compatible with training.

    This function is replaced immediately below in main's execution path by
    the validation feature frame prepared from the source parquet.
    """
    raise RuntimeError(
        "Internal setup error: validation feature frame was not passed."
    )


if __name__ == "__main__":
    main()
