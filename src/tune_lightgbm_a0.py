
"""Bounded LightGBM hyperparameter search for PaySim fraud detection.

Primary selection metric: fraud value caught at the fixed alert budget.
Secondary selection metric: average precision.

The search uses at most 12 configurations. It does not read the locked
test period or use validation for early stopping.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import joblib

from src.train_lightgbm_baseline import (
    CATEGORICAL_FEATURES,
    EFFECTIVE_TRAIN_END_STEP,
    FEATURE_COLUMNS,
    VALIDATION_START_STEP,
    VALIDATION_END_STEP,
    load_data,
    evaluate_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Best rule result from Query 28, on the same validation period and budget.
BEST_RULE_FRAUD_VALUE = 1_567_743_893.0
REQUIRED_VALUE_MULTIPLIER = 1.05
REQUIRED_FRAUD_VALUE = (
    BEST_RULE_FRAUD_VALUE * REQUIRED_VALUE_MULTIPLIER
)

# All trials are unweighted and use the complete A0 feature set.
# Trial 1 reproduces the current unweighted baseline.
TRIALS = [
    {
        "trial": 1,
        "name": "baseline_unweighted",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 2,
        "name": "more_trees_slower_learning",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 3,
        "name": "deep_trees_more_iterations",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 4,
        "name": "fewer_leaves",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 5,
        "name": "fewer_leaves_more_regularization",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 200,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "trial": 6,
        "name": "more_leaves",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 7,
        "name": "more_leaves_more_iterations",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 200,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "trial": 8,
        "name": "larger_leaf_minimum",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 500,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 9,
        "name": "larger_leaf_minimum_slow_learning",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 500,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "trial": 10,
        "name": "lower_feature_sampling",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "trial": 11,
        "name": "lower_row_sampling",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.7,
        "reg_lambda": 1.0,
    },
    {
        "trial": 12,
        "name": "stronger_regularization",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "max_bin": 127,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 10.0,
    },
]


def main() -> None:
    if len(TRIALS) > 20:
        raise ValueError("The tuning budget cannot exceed 20 trials.")

    X_train, X_val, validation_meta, y_train, train_summary = load_data()

    # Validate expected dimensions before training.
    if list(X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("Training feature columns don't match the A0 contract.")
    if list(X_val.columns) != FEATURE_COLUMNS:
        raise ValueError("Validation feature columns don't match the A0 contract.")

    print("LightGBM tuning protocol")
    print("------------------------")
    print(f"Trials: {len(TRIALS)}")
    print(f"Training cutoff: step {EFFECTIVE_TRAIN_END_STEP}")
    print(
        f"Validation period: steps "
        f"{VALIDATION_START_STEP}-{VALIDATION_END_STEP}"
    )
    print(f"Training rows: {len(y_train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation_meta):,}")
    print(
        f"Validation fraud cases: "
        f"{int(validation_meta['isFraud'].sum()):,}"
    )
    print(f"Best-rule fraud value: {BEST_RULE_FRAUD_VALUE:,.2f}")
    print(f"Required value for +5% hurdle: {REQUIRED_FRAUD_VALUE:,.2f}")

    results = []
    best_key = None
    best_model = None
    best_trial = None

    for trial in TRIALS:
        name = trial["name"]
        params = {
            key: value
            for key, value in trial.items()
            if key not in {"trial", "name"}
        }

        print(f"\n{'=' * 68}")
        print(f"Trial {trial['trial']}/{len(TRIALS)}: {name}")
        print(f"Parameters: {params}")
        print(f"{'=' * 68}")

        model = lgb.LGBMClassifier(
            objective="binary",
            **params,
            scale_pos_weight=1.0,
            subsample_freq=1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            importance_type="gain",
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=CATEGORICAL_FEATURES,
        )

        scores = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(
            strategy=name,
            scores=scores,
            y_train=y_train,
            fit_rows=len(y_train),
            fit_fraud=int(y_train.sum()),
            validation_meta=validation_meta,
        )

        value_uplift_pct = (
            100.0
            * (metrics["fraud_value_caught"] / BEST_RULE_FRAUD_VALUE - 1.0)
        )

        metrics["trial"] = trial["trial"]
        metrics["name"] = name
        metrics["value_uplift_vs_best_rule_pct"] = value_uplift_pct
        metrics["required_fraud_value_for_5pct_uplift"] = (
            REQUIRED_FRAUD_VALUE
        )
        metrics["meets_5pct_value_hurdle"] = (
            metrics["fraud_value_caught"] >= REQUIRED_FRAUD_VALUE
        )

        # Store actual parameters alongside metrics for reproducibility.
        for key, value in params.items():
            metrics[f"param_{key}"] = value

        results.append(metrics)

        # Primary criterion is value captured. Break exact ties using
        # average precision, then fraud cases caught.
        current_key = (
            metrics["fraud_value_caught"],
            metrics["average_precision"],
            metrics["true_positives"],
        )

        if best_key is None or current_key > best_key:
            best_key = current_key
            best_trial = trial
            best_model = model

            # Preserve the best-so-far model if the process is interrupted.
            best_model_path = MODELS_DIR / "lightgbm_a0_tuned_best.joblib"
            joblib.dump(best_model, best_model_path, compress=3)

        print(
            f"Average precision: {metrics['average_precision']:.6f}\n"
            f"Alerts: {metrics['alerts_used']:,} / "
            f"{metrics['available_alert_capacity']:,}\n"
            f"True positives: {metrics['true_positives']:,}\n"
            f"False positives: {metrics['false_positives']:,}\n"
            f"Precision: {metrics['precision_pct']:.4f}%\n"
            f"Recall: {metrics['recall_pct']:.4f}%\n"
            f"Dollar-weighted recall: "
            f"{metrics['dollar_weighted_recall_pct']:.4f}%\n"
            f"Fraud value caught: "
            f"{metrics['fraud_value_caught']:,.2f}\n"
            f"Value uplift vs rule: {value_uplift_pct:+.3f}%"
        )

        if best_model is not model:
            del model
        del scores
        gc.collect()

    results_df = pd.DataFrame(results)

    # Same selection order described above, for the report and final display.
    results_df = results_df.sort_values(
        by=[
            "fraud_value_caught",
            "average_precision",
            "true_positives",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    results_csv = REPORTS_DIR / "lightgbm_a0_tuning_results.csv"
    results_json = REPORTS_DIR / "lightgbm_a0_tuning_report.json"
    importance_csv = REPORTS_DIR / "lightgbm_a0_tuned_best_importance.csv"

    results_df.to_csv(results_csv, index=False)

    with results_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_count": len(TRIALS),
                "selection_metric": "fraud_value_caught",
                "secondary_metrics": [
                    "average_precision",
                    "true_positives",
                ],
                "training_cutoff_step": EFFECTIVE_TRAIN_END_STEP,
                "validation_steps": [
                    VALIDATION_START_STEP,
                    VALIDATION_END_STEP,
                ],
                "best_rule_fraud_value": BEST_RULE_FRAUD_VALUE,
                "required_fraud_value_for_5pct_uplift": REQUIRED_FRAUD_VALUE,
                "best_trial": best_trial,
                "best_model_path": "models/lightgbm_a0_tuned_best.joblib",
                "results": results,
                "notes": [
                    "Validation was used for hyperparameter selection.",
                    "The validation winner is not an unbiased final estimate.",
                    "The locked test period was not loaded.",
                    "Net benefit has not been calculated in this script.",
                ],
            },
            f,
            indent=2,
        )

    if best_model is None:
        raise RuntimeError("No successful tuning trial produced a model.")

    best_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_gain": best_model.booster_.feature_importance(
                importance_type="gain"
            ),
        }
    ).sort_values("importance_gain", ascending=False)

    best_importance.to_csv(importance_csv, index=False)

    display_columns = [
        "trial",
        "name",
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

    print("\nTuning results, ranked by fraud value caught")
    print(results_df[display_columns].to_string(index=False))
    print(f"\nBest trial: {best_trial['name']}")
    print(f"Saved best model: models/lightgbm_a0_tuned_best.joblib")
    print(f"Saved results: {results_csv.relative_to(ROOT)}")
    print(f"Saved report: {results_json.relative_to(ROOT)}")
    print(f"Saved feature importance: {importance_csv.relative_to(ROOT)}")

    # Keep this variable explicit: it is the selected validation model,
    # not a final deployment decision.
    del best_model
    gc.collect()


if __name__ == "__main__":
    main()
