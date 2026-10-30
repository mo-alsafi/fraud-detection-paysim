
"""SHAP explainability for the selected A0 LightGBM model.

Uses:
- Selected validation model: models/lightgbm_a0_tuned_best.joblib
- Validation period: steps 446-594
- All validation fraud cases
- An equally sized random sample of legitimate validation cases

Explains the model's raw fraud log-odds, not transaction loss or causality.
Does not read or score the locked test period.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "SHAP is not installed. Install it with:\n"
        "python -m pip install shap"
    ) from exc

from src.train_lightgbm_baseline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    VALIDATION_START_STEP,
    VALIDATION_END_STEP,
    load_data,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_a0_tuned_best.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IMPORTANCE_CSV = REPORTS_DIR / "shap_a0_validation_importance.csv"
SUMMARY_JSON = REPORTS_DIR / "shap_a0_validation_summary.json"
BEESWARM_PNG = REPORTS_DIR / "shap_a0_validation_beeswarm.png"
GLOBAL_IMPORTANCE_PNG = (
    REPORTS_DIR / "shap_a0_validation_global_importance.png"
)

RANDOM_SEED = 42


def prepare_explanation_sample(
    X_val: pd.DataFrame,
    validation_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Use all fraud rows and an equal-sized random legitimate sample."""
    labels = validation_meta["isFraud"].to_numpy(dtype=np.int8)

    fraud_positions = np.flatnonzero(labels == 1)
    legitimate_positions = np.flatnonzero(labels == 0)

    if len(fraud_positions) == 0 or len(legitimate_positions) == 0:
        raise ValueError(
            "Validation sample must contain both fraud and legitimate rows."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    legitimate_sample_size = min(
        len(fraud_positions),
        len(legitimate_positions),
    )

    sampled_legitimate = rng.choice(
        legitimate_positions,
        size=legitimate_sample_size,
        replace=False,
    )

    positions = np.concatenate(
        [fraud_positions, sampled_legitimate]
    )
    rng.shuffle(positions)

    X_explain = X_val.iloc[positions].copy()
    y_explain = labels[positions]

    return X_explain, y_explain


def normalize_shap_output(
    raw_shap_values,
    expected_features: int,
) -> np.ndarray:
    """Normalize SHAP output across supported SHAP versions."""
    if isinstance(raw_shap_values, list):
        # Binary classifier: choose the positive/fraud class.
        raw_shap_values = raw_shap_values[-1]

    values = np.asarray(raw_shap_values)

    if values.ndim == 3:
        # Some SHAP versions return a separate output dimension.
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
        else:
            raise ValueError(
                f"Unexpected three-dimensional SHAP output: {values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D SHAP matrix, got shape {values.shape}"
        )

    # Some prediction-contribution interfaces append a base-value column.
    if values.shape[1] == expected_features + 1:
        values = values[:, :-1]

    if values.shape[1] != expected_features:
        raise ValueError(
            "SHAP feature count does not match model feature count: "
            f"{values.shape[1]} vs {expected_features}"
        )

    return values


def build_importance_table(
    shap_matrix: np.ndarray,
    y_explain: np.ndarray,
    feature_names: list[str],
    validation_fraud_rate: float,
) -> pd.DataFrame:
    """Calculate class-specific and prevalence-weighted importance."""
    fraud_mask = y_explain == 1
    legitimate_mask = y_explain == 0

    if not fraud_mask.any() or not legitimate_mask.any():
        raise ValueError("Explanation sample must contain both classes.")

    fraud_shap = shap_matrix[fraud_mask]
    legitimate_shap = shap_matrix[legitimate_mask]

    fraud_mean_abs = np.abs(fraud_shap).mean(axis=0)
    legitimate_mean_abs = np.abs(legitimate_shap).mean(axis=0)

    # Weight the class-specific values using the actual validation
    # prevalence, not the balanced explanation-sample prevalence.
    population_weighted = (
        validation_fraud_rate * fraud_mean_abs
        + (1.0 - validation_fraud_rate) * legitimate_mean_abs
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "population_weighted_mean_abs_shap": population_weighted,
            "fraud_mean_abs_shap": fraud_mean_abs,
            "legitimate_mean_abs_shap": legitimate_mean_abs,
            "fraud_mean_shap": fraud_shap.mean(axis=0),
            "legitimate_mean_shap": legitimate_shap.mean(axis=0),
        }
    )

    result = result.sort_values(
        "population_weighted_mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return result


def create_global_importance_plot(
    importance: pd.DataFrame,
) -> None:
    """Plot validation-prevalence-weighted mean absolute SHAP values."""
    plot_df = importance.sort_values(
        "population_weighted_mean_abs_shap",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        plot_df["feature"],
        plot_df["population_weighted_mean_abs_shap"],
    )

    ax.set_title(
        "A0 SHAP Global Importance\n"
        "Weighted using validation fraud prevalence"
    )
    ax.set_xlabel("Mean absolute SHAP value (raw log-odds)")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(GLOBAL_IMPORTANCE_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_beeswarm_plot(
    shap_matrix: np.ndarray,
    X_explain: pd.DataFrame,
) -> None:
    """Plot the balanced fraud/legitimate explanation sample."""
    shap.summary_plot(
        shap_matrix,
        X_explain,
        feature_names=list(X_explain.columns),
        max_display=len(X_explain.columns),
        show=False,
    )

    plt.title(
        "A0 SHAP Beeswarm\n"
        "All validation fraud cases + matched legitimate sample"
    )
    plt.tight_layout()
    plt.savefig(BEESWARM_PNG, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected A0 model not found: {MODEL_PATH}"
        )

    print("Loading the selected A0 validation model")
    print("----------------------------------------")
    print(f"Model: {MODEL_PATH.relative_to(ROOT)}")

    # Reuse the same feature preparation used during A0 training.
    _, X_val, validation_meta, _, _ = load_data()

    if list(X_val.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Validation feature columns do not match the A0 feature contract."
        )

    if list(X_val.columns) != list(X_val.columns):
        raise ValueError("Unexpected feature-column mismatch.")

    if not validation_meta["step"].between(
        VALIDATION_START_STEP,
        VALIDATION_END_STEP,
    ).all():
        raise ValueError("Unexpected rows found outside the validation period.")

    model = joblib.load(MODEL_PATH)

    if list(model.booster_.feature_name()) != list(X_val.columns):
        raise ValueError(
            "Loaded model features do not match validation feature columns."
        )

    y_val = validation_meta["isFraud"].to_numpy(dtype=np.int8)
    fraud_count = int(y_val.sum())
    legitimate_count = int(len(y_val) - fraud_count)
    fraud_rate = fraud_count / len(y_val)

    print(f"Validation rows: {len(y_val):,}")
    print(f"Validation fraud cases: {fraud_count:,}")
    print(f"Validation legitimate cases: {legitimate_count:,}")
    print(f"Validation fraud rate: {100.0 * fraud_rate:.6f}%")

    X_explain, y_explain = prepare_explanation_sample(
        X_val,
        validation_meta,
    )

    print("\nSHAP explanation sample")
    print("-----------------------")
    print(f"Rows explained: {len(X_explain):,}")
    print(f"Fraud cases: {int(y_explain.sum()):,}")
    print(f"Legitimate cases: {int((y_explain == 0).sum()):,}")
    print("Explaining raw fraud log-odds, not calibrated loss.")

    # TreeExplainer uses the fitted LightGBM tree structure.
    explainer = shap.TreeExplainer(
        model.booster_,
        model_output="raw",
    )

    raw_shap = explainer.shap_values(
        X_explain,
        check_additivity=False,
    )

    shap_matrix = normalize_shap_output(
        raw_shap,
        expected_features=len(FEATURE_COLUMNS),
    )

    importance = build_importance_table(
        shap_matrix=shap_matrix,
        y_explain=y_explain,
        feature_names=list(X_explain.columns),
        validation_fraud_rate=fraud_rate,
    )

    importance.to_csv(IMPORTANCE_CSV, index=False)

    create_global_importance_plot(importance)
    create_beeswarm_plot(shap_matrix, X_explain)

    summary = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "feature_set": "A0 only",
        "validation_steps": [
            VALIDATION_START_STEP,
            VALIDATION_END_STEP,
        ],
        "validation_rows": int(len(y_val)),
        "validation_fraud_cases": fraud_count,
        "validation_legitimate_cases": legitimate_count,
        "validation_fraud_rate_pct": 100.0 * fraud_rate,
        "explanation_sample_rows": int(len(X_explain)),
        "explanation_sample_fraud_cases": int(y_explain.sum()),
        "explanation_sample_legitimate_cases": int(
            np.sum(y_explain == 0)
        ),
        "sample_method": (
            "All validation fraud cases plus an equally sized "
            "random sample of validation legitimate cases."
        ),
        "shap_version": shap.__version__,
        "lightgbm_version": lgb.__version__,
        "shap_output": "raw model output (fraud log-odds)",
        "importance_method": (
            "Class-specific mean absolute SHAP, weighted by actual "
            "validation class prevalence for the overall ranking."
        ),
        "outputs": {
            "importance_csv": str(IMPORTANCE_CSV.relative_to(ROOT)),
            "beeswarm_plot": str(BEESWARM_PNG.relative_to(ROOT)),
            "global_importance_plot": str(
                GLOBAL_IMPORTANCE_PNG.relative_to(ROOT)
            ),
        },
        "limitations": [
            "SHAP explains model behavior; it does not establish causality.",
            "The validation period was used during model selection.",
            "The balanced explanation sample is not prevalence representative.",
            "Overall importance is weighted using actual validation prevalence.",
            "The locked test period was not loaded or scored.",
            "These explanations do not establish performance on real payment data.",
        ],
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    display_columns = [
        "feature",
        "population_weighted_mean_abs_shap",
        "fraud_mean_abs_shap",
        "legitimate_mean_abs_shap",
        "fraud_mean_shap",
        "legitimate_mean_shap",
    ]

    print("\nSHAP feature importance")
    print("-----------------------")
    print(importance[display_columns].to_string(index=False))

    print("\nSaved outputs")
    print("-------------")
    print(f"Importance CSV: {IMPORTANCE_CSV.relative_to(ROOT)}")
    print(f"Global importance plot: {GLOBAL_IMPORTANCE_PNG.relative_to(ROOT)}")
    print(f"Beeswarm plot: {BEESWARM_PNG.relative_to(ROOT)}")
    print(f"Summary JSON: {SUMMARY_JSON.relative_to(ROOT)}")

    del shap_matrix, raw_shap, X_explain
    gc.collect()


if __name__ == "__main__":
    main()
