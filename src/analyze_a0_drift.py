
"""Analyze A0 feature drift and daily validation stability.

Compares:
- Training: steps 1-421
- Validation: steps 446-594

The 24-step label-delay gap (422-445) is excluded.
The locked test period (595-743) is not loaded.

Outputs:
- Feature PSI
- Training-versus-validation missingness
- Fraud prevalence comparison
- Daily validation policy performance
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_lightgbm_baseline import (
    CATEGORICAL_FEATURES,
    EFFECTIVE_TRAIN_END_STEP,
    FEATURE_COLUMNS,
    VALIDATION_START_STEP,
    VALIDATION_END_STEP,
)

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT / "data" / "processed" / "paysim_a0_features.parquet"
)
MODEL_PATH = ROOT / "models" / "lightgbm_a0_tuned_best.joblib"

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PSI_CSV = REPORTS_DIR / "a0_feature_drift_psi.csv"
DAILY_CSV = REPORTS_DIR / "a0_daily_validation_stability.csv"
REPORT_JSON = REPORTS_DIR / "a0_drift_analysis_report.json"

TRAIN_END = EFFECTIVE_TRAIN_END_STEP
VALIDATION_START = VALIDATION_START_STEP
VALIDATION_END = VALIDATION_END_STEP

ALERTS_PER_DAY = 500
STEPS_PER_DAY = 24
RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "log_amount",
    "sender_txn_count_prev_24_steps",
    "sender_amount_sum_prev_24_steps",
    "destination_txn_count_prev_24_steps",
    "destination_amount_sum_prev_24_steps",
    "amount_to_sender_past_mean_ratio",
]

CATEGORICAL_FEATURES_FOR_DRIFT = [
    "hour_of_day",
    "type",
    "destination_is_merchant",
    "is_first_time_destination_for_sender",
    "amount_to_sender_past_mean_ratio_is_missing",
]

REQUIRED_COLUMNS = [
    "step",
    "isFraud",
    "amount",
    *[
        col for col in FEATURE_COLUMNS
        if col != "amount_to_sender_past_mean_ratio_is_missing"
    ],
]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same A0 feature preparation used in model training."""

    frame["step"] = pd.to_numeric(
        frame["step"], errors="raise"
    ).astype("int16")

    frame["isFraud"] = pd.to_numeric(
        frame["isFraud"], errors="raise"
    ).astype("int8")

    frame["amount"] = pd.to_numeric(
        frame["amount"], errors="raise"
    ).astype("float64")

    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("category")

    ratio = "amount_to_sender_past_mean_ratio"

    frame[ratio] = pd.to_numeric(
        frame[ratio], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    frame["amount_to_sender_past_mean_ratio_is_missing"] = (
        frame[ratio].isna().astype("int8")
    )

    for col in NUMERIC_FEATURES:
        if col == ratio:
            continue

        frame[col] = pd.to_numeric(
            frame[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    for col in NUMERIC_FEATURES:
        frame[col] = frame[col].astype("float32")

    return frame


def psi_from_counts(
    train_counts: np.ndarray,
    validation_counts: np.ndarray,
) -> float:
    """Calculate PSI with small smoothing to avoid zero-frequency errors."""

    epsilon = 1e-6

    train_pct = (
        train_counts + epsilon
    ) / (train_counts.sum() + epsilon * len(train_counts))

    validation_pct = (
        validation_counts + epsilon
    ) / (
        validation_counts.sum()
        + epsilon * len(validation_counts)
    )

    return float(
        np.sum(
            (validation_pct - train_pct)
            * np.log(validation_pct / train_pct)
        )
    )


def categorical_psi(
    train_values: pd.Series,
    validation_values: pd.Series,
) -> float:
    """Calculate PSI for categorical or discrete features."""

    train = train_values.astype("string").fillna("<MISSING>")
    validation = validation_values.astype("string").fillna("<MISSING>")

    categories = sorted(
        set(train.unique()).union(set(validation.unique()))
    )

    train_counts = (
        train.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=np.float64)
    )

    validation_counts = (
        validation.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=np.float64)
    )

    return psi_from_counts(train_counts, validation_counts)


def numeric_psi(
    train_values: pd.Series,
    validation_values: pd.Series,
) -> float:
    """Calculate PSI using quantile boundaries learned from training data."""

    train = pd.to_numeric(
        train_values, errors="coerce"
    ).to_numpy(dtype=np.float64)

    validation = pd.to_numeric(
        validation_values, errors="coerce"
    ).to_numpy(dtype=np.float64)

    train = np.where(np.isfinite(train), train, np.nan)
    validation = np.where(np.isfinite(validation), validation, np.nan)

    finite_train = train[np.isfinite(train)]

    if len(finite_train) == 0:
        return float("nan")

    # Create approximately equal-frequency training bins.
    quantiles = np.nanquantile(
        finite_train,
        np.linspace(0, 1, 11),
    )

    interior_edges = np.unique(quantiles)[1:-1]

    # One extra bucket is reserved for missing values.
    number_of_bins = len(interior_edges) + 2

    def assign_bins(values: np.ndarray) -> np.ndarray:
        bins = np.full(
            len(values),
            number_of_bins - 1,
            dtype=np.int32,
        )

        valid = np.isfinite(values)
        bins[valid] = np.digitize(
            values[valid],
            interior_edges,
        )

        return bins

    train_bins = assign_bins(train)
    validation_bins = assign_bins(validation)

    train_counts = np.bincount(
        train_bins,
        minlength=number_of_bins,
    ).astype(np.float64)

    validation_counts = np.bincount(
        validation_bins,
        minlength=number_of_bins,
    ).astype(np.float64)

    return psi_from_counts(
        train_counts,
        validation_counts,
    )


def interpret_psi(value: float) -> str:
    """Return a descriptive, non-binding PSI guideline."""

    if not np.isfinite(value):
        return "undefined"
    if value < 0.10:
        return "low"
    if value < 0.25:
        return "moderate"
    return "high"


def calculate_feature_drift(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate PSI and missingness for all approved A0 features."""

    rows = []

    for feature in FEATURE_COLUMNS:
        train_missing = float(train[feature].isna().mean())
        validation_missing = float(validation[feature].isna().mean())

        if feature in CATEGORICAL_FEATURES_FOR_DRIFT:
            psi = categorical_psi(
                train[feature],
                validation[feature],
            )
        else:
            psi = numeric_psi(
                train[feature],
                validation[feature],
            )

        rows.append(
            {
                "feature": feature,
                "feature_type": (
                    "categorical"
                    if feature in CATEGORICAL_FEATURES_FOR_DRIFT
                    else "numeric"
                ),
                "train_missing_pct": 100.0 * train_missing,
                "validation_missing_pct": 100.0 * validation_missing,
                "missing_rate_change_percentage_points": (
                    100.0 * (validation_missing - train_missing)
                ),
                "psi": psi,
                "psi_guideline": interpret_psi(psi),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


def calculate_daily_metrics(
    scores: np.ndarray,
    validation_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate frozen score-times-amount policy metrics per day."""

    steps = validation_meta["step"].to_numpy(dtype=np.int32)
    amounts = validation_meta["amount"].to_numpy(dtype=np.float64)
    labels = validation_meta["isFraud"].to_numpy(dtype=np.int8)

    priority = scores * amounts
    simulation_days = (steps - 1) // STEPS_PER_DAY

    rows = []

    for day in np.unique(simulation_days):
        positions = np.flatnonzero(simulation_days == day)

        day_steps = np.unique(steps[positions])
        covered_steps = len(day_steps)

        capacity = int(
            np.floor(
                ALERTS_PER_DAY
                * covered_steps
                / STEPS_PER_DAY
            )
        )

        # Reproduce the frozen queue's deterministic tie-breaking.
        order = np.lexsort(
            (
                positions,
                -amounts[positions],
                -priority[positions],
            )
        )

        chosen = positions[order[:capacity]]

        selected = np.zeros(
            len(positions),
            dtype=bool,
        )
        selected[order[:capacity]] = True

        day_labels = labels[positions]
        day_amounts = amounts[positions]

        tp = int(
            np.sum(selected & (day_labels == 1))
        )
        fp = int(
            np.sum(selected & (day_labels == 0))
        )
        fraud_count = int(
            np.sum(day_labels == 1)
        )

        total_fraud_value = float(
            day_amounts[day_labels == 1].sum()
        )

        caught_fraud_value = float(
            day_amounts[selected & (day_labels == 1)].sum()
        )

        alerts = int(selected.sum())

        rows.append(
            {
                "simulation_day": int(day),
                "first_step": int(day_steps.min()),
                "last_step": int(day_steps.max()),
                "covered_steps": int(covered_steps),
                "transactions": int(len(positions)),
                "fraud_cases": fraud_count,
                "fraud_rate_pct": (
                    100.0 * fraud_count / len(positions)
                ),
                "alert_capacity": capacity,
                "alerts_used": alerts,
                "true_positives": tp,
                "false_positives": fp,
                "precision_pct": (
                    100.0 * tp / alerts if alerts else 0.0
                ),
                "recall_pct": (
                    100.0 * tp / fraud_count
                    if fraud_count else 0.0
                ),
                "fraud_value_total": total_fraud_value,
                "fraud_value_caught": caught_fraud_value,
                "dollar_weighted_recall_pct": (
                    100.0 * caught_fraud_value / total_fraud_value
                    if total_fraud_value else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"A0 feature file not found: {FEATURES_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected model not found: {MODEL_PATH}"
        )

    if EFFECTIVE_TRAIN_END_STEP != TRAIN_END:
        raise RuntimeError(
            "Training cutoff no longer matches the frozen protocol."
        )

    print("A0 drift analysis")
    print("-----------------")
    print(f"Training period ends: step {TRAIN_END}")
    print(f"Validation period: {VALIDATION_START}-{VALIDATION_END}")
    print("Purged gap: steps 422-445")
    print("Locked test period: not loaded")

    # Read only through the end of validation.
    # Test steps 595-743 are deliberately excluded from this read.
    frame = pd.read_parquet(
        FEATURES_PATH,
        columns=REQUIRED_COLUMNS,
        filters=[("step", "<=", VALIDATION_END)],
    )

    frame = normalize_frame(frame)

    train_mask = frame["step"] <= TRAIN_END

    validation_mask = (
        (frame["step"] >= VALIDATION_START)
        & (frame["step"] <= VALIDATION_END)
    )

    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()

    train_rows = len(train)
    validation_rows = len(validation)

    train_fraud = int(train["isFraud"].sum())
    validation_fraud = int(validation["isFraud"].sum())

    train_fraud_rate = train_fraud / train_rows
    validation_fraud_rate = validation_fraud / validation_rows

    fraud_rate_ratio = (
        validation_fraud_rate / train_fraud_rate
        if train_fraud_rate > 0 else None
    )

    print("\nClass prevalence")
    print("----------------")
    print(f"Training rows: {train_rows:,}")
    print(f"Training fraud cases: {train_fraud:,}")
    print(f"Training fraud rate: {100.0 * train_fraud_rate:.6f}%")
    print(f"Validation rows: {validation_rows:,}")
    print(f"Validation fraud cases: {validation_fraud:,}")
    print(
        "Validation fraud rate: "
        f"{100.0 * validation_fraud_rate:.6f}%"
    )
    print(f"Fraud-rate ratio: {fraud_rate_ratio:.2f}x")

    # -----------------------------------------------------------------
    # 1. Feature drift
    # -----------------------------------------------------------------

    print("\nCalculating A0 feature PSI")
    psi_df = calculate_feature_drift(
        train=train,
        validation=validation,
    )

    psi_df.to_csv(PSI_CSV, index=False)

    print("\nFeature drift ranking")
    print("---------------------")
    print(psi_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 2. Daily validation policy performance
    # -----------------------------------------------------------------

    X_validation = validation[FEATURE_COLUMNS].reset_index(drop=True)

    validation_meta = validation[
        ["step", "isFraud", "amount"]
    ].reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    if list(model.booster_.feature_name()) != FEATURE_COLUMNS:
        raise ValueError(
            "Selected model's feature order differs from A0 contract."
        )

    scores = model.predict_proba(X_validation)[:, 1]

    daily_df = calculate_daily_metrics(
        scores=scores,
        validation_meta=validation_meta,
    )

    daily_df.to_csv(DAILY_CSV, index=False)

    print("\nDaily validation stability")
    print("--------------------------")
    print(daily_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 3. Summarize major drift signals
    # -----------------------------------------------------------------

    high_drift = psi_df.loc[
        psi_df["psi"] >= 0.25,
        "feature",
    ].tolist()

    moderate_drift = psi_df.loc[
        (psi_df["psi"] >= 0.10)
        & (psi_df["psi"] < 0.25),
        "feature",
    ].tolist()

    report = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "feature_set": "A0 only",
        "training_steps": [1, TRAIN_END],
        "validation_steps": [
            VALIDATION_START,
            VALIDATION_END,
        ],
        "excluded_steps": [422, 445],
        "test_period_loaded": False,
        "training_rows": int(train_rows),
        "training_fraud_cases": train_fraud,
        "training_fraud_rate_pct": 100.0 * train_fraud_rate,
        "validation_rows": int(validation_rows),
        "validation_fraud_cases": validation_fraud,
        "validation_fraud_rate_pct": (
            100.0 * validation_fraud_rate
        ),
        "fraud_rate_ratio_validation_to_training": (
            float(fraud_rate_ratio)
            if fraud_rate_ratio is not None else None
        ),
        "psi_guidelines": {
            "below_0_10": "low drift",
            "0_10_to_below_0_25": "moderate drift",
            "0_25_or_higher": "high drift",
            "note": (
                "PSI thresholds are rough monitoring guidelines, "
                "not universal statistical significance boundaries."
            ),
        },
        "high_drift_features": high_drift,
        "moderate_drift_features": moderate_drift,
        "feature_psi": psi_df.to_dict(orient="records"),
        "daily_validation_metrics": daily_df.to_dict(
            orient="records"
        ),
        "outputs": {
            "feature_psi_csv": str(PSI_CSV.relative_to(ROOT)),
            "daily_stability_csv": str(DAILY_CSV.relative_to(ROOT)),
        },
        "limitations": [
            "Feature PSI is descriptive and depends on binning.",
            "Validation was used for model and policy selection.",
            "Daily metrics can be volatile when fraud counts are small.",
            "PSI does not establish the cause of drift.",
            "The locked test period was not loaded or rescored.",
        ],
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved outputs")
    print("-------------")
    print(f"Feature PSI: {PSI_CSV.relative_to(ROOT)}")
    print(f"Daily performance: {DAILY_CSV.relative_to(ROOT)}")
    print(f"JSON report: {REPORT_JSON.relative_to(ROOT)}")

    del frame, train, validation, X_validation
    del scores, model
    gc.collect()


if __name__ == "__main__":
    main()
    