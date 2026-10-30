"""Production-grade inference and feature-contract validation for PaySim A0 model.

Ensures incoming data strictly adheres to the approved A0 feature schema,
applies missing value imputation, computes fraud scores and priority scores,
and selects the top alert queue based on the daily budget policy.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "lgbm_tuned_best.joblib"

APPROVED_A0_FEATURES = [
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
    "amount_to_sender_past_mean_ratio_is_missing",
]

CATEGORICAL_FEATURES = [
    "type",
    "destination_is_merchant",
    "is_first_time_destination_for_sender",
    "hour_of_day",
]


class PaySimA0Scorer:
    """Encapsulates model loading, feature validation, and scoring logic."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Frozen model not found at {self.model_path}")
        
        print(f"Loading LightGBM model from {self.model_path.relative_to(ROOT)}...")
        self.model = joblib.load(self.model_path)

    def validate_and_prep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates feature contract and applies required preprocessing/imputation."""
        missing_cols = [col for col in APPROVED_A0_FEATURES if col not in df.columns]
        
        # Handle missing ratio imputation gracefully if raw columns are present
        working_df = df.copy()
        if "amount_to_sender_past_mean_ratio" in working_df.columns:
            working_df["amount_to_sender_past_mean_ratio_is_missing"] = working_df["amount_to_sender_past_mean_ratio"].isna().astype(int)
            working_df["amount_to_sender_past_mean_ratio"] = working_df["amount_to_sender_past_mean_ratio"].fillna(0.0)
        else:
            if "amount_to_sender_past_mean_ratio_is_missing" not in working_df.columns:
                working_df["amount_to_sender_past_mean_ratio_is_missing"] = 1.0
            if "amount_to_sender_past_mean_ratio" not in working_df.columns:
                working_df["amount_to_sender_past_mean_ratio"] = 0.0

        # Check remaining missing features (excluding ratio if handled)
        strict_missing = [c for c in APPROVED_A0_FEATURES if c not in working_df.columns]
        if strict_missing:
            raise ValueError(f"Feature contract violation: Missing required columns: {strict_missing}")

        # Ensure correct data types for LightGBM categories
        for col in CATEGORICAL_FEATURES:
            if col in working_df.columns:
                working_df[col] = working_df[col].astype("category")

        return working_df[APPROVED_A0_FEATURES]

    def score_transactions(
        self,
        df: pd.DataFrame,
        daily_budget: int = 500,
        steps_per_day: int = 24,
    ) -> pd.DataFrame:
        """Scores transactions and flags alerts based on priority score (fraud_score * amount)."""
        required_input_cols = ["step", "amount"]
        for col in required_input_cols:
            if col not in df.columns:
                raise ValueError(f"Input dataframe must contain '{col}' for operational queue ranking.")

        X = self.validate_and_prep_features(df)
        
        print(f"Scoring {len(df):,} transactions...")
        scores = self.model.predict_proba(X)[:, 1]

        result_df = df.copy()
        result_df["fraud_score"] = scores
        result_df["priority_score"] = result_df["fraud_score"] * result_df["amount"]
        result_df["simulation_day"] = np.floor((result_df["step"] - 1) / steps_per_day).astype(int) + 1

        # Apply daily budget allocation
        selected_indices = []
        for _, group in result_df.groupby("simulation_day"):
            n_steps_in_day = group["step"].nunique()
            day_budget = int(np.floor(daily_budget * (n_steps_in_day / steps_per_day)))
            if day_budget < 0:
                day_budget = 0

            eligible_group = group.sort_values(
                by=["priority_score", "amount", "step"], ascending=[False, False, True]
            )
            k = min(len(eligible_group), day_budget)
            selected_indices.extend(eligible_group.head(k).index.tolist())

        alert_mask = pd.Series(False, index=result_df.index)
        if selected_indices:
            alert_mask.loc[selected_indices] = True

        result_df["alert"] = alert_mask
        return result_df


if __name__ == "__main__":
    print("Testing PaySimA0Scorer on validation split...")
    feats_path = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
    if feats_path.is_file():
        sample_df = pd.read_parquet(feats_path)
        val_sample = sample_df[(sample_df["step"] >= 446) & (sample_df["step"] <= 450)].copy()
        
        scorer = PaySimA0Scorer()
        scored = scorer.score_transactions(val_sample)
        
        print(f"\nScored {len(scored):,} rows successfully.")
        print(f"Alerts triggered: {scored['alert'].sum():,}")
        print(scored[["step", "type", "amount", "fraud_score", "priority_score", "alert"]].head(10).to_string(index=False))