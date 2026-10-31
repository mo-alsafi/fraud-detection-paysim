
"""Reusable inference interface for the PaySim A0 LightGBM model.

The interface:
1. Validates incoming A0 features.
2. Preserves the same missing-value behavior used during training.
3. Loads and validates the frozen final model.
4. Generates fraud scores and value-priority scores.
5. Applies the fixed daily alert budget.

This is a research scoring interface, not a production payment gateway.
A0 features must already have been engineered using past-only information.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_contract import (
    A0_FEATURES,
    assert_deployable_feature_set,
    validate_feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]

# This is the model used in the one-time locked test evaluation.
DEFAULT_MODEL_PATH = (
    ROOT / "models" / "lightgbm_a0_final_test_model.joblib"
)

MISSINGNESS_FEATURE = (
    "amount_to_sender_past_mean_ratio_is_missing"
)

MODEL_FEATURES = list(A0_FEATURES) + [MISSINGNESS_FEATURE]

# These are the only categorical columns in the training pipeline.
CATEGORICAL_FEATURES = [
    "hour_of_day",
    "type",
]

BINARY_NUMERIC_FEATURES = [
    "destination_is_merchant",
    "is_first_time_destination_for_sender",
]

METADATA_COLUMNS = {
    "transaction_id",
    "step",
    "amount",
}

FORBIDDEN_COLUMNS = {
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "oldbalanceDest",
    "newbalanceOrig",
    "newbalanceDest",
    "drain_ratio",
    "amount_to_oldbalanceDest_ratio",
}

ALLOWED_INPUT_COLUMNS = (
    set(A0_FEATURES)
    | {MISSINGNESS_FEATURE}
    | METADATA_COLUMNS
)


class PaySimA0Scorer:
    """Load, validate, and score transactions using the frozen A0 model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        assert_deployable_feature_set("A0")

        path = Path(model_path)

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Final A0 model not found: {path}\n"
                "The model artifact must be available locally."
            )

        self.model_path = path
        self.model = joblib.load(path)

        self.category_levels = self._validate_model()

    def _validate_model(self) -> dict[str, list]:
        """Verify the loaded model against the approved A0 contract."""

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "The saved model must implement predict_proba()."
            )

        booster = getattr(self.model, "booster_", None)

        if booster is None:
            raise TypeError(
                "Expected a fitted LightGBM sklearn classifier."
            )

        model_features = list(booster.feature_name())

        if model_features != MODEL_FEATURES:
            raise ValueError(
                "Saved model feature order does not match the A0 "
                "inference contract.\n"
                f"Expected: {MODEL_FEATURES}\n"
                f"Actual:   {model_features}"
            )

        classes = getattr(self.model, "classes_", None)

        if classes is None or list(classes) != [0, 1]:
            raise ValueError(
                "Expected a binary classifier with classes [0, 1]."
            )

        stored_categories = getattr(
            booster,
            "pandas_categorical",
            None,
        )

        if stored_categories is None:
            raise RuntimeError(
                "The model does not contain saved pandas categorical "
                "metadata. Cannot safely prepare categorical inputs."
            )

        if len(stored_categories) != len(CATEGORICAL_FEATURES):
            raise RuntimeError(
                "Model categorical metadata does not match the "
                "expected A0 categorical feature list."
            )

        category_levels = {
            feature: list(levels)
            for feature, levels in zip(
                CATEGORICAL_FEATURES,
                stored_categories,
            )
        }

        for feature, levels in category_levels.items():
            if not levels:
                raise RuntimeError(
                    f"Model has no saved categories for {feature}."
                )

        return category_levels

    @staticmethod
    def _validate_metadata(df: pd.DataFrame) -> None:
        """Validate transaction metadata used by the operational queue."""

        if "amount" not in df.columns:
            raise ValueError(
                "Input must include amount for value-priority ranking."
            )

        if "step" not in df.columns:
            raise ValueError(
                "Input must include step for daily alert allocation."
            )

        amount = pd.to_numeric(df["amount"], errors="coerce")
        steps = pd.to_numeric(df["step"], errors="coerce")

        if amount.isna().any() or steps.isna().any():
            raise ValueError(
                "amount and step must not contain missing or "
                "non-numeric values."
            )

        amount_values = amount.to_numpy(dtype=np.float64)
        step_values = steps.to_numpy(dtype=np.float64)

        if not np.isfinite(amount_values).all():
            raise ValueError("amount contains non-finite values.")

        if (amount_values < 0).any():
            raise ValueError("amount cannot contain negative values.")

        if (
            not np.isfinite(step_values).all()
            or (step_values < 1).any()
            or not np.equal(step_values, np.floor(step_values)).all()
        ):
            raise ValueError(
                "step must contain positive, finite integer values."
            )

        if "transaction_id" in df.columns:
            if df["transaction_id"].isna().any():
                raise ValueError(
                    "transaction_id contains missing values."
                )

            if df["transaction_id"].duplicated().any():
                raise ValueError(
                    "transaction_id contains duplicates."
                )

    def validate_and_prep_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the exact 11-column A0 model input matrix.

        The historical sender ratio remains NaN when undefined, matching
        model training. LightGBM handles those missing values natively.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if df.empty:
            raise ValueError("Cannot score an empty DataFrame.")

        if not df.columns.is_unique:
            raise ValueError(
                "Input contains duplicate column names."
            )

        forbidden = FORBIDDEN_COLUMNS.intersection(df.columns)

        if forbidden:
            raise ValueError(
                "Forbidden target, account identifier, or balance "
                f"columns detected: {sorted(forbidden)}."
            )

        unexpected = set(df.columns) - ALLOWED_INPUT_COLUMNS

        if unexpected:
            raise ValueError(
                "Unexpected input columns detected: "
                f"{sorted(unexpected)}."
            )

        missing = set(A0_FEATURES) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required A0 feature columns: {sorted(missing)}."
            )

        validate_feature_columns(
            "A0",
            list(A0_FEATURES),
        )

        self._validate_metadata(df)

        prepared = df.copy()

        # Preserve undefined ratios as NaN.
        # This matches the training pipeline; do not impute zero.
        ratio_column = "amount_to_sender_past_mean_ratio"

        ratio = pd.to_numeric(
            prepared[ratio_column],
            errors="coerce",
        )

        invalid_ratio = (
            prepared[ratio_column].notna()
            & ratio.isna()
        )

        if invalid_ratio.any():
            raise ValueError(
                f"{ratio_column} contains invalid nonnumeric values."
            )

        ratio = ratio.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        prepared[ratio_column] = ratio

        expected_missingness = (
            ratio.isna().astype("int8")
        )

        # If a missingness flag is supplied, verify it rather than
        # silently replacing it.
        if MISSINGNESS_FEATURE in prepared.columns:
            provided = pd.to_numeric(
                prepared[MISSINGNESS_FEATURE],
                errors="coerce",
            )

            if provided.isna().any():
                raise ValueError(
                    f"{MISSINGNESS_FEATURE} contains invalid values."
                )

            if not provided.isin([0, 1]).all():
                raise ValueError(
                    f"{MISSINGNESS_FEATURE} must contain only 0 or 1."
                )

            if not np.array_equal(
                provided.to_numpy(dtype=np.int8),
                expected_missingness.to_numpy(dtype=np.int8),
            ):
                raise ValueError(
                    "The supplied ratio missingness indicator "
                    "does not match the ratio values."
                )

        prepared[MISSINGNESS_FEATURE] = expected_missingness

        # Restore the categories exactly as saved during training.
        for feature in CATEGORICAL_FEATURES:
            values = prepared[feature]

            if values.isna().any():
                raise ValueError(
                    f"Categorical feature {feature} contains missing values."
                )

            levels = self.category_levels[feature]

            unknown = ~values.isin(levels)

            if unknown.any():
                unknown_values = (
                    values.loc[unknown]
                    .drop_duplicates()
                    .tolist()
                )

                raise ValueError(
                    f"Unknown values in {feature}: {unknown_values}. "
                    "These categories were not present during model training."
                )

            prepared[feature] = pd.Series(
                pd.Categorical(
                    values,
                    categories=levels,
                ),
                index=prepared.index,
                name=feature,
            )

        # Convert other model inputs to the same numeric dtype
        # used during model training.
        for feature in MODEL_FEATURES:
            if feature in CATEGORICAL_FEATURES:
                continue

            if feature == MISSINGNESS_FEATURE:
                prepared[feature] = (
                    prepared[feature].astype("int8")
                )
                continue

            if feature == ratio_column:
                # NaN is permitted for this historical feature.
                prepared[feature] = (
                    prepared[feature].astype("float32")
                )
                continue

            numeric = pd.to_numeric(
                prepared[feature],
                errors="coerce",
            )

            invalid = (
                prepared[feature].notna()
                & numeric.isna()
            )

            if invalid.any():
                raise ValueError(
                    f"{feature} contains invalid nonnumeric values."
                )

            values = numeric.to_numpy(dtype=np.float64)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{feature} contains missing or non-finite values."
                )

            prepared[feature] = numeric.astype("float32")

        for feature in BINARY_NUMERIC_FEATURES:
            if not prepared[feature].isin([0, 1]).all():
                raise ValueError(
                    f"{feature} must contain only 0 or 1."
                )

        return prepared.loc[:, MODEL_FEATURES].copy()

    def score_transactions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate fraud and value-priority scores without alert selection."""

        X = self.validate_and_prep_features(df)

        probabilities = np.asarray(
            self.model.predict_proba(X)
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != 2
        ):
            raise ValueError(
                "Model returned an unexpected probability matrix."
            )

        fraud_scores = probabilities[:, 1].astype(np.float64)

        if not np.isfinite(fraud_scores).all():
            raise ValueError(
                "Model produced non-finite fraud scores."
            )

        if ((fraud_scores < 0) | (fraud_scores > 1)).any():
            raise ValueError(
                "Fraud scores must be between 0 and 1."
            )

        result = df.copy()

        amounts = pd.to_numeric(
            df["amount"],
            errors="raise",
        ).to_numpy(dtype=np.float64)

        result["fraud_score"] = fraud_scores

        # Ranking heuristic, not a calibrated monetary-loss prediction.
        result["priority_score"] = fraud_scores * amounts

        return result

    @staticmethod
    def prioritize_alerts(
        scored_df: pd.DataFrame,
        daily_budget: int = 500,
        steps_per_day: int = 24,
    ) -> pd.DataFrame:
        """Select the highest-priority alerts within each simulation day.

        Daily capacity is prorated for partial simulation days.
        Ties are broken by amount descending and then original row order.
        """

        if (
            isinstance(daily_budget, bool)
            or not isinstance(daily_budget, (int, np.integer))
            or daily_budget < 0
        ):
            raise ValueError(
                "daily_budget must be a nonnegative integer."
            )

        if (
            isinstance(steps_per_day, bool)
            or not isinstance(steps_per_day, (int, np.integer))
            or steps_per_day < 1
        ):
            raise ValueError(
                "steps_per_day must be a positive integer."
            )

        required = {
            "step",
            "amount",
            "fraud_score",
            "priority_score",
        }

        missing = required - set(scored_df.columns)

        if missing:
            raise ValueError(
                f"Missing columns needed for alert selection: "
                f"{sorted(missing)}."
            )

        result = scored_df.copy()

        steps = pd.to_numeric(
            result["step"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        amounts = pd.to_numeric(
            result["amount"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        scores = pd.to_numeric(
            result["fraud_score"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        priorities = pd.to_numeric(
            result["priority_score"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        for name, values in {
            "step": steps,
            "amount": amounts,
            "fraud_score": scores,
            "priority_score": priorities,
        }.items():
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{name} contains non-finite values."
                )

        if (
            (steps < 1).any()
            or not np.equal(steps, np.floor(steps)).all()
        ):
            raise ValueError(
                "step must contain positive integer values."
            )

        if (amounts < 0).any():
            raise ValueError(
                "amount cannot contain negative values."
            )

        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(
                "fraud_score must be between 0 and 1."
            )

        if (priorities < 0).any():
            raise ValueError(
                "priority_score cannot contain negative values."
            )

        # Confirm the frozen ranking formula has not been changed.
        expected_priority = scores * amounts

        if not np.allclose(
            priorities,
            expected_priority,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "priority_score must equal fraud_score * amount."
            )

        integer_steps = steps.astype(np.int64)

        day_ids = (
            integer_steps - 1
        ) // int(steps_per_day)

        selected = np.zeros(
            len(result),
            dtype=bool,
        )

        daily_ranks = np.zeros(
            len(result),
            dtype=np.int32,
        )

        daily_capacities = np.zeros(
            len(result),
            dtype=np.int32,
        )

        for day in np.unique(day_ids):
            positions = np.flatnonzero(
                day_ids == day
            )

            covered_steps = np.unique(
                integer_steps[positions]
            ).size

            capacity = int(
                np.floor(
                    daily_budget
                    * covered_steps
                    / steps_per_day
                )
            )

            daily_capacities[positions] = capacity

            # Match the deterministic ordering used in the evaluation:
            # priority descending, amount descending, original row position.
            order = np.lexsort(
                (
                    positions,
                    -amounts[positions],
                    -priorities[positions],
                )
            )

            daily_ranks[
                positions[order]
            ] = np.arange(
                1,
                len(positions) + 1,
                dtype=np.int32,
            )

            chosen = positions[
                order[:capacity]
            ]

            selected[chosen] = True

        result["simulation_day"] = day_ids + 1
        result["daily_priority_rank"] = daily_ranks
        result["daily_alert_capacity"] = daily_capacities
        result["alert"] = selected

        return result

    def score_and_prioritize(
        self,
        df: pd.DataFrame,
        daily_budget: int = 500,
        steps_per_day: int = 24,
    ) -> pd.DataFrame:
        """Convenience method: validate, score, and select alerts."""

        scored = self.score_transactions(df)

        return self.prioritize_alerts(
            scored_df=scored,
            daily_budget=daily_budget,
            steps_per_day=steps_per_day,
        )
