
from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "type",
    "amount",
    "isFlaggedFraud",
}


def _validate_inputs(frame: pd.DataFrame) -> None:
    """Validate the columns needed by the baseline rules."""
    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Cannot score an empty DataFrame.")

    if frame["type"].isna().any():
        raise ValueError("Transaction type contains missing values.")

    amounts = pd.to_numeric(frame["amount"], errors="raise")

    if not amounts.notna().all():
        raise ValueError("Transaction amount contains missing values.")

    if not amounts.map(lambda value: value == value).all():
        raise ValueError("Transaction amounts contain invalid values.")

    if (amounts < 0).any():
        raise ValueError("Transaction amounts cannot be negative.")

    if not frame["isFlaggedFraud"].isin([0, 1]).all():
        raise ValueError("isFlaggedFraud must contain only 0 or 1.")


def existing_flag_rule(frame: pd.DataFrame) -> pd.Series:
    """Use the dataset's existing isFlaggedFraud indicator."""
    _validate_inputs(frame)

    return frame["isFlaggedFraud"].eq(1)


def transfer_cashout_rule(frame: pd.DataFrame) -> pd.Series:
    """Flag every TRANSFER and CASH_OUT transaction."""
    _validate_inputs(frame)

    return frame["type"].isin(["TRANSFER", "CASH_OUT"])


def high_value_transfer_rule(
    frame: pd.DataFrame,
    threshold: float = 200_000.0,
) -> pd.Series:
    """Flag TRANSFER transactions strictly above the threshold."""
    _validate_inputs(frame)

    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")

    return (
        frame["type"].eq("TRANSFER")
        & frame["amount"].gt(threshold)
    )


def high_value_movement_rule(
    frame: pd.DataFrame,
    threshold: float = 100_000.0,
) -> pd.Series:
    """
    Flag high-value TRANSFER and CASH_OUT transactions.

    This is a simple, fixed exploratory rule. Its threshold
    has not been optimized using validation performance.
    """
    _validate_inputs(frame)

    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")

    eligible_type = frame["type"].isin(["TRANSFER", "CASH_OUT"])

    return eligible_type & frame["amount"].ge(threshold)


def build_rule_alerts(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return all declared rule policies as Boolean alert masks."""
    return pd.DataFrame(
        {
            "existing_flag": existing_flag_rule(frame),
            "transfer_cashout": transfer_cashout_rule(frame),
            "high_value_transfer": high_value_transfer_rule(frame),
            "high_value_movement": high_value_movement_rule(frame),
        },
        index=frame.index,
    )
