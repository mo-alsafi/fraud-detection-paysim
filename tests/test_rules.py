
import pandas as pd
import pytest

from src.rules import (
    build_rule_alerts,
    existing_flag_rule,
    high_value_movement_rule,
    high_value_transfer_rule,
    transfer_cashout_rule,
)


@pytest.fixture
def sample_transactions():
    return pd.DataFrame(
        {
            "type": [
                "TRANSFER",
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "CASH_OUT",
                "TRANSFER",
            ],
            "amount": [
                200_000,
                200_000.01,
                150_000,
                500_000,
                99_999,
                300_000,
            ],
            "isFlaggedFraud": [0, 1, 0, 0, 0, 0],
        }
    )


def test_existing_flag_rule(sample_transactions):
    result = existing_flag_rule(sample_transactions)

    assert result.tolist() == [
        False, True, False, False, False, False
    ]


def test_transfer_cashout_rule(sample_transactions):
    result = transfer_cashout_rule(sample_transactions)

    assert result.tolist() == [
        True, True, True, False, True, True
    ]


def test_high_value_transfer_boundary(sample_transactions):
    result = high_value_transfer_rule(sample_transactions)

    assert result.tolist() == [
        False, True, False, False, False, True
    ]


def test_high_value_movement_rule(sample_transactions):
    result = high_value_movement_rule(sample_transactions)

    assert result.tolist() == [
        True, True, True, False, False, True
    ]


def test_negative_threshold_is_rejected(sample_transactions):
    with pytest.raises(ValueError, match="Threshold"):
        high_value_transfer_rule(
            sample_transactions,
            threshold=-1,
        )


def test_missing_required_column_is_rejected():
    frame = pd.DataFrame(
        {
            "type": ["TRANSFER"],
            "amount": [100],
        }
    )

    with pytest.raises(ValueError, match="Missing required"):
        existing_flag_rule(frame)


def test_invalid_flag_is_rejected():
    frame = pd.DataFrame(
        {
            "type": ["TRANSFER"],
            "amount": [100],
            "isFlaggedFraud": [2],
        }
    )

    with pytest.raises(ValueError, match="0 or 1"):
        existing_flag_rule(frame)


def test_alert_builder_returns_all_four_rules(sample_transactions):
    result = build_rule_alerts(sample_transactions)

    assert list(result.columns) == [
        "existing_flag",
        "transfer_cashout",
        "high_value_transfer",
        "high_value_movement",
    ]

    assert result.shape == (6, 4)
    assert all(dtype == bool for dtype in result.dtypes)
