
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.train_logistic_baseline import (
    NUMERIC_FEATURES,
    build_model_matrix,
    prepare_numeric_matrix,
    alert_mask_at_daily_budget,
)


@pytest.fixture
def sample_batch():
    return pd.DataFrame({
        "hour_of_day": [0, 23],
        "type": ["TRANSFER", "CASH_OUT"],
        "log_amount": [4.0, 5.0],
        "destination_is_merchant": [False, True],
        "sender_txn_count_prev_24_steps": [0, 2],
        "sender_amount_sum_prev_24_steps": [0.0, 150.0],
        "destination_txn_count_prev_24_steps": [1, 3],
        "destination_amount_sum_prev_24_steps": [20.0, 300.0],
        "is_first_time_destination_for_sender": [True, False],
        "amount_to_sender_past_mean_ratio": [np.nan, 2.0],
    })


def test_missing_ratio_is_distinguished_from_zero(sample_batch):
    numeric = prepare_numeric_matrix(sample_batch)

    ratio_index = NUMERIC_FEATURES.index(
        "amount_to_sender_past_mean_ratio"
    )
    missing_index = NUMERIC_FEATURES.index("ratio_is_missing")

    assert numeric[0, ratio_index] == 0.0
    assert numeric[0, missing_index] == 1.0

    assert numeric[1, ratio_index] == pytest.approx(np.log1p(2.0))
    assert numeric[1, missing_index] == 0.0


def test_numeric_transform_returns_finite_values(sample_batch):
    numeric = prepare_numeric_matrix(sample_batch)

    assert numeric.shape == (2, len(NUMERIC_FEATURES))
    assert np.isfinite(numeric).all()


def test_one_hot_matrix_has_expected_shape(sample_batch):
    numeric = prepare_numeric_matrix(sample_batch)
    scaler = StandardScaler().fit(numeric)

    matrix = build_model_matrix(sample_batch, scaler)

    # 9 numeric inputs + 24 hour categories + 5 type categories.
    assert matrix.shape == (2, 38)

    n_numeric = len(NUMERIC_FEATURES)

    # Hour 0 and hour 23 each activate their own category.
    assert matrix[0, n_numeric + 0] == 1.0
    assert matrix[1, n_numeric + 23] == 1.0

    # TRANSFER has type index 4; CASH_OUT has type index 1.
    type_start = n_numeric + 24
    assert matrix[0, type_start + 4] == 1.0
    assert matrix[1, type_start + 1] == 1.0


def test_top_alert_budget_handles_partial_periods():
    steps = np.array([
        446, 447, 456,      # 11 steps of a partially covered day
        457, 458, 480,      # full 24-step group
        481,                # next full group
        594,                # 18 steps of the final group
    ])

    scores = np.array([
        0.9, 0.8, 0.7,
        0.3, 0.2, 0.1,
        0.5,
        0.6,
    ])

    alerts, permitted_capacity = alert_mask_at_daily_budget(
        scores,
        steps,
        alerts_per_day=2,
        period_start=446,
        period_end=594,
    )

    # Partial day 446-456 receives zero alerts at a budget of
    # two per full 24 steps. The full group selects the two
    # highest scores; the remaining groups select their rows.
    assert alerts.tolist() == [
        False, False, False,
        True, True, False,
        True,
        True,
    ]

    assert permitted_capacity == 5
    assert int(alerts.sum()) <= permitted_capacity


def test_invalid_steps_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        alert_mask_at_daily_budget(
            scores=np.array([0.8]),
            steps=np.array([445]),
            alerts_per_day=500,
            period_start=446,
            period_end=594,
        )
