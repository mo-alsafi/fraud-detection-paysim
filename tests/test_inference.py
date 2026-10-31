
"""Unit tests for PaySim A0 inference and alert prioritization."""

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MISSINGNESS_FEATURE,
    PaySimA0Scorer,
)


class FakeBooster:
    """Minimal booster metadata for inference contract tests."""

    pandas_categorical = [
        list(range(24)),
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
    ]

    @staticmethod
    def feature_name():
        return MODEL_FEATURES.copy()


class FakeModel:
    """Deterministic model used only in unit tests."""

    booster_ = FakeBooster()
    classes_ = np.array([0, 1])

    @staticmethod
    def predict_proba(X):
        scores = np.clip(
            X["log_amount"].to_numpy(dtype=float),
            0.0,
            1.0,
        )

        return np.column_stack(
            [1.0 - scores, scores]
        )


def make_frame():
    return pd.DataFrame(
        {
            "transaction_id": [101, 102, 103, 104],
            "step": [1, 2, 3, 4],
            "amount": [100.0, 500.0, 200.0, 1000.0],
            "hour_of_day": [1, 2, 3, 4],
            "type": [
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "TRANSFER",
            ],
            "log_amount": [0.1, 0.8, 0.5, 0.3],
            "destination_is_merchant": [0, 0, 1, 0],
            "sender_txn_count_prev_24_steps": [0, 1, 0, 2],
            "sender_amount_sum_prev_24_steps": [
                0.0, 100.0, 0.0, 300.0
            ],
            "destination_txn_count_prev_24_steps": [0, 1, 2, 1],
            "destination_amount_sum_prev_24_steps": [
                0.0, 50.0, 500.0, 200.0
            ],
            "is_first_time_destination_for_sender": [1, 0, 1, 0],
            "amount_to_sender_past_mean_ratio": [
                np.nan, 1.5, np.nan, 0.8
            ],
        }
    )


@pytest.fixture
def scorer():
    instance = PaySimA0Scorer.__new__(PaySimA0Scorer)
    instance.model = FakeModel()
    instance.category_levels = instance._validate_model()
    return instance


def test_model_feature_schema_matches_contract(scorer):
    frame = make_frame()

    prepared = scorer.validate_and_prep_features(frame)

    assert list(prepared.columns) == MODEL_FEATURES
    assert prepared.shape == (4, len(MODEL_FEATURES))


def test_missing_ratio_is_preserved_as_nan(scorer):
    frame = make_frame()

    prepared = scorer.validate_and_prep_features(frame)

    assert pd.isna(
        prepared.loc[0, "amount_to_sender_past_mean_ratio"]
    )

    assert prepared.loc[0, MISSINGNESS_FEATURE] == 1
    assert prepared.loc[1, MISSINGNESS_FEATURE] == 0


def test_categorical_columns_match_training_contract(scorer):
    prepared = scorer.validate_and_prep_features(make_frame())

    for feature in CATEGORICAL_FEATURES:
        assert isinstance(
            prepared[feature].dtype,
            pd.CategoricalDtype,
        )

    # These were numeric in model training, not categorical.
    assert not isinstance(
        prepared["destination_is_merchant"].dtype,
        pd.CategoricalDtype,
    )

    assert not isinstance(
        prepared["is_first_time_destination_for_sender"].dtype,
        pd.CategoricalDtype,
    )


@pytest.mark.parametrize(
    "forbidden_column",
    [
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
    ],
)
def test_forbidden_columns_are_rejected(scorer, forbidden_column):
    frame = make_frame()
    frame[forbidden_column] = 1

    with pytest.raises(ValueError, match="Forbidden"):
        scorer.validate_and_prep_features(frame)


def test_missing_required_feature_is_rejected(scorer):
    frame = make_frame().drop(
        columns=["destination_txn_count_prev_24_steps"]
    )

    with pytest.raises(ValueError, match="Missing required"):
        scorer.validate_and_prep_features(frame)


def test_unknown_transaction_type_is_rejected(scorer):
    frame = make_frame()
    frame.loc[0, "type"] = "UNKNOWN_TYPE"

    with pytest.raises(ValueError, match="Unknown values"):
        scorer.validate_and_prep_features(frame)


def test_inconsistent_missingness_flag_is_rejected(scorer):
    frame = make_frame()
    frame[MISSINGNESS_FEATURE] = [0, 0, 1, 0]

    with pytest.raises(ValueError, match="does not match"):
        scorer.validate_and_prep_features(frame)


def test_duplicate_transaction_ids_are_rejected(scorer):
    frame = make_frame()
    frame.loc[1, "transaction_id"] = frame.loc[0, "transaction_id"]

    with pytest.raises(ValueError, match="duplicates"):
        scorer.validate_and_prep_features(frame)


def test_score_transaction_returns_scores_and_priority(scorer):
    frame = make_frame()

    result = scorer.score_transactions(frame)

    expected_scores = frame["log_amount"].to_numpy()
    expected_priority = expected_scores * frame["amount"].to_numpy()

    np.testing.assert_allclose(
        result["fraud_score"].to_numpy(),
        expected_scores,
    )

    np.testing.assert_allclose(
        result["priority_score"].to_numpy(),
        expected_priority,
    )


def test_alert_budget_and_priority_tie_break():
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "amount": [100.0, 200.0, 200.0, 100.0],
            "fraud_score": [0.4, 0.2, 0.1, 0.1],
        }
    )

    frame["priority_score"] = (
        frame["fraud_score"] * frame["amount"]
    )

    result = PaySimA0Scorer.prioritize_alerts(
        frame,
        daily_budget=1,
        steps_per_day=4,
    )

    assert int(result["alert"].sum()) == 1

    # First two rows tie on priority; the second has a higher amount.
    assert not result.loc[0, "alert"]
    assert result.loc[1, "alert"]


def test_partial_day_budget_is_prorated():
    frame = pd.DataFrame(
        {
            "step": [1, 2],
            "amount": [100.0, 500.0],
            "fraud_score": [0.2, 0.8],
        }
    )

    frame["priority_score"] = (
        frame["fraud_score"] * frame["amount"]
    )

    result = PaySimA0Scorer.prioritize_alerts(
        frame,
        daily_budget=2,
        steps_per_day=4,
    )

    # 2 * 2 covered steps / 4 steps = 1 alert.
    assert int(result["daily_alert_capacity"].max()) == 1
    assert int(result["alert"].sum()) == 1
    assert result.loc[1, "alert"]


def test_changed_priority_formula_is_rejected():
    frame = pd.DataFrame(
        {
            "step": [1],
            "amount": [100.0],
            "fraud_score": [0.5],
            "priority_score": [999.0],
        }
    )

    with pytest.raises(ValueError, match="fraud_score"):
        PaySimA0Scorer.prioritize_alerts(frame)


def test_invalid_daily_budget_is_rejected():
    frame = pd.DataFrame(
        {
            "step": [1],
            "amount": [100.0],
            "fraud_score": [0.5],
            "priority_score": [50.0],
        }
    )

    with pytest.raises(ValueError, match="daily_budget"):
        PaySimA0Scorer.prioritize_alerts(
            frame,
            daily_budget=-1,
        )
