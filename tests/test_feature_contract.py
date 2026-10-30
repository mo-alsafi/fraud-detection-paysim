
import numpy as np
import pytest

from src.evaluation import (
    LABEL_DELAY_STEPS,
    past_label_counts,
)
from src.feature_contract import (
    A0_FEATURES,
    FEATURE_SETS,
    PRE_TRANSACTION_BALANCE_FEATURES,
    POST_TRANSACTION_BALANCE_FEATURES,
    assert_deployable_feature_set,
    get_feature_columns,
    get_feature_set_purpose,
    validate_feature_columns,
)


def test_a0_contains_no_balance_or_target_columns():
    columns = set(get_feature_columns("A0"))

    balance_columns = {
        "oldbalanceOrg",
        "oldbalanceDest",
        "newbalanceOrig",
        "newbalanceDest",
        "drain_ratio",
        "amount_to_oldbalanceDest_ratio",
    }

    assert columns.isdisjoint(balance_columns)
    assert "isFraud" not in columns
    assert "nameOrig" not in columns
    assert "nameDest" not in columns


def test_a1_extends_a0_with_pre_transaction_balances():
    expected = A0_FEATURES + PRE_TRANSACTION_BALANCE_FEATURES

    assert get_feature_columns("A1") == expected


def test_b_extends_a1_with_post_transaction_balances():
    expected = (
        FEATURE_SETS["A1"]
        + POST_TRANSACTION_BALANCE_FEATURES
    )

    assert get_feature_columns("B") == expected


def test_only_a0_is_approved_for_deployment():
    assert_deployable_feature_set("A0")

    with pytest.raises(ValueError, match="not approved"):
        assert_deployable_feature_set("A1")

    with pytest.raises(ValueError, match="not approved"):
        assert_deployable_feature_set("B")


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature set"):
        get_feature_columns("A3")


def test_feature_column_order_is_enforced():
    expected = get_feature_columns("A0")

    validate_feature_columns("A0", expected)

    wrong_order = list(expected)
    wrong_order[0], wrong_order[1] = (
        wrong_order[1],
        wrong_order[0],
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_feature_columns("A0", wrong_order)


def test_duplicate_feature_columns_are_rejected():
    expected = get_feature_columns("A0")
    duplicated = list(expected) + [expected[0]]

    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_columns("A0", duplicated)


def test_label_delay_excludes_labels_that_are_too_recent():
    # For step 100, the cutoff is 100 - 24 = 76.
    counts = past_label_counts(
        query_accounts=["A"],
        query_steps=[100],
        event_accounts=["A", "A", "A", "A"],
        event_steps=[75, 76, 77, 99],
        event_labels=[1, 1, 1, 1],
        label_delay_steps=LABEL_DELAY_STEPS,
    )

    assert counts.tolist() == [2]


def test_future_label_changes_do_not_change_earlier_features():
    query_accounts = ["A", "A"]
    query_steps = [100, 120]

    event_accounts = ["A", "A", "A", "A"]
    event_steps = [50, 76, 130, 150]
    event_labels = [1, 1, 1, 0]

    original = past_label_counts(
        query_accounts,
        query_steps,
        event_accounts,
        event_steps,
        event_labels,
        label_delay_steps=LABEL_DELAY_STEPS,
    )

    changed_labels = list(event_labels)
    changed_labels[2] = 0
    changed_labels[3] = 1

    perturbed = past_label_counts(
        query_accounts,
        query_steps,
        event_accounts,
        event_steps,
        changed_labels,
        label_delay_steps=LABEL_DELAY_STEPS,
    )

    assert np.array_equal(original, perturbed)
