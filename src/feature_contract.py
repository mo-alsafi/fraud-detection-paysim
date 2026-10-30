
from __future__ import annotations


# Features that do not directly use the four PaySim balance columns.
A0_FEATURES = (
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
)

# Pre-transaction balances and ratios.
PRE_TRANSACTION_BALANCE_FEATURES = (
    "oldbalanceOrg",
    "oldbalanceDest",
    "drain_ratio",
    "amount_to_oldbalanceDest_ratio",
)

# Post-transaction balances: never eligible for pre-approval scoring.
POST_TRANSACTION_BALANCE_FEATURES = (
    "newbalanceOrig",
    "newbalanceDest",
)

FEATURE_SETS = {
    "A0": A0_FEATURES,
    "A1": A0_FEATURES + PRE_TRANSACTION_BALANCE_FEATURES,
    "B": (
        A0_FEATURES
        + PRE_TRANSACTION_BALANCE_FEATURES
        + POST_TRANSACTION_BALANCE_FEATURES
    ),
}

FEATURE_SET_PURPOSE = {
    "A0": "primary_deployable_candidate",
    "A1": "diagnostic_pre_transaction_balance_experiment",
    "B": "leakage_demonstration_only",
}

# Only A0 is initially approved as a deployable candidate.
DEPLOYABLE_FEATURE_SET = "A0"

# These fields may be required to construct features or evaluate
# models, but must not be passed as raw model features.
IDENTIFIER_COLUMNS = (
    "nameOrig",
    "nameDest",
)

TARGET_COLUMNS = (
    "isFraud",
)

BASELINE_ONLY_COLUMNS = (
    "isFlaggedFraud",
)

POST_TRANSACTION_COLUMNS = (
    "newbalanceOrig",
    "newbalanceDest",
)

FORBIDDEN_MODEL_COLUMNS = frozenset(
    IDENTIFIER_COLUMNS
    + TARGET_COLUMNS
    + POST_TRANSACTION_COLUMNS
)


def get_feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return the declared feature columns in their fixed order."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        valid = ", ".join(FEATURE_SETS)
        raise ValueError(
            f"Unknown feature set {feature_set!r}. "
            f"Choose one of: {valid}."
        ) from exc


def get_feature_set_purpose(feature_set: str) -> str:
    """Return the declared purpose of a feature set."""
    try:
        return FEATURE_SET_PURPOSE[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature set: {feature_set!r}"
        ) from exc


def assert_deployable_feature_set(feature_set: str) -> None:
    """
    Reject feature sets not approved for pre-approval deployment.

    A1 remains diagnostic until its data-reliability concerns are
    resolved. B is never approved for pre-approval deployment.
    """
    if feature_set != DEPLOYABLE_FEATURE_SET:
        purpose = get_feature_set_purpose(feature_set)

        raise ValueError(
            f"Feature set {feature_set!r} is not approved for "
            f"deployment. Declared purpose: {purpose!r}. "
            f"The currently approved candidate is "
            f"{DEPLOYABLE_FEATURE_SET!r}."
        )


def validate_feature_columns(
    feature_set: str,
    actual_columns: tuple[str, ...] | list[str],
) -> None:
    """
    Check that the generated feature matrix matches the declared
    schema exactly, including column order.
    """
    expected = get_feature_columns(feature_set)
    actual = tuple(actual_columns)

    if len(actual) != len(set(actual)):
        raise ValueError(
            "Feature matrix contains duplicate column names."
        )

    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]

        raise ValueError(
            f"Feature schema mismatch for {feature_set!r}. "
            f"Missing={missing}; unexpected={unexpected}; "
            f"expected_order={expected}; actual_order={actual}."
        )

    forbidden = set(actual) & FORBIDDEN_MODEL_COLUMNS

    if forbidden:
        raise ValueError(
            f"Forbidden raw columns entered the feature matrix: "
            f"{sorted(forbidden)}."
        )
