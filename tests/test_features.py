
import duckdb
import pandas as pd
import pytest

from src.feature_contract import A0_FEATURES
from src.features import build_a0_features


def make_transactions() -> pd.DataFrame:
    """Create a small dataset with repeated pairs and account history."""
    rows = [
        # step, type, amount, sender, destination, isFraud
        (1, "TRANSFER", 10.0, "A", "B", 0),
        (1, "TRANSFER", 40.0, "A", "B", 0),
        (2, "TRANSFER", 20.0, "A", "C", 0),
        (3, "TRANSFER", 5.0, "X", "A", 0),
        (25, "TRANSFER", 30.0, "A", "B", 0),
        (26, "TRANSFER", 40.0, "A", "D", 0),
        (50, "TRANSFER", 15.0, "A", "E", 1),
    ]

    result = pd.DataFrame(
        rows,
        columns=[
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "isFraud",
        ],
    )

    result["oldbalanceOrg"] = 100.0
    result["newbalanceOrig"] = 90.0
    result["oldbalanceDest"] = 50.0
    result["newbalanceDest"] = 60.0
    result["isFlaggedFraud"] = 0

    return result[
        [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ]
    ]


def write_parquet(
    frame: pd.DataFrame,
    path,
) -> None:
    """Write a test fixture to Parquet with DuckDB."""
    connection = duckdb.connect(database=":memory:")

    try:
        connection.register("test_frame", frame)
        connection.execute(
            f"COPY test_frame TO '{str(path).replace(chr(92), '/')}' "
            "(FORMAT PARQUET)"
        )
    finally:
        connection.close()


def read_features(path) -> pd.DataFrame:
    """Read the small generated test output ordered by row ID."""
    connection = duckdb.connect(database=":memory:")

    try:
        path_sql = "'" + str(path).replace("\\", "/").replace("'", "''") + "'"

        return connection.execute(
            f"""
            SELECT *
            FROM read_parquet({path_sql})
            ORDER BY transaction_id
            """
        ).fetchdf()
    finally:
        connection.close()


def test_feature_builder_preserves_rows_and_contract(tmp_path):
    source = tmp_path / "input.parquet"
    output = tmp_path / "features.parquet"
    report = tmp_path / "report.json"

    write_parquet(make_transactions(), source)

    summary = build_a0_features(
        source,
        output,
        report_path=report,
    )

    features = read_features(output)

    assert len(features) == len(make_transactions())
    assert features["transaction_id"].is_unique
    assert list(A0_FEATURES) == [
        column for column in features.columns
        if column in A0_FEATURES
    ]
    assert summary["row_count"] == len(make_transactions())
    assert report.exists()


def test_same_step_transactions_do_not_enter_each_others_history(tmp_path):
    source = tmp_path / "input.parquet"
    output = tmp_path / "features.parquet"

    write_parquet(make_transactions(), source)

    build_a0_features(source, output)
    features = read_features(output).set_index("transaction_id")

    # The first two transactions occur at step 1.
    # Neither may use the other transaction as prior history.
    for transaction_id in (1, 2):
        row = features.loc[transaction_id]

        assert row["sender_txn_count_prev_24_steps"] == 0
        assert row["sender_amount_sum_prev_24_steps"] == 0.0
        assert pd.isna(row["amount_to_sender_past_mean_ratio"])


def test_history_window_obeys_the_24_step_boundary(tmp_path):
    source = tmp_path / "input.parquet"
    output = tmp_path / "features.parquet"

    write_parquet(make_transactions(), source)

    build_a0_features(source, output)
    features = read_features(output).set_index("transaction_id")

    # Row 5 is at step 25. Its prior 24-step range is steps 1–24.
    row_25 = features.loc[5]

    assert row_25["sender_txn_count_prev_24_steps"] == 4
    assert row_25["sender_amount_sum_prev_24_steps"] == pytest.approx(75.0)

    # Its destination B previously received amounts 10 and 40.
    assert row_25["destination_txn_count_prev_24_steps"] == 2
    assert row_25["destination_amount_sum_prev_24_steps"] == pytest.approx(50.0)

    # A -> B existed at an earlier step.
    assert not row_25["is_first_time_destination_for_sender"]

    # Mean of A's earlier outgoing transactions:
    # (10 + 40 + 20) / 3 = 23.333...
    assert row_25["amount_to_sender_past_mean_ratio"] == pytest.approx(
        30.0 / ((10.0 + 40.0 + 20.0) / 3.0)
    )

    # Row 6 is at step 26. Its window is steps 2–25,
    # so transactions from step 1 must no longer be counted.
    row_26 = features.loc[6]

    assert row_26["sender_txn_count_prev_24_steps"] == 3
    assert row_26["sender_amount_sum_prev_24_steps"] == pytest.approx(55.0)


def test_future_row_changes_do_not_change_earlier_features(tmp_path):
    baseline = make_transactions()

    # Add a future transaction involving a historical account and
    # an already-observed sender-destination pair.
    future = pd.DataFrame(
        [
            {
                "step": 80,
                "type": "TRANSFER",
                "amount": 999.0,
                "nameOrig": "A",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 1.0,
                "nameDest": "B",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 999.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        ]
    )

    altered_future = pd.concat(
        [baseline, future],
        ignore_index=True,
    )

    baseline_source = tmp_path / "baseline.parquet"
    altered_source = tmp_path / "altered.parquet"

    baseline_output = tmp_path / "baseline_features.parquet"
    altered_output = tmp_path / "altered_features.parquet"

    write_parquet(baseline, baseline_source)
    write_parquet(altered_future, altered_source)

    build_a0_features(baseline_source, baseline_output)
    build_a0_features(altered_source, altered_output)

    original_features = read_features(baseline_output)
    changed_features = read_features(altered_output)

    # Compare only the original transactions; the future row is extra.
    columns_to_compare = list(A0_FEATURES)

    pd.testing.assert_frame_equal(
        original_features[columns_to_compare].reset_index(drop=True),
        changed_features.iloc[:len(baseline)][columns_to_compare]
        .reset_index(drop=True),
        check_dtype=False,
    )


def test_invalid_input_amount_is_rejected(tmp_path):
    frame = make_transactions()
    frame.loc[0, "amount"] = -10.0

    source = tmp_path / "invalid.parquet"
    output = tmp_path / "features.parquet"

    write_parquet(frame, source)

    with pytest.raises(ValueError, match="invalid amounts"):
        build_a0_features(source, output)
