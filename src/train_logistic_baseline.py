
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.feature_contract import (
    A0_FEATURES,
    validate_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "paysim_a0_features.parquet"
)

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_a0_sgd.joblib"

REPORT_PATH = (
    PROJECT_ROOT / "reports"
    / "logistic_a0_validation_metrics.json"
)

CSV_REPORT_PATH = (
    PROJECT_ROOT / "reports"
    / "logistic_a0_validation_metrics.csv"
)

TRAIN_START, TRAIN_END = 1, 445
VALIDATION_START, VALIDATION_END = 446, 594

DEFAULT_ALERTS_PER_DAY = 500
DEFAULT_EPOCHS = 3
BATCH_VECTORS = 48
RANDOM_SEED = 42

METADATA_COLUMNS = (
    "transaction_id",
    "step",
    "isFraud",
    "amount",
)

# Hour and transaction type are categorical, not continuous.
HOUR_CATEGORIES = tuple(range(24))

TYPE_CATEGORIES = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

TYPE_TO_INDEX = {
    value: index
    for index, value in enumerate(TYPE_CATEGORIES)
}

BASE_NUMERIC_FEATURES = (
    "log_amount",
    "destination_is_merchant",
    "sender_txn_count_prev_24_steps",
    "sender_amount_sum_prev_24_steps",
    "destination_txn_count_prev_24_steps",
    "destination_amount_sum_prev_24_steps",
    "is_first_time_destination_for_sender",
    "amount_to_sender_past_mean_ratio",
)

# Keep this indicator separate from the ratio's imputed value.
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + (
    "ratio_is_missing",
)

# Account activity and the historical ratio are non-negative.
# log1p compresses their long right tails before standardization.
LOG1P_FEATURES = (
    "sender_txn_count_prev_24_steps",
    "sender_amount_sum_prev_24_steps",
    "destination_txn_count_prev_24_steps",
    "destination_amount_sum_prev_24_steps",
    "amount_to_sender_past_mean_ratio",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sql_literal(value: str | Path) -> str:
    """Quote a string literal safely for our DuckDB query."""
    value = str(value).replace("\\", "/")
    return "'" + value.replace("'", "''") + "'"


def select_sql(
    data_path: Path,
    lower_step: int,
    upper_step: int,
) -> str:
    """Build the fixed-column query for one temporal block."""
    columns = (
        list(METADATA_COLUMNS)
        + list(A0_FEATURES)
    )

    selected_columns = ", ".join(
        f'"{column}"'
        for column in columns
    )

    return f"""
        SELECT {selected_columns}
        FROM read_parquet({sql_literal(data_path)})
        WHERE step BETWEEN {int(lower_step)} AND {int(upper_step)}
    """


def iter_batches(
    connection: duckdb.DuckDBPyConnection,
    query: str,
):
    """
    Execute a query and yield manageable DataFrame batches.

    DuckDB's fetch_df_chunk prevents this helper from materializing
    the entire training dataset as a pandas DataFrame.
    """
    connection.execute(query)

    while True:
        batch = connection.fetch_df_chunk(
            vectors_per_chunk=BATCH_VECTORS
        )

        if batch is None or batch.empty:
            break

        yield batch


def prepare_numeric_matrix(
    frame: pd.DataFrame,
) -> np.ndarray:
    """
    Prepare numeric model inputs without fitting any statistics.

    The historical-mean ratio is imputed to zero with a separate
    missingness indicator, so zero and missing remain distinguishable.
    """
    data = {}

    for column in BASE_NUMERIC_FEATURES:
        if column == "amount_to_sender_past_mean_ratio":
            continue

        data[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        ).astype("float64")

    ratio = pd.to_numeric(
        frame["amount_to_sender_past_mean_ratio"],
        errors="raise",
    ).astype("float64")

    data["amount_to_sender_past_mean_ratio"] = (
        ratio.fillna(0.0)
    )

    data["ratio_is_missing"] = (
        ratio.isna().astype("float64")
    )

    numeric_frame = pd.DataFrame(
        data,
        index=frame.index,
    )[list(NUMERIC_FEATURES)]

    for column in LOG1P_FEATURES:
        values = numeric_frame[column].to_numpy(
            dtype=np.float64
        )

        if np.any(values < 0):
            raise ValueError(
                f"{column} contains negative values; "
                "log1p transformation is invalid."
            )

        numeric_frame[column] = np.log1p(values)

    values = numeric_frame.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite numeric features remain after preprocessing."
        )

    return values


def build_model_matrix(
    frame: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """Scale numeric inputs and one-hot encode hour and type."""
    numeric_values = prepare_numeric_matrix(frame)
    scaled_values = scaler.transform(numeric_values).astype(
        np.float32
    )

    hours = pd.to_numeric(
        frame["hour_of_day"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if np.any((hours < 0) | (hours > 23)):
        raise ValueError(
            "hour_of_day must contain values from 0 to 23."
        )

    type_codes = frame["type"].map(TYPE_TO_INDEX)

    if type_codes.isna().any():
        unexpected = sorted(
            set(frame.loc[type_codes.isna(), "type"])
        )
        raise ValueError(
            f"Unexpected transaction types: {unexpected}"
        )

    type_codes = type_codes.to_numpy(dtype=np.int64)

    n_rows = len(frame)
    n_numeric = len(NUMERIC_FEATURES)
    n_hours = len(HOUR_CATEGORIES)
    n_types = len(TYPE_CATEGORIES)

    matrix = np.zeros(
        (n_rows, n_numeric + n_hours + n_types),
        dtype=np.float32,
    )

    matrix[:, :n_numeric] = scaled_values

    row_indices = np.arange(n_rows)

    hour_start = n_numeric
    type_start = n_numeric + n_hours

    matrix[
        row_indices,
        hour_start + hours
    ] = 1.0

    matrix[
        row_indices,
        type_start + type_codes
    ] = 1.0

    return matrix


def alert_mask_at_daily_budget(
    scores: np.ndarray,
    steps: np.ndarray,
    *,
    alerts_per_day: int,
    period_start: int,
    period_end: int,
) -> tuple[np.ndarray, int]:
    """
    Choose the top-ranked alerts within each 24-step group.

    Partial groups at temporal boundaries receive a proportionally
    smaller alert budget. Ties are resolved by input order.
    """
    if (
        isinstance(alerts_per_day, bool)
        or alerts_per_day < 1
    ):
        raise ValueError("alerts_per_day must be positive.")

    scores = np.asarray(scores, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)

    if scores.ndim != 1 or steps.ndim != 1:
        raise ValueError("Scores and steps must be 1D arrays.")

    if len(scores) != len(steps) or len(scores) == 0:
        raise ValueError(
            "Scores and steps must have equal non-zero length."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    if np.any(
        (steps < period_start) | (steps > period_end)
    ):
        raise ValueError(
            "Steps lie outside the declared evaluation period."
        )

    alerts = np.zeros(len(scores), dtype=bool)
    total_capacity = 0

    simulation_days = (steps - 1) // 24

    for day in np.unique(simulation_days):
        positions = np.flatnonzero(simulation_days == day)

        day_start = int(day * 24 + 1)
        day_end = int(day * 24 + 24)

        covered_start = max(day_start, period_start)
        covered_end = min(day_end, period_end)

        covered_steps = max(
            0,
            covered_end - covered_start + 1,
        )

        budget = math.floor(
            alerts_per_day * covered_steps / 24
        )

        total_capacity += budget

        if budget == 0:
            continue

        count_to_alert = min(budget, len(positions))

        order = np.argsort(
            -scores[positions],
            kind="stable",
        )

        selected = positions[
            order[:count_to_alert]
        ]

        alerts[selected] = True

    return alerts, total_capacity


def calculate_validation_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    alerts: np.ndarray,
    amounts: np.ndarray,
    *,
    permitted_capacity: int,
) -> dict:
    """Calculate ranking, detection, and value-weighted metrics."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    alerts = np.asarray(alerts, dtype=bool)
    amounts = np.asarray(amounts, dtype=np.float64)

    lengths = {
        len(labels),
        len(scores),
        len(alerts),
        len(amounts),
    }

    if len(lengths) != 1 or len(labels) == 0:
        raise ValueError(
            "Labels, scores, alerts and amounts must have equal "
            "non-zero lengths."
        )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    if not np.isfinite(amounts).all() or np.any(amounts < 0):
        raise ValueError(
            "Transaction amounts must be finite and non-negative."
        )

    fraud = labels == 1
    legitimate = ~fraud

    true_positives = int(np.sum(alerts & fraud))
    false_positives = int(np.sum(alerts & legitimate))
    false_negatives = int(np.sum(~alerts & fraud))

    alert_count = int(np.sum(alerts))
    fraud_count = int(np.sum(fraud))
    legitimate_count = int(np.sum(legitimate))

    caught_value = float(
        amounts[alerts & fraud].sum()
    )
    total_fraud_value = float(
        amounts[fraud].sum()
    )

    return {
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "alert_budget_per_full_24_steps": int(
            permitted_capacity
        ),
        "alert_count": alert_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision_pct": (
            100.0 * true_positives / alert_count
            if alert_count else 0.0
        ),
        "recall_pct": (
            100.0 * true_positives / fraud_count
            if fraud_count else 0.0
        ),
        "false_positive_rate_pct": (
            100.0 * false_positives / legitimate_count
            if legitimate_count else 0.0
        ),
        "fraudulent_transaction_value_caught": caught_value,
        "total_fraudulent_transaction_value": total_fraud_value,
        "dollar_weighted_recall_pct": (
            100.0 * caught_value / total_fraud_value
            if total_fraud_value > 0 else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an A0 logistic-loss baseline and evaluate "
            "it on validation data."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of complete SGD passes over training data.",
    )

    parser.add_argument(
        "--alerts-per-day",
        type=int,
        default=DEFAULT_ALERTS_PER_DAY,
        help="Alert capacity per 24 simulation steps.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model and report files.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.alerts_per_day < 1:
        raise ValueError(
            "--alerts-per-day must be a positive integer."
        )

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"A0 feature file not found: {DATA_PATH}"
        )

    if MODEL_PATH.exists() and not args.overwrite:
        raise FileExistsError(
            f"Model already exists at {MODEL_PATH}. "
            "Use --overwrite only when intentionally rebuilding it."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_directory = (
        PROJECT_ROOT / "data" / "processed" / ".duckdb_tmp"
    )
    temp_directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(database=":memory:")

    try:
        connection.execute(
            "SET preserve_insertion_order = true"
        )
        connection.execute(
            f"SET temp_directory = {sql_literal(temp_directory)}"
        )

        # Confirm the complete A0 input contract before training.
        schema_rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet("
            f"{sql_literal(DATA_PATH)})"
        ).fetchall()

        available_columns = [row[0] for row in schema_rows]
        available_set = set(available_columns)

        required_columns = set(METADATA_COLUMNS) | set(A0_FEATURES)
        missing_columns = required_columns - available_set

        if missing_columns:
            raise ValueError(
                f"A0 feature file is missing columns: "
                f"{sorted(missing_columns)}"
            )

        observed_feature_order = [
            column
            for column in available_columns
            if column in set(A0_FEATURES)
        ]

        validate_feature_columns(
            "A0",
            observed_feature_order,
        )

        # Compute class weights from training labels only.
        train_count, positive_count = connection.execute(
            f"""
            SELECT
                COUNT(*),
                SUM(CAST(isFraud AS BIGINT))
            FROM read_parquet({sql_literal(DATA_PATH)})
            WHERE step BETWEEN {TRAIN_START} AND {TRAIN_END}
            """
        ).fetchone()

        train_count = int(train_count)
        positive_count = int(positive_count or 0)
        negative_count = train_count - positive_count

        if (
            train_count == 0
            or positive_count == 0
            or negative_count == 0
        ):
            raise ValueError(
                "Training data must contain both classes."
            )

        class_weights = {
            0: train_count / (2.0 * negative_count),
            1: train_count / (2.0 * positive_count),
        }

        print(f"Training rows: {train_count:,}")
        print(f"Fraud cases in training: {positive_count:,}")
        print(f"Class weights: {class_weights}")

        train_query = select_sql(
            DATA_PATH,
            TRAIN_START,
            TRAIN_END,
        )

        # Fit the scaler using training observations only.
        print("\nFitting numeric scaler on training batches...")

        scaler = StandardScaler()
        rows_scaled = 0

        for batch in iter_batches(connection, train_query):
            numeric_batch = prepare_numeric_matrix(batch)
            scaler.partial_fit(numeric_batch)
            rows_scaled += len(batch)

        if rows_scaled != train_count:
            raise RuntimeError(
                f"Scaler pass read {rows_scaled:,} rows, "
                f"expected {train_count:,}."
            )

        # Materialize one deterministically shuffled training table.
        # This avoids repeatedly sorting the entire dataset per epoch.
        print("\nPreparing a shuffled training table...")

        selected_columns = ", ".join(
            f'"{column}"'
            for column in list(METADATA_COLUMNS) + list(A0_FEATURES)
        )

        connection.execute(
            f"""
            CREATE TEMP TABLE shuffled_training AS
            SELECT {selected_columns}
            FROM read_parquet({sql_literal(DATA_PATH)})
            WHERE step BETWEEN {TRAIN_START} AND {TRAIN_END}
            ORDER BY hash(transaction_id + {RANDOM_SEED}),
                     transaction_id
            """
        )

        shuffled_rows = connection.execute(
            "SELECT COUNT(*) FROM shuffled_training"
        ).fetchone()[0]

        if int(shuffled_rows) != train_count:
            raise RuntimeError(
                "Shuffled training table has an unexpected row count."
            )

        # SGDClassifier with logistic loss is our scalable
        # logistic-regression baseline.
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            average=True,
        )

        print("\nTraining logistic-loss model...")

        first_batch = True

        for epoch in range(args.epochs):
            connection.execute(
                "SELECT * FROM shuffled_training"
            )

            rows_in_epoch = 0

            while True:
                batch = connection.fetch_df_chunk(
                    vectors_per_chunk=BATCH_VECTORS
                )

                if batch is None or batch.empty:
                    break

                X_batch = build_model_matrix(batch, scaler)

                y_batch = batch["isFraud"].to_numpy(
                    dtype=np.int64
                )

                if first_batch:
                    model.partial_fit(
                        X_batch,
                        y_batch,
                        classes=np.array([0, 1]),
                    )
                    first_batch = False
                else:
                    model.partial_fit(
                        X_batch,
                        y_batch,
                    )

                rows_in_epoch += len(batch)

            if rows_in_epoch != train_count:
                raise RuntimeError(
                    f"Epoch {epoch + 1} processed "
                    f"{rows_in_epoch:,} rows instead of "
                    f"{train_count:,}."
                )

            print(
                f"Completed epoch {epoch + 1}/{args.epochs}: "
                f"{rows_in_epoch:,} rows"
            )

        # Read validation data only after model fitting is complete.
        # The test block is never queried by this script.
        print("\nScoring validation data...")

        validation_query = select_sql(
            DATA_PATH,
            VALIDATION_START,
            VALIDATION_END,
        ) + ' ORDER BY "transaction_id"'

        validation = connection.execute(
            validation_query
        ).fetchdf()

    finally:
        connection.close()

    if validation.empty:
        raise RuntimeError("Validation dataset is empty.")

    # The decision function is a ranking score, not a calibrated
    # fraud probability.
    X_validation = build_model_matrix(validation, scaler)

    scores = model.decision_function(X_validation)

    y_validation = validation["isFraud"].to_numpy(
        dtype=np.int64
    )

    validation_steps = validation["step"].to_numpy(
        dtype=np.int64
    )

    validation_amounts = validation["amount"].to_numpy(
        dtype=np.float64
    )

    alerts, total_capacity = alert_mask_at_daily_budget(
        scores,
        validation_steps,
        alerts_per_day=args.alerts_per_day,
        period_start=VALIDATION_START,
        period_end=VALIDATION_END,
    )

    metrics = calculate_validation_metrics(
        y_validation,
        scores,
        alerts,
        validation_amounts,
        permitted_capacity=total_capacity,
    )

    if metrics["alert_count"] > total_capacity:
        raise RuntimeError(
            "Alert policy exceeded its computed validation capacity."
        )

    bundle = {
        "estimator": model,
        "scaler": scaler,
        "feature_set": "A0",
        "feature_columns": list(A0_FEATURES),
        "numeric_features_after_preprocessing": list(NUMERIC_FEATURES),
        "log1p_features": list(LOG1P_FEATURES),
        "hour_categories": list(HOUR_CATEGORIES),
        "type_categories": list(TYPE_CATEGORIES),
        "metadata": {
            "train_steps": [TRAIN_START, TRAIN_END],
            "validation_steps": [
                VALIDATION_START,
                VALIDATION_END,
            ],
            "epochs": args.epochs,
            "random_seed": RANDOM_SEED,
            "class_weights": class_weights,
            "alerts_per_day": args.alerts_per_day,
            "score_type": "decision_function; ranking score, not calibrated probability",
        },
    }

    joblib.dump(bundle, MODEL_PATH)

    report = {
        "evaluated_at_utc": utc_now(),
        "model": "SGDClassifier with logistic loss",
        "purpose": "A0 class-weighted logistic-regression baseline",
        "feature_set": "A0",
        "training": {
            "start_step": TRAIN_START,
            "end_step": TRAIN_END,
            "rows": train_count,
            "fraud_cases": positive_count,
            "legitimate_cases": negative_count,
            "class_weights": {
                str(key): float(value)
                for key, value in class_weights.items()
            },
            "epochs": args.epochs,
            "random_seed": RANDOM_SEED,
        },
        "validation": {
            "start_step": VALIDATION_START,
            "end_step": VALIDATION_END,
            "rows": int(len(validation)),
            "fraud_cases": int(y_validation.sum()),
            "legitimate_cases": int((y_validation == 0).sum()),
            **metrics,
        },
        "preprocessing": {
            "numeric_scaler_fit_on": "training only",
            "categorical_encoding": "one-hot hour and transaction type",
            "missing_ratio_policy": (
                "fill with zero after creating a separate missingness indicator"
            ),
            "skew_handling": "log1p on non-negative activity features and historical ratio",
        },
        "versions": {
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "model_path": MODEL_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "note": (
            "Validation-only model evaluation. The test block was "
            "not used for fitting or scoring by this script. "
            "Historical exploratory analysis has already exposed "
            "test-period aggregate statistics."
        ),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    pd.DataFrame([metrics]).to_csv(
        CSV_REPORT_PATH,
        index=False,
    )

    print("\nValidation results")
    print("------------------")
    print(f"Average precision: {metrics['average_precision']:.6f}")
    print(f"Alert budget available: {total_capacity:,}")
    print(f"Alerts used: {metrics['alert_count']:,}")
    print(f"Precision: {metrics['precision_pct']:.4f}%")
    print(f"Recall: {metrics['recall_pct']:.4f}%")
    print(
        "False-positive rate: "
        f"{metrics['false_positive_rate_pct']:.4f}%"
    )
    print(
        "Dollar-weighted recall: "
        f"{metrics['dollar_weighted_recall_pct']:.4f}%"
    )
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {REPORT_PATH}")
    print(f"CSV metrics saved to: {CSV_REPORT_PATH}")


if __name__ == "__main__":
    main()
