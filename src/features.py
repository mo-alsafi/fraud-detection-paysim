
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.feature_contract import (
    A0_FEATURES,
    validate_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "paysim.parquet"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed"
    / "paysim_a0_features.parquet"
)

DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "a0_feature_build_report.json"
)

METADATA_COLUMNS = (
    "transaction_id",
    "step",
    "isFraud",
    "amount",
)

EXPECTED_OUTPUT_COLUMNS = list(METADATA_COLUMNS) + list(A0_FEATURES)


def _resolve_path(path: str | Path) -> Path:
    """Resolve a relative path from the project root."""
    result = Path(path)

    if not result.is_absolute():
        result = PROJECT_ROOT / result

    return result.resolve()


def _sql_literal(value: str | Path) -> str:
    """Safely quote a value as a DuckDB SQL string literal."""
    text = str(value).replace("\\", "/")
    return "'" + text.replace("'", "''") + "'"


def _utc_now() -> str:
    """Return a UTC timestamp in ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_a0_features(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
    report_path: str | Path | None = None,
) -> dict:
    """
    Build the A0 feature dataset using DuckDB.

    Historical activity features use transactions in the
    24 preceding simulation steps: [step - 24, step - 1].

    Same-step transactions are excluded because their relative
    order is not known.

    Returns a summary dictionary after validating the output.
    """
    source = _resolve_path(input_path)
    output = _resolve_path(output_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Input Parquet file not found: {source}"
        )

    if source == output:
        raise ValueError(
            "Input and output paths must be different."
        )

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. "
            "Use overwrite=True to rebuild it intentionally."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output.with_name(
        f"{output.stem}.tmp{output.suffix}"
    )

    temporary_output.unlink(missing_ok=True)

    temp_directory = output.parent / ".duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    if report_path is None:
        if output == DEFAULT_OUTPUT.resolve():
            report = DEFAULT_REPORT
        else:
            report = output.with_name(
                f"{output.stem}_build_report.json"
            )
    else:
        report = _resolve_path(report_path)

    report.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(database=":memory:")

    try:
        # Preserve Parquet input order when generating a row ID.
        connection.execute(
            "SET preserve_insertion_order = true"
        )
        connection.execute(
            f"SET temp_directory = "
            f"{_sql_literal(temp_directory)}"
        )

        source_sql = _sql_literal(source)

        # Confirm the columns needed for feature construction exist.
        source_columns = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({source_sql})"
        ).fetchall()

        available_columns = {
            row[0] for row in source_columns
        }

        required_columns = {
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "isFraud",
        }

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Source data is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # Validate essential input conditions before applying LN
        # or computing account histories.
        bad_input = connection.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE step IS NULL
                       OR type IS NULL
                       OR amount IS NULL
                       OR nameOrig IS NULL
                       OR nameDest IS NULL
                       OR isFraud IS NULL
                ) AS rows_with_null_required_fields,

                COUNT(*) FILTER (
                    WHERE NOT isfinite(amount)
                       OR amount < 0
                ) AS rows_with_invalid_amounts,

                COUNT(*) FILTER (
                    WHERE isFraud NOT IN (0, 1)
                ) AS rows_with_invalid_labels

            FROM read_parquet({source_sql})
            """
        ).fetchone()

        (
            rows_with_nulls,
            rows_with_invalid_amounts,
            rows_with_invalid_labels,
        ) = bad_input

        if rows_with_nulls:
            raise ValueError(
                f"Found {rows_with_nulls:,} rows with "
                "NULL values in required input fields."
            )

        if rows_with_invalid_amounts:
            raise ValueError(
                f"Found {rows_with_invalid_amounts:,} rows "
                "with negative or non-finite amounts."
            )

        if rows_with_invalid_labels:
            raise ValueError(
                f"Found {rows_with_invalid_labels:,} rows "
                "with invalid fraud labels."
            )

        # Step 1: Assign an identifier to each source row.
        # This identifier is created before joins and windows.
        connection.execute(
            f"""
            CREATE TEMP TABLE transaction_base AS
            SELECT
                CAST(
                    ROW_NUMBER() OVER () AS BIGINT
                ) AS transaction_id,

                step::INTEGER AS step,
                type::VARCHAR AS type,
                amount::DOUBLE AS amount,
                nameOrig::VARCHAR AS nameOrig,
                nameDest::VARCHAR AS nameDest,
                isFraud::UTINYINT AS isFraud

            FROM read_parquet({source_sql})
            """
        )

        source_count = connection.execute(
            "SELECT COUNT(*) FROM transaction_base"
        ).fetchone()[0]

        if source_count == 0:
            raise ValueError("Source dataset contains no rows.")

        # Step 2: Create one event for every account involved
        # in a transaction, counting a self-transaction once.
        connection.execute(
            """
            CREATE TEMP TABLE account_events AS

            SELECT
                transaction_id,
                step,
                nameOrig AS account_id,
                amount
            FROM transaction_base

            UNION ALL

            SELECT
                transaction_id,
                step,
                nameDest AS account_id,
                amount
            FROM transaction_base
            WHERE nameDest <> nameOrig
            """
        )

        # Step 3: Calculate historical activity for each account.
        # RANGE covers [current step - 24, current step - 1].
        # This excludes every event at the current simulation step.
        connection.execute(
            """
            CREATE TEMP TABLE account_history AS
            SELECT
                transaction_id,
                account_id,

                COUNT(*) OVER (
                    PARTITION BY account_id
                    ORDER BY step
                    RANGE BETWEEN 24 PRECEDING
                              AND 1 PRECEDING
                ) AS txn_count_prev_24_steps,

                COALESCE(
                    SUM(amount) OVER (
                        PARTITION BY account_id
                        ORDER BY step
                        RANGE BETWEEN 24 PRECEDING
                                  AND 1 PRECEDING
                    ),
                    0.0
                ) AS amount_sum_prev_24_steps

            FROM account_events
            """
        )

        # Step 4: Count earlier occurrences of each directed pair.
        # A pair from the same step does not make another row
        # at that step a repeat; only a strictly earlier step counts.
        connection.execute(
            """
            CREATE TEMP TABLE pair_history AS
            SELECT
                transaction_id,

                COUNT(*) OVER (
                    PARTITION BY nameOrig, nameDest
                    ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING
                              AND 1 PRECEDING
                ) AS prior_pair_count

            FROM transaction_base
            """
        )

        # Step 5: Calculate the sender's historical mean amount
        # from earlier outgoing transactions only.
        connection.execute(
            """
            CREATE TEMP TABLE sender_past_mean AS
            SELECT
                transaction_id,

                AVG(amount) OVER (
                    PARTITION BY nameOrig
                    ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING
                              AND 1 PRECEDING
                ) AS past_mean_outgoing_amount

            FROM transaction_base
            """
        )

        # Step 6: Assemble one output row per original transaction.
        connection.execute(
            """
            CREATE TEMP TABLE a0_features AS
            SELECT
                b.transaction_id,
                b.step,
                b.isFraud,
                b.amount,

                CAST(
                    (b.step - 1) % 24 AS INTEGER
                ) AS hour_of_day,

                b.type,

                LN(1.0 + b.amount) AS log_amount,

                (b.nameDest LIKE 'M%')
                    AS destination_is_merchant,

                COALESCE(
                    sender_hist.txn_count_prev_24_steps,
                    0
                )::BIGINT
                    AS sender_txn_count_prev_24_steps,

                COALESCE(
                    sender_hist.amount_sum_prev_24_steps,
                    0.0
                )::DOUBLE
                    AS sender_amount_sum_prev_24_steps,

                COALESCE(
                    dest_hist.txn_count_prev_24_steps,
                    0
                )::BIGINT
                    AS destination_txn_count_prev_24_steps,

                COALESCE(
                    dest_hist.amount_sum_prev_24_steps,
                    0.0
                )::DOUBLE
                    AS destination_amount_sum_prev_24_steps,

                (
                    COALESCE(pair.prior_pair_count, 0) = 0
                ) AS is_first_time_destination_for_sender,

                (
                    b.amount
                    / NULLIF(mean.past_mean_outgoing_amount, 0.0)
                ) AS amount_to_sender_past_mean_ratio

            FROM transaction_base AS b

            LEFT JOIN account_history AS sender_hist
                ON sender_hist.transaction_id = b.transaction_id
               AND sender_hist.account_id = b.nameOrig

            LEFT JOIN account_history AS dest_hist
                ON dest_hist.transaction_id = b.transaction_id
               AND dest_hist.account_id = b.nameDest

            LEFT JOIN pair_history AS pair
                ON pair.transaction_id = b.transaction_id

            LEFT JOIN sender_past_mean AS mean
                ON mean.transaction_id = b.transaction_id
            """
        )

        # Step 7: Verify the generated schema against the contract.
        actual_columns = [
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM a0_features"
            ).fetchall()
        ]

        if actual_columns != EXPECTED_OUTPUT_COLUMNS:
            raise RuntimeError(
                "Output columns do not match the expected schema.\n"
                f"Expected: {EXPECTED_OUTPUT_COLUMNS}\n"
                f"Actual: {actual_columns}"
            )

        validate_feature_columns(
            "A0",
            actual_columns[len(METADATA_COLUMNS):],
        )

        # Step 8: Validate row preservation and generated features.
        validation = connection.execute(
            """
            SELECT
                COUNT(*) AS output_rows,

                COUNT(DISTINCT transaction_id)
                    AS distinct_transaction_ids,

                COUNT(*) FILTER (
                    WHERE transaction_id IS NULL
                ) AS missing_transaction_ids,

                COUNT(*) FILTER (
                    WHERE sender_txn_count_prev_24_steps IS NULL
                       OR sender_amount_sum_prev_24_steps IS NULL
                       OR destination_txn_count_prev_24_steps IS NULL
                       OR destination_amount_sum_prev_24_steps IS NULL
                       OR is_first_time_destination_for_sender IS NULL
                       OR NOT isfinite(log_amount)
                ) AS invalid_required_features,

                COUNT(*) FILTER (
                    WHERE amount_to_sender_past_mean_ratio IS NULL
                ) AS undefined_past_mean_ratios

            FROM a0_features
            """
        ).fetchone()

        (
            output_rows,
            distinct_ids,
            missing_ids,
            invalid_required_features,
            undefined_ratios,
        ) = validation

        if output_rows != source_count:
            raise RuntimeError(
                f"Row count changed: source={source_count:,}, "
                f"output={output_rows:,}."
            )

        if distinct_ids != output_rows or missing_ids:
            raise RuntimeError(
                "Generated transaction IDs are not unique and complete."
            )

        if invalid_required_features:
            raise RuntimeError(
                f"Found {invalid_required_features:,} rows with "
                "invalid required features."
            )

        # Step 9: Write compressed Parquet to a temporary file.
        connection.execute(
            f"""
            COPY a0_features
            TO {_sql_literal(temporary_output)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        # Confirm the output file itself can be scanned.
        parquet_rows = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet("
            f"{_sql_literal(temporary_output)})"
        ).fetchone()[0]

        if parquet_rows != source_count:
            raise RuntimeError(
                "Written Parquet row count does not match source."
            )

        # Replace the final output only after successful validation.
        connection.close()
        os.replace(temporary_output, output)

        summary = {
            "built_at_utc": _utc_now(),
            "feature_set": "A0",
            "purpose": "primary_deployable_candidate",
            "input_path": (
                source.relative_to(PROJECT_ROOT).as_posix()
                if source.is_relative_to(PROJECT_ROOT)
                else str(source)
            ),
            "output_path": (
                output.relative_to(PROJECT_ROOT).as_posix()
                if output.is_relative_to(PROJECT_ROOT)
                else str(output)
            ),
            "row_count": int(output_rows),
            "distinct_transaction_ids": int(distinct_ids),
            "feature_columns": list(A0_FEATURES),
            "metadata_columns": list(METADATA_COLUMNS),
            "undefined_past_mean_ratios": int(undefined_ratios),
            "history_window": "[step - 24, step - 1]",
            "same_step_history_allowed": False,
            "transaction_id_method": (
                "row_number over insertion-preserved source Parquet order"
            ),
            "notes": [
                "Account history counts events where the account "
                "appears as origin or destination.",
                "The past mean amount uses earlier outgoing "
                "transactions only.",
                "Undefined past-mean ratios are intentionally NULL.",
                "The target label and metadata columns must not "
                "be passed to the model as features.",
            ],
            "output_size_bytes": output.stat().st_size,
        }

        with report.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")

        print("A0 feature build completed successfully.")
        print(f"Rows: {output_rows:,}")
        print(f"Features: {len(A0_FEATURES)}")
        print(f"Undefined past-mean ratios: {undefined_ratios:,}")
        print(f"Output: {output}")
        print(f"Report: {report}")

        return summary

    except Exception:
        try:
            connection.close()
        except Exception:
            pass

        temporary_output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    build_a0_features()
