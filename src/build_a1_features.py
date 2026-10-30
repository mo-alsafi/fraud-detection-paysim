
"""Build A1 features by adding pre-transaction balances to the A0 table.

A1 adds:
- oldbalanceOrg
- oldbalanceDest
- drain_ratio = amount / oldbalanceOrg, where oldbalanceOrg > 0
- amount_to_oldbalanceDest_ratio = amount / oldbalanceDest,
  where oldbalanceDest > 0

Post-transaction balance columns are deliberately excluded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "processed" / "paysim.parquet"
A0_PATH = ROOT / "data" / "processed" / "paysim_a0_features.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "paysim_a1_features.parquet"
REPORT_PATH = ROOT / "reports" / "a1_feature_build_report.json"

A0_FEATURE_COLUMNS = [
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
]

MISSINGNESS_FEATURE = "amount_to_sender_past_mean_ratio_is_missing"

A1_BALANCE_FEATURE_COLUMNS = [
    "oldbalanceOrg",
    "oldbalanceDest",
    "drain_ratio",
    "amount_to_oldbalanceDest_ratio",
]

A1_FEATURE_COLUMNS = (
    A0_FEATURE_COLUMNS
    + [MISSINGNESS_FEATURE]
    + A1_BALANCE_FEATURE_COLUMNS
)


def sql_path(path: Path) -> str:
    """Return an absolute path safely quoted as a DuckDB SQL string."""
    value = str(path.resolve()).replace("\\", "/")
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw Parquet not found: {RAW_PATH}")

    if not A0_PATH.exists():
        raise FileNotFoundError(
            f"A0 feature table not found: {A0_PATH}\n"
            "Build the A0 features before building A1."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Avoid a stale output being mistaken for a new successful build.
    OUTPUT_PATH.unlink(missing_ok=True)

    raw_sql = sql_path(RAW_PATH)
    a0_sql = sql_path(A0_PATH)
    output_sql = sql_path(OUTPUT_PATH)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = true")

    try:
        # The original A0 build report states that transaction_id was
        # assigned with ROW_NUMBER() OVER () on the insertion-preserved
        # source. Recreate the same key and verify its alignment below.
        con.execute(
            f"""
            CREATE TEMP VIEW raw_with_id AS
            SELECT
                CAST(ROW_NUMBER() OVER () AS BIGINT) AS transaction_id,
                step,
                amount,
                isFraud,
                oldbalanceOrg,
                oldbalanceDest
            FROM read_parquet({raw_sql})
            """
        )

        alignment = con.execute(
            f"""
            SELECT
                COUNT(*) AS a0_rows,
                COUNT(*) FILTER (
                    WHERE r.transaction_id IS NULL
                ) AS missing_raw_matches,
                COUNT(*) FILTER (
                    WHERE
                        CAST(a.step AS BIGINT)
                            IS DISTINCT FROM CAST(r.step AS BIGINT)
                        OR CAST(a.amount AS DOUBLE)
                            IS DISTINCT FROM CAST(r.amount AS DOUBLE)
                        OR CAST(a.isFraud AS INTEGER)
                            IS DISTINCT FROM CAST(r.isFraud AS INTEGER)
                ) AS mismatched_transactions
            FROM read_parquet({a0_sql}) AS a
            LEFT JOIN raw_with_id AS r
                USING (transaction_id)
            """
        ).fetchone()

        a0_rows, missing_matches, mismatches = map(int, alignment)

        print("Transaction alignment check")
        print("---------------------------")
        print(f"A0 rows checked: {a0_rows:,}")
        print(f"Missing raw matches: {missing_matches:,}")
        print(f"Transactions with mismatched metadata: {mismatches:,}")

        if missing_matches != 0 or mismatches != 0:
            raise RuntimeError(
                "A0 transaction IDs do not align with the raw source. "
                "A1 was not built. Do not bypass this check; inspect how "
                "transaction_id was constructed in src/features.py."
            )

        con.execute(
            f"""
            COPY (
                SELECT
                    a.transaction_id,
                    a.step,
                    a.isFraud,
                    a.amount,

                    a.hour_of_day,
                    a."type",
                    a.log_amount,
                    a.destination_is_merchant,
                    a.sender_txn_count_prev_24_steps,
                    a.sender_amount_sum_prev_24_steps,
                    a.destination_txn_count_prev_24_steps,
                    a.destination_amount_sum_prev_24_steps,
                    a.is_first_time_destination_for_sender,
                    a.amount_to_sender_past_mean_ratio,

                    CASE
                        WHEN a.amount_to_sender_past_mean_ratio IS NULL
                            THEN 1
                        ELSE 0
                    END::TINYINT
                        AS {MISSINGNESS_FEATURE},

                    CAST(r.oldbalanceOrg AS DOUBLE)
                        AS oldbalanceOrg,

                    CAST(r.oldbalanceDest AS DOUBLE)
                        AS oldbalanceDest,

                    CASE
                        WHEN r.oldbalanceOrg > 0
                            THEN CAST(a.amount AS DOUBLE)
                                 / CAST(r.oldbalanceOrg AS DOUBLE)
                        ELSE NULL
                    END AS drain_ratio,

                    CASE
                        WHEN r.oldbalanceDest > 0
                            THEN CAST(a.amount AS DOUBLE)
                                 / CAST(r.oldbalanceDest AS DOUBLE)
                        ELSE NULL
                    END AS amount_to_oldbalanceDest_ratio

                FROM read_parquet({a0_sql}) AS a
                INNER JOIN raw_with_id AS r
                    USING (transaction_id)
            )
            TO {output_sql}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        output_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT transaction_id) AS distinct_transaction_ids,
                MIN(step) AS min_step,
                MAX(step) AS max_step,
                COUNT(*) FILTER (
                    WHERE drain_ratio IS NULL
                ) AS undefined_drain_ratio,
                COUNT(*) FILTER (
                    WHERE amount_to_oldbalanceDest_ratio IS NULL
                ) AS undefined_destination_ratio,
                COUNT(*) FILTER (
                    WHERE oldbalanceOrg IS NULL
                ) AS missing_sender_balance,
                COUNT(*) FILTER (
                    WHERE oldbalanceDest IS NULL
                ) AS missing_destination_balance
            FROM read_parquet({output_sql})
            """
        ).fetchone()

        (
            row_count,
            distinct_ids,
            min_step,
            max_step,
            undefined_drain_ratio,
            undefined_destination_ratio,
            missing_sender_balance,
            missing_destination_balance,
        ) = output_stats

        row_count = int(row_count)
        distinct_ids = int(distinct_ids)

        if row_count != a0_rows or distinct_ids != row_count:
            raise RuntimeError(
                "A1 row-count or transaction-ID validation failed."
            )

        report = {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_set": "A1",
            "input_raw": str(RAW_PATH.relative_to(ROOT)),
            "input_a0": str(A0_PATH.relative_to(ROOT)),
            "output": str(OUTPUT_PATH.relative_to(ROOT)),
            "row_count": row_count,
            "distinct_transaction_ids": distinct_ids,
            "min_step": int(min_step),
            "max_step": int(max_step),
            "raw_a0_alignment": {
                "missing_raw_matches": missing_matches,
                "mismatched_transactions": mismatches,
            },
            "feature_columns": A1_FEATURE_COLUMNS,
            "undefined_drain_ratio": int(undefined_drain_ratio),
            "undefined_destination_ratio": int(
                undefined_destination_ratio
            ),
            "missing_sender_balance": int(missing_sender_balance),
            "missing_destination_balance": int(
                missing_destination_balance
            ),
            "post_transaction_balance_features_included": False,
            "output_size_bytes": OUTPUT_PATH.stat().st_size,
            "notes": [
                "A1 contains A0 plus pre-transaction balance features.",
                "Ratios are NULL when their denominator is not positive.",
                "Post-transaction balances are excluded.",
                "Raw transaction IDs and account names are not model features.",
                "Alignment validation passed before the join was written.",
            ],
        }

        with REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\nA1 feature build successful")
        print("---------------------------")
        print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")
        print(f"Rows: {row_count:,}")
        print(f"Unique transaction IDs: {distinct_ids:,}")
        print(f"Undefined drain ratios: {int(undefined_drain_ratio):,}")
        print(
            "Undefined destination ratios: "
            f"{int(undefined_destination_ratio):,}"
        )
        print(
            "Missing sender balances: "
            f"{int(missing_sender_balance):,}"
        )
        print(
            "Missing destination balances: "
            f"{int(missing_destination_balance):,}"
        )
        print(f"Report: {REPORT_PATH.relative_to(ROOT)}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
