
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paysim.parquet"
)
MANIFEST_PATH = (
    PROJECT_ROOT / "reports" / "data_retrieval_manifest.json"
)
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "data_validation_report.json"
)

EXPECTED_ROW_COUNT = 6_362_620
EXPECTED_STEP_MIN = 1
EXPECTED_STEP_MAX = 743

EXPECTED_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

EXPECTED_SCHEMA = [
    ("step", "INTEGER"),
    ("type", "VARCHAR"),
    ("amount", "DOUBLE"),
    ("nameOrig", "VARCHAR"),
    ("oldbalanceOrg", "DOUBLE"),
    ("newbalanceOrig", "DOUBLE"),
    ("nameDest", "VARCHAR"),
    ("oldbalanceDest", "DOUBLE"),
    ("newbalanceDest", "DOUBLE"),
    ("isFraud", "UTINYINT"),
    ("isFlaggedFraud", "UTINYINT"),
]


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest_row_count() -> int:
    """Return the observed source row count from the manifest."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Retrieval manifest is missing. Complete the previous "
            "task before validating the processed dataset."
        )

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    return int(
        manifest["file"]["row_count_excluding_header"]
    )


def collect_summary(connection: duckdb.DuckDBPyConnection) -> dict:
    """Collect the data-quality statistics needed by this project."""

    result = connection.execute(
        """
        SELECT
            COUNT(*) AS row_count,

            COUNT(*) FILTER (
                WHERE step IS NULL
                   OR type IS NULL
                   OR amount IS NULL
                   OR nameOrig IS NULL
                   OR oldbalanceOrg IS NULL
                   OR newbalanceOrig IS NULL
                   OR nameDest IS NULL
                   OR oldbalanceDest IS NULL
                   OR newbalanceDest IS NULL
                   OR isFraud IS NULL
                   OR isFlaggedFraud IS NULL
            ) AS rows_with_nulls,

            COUNT(*) FILTER (
                WHERE isFraud NOT IN (0, 1)
                   OR isFlaggedFraud NOT IN (0, 1)
            ) AS rows_with_invalid_labels,

            MIN(step) AS min_step,
            MAX(step) AS max_step,
            COUNT(DISTINCT step) AS distinct_steps,
            COUNT(DISTINCT type) AS distinct_type_count

        FROM read_parquet(?)
        """,
        [str(PARQUET_PATH)],
    )

    columns = [description[0] for description in result.description]
    values = result.fetchone()

    return dict(zip(columns, values))


def collect_transaction_types(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    """Return the observed set of transaction types."""
    rows = connection.execute(
        """
        SELECT DISTINCT type
        FROM read_parquet(?)
        """,
        [str(PARQUET_PATH)],
    ).fetchall()

    return {row[0] for row in rows}


def collect_schema(
    connection: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str]]:
    """Return the observed column names and DuckDB types."""
    rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(PARQUET_PATH)],
    ).fetchall()

    return [(row[0], row[1]) for row in rows]


def get_validation_errors(
    summary: dict,
    observed_types: set[str],
    actual_schema: list[tuple[str, str]],
    manifest_row_count: int,
) -> list[str]:
    """
    Compare observed statistics with the project's data contract.

    Return all detected problems, instead of stopping at the
    first failure. This makes debugging easier.
    """
    errors = []

    if summary["row_count"] != EXPECTED_ROW_COUNT:
        errors.append(
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"observed {summary['row_count']:,}."
        )

    if manifest_row_count != EXPECTED_ROW_COUNT:
        errors.append(
            "The source row count recorded in the manifest "
            f"is {manifest_row_count:,}, not "
            f"{EXPECTED_ROW_COUNT:,}."
        )

    if summary["row_count"] != manifest_row_count:
        errors.append(
            "The Parquet row count does not match the source "
            "row count recorded in the manifest."
        )

    if summary["min_step"] != EXPECTED_STEP_MIN:
        errors.append(
            f"Expected minimum step {EXPECTED_STEP_MIN}, "
            f"observed {summary['min_step']}."
        )

    if summary["max_step"] != EXPECTED_STEP_MAX:
        errors.append(
            f"Expected maximum step {EXPECTED_STEP_MAX}, "
            f"observed {summary['max_step']}."
        )

    # Supplemental completeness check: every integer step
    # in the expected range should appear in the dataset.
    expected_distinct_steps = (
        EXPECTED_STEP_MAX - EXPECTED_STEP_MIN + 1
    )

    if summary["distinct_steps"] != expected_distinct_steps:
        errors.append(
            f"Expected {expected_distinct_steps} distinct steps, "
            f"observed {summary['distinct_steps']}."
        )

    if summary["rows_with_nulls"] != 0:
        errors.append(
            f"Found {summary['rows_with_nulls']:,} rows with NULLs."
        )

    if summary["rows_with_invalid_labels"] != 0:
        errors.append(
            "Found "
            f"{summary['rows_with_invalid_labels']:,} rows "
            "with non-binary fraud labels."
        )

    if observed_types != EXPECTED_TYPES:
        missing = sorted(EXPECTED_TYPES - observed_types)
        unexpected = sorted(observed_types - EXPECTED_TYPES)

        errors.append(
            f"Unexpected transaction-type set. "
            f"Missing={missing}; unexpected={unexpected}."
        )

    if actual_schema != EXPECTED_SCHEMA:
        errors.append(
            "The Parquet schema does not match the expected "
            "column order or data types."
        )

    return errors


def main() -> None:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Processed Parquet file not found: "
            f"{PARQUET_PATH.relative_to(PROJECT_ROOT)}"
        )

    manifest_row_count = load_manifest_row_count()

    connection = duckdb.connect(database=":memory:")

    try:
        summary = collect_summary(connection)
        observed_types = collect_transaction_types(connection)
        actual_schema = collect_schema(connection)
    finally:
        connection.close()

    errors = get_validation_errors(
        summary=summary,
        observed_types=observed_types,
        actual_schema=actual_schema,
        manifest_row_count=manifest_row_count,
    )

    report = {
        "checked_at_utc": utc_now(),
        "dataset_path": "data/processed/paysim.parquet",
        "passed": not errors,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "manifest_row_count": manifest_row_count,
        "observed_summary": summary,
        "observed_transaction_types": sorted(observed_types),
        "observed_schema": [
            {"column": name, "type": dtype}
            for name, dtype in actual_schema
        ],
        "validation_errors": errors,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)
        file.write("\n")

    print("Data validation report:")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Rows with NULLs: {summary['rows_with_nulls']:,}")
    print(
        "Rows with invalid labels: "
        f"{summary['rows_with_invalid_labels']:,}"
    )
    print(
        f"Step range: {summary['min_step']} "
        f"to {summary['max_step']}"
    )
    print(f"Distinct steps: {summary['distinct_steps']}")
    print(f"Transaction types: {sorted(observed_types)}")
    print(f"Passed: {not errors}")
    print(
        "Report: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    if errors:
        print("\nValidation failures:")
        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
