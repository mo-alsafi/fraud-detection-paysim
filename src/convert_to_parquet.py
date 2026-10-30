from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    "data/raw/PS_20174392719_1491204439457_log.csv"
)
DEFAULT_OUTPUT = "data/processed/paysim.parquet"

EXPECTED_COLUMNS = [
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

# Explicit schema: column name -> DuckDB data type
COLUMN_TYPES = {
    "step": "INTEGER",
    "type": "VARCHAR",
    "amount": "DOUBLE",
    "nameOrig": "VARCHAR",
    "oldbalanceOrg": "DOUBLE",
    "newbalanceOrig": "DOUBLE",
    "nameDest": "VARCHAR",
    "oldbalanceDest": "DOUBLE",
    "newbalanceDest": "DOUBLE",
    "isFraud": "UTINYINT",
    "isFlaggedFraud": "UTINYINT",
}


def resolve_path(path: str) -> Path:
    """Resolve relative paths from the project root."""
    result = Path(path)

    if not result.is_absolute():
        result = PROJECT_ROOT / result

    return result.resolve()


def sql_literal(value: str) -> str:
    """Safely quote a string as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def verify_csv_header(csv_path: Path) -> None:
    """Confirm that the source CSV has the expected columns."""
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)

    if header != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected CSV header.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Observed: {header}"
        )


def get_expected_row_count() -> int | None:
    """Read the observed source row count from our manifest."""
    manifest_path = (
        PROJECT_ROOT
        / "reports"
        / "data_retrieval_manifest.json"
    )

    if not manifest_path.exists():
        return None

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    return manifest["file"]["row_count_excluding_header"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the PaySim CSV to typed Parquet."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Raw CSV path relative to the project root.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Processed Parquet path relative to the project root.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output after successful conversion.",
    )

    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}"
        )

    if input_path == output_path:
        raise ValueError("Input and output paths must differ.")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: "
            f"{output_path.relative_to(PROJECT_ROOT)}\n"
            "Use --overwrite to replace it after validation."
        )

    verify_csv_header(input_path)

    expected_rows = get_expected_row_count()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first, preserving any existing
    # output until the new conversion passes its checks.
    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )

    temp_path.unlink(missing_ok=True)

    columns_sql = "{ " + ", ".join(
        f"{sql_literal(name)}: {sql_literal(dtype)}"
        for name, dtype in COLUMN_TYPES.items()
    ) + " }"

    copy_sql = f"""
        COPY (
            SELECT *
            FROM read_csv(
                {sql_literal(input_path.as_posix())},
                header = true,
                delim = ',',
                auto_detect = false,
                columns = {columns_sql},
                strict_mode = true
            )
        )
        TO {sql_literal(temp_path.as_posix())}
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    connection = duckdb.connect(database=":memory:")

    try:
        print("Starting CSV-to-Parquet conversion...")
        connection.execute(copy_sql)

        # Verify the resulting Parquet schema.
        describe_sql = (
            "DESCRIBE SELECT * FROM read_parquet("
            f"{sql_literal(temp_path.as_posix())})"
        )

        actual_schema = [
            (row[0], row[1])
            for row in connection.execute(describe_sql).fetchall()
        ]

        expected_schema = list(COLUMN_TYPES.items())

        if actual_schema != expected_schema:
            raise RuntimeError(
                "Parquet schema does not match the declared schema.\n"
                f"Expected: {expected_schema}\n"
                f"Observed: {actual_schema}"
            )

        # Collect basic quality checks in one Parquet scan.
        validation_sql = f"""
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
                ) AS invalid_label_rows,

                MIN(step) AS min_step,
                MAX(step) AS max_step,
                COUNT(DISTINCT type) AS distinct_types

            FROM read_parquet(
                {sql_literal(temp_path.as_posix())}
            );
        """

        (
            row_count,
            rows_with_nulls,
            invalid_label_rows,
            min_step,
            max_step,
            distinct_types,
        ) = connection.execute(validation_sql).fetchone()

        if (
            expected_rows is not None
            and row_count != expected_rows
        ):
            raise RuntimeError(
                "Row count differs from the retrieval manifest: "
                f"expected {expected_rows:,}, observed {row_count:,}."
            )

        if rows_with_nulls != 0:
            raise RuntimeError(
                f"Found {rows_with_nulls:,} rows containing nulls."
            )

        if invalid_label_rows != 0:
            raise RuntimeError(
                f"Found {invalid_label_rows:,} rows with invalid labels."
            )

    except Exception:
        connection.close()
        temp_path.unlink(missing_ok=True)
        raise

    else:
        connection.close()
        temp_path.replace(output_path)

    print("\nConversion completed successfully.")
    print("Output:", output_path.relative_to(PROJECT_ROOT))
    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(actual_schema)}")
    print(f"Rows with nulls: {rows_with_nulls:,}")
    print(f"Invalid-label rows: {invalid_label_rows:,}")
    print(f"Step range: {min_step} to {max_step}")
    print(f"Distinct transaction types: {distinct_types}")
    print(f"Parquet size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()