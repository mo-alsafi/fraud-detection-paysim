
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "Synthetic Financial Datasets For Fraud Detection"
DATASET_SLUG = "ealaxi/paysim1"
DATASET_VERSION = "2"
DATASET_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"

EXPECTED_ROW_COUNT = 6_362_620

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


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute SHA-256 without reading the whole file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def inspect_csv(path: Path) -> dict:
    """
    Count CSV data records and inspect the header.

    Uses csv.reader to process one record at a time instead
    of loading the dataset into a DataFrame.
    """
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)

        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError("The CSV file is empty.") from exc

        row_count = 0
        wrong_field_count = 0

        for row in reader:
            row_count += 1

            if len(row) != len(columns):
                wrong_field_count += 1

    return {
        "row_count_excluding_header": row_count,
        "column_count": len(columns),
        "columns": columns,
        "records_with_wrong_field_count": wrong_field_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a PaySim data retrieval manifest."
    )

    parser.add_argument(
        "csv_path",
        help="CSV path relative to the repository root, "
             "or an absolute path inside the repository.",
    )

    parser.add_argument(
        "--retrieved-at-utc",
        required=True,
        help="UTC timestamp captured when retrieval began.",
    )

    parser.add_argument(
        "--retrieval-method",
        choices=["kaggle_cli", "manual"],
        required=True,
    )

    parser.add_argument(
        "--dataset-version",
        default=DATASET_VERSION,
        help="Kaggle dataset version used for this download.",
    )

    args = parser.parse_args()

    # Validate and normalize the supplied timestamp.
    try:
        parsed_timestamp = datetime.fromisoformat(
            args.retrieved_at_utc.replace("Z", "+00:00")
        )

        if parsed_timestamp.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        retrieved_at_utc = (
            parsed_timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except ValueError as exc:
        raise SystemExit(
            f"Invalid retrieval timestamp: {exc}"
        ) from exc

    csv_path = Path(args.csv_path)

    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    csv_path = csv_path.resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # Store a project-relative path, never a machine-specific path.
    try:
        relative_path = csv_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(
            "The CSV must be inside the project directory."
        ) from exc

    print("Calculating SHA-256...")
    file_hash = sha256_file(csv_path)

    print("Counting CSV records and inspecting columns...")
    csv_info = inspect_csv(csv_path)

    row_count = csv_info["row_count_excluding_header"]
    columns = csv_info["columns"]

    manifest = {
        "project": "fraud-detection-paysim",
        "dataset": {
            "name": DATASET_NAME,
            "slug": DATASET_SLUG,
            "version": args.dataset_version,
            "source_url": DATASET_URL,
            "synthetic": True,
        },
        "retrieval": {
            "method": args.retrieval_method,
            "retrieved_at_utc": retrieved_at_utc,
            "manifest_created_at_utc": utc_now(),
        },
        "file": {
            "relative_path": relative_path.as_posix(),
            "filename": csv_path.name,
            "size_bytes": csv_path.stat().st_size,
            "sha256": file_hash,
            **csv_info,
        },
        "reference_checks": {
            "expected_row_count": EXPECTED_ROW_COUNT,
            "row_count_matches_expected": (
                row_count == EXPECTED_ROW_COUNT
            ),
            "expected_columns_match": (
                columns == EXPECTED_COLUMNS
            ),
            "all_records_have_expected_field_count": (
                csv_info["records_with_wrong_field_count"] == 0
            ),
        },
    }

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "data_retrieval_manifest.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nManifest created:", output_path.relative_to(PROJECT_ROOT))
    print("Rows:", row_count)
    print("Columns:", len(columns))
    print("SHA-256:", file_hash)
    print("Size (bytes):", csv_path.stat().st_size)
    print("Expected row count matches:",
          manifest["reference_checks"]["row_count_matches_expected"])
    print("Expected columns match:",
          manifest["reference_checks"]["expected_columns_match"])
    print("Incorrect field counts:",
          csv_info["records_with_wrong_field_count"])

    if not all(manifest["reference_checks"].values()):
        print(
            "\nWARNING: At least one reference check failed. "
            "Investigate the file before continuing."
        )


if __name__ == "__main__":
    main()
