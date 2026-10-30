import os
import re
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = PROJECT_ROOT / "sql" / "01_parquet_profile.sql"
PROCESSED_DIR = Path("reports/parquet_profile")


def main() -> None:
    if not SQL_FILE.is_file():
        raise FileNotFoundError(
            f"SQL file not found: {SQL_FILE}"
        )

    # Ensure the output directory exists before running queries
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sql_text = SQL_FILE.read_text(encoding="utf-8")

    # Split by semicolon to separate the individual queries
    raw_statements = sql_text.split(";")
    statements = []

    for statement in raw_statements:
        stripped = statement.strip()
        if stripped:
            # 1. Parse a query name from SQL comments (e.g., "-- name: fetch_users")
            # This looks for any line starting with '-- name:' or '--name:'
            name_match = re.search(r"^--\s*name:\s*([^\n\r]+)", stripped, re.IGNORECASE | re.MULTILINE)
            
            if name_match:
                # Clean up the name string for terminal and filesystem safety
                query_name = name_match.group(1).strip().replace(" ", "_").lower()
            else:
                query_name = None
                
            statements.append({"text": stripped, "name": query_name})

    connection = duckdb.connect(database=":memory:")

    try:
        for index, item in enumerate(statements, start=1):
            statement_text = item["text"]
            
            # Determine the display/file name (Fallback to 'query_1' if name missing)
            display_name = item["name"] if item["name"] else f"query_{index}"
            
            print(f"\n{'=' * 60}")
            print(f"Query {index}: {display_name.upper()}")
            print("=" * 60)

            # Execute query and pull into a Pandas DataFrame
            result = connection.execute(statement_text).fetchdf()
            
            # Print to terminal
            print(result.to_string(index=False))

            # 2. Save the result to a CSV file in data/processed/
            csv_filename = f"{index:02d}_{display_name}.csv"
            csv_path = PROCESSED_DIR / csv_filename
            result.to_csv(csv_path, index=False)
            
            print(f"\n📁 Saved to: {csv_path}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()