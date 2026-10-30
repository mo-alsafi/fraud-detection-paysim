
from pathlib import Path
import argparse

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a SQL file using DuckDB."
    )
    parser.add_argument(
        "sql_file",
        help="SQL filename located inside the project's sql directory.",
    )
    args = parser.parse_args()

    # Accept a filename only, not an arbitrary path.
    if Path(args.sql_file).name != args.sql_file:
        raise ValueError("Pass a SQL filename from the sql directory.")

    sql_path = SQL_DIR / args.sql_file

    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8").strip()

    if not sql_text:
        raise ValueError(f"SQL file is empty: {sql_path.name}")

    # These exercise files contain one SELECT query each.
    connection = duckdb.connect(database=":memory:")

    try:
        result = connection.execute(sql_text)
        columns = [column[0] for column in result.description]
        rows = result.fetchall()

        print(f"\nQuery: {sql_path.name}")
        print(f"Rows returned: {len(rows)}\n")
        print(" | ".join(columns))
        print("-" * 80)

        for row in rows:
            print(" | ".join(str(value) for value in row))

    finally:
        connection.close()


if __name__ == "__main__":
    main()
