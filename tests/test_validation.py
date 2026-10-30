from copy import deepcopy

from src.validate_data import (
    EXPECTED_ROW_COUNT,
    EXPECTED_SCHEMA,
    EXPECTED_STEP_MIN,
    EXPECTED_STEP_MAX,
    EXPECTED_TYPES,
    get_validation_errors,
)


def valid_summary() -> dict:
    """Create a small, representative valid statistics object."""
    return {
        "row_count": EXPECTED_ROW_COUNT,
        "rows_with_nulls": 0,
        "rows_with_invalid_labels": 0,
        "min_step": EXPECTED_STEP_MIN,
        "max_step": EXPECTED_STEP_MAX,
        "distinct_steps": (
            EXPECTED_STEP_MAX - EXPECTED_STEP_MIN + 1
        ),
        "distinct_type_count": len(EXPECTED_TYPES),
    }


def validate_fixture(
    summary=None,
    observed_types=None,
    schema=None,
    manifest_row_count=None,
):
    """Run the validation contract against test values."""
    return get_validation_errors(
        summary=summary if summary is not None else valid_summary(),
        observed_types=(
            observed_types
            if observed_types is not None
            else set(EXPECTED_TYPES)
        ),
        actual_schema=(
            schema if schema is not None else list(EXPECTED_SCHEMA)
        ),
        manifest_row_count=(
            manifest_row_count
            if manifest_row_count is not None
            else EXPECTED_ROW_COUNT
        ),
    )


def test_valid_contract_passes():
    assert validate_fixture() == []


def test_wrong_row_count_fails():
    summary = valid_summary()
    summary["row_count"] -= 1

    errors = validate_fixture(summary=summary)

    assert any("Expected" in error and "rows" in error for error in errors)


def test_null_rows_fail():
    summary = valid_summary()
    summary["rows_with_nulls"] = 3

    errors = validate_fixture(summary=summary)

    assert any("NULLs" in error for error in errors)


def test_invalid_labels_fail():
    summary = valid_summary()
    summary["rows_with_invalid_labels"] = 1

    errors = validate_fixture(summary=summary)

    assert any("non-binary" in error for error in errors)


def test_wrong_step_range_fails():
    summary = valid_summary()
    summary["max_step"] = EXPECTED_STEP_MAX - 1

    errors = validate_fixture(summary=summary)

    assert any("maximum step" in error for error in errors)


def test_missing_transaction_type_fails():
    observed = set(EXPECTED_TYPES)
    observed.remove("TRANSFER")

    errors = validate_fixture(observed_types=observed)

    assert any("transaction-type set" in error for error in errors)


def test_wrong_schema_fails():
    wrong_schema = deepcopy(EXPECTED_SCHEMA)
    wrong_schema[0] = ("step", "VARCHAR")

    errors = validate_fixture(schema=wrong_schema)

    assert any("Parquet schema" in error for error in errors)


def test_manifest_row_count_mismatch_fails():
    errors = validate_fixture(manifest_row_count=100)

    assert any("manifest" in error.lower() for error in errors)