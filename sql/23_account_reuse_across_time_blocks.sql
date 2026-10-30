
WITH account_events AS (
    -- Accounts acting as transaction origins
    SELECT
        nameOrig AS account_id,
        step
    FROM read_parquet('data/processed/paysim.parquet')

    UNION ALL

    -- Accounts acting as transaction destinations
    SELECT
        nameDest AS account_id,
        step
    FROM read_parquet('data/processed/paysim.parquet')
),

account_blocks AS (
    SELECT
        account_id,

        MAX(
            CASE WHEN step BETWEEN 1 AND 445
                 THEN 1 ELSE 0 END
        ) AS in_training_period,

        MAX(
            CASE WHEN step BETWEEN 446 AND 594
                 THEN 1 ELSE 0 END
        ) AS in_validation_period,

        MAX(
            CASE WHEN step BETWEEN 595 AND 743
                 THEN 1 ELSE 0 END
        ) AS in_test_period

    FROM account_events
    GROUP BY account_id
)

SELECT
    COUNT(*) AS distinct_accounts,

    COUNT(*) FILTER (
        WHERE in_training_period = 1
    ) AS accounts_in_training,

    COUNT(*) FILTER (
        WHERE in_validation_period = 1
    ) AS accounts_in_validation,

    COUNT(*) FILTER (
        WHERE in_test_period = 1
    ) AS accounts_in_test,

    COUNT(*) FILTER (
        WHERE in_training_period
            + in_validation_period
            + in_test_period >= 2
    ) AS accounts_in_multiple_periods,

    COUNT(*) FILTER (
        WHERE in_training_period = 1
          AND in_validation_period = 1
    ) AS training_validation_overlap,

    COUNT(*) FILTER (
        WHERE in_training_period = 1
          AND in_test_period = 1
    ) AS training_test_overlap,

    COUNT(*) FILTER (
        WHERE in_validation_period = 1
          AND in_test_period = 1
    ) AS validation_test_overlap,

    COUNT(*) FILTER (
        WHERE in_training_period = 1
          AND in_validation_period = 1
          AND in_test_period = 1
    ) AS accounts_in_all_three_periods

FROM account_blocks;
