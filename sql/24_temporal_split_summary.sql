
WITH assigned_blocks AS (
    SELECT
        step,
        isFraud,
        amount,

        CASE
            WHEN step BETWEEN 1 AND 445
                THEN 'train'
            WHEN step BETWEEN 446 AND 594
                THEN 'validation'
            WHEN step BETWEEN 595 AND 743
                THEN 'test'
            ELSE 'outside_protocol'
        END AS data_block

    FROM read_parquet('data/processed/paysim.parquet')
)

SELECT
    data_block,

    MIN(step) AS minimum_step,
    MAX(step) AS maximum_step,

    COUNT(*) AS transaction_count,

    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

    COUNT(*) - SUM(CAST(isFraud AS BIGINT))
        AS legitimate_count,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct,

    SUM(
        CASE
            WHEN isFraud = 1 THEN amount
            ELSE 0
        END
    ) AS fraudulent_transaction_value

FROM assigned_blocks

GROUP BY data_block

ORDER BY
    CASE data_block
        WHEN 'train' THEN 1
        WHEN 'validation' THEN 2
        WHEN 'test' THEN 3
        ELSE 4
    END;
