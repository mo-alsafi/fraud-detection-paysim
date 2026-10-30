WITH type_summary AS (
    SELECT
        type,
        COUNT(*) AS total_transactions,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_count,
        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS fraudulent_transaction_value
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY type
)
SELECT
    type,
    total_transactions,
    fraud_count,
    total_transactions - fraud_count
        AS legitimate_transactions,
    ROUND(
        100.0 * fraud_count / total_transactions,
        6
    ) AS fraud_rate_pct,
    ROUND(
        100.0 * fraud_count
        / NULLIF(SUM(fraud_count) OVER (), 0),
        4
    ) AS share_of_all_fraud_pct,
    fraudulent_transaction_value
FROM type_summary
ORDER BY fraud_rate_pct DESC;