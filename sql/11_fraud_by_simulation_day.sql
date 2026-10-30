WITH daily_summary AS (
    SELECT
        CAST(
            FLOOR((step - 1) / 24.0) + 1
            AS INTEGER
        ) AS simulation_day,
        COUNT(*) AS total_transactions,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_count,
        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS fraudulent_transaction_value
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY simulation_day
)
SELECT
    simulation_day,
    total_transactions,
    fraud_count,
    ROUND(
        100.0 * fraud_count / total_transactions,
        6
    ) AS fraud_rate_pct,
    fraudulent_transaction_value
FROM daily_summary
ORDER BY simulation_day;