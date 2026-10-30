WITH hourly_summary AS (
    SELECT
        (step - 1) % 24 AS simulated_hour,

        COUNT(*) AS total_transactions,

        SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS fraudulent_transaction_value

    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY simulated_hour
)
SELECT
    simulated_hour,
    total_transactions,
    fraud_count,

    ROUND(
        100.0 * fraud_count / total_transactions,
        6
    ) AS fraud_rate_pct,

    fraudulent_transaction_value

FROM hourly_summary
ORDER BY simulated_hour;