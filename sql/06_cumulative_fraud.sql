WITH fraud_by_step AS (
    SELECT
        step,
        COUNT(*) AS transaction_count,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_count
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY step
)
SELECT
    step,
    transaction_count,
    fraud_count,
    SUM(fraud_count) OVER (
        ORDER BY step
        ROWS BETWEEN UNBOUNDED PRECEDING
                 AND CURRENT ROW
    ) AS cumulative_fraud_count
FROM fraud_by_step
ORDER BY step;