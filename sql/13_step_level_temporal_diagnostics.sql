SELECT
    step,
    (step - 1) % 24 AS simulated_hour,

    COUNT(*) AS total_transactions,

    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

    COUNT(*) FILTER (
        WHERE isFraud = 0
    ) AS legitimate_count,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        4
    ) AS fraud_rate_pct,

    COUNT(DISTINCT type) AS transaction_types_present

FROM read_parquet('data/processed/paysim.parquet')

WHERE
    step BETWEEN 49 AND 72
    OR step BETWEEN 721 AND 743
    OR (step - 1) % 24 IN (2, 3, 4)

GROUP BY step
ORDER BY step;