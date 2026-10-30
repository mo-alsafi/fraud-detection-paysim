SELECT
    ((step - 1) % 24) AS simulated_hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,
    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY simulated_hour_of_day
ORDER BY simulated_hour_of_day;