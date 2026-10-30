SELECT
    nameDest,
    COUNT(*) AS transaction_count,
    COUNT(DISTINCT type) AS transaction_type_count,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_count
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY nameDest
HAVING COUNT(*) > 1
ORDER BY transaction_count DESC
LIMIT 20;