SELECT
    type,
    COUNT(*) AS transaction_count
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY type
ORDER BY transaction_count DESC;