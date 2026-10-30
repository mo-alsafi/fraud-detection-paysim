SELECT
    COUNT(*) AS total_transactions,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_transactions,
    COUNT(*) -
        SUM(CAST(isFraud AS BIGINT)) AS legitimate_transactions,
    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct,
    SUM(
        CASE
            WHEN isFraud = 1 THEN amount
            ELSE 0
        END
    ) AS total_fraudulent_transaction_value,
    AVG(
        CASE
            WHEN isFraud = 1 THEN amount
        END
    ) AS average_fraud_transaction_amount
FROM read_parquet('data/processed/paysim.parquet');