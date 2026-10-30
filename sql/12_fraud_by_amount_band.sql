WITH banded_transactions AS (
    SELECT
        CASE
            WHEN amount < 100 THEN '<100'
            WHEN amount < 1000 THEN '100-<1,000'
            WHEN amount < 10000 THEN '1,000-<10,000'
            WHEN amount < 100000 THEN '10,000-<100,000'
            ELSE '>=100,000'
        END AS amount_band,
        CASE
            WHEN amount < 100 THEN 1
            WHEN amount < 1000 THEN 2
            WHEN amount < 10000 THEN 3
            WHEN amount < 100000 THEN 4
            ELSE 5
        END AS band_order,
        amount,
        isFraud
    FROM read_parquet('data/processed/paysim.parquet')
)
SELECT
    amount_band,
    COUNT(*) AS total_transactions,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,
    COUNT(*) - SUM(CAST(isFraud AS BIGINT))
        AS legitimate_transactions,
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
FROM banded_transactions
GROUP BY amount_band, band_order
ORDER BY band_order;