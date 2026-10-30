
WITH transactions AS (
    SELECT
        isFraud,
        CASE
            WHEN oldbalanceOrg IS NULL
              OR oldbalanceOrg <= 0
                THEN 'Undefined: old balance <= 0'

            WHEN amount / oldbalanceOrg < 0.1
                THEN '<0.1'

            WHEN amount / oldbalanceOrg < 0.5
                THEN '0.1-<0.5'

            WHEN amount / oldbalanceOrg < 0.9
                THEN '0.5-<0.9'

            WHEN amount / oldbalanceOrg <= 1.1
                THEN '0.9-1.1 (near 1)'

            ELSE '>1.1'
        END AS drain_ratio_band,

        CASE
            WHEN oldbalanceOrg IS NULL
              OR oldbalanceOrg <= 0 THEN 1
            WHEN amount / oldbalanceOrg < 0.1 THEN 2
            WHEN amount / oldbalanceOrg < 0.5 THEN 3
            WHEN amount / oldbalanceOrg < 0.9 THEN 4
            WHEN amount / oldbalanceOrg <= 1.1 THEN 5
            ELSE 6
        END AS band_order

    FROM read_parquet('data/processed/paysim.parquet')
),

band_counts AS (
    SELECT
        isFraud,
        drain_ratio_band,
        band_order,
        COUNT(*) AS transaction_count

    FROM transactions

    GROUP BY
        isFraud,
        drain_ratio_band,
        band_order
)

SELECT
    CASE
        WHEN isFraud = 1 THEN 'Fraud'
        ELSE 'Legitimate'
    END AS transaction_class,

    drain_ratio_band,

    transaction_count,

    SUM(transaction_count) OVER (
        PARTITION BY isFraud
    ) AS total_transactions_in_class,

    ROUND(
        100.0 * transaction_count
        / SUM(transaction_count) OVER (
            PARTITION BY isFraud
        ),
        4
    ) AS pct_of_transaction_class

FROM band_counts

ORDER BY
    isFraud DESC,
    band_order;
