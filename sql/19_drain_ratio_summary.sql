
WITH transactions AS (
    SELECT
        isFraud,
        amount,
        oldbalanceOrg,

        CASE
            WHEN oldbalanceOrg > 0
            THEN amount / oldbalanceOrg
            ELSE NULL
        END AS drain_ratio

    FROM read_parquet('data/processed/paysim.parquet')
)

SELECT
    CASE
        WHEN isFraud = 1 THEN 'Fraud'
        ELSE 'Legitimate'
    END AS transaction_class,

    COUNT(*) AS total_transactions,

    COUNT(*) FILTER (
        WHERE oldbalanceOrg > 0
    ) AS transactions_with_defined_ratio,

    COUNT(*) FILTER (
        WHERE oldbalanceOrg IS NULL
           OR oldbalanceOrg <= 0
    ) AS transactions_with_undefined_ratio,

    COUNT(*) FILTER (
        WHERE drain_ratio BETWEEN 0.9 AND 1.1
    ) AS transactions_near_one,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE drain_ratio BETWEEN 0.9 AND 1.1
        ) / COUNT(*),
        4
    ) AS pct_near_one_of_all_transactions,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE drain_ratio BETWEEN 0.9 AND 1.1
        )
        / NULLIF(
            COUNT(*) FILTER (
                WHERE oldbalanceOrg > 0
            ),
            0
        ),
        4
    ) AS pct_near_one_of_defined_ratios,

    MEDIAN(drain_ratio) AS median_drain_ratio,

    AVG(drain_ratio) AS mean_drain_ratio,

    MAX(drain_ratio) AS maximum_drain_ratio

FROM transactions

GROUP BY isFraud

ORDER BY isFraud DESC;
