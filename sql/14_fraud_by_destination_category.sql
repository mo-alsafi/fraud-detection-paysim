
WITH classified_transactions AS (
    SELECT
        CASE
            WHEN nameDest LIKE 'M%' THEN 'Merchant'
            ELSE 'Customer'
        END AS destination_category,

        nameDest,
        amount,
        isFraud

    FROM read_parquet('data/processed/paysim.parquet')
)
SELECT
    destination_category,

    COUNT(*) AS total_transactions,

    COUNT(DISTINCT nameDest) AS distinct_destinations,

    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

    COUNT(*) - SUM(CAST(isFraud AS BIGINT))
        AS legitimate_transactions,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT))
        / NULLIF(
            SUM(SUM(CAST(isFraud AS BIGINT))) OVER (),
            0
        ),
        4
    ) AS share_of_all_fraud_pct,

    SUM(
        CASE
            WHEN isFraud = 1 THEN amount
            ELSE 0
        END
    ) AS fraudulent_transaction_value,

    COUNT(*) FILTER (
        WHERE isFraud = 1
          AND nameDest LIKE 'M%'
    ) AS fraud_cases_sent_to_merchants

FROM classified_transactions
GROUP BY destination_category
ORDER BY destination_category;
