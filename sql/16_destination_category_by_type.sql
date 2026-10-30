SELECT
    CASE
        WHEN nameDest LIKE 'M%' THEN 'Merchant'
        ELSE 'Customer'
    END AS destination_category,

    type,

    COUNT(*) AS total_transactions,

    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct

FROM read_parquet('data/processed/paysim.parquet')

GROUP BY
    destination_category,
    type

ORDER BY
    destination_category,
    type;