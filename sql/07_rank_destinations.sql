WITH destination_activity AS (
    SELECT
        type,
        nameDest,
        COUNT(*) AS transaction_count,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_count
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY type, nameDest
),
ranked_destinations AS (
    SELECT
        type,
        nameDest,
        transaction_count,
        fraud_count,
        ROW_NUMBER() OVER (
            PARTITION BY type
            ORDER BY transaction_count DESC, nameDest
        ) AS destination_rank
    FROM destination_activity
)
SELECT
    type,
    nameDest,
    transaction_count,
    fraud_count,
    destination_rank
FROM ranked_destinations
WHERE destination_rank <= 10
ORDER BY type, destination_rank;