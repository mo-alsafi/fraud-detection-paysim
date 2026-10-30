-- name: Inspect_a_sample
SELECT *
FROM read_parquet('data/processed/paysim.parquet')
LIMIT 10;


-- name: Count_transactions_by_type
SELECT
    type,
    COUNT(*) AS transaction_count
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY type
ORDER BY transaction_count DESC;


-- name: fraud_rates_by_type
SELECT
    type,
    COUNT(*) AS total_transactions,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_transactions,
    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY type
ORDER BY fraud_rate_pct DESC;


-- name: Count_fraud_by_simulation_time_step
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_transactions,
    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY step
ORDER BY step;


--name: daily transaction volumes
WITH daily_volume AS (
    SELECT
        CAST(
            FLOOR((step - 1) / 24.0) + 1
            AS INTEGER
        ) AS simulation_day,
        COUNT(*) AS total_transactions,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_transactions
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY simulation_day
)
SELECT
    simulation_day,
    total_transactions,
    fraud_transactions
FROM daily_volume
ORDER BY simulation_day;


--name: frequently_reused_destination_accounts
SELECT
    nameDest,
    COUNT(*) AS transaction_count,
    COUNT(DISTINCT type) AS distinct_transaction_types,
    SUM(CAST(isFraud AS BIGINT)) AS fraud_transactions
FROM read_parquet('data/processed/paysim.parquet')
GROUP BY nameDest
HAVING COUNT(*) > 1
ORDER BY transaction_count DESC
LIMIT 20;


-- name: cumulative_fraud_counts
WITH fraud_by_step AS (
    SELECT
        step,
        COUNT(*) AS transaction_count,
        SUM(CAST(isFraud AS BIGINT)) AS fraud_count
    FROM read_parquet('data/processed/paysim.parquet')
    GROUP BY step
)
SELECT
    step,
    transaction_count,
    fraud_count,
    SUM(fraud_count) OVER (
        ORDER BY step
        ROWS BETWEEN UNBOUNDED PRECEDING
                 AND CURRENT ROW
    ) AS cumulative_fraud_count
FROM fraud_by_step
ORDER BY step;


--name: Rank_destination_accounts_within_each_transaction_type
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