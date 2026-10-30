
WITH train_transactions AS (
    SELECT
        step,
        nameOrig,
        amount,
        isFraud

    FROM read_parquet('data/processed/paysim.parquet')

    WHERE step BETWEEN 1 AND 445
),

history AS (
    SELECT
        step,
        nameOrig,
        amount,
        isFraud,

        AVG(amount) OVER (
            PARTITION BY nameOrig
            ORDER BY step
            RANGE BETWEEN UNBOUNDED PRECEDING
                      AND 1 PRECEDING
        ) AS past_mean_outgoing_amount

    FROM train_transactions
),

feature_status AS (
    SELECT
        isFraud,

        past_mean_outgoing_amount IS NOT NULL
            AS has_prior_outgoing_history,

        amount / NULLIF(
            past_mean_outgoing_amount,
            0.0
        ) AS amount_to_sender_past_mean_ratio

    FROM history
)

SELECT
    CASE
        WHEN has_prior_outgoing_history
            THEN 'Prior outgoing history available'
        ELSE 'No prior outgoing history'
    END AS history_status,

    COUNT(*) AS total_transactions,

    SUM(CAST(isFraud AS BIGINT)) AS fraud_count,

    COUNT(*) - SUM(CAST(isFraud AS BIGINT))
        AS legitimate_count,

    ROUND(
        100.0 * SUM(CAST(isFraud AS BIGINT)) / COUNT(*),
        6
    ) AS fraud_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        4
    ) AS share_of_training_transactions_pct,

    MEDIAN(amount_to_sender_past_mean_ratio)
        AS median_amount_to_past_mean_ratio

FROM feature_status

GROUP BY has_prior_outgoing_history

ORDER BY has_prior_outgoing_history DESC;
