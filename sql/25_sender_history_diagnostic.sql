
WITH sender_history AS (
    SELECT
        step,
        nameOrig,
        amount,

        COUNT(*) OVER (
            PARTITION BY nameOrig
            ORDER BY step
            RANGE BETWEEN UNBOUNDED PRECEDING
                      AND 1 PRECEDING
        ) AS prior_outgoing_count,

        AVG(amount) OVER (
            PARTITION BY nameOrig
            ORDER BY step
            RANGE BETWEEN UNBOUNDED PRECEDING
                      AND 1 PRECEDING
        ) AS past_mean_outgoing_amount

    FROM read_parquet('data/processed/paysim.parquet')
)

SELECT
    COUNT(*) AS total_transactions,

    COUNT(*) FILTER (
        WHERE prior_outgoing_count = 0
    ) AS no_prior_outgoing_transactions,

    COUNT(*) FILTER (
        WHERE prior_outgoing_count > 0
    ) AS has_prior_outgoing_transactions,

    COUNT(*) FILTER (
        WHERE past_mean_outgoing_amount > 0
    ) AS positive_historical_mean,

    COUNT(*) FILTER (
        WHERE prior_outgoing_count > 0
          AND past_mean_outgoing_amount = 0
    ) AS zero_historical_mean,

    COUNT(*) FILTER (
        WHERE prior_outgoing_count > 0
          AND past_mean_outgoing_amount IS NULL
    ) AS unexpected_missing_historical_mean

FROM sender_history;
