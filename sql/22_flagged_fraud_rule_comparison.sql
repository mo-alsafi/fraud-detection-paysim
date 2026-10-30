
WITH transactions AS (
    SELECT
        type,
        amount,
        isFraud,
        isFlaggedFraud,

        (isFlaggedFraud = 1) AS builtin_alert,

        (
            type = 'TRANSFER'
            AND amount > 200000
        ) AS naive_alert

    FROM read_parquet('data/processed/paysim.parquet')
),

policy_results AS (
    SELECT
        'Existing isFlaggedFraud indicator' AS policy,
        isFraud,
        amount,
        builtin_alert AS alert
    FROM transactions

    UNION ALL

    SELECT
        'Every TRANSFER above 200,000' AS policy,
        isFraud,
        amount,
        naive_alert AS alert
    FROM transactions
),

policy_summary AS (
    SELECT
        policy,

        COUNT(*) FILTER (
            WHERE alert
        ) AS alert_count,

        COUNT(*) FILTER (
            WHERE alert AND isFraud = 1
        ) AS true_positives,

        COUNT(*) FILTER (
            WHERE alert AND isFraud = 0
        ) AS false_positives,

        COUNT(*) FILTER (
            WHERE NOT alert AND isFraud = 1
        ) AS false_negatives,

        COUNT(*) FILTER (
            WHERE isFraud = 1
        ) AS total_fraud_cases,

        COUNT(*) FILTER (
            WHERE isFraud = 0
        ) AS total_legitimate_transactions,

        SUM(
            CASE
                WHEN isFraud = 1 AND alert THEN amount
                ELSE 0
            END
        ) AS fraudulent_transaction_value_caught,

        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS total_fraudulent_transaction_value

    FROM policy_results
    GROUP BY policy
)

SELECT
    policy,

    alert_count,

    true_positives,

    false_positives,

    false_negatives,

    ROUND(
        100.0 * true_positives / NULLIF(alert_count, 0),
        4
    ) AS precision_pct,

    ROUND(
        100.0 * true_positives / NULLIF(total_fraud_cases, 0),
        4
    ) AS recall_pct,

    ROUND(
        100.0 * false_positives
        / NULLIF(total_legitimate_transactions, 0),
        4
    ) AS false_positive_rate_pct,

    fraudulent_transaction_value_caught,

    total_fraudulent_transaction_value,

    ROUND(
        100.0 * fraudulent_transaction_value_caught
        / NULLIF(total_fraudulent_transaction_value, 0),
        4
    ) AS dollar_weighted_recall_pct

FROM policy_summary

ORDER BY policy;
