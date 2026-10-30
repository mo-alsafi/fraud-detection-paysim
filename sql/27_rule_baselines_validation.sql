
WITH validation AS (
    SELECT
        step,
        type,
        amount,
        isFraud,
        isFlaggedFraud
    FROM read_parquet('data/processed/paysim.parquet')
    WHERE step BETWEEN 446 AND 594
),

policy_alerts AS (
    SELECT
        'Existing isFlaggedFraud' AS policy,
        isFraud,
        amount,
        isFlaggedFraud = 1 AS alert
    FROM validation

    UNION ALL

    SELECT
        'All TRANSFER and CASH_OUT',
        isFraud,
        amount,
        type IN ('TRANSFER', 'CASH_OUT')
    FROM validation

    UNION ALL

    SELECT
        'TRANSFER above 200,000',
        isFraud,
        amount,
        type = 'TRANSFER' AND amount > 200000
    FROM validation

    UNION ALL

    SELECT
        'High-value TRANSFER/CASH_OUT',
        isFraud,
        amount,
        type IN ('TRANSFER', 'CASH_OUT')
            AND amount >= 100000
    FROM validation
),

metrics AS (
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
        ) AS fraud_value_caught,

        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS total_fraud_value

    FROM policy_alerts
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

    fraud_value_caught,
    total_fraud_value,

    ROUND(
        100.0 * fraud_value_caught
        / NULLIF(total_fraud_value, 0),
        4
    ) AS dollar_weighted_recall_pct

FROM metrics
ORDER BY policy;
