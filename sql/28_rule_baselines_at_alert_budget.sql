
WITH validation_base AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY
                step,
                type,
                nameOrig,
                nameDest,
                amount,
                isFraud,
                isFlaggedFraud,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest
        ) AS transaction_id,

        step,

        CAST(
            FLOOR((step - 1) / 24.0)
            AS INTEGER
        ) AS simulation_day,

        type,
        amount,
        isFraud,
        isFlaggedFraud

    FROM read_parquet('data/processed/paysim.parquet')

    WHERE step BETWEEN 446 AND 594
),

day_budgets AS (
    SELECT
        simulation_day,

        CAST(
            FLOOR(
                500.0
                * (
                    LEAST(
                        simulation_day * 24 + 24,
                        594
                    )
                    -
                    GREATEST(
                        simulation_day * 24 + 1,
                        446
                    )
                    + 1
                )
                / 24.0
            ) AS BIGINT
        ) AS daily_budget

    FROM (
        SELECT DISTINCT simulation_day
        FROM validation_base
    )
),

policy_candidates AS (
    SELECT
        'Existing isFlaggedFraud' AS policy,
        transaction_id,
        simulation_day,
        amount,
        isFraud,
        isFlaggedFraud = 1 AS eligible_for_alert
    FROM validation_base

    UNION ALL

    SELECT
        'All TRANSFER and CASH_OUT',
        transaction_id,
        simulation_day,
        amount,
        isFraud,
        type IN ('TRANSFER', 'CASH_OUT')
    FROM validation_base

    UNION ALL

    SELECT
        'TRANSFER above 200,000',
        transaction_id,
        simulation_day,
        amount,
        isFraud,
        type = 'TRANSFER' AND amount > 200000
    FROM validation_base

    UNION ALL

    SELECT
        'High-value TRANSFER/CASH_OUT',
        transaction_id,
        simulation_day,
        amount,
        isFraud,
        type IN ('TRANSFER', 'CASH_OUT')
            AND amount >= 100000
    FROM validation_base
),

ranked_candidates AS (
    SELECT
        p.*,
        b.daily_budget,

        ROW_NUMBER() OVER (
            PARTITION BY p.policy, p.simulation_day
            ORDER BY
                p.eligible_for_alert DESC,
                p.amount DESC,
                p.transaction_id
        ) AS policy_rank

    FROM policy_candidates AS p

    INNER JOIN day_budgets AS b
        ON p.simulation_day = b.simulation_day
),

selected_alerts AS (
    SELECT
        *,

        (
            eligible_for_alert
            AND policy_rank <= daily_budget
        ) AS alert

    FROM ranked_candidates
),

policy_metrics AS (
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
                WHEN alert AND isFraud = 1
                THEN amount
                ELSE 0
            END
        ) AS fraud_value_caught,

        SUM(
            CASE
                WHEN isFraud = 1 THEN amount
                ELSE 0
            END
        ) AS total_fraud_value

    FROM selected_alerts

    GROUP BY policy
),

total_capacity AS (
    SELECT SUM(daily_budget) AS available_alert_capacity
    FROM day_budgets
)

SELECT
    m.policy,

    c.available_alert_capacity,

    m.alert_count,

    m.true_positives,

    m.false_positives,

    m.false_negatives,

    ROUND(
        100.0 * m.true_positives
        / NULLIF(m.alert_count, 0),
        4
    ) AS precision_pct,

    ROUND(
        100.0 * m.true_positives
        / NULLIF(m.total_fraud_cases, 0),
        4
    ) AS recall_pct,

    ROUND(
        100.0 * m.false_positives
        / NULLIF(m.total_legitimate_transactions, 0),
        4
    ) AS false_positive_rate_pct,

    m.fraud_value_caught,

    m.total_fraud_value,

    ROUND(
        100.0 * m.fraud_value_caught
        / NULLIF(m.total_fraud_value, 0),
        4
    ) AS dollar_weighted_recall_pct

FROM policy_metrics AS m

CROSS JOIN total_capacity AS c

ORDER BY m.policy;
