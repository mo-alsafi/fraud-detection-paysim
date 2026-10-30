WITH transactions AS (
    SELECT
        type,
        isFraud,
        nameDest,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    FROM read_parquet('data/processed/paysim.parquet')
)

SELECT
    type,
    isFraud,

    COUNT(*) AS transaction_count,

    -- Sender balance unchanged
    COUNT(*) FILTER (
        WHERE ABS(oldbalanceOrg - newbalanceOrig) <= 0.01
    ) AS sender_balance_unchanged_count,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE ABS(oldbalanceOrg - newbalanceOrig) <= 0.01
        ) / COUNT(*),
        4
    ) AS sender_balance_unchanged_pct,

    -- Does the simple debit equation hold?
    COUNT(*) FILTER (
        WHERE ABS(
            (oldbalanceOrg - amount) - newbalanceOrig
        ) <= 0.01
    ) AS sender_debit_equation_matches,

    -- Sender balance ends at zero
    COUNT(*) FILTER (
        WHERE newbalanceOrig = 0
    ) AS sender_new_balance_zero_count,

    -- Merchant destination balances are not comparable:
    -- the dataset does not provide meaningful balances for M* accounts.
    COUNT(*) FILTER (
        WHERE nameDest NOT LIKE 'M%'
    ) AS customer_destination_transactions,

    COUNT(*) FILTER (
        WHERE nameDest NOT LIKE 'M%'
          AND ABS(oldbalanceDest - newbalanceDest) <= 0.01
    ) AS customer_destination_balance_unchanged,

    COUNT(*) FILTER (
        WHERE nameDest NOT LIKE 'M%'
          AND ABS(
              (oldbalanceDest + amount) - newbalanceDest
          ) <= 0.01
    ) AS destination_credit_equation_matches,

    COUNT(*) FILTER (
        WHERE nameDest NOT LIKE 'M%'
          AND newbalanceDest = 0
    ) AS customer_destination_new_balance_zero

FROM transactions

GROUP BY type, isFraud

ORDER BY type, isFraud DESC;