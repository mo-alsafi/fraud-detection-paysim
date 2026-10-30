WITH fraud_transactions AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY
                step,
                type,
                nameOrig,
                nameDest,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest
        ) AS fraud_tx_id,

        step,
        type,
        nameOrig,
        nameDest,
        amount

    FROM read_parquet('data/processed/paysim.parquet')

    WHERE isFraud = 1
      AND type IN ('TRANSFER', 'CASH_OUT')
),

fraudulent_transfers AS (
    SELECT *
    FROM fraud_transactions
    WHERE type = 'TRANSFER'
),

fraudulent_cashouts AS (
    SELECT *
    FROM fraud_transactions
    WHERE type = 'CASH_OUT'
),

sequence_matches AS (
    SELECT
        t.fraud_tx_id AS transfer_id,
        c.fraud_tx_id AS cashout_id,

        t.nameDest AS intermediate_account,

        t.step AS transfer_step,
        c.step AS cashout_step,

        c.step - t.step AS gap_steps,

        t.amount AS transfer_amount,
        c.amount AS cashout_amount

    FROM fraudulent_transfers AS t

    INNER JOIN fraudulent_cashouts AS c
        ON t.nameDest = c.nameOrig
       AND c.step > t.step
)

SELECT
    (
        SELECT COUNT(*)
        FROM fraudulent_transfers
    ) AS total_fraudulent_transfers,

    COUNT(*) AS matching_transfer_cashout_pairs,

    COUNT(DISTINCT transfer_id)
        AS matched_fraudulent_transfer_rows,

    ROUND(
        100.0 * COUNT(DISTINCT transfer_id)
        / NULLIF(
            (SELECT COUNT(*) FROM fraudulent_transfers),
            0
        ),
        4
    ) AS pct_transfers_with_later_fraudulent_cashout,

    COUNT(DISTINCT cashout_id)
        AS matched_fraudulent_cashout_rows,

    COUNT(DISTINCT intermediate_account)
        AS distinct_intermediate_accounts,

    MIN(gap_steps) AS minimum_gap_steps,

    MEDIAN(gap_steps) AS median_gap_steps,

    AVG(gap_steps) AS average_gap_steps,

    MAX(gap_steps) AS maximum_gap_steps

FROM sequence_matches;