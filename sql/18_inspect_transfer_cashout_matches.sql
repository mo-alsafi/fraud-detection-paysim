WITH fraudulent_transfers AS (
    SELECT
        step AS transfer_step,
        nameOrig AS transfer_origin,
        nameDest AS transfer_destination,
        amount AS transfer_amount
    FROM read_parquet('data/processed/paysim.parquet')
    WHERE type = 'TRANSFER'
      AND isFraud = 1
),

cashouts AS (
    SELECT
        step AS cashout_step,
        nameOrig AS cashout_origin,
        nameDest AS cashout_destination,
        amount AS cashout_amount,
        isFraud AS cashout_is_fraud
    FROM read_parquet('data/processed/paysim.parquet')
    WHERE type = 'CASH_OUT'
)

SELECT
    t.transfer_step,
    t.transfer_origin,
    t.transfer_destination,
    t.transfer_amount,

    c.cashout_step,
    c.cashout_origin,
    c.cashout_destination,
    c.cashout_amount,
    c.cashout_is_fraud,

    c.cashout_step - t.transfer_step AS gap_steps

FROM fraudulent_transfers AS t
INNER JOIN cashouts AS c
    ON t.transfer_destination = c.cashout_origin

ORDER BY
    t.transfer_step,
    c.cashout_step;