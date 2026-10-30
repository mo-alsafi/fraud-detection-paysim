WITH fraudulent_transfers AS (
    SELECT
        step AS transfer_step,
        nameDest AS transfer_destination,
        amount AS transfer_amount

    FROM read_parquet('data/processed/paysim.parquet')

    WHERE isFraud = 1
      AND type = 'TRANSFER'
),

all_cashouts AS (
    SELECT
        step AS cashout_step,
        nameOrig AS cashout_origin,
        isFraud AS cashout_is_fraud

    FROM read_parquet('data/processed/paysim.parquet')

    WHERE type = 'CASH_OUT'
),

account_matches AS (
    SELECT
        t.transfer_step,
        t.transfer_destination,
        c.cashout_step,
        c.cashout_origin,
        c.cashout_is_fraud

    FROM fraudulent_transfers AS t

    LEFT JOIN all_cashouts AS c
        ON t.transfer_destination = c.cashout_origin
)

SELECT
    (
        SELECT COUNT(*)
        FROM fraudulent_transfers
    ) AS fraudulent_transfer_count,

    COUNT(*) FILTER (
        WHERE cashout_origin IS NOT NULL
    ) AS matching_pairs_any_time,

    COUNT(DISTINCT transfer_destination) FILTER (
        WHERE cashout_origin IS NOT NULL
    ) AS distinct_destinations_with_cashout_origin_match,

    COUNT(*) FILTER (
        WHERE cashout_origin IS NOT NULL
          AND cashout_step = transfer_step
    ) AS matching_pairs_same_step,

    COUNT(*) FILTER (
        WHERE cashout_origin IS NOT NULL
          AND cashout_step > transfer_step
    ) AS matching_pairs_later_cashout,

    COUNT(*) FILTER (
        WHERE cashout_origin IS NOT NULL
          AND cashout_step > transfer_step
          AND cashout_is_fraud = 1
    ) AS later_fraudulent_cashout_pairs,

    COUNT(DISTINCT transfer_destination) FILTER (
        WHERE cashout_origin IS NOT NULL
          AND cashout_step > transfer_step
          AND cashout_is_fraud = 1
    ) AS destinations_with_later_fraudulent_cashout

FROM account_matches;