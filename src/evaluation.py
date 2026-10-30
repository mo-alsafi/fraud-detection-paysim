
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen initial chronological protocol.
SPLIT_BOUNDS = {
    "train": (1, 445),
    "validation": (446, 594),
    "test": (595, 743),
}

# Initial operational assumption: one simulated day of label delay.
# This is a project assumption, not a measured PaySim property.
LABEL_DELAY_STEPS = 24

DEFAULT_TEST_LOCK = (
    PROJECT_ROOT / "reports" / "test_evaluation_lock.json"
)


def _integer_steps(values: Sequence, name: str) -> np.ndarray:
    """Validate and return finite integer simulation steps."""
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric steps.") from exc

    if numeric.ndim != 1 or numeric.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence.")

    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains missing or non-finite steps.")

    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} contains non-integer steps.")

    return numeric.astype(np.int64)


def validate_temporal_splits(
    splits: Mapping[str, pd.DataFrame],
    step_col: str = "step",
) -> None:
    """Raise if splits violate the declared chronological boundaries."""
    required = set(SPLIT_BOUNDS)

    if set(splits) != required:
        raise ValueError(
            f"Expected exactly these split names: {sorted(required)}."
        )

    observed_ranges = {}

    for split_name, (lower, upper) in SPLIT_BOUNDS.items():
        frame = splits[split_name]

        if frame.empty:
            raise ValueError(f"{split_name} split is empty.")

        if step_col not in frame.columns:
            raise ValueError(
                f"{split_name} split lacks column {step_col!r}."
            )

        steps = _integer_steps(
            frame[step_col].to_numpy(),
            f"{split_name}.{step_col}",
        )

        if np.any((steps < lower) | (steps > upper)):
            raise ValueError(
                f"{split_name} contains steps outside "
                f"the declared interval [{lower}, {upper}]."
            )

        observed_ranges[split_name] = (
            int(steps.min()),
            int(steps.max()),
        )

    if (
        observed_ranges["train"][1]
        >= observed_ranges["validation"][0]
    ):
        raise ValueError(
            "Training and validation overlap in time."
        )

    if (
        observed_ranges["validation"][1]
        >= observed_ranges["test"][0]
    ):
        raise ValueError(
            "Validation and test overlap in time."
        )


def split_by_step(
    frame: pd.DataFrame,
    step_col: str = "step",
) -> dict[str, pd.DataFrame]:
    """
    Split an existing DataFrame using the frozen time boundaries.

    For the full PaySim dataset, prefer DuckDB WHERE clauses to
    materializing the entire dataset as a pandas DataFrame.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if step_col not in frame.columns:
        raise ValueError(f"Missing step column: {step_col!r}")

    steps = _integer_steps(frame[step_col].to_numpy(), step_col)

    if np.any((steps < 1) | (steps > 743)):
        raise ValueError("Steps must be between 1 and 743.")

    train_start, train_end = SPLIT_BOUNDS["train"]
    val_start, val_end = SPLIT_BOUNDS["validation"]
    test_start, test_end = SPLIT_BOUNDS["test"]

    splits = {
        "train": frame.loc[
            (steps >= train_start) & (steps <= train_end)
        ].copy(),
        "validation": frame.loc[
            (steps >= val_start) & (steps <= val_end)
        ].copy(),
        "test": frame.loc[
            (steps >= test_start) & (steps <= test_end)
        ].copy(),
    }

    validate_temporal_splits(splits, step_col=step_col)
    return splits


def top_k_alert_mask_per_day(
    scores: Sequence[float],
    steps: Sequence[int],
    alerts_per_day: int,
) -> np.ndarray:
    """
    Select the highest-scored transactions within each 24-step
    simulation-day group.

    Ties are resolved by input order. Partial groups at split
    boundaries require explicit operational treatment.
    """
    if (
        isinstance(alerts_per_day, bool)
        or not isinstance(alerts_per_day, (int, np.integer))
        or alerts_per_day < 1
    ):
        raise ValueError("alerts_per_day must be a positive integer.")

    risk_scores = np.asarray(scores, dtype=float)
    step_values = _integer_steps(steps, "steps")

    if risk_scores.ndim != 1 or len(risk_scores) != len(step_values):
        raise ValueError("Scores and steps must have the same length.")

    if not np.isfinite(risk_scores).all():
        raise ValueError("Scores contain non-finite values.")

    if np.any((step_values < 1) | (step_values > 743)):
        raise ValueError("Steps must be between 1 and 743.")

    # Use the dataset's absolute simulation-day grouping.
    simulation_days = (step_values - 1) // 24

    alerts = np.zeros(len(risk_scores), dtype=bool)

    for day in np.unique(simulation_days):
        positions = np.flatnonzero(simulation_days == day)

        order = np.argsort(
            -risk_scores[positions],
            kind="stable",
        )

        selected = positions[order[:alerts_per_day]]
        alerts[selected] = True

    return alerts


def calculate_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    alerts: Sequence[bool],
    amounts: Sequence[float],
) -> dict:
    """Calculate ranking, alert-policy, and value-weighted metrics."""
    labels = np.asarray(y_true)
    risk_scores = np.asarray(scores, dtype=float)
    raw_alerts = np.asarray(alerts)
    values = np.asarray(amounts, dtype=float)

    lengths = {
        len(labels),
        len(risk_scores),
        len(raw_alerts),
        len(values),
    }

    if len(lengths) != 1 or len(labels) == 0:
        raise ValueError("All inputs must have the same non-zero length.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Evaluation requires both fraud and legitimate cases."
        )

    if not np.isfinite(risk_scores).all():
        raise ValueError("Scores must be finite.")

    if not np.isin(raw_alerts, [0, 1, False, True]).all():
        raise ValueError("Alerts must be Boolean or binary.")

    alerts_array = raw_alerts.astype(bool)

    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Amounts must be finite and non-negative.")

    fraud = labels == 1
    legitimate = ~fraud

    true_positives = int(np.sum(alerts_array & fraud))
    false_positives = int(np.sum(alerts_array & legitimate))
    false_negatives = int(np.sum(~alerts_array & fraud))

    alert_count = int(np.sum(alerts_array))
    fraud_count = int(np.sum(fraud))
    legitimate_count = int(np.sum(legitimate))

    precision = (
        true_positives / alert_count
        if alert_count
        else 0.0
    )

    recall = true_positives / fraud_count
    false_positive_rate = false_positives / legitimate_count

    total_fraud_value = float(values[fraud].sum())
    caught_fraud_value = float(values[fraud & alerts_array].sum())

    value_weighted_recall = (
        caught_fraud_value / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )

    return {
        "average_precision": float(
            average_precision_score(labels, risk_scores)
        ),
        "alert_count": alert_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "fraudulent_transaction_value_caught": caught_fraud_value,
        "total_fraudulent_transaction_value": total_fraud_value,
        "dollar_weighted_recall": float(value_weighted_recall),
    }


def evaluate_policies(
    y_true: Sequence[int],
    policy_scores: Mapping[str, Sequence[float]],
    policy_alerts: Mapping[str, Sequence[bool]],
    amounts: Sequence[float],
    *,
    split: str,
    final: bool = False,
    steps: Sequence[int] | None = None,
    lock_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple policies in one call.

    The test split requires final=True. A persistent lock prevents
    another formal test evaluation from this project workspace.

    Include all policies to be compared in the same final call.
    This guard cannot undo earlier exploratory inspection of test data.
    """
    if split not in SPLIT_BOUNDS:
        raise ValueError(f"Unknown split: {split!r}")

    if split == "test" and not final:
        raise RuntimeError(
            "Test evaluation requires final=True after model and "
            "policy selection are complete."
        )

    if split != "test" and final:
        raise ValueError(
            "final=True is reserved for the formal test evaluation."
        )

    if not policy_scores:
        raise ValueError("At least one policy is required.")

    if set(policy_scores) != set(policy_alerts):
        raise ValueError(
            "Policy score and alert-mask names must match exactly."
        )

    y = np.asarray(y_true)
    value_array = np.asarray(amounts, dtype=float)

    if steps is not None:
        step_array = _integer_steps(steps, "steps")

        if len(step_array) != len(y):
            raise ValueError("Steps and labels must have equal length.")

        lower, upper = SPLIT_BOUNDS[split]

        if np.any((step_array < lower) | (step_array > upper)):
            raise ValueError(
                f"Rows passed for {split} must have steps "
                f"between {lower} and {upper}."
            )
    else:
        step_array = None

    resolved_lock = (
        Path(lock_path)
        if lock_path is not None
        else DEFAULT_TEST_LOCK
    )

    if split == "test" and resolved_lock.exists():
        raise RuntimeError(
            f"The formal test evaluation is already recorded at "
            f"{resolved_lock}. Do not rerun it to compare new models."
        )

    records = []

    for policy_name in policy_scores:
        metrics = calculate_metrics(
            y_true=y,
            scores=policy_scores[policy_name],
            alerts=policy_alerts[policy_name],
            amounts=value_array,
        )

        records.append({
            "split": split,
            "policy": policy_name,
            **metrics,
        })

    results = pd.DataFrame(records)

    if split == "test":
        lock_record = {
            "evaluated_at_utc": (
                datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "split": "test",
            "final": True,
            "row_count": int(len(y)),
            "step_min": (
                int(step_array.min())
                if step_array is not None
                else None
            ),
            "step_max": (
                int(step_array.max())
                if step_array is not None
                else None
            ),
            "policies_evaluated": list(policy_scores),
            "results": results.to_dict(orient="records"),
            "note": (
                "Formal evaluation recorded. This lock does not "
                "undo prior exploratory inspection of the test data."
            ),
        }

        resolved_lock.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation also protects against a second process
        # creating the same lock concurrently.
        try:
            with resolved_lock.open(
                "x",
                encoding="utf-8",
            ) as file:
                json.dump(lock_record, file, indent=2)
                file.write("\n")
        except FileExistsError as exc:
            raise RuntimeError(
                "The formal test evaluation has already been recorded. "
                "Do not rerun it to compare new models."
            ) from exc

    return results


def past_label_counts(
    query_accounts: Sequence[str],
    query_steps: Sequence[int],
    event_accounts: Sequence[str],
    event_steps: Sequence[int],
    event_labels: Sequence[int],
    *,
    label_delay_steps: int = LABEL_DELAY_STEPS,
) -> np.ndarray:
    """
    Count fraud-labelled events for each queried account whose
    steps are no newer than query_step - label_delay_steps.

    This generic helper defines the information boundary for tests.
    The full production feature pipeline should implement the same
    rule in DuckDB rather than using this Python helper on all rows.
    """
    if (
        isinstance(label_delay_steps, bool)
        or not isinstance(label_delay_steps, (int, np.integer))
        or label_delay_steps < 1
    ):
        raise ValueError("Label delay must be a positive integer.")

    query_accts = np.asarray(query_accounts, dtype=object)
    event_accts = np.asarray(event_accounts, dtype=object)

    q_steps = _integer_steps(query_steps, "query_steps")
    e_steps = _integer_steps(event_steps, "event_steps")

    labels = np.asarray(event_labels)

    if query_accts.ndim != 1 or len(query_accts) != len(q_steps):
        raise ValueError("Query accounts and steps must have equal length.")

    if event_accts.ndim != 1 or len(event_accts) != len(e_steps):
        raise ValueError("Event accounts and steps must have equal length.")

    if len(event_accts) != len(labels):
        raise ValueError("Event accounts, steps, and labels must align.")

    if pd.isna(query_accts).any() or pd.isna(event_accts).any():
        raise ValueError("Account identifiers cannot be missing.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Event labels must be binary.")

    labels = labels.astype(np.int64)

    # Index each account's historical label stream in chronological
    # order, together with its cumulative fraud-label counts.
    history_by_account = {}

    for account in np.unique(event_accts):
        positions = np.flatnonzero(event_accts == account)
        order = np.argsort(e_steps[positions], kind="stable")

        sorted_steps = e_steps[positions][order]
        sorted_labels = labels[positions][order]

        cumulative_counts = np.concatenate((
            np.array([0], dtype=np.int64),
            np.cumsum(sorted_labels, dtype=np.int64),
        ))

        history_by_account[account] = (
            sorted_steps,
            cumulative_counts,
        )

    output = np.zeros(len(query_steps), dtype=np.int64)

    for i, (account, query_step) in enumerate(
        zip(query_accts, q_steps)
    ):
        history = history_by_account.get(account)

        if history is None:
            continue

        historical_steps, cumulative_counts = history
        cutoff = int(query_step) - label_delay_steps

        # "right" includes events exactly LABEL_DELAY_STEPS old.
        position = int(
            np.searchsorted(
                historical_steps,
                cutoff,
                side="right",
            )
        )

        output[i] = cumulative_counts[position]

    return output
