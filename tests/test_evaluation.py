
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    LABEL_DELAY_STEPS,
    calculate_metrics,
    evaluate_policies,
    past_label_counts,
    split_by_step,
    top_k_alert_mask_per_day,
    validate_temporal_splits,
)


def test_splits_use_disjoint_chronological_boundaries():
    frame = pd.DataFrame({
        "step": [1, 445, 446, 594, 595, 743],
        "value": [10, 20, 30, 40, 50, 60],
    })

    splits = split_by_step(frame)

    assert splits["train"]["step"].max() == 445
    assert splits["validation"]["step"].min() == 446
    assert splits["validation"]["step"].max() == 594
    assert splits["test"]["step"].min() == 595
    assert splits["test"]["step"].max() == 743

    validate_temporal_splits(splits)


def test_overlapping_time_ranges_are_rejected():
    splits = {
        "train": pd.DataFrame({"step": [1, 446]}),
        "validation": pd.DataFrame({"step": [446, 594]}),
        "test": pd.DataFrame({"step": [595, 743]}),
    }

    with pytest.raises(ValueError, match="outside"):
        validate_temporal_splits(splits)


def test_missing_steps_are_rejected():
    frame = pd.DataFrame({"step": [1, None, 595]})

    with pytest.raises(ValueError):
        split_by_step(frame)


def test_test_evaluation_requires_final_true(tmp_path):
    labels = [0, 1, 0, 1]
    scores = {"candidate": [0.1, 0.9, 0.2, 0.8]}
    alerts = {"candidate": [False, True, False, True]}
    amounts = [100.0, 500.0, 200.0, 700.0]

    lock_path = tmp_path / "test_evaluation_lock.json"

    with pytest.raises(RuntimeError, match="final=True"):
        evaluate_policies(
            labels,
            scores,
            alerts,
            amounts,
            split="test",
            final=False,
            lock_path=lock_path,
        )

    assert not lock_path.exists()


def test_formal_test_evaluation_can_only_be_recorded_once(tmp_path):
    labels = [0, 1, 0, 1]
    scores = {
        "model": [0.1, 0.9, 0.2, 0.8],
        "rule": [0.0, 1.0, 0.0, 0.0],
    }
    alerts = {
        "model": [False, True, False, True],
        "rule": [False, True, False, False],
    }
    amounts = [100.0, 500.0, 200.0, 700.0]
    steps = [595, 596, 597, 598]

    lock_path = tmp_path / "test_evaluation_lock.json"

    results = evaluate_policies(
        labels,
        scores,
        alerts,
        amounts,
        split="test",
        final=True,
        steps=steps,
        lock_path=lock_path,
    )

    assert set(results["policy"]) == {"model", "rule"}
    assert lock_path.exists()

    record = json.loads(lock_path.read_text(encoding="utf-8"))
    assert record["final"] is True
    assert record["row_count"] == 4

    with pytest.raises(RuntimeError, match="already recorded"):
        evaluate_policies(
            labels,
            scores,
            alerts,
            amounts,
            split="test",
            final=True,
            steps=steps,
            lock_path=lock_path,
        )


def test_label_history_respects_24_step_delay():
    # At query step 100, only events at steps <= 76 are eligible.
    counts = past_label_counts(
        query_accounts=["A"],
        query_steps=[100],
        event_accounts=["A", "A", "A", "A"],
        event_steps=[20, 76, 77, 99],
        event_labels=[1, 1, 1, 1],
        label_delay_steps=LABEL_DELAY_STEPS,
    )

    assert counts.tolist() == [2]


def test_future_rows_cannot_change_earlier_history_features():
    query_accounts = ["A", "A", "A"]
    query_steps = [50, 100, 120]

    event_accounts = ["A", "A", "A", "A"]
    event_steps = [10, 76, 101, 150]
    event_labels = [1, 1, 0, 1]

    baseline = past_label_counts(
        query_accounts,
        query_steps,
        event_accounts,
        event_steps,
        event_labels,
        label_delay_steps=24,
    )

    # Perturb a future row, beyond every query time.
    changed_labels = list(event_labels)
    changed_labels[3] = 0

    perturbed = past_label_counts(
        query_accounts,
        query_steps,
        event_accounts,
        event_steps,
        changed_labels,
        label_delay_steps=24,
    )

    assert np.array_equal(baseline, perturbed)


def test_top_k_helper_obeys_daily_capacity():
    steps = [1, 2, 3, 24, 25, 26]
    scores = [0.1, 0.9, 0.8, 0.2, 0.3, 0.7]

    alerts = top_k_alert_mask_per_day(
        scores,
        steps,
        alerts_per_day=1,
    )

    # Steps 1-24 form one group; steps 25-26 form another.
    assert alerts.tolist() == [
        False, True, False, False, False, True
    ]


def test_metrics_calculate_expected_values():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8]
    alerts = [False, True, False, False]
    amounts = [100.0, 500.0, 200.0, 700.0]

    metrics = calculate_metrics(
        labels,
        scores,
        alerts,
        amounts,
    )

    assert metrics["alert_count"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["dollar_weighted_recall"] == pytest.approx(
        500.0 / 1200.0
    )
