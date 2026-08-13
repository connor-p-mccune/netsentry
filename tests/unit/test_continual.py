"""Continual-learning metrics and the replay buffer they depend on.

The retention-matrix metrics are easy to get subtly wrong in ways that flatter the result --
averaging over the wrong triangle, or comparing an old task's final score against the *best*
score anyone achieved instead of against the score it earned when it was learned. Each one is
pinned here on a matrix small enough to verify by hand.
"""

from __future__ import annotations

import numpy as np

from netsentry.training.continual import (
    ReservoirBuffer,
    StrategyResult,
    backward_transfer,
    forward_transfer,
)


def _result(matrix: np.ndarray) -> StrategyResult:
    n = matrix.shape[0]
    return StrategyResult(
        name="test",
        description="",
        matrix=matrix,
        seconds=[1.0] * n,
        trees=[10] * n,
        rows_touched=[100] * n,
        buffer_rows=0,
    )


def test_backward_transfer_is_zero_when_nothing_is_forgotten() -> None:
    matrix = np.array([[0.8, 0.1], [0.8, 0.9]])
    assert backward_transfer(matrix) == 0.0


def test_backward_transfer_is_negative_when_old_tasks_decay() -> None:
    # Task 0 was learned at 0.80 and ends at 0.50; task 1 at 0.90 and ends at 0.90.
    matrix = np.array([[0.8, 0.1, 0.1], [0.7, 0.9, 0.2], [0.5, 0.9, 0.6]])
    assert np.isclose(backward_transfer(matrix), np.mean([0.5 - 0.8, 0.9 - 0.9]))
    assert backward_transfer(matrix) < 0


def test_backward_transfer_compares_against_when_the_task_was_learned() -> None:
    # An intermediate row scoring higher than the diagonal must not become the reference:
    # forgetting is measured from what the task earned on arrival, not from its best moment.
    matrix = np.array([[0.5, 0.1], [0.5, 0.9]])
    lucky = np.array([[0.5, 0.1], [0.5, 0.9]])
    lucky[0, 0] = 0.5
    assert backward_transfer(matrix) == backward_transfer(lucky) == 0.0


def test_backward_transfer_of_a_single_task_is_zero() -> None:
    assert backward_transfer(np.array([[0.7]])) == 0.0


def test_forward_transfer_is_zero_shot_minus_prevalence() -> None:
    # R[0][1] = 0.30 against a 0.10 base rate -> +0.20 of genuinely transferable signal.
    matrix = np.array([[0.8, 0.3], [0.7, 0.9]])
    assert np.isclose(forward_transfer(matrix, np.array([0.5, 0.1])), 0.2)


def test_forward_transfer_is_negative_when_the_model_is_worse_than_chance() -> None:
    matrix = np.array([[0.8, 0.05], [0.7, 0.9]])
    assert forward_transfer(matrix, np.array([0.5, 0.4])) < 0


def test_strategy_average_uses_the_final_row_only() -> None:
    result = _result(np.array([[0.9, 0.0], [0.1, 0.5]]))
    assert np.isclose(result.average, 0.3)
    assert np.isclose(result.learning_accuracy, 0.7)


def test_reservoir_keeps_everything_while_it_fits() -> None:
    rng = np.random.default_rng(0)
    buffer = ReservoirBuffer(capacity=10, n_features=2)
    x = np.arange(12, dtype=float).reshape(6, 2)
    buffer.add(x, np.zeros(6, dtype=int), rng)
    kept, _ = buffer.rows
    assert len(kept) == 6
    assert np.allclose(np.sort(kept[:, 0]), np.sort(x[:, 0]))


def test_reservoir_never_exceeds_its_capacity() -> None:
    rng = np.random.default_rng(1)
    buffer = ReservoirBuffer(capacity=5, n_features=1)
    for _ in range(4):
        buffer.add(np.arange(20, dtype=float).reshape(20, 1), np.zeros(20, dtype=int), rng)
    kept, labels = buffer.rows
    assert len(kept) == 5 and len(labels) == 5
    assert buffer.n_seen == 80


def test_reservoir_is_uniform_over_the_whole_stream() -> None:
    """The property that makes replay a memory rather than a recency bias.

    Every row seen must survive with probability capacity/n. If the buffer were "keep the last
    k" -- or a fresh subsample of buffer-plus-new-batch each round -- the early rows would be
    systematically under-represented, replay would silently become a recency policy, and the
    forgetting it appears to prevent would be measured against the wrong baseline.
    """
    capacity, stream, trials = 4, 40, 400
    early = 0
    late = 0
    rng = np.random.default_rng(7)
    for _ in range(trials):
        buffer = ReservoirBuffer(capacity=capacity, n_features=1)
        for start in range(0, stream, 10):  # arrives in batches, as tasks do
            rows = np.arange(start, start + 10, dtype=float).reshape(10, 1)
            buffer.add(rows, np.zeros(10, dtype=int), rng)
        kept = set(buffer.rows[0][:, 0].tolist())
        early += int(0.0 in kept)
        late += int(float(stream - 1) in kept)
    expected = trials * capacity / stream
    assert abs(early - expected) < 0.4 * expected
    assert abs(late - expected) < 0.4 * expected


def test_a_zero_capacity_reservoir_remembers_nothing_but_still_counts() -> None:
    # The left end of the buffer sweep: replay with an empty buffer must *be* naive fine-tuning.
    rng = np.random.default_rng(2)
    buffer = ReservoirBuffer(capacity=0, n_features=3)
    buffer.add(np.ones((5, 3)), np.ones(5, dtype=int), rng)
    kept, labels = buffer.rows
    assert len(kept) == 0 and len(labels) == 0
    assert buffer.n_seen == 5
