"""The corruptions, the predictions, and the rule that makes the suite mean something.

The predictions are derivations, not conventions, so they are tested as such: an uninformative
ranking scores at the prevalence and a threshold at a 1% budget detects 1% of attacks. The other
load-bearing test is `suite_holds`, which must require the *positive* arms to separate -- a
harness that returned chance unconditionally would pass every negative control, and a suite that
called that a success would be worse than no suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.controls import (
    NEGATIVE,
    POSITIVE,
    ControlRow,
    ControlsStudy,
    Prediction,
    chance,
    constant_features,
    noise_features,
    permute_columns,
    permute_labels,
    signal,
)

# --------------------------------------------------------------------------------------
# The corruptions.
# --------------------------------------------------------------------------------------


def test_permuting_labels_keeps_the_class_balance_exactly() -> None:
    """The prediction is only exact because the prevalence does not move."""
    y = np.array([1] * 30 + [0] * 70)
    permuted = permute_labels(y, np.random.default_rng(0))
    assert int(permuted.sum()) == 30


def test_permuting_labels_actually_moves_them() -> None:
    y = np.array([1] * 30 + [0] * 70)
    assert not np.array_equal(permute_labels(y, np.random.default_rng(0)), y)


def test_shuffling_columns_keeps_every_column_distribution() -> None:
    """The strict control: marginals survive, only the row correspondence breaks."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4)) * np.array([1.0, 10.0, 100.0, 0.1])
    scrambled = permute_columns(x, rng)
    for column in range(x.shape[1]):
        assert np.allclose(np.sort(scrambled[:, column]), np.sort(x[:, column]))


def test_shuffling_columns_breaks_the_row_correspondence() -> None:
    rng = np.random.default_rng(2)
    x = np.tile(np.arange(200.0).reshape(-1, 1), (1, 3))
    scrambled = permute_columns(x, rng)
    assert not np.allclose(scrambled[:, 0], scrambled[:, 1])


def test_shuffling_columns_does_not_mutate_its_input() -> None:
    x = np.arange(20.0).reshape(10, 2)
    permute_columns(x, np.random.default_rng(3))
    assert x[0, 0] == 0.0 and x[9, 1] == 19.0


def test_noise_features_keep_only_the_shape() -> None:
    x = np.ones((50, 3)) * 1000.0
    noisy = noise_features(x, np.random.default_rng(4))
    assert noisy.shape == x.shape
    assert abs(float(np.mean(noisy))) < 0.5


def test_constant_features_leave_nothing_to_split_on() -> None:
    x = np.random.default_rng(5).normal(size=(20, 4))
    flat = constant_features(x)
    assert flat.shape == x.shape
    assert len(np.unique(flat)) == 1


# --------------------------------------------------------------------------------------
# The predictions.
# --------------------------------------------------------------------------------------


def test_chance_is_the_prevalence_and_the_budget() -> None:
    """Both derivations, not conventions: precision at the base rate, TPR at the FPR."""
    predicted = chance(prevalence=0.25, budget=0.01, tolerance=0.03)
    assert predicted.pr_auc == pytest.approx(0.25)
    assert predicted.detection == pytest.approx(0.01)
    assert predicted.direction == NEGATIVE


def test_a_negative_prediction_needs_both_numbers_to_land() -> None:
    predicted = chance(0.25, 0.01, 0.03)
    assert predicted.holds(0.26, 0.02)
    assert not predicted.holds(0.26, 0.30)  # detection wrong
    assert not predicted.holds(0.60, 0.02)  # PR-AUC wrong


def test_a_negative_prediction_fails_in_either_direction() -> None:
    """Scoring far *below* chance is as much a defect as scoring above it."""
    predicted = chance(0.25, 0.01, 0.03)
    assert not predicted.holds(0.10, 0.01)


def test_a_positive_prediction_is_a_floor_not_a_band() -> None:
    predicted = signal(floor=0.40, tolerance=0.03)
    assert predicted.holds(0.99, 0.0)
    assert predicted.holds(0.38, 0.0)  # inside tolerance of the floor
    assert not predicted.holds(0.20, 0.0)
    assert predicted.direction == POSITIVE


# --------------------------------------------------------------------------------------
# The rows and the suite.
# --------------------------------------------------------------------------------------


def _row(name: str, direction: str, predicted: Prediction, pr_auc: float, detection: float = 0.01):
    return ControlRow(
        name=name,
        describes="",
        direction=direction,
        predicted=predicted,
        pr_auc=pr_auc,
        detection=detection,
        realised_fpr=0.01,
    )


def _suite(rows: list[ControlRow]) -> ControlsStudy:
    return ControlsStudy(
        rows=rows, prevalence=0.25, budget=0.01, tolerance=0.03, n_train=100, n_test=100
    )


def _passing() -> list[ControlRow]:
    return [
        _row("intact", POSITIVE, signal(0.40, 0.03), 0.53, 0.21),
        _row("permuted labels", NEGATIVE, chance(0.25, 0.01, 0.03), 0.2535),
        _row("shuffled columns", NEGATIVE, chance(0.25, 0.01, 0.03), 0.2445),
        _row("leaked", POSITIVE, signal(0.90, 0.03), 1.0, 1.0),
    ]


def test_a_clean_suite_holds() -> None:
    assert _suite(_passing()).suite_holds()


def test_a_negative_arm_above_chance_breaks_the_suite() -> None:
    rows = _passing()
    rows[1] = _row("permuted labels", NEGATIVE, chance(0.25, 0.01, 0.03), 0.62)
    study = _suite(rows)
    assert not study.suite_holds()
    assert [row.name for row in study.failures()] == ["permuted labels"]


def test_a_positive_arm_that_fails_breaks_the_suite_too() -> None:
    """The half people leave out: a harness returning chance always would pass the negatives."""
    rows = _passing()
    rows[0] = _row("intact", POSITIVE, signal(0.40, 0.03), 0.25, 0.01)
    study = _suite(rows)
    assert not study.suite_holds()
    assert [row.name for row in study.failures()] == ["intact"]


def test_the_excess_measures_the_size_of_a_defect() -> None:
    row = _row("permuted labels", NEGATIVE, chance(0.25, 0.01, 0.03), 0.62)
    assert row.excess == pytest.approx(0.37)


def test_the_worst_negative_is_the_one_furthest_above_chance() -> None:
    study = _suite(_passing())
    assert study.worst_negative().name == "permuted labels"


def test_the_arms_are_separated_by_direction() -> None:
    study = _suite(_passing())
    assert len(study.negatives()) == 2
    assert len(study.positives()) == 2


def test_an_expectation_reads_as_a_band_or_a_floor() -> None:
    assert "+/-" in _row("n", NEGATIVE, chance(0.25, 0.01, 0.03), 0.25).expectation
    assert "at least" in _row("p", POSITIVE, signal(0.40, 0.03), 0.53).expectation
