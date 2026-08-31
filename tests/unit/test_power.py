"""The resolution machinery: bootstrap, pairing, the permutation null, and the bar.

The load-bearing claim of the study is that pairing removes the noise two models share, so that
is tested as an invariant rather than illustrated by a table. The rest pins the arithmetic --
a minimum detectable effect is 2.80 standard errors and nothing more -- and the formatting
decision that keeps a 0.1% budget from rendering as `0.0%`.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.power import (
    POWER_FACTOR,
    ComparisonRow,
    MetricRow,
    PowerStudy,
    PublishedClaim,
    alert_rate,
    as_percent,
    bootstrap,
    detection_rate,
    fixed_threshold_fpr,
    fixed_threshold_tpr,
    interval,
    label,
    paired_bootstrap,
    permutation_null,
    pr_auc,
    roc_auc,
    standard_error,
)

Y = np.tile([0, 1], 200)


def _correlated(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two scorers that mostly agree -- the situation pairing is for."""
    shared = rng.random(len(Y)) * 0.5 + Y * 0.3
    return shared + rng.normal(0, 0.02, len(Y)), shared + rng.normal(0, 0.02, len(Y))


# --------------------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------------------


def test_a_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 1, 1])
    assert pr_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert roc_auc(y, y.astype(float)) == pytest.approx(1.0)


def test_a_single_class_sample_is_undefined_rather_than_wrong() -> None:
    """Bootstrap resamples can lose a rare class; a silent zero would bias every interval."""
    y = np.zeros(4, dtype=int)
    assert np.isnan(pr_auc(y, np.arange(4.0)))
    assert np.isnan(roc_auc(y, np.arange(4.0)))


def test_the_threshold_does_not_move_with_the_sample() -> None:
    """The deployed cut is chosen once on validation; re-picking it would measure something else."""
    metric = fixed_threshold_tpr(0.5)
    y = np.array([1, 1, 0, 0])
    assert metric(y, np.array([0.9, 0.1, 0.2, 0.3])) == pytest.approx(0.5)
    assert metric(y, np.array([0.9, 0.6, 0.2, 0.3])) == pytest.approx(1.0)


def test_the_false_positive_rate_counts_only_benign_flows() -> None:
    metric = fixed_threshold_fpr(0.5)
    y = np.array([1, 1, 0, 0])
    assert metric(y, np.array([0.9, 0.9, 0.9, 0.1])) == pytest.approx(0.5)


def test_the_alert_rate_counts_every_flow() -> None:
    metric = alert_rate(0.5)
    assert metric(np.array([1, 0, 0, 0]), np.array([0.9, 0.9, 0.1, 0.1])) == pytest.approx(0.5)


def test_detection_rate_reads_decisions_rather_than_scores() -> None:
    """Two models with different thresholds are only comparable through their alert bits."""
    y = np.array([1, 1, 0, 0])
    assert detection_rate(y, np.array([1.0, 0.0, 1.0, 1.0])) == pytest.approx(0.5)


def test_detection_rate_survives_a_sample_with_no_attacks() -> None:
    assert detection_rate(np.zeros(3, dtype=int), np.ones(3)) == 0.0


# --------------------------------------------------------------------------------------
# The bootstrap.
# --------------------------------------------------------------------------------------


def test_a_bigger_sample_gives_a_smaller_standard_error() -> None:
    rng = np.random.default_rng(0)
    small = np.tile([0, 1], 25)
    large = np.tile([0, 1], 500)
    small_error = standard_error(bootstrap(pr_auc, small, rng.random(len(small)), 80, rng))
    large_error = standard_error(bootstrap(pr_auc, large, rng.random(len(large)), 80, rng))
    assert large_error < small_error


def test_pairing_narrows_the_interval_when_the_models_agree() -> None:
    """The study's load-bearing claim, as an invariant rather than a table."""
    rng = np.random.default_rng(1)
    first, second = _correlated(rng)
    paired = paired_bootstrap(pr_auc, Y, first, second, 120, np.random.default_rng(2))
    left = bootstrap(pr_auc, Y, first, 120, np.random.default_rng(3))
    right = bootstrap(pr_auc, Y, second, 120, np.random.default_rng(4))
    assert standard_error(paired) < standard_error(left - right)


def test_the_paired_difference_is_centred_on_the_observed_one() -> None:
    rng = np.random.default_rng(5)
    first, second = _correlated(rng)
    draws = paired_bootstrap(pr_auc, Y, first, second, 200, np.random.default_rng(6))
    observed = pr_auc(Y, first) - pr_auc(Y, second)
    assert float(np.mean(draws)) == pytest.approx(observed, abs=0.02)


def test_an_interval_ignores_undefined_resamples() -> None:
    draws = np.array([0.1, 0.2, np.nan, 0.3, np.inf])
    low, high = interval(draws, 0.95)
    assert 0.1 <= low <= high <= 0.3


def test_an_interval_with_nothing_finite_is_undefined_rather_than_zero() -> None:
    low, high = interval(np.array([np.nan, np.nan]), 0.95)
    assert np.isnan(low) and np.isnan(high)


# --------------------------------------------------------------------------------------
# The permutation null.
# --------------------------------------------------------------------------------------


def test_two_identical_scorers_produce_no_difference_under_the_null() -> None:
    rng = np.random.default_rng(7)
    scores = rng.random(len(Y))
    null = permutation_null(pr_auc, Y, scores, scores, 50, rng)
    assert np.allclose(null, 0.0)


def test_the_null_is_centred_on_zero() -> None:
    rng = np.random.default_rng(8)
    first, second = _correlated(rng)
    null = permutation_null(pr_auc, Y, first, second, 200, np.random.default_rng(9))
    assert float(np.mean(null)) == pytest.approx(0.0, abs=0.02)


def test_a_real_difference_falls_outside_the_null() -> None:
    rng = np.random.default_rng(10)
    good = Y + rng.normal(0, 0.1, len(Y))
    bad = rng.random(len(Y))
    null = permutation_null(pr_auc, Y, good, bad, 200, np.random.default_rng(11))
    observed = pr_auc(Y, good) - pr_auc(Y, bad)
    assert float(np.mean(np.abs(null) >= abs(observed))) < 0.05


# --------------------------------------------------------------------------------------
# The bar, and how it is read.
# --------------------------------------------------------------------------------------


def _row(value: float, error: float, decisive: int = 10, percent: bool = False) -> MetricRow:
    return MetricRow(
        name="pr_auc",
        value=value,
        low=value - 2 * error,
        high=value + 2 * error,
        error=error,
        decisive=decisive,
        percent=percent,
    )


def test_the_detectable_effect_is_two_point_eight_standard_errors() -> None:
    assert _row(0.5, 0.01).detectable == pytest.approx(POWER_FACTOR * 0.01)
    assert abs(POWER_FACTOR - 2.8016) < 1e-3


def test_the_relative_bar_is_the_bar_over_the_value() -> None:
    row = _row(0.5, 0.01)
    assert row.relative == pytest.approx(row.detectable / 0.5)


def test_a_metric_of_zero_has_an_infinite_relative_bar() -> None:
    assert _row(0.0, 0.01).relative == float("inf")


def test_a_small_rate_keeps_its_digits() -> None:
    """One decimal place renders a 0.1% budget as `0.0%`, which is not a number."""
    assert as_percent(0.00048) == "0.048%"
    assert as_percent(0.207) == "20.7%"
    assert as_percent(-0.0019, signed=True) == "-0.190%"


def test_a_claim_at_exactly_the_bar_clears_it() -> None:
    claim = PublishedClaim("r.md", "d", 0.017, "pr_auc", 0.017)
    assert claim.clears and claim.marginal


def test_a_claim_just_under_the_bar_does_not() -> None:
    claim = PublishedClaim("r.md", "d", 0.0169, "pr_auc", 0.017)
    assert not claim.clears


def test_a_claim_well_over_the_bar_is_not_marginal() -> None:
    claim = PublishedClaim("r.md", "d", 0.257, "pr_auc", 0.017)
    assert claim.clears and not claim.marginal


def test_a_negative_claim_is_judged_on_its_size() -> None:
    assert PublishedClaim("r.md", "d", -0.03, "pr_auc", 0.017).clears


# --------------------------------------------------------------------------------------
# The record.
# --------------------------------------------------------------------------------------


def _study(metrics: list[MetricRow]) -> PowerStudy:
    return PowerStudy(
        metrics=metrics,
        comparisons=[ComparisonRow("PR-AUC", 0.01, -0.01, 0.03, -0.05, 0.07, 0.4)],
        published=[],
        n_test=100,
        n_attacks=25,
        n_benign=75,
    )


def test_the_operational_metric_is_the_tightest_budget() -> None:
    rows = [
        MetricRow(name="tpr_at_0.01", value=0.2, low=0.19, high=0.21, error=0.005, decisive=100),
        MetricRow(name="tpr_at_0.001", value=0.09, low=0.08, high=0.1, error=0.004, decisive=50),
        MetricRow(name="pr_auc", value=0.53, low=0.52, high=0.54, error=0.006, decisive=250),
    ]
    assert _study(rows).operational().name == "tpr_at_0.001"


def test_the_least_certain_metric_is_the_one_with_the_widest_relative_bar() -> None:
    rows = [
        MetricRow(name="pr_auc", value=0.53, low=0.5, high=0.56, error=0.006, decisive=250),
        MetricRow(name="fpr_at_0.001", value=0.0005, low=0.0, high=0.001, error=0.0002, decisive=9),
    ]
    study = _study(rows)
    assert study.least_certain().name == "fpr_at_0.001"
    assert study.most_certain().name == "pr_auc"


def test_pairing_narrowing_is_the_ratio_of_half_widths() -> None:
    row = ComparisonRow("PR-AUC", 0.01, -0.01, 0.03, -0.05, 0.07, 0.4)
    assert row.narrowing == pytest.approx(0.06 / 0.02)


def test_a_difference_whose_interval_spans_zero_is_not_significant() -> None:
    assert not ComparisonRow("m", 0.01, -0.01, 0.03, -0.05, 0.07, 0.4).significant
    assert ComparisonRow("m", 0.04, 0.01, 0.07, -0.01, 0.09, 0.02).significant


def test_metric_labels_read_as_prose() -> None:
    assert label("pr_auc") == "PR-AUC"
    assert label("tpr_at_0.001") == "TPR at a 0.1% budget"
