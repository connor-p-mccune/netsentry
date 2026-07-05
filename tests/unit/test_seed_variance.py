"""Seed-variance summary statistics: the pure math behind the training-noise audit."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.seed_variance import summarize_seed_runs


def test_summary_matches_hand_computed_stats() -> None:
    runs = [{"pr_auc": 0.50}, {"pr_auc": 0.52}, {"pr_auc": 0.54}]
    stats = summarize_seed_runs(runs)["pr_auc"]
    assert np.isclose(stats.mean, 0.52)
    assert np.isclose(stats.std, np.std([0.50, 0.52, 0.54], ddof=1))  # sample sd, not population
    assert stats.low == 0.50 and stats.high == 0.54
    assert np.isclose(stats.spread, 0.04)


def test_summary_covers_every_metric_key() -> None:
    runs = [
        {"pr_auc": 0.5, "tpr_fpr_1pct": 0.2},
        {"pr_auc": 0.6, "tpr_fpr_1pct": 0.3},
    ]
    stats = summarize_seed_runs(runs)
    assert set(stats) == {"pr_auc", "tpr_fpr_1pct"}
    assert np.isclose(stats["tpr_fpr_1pct"].mean, 0.25)


def test_single_run_reports_zero_spread_not_nan() -> None:
    stats = summarize_seed_runs([{"pr_auc": 0.5}])["pr_auc"]
    assert stats.std == 0.0  # ddof=1 on one sample would be NaN; zero is the honest floor
    assert stats.spread == 0.0


def test_empty_runs_yield_empty_summary() -> None:
    assert summarize_seed_runs([]) == {}


def test_identical_runs_have_zero_noise() -> None:
    runs = [{"pr_auc": 0.5}] * 4
    stats = summarize_seed_runs(runs)["pr_auc"]
    assert stats.std == 0.0 and stats.low == stats.high == 0.5
