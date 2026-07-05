"""Promotion policy: the margins-and-evidence logic that guards the champion."""

from __future__ import annotations

from netsentry.config import Settings
from netsentry.evaluation.confidence import DiffResult
from netsentry.models.promotion import decide_promotion


def test_empty_registry_bootstraps_the_champion(settings: Settings) -> None:
    promote, reason = decide_promotion(settings.promotion, None, None)
    assert promote and "seeds the registry" in reason


def test_credible_regression_is_held(settings: Settings) -> None:
    # The whole CI sits below the non-inferiority margin: an unambiguous regression.
    pr = DiffResult(diff=-0.02, low=-0.03, high=-0.01, p_value=1.0)
    promote, reason = decide_promotion(settings.promotion, pr, None)
    assert not promote and "non-inferiority margin" in reason


def test_within_noise_rolls_forward_under_non_inferiority(settings: Settings) -> None:
    # Delta straddles zero but the lower bound clears the margin: parity promotes.
    settings.promotion.policy = "non_inferiority"
    settings.promotion.metric_margin = 0.005
    pr = DiffResult(diff=0.001, low=-0.003, high=0.005, p_value=0.4)
    promote, reason = decide_promotion(settings.promotion, pr, None)
    assert promote and "non-inferior" in reason


def test_superiority_policy_demands_ci_above_zero(settings: Settings) -> None:
    settings.promotion.policy = "superiority"
    parity = DiffResult(diff=0.001, low=-0.003, high=0.005, p_value=0.4)
    promote, reason = decide_promotion(settings.promotion, parity, None)
    assert not promote and "not proven better" in reason

    better = DiffResult(diff=0.02, low=0.01, high=0.03, p_value=0.001)
    promote, reason = decide_promotion(settings.promotion, better, None)
    assert promote and "credibly better" in reason


def test_tpr_regression_blocks_even_when_pr_auc_is_fine(settings: Settings) -> None:
    pr = DiffResult(diff=0.01, low=0.005, high=0.02, p_value=0.01)
    tpr = DiffResult(diff=-0.10, low=-0.15, high=-0.05, p_value=1.0)
    promote, reason = decide_promotion(settings.promotion, pr, tpr)
    assert not promote and "detection regression" in reason

    # The TPR guard is operator-disableable (e.g. when profiles are incomparable).
    settings.promotion.require_tpr_non_inferior = False
    promote, _ = decide_promotion(settings.promotion, pr, tpr)
    assert promote
