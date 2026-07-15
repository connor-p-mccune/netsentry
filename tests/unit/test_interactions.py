"""Friedman's H-statistic: the additive/interaction decomposition, checked on functions
with a known interaction structure, plus the end-to-end pairwise estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.explain.interactions import h_statistic, pairwise_h


def test_h_is_zero_for_a_purely_additive_response() -> None:
    # PD_jk exactly equals PD_j + PD_k -> no interaction.
    pd_j = np.array([0.0, 1.0, 2.0, 3.0])
    pd_k = np.array([0.0, 0.5, 1.0, 1.5])
    pd_jk = pd_j + pd_k + 7.0  # an additive constant washes out under centring
    assert h_statistic(pd_jk, pd_j, pd_k) == 0.0


def test_h_is_one_when_the_joint_is_pure_interaction() -> None:
    # Marginals are flat (mean-centred to zero); all structure is in the joint term.
    pd_j = np.zeros(5)
    pd_k = np.zeros(5)
    pd_jk = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert h_statistic(pd_jk, pd_j, pd_k) == 1.0


def test_h_is_zero_for_a_constant_joint_response() -> None:
    assert h_statistic(np.full(4, 0.3), np.zeros(4), np.zeros(4)) == 0.0


def test_h_is_clipped_to_unit_interval() -> None:
    rng = np.random.default_rng(0)
    pd_jk = rng.standard_normal(50)
    pd_j = rng.standard_normal(50)
    pd_k = rng.standard_normal(50)
    h = h_statistic(pd_jk, pd_j, pd_k)
    assert 0.0 <= h <= 1.0


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # Mean-centred features so a product score has no additive leakage.
    a = rng.uniform(-1, 1, size=200)
    b = rng.uniform(-1, 1, size=200)
    c = rng.uniform(-1, 1, size=200)
    return pd.DataFrame({"a": a - a.mean(), "b": b - b.mean(), "c": c - c.mean()})


def test_pairwise_h_recovers_a_multiplicative_interaction() -> None:
    sample = _sample()

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        return (frame["a"] * frame["b"]).to_numpy()

    # a*b is a pure interaction; a-c and b-c are additive (c does not appear).
    assert pairwise_h(score_fn, sample, "a", "b") > 0.8
    assert pairwise_h(score_fn, sample, "a", "c") < 0.2


def test_pairwise_h_is_low_for_an_additive_model() -> None:
    sample = _sample()

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        return (2.0 * frame["a"] + 3.0 * frame["b"]).to_numpy()

    assert pairwise_h(score_fn, sample, "a", "b") < 0.05
