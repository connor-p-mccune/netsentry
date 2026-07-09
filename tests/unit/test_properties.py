"""Property-based invariants (hypothesis) for the logic the results stand on.

Example-based tests pin known cases; these assert the *contracts* hold for every
input hypothesis can dream up: the FPR budget is never exceeded on the selecting
set, detection is monotone in the budget, confusion rates stay coherent, PSI
behaves like the divergence it is, and cleaning's guarantees (stripped headers,
no Inf, no identifiers, no duplicates, coherent targets) survive adversarial
frames — the quirky fixture, generalized.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp
from hypothesis import strategies as st

from netsentry.config import Settings
from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET, clean_dataframe
from netsentry.evaluation.metrics import (
    attack_probability,
    rates_at_threshold,
    threshold_at_fpr,
    tpr_at_fpr,
)
from netsentry.monitoring.drift import population_stability_index

SETTINGS = Settings()  # pure defaults; no YAML/env needed for these contracts

# --- shared strategies --------------------------------------------------------

# A scored binary problem: labels with both classes present, scores in [0, 1].
_scored = st.integers(min_value=4, max_value=60).flatmap(
    lambda n: st.tuples(
        st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n).filter(
            lambda ys: 0 < sum(ys) < len(ys)
        ),
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n),
    )
)

_budget = st.sampled_from([0.001, 0.01, 0.05, 0.1, 0.25])


# --- operating-point metrics ----------------------------------------------------


@given(_scored, _budget)
@hyp(deadline=None)
def test_fpr_budget_is_never_exceeded_on_the_selecting_set(
    problem: tuple[list[int], list[float]], budget: float
) -> None:
    y, s = np.array(problem[0]), np.array(problem[1])
    threshold = threshold_at_fpr(y, s, budget)
    # The guarantee the whole thresholding story rests on: on the set used to
    # choose it, the threshold's realized FPR respects the budget.
    assert rates_at_threshold(y, s, threshold)["fpr"] <= budget + 1e-12


@given(_scored)
@hyp(deadline=None)
def test_detection_is_monotone_in_the_fpr_budget(problem: tuple[list[int], list[float]]) -> None:
    y, s = np.array(problem[0]), np.array(problem[1])
    tprs = [tpr_at_fpr(y, s, budget)[1] for budget in (0.001, 0.01, 0.1, 0.5)]
    # Loosening the false-positive budget can never lose detection.
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(tprs))


@given(_scored, st.floats(min_value=-0.5, max_value=1.5, allow_nan=False))
@hyp(deadline=None)
def test_rates_are_coherent_at_any_threshold(
    problem: tuple[list[int], list[float]], threshold: float
) -> None:
    y, s = np.array(problem[0]), np.array(problem[1])
    rates = rates_at_threshold(y, s, threshold)
    assert rates["tp"] + rates["fn"] == int(y.sum())
    assert rates["fp"] + rates["tn"] == int((y == 0).sum())
    for key in ("tpr", "fpr", "precision"):
        assert 0.0 <= rates[key] <= 1.0


@given(
    st.integers(min_value=1, max_value=40),
    st.integers(min_value=2, max_value=5),
    st.randoms(use_true_random=False),
)
@hyp(deadline=None)
def test_attack_probability_is_a_probability(n: int, k: int, rnd: object) -> None:
    rng = np.random.default_rng(getattr(rnd, "randint")(0, 2**31))  # noqa: B009
    raw = rng.random((n, k)) + 1e-9
    proba = raw / raw.sum(axis=1, keepdims=True)
    for classes in (np.arange(k), np.array(["BENIGN", *[f"atk{i}" for i in range(k - 1)]])):
        p = attack_probability(proba, classes)
        assert p.shape == (n,)
        assert np.all((p >= 0.0) & (p <= 1.0 + 1e-12))


# --- drift (PSI) ---------------------------------------------------------------

_sample = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=8, max_size=200
)


@given(_sample)
@hyp(deadline=None)
def test_psi_of_a_distribution_against_itself_is_zero(values: list[float]) -> None:
    x = np.array(values)
    assert population_stability_index(x, x) == 0.0


@given(_sample, _sample)
@hyp(deadline=None)
def test_psi_is_nonnegative_and_finite(a: list[float], b: list[float]) -> None:
    psi = population_stability_index(np.array(a), np.array(b))
    assert np.isfinite(psi)
    assert psi >= 0.0


@given(_sample, st.floats(min_value=1e7, max_value=1e9))
@hyp(deadline=None)
def test_psi_flags_a_disjoint_shift_as_major(values: list[float], shift: float) -> None:
    x = np.array(values)
    # Every shifted value lands beyond the reference's top bin edge: a total
    # migration must read as major drift (>= 0.25), or the alarm is useless.
    assert population_stability_index(x, x + shift) >= 0.25


# --- cleaning ------------------------------------------------------------------


@st.composite
def raw_frames(draw: st.DrawFn) -> pd.DataFrame:
    """Adversarial raw frames: padded headers, Infs, dupes, identifiers, sentinels."""
    n = draw(st.integers(min_value=2, max_value=12))
    labels = draw(
        st.lists(
            st.sampled_from(["BENIGN", "DoS Hulk", "PortScan", "Web Attack \x96 XSS"]),
            min_size=n,
            max_size=n,
        )
    )
    rates = draw(
        st.lists(
            st.one_of(
                st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
                st.just(float("inf")),
                st.just(float("-inf")),
            ),
            min_size=n,
            max_size=n,
        )
    )
    pad_headers = draw(st.booleans())
    frame = pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n)],
            "Source IP": ["1.1.1.1"] * n,
            "Destination Port": draw(
                st.lists(st.integers(min_value=0, max_value=65535), min_size=n, max_size=n)
            ),
            "Flow Duration": draw(
                st.lists(st.integers(min_value=0, max_value=10**9), min_size=n, max_size=n)
            ),
            "Flow Bytes/s": rates,
            "Init_Win_bytes_forward": draw(
                st.lists(st.sampled_from([-1, 0, 256, 8192]), min_size=n, max_size=n)
            ),
            "Label": labels,
        }
    )
    if draw(st.booleans()):  # plant an exact duplicate row
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    if pad_headers:
        frame.columns = [f" {c} " for c in frame.columns]
    return frame


@given(raw_frames())
@hyp(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_cleaning_guarantees_hold_for_any_raw_frame(frame: pd.DataFrame) -> None:
    cleaned = clean_dataframe(frame, SETTINGS)

    # Headers stripped; no identifier/leaky column survives.
    assert all(col == col.strip() for col in cleaned.columns)
    assert not set(schema.identifier_columns()) & set(cleaned.columns)
    # No Inf anywhere (the classic CIC defect), regardless of where it appeared.
    numeric = cleaned.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()
    # Exact duplicates are gone.
    assert not cleaned.duplicated().any()
    # Targets exist and agree with the (normalised) label.
    assert set(cleaned[BINARY_TARGET].unique()) <= {0, 1}
    is_attack = cleaned[schema.LABEL_COLUMN] != SETTINGS.labels.benign_label
    assert (cleaned[BINARY_TARGET] == is_attack.astype(int)).all()
