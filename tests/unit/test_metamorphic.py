"""Metamorphic testing: the transformations, the violation measures, and the relation harness.

The point of a metamorphic oracle is that it fires on a broken system and stays silent on a
sound one. Both halves are pinned here: the transformations are checked to do exactly what the
report claims (timing up, rates down, counts untouched), and the harness is run against a
deliberately batch-dependent scorer to prove it actually detects the defect it advertises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.robustness.metamorphic import (
    Relation,
    RelationResult,
    check_relations,
    max_score_delta,
    rate_columns,
    rescale_clock,
    round_significant,
    timing_columns,
    verdict_flip_rate,
)


@pytest.fixture
def flows() -> pd.DataFrame:
    """A tiny frame carrying one column from each family the relations touch."""
    return pd.DataFrame(
        {
            "Flow Duration": [1000.0, 2000.0, 40.0],
            "Flow IAT Mean": [100.0, 250.0, 5.0],
            "Flow Bytes/s": [500.0, 1000.0, 20.0],
            "Flow Packets/s": [10.0, 20.0, 1.0],
            "Total Fwd Packets": [8.0, 16.0, 2.0],
            "SYN Flag Count": [1.0, 0.0, 1.0],
        }
    )


def test_column_families_partition_by_meaning(flows: pd.DataFrame) -> None:
    cols = list(flows.columns)
    assert set(timing_columns(cols)) == {"Flow Duration", "Flow IAT Mean"}
    assert set(rate_columns(cols)) == {"Flow Bytes/s", "Flow Packets/s"}


def test_clock_rescale_moves_times_up_and_rates_down_and_leaves_counts_alone(
    flows: pd.DataFrame,
) -> None:
    out = rescale_clock(flows, 2.0)
    assert np.allclose(out["Flow Duration"], flows["Flow Duration"] * 2)
    assert np.allclose(out["Flow IAT Mean"], flows["Flow IAT Mean"] * 2)
    assert np.allclose(out["Flow Bytes/s"], flows["Flow Bytes/s"] / 2)
    assert np.allclose(out["Total Fwd Packets"], flows["Total Fwd Packets"])
    assert np.allclose(out["SYN Flag Count"], flows["SYN Flag Count"])


def test_clock_rescale_is_invertible(flows: pd.DataFrame) -> None:
    # A re-timing followed by its inverse must return the original record exactly.
    there_and_back = rescale_clock(rescale_clock(flows, 2.5), 1 / 2.5)
    pd.testing.assert_frame_equal(there_and_back, flows)


def test_clock_rescale_rejects_a_nonpositive_factor(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="positive"):
        rescale_clock(flows, 0.0)


def test_round_significant_keeps_the_requested_digits() -> None:
    frame = pd.DataFrame({"a": [123456.0, 0.00123456, -98765.0, 0.0]})
    out = round_significant(frame, 3)
    assert np.allclose(out["a"], [123000.0, 0.00123, -98800.0, 0.0])


def test_round_significant_preserves_nan_rather_than_inventing_a_value() -> None:
    frame = pd.DataFrame({"a": [1.23456, np.nan]})
    out = round_significant(frame, 3)
    assert out["a"].iloc[0] == pytest.approx(1.23)
    assert np.isnan(out["a"].iloc[1])


def test_round_significant_rejects_zero_digits() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        round_significant(pd.DataFrame({"a": [1.0]}), 0)


def test_verdict_flip_rate_counts_only_decisions_that_cross_the_threshold() -> None:
    before = np.array([0.1, 0.9, 0.4, 0.6])
    after = np.array([0.2, 0.95, 0.7, 0.55])  # only the third crosses 0.5
    assert verdict_flip_rate(before, after, 0.5) == pytest.approx(0.25)


def test_verdict_flip_rate_ignores_rows_the_relation_did_not_score() -> None:
    # The single-vs-batch relation only scores a capped prefix; the rest come back NaN and
    # must not be counted as agreements or as flips.
    before = np.array([0.1, 0.9, 0.4])
    after = np.array([0.9, np.nan, np.nan])
    assert verdict_flip_rate(before, after, 0.5) == pytest.approx(1.0)


def test_max_score_delta_is_zero_for_an_identical_rescoring() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    assert max_score_delta(scores, scores) == 0.0


def test_structural_relations_demand_exactness_and_semantic_ones_only_demand_the_verdict() -> None:
    # A tiny score movement that crosses no threshold is a *finding* on a semantic relation
    # (the model wobbled) but a *defect* on a structural one (the same input scored twice).
    common = {"statement": "s", "rationale": "r", "flip_rate": 0.0, "max_delta": 1e-9}
    structural = RelationResult(relation="x", kind="structural", **common)  # type: ignore[arg-type]
    semantic = RelationResult(relation="x", kind="semantic", **common)  # type: ignore[arg-type]
    assert not structural.holds
    assert semantic.holds


def _identity_relation() -> Relation:
    return Relation(
        name="identity",
        kind="structural",
        statement="rescoring the same frame gives the same scores",
        rationale="a scoring function is a function",
        evaluate=lambda score, frame, rng, cap: score(frame),
    )


def _permutation_relation() -> Relation:
    def evaluate(score, frame, rng, cap):  # type: ignore[no-untyped-def]
        order = rng.permutation(len(frame))
        shuffled = score(frame.iloc[order])
        out = np.empty(len(frame), dtype=float)
        out[order] = shuffled
        return out

    return Relation(
        name="batch permutation",
        kind="structural",
        statement="a verdict does not depend on batch position",
        rationale="batching is a transport detail",
        evaluate=evaluate,
    )


def test_relations_hold_for_a_pure_row_wise_scorer(flows: pd.DataFrame) -> None:
    def score(frame: pd.DataFrame) -> np.ndarray:
        return frame["Flow Duration"].to_numpy(dtype=float) / 10000.0

    results = check_relations(
        score, flows, [_identity_relation(), _permutation_relation()], 0.15, seed=0, single_cap=3
    )
    assert all(r.holds for r in results)
    assert all(r.max_delta == 0.0 for r in results)


def test_the_harness_catches_a_batch_dependent_scorer(flows: pd.DataFrame) -> None:
    # The archetype defect: standardising with the request's own statistics. It is a pure
    # function of the *batch*, so it survives any single-batch offline evaluation, and the
    # permutation relation is blind to it too -- only a change of batch membership exposes it.
    def score(frame: pd.DataFrame) -> np.ndarray:
        values = frame["Flow Duration"].to_numpy(dtype=float)
        return (values - values.mean()) / (values.std() + 1e-9)

    def halved(_: object, frame: pd.DataFrame, rng: object, cap: int) -> np.ndarray:
        head = score(frame.iloc[: max(1, len(frame) // 2)])
        out = np.full(len(frame), np.nan)
        out[: len(head)] = head
        return out

    relation = Relation(
        name="batch subset",
        kind="structural",
        statement="a verdict does not depend on which other flows share the batch",
        rationale="batch membership is not an input to the decision",
        evaluate=halved,  # type: ignore[arg-type]
    )
    results = check_relations(score, flows, [relation], 0.0, seed=0, single_cap=3)
    assert not results[0].holds
    assert results[0].max_delta > 0.0
