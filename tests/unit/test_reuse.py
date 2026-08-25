"""The static count, the two candidate generators, and the mechanisms built on them.

The audit is the part that must not be allowed to drift, because the report states its count as
a fact about the repository. The rest pins the properties the study's conclusions rest on: that
a jittered candidate is genuinely indistinguishable and a perturbed one genuinely is not, that
the confidence gate refuses noise and accepts a real edge, and that Thresholdout stops answering
when its budget is gone rather than quietly serving reference answers forever.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from netsentry.evaluation.reuse import (
    ArmRow,
    ReuseStudy,
    RoundRow,
    SplitRead,
    _confidence_gate,
    _incumbent,
    _naive,
    _thresholdout,
    bootstrap_halfwidth,
    jitter,
    perturb,
    split_reads,
    summarise_audit,
)

# --------------------------------------------------------------------------------------
# The static count.
# --------------------------------------------------------------------------------------


def test_a_literal_split_read_is_counted() -> None:
    source = 'frame = load_split(settings, "temporal", "test")\n'
    reads = split_reads(source, "m.py")
    assert [(r.part, r.line) for r in reads] == [("test", 1)]


def test_a_keyword_split_read_is_counted() -> None:
    source = 'frame = load_split(settings, strategy="temporal", part="val")\n'
    assert [r.part for r in split_reads(source, "m.py")] == ["val"]


def test_a_qualified_call_is_counted() -> None:
    source = 'frame = split.load_split(settings, "temporal", "test")\n'
    assert [r.part for r in split_reads(source, "m.py")] == ["test"]


def test_a_non_literal_partition_is_not_guessed_at() -> None:
    """Undercounting is the safe direction for an audit that reports its own count as a fact."""
    source = "for part in parts:\n    frame = load_split(settings, 'temporal', part)\n"
    assert split_reads(source, "m.py") == []


def test_an_unrelated_call_is_not_counted() -> None:
    source = 'frame = load_frame(settings, "temporal", "test")\n'
    assert split_reads(source, "m.py") == []


def test_unparseable_source_yields_nothing_rather_than_raising() -> None:
    assert split_reads("def broken(:\n", "m.py") == []


def test_the_summary_counts_reads_and_distinct_modules() -> None:
    reads = [
        SplitRead("a.py", 1, "test"),
        SplitRead("a.py", 9, "test"),
        SplitRead("b.py", 3, "test"),
        SplitRead("b.py", 4, "val"),
    ]
    rows = {row.part: row for row in summarise_audit(reads)}
    assert (rows["test"].reads, rows["test"].modules) == (3, 2)
    assert (rows["val"].reads, rows["val"].modules) == (1, 1)


def test_the_summary_is_ordered_by_exposure() -> None:
    reads = [
        SplitRead("a.py", 1, "val"),
        SplitRead("a.py", 2, "test"),
        SplitRead("b.py", 1, "test"),
    ]
    assert [row.part for row in summarise_audit(reads)] == ["test", "val"]


# --------------------------------------------------------------------------------------
# The two candidate generators, which the whole contrast rests on.
# --------------------------------------------------------------------------------------


def test_a_jittered_candidate_carries_no_information_about_the_flow() -> None:
    """Independent per-flow noise is what makes a pool indistinguishable in truth."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 1.0, 500)
    first = jitter(base, 0.1, rng)
    second = jitter(base, 0.1, rng)
    residual_one, residual_two = first - base, second - base
    assert abs(float(np.corrcoef(residual_one, residual_two)[0, 1])) < 0.2


def test_a_perturbed_candidate_moves_similar_flows_together() -> None:
    """A feature-direction nudge is a real change of detector, not noise."""
    features = np.repeat(np.arange(50.0).reshape(-1, 1), 3, axis=1)
    direction = np.array([1.0, 0.0, 0.0])
    nudged = perturb(np.zeros(50), features, direction, 0.01)
    assert np.all(np.diff(nudged) > 0)


def test_the_bootstrap_halfwidth_shrinks_with_sample_size() -> None:
    rng = np.random.default_rng(1)
    y_small = np.tile([0, 1], 40)
    y_large = np.tile([0, 1], 400)
    small = bootstrap_halfwidth(y_small, rng.random(len(y_small)), 60, rng, 0.95)
    large = bootstrap_halfwidth(y_large, rng.random(len(y_large)), 60, rng, 0.95)
    assert large < small


def test_the_bootstrap_halfwidth_is_non_negative() -> None:
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 50)
    assert bootstrap_halfwidth(y, rng.random(100), 40, rng, 0.9) >= 0.0


# --------------------------------------------------------------------------------------
# The strategies.
# --------------------------------------------------------------------------------------

Y = np.tile([0, 1], 60)


def _pool(rng: np.random.Generator, size: int, planted_at: int | None = None) -> list:
    """A pool of indistinguishable candidates, optionally with one real improvement in it."""
    base = rng.random(len(Y)) * 0.4 + Y * 0.2
    pool = []
    for index in range(size):
        scores = base + rng.normal(0.0, 0.15, len(Y))
        if planted_at is not None and index == planted_at:
            scores = base + Y * 0.6
        pool.append((scores, scores[::-1], scores, scores))
    return pool


def test_the_incumbent_arm_asks_nothing() -> None:
    pool = _pool(np.random.default_rng(3), 5)
    arm = _incumbent(Y, Y, pool)
    assert arm.queries == 0 and arm.adopted == 0 and arm.budget_spent == 0


def test_the_naive_analyst_answers_every_query() -> None:
    pool = _pool(np.random.default_rng(4), 21)
    arm, trace = _naive(Y, Y, pool, rounds=4, per_round=5, planted=-1)
    assert arm.queries == 20
    assert len(trace) == 4
    assert trace[-1].queries == 20


def test_the_naive_analyst_never_reports_worse_than_it_started() -> None:
    """Selection is a running maximum, so the reported number can only climb."""
    pool = _pool(np.random.default_rng(5), 21)
    arm, trace = _naive(Y, Y, pool, rounds=4, per_round=5, planted=-1)
    assert arm.reported >= trace[0].reported
    assert all(later.reported >= earlier.reported for earlier, later in pairwise(trace))


def test_the_confidence_gate_refuses_differences_inside_the_noise() -> None:
    pool = _pool(np.random.default_rng(6), 21)
    arm = _confidence_gate(Y, Y, pool, rounds=4, per_round=5, planted=-1, halfwidth=1.0)
    assert arm.adopted == 0


def test_the_confidence_gate_still_takes_a_real_improvement() -> None:
    """A gate that never adopts anything is honest and useless; this is the difference."""
    pool = _pool(np.random.default_rng(7), 21, planted_at=11)
    arm = _confidence_gate(Y, Y, pool, rounds=4, per_round=5, planted=11, halfwidth=0.02)
    assert arm.found_planted


def test_thresholdout_stops_answering_when_its_budget_is_gone() -> None:
    """Serving reference answers past exhaustion would keep the guarantee's name without it."""
    rng = np.random.default_rng(8)
    pool = _pool(rng, 21)
    disagreeing = np.zeros(len(Y))  # a reference that always disagrees burns budget every query
    pool = [(scores, sealed, disagreeing, disagreeing) for scores, sealed, _, _ in pool]
    arm = _thresholdout(
        "t",
        Y,
        Y,
        2,
        Y,
        pool,
        rounds=4,
        per_round=5,
        planted=-1,
        tolerance=0.0,
        noise=1e-9,
        budget=3,
        rng=rng,
    )
    assert arm.budget_spent == 3
    assert arm.queries == 3


def test_thresholdout_answers_for_free_when_the_reference_agrees() -> None:
    rng = np.random.default_rng(9)
    pool = _pool(rng, 21)
    arm = _thresholdout(
        "t",
        Y,
        Y,
        2,
        Y,
        pool,
        rounds=4,
        per_round=5,
        planted=-1,
        tolerance=1.0,
        noise=1e-9,
        budget=1,
        rng=rng,
    )
    assert arm.budget_spent == 0
    assert arm.queries == 20


# --------------------------------------------------------------------------------------
# The record.
# --------------------------------------------------------------------------------------


def _arm(name: str, reported: float, sealed: float, found: bool = False) -> ArmRow:
    return ArmRow(
        name=name,
        reported=reported,
        sealed=sealed,
        queries=1,
        budget_spent=0,
        adopted=0,
        found_planted=found,
    )


def _study(harm: list[ArmRow], power: list[ArmRow], contrast: list[ArmRow]) -> ReuseStudy:
    return ReuseStudy(
        audit=[],
        audit_modules=0,
        harm=harm,
        power=power,
        contrast=contrast,
        trace=[RoundRow(1, 0.5, 0.5)],
        planted_edge=0.04,
        halfwidth=0.02,
        n_holdout=10,
        n_sealed=10,
        n_reference=10,
        candidates=3,
    )


def test_the_selection_cost_nets_out_the_sampling_floor() -> None:
    """Charging selection for noise it did not cause would inflate the study's own finding."""
    harm = [_arm("incumbent", 0.60, 0.55), _arm("naive", 0.70, 0.60)]
    study = _study(harm, harm, harm)
    assert study.selection_cost() == pytest.approx(0.05)


def test_the_best_fix_must_have_found_the_planted_improvement() -> None:
    honest_but_useless = _arm("never adopts", 0.60, 0.55)
    real_fix = _arm("gate", 0.63, 0.57)
    harm = [_arm("incumbent", 0.60, 0.55), _arm("naive", 0.70, 0.60), honest_but_useless, real_fix]
    power = [
        _arm("incumbent", 0.60, 0.55),
        _arm("naive", 0.70, 0.60, found=True),
        _arm("never adopts", 0.60, 0.55, found=False),
        _arm("gate", 0.63, 0.57, found=True),
    ]
    study = _study(harm, power, harm)
    assert study.best_fix().name == "gate"
