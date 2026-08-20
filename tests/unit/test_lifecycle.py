"""The lifecycle model, checked without a server, and the machine checked against fake ones.

The model is a pure function of (state, operation, response), which is what makes it testable
on its own: a stub service can answer anything at all, and the tests assert the model reaches
the verdict the contract requires. The stub is the point -- a checker exercised only against a
correct implementation has never been shown to fail.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hyp
from hypothesis import strategies as st

from netsentry.serving.lifecycle import (
    HEALTH,
    HEAVY_OPERATIONS,
    MALFORMED,
    METRICS,
    MUTANTS,
    OPERATIONS,
    PREDICT,
    PREDICT_NO_KEY,
    RELOAD_CANARY_FAIL,
    RELOAD_ESCAPE,
    RELOAD_MISSING,
    RELOAD_VALID,
    Exploration,
    LifecycleState,
    apply_mutation,
    check_step,
    explore,
    grade,
    plan_operations,
)

START = LifecycleState(version="1.0.0", canary_ok=True, api_key="k")


def _ok(operation: str, version: str = "1.0.0") -> tuple[int, dict[str, Any] | None, str | None]:
    """How a correct service answers each operation."""
    if operation in {PREDICT, "predict batch", HEALTH}:
        return 200, {"model_version": version, "status": "ok"}, None
    if operation == PREDICT_NO_KEY:
        return 401, {"detail": "invalid or missing API key"}, None
    if operation == METRICS:
        return 200, None, None
    if operation == RELOAD_VALID:
        return 200, {"model_version": version}, version
    if operation == RELOAD_CANARY_FAIL:
        return 409, {"detail": "candidate rejected"}, None
    if operation == RELOAD_MISSING:
        return 404, {"detail": "no bundle"}, None
    if operation == RELOAD_ESCAPE:
        return 400, {"detail": "outside models dir"}, None
    if operation == MALFORMED:
        return 422, {"detail": "validation error"}, None
    raise AssertionError(operation)


def _correct_service(version: str = "1.0.0"):  # type: ignore[no-untyped-def]
    return lambda operation: _ok(operation, version)


# --------------------------------------------------------------------------------------
# The model agrees with a correct service, on every operation.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_correct_response_produces_no_violation(operation: str) -> None:
    status, body, candidate = _ok(operation)
    _, violations = check_step(START, operation, status, body, candidate)
    assert not violations, violations


def test_a_full_walk_against_a_correct_service_is_clean() -> None:
    run = explore(_correct_service(), START, 60, np.random.default_rng(0))
    assert run.clean
    assert run.n_operations == 60


# --------------------------------------------------------------------------------------
# ...and disagrees with every way of getting it wrong.
# --------------------------------------------------------------------------------------


def test_a_refused_reload_that_returns_success_is_caught() -> None:
    _, violations = check_step(START, RELOAD_CANARY_FAIL, 200, {"model_version": "2.0.0"}, None)
    assert violations


def test_a_prediction_naming_a_version_the_service_never_loaded_is_caught() -> None:
    _, violations = check_step(START, PREDICT, 200, {"model_version": "9.9.9"}, None)
    assert violations


def test_a_successful_reload_moves_the_model_and_the_next_prediction_agrees() -> None:
    state, violations = check_step(START, RELOAD_VALID, 200, {"model_version": "2.0.0"}, "2.0.0")
    assert not violations
    assert state.version == "2.0.0"
    _, after = check_step(state, PREDICT, 200, {"model_version": "2.0.0"}, None)
    assert not after
    # ...and the version it replaced is now wrong.
    _, stale = check_step(state, PREDICT, 200, {"model_version": "1.0.0"}, None)
    assert stale


def test_health_claiming_ok_with_a_failing_canary_is_caught() -> None:
    body = {"model_version": "1.0.0", "status": "ok", "canary": {"ok": False, "n": 8}}
    _, violations = check_step(START, HEALTH, 200, body, None)
    assert violations


def test_an_unauthenticated_request_that_succeeds_is_caught_when_a_key_is_set() -> None:
    _, violations = check_step(START, PREDICT_NO_KEY, 200, {"model_version": "1.0.0"}, None)
    assert violations


def test_an_unauthenticated_request_is_fine_when_no_key_is_configured() -> None:
    open_state = LifecycleState(version="1.0.0", canary_ok=True, api_key=None)
    _, violations = check_step(open_state, PREDICT_NO_KEY, 200, {"model_version": "1.0.0"}, None)
    assert not violations


# --------------------------------------------------------------------------------------
# The mutants: the evidence that the machine can fail.
# --------------------------------------------------------------------------------------


def test_every_mutant_is_caught_on_a_walk_that_covers_every_operation() -> None:
    """Each injected regression must produce at least one disagreement."""
    rng = np.random.default_rng(1)
    plan = plan_operations(120, rng, min_heavy=6, min_light=6)
    clean = explore(_correct_service(), START, len(plan), rng, plan=plan)
    assert clean.clean, clean.violations

    missed = []
    for mutant in MUTANTS:
        mutated = grade(apply_mutation(clean.transcript, mutant.rewrite), START, mutant.name)
        if mutated.clean:
            missed.append(mutant.name)
    assert not missed, f"regressions the machine did not notice: {missed}"


def test_a_mutation_only_changes_the_operations_it_targets() -> None:
    rng = np.random.default_rng(2)
    plan = plan_operations(60, rng, min_heavy=3, min_light=3)
    clean = explore(_correct_service(), START, len(plan), rng, plan=plan)
    for mutant in MUTANTS:
        mutated = apply_mutation(clean.transcript, mutant.rewrite)
        changed = {
            operation
            for (operation, *rest), original in zip(mutated, clean.transcript, strict=True)
            if (operation, *rest) != original
        }
        assert changed, mutant.name  # it has to change something
        assert len(changed) <= 2, (mutant.name, changed)  # ...and not everything


# --------------------------------------------------------------------------------------
# The schedule.
# --------------------------------------------------------------------------------------


def test_the_schedule_covers_every_operation() -> None:
    """The regression this replaced: a weighted draw left successful reloads unexercised."""
    for seed in range(5):
        plan = plan_operations(200, np.random.default_rng(seed))
        counts = {operation: plan.count(operation) for operation in OPERATIONS}
        assert min(counts.values()) > 0, counts
        for heavy in HEAVY_OPERATIONS:
            assert counts[heavy] >= 4


def test_the_schedule_keeps_the_expensive_operations_rare() -> None:
    plan = plan_operations(200, np.random.default_rng(3))
    heavy = sum(plan.count(operation) for operation in HEAVY_OPERATIONS)
    assert heavy < len(plan) * 0.15  # each one builds an inference engine


def test_the_schedule_is_deterministic_given_the_seed() -> None:
    first = plan_operations(120, np.random.default_rng(4))
    second = plan_operations(120, np.random.default_rng(4))
    assert first == second


# --------------------------------------------------------------------------------------
# A property nobody should have to enumerate by hand.
# --------------------------------------------------------------------------------------


@hyp(deadline=None)  # per-example wall-clock varies with suite load; the size bound is the guard
@given(
    operations=st.lists(st.sampled_from(OPERATIONS), min_size=1, max_size=40),
    version=st.sampled_from(["1.0.0", "2.0.0"]),
)
def test_a_correct_service_is_clean_under_any_sequence(operations: list[str], version: str) -> None:
    """Hypothesis drives the sequence; the model must never accuse a correct service.

    False alarms are what get a conformance checker deleted, so this is the property that
    matters more than the detection one.
    """
    start = LifecycleState(version=version, canary_ok=True, api_key="k")
    transcript = [(operation, *_ok(operation, version)) for operation in operations]
    run: Exploration = grade(transcript, start, "hypothesis")  # type: ignore[arg-type]
    assert run.clean, run.violations[0].violations if run.violations else ""
