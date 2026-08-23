"""The search methods, the cost model, and the two properties the report's claims rest on.

The search algorithms are driven with a stub evaluator rather than a real model: what has to
be right is the *schedule* -- how many configurations survive each cut, at what fidelity, for
what budget -- and a stub makes that checkable exactly instead of approximately. The
winner's-curse machinery gets its own pair of tests, because a curve that rises for any input
would prove nothing, so it is shown both firing and staying flat.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.training.multifidelity import (
    Configuration,
    LadderRow,
    Trial,
    _winners_curse,
    cost_model,
    hyperband,
    random_search,
    sample_configuration,
    successive_halving,
)


def _stub(scores: dict[tuple[float, ...], float] | None = None):  # type: ignore[no-untyped-def]
    """An evaluator that records every call and scores a configuration deterministically."""
    calls: list[tuple[Configuration, int]] = []

    def evaluate(configuration: Configuration, fidelity: int) -> Trial:
        calls.append((configuration, fidelity))
        value = (
            scores.get(configuration.key(), 0.0)
            if scores is not None
            else configuration.learning_rate
        )
        return Trial(configuration, fidelity, validation=value, test=value, seconds=0.0)

    return evaluate, calls


def _configuration(rate: float) -> Configuration:
    return Configuration(
        learning_rate=rate,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )


# --------------------------------------------------------------------------------------
# The search space.
# --------------------------------------------------------------------------------------


def test_configurations_stay_inside_the_search_space() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        configuration = sample_configuration(rng)
        assert 0.01 <= configuration.learning_rate <= 0.3
        assert 8 <= configuration.num_leaves < 128
        assert 5 <= configuration.min_child_samples < 200
        assert 0.5 <= configuration.subsample <= 1.0
        assert 1e-3 <= configuration.reg_lambda <= 10.0


def test_the_learning_rate_is_sampled_in_log_space() -> None:
    """Sampling a scale uniformly is how a random-search baseline is quietly made weak."""
    rng = np.random.default_rng(1)
    rates = np.array([sample_configuration(rng).learning_rate for _ in range(4000)])
    geometric_midpoint = float(np.sqrt(0.01 * 0.3))
    assert float(np.median(rates)) == pytest.approx(geometric_midpoint, rel=0.15)


def test_the_same_configuration_has_the_same_identity() -> None:
    assert _configuration(0.1).key() == _configuration(0.1).key()
    assert _configuration(0.1).key() != _configuration(0.2).key()


# --------------------------------------------------------------------------------------
# The schedules.
# --------------------------------------------------------------------------------------


def test_random_search_spends_its_whole_budget_at_full_fidelity() -> None:
    evaluate, calls = _stub()
    trials = random_search(evaluate, np.random.default_rng(2), full_fidelity=81, budget=810)
    assert len(trials) == 10
    assert {fidelity for _, fidelity in calls} == {81}
    assert sum(trial.units for trial in trials) == 810


def test_successive_halving_keeps_one_in_eta_at_every_rung() -> None:
    evaluate, calls = _stub()
    configurations = [_configuration(0.01 * (index + 1)) for index in range(27)]
    successive_halving(evaluate, configurations, ladder=[1, 3, 9], eta=3)
    per_rung = {fidelity: sum(1 for _, f in calls if f == fidelity) for fidelity in (1, 3, 9)}
    assert per_rung == {1: 27, 3: 9, 9: 3}


def test_successive_halving_promotes_the_configurations_that_scored_best() -> None:
    """The whole method is the cut; a cut that keeps the wrong survivors is not a method."""
    winners = {_configuration(0.5).key(): 1.0, _configuration(0.4).key(): 0.9}
    configurations = [_configuration(0.5), _configuration(0.4)] + [
        _configuration(0.01 * (index + 1)) for index in range(7)
    ]
    evaluate, calls = _stub({**winners})
    successive_halving(evaluate, configurations, ladder=[1, 3], eta=3)
    promoted = {configuration.key() for configuration, fidelity in calls if fidelity == 3}
    assert set(winners) <= promoted  # both scorers survive
    assert len(promoted) == len(configurations) // 3  # and only one in eta does


def test_successive_halving_stops_when_one_configuration_is_left() -> None:
    evaluate, calls = _stub()
    successive_halving(
        evaluate, [_configuration(0.1), _configuration(0.2)], ladder=[1, 3, 9, 27], eta=3
    )
    assert max(fidelity for _, fidelity in calls) < 27


def test_hyperband_stays_inside_its_budget_and_uses_several_fidelities() -> None:
    evaluate, calls = _stub()
    trials = hyperband(evaluate, np.random.default_rng(3), full_fidelity=27, eta=3, budget=400)
    assert sum(trial.units for trial in trials) <= 400
    assert len({fidelity for _, fidelity in calls}) >= 2


def test_hyperband_terminates_when_no_bracket_fits() -> None:
    """A budget below the cheapest bracket must return, not loop forever looking for room."""
    evaluate, _ = _stub()
    assert hyperband(evaluate, np.random.default_rng(4), full_fidelity=81, eta=3, budget=1) == []


def test_hyperband_keeps_spending_when_the_budget_exceeds_one_pass() -> None:
    evaluate, _ = _stub()
    small = hyperband(evaluate, np.random.default_rng(5), full_fidelity=9, eta=3, budget=60)
    large = hyperband(evaluate, np.random.default_rng(5), full_fidelity=9, eta=3, budget=600)
    assert sum(t.units for t in large) > sum(t.units for t in small)


# --------------------------------------------------------------------------------------
# The cost model.
# --------------------------------------------------------------------------------------


def test_the_cost_model_recovers_a_known_fixed_and_marginal_cost() -> None:
    rows = [
        LadderRow(
            fidelity=fidelity,
            share_of_full=fidelity / 81,
            rank_correlation=0.0,
            learning_rate_correlation=0.0,
            top_config_kept=True,
            seconds=0.0,
            seconds_per_fit=0.9 + 0.012 * fidelity,
        )
        for fidelity in (1, 3, 9, 27, 81)
    ]
    fixed, marginal = cost_model(rows)
    assert fixed == pytest.approx(0.9, abs=1e-6)
    assert marginal == pytest.approx(0.012, abs=1e-6)


def test_the_cost_model_never_reports_a_negative_cost() -> None:
    rows = [
        LadderRow(1, 0.1, 0.0, 0.0, True, 0.0, 5.0),
        LadderRow(9, 1.0, 0.0, 0.0, True, 0.0, 1.0),
    ]
    fixed, marginal = cost_model(rows)
    assert fixed >= 0.0 and marginal >= 0.0


# --------------------------------------------------------------------------------------
# The winner's curse.
# --------------------------------------------------------------------------------------


def _pool(validation: list[float], test: list[float]) -> list[Trial]:
    return [
        Trial(_configuration(0.1), 81, validation=v, test=t, seconds=0.0)
        for v, t in zip(validation, test, strict=True)
    ]


def test_selecting_on_noise_raises_the_reported_score_and_not_the_delivered_one() -> None:
    """The curse in its pure form: validation is noise, deployment is constant."""
    rng = np.random.default_rng(6)
    noise = list(rng.normal(size=60))
    rows = _winners_curse(_pool(noise, [0.5] * 60), 400, rng)
    assert rows[-1].reported > rows[0].reported + 0.5
    assert rows[-1].delivered == pytest.approx(0.5)


def test_when_validation_is_the_objective_both_columns_move_together() -> None:
    """The control: a curve that rises for any input would say nothing about selection."""
    rng = np.random.default_rng(7)
    values = list(rng.normal(size=60))
    rows = _winners_curse(_pool(values, values), 400, rng)
    assert rows[-1].reported == pytest.approx(rows[-1].delivered)
    assert rows[-1].optimism == pytest.approx(0.0, abs=1e-9)


def test_the_curve_has_one_row_per_pool_size() -> None:
    rng = np.random.default_rng(8)
    rows = _winners_curse(_pool([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]), 50, rng)
    assert [row.trials for row in rows] == [1, 2, 3]
