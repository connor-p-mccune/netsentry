"""The game solved on hand-built payoff matrices, where the right answer is known by inspection.

Solution concepts are exactly the kind of code that looks right and is off by one: a Stackelberg
solver that forgets the follower re-optimises collapses to argmax, and a Nash check that only
tests one player's incentive accepts cells nobody would sit in. Every matrix here is small enough
to solve on paper first.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.robustness.strategic import (
    arms_race,
    attacker_utility,
    best_response,
    cycle_length,
    pure_nash,
    stackelberg_solution,
)

FRACTIONS = np.array([0.0, 0.5, 1.0])


def test_attacker_gains_nothing_from_total_mimicry() -> None:
    # A flow indistinguishable from benign traffic is not carrying out an attack, whatever the
    # detector does. This term is what stops the best reply from being "mimic completely".
    assert attacker_utility(detection=0.0, fraction=1.0, effectiveness_exponent=1.0) == 0.0


def test_attacker_gains_nothing_from_a_certainly_detected_attack() -> None:
    assert attacker_utility(detection=1.0, fraction=0.0, effectiveness_exponent=1.0) == 0.0


def test_attacker_utility_trades_evasion_against_effectiveness() -> None:
    # Half-disguised and half-detected is worth (1-0.5)*(1-0.5).
    assert attacker_utility(0.5, 0.5, 1.0) == pytest.approx(0.25)


def test_attacker_utility_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="fraction"):
        attacker_utility(0.5, 1.5, 1.0)


def test_best_response_prefers_partial_disguise_over_total() -> None:
    # Detection drops all the way to zero under full mimicry, but so does the attack's value.
    detections = np.array([0.9, 0.4, 0.0])
    idx, utility = best_response(detections, FRACTIONS, 1.0)
    assert idx == 1  # (1-0.4)*(1-0.5) = 0.30 beats 0.10 and 0.00
    assert utility == pytest.approx(0.30)


def test_best_response_breaks_ties_toward_the_cheaper_disguise() -> None:
    # Two options with identical utility: the attacker takes the one that costs less.
    detections = np.array([0.5, 0.0, 0.0])
    idx, _ = best_response(detections, np.array([0.0, 0.5, 0.75]), 1.0)
    assert idx == 0


def test_stackelberg_beats_the_row_that_is_best_against_todays_attack() -> None:
    # Row 0 is superb against the undisguised attack and collapses once the attacker adapts.
    # Row 1 is mediocre everywhere, and therefore the right commitment: against a defence with
    # no soft spot the disguise costs the attacker more than it buys, so they decline to use it
    # and detection stays at the *undisguised* cell rather than at row 1's worst entry.
    payoff = np.array(
        [
            [0.95, 0.05, 0.0],  # brittle specialist
            [0.60, 0.55, 0.0],  # flat generalist
        ]
    )
    naive = int(np.argmax(payoff[:, 0]))
    row, col, detection = stackelberg_solution(payoff, FRACTIONS, 1.0)
    assert naive == 0  # what a myopic defender would deploy
    assert row == 1  # what a defender who thinks one move ahead deploys
    assert col == 0  # and the attacker's reply is to stop disguising
    assert detection == pytest.approx(0.60)
    assert detection > payoff[0, best_response(payoff[0], FRACTIONS, 1.0)[0]]


def test_stackelberg_returns_the_cell_the_attacker_actually_plays() -> None:
    payoff = np.array([[0.8, 0.2, 0.0]])
    _, col, detection = stackelberg_solution(payoff, FRACTIONS, 1.0)
    assert col == best_response(payoff[0], FRACTIONS, 1.0)[0]
    assert detection == payoff[0, col]


def test_pure_nash_finds_the_cell_neither_side_leaves() -> None:
    # Solved by hand with fractions [0, 0.5]. Row 0: utilities 0.10 and 0.35, so the attacker
    # replies with column 1. Column 1: detections 0.3 and 0.2, so the defender replies with
    # row 0. Both conditions hold at (0, 1) and nowhere else.
    payoff = np.array([[0.9, 0.3], [0.4, 0.2]])
    fractions = np.array([0.0, 0.5])
    assert pure_nash(payoff, fractions, 1.0) == [(0, 1)]


def test_pure_nash_requires_both_players_to_be_content_not_just_one() -> None:
    # (1, 0) is the defender's best column reply but not the attacker's row reply, so a check
    # that tested only one side would wrongly accept it.
    payoff = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]])
    assert pure_nash(payoff, FRACTIONS, 1.0) == []


def test_a_rock_paper_scissors_matrix_has_no_equilibrium_and_never_settles() -> None:
    # Each defence is countered by one attack, and each attack is answered by a different
    # defence. With no cost to disguise, the attacker simply minimises detection.
    payoff = np.array(
        [
            [0.9, 0.1, 0.5],
            [0.5, 0.9, 0.1],
            [0.1, 0.5, 0.9],
        ]
    )
    free = np.array([0.0, 0.0, 0.0])  # disguise costs the attacker nothing here
    assert pure_nash(payoff, free, 1.0) == []
    trajectory = arms_race(payoff, free, 1.0, rounds=9)
    assert cycle_length(trajectory) == 3  # the defender chases the attack around the cycle
    assert len({d for d, _, _ in trajectory}) == 3


def test_arms_race_converges_and_reports_a_unit_cycle() -> None:
    payoff = np.array([[0.6, 0.5, 0.0]])  # one defence: nothing to switch to
    trajectory = arms_race(payoff, FRACTIONS, 1.0, rounds=4)
    assert cycle_length(trajectory) == 1
    assert {(d, a) for d, a, _ in trajectory} == {(0, trajectory[0][1])}


def test_arms_race_cycles_when_each_defence_counters_the_other() -> None:
    # Defence 0 is strong against attack 1 and weak against attack 0; defence 1 the reverse.
    # A myopic defender chases the last attack forever.
    payoff = np.array(
        [
            [0.10, 0.90, 0.0],
            [0.90, 0.10, 0.0],
        ]
    )
    trajectory = arms_race(payoff, FRACTIONS, 1.0, rounds=8)
    assert cycle_length(trajectory) == 2
    assert len({d for d, _, _ in trajectory}) == 2  # the defender keeps swapping


def test_cycle_length_of_an_empty_trajectory_is_zero() -> None:
    assert cycle_length([]) == 0
