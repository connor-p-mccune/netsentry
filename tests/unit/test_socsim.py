"""SOC queue simulation: the event-driven core, hand-checked against known shifts."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config.settings import SocSimConfig
from netsentry.evaluation.socsim import _summarize_run, run_shift, simulate_queue


def test_single_server_fifo_serializes_waiting_tickets() -> None:
    # Two tickets, one server, deterministic service. Ticket 0 grabs the free
    # server at t=0; ticket 1 arrives at t=1 and waits until t=5.
    result = simulate_queue(
        arrivals=np.array([0.0, 1.0]),
        services=np.array([5.0, 5.0]),
        scores=np.array([0.5, 0.5]),
        n_servers=1,
        discipline="fifo",
    )
    assert list(result.start) == [0.0, 5.0]
    assert list(result.wait) == [0.0, 4.0]


def test_two_servers_run_in_parallel_no_wait() -> None:
    # Two tickets arriving together, two servers: both start immediately.
    result = simulate_queue(
        arrivals=np.array([0.0, 0.0]),
        services=np.array([5.0, 5.0]),
        scores=np.array([0.1, 0.9]),
        n_servers=2,
        discipline="fifo",
    )
    assert list(result.wait) == [0.0, 0.0]


def test_priority_lets_a_late_attack_jump_the_benign_backlog() -> None:
    # One server, three tickets arriving at 0/1/2, each 10 min. The attack (index
    # 2, top score) arrives last into a backlog. FIFO serves it third — it starts
    # at t=20 (after 0..10 and 10..20), so it waits 18. Priority serves it second,
    # right after the in-service ticket frees at t=10, so it waits 8.
    arrivals = np.array([0.0, 1.0, 2.0])
    services = np.array([10.0, 10.0, 10.0])
    scores = np.array([0.10, 0.20, 0.90])
    fifo = simulate_queue(arrivals, services, scores, n_servers=1, discipline="fifo")
    prio = simulate_queue(arrivals, services, scores, n_servers=1, discipline="priority")
    assert fifo.wait[2] == pytest.approx(18.0)  # served last: starts at t=20
    assert prio.wait[2] == pytest.approx(8.0)  # served second: starts at t=10
    assert prio.wait[2] < fifo.wait[2]
    # The ticket already in service (index 0) is never bumped under either policy.
    assert fifo.start[0] == prio.start[0] == 0.0


def test_every_ticket_is_eventually_served() -> None:
    rng = np.random.default_rng(0)
    n = 200
    arrivals = rng.uniform(0, 480, size=n)
    services = rng.exponential(8.0, size=n)
    scores = rng.uniform(size=n)
    result = simulate_queue(arrivals, services, scores, n_servers=3, discipline="priority")
    assert np.isfinite(result.start).all()
    assert (result.wait >= -1e-9).all()  # no ticket starts before it arrives


def test_summarize_run_attack_sla_and_backlog() -> None:
    # Three attack alerts: one served fast (in SLA), one served slow (out of SLA
    # but within the shift), one that starts after the horizon (backlog).
    from netsentry.evaluation.socsim import QueueResult

    result = QueueResult(
        start=np.array([5.0, 100.0, 500.0]),
        wait=np.array([5.0, 100.0, 480.0]),
    )
    summary = _summarize_run(
        result,
        is_attack=np.array([1, 1, 1]),
        services=np.array([10.0, 10.0, 10.0]),
        horizon=480.0,
        sla=30.0,
        n_servers=1,
    )
    assert summary["attack_sla"] == pytest.approx(1 / 3)  # only the first is in-SLA
    assert summary["attack_backlog"] == pytest.approx(1 / 3)  # only the third overflows


def test_run_shift_is_seeded_and_reports_load() -> None:
    cfg = SocSimConfig(n_runs=5, arrivals_per_shift=100)
    scores = np.concatenate([np.full(90, 0.3), np.full(10, 0.95)])
    is_attack = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
    first = run_shift(scores, is_attack, cfg, n_servers=2, discipline="fifo", seed=7)
    second = run_shift(scores, is_attack, cfg, n_servers=2, discipline="fifo", seed=7)
    assert first.attack_sla == second.attack_sla  # seeded: identical medians
    assert first.rho > 0  # offered load is reported


def test_priority_never_hurts_attack_sla_under_load() -> None:
    # A saturated single-server shift: risk-ordering the queue can only help the
    # attacks reach an analyst sooner, never hurt them.
    cfg = SocSimConfig(n_runs=8, arrivals_per_shift=120, minutes_per_alert_mean=8.0)
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 1.0, size=120)
    is_attack = (scores > 0.8).astype(int)
    fifo = run_shift(scores, is_attack, cfg, n_servers=1, discipline="fifo", seed=1)
    prio = run_shift(scores, is_attack, cfg, n_servers=1, discipline="priority", seed=1)
    assert prio.attack_sla >= fifo.attack_sla - 1e-9
