"""The batching queue, checked against arithmetic rather than against itself.

A discrete-event simulator is easy to write and easy to write wrongly: an off-by-one in the
clock produces latencies that look plausible, scale plausibly, and are wrong. So the tests pin
it to quantities that can be derived by hand — the service time at zero load, conservation of
requests, the equilibrium batch size, and the capacity ceiling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.serving.batching import (
    ServicePoint,
    equilibrium_batch_size,
    fit_affine,
    measure_service_curve,
    simulate_queue,
    theoretical_mean_latency,
)

FIXED = 0.010  # 10 ms of fixed cost per call
MARGINAL = 0.00002  # 20 microseconds per extra flow


def _simulate(rate: float, max_batch: int, max_wait: float, n: int = 6000) -> tuple:
    return simulate_queue(
        arrival_rate=rate,
        n_requests=n,
        fixed=FIXED,
        marginal=MARGINAL,
        max_batch=max_batch,
        max_wait=max_wait,
        rng=np.random.default_rng(0),
    )


# --------------------------------------------------------------------------------------
# The service curve.
# --------------------------------------------------------------------------------------


def test_fit_affine_recovers_known_coefficients() -> None:
    points = [ServicePoint(size, 0.004 + 0.00003 * size) for size in (1, 2, 8, 64, 256)]
    fixed, marginal = fit_affine(points)
    assert fixed == pytest.approx(0.004, rel=1e-6)
    assert marginal == pytest.approx(0.00003, rel=1e-6)


def test_per_flow_cost_falls_with_batch_size_under_a_fixed_cost() -> None:
    single = ServicePoint(1, 0.010)
    hundred = ServicePoint(100, 0.012)
    assert single.per_flow_ms == pytest.approx(10.0)
    assert hundred.per_flow_ms == pytest.approx(0.12)


def test_measure_service_curve_times_every_requested_size() -> None:
    frame = pd.DataFrame({"a": np.arange(50.0)})
    points = measure_service_curve(lambda f: np.asarray(f), frame, [1, 5, 20], repeats=3)
    assert [p.batch_size for p in points] == [1, 5, 20]
    assert all(p.seconds >= 0.0 for p in points)


# --------------------------------------------------------------------------------------
# The simulator.
# --------------------------------------------------------------------------------------


def test_every_request_is_served_exactly_once() -> None:
    latencies, throughput, mean_batch = _simulate(500.0, 32, 0.002)
    assert len(latencies) == 6000
    assert (latencies > 0).all()
    assert throughput > 0 and mean_batch >= 1.0


def test_no_request_finishes_faster_than_the_service_time() -> None:
    # The floor is the service time of its own batch: nothing can be answered before the call
    # that answers it has run.
    latencies, _, mean_batch = _simulate(100.0, 16, 0.0)
    assert float(latencies.min()) >= FIXED + MARGINAL - 1e-9
    assert float(latencies.min()) < FIXED + MARGINAL * mean_batch + 0.01


def test_at_vanishing_load_latency_is_just_the_service_time() -> None:
    # One request at a time with nobody else in the system: latency must equal the service
    # time exactly, which is the simplest possible check on the clock arithmetic.
    latencies, _, _ = _simulate(0.5, 1, 0.0, n=200)
    assert float(np.median(latencies)) == pytest.approx(FIXED + MARGINAL, abs=1e-6)


def test_the_waiting_timer_bounds_the_extra_delay() -> None:
    # An idle server with a 3 ms timer adds at most 3 ms before giving up and serving alone.
    wait = 0.003
    latencies, _, _ = _simulate(1.0, 32, wait, n=300)
    assert float(np.median(latencies)) == pytest.approx(FIXED + MARGINAL + wait, abs=5e-4)


def test_an_unbatched_server_saturates_where_arithmetic_says_it_must() -> None:
    capacity = 1.0 / (FIXED + MARGINAL)
    _, throughput, _ = _simulate(4 * capacity, 1, 0.0)
    assert throughput == pytest.approx(capacity, rel=0.02)  # it cannot exceed its own rate


def test_batching_raises_the_ceiling_the_marginal_cost_implies() -> None:
    rate = 2000.0  # far past the unbatched capacity of ~100/s
    _, plain_throughput, _ = _simulate(rate, 1, 0.0)
    _, batched_throughput, _ = _simulate(rate, 256, 0.002)
    assert plain_throughput < 110.0
    assert batched_throughput > 0.95 * rate


# --------------------------------------------------------------------------------------
# The closed form.
# --------------------------------------------------------------------------------------


def test_the_equilibrium_batch_size_predicts_the_simulated_one() -> None:
    for rate in (200.0, 800.0, 2000.0, 5000.0):
        _, _, mean_batch = _simulate(rate, 4096, 0.0, n=20000)
        predicted = equilibrium_batch_size(rate, FIXED, MARGINAL)
        assert mean_batch == pytest.approx(predicted, rel=0.1)


def test_the_predicted_latency_tracks_the_simulated_one() -> None:
    for rate in (200.0, 2000.0):
        latencies, _, _ = _simulate(rate, 4096, 0.0, n=20000)
        predicted = theoretical_mean_latency(rate, FIXED, MARGINAL)
        assert float(np.mean(latencies)) == pytest.approx(predicted, rel=0.15)


def test_capacity_is_the_reciprocal_of_the_marginal_cost() -> None:
    # Past 1 / marginal the per-flow work alone outruns the server, and the model says so by
    # returning an infinite batch size rather than a large finite one.
    capacity = 1.0 / MARGINAL
    assert np.isfinite(equilibrium_batch_size(0.5 * capacity, FIXED, MARGINAL))
    assert not np.isfinite(equilibrium_batch_size(capacity, FIXED, MARGINAL))
    assert theoretical_mean_latency(2 * capacity, FIXED, MARGINAL) == float("inf")


def test_the_equilibrium_batch_never_drops_below_one() -> None:
    assert equilibrium_batch_size(1e-6, FIXED, MARGINAL) == 1.0
