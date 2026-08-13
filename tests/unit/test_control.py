"""The control loop, tested against the control-theory properties it claims.

The valuable tests here are the two textbook ones -- proportional control leaves a steady-state
offset, the integral term removes it -- run against a toy plant simple enough that the right
answer is not in doubt. If those hold, the controller is a controller; if they do not, the rest
of the report is decoration.
"""

from __future__ import annotations

import numpy as np

from netsentry.monitoring.control import (
    LogRatePI,
    ScoreSpaceTracker,
    StaticThreshold,
    oscillation,
    overshoot,
    realised_rates,
    settling_time,
    simulate_loop,
    steady_state_error,
)

# A reference distribution whose quantiles are easy to reason about.
REFERENCE = np.linspace(0.0, 1.0, 10_001)


def _plant(threshold: float, disturbance: float, n: int = 1000) -> float:
    """Toy plant: realised volume is the nominal tail rate scaled by a disturbance."""
    return float((1.0 - threshold) * n * disturbance)


def _run(controller: LogRatePI, setpoint: float, steps: int, *, drift: float = 1.0) -> list[float]:
    """Drive the loop against a disturbance that grows by ``drift`` every batch."""
    threshold = float(np.quantile(REFERENCE, 1.0 - 10.0**controller.log_rate))
    disturbance = 1.0
    volumes = []
    for _ in range(steps):
        volume = _plant(threshold, disturbance)
        volumes.append(volume)
        threshold = controller.update(setpoint, volume, REFERENCE)
        disturbance *= drift
    return volumes


def test_a_static_disturbance_is_rejected_completely() -> None:
    # With the actuator and the error in the same (log) units, a constant multiplicative
    # disturbance has an exact fixed point, and proportional control alone reaches it.
    controller = LogRatePI(kp=0.5, ki=0.0, log_rate=-2.0, max_step=1.0)
    volumes = _run(controller, setpoint=10.0, steps=60)
    assert abs(steady_state_error(np.array(volumes), 10.0)) < 0.01


def test_proportional_control_lags_a_drifting_disturbance() -> None:
    """The case that separates P from PI: a target that keeps moving."""
    controller = LogRatePI(kp=0.4, ki=0.0, log_rate=-2.0, max_step=1.0)
    volumes = _run(controller, setpoint=10.0, steps=60, drift=1.1)
    assert steady_state_error(np.array(volumes), 10.0) > 0.05  # permanently behind


def test_the_integral_term_catches_up_with_the_drift() -> None:
    proportional = LogRatePI(kp=0.4, ki=0.0, log_rate=-2.0, max_step=1.0)
    integral = LogRatePI(kp=0.4, ki=0.4, log_rate=-2.0, max_step=1.0)
    p_error = abs(steady_state_error(np.array(_run(proportional, 10.0, 60, drift=1.1)), 10.0))
    pi_error = abs(steady_state_error(np.array(_run(integral, 10.0, 60, drift=1.1)), 10.0))
    assert pi_error < p_error / 2


def test_the_controller_tightens_when_the_queue_overflows() -> None:
    controller = LogRatePI(kp=0.5, log_rate=-2.0)
    before = controller.log_rate
    controller.update(setpoint=10.0, measured=100.0, reference=REFERENCE)
    assert controller.log_rate < before  # ten times the budget: alert on fewer flows


def test_the_controller_loosens_when_the_queue_is_idle() -> None:
    controller = LogRatePI(kp=0.5, log_rate=-3.0)
    before = controller.log_rate
    controller.update(setpoint=10.0, measured=0.0, reference=REFERENCE)
    assert controller.log_rate > before


def test_the_rate_limit_bounds_actuator_movement() -> None:
    controller = LogRatePI(kp=5.0, log_rate=-2.0, max_step=0.05)
    controller.update(setpoint=10.0, measured=10_000.0, reference=REFERENCE)
    assert np.isclose(controller.log_rate, -2.05)


def test_anti_windup_stops_the_integrator_running_away_at_saturation() -> None:
    """Without this, a surge leaves the loop stuck at its floor long after the surge ends."""
    guarded = LogRatePI(kp=1.0, ki=1.0, log_rate=-4.9, max_step=1.0, anti_windup=True)
    naive = LogRatePI(kp=1.0, ki=1.0, log_rate=-4.9, max_step=1.0, anti_windup=False)
    for _ in range(30):  # a sustained overflow that pins the actuator at its floor
        guarded.update(setpoint=10.0, measured=1000.0, reference=REFERENCE)
        naive.update(setpoint=10.0, measured=1000.0, reference=REFERENCE)
    assert abs(guarded.integral) < abs(naive.integral)
    # Recovery: with the surge over, the guarded loop returns to a usable rate much sooner.
    for _ in range(10):
        guarded.update(setpoint=10.0, measured=0.0, reference=REFERENCE)
        naive.update(setpoint=10.0, measured=0.0, reference=REFERENCE)
    assert guarded.log_rate > naive.log_rate


def test_freeze_above_ignores_an_excursion_that_is_an_incident_not_an_error() -> None:
    controller = LogRatePI(kp=0.5, ki=0.5, log_rate=-2.0, freeze_above=0.5)
    controller.update(setpoint=10.0, measured=10_000.0, reference=REFERENCE)  # 3 decades out
    assert controller.integral == 0.0  # the integrator refused to learn from the surge


def test_static_threshold_ignores_every_measurement() -> None:
    policy = StaticThreshold(log_rate=-2.0)
    first = policy.update(10.0, 0.0, REFERENCE)
    second = policy.update(10.0, 5_000.0, REFERENCE)
    assert first == second


def test_score_space_tracker_moves_toward_the_target_rate() -> None:
    tracker = ScoreSpaceTracker(target_rate=0.02, step=0.5, scale=1.0, threshold=0.5)
    tracker.update(setpoint=10.0, measured=50.0, reference=REFERENCE)  # five times the budget
    assert tracker.threshold > 0.5
    loose = ScoreSpaceTracker(target_rate=0.02, step=0.5, scale=1.0, threshold=0.5)
    loose.update(setpoint=10.0, measured=0.0, reference=REFERENCE)
    assert loose.threshold < 0.5


def test_a_batch_is_never_judged_by_a_threshold_derived_from_itself() -> None:
    """The loop must be causal, or its alert count is a constant by construction."""
    scores = np.concatenate([np.zeros(50), np.ones(50)])
    labels = np.zeros(100, dtype=int)
    trace = simulate_loop(
        scores,
        labels,
        REFERENCE,
        LogRatePI(kp=1.0, log_rate=-1.0, max_step=1.0),
        [(0, 50), (50, 100)],
        setpoint=5.0,
        tolerance=0.5,
        initial_threshold=0.5,
        name="causal",
    )
    assert trace.thresholds[0] == 0.5  # the first batch used the shipped threshold
    assert trace.volumes[0] == 0  # and no alert, because that batch is all zeros


def test_measurement_delay_holds_the_feedback_back() -> None:
    delayed = simulate_loop(
        np.linspace(0, 1, 300),
        np.zeros(300, dtype=int),
        REFERENCE,
        LogRatePI(kp=1.0, log_rate=-1.0, max_step=1.0),
        [(0, 100), (100, 200), (200, 300)],
        setpoint=5.0,
        tolerance=0.5,
        initial_threshold=0.5,
        delay=2,
        name="delayed",
    )
    # With a two-batch delay the first three thresholds cannot yet reflect any measurement.
    assert delayed.thresholds[0] == delayed.thresholds[1] == delayed.thresholds[2]


def test_overshoot_and_steady_state_are_read_off_the_right_places() -> None:
    volumes = np.array([30.0, 20.0, 12.0, 10.0, 10.0])
    assert np.isclose(overshoot(volumes, 10.0), 2.0)  # peak of 30 is 200% above the setpoint
    assert np.isclose(steady_state_error(volumes, 10.0, tail=2), 0.0)


def test_settling_uses_the_last_exit_not_the_first_entry() -> None:
    # In the band at index 1, out again at 3: an oscillating loop has not settled at 1.
    volumes = np.array([50.0, 10.0, 10.0, 40.0, 10.0])
    assert settling_time(volumes, 10.0, tolerance=0.25) == 4


def test_oscillation_is_mean_absolute_movement() -> None:
    assert np.isclose(oscillation(np.array([1.0, 2.0, 1.0])), 1.0)
    assert oscillation(np.array([5.0])) == 0.0


def test_realised_rates_translate_thresholds_into_operator_units() -> None:
    rates = realised_rates(np.array([0.5, 0.99]), REFERENCE)
    assert np.isclose(rates[0], 0.5, atol=0.01)
    assert np.isclose(rates[1], 0.01, atol=0.01)
