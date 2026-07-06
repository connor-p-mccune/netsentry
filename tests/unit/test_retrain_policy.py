"""Retrain-trigger logic: the pure decision rules behind the policy study."""

from __future__ import annotations

from netsentry.monitoring.retrain_policy import (
    AlwaysTrigger,
    DriftTrigger,
    NeverTrigger,
    PeriodicTrigger,
)


def test_never_and_always_are_the_floor_and_ceiling() -> None:
    never, always = NeverTrigger(), AlwaysTrigger()
    for i, psi in enumerate([0.0, 0.5, 10.0]):
        assert not never.should_retrain(i, psi)
        assert always.should_retrain(i, psi)


def test_periodic_fires_on_the_calendar_not_the_signal() -> None:
    trigger = PeriodicTrigger(every=3)
    fired = [i for i in range(9) if trigger.should_retrain(i, score_psi=0.0)]
    assert fired == [2, 5, 8]  # after every 3rd batch, regardless of drift


def test_drift_trigger_respects_threshold() -> None:
    trigger = DriftTrigger(threshold=0.25, cooldown=1)
    assert not trigger.should_retrain(0, score_psi=0.24)  # below the line: hold
    assert trigger.should_retrain(1, score_psi=0.25)  # at the line: fire


def test_drift_trigger_cooldown_suppresses_refires() -> None:
    trigger = DriftTrigger(threshold=0.25, cooldown=3)
    assert trigger.should_retrain(0, score_psi=0.9)
    # Alarms inside the cooldown window are suppressed even though PSI stays high...
    assert not trigger.should_retrain(1, score_psi=0.9)
    assert not trigger.should_retrain(2, score_psi=0.9)
    # ...and the trigger re-arms once the window has passed.
    assert trigger.should_retrain(3, score_psi=0.9)


def test_cooldown_counts_from_the_last_fire_not_the_last_alarm() -> None:
    trigger = DriftTrigger(threshold=0.25, cooldown=2)
    assert trigger.should_retrain(0, score_psi=0.9)
    assert not trigger.should_retrain(1, score_psi=0.9)  # suppressed, must not reset clock
    assert trigger.should_retrain(2, score_psi=0.9)  # 2 batches after the fire at 0
