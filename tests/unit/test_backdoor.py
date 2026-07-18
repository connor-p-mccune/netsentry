"""Backdoor poisoning: the trigger mechanics, the ASR metric, and the spectral detector.

The report's alarming claim — a tiny poison budget backdoors the model while clean metrics
stay green, and a blind spectral audit removes it — rests on three pure pieces: the trigger
is stamped and mislabeled correctly, attack success is measured over the right denominator,
and the spectral-signature score actually concentrates on a planted cluster. Each is pinned
here without needing a fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.robustness.backdoor import (
    attack_success_rate,
    poison_training_set,
    spectral_signature_scores,
    stamp_trigger,
)

TRIGGER = {"Init_Win_bytes_forward": 4242.0, "Fwd IAT Min": 4242.0}


def _frame(n_attack: int = 40, n_benign: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_attack + n_benign
    labels = np.array([1] * n_attack + [0] * n_benign)
    return pd.DataFrame(
        {
            "Init_Win_bytes_forward": rng.integers(0, 1000, n).astype(float),
            "Fwd IAT Min": rng.integers(0, 1000, n).astype(float),
            "Flow Duration": rng.integers(0, 5000, n).astype(float),
            BINARY_TARGET: labels,
            MULTICLASS_TARGET: np.where(labels == 1, "DoS Hulk", "BENIGN"),
        }
    )


def test_stamp_trigger_writes_exact_values_and_leaves_others() -> None:
    df = _frame()
    stamped = stamp_trigger(df, TRIGGER)
    assert (stamped["Init_Win_bytes_forward"] == 4242.0).all()
    assert (stamped["Fwd IAT Min"] == 4242.0).all()
    # A non-trigger feature and the original frame are untouched.
    assert stamped["Flow Duration"].equals(df["Flow Duration"])
    assert not (df["Init_Win_bytes_forward"] == 4242.0).all()


def test_stamp_trigger_ignores_absent_features() -> None:
    df = _frame()
    stamped = stamp_trigger(df, {"Nonexistent Feature": 1.0})
    assert stamped.equals(df)  # nothing to write, nothing changed


def test_poison_injects_triggered_benign_labeled_attack_rows() -> None:
    df = _frame(n_attack=40, n_benign=60)
    poisoned, mask = poison_training_set(df, TRIGGER, rate=0.1, benign_label="BENIGN", seed=1)
    n_inject = int(len(df) * 0.1)
    assert mask.sum() == n_inject
    injected = poisoned.loc[mask]
    # Injected rows wear the trigger and are labeled benign on BOTH targets...
    assert (injected["Init_Win_bytes_forward"] == 4242.0).all()
    assert (injected[BINARY_TARGET] == 0).all()
    assert (injected[MULTICLASS_TARGET] == "BENIGN").all()
    # ...and the original rows are preserved untouched at the front.
    assert poisoned.iloc[: len(df)][BINARY_TARGET].tolist() == df[BINARY_TARGET].tolist()


def test_poison_zero_rate_is_a_noop() -> None:
    df = _frame()
    poisoned, mask = poison_training_set(df, TRIGGER, rate=0.0, benign_label="BENIGN", seed=1)
    assert mask.sum() == 0 and len(poisoned) == len(df)


def test_attack_success_rate_uses_the_detained_denominator() -> None:
    # Five triggered scores; the benign threshold is 0.5. Three are below it (escaped).
    trig = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
    # Detained = the three attacks the clean model caught (indices 0, 2, 4). Of those,
    # index 0 escapes (0.1 < 0.5), index 2 does not (0.9), index 4 does not (0.8): 1/3.
    detained = np.array([True, False, True, False, True])
    assert attack_success_rate(trig, 0.5, detained) == 1 / 3
    # Without the mask it is measured over all five (indices 0, 1, 3 escape): 3/5.
    assert attack_success_rate(trig, 0.5) == 3 / 5


def test_attack_success_rate_empty_detained_is_nan() -> None:
    trig = np.array([0.1, 0.2])
    assert np.isnan(attack_success_rate(trig, 0.5, np.array([False, False])))


def test_spectral_signature_flags_a_planted_cluster() -> None:
    # A benign blob plus a tight poisoned cluster shoved along one direction. The squared
    # projection on the top singular vector must rank the cluster in its tail.
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 1, size=(400, 8))
    poison = rng.normal(0, 0.05, size=(20, 8)) + np.array([6.0, 0, 0, 0, 0, 0, 0, 0])
    rep = np.vstack([clean, poison])
    scores = spectral_signature_scores(rep)
    order = np.argsort(-scores)
    # The 20 poisoned rows (indices 400-419) should dominate the top of the ranking.
    top20 = set(order[:20].tolist())
    caught = len(top20 & set(range(400, 420)))
    assert caught >= 18  # near-perfect separation on a clean planted cluster


def test_spectral_signature_empty_is_safe() -> None:
    assert spectral_signature_scores(np.empty((0, 5))).shape == (0,)
