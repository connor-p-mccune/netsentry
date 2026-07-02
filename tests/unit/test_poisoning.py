"""Poisoning-primitive tests: label flips corrupt both targets at the right rate
without touching clean rows, and benign-pool contamination injects the right count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.robustness.poisoning import contaminate_benign_pool, flip_attack_labels


def _labeled_frame(n_attack: int = 40, n_benign: int = 60) -> pd.DataFrame:
    binary = [1] * n_attack + [0] * n_benign
    multiclass = ["DoS Hulk"] * n_attack + ["BENIGN"] * n_benign
    return pd.DataFrame(
        {
            BINARY_TARGET: binary,
            MULTICLASS_TARGET: multiclass,
            "Flow Duration": np.arange(n_attack + n_benign, dtype=float),
        }
    )


def test_flip_rate_zero_is_identity() -> None:
    df = _labeled_frame()
    out, n = flip_attack_labels(df, 0.0, "BENIGN", seed=1)
    assert n == 0
    pd.testing.assert_frame_equal(out, df)


def test_flip_relabels_both_targets_and_only_attacks() -> None:
    df = _labeled_frame(n_attack=40, n_benign=60)
    out, n = flip_attack_labels(df, 0.25, "BENIGN", seed=7)
    assert n == 10  # 25% of 40 attack rows
    # 10 attacks became benign in both targets; the 60 original benign are untouched.
    assert int((out[BINARY_TARGET] == 1).sum()) == 30
    assert int((out[MULTICLASS_TARGET] == "BENIGN").sum()) == 70
    # Consistency: every binary-benign row is multiclass-BENIGN.
    assert ((out[BINARY_TARGET] == 0) == (out[MULTICLASS_TARGET] == "BENIGN")).all()


def test_flip_does_not_mutate_input() -> None:
    df = _labeled_frame()
    before = df[BINARY_TARGET].sum()
    flip_attack_labels(df, 0.5, "BENIGN", seed=3)
    assert df[BINARY_TARGET].sum() == before


def test_flip_is_seed_deterministic() -> None:
    df = _labeled_frame()
    a, _ = flip_attack_labels(df, 0.3, "BENIGN", seed=11)
    b, _ = flip_attack_labels(df, 0.3, "BENIGN", seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_contamination_injects_expected_count() -> None:
    df = _labeled_frame(n_attack=40, n_benign=60)
    benign = df[df[BINARY_TARGET] == 0]
    attacks = df[df[BINARY_TARGET] == 1]
    out, n = contaminate_benign_pool(benign, attacks, 0.1, seed=5)
    assert n == 6  # 10% of 60 benign rows
    assert len(out) == len(benign) + 6


def test_contamination_rate_zero_is_identity() -> None:
    df = _labeled_frame()
    benign = df[df[BINARY_TARGET] == 0]
    attacks = df[df[BINARY_TARGET] == 1]
    out, n = contaminate_benign_pool(benign, attacks, 0.0, seed=5)
    assert n == 0 and len(out) == len(benign)


def test_contamination_handles_no_attacks() -> None:
    df = _labeled_frame(n_attack=0, n_benign=50)
    benign = df[df[BINARY_TARGET] == 0]
    empty_attacks = df[df[BINARY_TARGET] == 1]
    out, n = contaminate_benign_pool(benign, empty_attacks, 0.2, seed=5)
    assert n == 0 and len(out) == len(benign)
