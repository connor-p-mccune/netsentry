"""Serve-time sensor faults: the fault injectors and the marginal-blindness of PSI.

The report's central claim — that a mis-assembly fault is invisible to a marginal drift
monitor — is a mathematical property of PSI, not an empirical accident, so it is pinned
here directly. The rest guards the fault injectors: each mode must break what it says it
breaks and leave the rest of the frame untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.robustness.degradation import FaultOutcome, apply_fault, psi_of_fault


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Flow Duration": rng.lognormal(5, 1, size=400),
            "Total Fwd Packets": rng.integers(1, 50, size=400).astype(float),
            "Untouched": np.arange(400, dtype=float),
        }
    )


def test_missing_fault_nulls_only_the_named_columns(frame: pd.DataFrame) -> None:
    out = apply_fault(frame, ["Flow Duration"], "missing", np.random.default_rng(0))
    assert out["Flow Duration"].isna().all()
    assert out["Untouched"].equals(frame["Untouched"])
    assert not out["Total Fwd Packets"].isna().any()


def test_stuck_fault_freezes_the_column_at_zero(frame: pd.DataFrame) -> None:
    out = apply_fault(frame, ["Total Fwd Packets"], "stuck", np.random.default_rng(0))
    assert (out["Total Fwd Packets"] == 0.0).all()
    assert out["Flow Duration"].equals(frame["Flow Duration"])


def test_shuffled_fault_preserves_the_multiset_of_values(frame: pd.DataFrame) -> None:
    out = apply_fault(frame, ["Flow Duration"], "shuffled", np.random.default_rng(1))
    # Same values, different rows: the marginal survives exactly, the joint does not.
    assert sorted(out["Flow Duration"].tolist()) == sorted(frame["Flow Duration"].tolist())
    assert not out["Flow Duration"].equals(frame["Flow Duration"])


def test_shuffled_fault_uses_one_permutation_across_the_group(frame: pd.DataFrame) -> None:
    cols = ["Flow Duration", "Total Fwd Packets"]
    out = apply_fault(frame, cols, "shuffled", np.random.default_rng(2))
    # A single mis-assembly moves the group's fields together, so the pairing *within*
    # the group is preserved even though its pairing with the rest of the row is not.
    pairs = set(zip(frame["Flow Duration"], frame["Total Fwd Packets"], strict=True))
    assert set(zip(out["Flow Duration"], out["Total Fwd Packets"], strict=True)) == pairs


def test_unknown_fault_mode_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown fault mode"):
        apply_fault(frame, ["Flow Duration"], "corrupted", np.random.default_rng(0))


def test_shuffle_is_invisible_to_psi_because_psi_is_marginal(frame: pd.DataFrame) -> None:
    # The report's headline claim. Permuting rows leaves every bin count identical, so a
    # per-feature marginal statistic sees exactly nothing.
    shuffled = apply_fault(frame, ["Flow Duration"], "shuffled", np.random.default_rng(3))
    assert psi_of_fault(frame, shuffled, ["Flow Duration"], bins=10) == pytest.approx(0.0)


def test_stuck_fault_is_loudly_visible_to_psi(frame: pd.DataFrame) -> None:
    stuck = apply_fault(frame, ["Flow Duration"], "stuck", np.random.default_rng(0))
    assert psi_of_fault(frame, stuck, ["Flow Duration"], bins=10) > 1.0


def test_all_null_feature_scores_as_infinitely_drifted(frame: pd.DataFrame) -> None:
    # PSI is undefined with no observations; an ingestion failure is not a subtle shift.
    missing = apply_fault(frame, ["Flow Duration"], "missing", np.random.default_rng(0))
    assert psi_of_fault(frame, missing, ["Flow Duration"], bins=10) == float("inf")


def test_psi_of_fault_ignores_columns_absent_from_either_frame(frame: pd.DataFrame) -> None:
    assert psi_of_fault(frame, frame, ["Not A Column"], bins=10) == 0.0


def _outcome(psi: float, level: str, baseline: float, delta_level: str) -> FaultOutcome:
    return FaultOutcome(
        group="timing",
        mode="shuffled",
        n_features=3,
        pr_auc=0.2,
        tpr=0.05,
        fpr=0.001,
        precision=0.5,
        alerts_per_day=900,
        psi=psi,
        psi_level=level,
        baseline_psi=baseline,
        psi_delta_level=delta_level,
    )


def test_outcome_flags_a_quiet_monitor() -> None:
    assert not _outcome(0.001, "none", 0.0, "none").detected_by_monitor
    assert _outcome(1.4, "major", 0.0, "major").detected_by_monitor


def test_pre_existing_drift_is_not_credited_to_the_fault() -> None:
    # The healthy temporal test set already sits at PSI 0.12 (moderate). A fault that leaves
    # it there added nothing, so the monitor gets no credit for the alarm it was already
    # raising — otherwise every fault on an already-drifted family would read as "detected".
    already_drifted = _outcome(psi=0.12, level="moderate", baseline=0.12, delta_level="none")
    assert not already_drifted.detected_by_monitor
