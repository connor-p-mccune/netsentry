"""Adversarial-evasion robustness for the supervised detector.

The model card lists "not adversarially robust" as a limitation; this package
*measures* that limitation rather than leaving it asserted — how far an attacker
who shapes the controllable parts of a flow can push the detection rate down.
"""

from __future__ import annotations

from netsentry.robustness.evasion import (
    EvasionStudy,
    controllable_indices,
    mimicry_perturb,
    run_evasion_study,
)

__all__ = [
    "EvasionStudy",
    "controllable_indices",
    "mimicry_perturb",
    "run_evasion_study",
]
