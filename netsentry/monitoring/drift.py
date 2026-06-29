"""Population Stability Index (PSI) drift metrics.

PSI measures how far a feature's distribution has moved from a reference
(training) distribution to a current (production) one. Standard reading:
``< 0.1`` no real shift, ``0.1-0.25`` moderate, ``>= 0.25`` major. Watching PSI on
inputs and on the model's output score is how model decay is caught in
production *before* the labels — and the damage — arrive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
_EPS = 1e-6  # proportion floor so an empty bin can't send the log to -inf


def quantile_bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile bin edges from the reference, with open (+/-inf) outer edges.

    Reference quantiles keep each bin roughly equally populated (the standard PSI
    construction); open outer edges guarantee every current value lands in a bin.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))).astype(float)
    if edges.size < 2:  # constant feature
        edges = np.array([edges[0], edges[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of finite ``values`` falling in each bin defined by ``edges``."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    counts, _ = np.histogram(finite, bins=edges)
    n_bins = len(edges) - 1
    total = int(counts.sum())
    if total == 0:
        return np.full(n_bins, 1.0 / n_bins)
    return counts / total


def psi_from_proportions(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    """PSI between two proportion vectors that share the same binning."""
    ref = np.clip(ref_props, _EPS, None)
    cur = np.clip(cur_props, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """PSI of ``current`` against ``reference`` for a single feature."""
    edges = quantile_bin_edges(reference, bins)
    return psi_from_proportions(bin_proportions(reference, edges), bin_proportions(current, edges))


def classify_psi(psi: float, *, moderate: float = PSI_MODERATE, major: float = PSI_MAJOR) -> str:
    """Bucket a PSI value as ``none`` / ``moderate`` / ``major``."""
    if psi >= major:
        return "major"
    if psi >= moderate:
        return "moderate"
    return "none"


@dataclass
class DriftReport:
    """Per-feature (and optional model-score) PSI with severity helpers."""

    feature_psi: dict[str, float]
    score_psi: float | None = None
    moderate: float = PSI_MODERATE
    major: float = PSI_MAJOR

    @property
    def max_psi(self) -> float:
        return max(self.feature_psi.values(), default=0.0)

    @property
    def mean_psi(self) -> float:
        values = list(self.feature_psi.values())
        return float(np.mean(values)) if values else 0.0

    def drifted(self, *, level: str = "moderate") -> list[str]:
        """Features at or above a severity level, worst first."""
        cutoff = self.major if level == "major" else self.moderate
        ranked = sorted(self.feature_psi.items(), key=lambda kv: kv[1], reverse=True)
        return [feature for feature, psi in ranked if psi >= cutoff]

    def classify(self, psi: float) -> str:
        return classify_psi(psi, moderate=self.moderate, major=self.major)


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], *, bins: int = 10
) -> dict[str, float]:
    """PSI per feature, for the columns present in both frames."""
    out: dict[str, float] = {}
    for col in columns:
        if col in reference.columns and col in current.columns:
            out[col] = population_stability_index(
                reference[col].to_numpy(), current[col].to_numpy(), bins=bins
            )
    return out


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    bins: int = 10,
    moderate: float = PSI_MODERATE,
    major: float = PSI_MAJOR,
    score_reference: np.ndarray | None = None,
    score_current: np.ndarray | None = None,
) -> DriftReport:
    """Build a :class:`DriftReport` for feature drift, plus optional score drift."""
    feature_psi = feature_drift(reference, current, columns, bins=bins)
    score_psi: float | None = None
    if score_reference is not None and score_current is not None:
        score_psi = population_stability_index(score_reference, score_current, bins=bins)
    return DriftReport(feature_psi=feature_psi, score_psi=score_psi, moderate=moderate, major=major)
