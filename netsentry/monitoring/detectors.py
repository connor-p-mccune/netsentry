"""Statistical and online concept-drift detectors — the significance PSI omits.

The PSI drift report (``monitoring/drift.py``) answers *how much* a distribution
moved. It is an effect size with a rule-of-thumb cutoff, not a test, and it is
computed on static batches. Production drift monitoring wants two more things,
supplied here as small, dependency-light, textbook detectors:

- **Kolmogorov-Smirnov two-sample test** per feature, with **Benjamini-Hochberg**
  false-discovery-rate control across features — so "N features drifted" is a
  multiplicity-corrected count with p-values, not a threshold on an effect size.
- **Page-Hinkley** and **DDM** (Gama et al., 2004): *online* detectors that consume
  a stream and report *when* it changed — the change-point PSI-on-batches cannot
  give. Page-Hinkley watches the model-score stream for a sustained mean increase;
  DDM watches the model-error stream and raises a warning then a drift alarm as the
  error rate climbs statistically above its running minimum.

All functions are pure (arrays in, results out) so they unit-test cleanly against
planted shifts; the reporting layer lives in ``monitoring/report.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR procedure.

    Returns a boolean mask of the hypotheses declared significant at false-discovery
    rate ``alpha`` and the critical p-value (0.0 when nothing is significant). BH
    controls the expected share of false positives among rejections — the right knob
    when testing many features at once, where an uncorrected 5% per-test rate would
    flag ~5% of *stable* features as drifted by chance.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool), 0.0
    crit = float(ranked[np.max(np.where(below)[0])])
    return p <= crit, crit


@dataclass
class FeatureKS:
    """One feature's KS two-sample result (reference vs current)."""

    feature: str
    statistic: float
    p_value: float
    significant: bool = False


def ks_feature_tests(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], *, alpha: float = 0.05
) -> list[FeatureKS]:
    """Per-feature KS two-sample tests with BH-FDR significance across features.

    Columns absent from either frame, or without at least two finite values on both
    sides, are skipped (KS is undefined there). Results are returned worst-first by
    the KS statistic.
    """
    results: list[FeatureKS] = []
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if len(ref) < 2 or len(cur) < 2:
            continue
        stat, p_value = ks_2samp(ref, cur)
        results.append(FeatureKS(col, float(stat), float(p_value)))

    if results:
        significant, _ = benjamini_hochberg(np.array([r.p_value for r in results]), alpha)
        for r, flag in zip(results, significant, strict=True):
            r.significant = bool(flag)
    results.sort(key=lambda r: r.statistic, reverse=True)
    return results


def page_hinkley(stream: np.ndarray, *, delta: float = 0.005, lam: float = 50.0) -> int | None:
    """Page-Hinkley change-point for a sustained *increase* in a stream's mean.

    Tracks the cumulative deviation of each value from the running mean (minus a
    tolerance ``delta`` that absorbs benign wobble) and its running minimum; alarms
    the first time the gap between the two exceeds ``lam``. Returns the 0-based index
    where it would fire, or ``None`` if the stream never drifts. This is the standard
    one-sided formulation used for rising error/score streams.
    """
    values = np.asarray(stream, dtype=float)
    mean = 0.0
    cumulative = 0.0
    min_cumulative = 0.0
    for i, x in enumerate(values):
        mean += (x - mean) / (i + 1)
        cumulative += x - mean - delta
        min_cumulative = min(min_cumulative, cumulative)
        if cumulative - min_cumulative > lam:
            return i
    return None


@dataclass
class DDMResult:
    """DDM outcome over an error stream: first warning and first drift index."""

    warning_index: int | None
    drift_index: int | None


def ddm(
    errors: np.ndarray,
    *,
    warn_level: float = 2.0,
    drift_level: float = 3.0,
    min_samples: int = 30,
) -> DDMResult:
    """Drift Detection Method (Gama et al., 2004) on a binary error stream.

    For a stream of 0/1 errors it tracks the running error rate ``p`` and its
    binomial standard deviation ``s = sqrt(p(1-p)/n)``, remembering the minimum of
    ``p + s`` seen so far. A *warning* is raised when ``p + s >= p_min +
    warn_level * s_min`` and a *drift* alarm when the multiplier reaches
    ``drift_level`` — the error rate has climbed a statistically meaningful margin
    above its best point. Returns the first index of each (either may be ``None``).
    """
    e = np.asarray(errors, dtype=float)
    p = 0.0
    p_min = np.inf
    s_min = np.inf
    warning_index: int | None = None
    drift_index: int | None = None
    for i, err in enumerate(e):
        n = i + 1
        p += (err - p) / n
        s = float(np.sqrt(p * (1.0 - p) / n))
        # Don't arm until there are enough samples *and* a non-zero error rate to
        # estimate: locking a (0, 0) baseline from an all-correct warmup would make
        # the very first error trip the alarm. Once p > 0 the baseline is real.
        if n < min_samples or p <= 0.0:
            continue
        if p + s < p_min + s_min:
            p_min, s_min = p, s
        if drift_index is None and p + s >= p_min + drift_level * s_min:
            drift_index = i
        elif warning_index is None and p + s >= p_min + warn_level * s_min:
            warning_index = i
    return DDMResult(warning_index, drift_index)
