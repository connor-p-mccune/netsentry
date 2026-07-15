"""Feature interactions (Friedman's H-statistic): what the PDP can only warn about.

The partial-dependence report ends on a caveat — a PDP assumes the swept feature is
independent of the others, and where features move together the average hides
**interaction**, with the ICE spread as the only hint. This study measures that
interaction directly. Friedman's H-statistic (Friedman & Popescu, 2008) decomposes a
feature pair's joint partial dependence into an additive part and a non-additive
(interaction) part, and reports the fraction of the joint variance the interaction
accounts for:

    H^2_jk = sum_i [ PD_jk(x_ji, x_ki) - PD_j(x_ji) - PD_k(x_ki) ]^2
             / sum_i [ PD_jk(x_ji, x_ki) ]^2                (all PDs mean-centred)

H = 0 means the two features act purely additively (the joint effect is exactly the
sum of the marginals — a PDP tells the whole story); H -> 1 means the effect of one
feature depends strongly on the other (the model has learned an interaction a
single-feature view cannot show). It is the natural completion of the interpretability
suite: SHAP says *which* features matter, PDP says *what shape* each one's response is,
the ablation says *which family is causally load-bearing*, and this says *which
features the model has entangled*.

Computed on the honest **temporal / binary** model, through the fitted pipeline, over a
background sample — the same scorer the partial-dependence study uses, so the two read
against each other. The cost is quadratic in the sample size per pair, so the sample is
deliberately small; the report states that the estimate is a Monte-Carlo one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.evaluation import plots
from netsentry.explain.partial_dependence import ScoreFn, _fit_scorer
from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "interactions.md"
FIGURE_NAME = "interactions.png"


def h_statistic(pd_jk: np.ndarray, pd_j: np.ndarray, pd_k: np.ndarray) -> float:
    """Friedman's pairwise H from the (uncentred) partial dependences at the sample points.

    Each partial dependence is mean-centred, then H is the square root of the share of
    the joint variance that the additive model ``PD_j + PD_k`` fails to explain. Clipped
    to [0, 1]: a finite-sample estimate can nudge just past 1 when the marginals are
    anti-correlated, and the interpretable quantity is a fraction.
    """
    jk = np.asarray(pd_jk, dtype=float)
    jk = jk - jk.mean()
    j = np.asarray(pd_j, dtype=float)
    j = j - j.mean()
    k = np.asarray(pd_k, dtype=float)
    k = k - k.mean()
    denominator = float(np.sum(jk**2))
    if denominator <= 1e-12:  # a constant joint response has no interaction to measure
        return 0.0
    numerator = float(np.sum((jk - j - k) ** 2))
    return float(np.sqrt(min(numerator / denominator, 1.0)))


def _pd_1d(score_fn: ScoreFn, sample: pd.DataFrame, feature: str, values: np.ndarray) -> np.ndarray:
    """Partial dependence of ``feature`` evaluated at each supplied value (mean over sample)."""
    out = np.empty(len(values), dtype=float)
    for i, value in enumerate(values):
        perturbed = sample.copy()
        perturbed[feature] = value
        out[i] = float(np.mean(score_fn(perturbed)))
    return out


def _pd_2d(
    score_fn: ScoreFn,
    sample: pd.DataFrame,
    fj: str,
    fk: str,
    vj: np.ndarray,
    vk: np.ndarray,
) -> np.ndarray:
    """Joint partial dependence of ``(fj, fk)`` at each observed pair (mean over sample)."""
    out = np.empty(len(vj), dtype=float)
    for i in range(len(vj)):
        perturbed = sample.copy()
        perturbed[fj] = vj[i]
        perturbed[fk] = vk[i]
        out[i] = float(np.mean(score_fn(perturbed)))
    return out


def pairwise_h(score_fn: ScoreFn, sample: pd.DataFrame, fj: str, fk: str) -> float:
    """Estimate Friedman's H for one feature pair over a background sample."""
    vj = sample[fj].to_numpy(dtype=float)
    vk = sample[fk].to_numpy(dtype=float)
    pd_j = _pd_1d(score_fn, sample, fj, vj)
    pd_k = _pd_1d(score_fn, sample, fk, vk)
    pd_jk = _pd_2d(score_fn, sample, fj, fk, vj, vk)
    return h_statistic(pd_jk, pd_j, pd_k)


@dataclass
class InteractionStudy:
    """Pairwise interaction strengths among the top model features."""

    features: list[str]
    matrix: np.ndarray  # symmetric K x K H-statistic matrix (zero diagonal)
    n_sample: int

    def pairs(self) -> list[tuple[str, str, float]]:
        """All feature pairs, strongest interaction first."""
        out: list[tuple[str, str, float]] = []
        for a, b in combinations(range(len(self.features)), 2):
            out.append((self.features[a], self.features[b], float(self.matrix[a, b])))
        out.sort(key=lambda t: t[2], reverse=True)
        return out

    def involvement(self) -> list[tuple[str, float]]:
        """Per-feature interaction involvement = its strongest pairwise H (a total-H proxy)."""
        out = [(f, float(self.matrix[i].max())) for i, f in enumerate(self.features)]
        out.sort(key=lambda t: t[1], reverse=True)
        return out


def compute_interactions(settings: Settings) -> InteractionStudy:
    """Compute the pairwise H-statistic matrix for the top features of the temporal model."""
    cfg = settings.interactions
    score_fn, train, val, ranking = _fit_scorer(settings)

    per_feature = ranking.groupby(level=0).max().sort_values(ascending=False)
    sample = val.sample(n=min(cfg.sample_rows, len(val)), random_state=settings.seed).reset_index(
        drop=True
    )
    # Rank raw features by importance, then take the top-k that (a) exist as columns and
    # (b) actually vary in the sample — a constant column has no interaction to measure,
    # and filtering before the cut keeps the full k rather than dropping into a short set.
    features = [
        f
        for f in per_feature.index
        if f in train.columns and sample[f].to_numpy(dtype=float).std() > 0
    ][: cfg.top_k]

    k = len(features)
    matrix = np.zeros((k, k), dtype=float)
    for a, b in combinations(range(k), 2):
        h = pairwise_h(score_fn, sample, features[a], features[b])
        matrix[a, b] = matrix[b, a] = h
    logger.info(
        "Interaction study",
        extra={"features": k, "max_h": round(float(matrix.max()), 3) if k else 0.0},
    )
    return InteractionStudy(features=features, matrix=matrix, n_sample=len(sample))


def run_interactions_report(settings: Settings) -> Path:
    """Compute the H-statistic matrix, plot it, and write the report."""
    study = compute_interactions(settings)

    fig = plots.plot_heatmap(
        study.matrix,
        study.features,
        title="Feature interaction strength (Friedman's H)",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        cbar_label="interaction strength H (0 = additive, 1 = fully entangled)",
    )

    report = _render(study, fig, settings)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote interactions report", extra={"path": str(out_path)})

    with track_run(settings, "interactions") as run:
        for a, b, h in study.pairs()[:8]:
            run.log_metrics({f"H_{a[:16]}_{b[:16]}": h})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _pairs_table(study: InteractionStudy, limit: int) -> str:
    rows = ["| feature pair | H (interaction) |", "|---|---|"]
    for a, b, h in study.pairs()[:limit]:
        rows.append(f"| {a} x {b} | {h:.3f} |")
    return "\n".join(rows)


def _involvement_table(study: InteractionStudy) -> str:
    rows = ["| feature | strongest interaction (H) |", "|---|---|"]
    for f, h in study.involvement():
        rows.append(f"| {f} | {h:.3f} |")
    return "\n".join(rows)


def _read(study: InteractionStudy) -> str:
    pairs = study.pairs()
    if not pairs:
        return "_Too few varying top features to estimate interactions on this data._"
    top_a, top_b, top_h = pairs[0]
    strong = top_h >= 0.3
    if strong:
        return (
            f"The strongest interaction the model has learned is **{top_a} x {top_b}** "
            f"(H = {top_h:.2f}): a large share of that pair's joint effect on the attack "
            "score is non-additive, so neither feature's partial-dependence curve tells the "
            "whole story about it — the effect of one bends with the value of the other. That "
            "is the concrete form of the caveat the partial-dependence report raises: the "
            "steepest-response features here (flow rates, packet counts, byte totals) are also "
            "the ones that move together in real traffic, and the model has entangled them "
            "rather than treating them as independent dials."
        )
    return (
        f"Interactions are mild on this stand-in — the strongest is {top_a} x {top_b} at "
        f"H = {top_h:.2f} — so the model's response is close to additive in its top features, "
        "and the partial-dependence curves are a faithful per-feature summary. That is itself "
        "worth stating: a low H is the licence to read the PDPs one feature at a time, which a "
        "PDP alone cannot justify."
    )


def _render(study: InteractionStudy, fig: Path, settings: Settings) -> str:
    return f"""# NetSentry - Feature Interactions (Friedman's H-statistic)

_Synthetic stand-in. Pairwise H-statistic among the top {len(study.features)} features
of the honest **temporal / binary** model, estimated over a background sample of
{study.n_sample} flows through the fitted pipeline — the same scorer the
partial-dependence study uses. H is a Monte-Carlo estimate; it ranges 0 (the pair acts
additively) to 1 (the effect of one feature fully depends on the other)._

## Why this report exists

The partial-dependence report shows the *shape* of each top feature's response but warns
that a PDP assumes the feature is independent of the others — where features move
together, the marginal curve hides **interaction**. This report measures that
interaction. Friedman's H (Friedman & Popescu, 2008) is the fraction of a feature pair's
joint-response variance that is *not* explained by adding the two marginal responses: a
model-agnostic, dimensionless interaction strength.

## Strongest interacting pairs

{_pairs_table(study, settings.interactions.max_pairs_reported)}

![Feature interaction matrix](../figures/{fig.name})

{_read(study)}

## Per-feature interaction involvement

Each feature's strongest pairwise H — a cheap proxy for Friedman's total-interaction
statistic (which additionally requires the full complement partial dependence). A high
value means the feature's contribution is context-dependent; a low value means it acts
like an independent dial the SHAP/PDP summaries capture faithfully.

{_involvement_table(study)}

## How to read this (and what it is not)

A high H does not say the interaction is *large* in absolute terms — only that, relative
to the pair's own joint effect, the effect is non-additive; a feature with a tiny
marginal effect can still show a high H. So H complements, and does not replace, the SHAP
importances and the PDP effect sizes: importance says how much a feature moves the score,
the PDP says in what shape, and H says whether that shape is fixed or bends with another
feature. Like the PDP, the estimate is computed by perturbing features independently, so
in strongly correlated regions it evaluates the model off the data manifold — which is
precisely why a measured interaction there matters, since it is where an additive reading
would most mislead. The causal value of a whole feature family remains the ablation
study's question; this is a diagnostic of the learned response surface, not a causal claim.
"""
