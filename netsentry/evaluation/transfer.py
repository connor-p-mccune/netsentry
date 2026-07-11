"""Threshold transfer — pricing "re-choose thresholds on labeled local traffic".

The cross-dataset study (and the Zeek ingestion notes) end on the same caveat:
the model's *ranking* transfers to a foreign schema, but a fixed-FPR operating
point does not — thresholds must be re-chosen on local traffic. This study turns
that advice into numbers: four threshold policies meet the same foreign dataset
at the primary FPR budget, ordered by how much local ground truth each needs.

1. **Transplant** — the source threshold (chosen on CIC validation) applied
   unchanged: zero local effort, and the policy every naive deployment starts
   with.
2. **Unsupervised quantile** — the (1 - budget) quantile of the *unlabeled*
   target scores. Free of labels, but only valid while the stream is
   benign-dominated; the report measures the violation by computing it both on
   the raw test mix and on a production-like low-prevalence mix.
3. **k labeled flows** — the threshold an analyst buys with ``k`` local labels,
   redrawn ``n_resamples`` times so the small-sample quantile noise (the refresh
   study's finding) is reported as spread, not hidden in one lucky draw.
4. **Oracle** — all target labels; the ceiling any label budget converges to.

Everything is scored on the raw attack-probability scale (matching the headline
evaluation); the foreign set is the synthetic NetFlow stand-in, so magnitudes
are illustrative and the *shape* — how fast compliance is bought back per label
— is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.cross_dataset import adapt_foreign_to_cic, generate_foreign
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.models.registry import ModelBundle, latest_bundle, load_bundle
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "threshold_transfer.md"


@dataclass
class PolicyPoint:
    """One threshold policy's realized operating point on the foreign set."""

    name: str
    threshold: float
    fpr: float
    tpr: float


@dataclass
class LabelBudgetPoint:
    """Realized-rate spread across redraws of ``k`` labeled target flows."""

    k: int
    fpr_median: float
    fpr_q25: float
    fpr_q75: float
    tpr_median: float
    compliance: float  # share of redraws whose realized FPR held the budget


def quantile_threshold(scores: np.ndarray, target_fpr: float) -> float:
    """The (1 - budget) quantile of an unlabeled score stream.

    Exact when every flow is benign; each attack in the stream pushes the
    quantile up, trading detection for an FPR under budget — the bias this
    study measures rather than assumes away.
    """
    return float(np.quantile(np.asarray(scores), 1.0 - target_fpr))


def compliance_share(fprs: np.ndarray, budget: float, factor: float) -> float:
    """Share of realized FPRs within ``factor`` of the budget, both sides.

    "Held the budget" means the queue is neither flooded (> factor * budget)
    nor starved (< budget / factor, which silently spends detection).
    """
    fprs = np.asarray(fprs, dtype=float)
    lo, hi = budget / factor, budget * factor
    return float(np.mean((fprs >= lo) & (fprs <= hi)))


def label_budget_trials(
    y: np.ndarray,
    scores: np.ndarray,
    k: int,
    budget: float,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(realized FPRs, realized TPRs) across seeded redraws of ``k`` labeled flows.

    Each trial chooses the threshold on its k-row sample at the budget, then
    realizes the rates on the full set — the spread is the price of estimating
    an extreme quantile from a small labeled window.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    k = min(k, len(y))
    fprs, tprs = [], []
    for _ in range(n_resamples):
        idx = rng.choice(len(y), size=k, replace=False)
        threshold = threshold_at_fpr(y[idx], scores[idx], budget)
        rates = rates_at_threshold(y, scores, threshold)
        fprs.append(rates["fpr"])
        tprs.append(rates["tpr"])
    return np.array(fprs), np.array(tprs)


def _load_deployed_bundle(settings: Settings) -> ModelBundle:
    path = settings.serving.artifact_path or latest_bundle(settings)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "No model bundle found. Train one with `netsentry train supervised` first."
        )
    return load_bundle(Path(path))


def _scores(bundle: ModelBundle, frame: pd.DataFrame, benign: str) -> np.ndarray:
    return attack_probability(bundle.predict_proba(frame), bundle.classes, benign)


def _production_mix(y: np.ndarray, scores: np.ndarray, attack_rate: float, seed: int) -> np.ndarray:
    """Scores of a low-prevalence resample: all benign flows + a thin attack share."""
    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]
    n_attack = int(len(benign_scores) * attack_rate / (1.0 - attack_rate))
    if n_attack == 0 or len(attack_scores) == 0:
        return np.asarray(benign_scores)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(attack_scores, size=min(n_attack, len(attack_scores)), replace=False)
    return np.concatenate([benign_scores, sampled])


def run_transfer_report(settings: Settings, *, bundle: ModelBundle | None = None) -> Path:
    """Compare the four threshold-transfer policies on the foreign set; write the report."""
    if bundle is None:
        bundle = _load_deployed_bundle(settings)
    benign = settings.labels.benign_label
    budget = settings.thresholds.primary_fpr
    cfg = settings.transfer

    val = load_split(settings, "temporal", "val")
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    s_val = _scores(bundle, val, benign)

    adapted = adapt_foreign_to_cic(generate_foreign(settings), settings)
    y_x = adapted[BINARY_TARGET].to_numpy().astype(int)
    s_x = _scores(bundle, adapted, benign)
    prevalence = float(np.mean(y_x))

    def _policy(name: str, threshold: float) -> PolicyPoint:
        rates = rates_at_threshold(y_x, s_x, threshold)
        return PolicyPoint(name, threshold, rates["fpr"], rates["tpr"])

    prod_rate = settings.cost.production_attack_rate
    policies = [
        _policy(
            "transplant (source threshold, zero local effort)",
            threshold_at_fpr(y_val, s_val, budget),
        ),
        _policy(
            f"quantile, unlabeled (as-is stream, {prevalence:.0%} attack)",
            quantile_threshold(s_x, budget),
        ),
        _policy(
            f"quantile, unlabeled ({prod_rate:.0%}-attack stream)",
            quantile_threshold(_production_mix(y_x, s_x, prod_rate, settings.seed), budget),
        ),
        _policy("oracle (all target labels)", threshold_at_fpr(y_x, s_x, budget)),
    ]

    budget_points: list[LabelBudgetPoint] = []
    for k in cfg.label_budgets:
        fprs, tprs = label_budget_trials(y_x, s_x, k, budget, cfg.n_resamples, settings.seed)
        budget_points.append(
            LabelBudgetPoint(
                k=k,
                fpr_median=float(np.median(fprs)),
                fpr_q25=float(np.quantile(fprs, 0.25)),
                fpr_q75=float(np.quantile(fprs, 0.75)),
                tpr_median=float(np.median(tprs)),
                compliance=compliance_share(fprs, budget, cfg.compliance_factor),
            )
        )
        logger.info(
            "Label-budget point",
            extra={"k": k, "fpr_median": budget_points[-1].fpr_median},
        )

    fig = _plot(settings, policies, budget_points, budget, len(y_x))
    report = _render(settings, policies, budget_points, budget, prevalence, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote threshold-transfer report", extra={"path": str(out_path)})

    transplant, oracle = policies[0], policies[-1]
    labels_needed = next((p.k for p in budget_points if p.compliance >= 0.5), None)
    with track_run(settings, "threshold_transfer") as run:
        run.log_metrics(
            {
                "transplant_fpr_ratio": transplant.fpr / budget if budget else float("nan"),
                "oracle_tpr": oracle.tpr,
                "labels_for_majority_compliance": (
                    float(labels_needed) if labels_needed is not None else float("nan")
                ),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(
    settings: Settings,
    policies: list[PolicyPoint],
    points: list[LabelBudgetPoint],
    budget: float,
    n_rows: int,
) -> Path:
    ks = np.array([p.k for p in points], dtype=float)
    # A realized FPR of exactly zero cannot sit on a log axis; plot it at the
    # half-flow resolution floor (finer than one false positive in the set).
    floor = 0.5 / n_rows

    def _safe(values: np.ndarray | list[float]) -> np.ndarray:
        return np.asarray(np.maximum(np.asarray(values, dtype=float), floor))

    series = {
        "k labeled flows (median over redraws)": (ks, _safe([p.fpr_median for p in points])),
        "budget": (ks, np.full_like(ks, budget)),
        "transplant": (ks, _safe(np.full_like(ks, policies[0].fpr))),
        "oracle (all labels)": (ks, _safe(np.full_like(ks, policies[-1].fpr))),
    }
    return plots.plot_lines(
        series,
        xlabel="Local labels spent on threshold selection",
        ylabel="Realized FPR on the foreign set",
        title="Buying back the operating point with local labels",
        out_path=settings.paths.figures_dir / "threshold_transfer.png",
        xscale="log",
        yscale="log",
    )


def _policy_table(policies: list[PolicyPoint], budget: float) -> str:
    rows = [
        "| policy | threshold | realized FPR | vs budget | detection (TPR) |",
        "|---|---|---|---|---|",
    ]
    for p in policies:
        ratio = p.fpr / budget if budget else float("inf")
        rows.append(
            f"| {p.name} | {p.threshold:.4f} | {p.fpr:.4%} | {ratio:,.1f}x "
            f"| {p.tpr * 100:.1f}% |"
        )
    return "\n".join(rows)


def _budget_table(points: list[LabelBudgetPoint], factor: float) -> str:
    rows = [
        f"| labels | realized FPR (median) | IQR | budget held (within {factor:g}x) "
        "| detection (median) |",
        "|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.k:,} | {p.fpr_median:.4%} | [{p.fpr_q25:.4%}, {p.fpr_q75:.4%}] "
            f"| {p.compliance * 100:.0f}% | {p.tpr_median * 100:.1f}% |"
        )
    return "\n".join(rows)


def _read(
    policies: list[PolicyPoint],
    points: list[LabelBudgetPoint],
    budget: float,
    prevalence: float,
) -> str:
    """Sign-aware reading so the prose tracks whatever the numbers did."""
    transplant, q_asis, q_prod, oracle = policies
    ratio = transplant.fpr / budget if budget else float("inf")
    if ratio > 2:
        transplant_read = (
            f"**The transplanted threshold runs {ratio:,.0f}x over budget** "
            f"({transplant.fpr:.3%} against {budget:.3%}): the score distribution moved "
            f"between schemas, so the source operating point floods the local queue — the "
            f"cross-dataset caveat, realized as alert volume."
        )
    elif ratio < 0.5:
        transplant_read = (
            f"**The transplanted threshold under-spends the budget** ({transplant.fpr:.4%} "
            f"against {budget:.3%}, {ratio:.2f}x): the foreign scores sit lower on the scale, "
            f"so the source cut silently buys less detection ({transplant.tpr * 100:.1f}% vs "
            f"the oracle's {oracle.tpr * 100:.1f}%) while *looking* quiet — the more insidious "
            f"failure, because nothing pages when a threshold is too strict."
        )
    else:
        transplant_read = (
            f"The transplanted threshold lands within {ratio:.1f}x of budget on this stand-in "
            f"— gentler than the real cross-schema case, where calibration shift is the norm; "
            f"the label-budget curve below is the machinery that matters when it is not."
        )

    quantile_read = (
        f"The unsupervised quantile is the tempting label-free fix, and the two rows show "
        f"its operating envelope: on the as-is stream ({prevalence:.0%} attack) it realizes "
        f"{q_asis.fpr:.4%} FPR at {q_asis.tpr * 100:.1f}% detection — every attack in the "
        f"stream pushes the quantile into the attack mass, so it under-alerts exactly when "
        f"traffic is hostile — while on a production-like mix it lands at {q_prod.fpr:.4%} "
        f"({q_prod.tpr * 100:.1f}% detection). It is a prevalence assumption wearing a "
        f"statistics costume, and it fails quietly in the direction of missed attacks."
    )

    labels_needed = next((p.k for p in points if p.compliance >= 0.5), None)
    strong = next((p.k for p in points if p.compliance >= 0.8), None)
    if labels_needed is not None:
        tail = (
            f"and {strong:,} labels hold it in at least 80% of redraws"
            if strong is not None
            else "though no tested budget holds it in 80% of redraws — the IQR column is the "
            "warning against trusting one draw"
        )
        label_read = (
            f"**The budget is bought back with labels, and the price is visible:** "
            f"{labels_needed:,} local labels hold the realized FPR within a factor of two in "
            f"at least half the redraws, {tail}. Estimating a {budget:.1%} quantile needs "
            f"roughly {1 / budget:,.0f} benign flows per expected false positive, which is "
            f"why the small budgets scatter across orders of magnitude — the refresh study's "
            f"small-window noise, met again at deployment."
        )
    else:
        label_read = (
            f"**No tested label budget reliably holds the FPR budget** (compliance never "
            f"reaches 50%). At a {budget:.1%} target, a k-row sample contains on the order of "
            f"k x {budget:.0e} expected false positives — the quantile simply is not "
            f"estimable from these window sizes, and the honest advice is a looser budget or "
            f"a bigger labeled window."
        )
    return f"{transplant_read}\n\n{quantile_read}\n\n{label_read}"


def _render(
    settings: Settings,
    policies: list[PolicyPoint],
    points: list[LabelBudgetPoint],
    budget: float,
    prevalence: float,
    fig: Path,
) -> str:
    cfg = settings.transfer
    return f"""# NetSentry — Threshold Transfer (the operating point, re-bought locally)

_Synthetic stand-in. The deployed bundle's raw attack scores on the foreign
NetFlow-schema set; every policy targets the primary {budget:.1%} FPR budget;
label-budget rows are medians over {cfg.n_resamples} seeded redraws. The foreign
set and its {prevalence:.0%} attack mix are the cross-dataset study's stand-in —
the shape, not the magnitudes, is the finding._

## Why this report exists

The cross-dataset study ends with "the ranking transfers, the calibration does
not — re-choose thresholds on labeled local traffic," and the Zeek ingestion
docs repeat it. This report prices that sentence: what does each level of local
effort — none, unlabeled traffic, k labels, all labels — actually buy at the
operating point?

## The four policies

{_policy_table(policies, budget)}

## Buying the budget back with labels

{_budget_table(points, cfg.compliance_factor)}

![Threshold transfer](../figures/{fig.name})

_Realized FPRs of exactly zero are plotted at the half-flow resolution floor;
the table carries the exact values._

## Read

{_read(policies, points, budget, prevalence)}

## Method & limits

- Scores are the bundle's raw attack probabilities (the headline evaluation's
  scale); thresholds are compared on that one scale throughout.
- Label-budget redraws resample the same finite foreign set, so the spread is a
  bootstrap-flavored estimate of sampling noise, not fresh traffic.
- "Budget held" counts a redraw whose realized FPR is within a factor of
  {cfg.compliance_factor:g} of budget **on either side** — an over-strict
  threshold silently spends detection, which is a failure too, not a win.
- The foreign set is synthetic; on real UNSW-NB15 / NF-*-v2 data the transplant
  row is expected to be worse and the label curve slower. The commands are
  identical.
"""
