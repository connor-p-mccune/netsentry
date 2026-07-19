"""Positive-unlabeled learning: train honestly from the labels a SOC actually has.

Every supervised study in this suite assumes the benign side of the training pool is
*verified* benign. A real deployment has nothing of the sort: incident response confirms a
subset of the attacks (tickets, IOC matches), and everything else is simply **unlabeled** —
overwhelmingly benign, but salted with the attacks nobody caught. Training "labeled vs rest"
as if the rest were clean plants the missed attacks as label noise on the negative side, and
selecting a false-positive budget against a contaminated "benign" pool distorts the operating
point.

Positive-unlabeled (PU) learning makes that situation explicit instead of ignoring it. Under
the SCAR assumption (labels are Selected Completely At Random from the positives), Elkan &
Noto (KDD 2008) show the "nontraditional" classifier ``g(x) = p(s=1|x)`` trained on
labeled-vs-unlabeled relates to the true posterior by a single constant:
``p(y=1|x) = g(x) / c`` with ``c = p(s=1|y=1)``, the label frequency. The constant is
estimable from a validation set (``c_hat = mean g over labeled positives``), which yields:

- **Corrected scores** ``g/c`` — monotone in ``g``, so ranking (PR-AUC) is unchanged; the
  value is calibration and everything downstream of it (thresholds, prevalence, costs).
- A **prevalence estimate** ``pi = E[g]/c`` — how many attacks are hiding in the unlabeled
  mass, with zero extra labels (the PU sibling of the [label-shift](label_shift.md) study).
- The **weighted retrain** (their second method): each unlabeled flow enters the training set
  twice, as a positive with weight ``w(x) = ((1-c)/c) * g/(1-g)`` (the posterior probability
  that an unlabeled example is positive) and as a negative with ``1-w`` — a genuinely
  different model, not a rescaling.
- A **PU-corrected operating point**: the estimated benign mass ``1-w`` replaces the
  contaminated head-count in the FPR denominator, so the budget is priced against the benign
  traffic that is actually there — same model, same scores, only the bookkeeping fixed.

The study sweeps the confirmed fraction ``c``, audits ``c_hat`` and the prevalence estimate
against the truth the estimators never saw, prices naive vs PU-weighted vs the
fully-supervised oracle on the honest temporal test split, and measures the budget
distortion directly. Sits with [weak supervision](weak_supervision.md) (zero labels),
[active learning](active_learning.md) (which labels to buy) and
[self-training](selftrain.md) (the pseudo-label shortcut) as the fourth answer to "what can
you train when nobody labels the benign side?".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import (
    attack_probability,
    rates_at_threshold,
    threshold_at_fpr,
)
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import PULearnConfig

logger = get_logger(__name__)

REPORT_NAME = "pu_learning.md"
FIGURE_NAME = "pu_learning.png"


def scar_labels(y: np.ndarray, label_frac: float, rng: np.random.Generator) -> np.ndarray:
    """SCAR labeling: mark a random ``label_frac`` of the true positives as confirmed.

    Returns ``s`` with ``s=1`` only where ``y=1`` (a confirmed attack) — the labeled set a
    SOC's incident tickets produce under the Selected-Completely-At-Random assumption.
    """
    y = np.asarray(y).astype(int)
    s = np.zeros(len(y), dtype=int)
    positives = np.flatnonzero(y == 1)
    n_labeled = round(label_frac * len(positives))
    if n_labeled:
        s[rng.choice(positives, size=n_labeled, replace=False)] = 1
    return s


def estimate_c(g_scores: np.ndarray, s: np.ndarray) -> float:
    """Elkan-Noto ``e1``: the mean nontraditional score over the labeled positives.

    Under SCAR with a calibrated ``g``, ``E[g(x) | s=1] = c`` when the true posterior is
    near 1 on positives; the report audits the estimate against the truth it never saw.
    """
    labeled = np.asarray(g_scores)[np.asarray(s) == 1]
    if len(labeled) == 0:
        return 1.0
    return float(np.clip(np.mean(labeled), 1e-6, 1.0))


def correct_scores(g_scores: np.ndarray, c_hat: float) -> np.ndarray:
    """The Elkan-Noto correction ``p(y=1|x) = g(x)/c``, capped at 1 (monotone in ``g``)."""
    return np.asarray(np.clip(np.asarray(g_scores, dtype=float) / c_hat, 0.0, 1.0))


def unlabeled_posterior_weights(
    g_scores: np.ndarray, c_hat: float, score_clip: float
) -> np.ndarray:
    """``p(y=1 | x, s=0) = ((1-c)/c) * g/(1-g)`` for unlabeled rows, clipped into [0, 1]."""
    g = np.clip(np.asarray(g_scores, dtype=float), score_clip, 1.0 - score_clip)
    w = ((1.0 - c_hat) / c_hat) * (g / (1.0 - g))
    return np.asarray(np.clip(w, 0.0, 1.0))


def expand_weighted(
    x: np.ndarray, s: np.ndarray, w_unlabeled: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the Elkan-Noto weighted design: positives once, each unlabeled row twice.

    Labeled positives carry (label 1, weight 1); an unlabeled row appears as a positive with
    weight ``w(x)`` and as a negative with ``1-w(x)``, so its total mass stays exactly 1.
    """
    s = np.asarray(s).astype(int)
    pos, unl = x[s == 1], x[s == 0]
    x_out = np.concatenate([pos, unl, unl])
    y_out = np.concatenate([np.ones(len(pos)), np.ones(len(unl)), np.zeros(len(unl))]).astype(int)
    w_out = np.concatenate([np.ones(len(pos)), w_unlabeled, 1.0 - w_unlabeled])
    return x_out, y_out, w_out


def prevalence_from_pu(mean_g: float, c_hat: float) -> float:
    """Elkan-Noto prevalence: ``E[y] = E[g]/c``, capped at 1."""
    return float(np.clip(mean_g / c_hat, 0.0, 1.0))


def pu_threshold_for_budget(
    scores_unlabeled: np.ndarray, benign_mass: np.ndarray, budget: float
) -> float:
    """Highest-detection threshold whose *estimated true* FPR stays within ``budget``.

    The naive bookkeeping counts every unlabeled row as benign; here each row contributes
    only its estimated benign mass ``1 - w(x)`` to the false-positive denominator, so hidden
    attacks stop inflating the apparent FPR and over-tightening the cut.
    """
    scores = np.asarray(scores_unlabeled, dtype=float)
    mass = np.asarray(benign_mass, dtype=float)
    total = float(mass.sum())
    if total <= 0:
        return float(np.max(scores)) if len(scores) else 1.0
    candidates = np.unique(scores)[::-1]  # descending: strictest first
    chosen = float(candidates[0]) + 1e-12  # above the max: zero alerts, trivially in budget
    for t in candidates:
        est_fpr = float(mass[scores >= t].sum()) / total
        if est_fpr > budget:
            break
        chosen = float(t)
    return chosen


@dataclass
class PUPoint:
    """One labeled-fraction setting: estimator recovery and detection."""

    label_frac: float
    n_labeled: int
    c_hat: float
    prevalence_hat: float
    naive_pr_auc: float
    weighted_pr_auc: float


@dataclass
class BudgetOutcome:
    """Operating-point arithmetic at the headline fraction: one model, three thresholds."""

    budget: float
    apparent_fpr: float  # what the naive bookkeeping believes its own cut costs
    naive_fpr: float  # that cut's realized FPR on true test benign
    naive_tpr: float
    pu_fpr: float  # the PU-corrected cut, realized
    pu_tpr: float
    oracle_fpr: float  # the true-label cut, realized (same model, same scores)
    oracle_tpr: float


@dataclass
class PUStudy:
    """The full positive-unlabeled study on the temporal/binary split."""

    n_train: int
    n_test: int
    train_prevalence: float
    oracle_pr_auc: float
    points: list[PUPoint]
    headline_frac: float
    budget: BudgetOutcome


def _attack_scores(model: SupervisedClassifier, x: np.ndarray, benign: str) -> np.ndarray:
    return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)


def run_pu_learning(settings: Settings) -> PUStudy:
    """Sweep the confirmed-attack fraction; price naive vs PU-weighted vs the oracle."""
    cfg: PULearnConfig = settings.pu_learning
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    # The fully-supervised ceiling: the deployed protocol with every label real.
    seed_everything(variant.seed)
    oracle = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    oracle_test = _attack_scores(oracle, x_test, benign)
    oracle_pr_auc = float(average_precision_score(y_test, oracle_test))

    # g must approximate p(s=1|x), so it trains without the balanced reweighting the
    # deployed model uses (balancing shifts the effective prior and breaks the c estimate).
    g_variant = variant.model_copy(deep=True)
    g_variant.supervised.class_weight = "none"

    points: list[PUPoint] = []
    budget_outcome: BudgetOutcome | None = None
    for frac in cfg.label_fracs:
        rng = np.random.default_rng(variant.seed + round(frac * 1000))
        s_train = scar_labels(y_train, frac, rng)
        s_val = scar_labels(y_val, frac, rng)

        seed_everything(variant.seed)
        naive = SupervisedClassifier(variant).fit(x_train, s_train, eval_set=(x_val, s_val))
        naive_val = _attack_scores(naive, x_val, benign)
        naive_test = _attack_scores(naive, x_test, benign)

        seed_everything(variant.seed)
        g_model = SupervisedClassifier(g_variant).fit(x_train, s_train, eval_set=(x_val, s_val))
        g_val = _attack_scores(g_model, x_val, benign)
        g_train = _attack_scores(g_model, x_train, benign)
        c_hat = estimate_c(g_val, s_val)
        prevalence_hat = prevalence_from_pu(float(np.mean(g_val)), c_hat)

        w_train = unlabeled_posterior_weights(g_train[s_train == 0], c_hat, cfg.score_clip)
        x_dup, y_dup, w_dup = expand_weighted(x_train, s_train, w_train)
        if len(x_dup) > cfg.max_weighted_rows:
            keep = rng.choice(len(x_dup), size=cfg.max_weighted_rows, replace=False)
            x_dup, y_dup, w_dup = x_dup[keep], y_dup[keep], w_dup[keep]
        seed_everything(variant.seed)
        weighted = SupervisedClassifier(g_variant).fit(x_dup, y_dup, sample_weight=w_dup)
        weighted_test = _attack_scores(weighted, x_test, benign)

        points.append(
            PUPoint(
                label_frac=float(frac),
                n_labeled=int(s_train.sum()),
                c_hat=c_hat,
                prevalence_hat=prevalence_hat,
                naive_pr_auc=float(average_precision_score(y_test, naive_test)),
                weighted_pr_auc=float(average_precision_score(y_test, weighted_test)),
            )
        )
        logger.info(
            "PU point done",
            extra={"frac": frac, "c_hat": round(c_hat, 4), "labeled": int(s_train.sum())},
        )

        if abs(frac - cfg.headline_frac) < 1e-9:
            # One model, three cuts: naive bookkeeping vs PU-corrected vs true labels.
            apparent_thr = threshold_at_fpr(s_val, naive_val, cfg.budget_fpr)
            apparent_fpr = rates_at_threshold(s_val, naive_val, apparent_thr)["fpr"]
            w_val = unlabeled_posterior_weights(g_val[s_val == 0], c_hat, cfg.score_clip)
            pu_thr = pu_threshold_for_budget(naive_val[s_val == 0], 1.0 - w_val, cfg.budget_fpr)
            oracle_thr = threshold_at_fpr(y_val, naive_val, cfg.budget_fpr)
            naive_rates = rates_at_threshold(y_test, naive_test, apparent_thr)
            pu_rates = rates_at_threshold(y_test, naive_test, pu_thr)
            oracle_rates = rates_at_threshold(y_test, naive_test, oracle_thr)
            budget_outcome = BudgetOutcome(
                budget=cfg.budget_fpr,
                apparent_fpr=float(apparent_fpr),
                naive_fpr=naive_rates["fpr"],
                naive_tpr=naive_rates["tpr"],
                pu_fpr=pu_rates["fpr"],
                pu_tpr=pu_rates["tpr"],
                oracle_fpr=oracle_rates["fpr"],
                oracle_tpr=oracle_rates["tpr"],
            )

    if budget_outcome is None:
        raise ValueError(
            f"pu_learning.headline_frac={cfg.headline_frac} must be one of label_fracs"
        )
    return PUStudy(
        n_train=len(y_train),
        n_test=len(y_test),
        train_prevalence=float(np.mean(y_train)),
        oracle_pr_auc=oracle_pr_auc,
        points=points,
        headline_frac=cfg.headline_frac,
        budget=budget_outcome,
    )


def run_pu_learning_report(settings: Settings) -> Path:
    """Run the PU study and write the report + figure."""
    study = run_pu_learning(settings)

    fracs = np.array([p.label_frac for p in study.points])
    series = {
        "naive (unlabeled treated as benign)": (
            fracs,
            np.array([p.naive_pr_auc for p in study.points]),
        ),
        "PU-weighted (Elkan-Noto)": (
            fracs,
            np.array([p.weighted_pr_auc for p in study.points]),
        ),
        "oracle (every label real)": (
            fracs,
            np.full(len(fracs), study.oracle_pr_auc),
        ),
    }
    fig = plots.plot_lines(
        series,
        xlabel="fraction of attacks with confirmed labels (c)",
        ylabel="test PR-AUC (temporal split)",
        title="Detection vs the confirmed-attack fraction: naive, PU-weighted, oracle",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote PU-learning report", extra={"path": str(out_path)})

    headline = next(p for p in study.points if p.label_frac == study.headline_frac)
    with track_run(settings, "pu_learning") as run:
        run.log_metrics(
            {
                "oracle_pr_auc": study.oracle_pr_auc,
                "headline_naive_pr_auc": headline.naive_pr_auc,
                "headline_weighted_pr_auc": headline.weighted_pr_auc,
                "headline_c_hat": headline.c_hat,
                "budget_naive_tpr": study.budget.naive_tpr,
                "budget_pu_tpr": study.budget.pu_tpr,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _sweep_table(study: PUStudy) -> str:
    rows = [
        "| confirmed fraction c | labeled attacks | c_hat | est. prevalence | naive PR-AUC "
        "| PU-weighted PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for p in study.points:
        marker = " (headline)" if p.label_frac == study.headline_frac else ""
        rows.append(
            f"| {p.label_frac:.2f}{marker} | {p.n_labeled:,} | {p.c_hat:.3f} "
            f"| {p.prevalence_hat:.3f} | {p.naive_pr_auc:.3f} | {p.weighted_pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _budget_table(b: BudgetOutcome) -> str:
    return "\n".join(
        [
            "| threshold policy | realized FPR (true benign) | detection (TPR) |",
            "|---|---|---|",
            f"| naive bookkeeping (unlabeled = benign) | {b.naive_fpr:.4f} | {b.naive_tpr:.1%} |",
            f"| PU-corrected denominator | {b.pu_fpr:.4f} | {b.pu_tpr:.1%} |",
            f"| oracle (true validation labels) | {b.oracle_fpr:.4f} | {b.oracle_tpr:.1%} |",
        ]
    )


def _estimator_read(study: PUStudy) -> str:
    errors = [abs(p.c_hat - p.label_frac) for p in study.points]
    mae = float(np.mean(errors))
    prev_err = float(
        np.mean([abs(p.prevalence_hat - study.train_prevalence) for p in study.points])
    )
    if mae <= 0.10:
        c_clause = (
            f"The label frequency is recoverable: `c_hat` tracks the true confirmed fraction "
            f"to a mean absolute error of {mae:.3f} across the sweep, from labeled positives "
            "and unlabeled flows alone."
        )
    else:
        c_clause = (
            f"`c_hat` misses the true confirmed fraction by {mae:.3f} on average — the "
            "estimator inherits `g`'s calibration (Elkan & Noto's stated dependence), and an "
            "uncalibrated gradient-boosted `g` pays for it here; the same failure axis the "
            "[label-shift](label_shift.md) study found for MLLS vs BBSE."
        )
    signed = float(np.mean([p.prevalence_hat - study.train_prevalence for p in study.points]))
    if abs(signed) <= 0.05:
        prev_clause = (
            f" The implied prevalence lands within {prev_err:.3f} of the true training attack "
            f"rate ({study.train_prevalence:.3f}) on average — a zero-extra-label answer to "
            '"how many attacks are hiding in the unlabeled mass?".'
        )
    else:
        prev_dir = "over" if signed > 0 else "under"
        c_dir = "under" if signed > 0 else "over"  # opposite: prevalence = E[g] / c_hat
        prev_clause = (
            f" The prevalence estimate `E[g]/c` inherits that same bias — it {prev_dir}shoots "
            f"the true training attack rate ({study.train_prevalence:.3f}) by {abs(signed):.3f} "
            f"on average, in the direction the {c_dir}estimated `c_hat` forces (dividing "
            f"`E[g]` by too {'small' if signed > 0 else 'large'} a constant). The point "
            "estimates that divide by `c_hat` therefore read as an order-of-magnitude sanity "
            "check on the hidden-attack mass, not a calibrated prior — while the *ranking* "
            "product below (the weighted retrain) survives the same `c_hat` error, because it "
            "never divides by it as sharply."
        )
    return c_clause + prev_clause


def _detection_read(study: PUStudy) -> str:
    headline = next(p for p in study.points if p.label_frac == study.headline_frac)
    gap = study.oracle_pr_auc - headline.naive_pr_auc
    delta = headline.weighted_pr_auc - headline.naive_pr_auc
    base = (
        f"At the headline fraction (c = {study.headline_frac:.2f}, "
        f"{headline.n_labeled:,} confirmed attacks), the naive model lands PR-AUC "
        f"{headline.naive_pr_auc:.3f} against the oracle's {study.oracle_pr_auc:.3f} — "
        f"{'a ' + format(gap, '.3f') + ' cost of' if gap > 0.005 else 'essentially no cost from'} "
        "training with the missed attacks planted on the benign side. "
    )
    if delta > 0.005:
        return base + (
            f"The Elkan-Noto weighted retrain recovers {delta:.3f} of it "
            f"(PR-AUC {headline.weighted_pr_auc:.3f}) by letting each unlabeled flow be "
            "partly positive instead of definitely benign."
        )
    if delta < -0.005:
        return base + (
            f"The weighted retrain *loses* {-delta:.3f} here (PR-AUC "
            f"{headline.weighted_pr_auc:.3f}): its weights inherit `g`'s calibration error "
            "twice (once in `c_hat`, once in the odds ratio), and on this stand-in that "
            "noise outweighs the contamination it removes — reported as it fell."
        )
    return base + (
        f"The weighted retrain lands level (PR-AUC {headline.weighted_pr_auc:.3f}): on this "
        "stand-in the gradient-boosted learner already tolerates the contamination rate, so "
        "the correction has little to buy — reported plainly."
    )


def _budget_read(study: PUStudy) -> str:
    b = study.budget
    inflation = b.apparent_fpr / max(b.naive_fpr, 1e-9)
    tpr_gain = b.pu_tpr - b.naive_tpr
    lead = (
        f"At the {b.budget:.1%} budget the naive bookkeeping believes its cut costs "
        f"{b.apparent_fpr:.4f} FPR, but the hidden attacks in its 'benign' denominator are "
        f"doing part of the scoring: against *true* benign traffic the same cut realizes "
        f"{b.naive_fpr:.4f}"
    )
    overshoots = b.pu_fpr > b.budget * 1.3
    if inflation > 1.5 and tpr_gain > 0.01:
        core = (
            f" — over-tightened {inflation:.1f}x, silently spending detection: it alerts on "
            f"just {b.naive_tpr:.1%} of attacks where the oracle cut, at the same true budget, "
            f"reaches {b.oracle_tpr:.1%}. That waste is the headline: a contaminated denominator "
            "makes a SOC believe it has spent a budget it has barely touched. Re-pricing the "
            "denominator with the estimated benign mass (same model, same scores) fixes the "
            f"direction — detection climbs to {b.pu_tpr:.1%}"
        )
        if overshoots:
            return (
                lead
                + core
                + (
                    f", but *overshoots* the budget to {b.pu_fpr:.4f} realized FPR "
                    f"({b.pu_fpr / b.budget:.1f}x over), because the same underestimated `c_hat` "
                    "understates the benign mass and relaxes the cut too far. The correction "
                    "moves the operating point the right way; landing it *on* budget needs the "
                    "calibrated `g` (the isotonic step the calibration module already ships) this "
                    "deliberately withholds to keep `c` estimable — the honest limit, named."
                )
            )
        return (
            lead
            + core
            + (
                f" at {b.pu_fpr:.4f} realized FPR, landing near the oracle cut's "
                f"{b.oracle_tpr:.1%} within the budget."
            )
        )
    return lead + (
        f". On this stand-in the distortion is modest (PU-corrected cut: {b.pu_tpr:.1%} TPR "
        f"at {b.pu_fpr:.4f} FPR; oracle cut: {b.oracle_tpr:.1%} at {b.oracle_fpr:.4f}) — "
        "the contamination is too thin at this prevalence to bend the quantile far, "
        "reported as it fell; the mechanism is asserted on a constructed stream in the "
        "unit tests."
    )


def _render(study: PUStudy, fig: Path) -> str:
    return f"""# NetSentry — Positive-Unlabeled Learning (the labels a SOC actually has)

_Synthetic stand-in. Honest temporal/binary split: {study.n_train:,} training flows
(attack prevalence {study.train_prevalence:.3f}), {study.n_test:,} test flows. Confirmed
labels are drawn SCAR from the attacks; everything else is unlabeled, never "benign"._

## Why this report exists

Every supervised number in this suite assumes someone verified the benign side of the
training pool. A real deployment has confirmed attacks (incident tickets) and an unlabeled
remainder that *contains the attacks nobody caught*. Treating that remainder as benign is
what a team does implicitly; PU learning (Elkan & Noto, KDD 2008) does it explicitly: under
SCAR, the labeled-vs-unlabeled classifier `g` relates to the true posterior through one
estimable constant `c = p(labeled | attack)`, which buys corrected scores, a hidden-attack
prevalence estimate, a principled weighted retrain, and a de-contaminated FPR denominator.

## The sweep: estimator recovery and detection

{_sweep_table(study)}

{_estimator_read(study)}

![Detection vs confirmed fraction](../figures/{fig.name})

{_detection_read(study)}

Ranking note, stated plainly: the Elkan-Noto *score correction* `g/c` is monotone, so it
cannot move PR-AUC by construction — the correction buys calibration (thresholds,
prevalence, costs), and only the *weighted retrain* can move ranking.

## The operating point: three cuts on one model

Thresholds chosen on validation at the {study.budget.budget:.1%} FPR budget; realized on the
temporal test split against true benign traffic. Apparent FPR under naive bookkeeping:
{study.budget.apparent_fpr:.4f}.

{_budget_table(study.budget)}

{_budget_read(study)}

## Scope

SCAR is an assumption, stated: confirmed labels arrive independently of the flow's features.
Real triage is biased toward the obvious (the loud DDoS gets a ticket, the quiet
exfiltration does not), which violates SCAR in the direction of overestimating `c` on the
easy attacks — the SAR generalisation is the named next step. `c_hat` inherits `g`'s
calibration, and this report audits rather than assumes it. The oracle column is the ceiling
this suite reports everywhere else; the honest reading of this study is the *gap* a
confirmed-only labeling regime opens, and how much of it the PU machinery closes with zero
additional labels."""
