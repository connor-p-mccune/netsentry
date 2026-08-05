"""Predictive multiplicity: how arbitrary is the verdict, given equally-good models?

Every number this project reports describes **one** model. But the training protocol has
free choices that no metric adjudicates — the seed, the row subsample, the column
subsample, the leaf count, the learning rate. Vary them and you get a family of models
that are, on the honest temporal split, *statistically indistinguishable* from the
deployed one. Leo Breiman called this the Rashomon effect: many equally-accurate stories
about the same data. Marx, Calmon & Ustun (*Predictive Multiplicity in Classification*,
ICML 2020) made it measurable, and the measurement is uncomfortable — because a flow
whose verdict flips across that family was never really decided by evidence. It was
decided by a coin-flip the engineer made months earlier.

Two quantities, both defined over the **Rashomon set** (the models within `epsilon` of
the deployed model's PR-AUC, all judged at the same validated false-positive budget so
nobody wins by simply alerting more):

- **Ambiguity** — the share of flows that *some* competing model in the set decides
  differently from the deployed model. This is the population of verdicts that are not
  robust to an arbitrary modelling choice.
- **Discrepancy** — the largest share of flows that a *single* competing model decides
  differently. Ambiguity is a union over models; discrepancy is the worst individual
  disagreement, and it bounds how much a deployment swap could churn the alert queue.

The report does three things beyond reciting those numbers. It sweeps `epsilon`, because
multiplicity is only meaningful relative to how much performance you would trade. It
asks **where** the arbitrariness lands — a system whose ambiguity concentrates on the
alerts it raises is far more troubling than one whose ambiguity sits in the benign bulk
nobody looks at. And it turns the measurement into a lever: a per-flow **vote fraction**
across the Rashomon set gives a three-way auto-alert / review / auto-clear policy that
routes the genuinely contested flows to a human, and prices that review load against the
ambiguity it removes.

This is the decision-level counterpart to the [importance-stability](importance_stability.md)
study, which asks whether the *explanations* are stable across refits. A stable
explanation of an arbitrary verdict is not much comfort.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import MultiplicityConfig

logger = get_logger(__name__)

REPORT_NAME = "multiplicity.md"
FIGURE_NAME = "multiplicity.png"
SWEEP_FIGURE_NAME = "multiplicity_sweep.png"


# --------------------------------------------------------------------------------------
# The measures (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
def rashomon_mask(pr_aucs: np.ndarray, champion_pr_auc: float, epsilon: float) -> np.ndarray:
    """Which competitors are near-optimal: PR-AUC within a *relative* ``epsilon`` of champion.

    Relative rather than absolute because PR-AUC's scale is set by the base rate — a
    0.01 absolute slack means something very different at 0.53 than at 0.99. A competitor
    that *beats* the champion is in the set too: it is certainly near-optimal, and
    excluding it would understate the multiplicity.
    """
    aucs = np.asarray(pr_aucs, dtype=float)
    floor = champion_pr_auc * (1.0 - epsilon)
    return np.asarray(aucs >= floor, dtype=bool)


def ambiguity(champion: np.ndarray, competitors: np.ndarray) -> float:
    """Share of flows that **at least one** near-optimal model decides differently.

    ``competitors`` is (n_models, n_flows) of 0/1 decisions; ``champion`` is (n_flows,).
    Ambiguity is a union over the Rashomon set, so it grows with the set and is the
    honest count of verdicts that a defensible alternative model would overturn.
    """
    champ = np.asarray(champion, dtype=int)
    comp = np.atleast_2d(np.asarray(competitors, dtype=int))
    if comp.size == 0:
        return 0.0
    differs = comp != champ[None, :]
    return float(np.mean(differs.any(axis=0)))


def discrepancy(champion: np.ndarray, competitors: np.ndarray) -> float:
    """Largest share of flows that a **single** near-optimal model decides differently.

    The worst-case churn of swapping the deployed model for one specific competitor — the
    quantity an operator feels, because deployments happen one model at a time.
    """
    champ = np.asarray(champion, dtype=int)
    comp = np.atleast_2d(np.asarray(competitors, dtype=int))
    if comp.size == 0:
        return 0.0
    return float(np.max(np.mean(comp != champ[None, :], axis=1)))


def vote_fraction(decisions: np.ndarray) -> np.ndarray:
    """Per-flow share of the Rashomon set voting *attack* — the multiplicity score.

    0 and 1 are unanimous (the verdict is a property of the data); values near 0.5 are
    flows the model family genuinely cannot agree on.
    """
    dec = np.atleast_2d(np.asarray(decisions, dtype=float))
    if dec.size == 0:
        return np.zeros(0, dtype=float)
    return np.asarray(dec.mean(axis=0), dtype=float)


@dataclass
class BandOutcome:
    """A three-way auto-alert / review / auto-clear routing at one review band."""

    lo: float
    hi: float
    review_rate: float
    auto_alert_rate: float
    residual_ambiguity: float
    auto_alert_precision: float
    auto_alert_recall: float
    attacks_routed_to_review: float


def route_by_vote(
    votes: np.ndarray,
    is_attack: np.ndarray,
    contested: np.ndarray,
    lo: float,
    hi: float,
) -> BandOutcome:
    """Route flows by their vote fraction and price the resulting review load.

    ``contested`` marks the flows the Rashomon set disagrees on at all. Residual ambiguity
    is measured over the **auto-decided** flows only — the whole point of an abstention
    band is to stop making arbitrary calls, so what matters is how much arbitrariness
    survives in the decisions still taken automatically.
    """
    votes = np.asarray(votes, dtype=float)
    attack = np.asarray(is_attack, dtype=bool)
    contested = np.asarray(contested, dtype=bool)
    n = len(votes)
    if n == 0:
        return BandOutcome(lo, hi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    review = (votes >= lo) & (votes <= hi)
    auto_alert = votes > hi
    auto = ~review
    residual = float(contested[auto].mean()) if auto.any() else 0.0
    n_alert = int(auto_alert.sum())
    n_attack = int(attack.sum())
    return BandOutcome(
        lo=lo,
        hi=hi,
        review_rate=float(review.mean()),
        auto_alert_rate=float(auto_alert.mean()),
        residual_ambiguity=residual,
        auto_alert_precision=float(attack[auto_alert].mean()) if n_alert else 0.0,
        auto_alert_recall=float((auto_alert & attack).sum() / n_attack) if n_attack else 0.0,
        attacks_routed_to_review=float((review & attack).sum() / n_attack) if n_attack else 0.0,
    )


# --------------------------------------------------------------------------------------
# The Rashomon pool
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateSpec:
    """One draw from the space of modelling choices no metric adjudicates."""

    seed: int
    subsample: float
    colsample_bytree: float
    num_leaves: int
    learning_rate: float

    def label(self) -> str:
        """Compact description for the report's pool table."""
        return (
            f"seed={self.seed}, rows={self.subsample:.2f}, cols={self.colsample_bytree:.2f}, "
            f"leaves={self.num_leaves}, lr={self.learning_rate:.3f}"
        )


@dataclass
class Candidate:
    """One trained near-optimal candidate and the modelling choice that produced it."""

    name: str
    spec: str
    pr_auc: float
    threshold: float
    decisions: np.ndarray
    scores: np.ndarray


def candidate_specs(cfg: MultiplicityConfig, seed: int) -> list[CandidateSpec]:
    """Sample the free modelling choices — the ones no metric adjudicates.

    Every knob here is a decision a competent engineer makes without a principled reason:
    which seed, how much of the data to bag, how many leaves, how fast to learn. Drawing
    them from a *plausible* neighbourhood (not a wild grid) is what makes the resulting
    disagreement an indictment rather than a strawman.
    """
    rng = np.random.default_rng(seed)
    return [
        CandidateSpec(
            seed=seed + 1 + i,
            subsample=float(rng.choice(cfg.subsample_choices)),
            colsample_bytree=float(rng.choice(cfg.colsample_choices)),
            num_leaves=int(rng.choice(cfg.num_leaves_choices)),
            learning_rate=float(rng.choice(cfg.learning_rate_choices)),
        )
        for i in range(cfg.n_models)
    ]


def _fit_candidate(
    settings: Settings,
    spec: CandidateSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    benign: str,
    target_fpr: float,
    name: str,
) -> Candidate:
    """Train one candidate and decide the test set at its own validated FPR budget.

    The threshold is re-chosen per candidate on validation. Without that, a candidate that
    simply scores higher on average would look like it "disagrees" everywhere, and the
    ambiguity would be measuring calibration drift instead of genuine multiplicity.
    """
    variant = settings.model_copy(deep=True)
    variant.seed = spec.seed
    variant.supervised.subsample = spec.subsample
    variant.supervised.colsample_bytree = spec.colsample_bytree
    variant.supervised.num_leaves = spec.num_leaves
    variant.supervised.learning_rate = spec.learning_rate
    seed_everything(variant.seed)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, target_fpr)
    return Candidate(
        name=name,
        spec=spec.label(),
        pr_auc=float(average_precision_score(y_test, s_test)),
        threshold=threshold,
        decisions=np.asarray(s_test >= threshold, dtype=int),
        scores=s_test,
    )


@dataclass
class SweepPoint:
    """Ambiguity/discrepancy at one Rashomon tolerance."""

    epsilon: float
    n_in_set: int
    ambiguity: float
    discrepancy: float


@dataclass
class MultiplicityStudy:
    """Everything the report renders."""

    champion: Candidate
    candidates: list[Candidate]
    epsilon: float
    n_in_set: int
    n_test: int
    target_fpr: float
    ambiguity: float
    discrepancy: float
    ambiguity_on_alerts: float
    ambiguity_on_clears: float
    ambiguity_on_attacks: float
    ambiguity_on_benign: float
    alert_rate: float
    sweep: list[SweepPoint]
    bands: list[BandOutcome]
    votes: np.ndarray
    contested: np.ndarray
    champion_alerts: np.ndarray


def run_multiplicity(settings: Settings) -> MultiplicityStudy:
    """Train a Rashomon pool on the honest split and measure how arbitrary the verdicts are."""
    cfg: MultiplicityConfig = settings.multiplicity
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
    target_fpr = variant.thresholds.primary_fpr

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    champion = _fit_candidate(
        variant,
        CandidateSpec(
            seed=variant.seed,
            subsample=variant.supervised.subsample,
            colsample_bytree=variant.supervised.colsample_bytree,
            num_leaves=variant.supervised.num_leaves,
            learning_rate=variant.supervised.learning_rate,
        ),
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        benign,
        target_fpr,
        name="deployed",
    )
    logger.info("Champion trained", extra={"pr_auc": round(champion.pr_auc, 4)})

    candidates: list[Candidate] = []
    for i, spec in enumerate(candidate_specs(cfg, variant.seed)):
        cand = _fit_candidate(
            variant,
            spec,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            benign,
            target_fpr,
            name=f"alt-{i + 1:02d}",
        )
        candidates.append(cand)
        logger.info(
            "Rashomon candidate trained",
            extra={"candidate": cand.name, "pr_auc": round(cand.pr_auc, 4)},
        )

    aucs = np.array([c.pr_auc for c in candidates])
    decisions = np.vstack([c.decisions for c in candidates]) if candidates else np.zeros((0, 0))

    in_set = rashomon_mask(aucs, champion.pr_auc, cfg.epsilon)
    member_decisions = decisions[in_set] if decisions.size else decisions
    amb = ambiguity(champion.decisions, member_decisions)
    disc = discrepancy(champion.decisions, member_decisions)

    # The whole Rashomon set, champion included, votes on each flow.
    all_decisions = (
        np.vstack([champion.decisions[None, :], member_decisions])
        if member_decisions.size
        else champion.decisions[None, :]
    )
    votes = vote_fraction(all_decisions)
    contested = (
        np.asarray((member_decisions != champion.decisions[None, :]).any(axis=0), dtype=bool)
        if member_decisions.size
        else np.zeros(len(champion.decisions), dtype=bool)
    )

    alerts = champion.decisions.astype(bool)
    attacks = y_test.astype(bool)
    sweep = []
    for eps in cfg.epsilon_sweep:
        mask = rashomon_mask(aucs, champion.pr_auc, eps)
        members = decisions[mask] if decisions.size else decisions
        sweep.append(
            SweepPoint(
                epsilon=eps,
                n_in_set=int(mask.sum()),
                ambiguity=ambiguity(champion.decisions, members),
                discrepancy=discrepancy(champion.decisions, members),
            )
        )

    bands = [route_by_vote(votes, attacks, contested, lo, hi) for lo, hi in cfg.review_bands]

    return MultiplicityStudy(
        champion=champion,
        candidates=candidates,
        epsilon=cfg.epsilon,
        n_in_set=int(in_set.sum()),
        n_test=len(y_test),
        target_fpr=target_fpr,
        ambiguity=amb,
        discrepancy=disc,
        ambiguity_on_alerts=float(contested[alerts].mean()) if alerts.any() else 0.0,
        ambiguity_on_clears=float(contested[~alerts].mean()) if (~alerts).any() else 0.0,
        ambiguity_on_attacks=float(contested[attacks].mean()) if attacks.any() else 0.0,
        ambiguity_on_benign=float(contested[~attacks].mean()) if (~attacks).any() else 0.0,
        alert_rate=float(alerts.mean()),
        sweep=sweep,
        bands=bands,
        votes=votes,
        contested=contested,
        champion_alerts=alerts,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_multiplicity_report(settings: Settings) -> Path:
    """Run the multiplicity study and write the report + figures."""
    study = run_multiplicity(settings)

    hist = plots.plot_hist_overlay(
        {
            "champion alerted": study.votes[study.champion_alerts],
            "champion cleared": study.votes[~study.champion_alerts],
        },
        xlabel="share of the Rashomon set voting attack",
        title="Unanimous or contested? The multiplicity score of every test flow",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        bins=25,
    )
    eps = np.array([p.epsilon for p in study.sweep])
    sweep_fig = plots.plot_lines(
        {
            "ambiguity (any near-optimal model flips it)": (
                eps,
                np.array([p.ambiguity for p in study.sweep]),
            ),
            "discrepancy (the worst single model)": (
                eps,
                np.array([p.discrepancy for p in study.sweep]),
            ),
        },
        xlabel="Rashomon tolerance epsilon (relative PR-AUC slack)",
        ylabel="share of test flows",
        title="How much arbitrariness a little performance slack buys",
        out_path=settings.paths.figures_dir / SWEEP_FIGURE_NAME,
    )

    report = _render(study, hist, sweep_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote multiplicity report", extra={"path": str(out_path)})

    with track_run(settings, "multiplicity") as run:
        run.log_params({"epsilon": study.epsilon, "n_candidates": len(study.candidates)})
        run.log_metrics(
            {
                "champion_pr_auc": study.champion.pr_auc,
                "n_in_rashomon_set": float(study.n_in_set),
                "ambiguity": study.ambiguity,
                "discrepancy": study.discrepancy,
                "ambiguity_on_alerts": study.ambiguity_on_alerts,
            }
        )
        run.log_artifact(hist)
        run.log_artifact(sweep_fig)
        run.log_artifact(out_path)
    return out_path


def _pool_table(study: MultiplicityStudy) -> str:
    rows = [
        "| model | modelling choice | PR-AUC | delta | in Rashomon set |",
        "|---|---|---|---|---|",
    ]
    rows.append(
        f"| **deployed** | {study.champion.spec} | **{study.champion.pr_auc:.3f}** | — | anchor |"
    )
    floor = study.champion.pr_auc * (1.0 - study.epsilon)
    for cand in study.candidates:
        delta = cand.pr_auc - study.champion.pr_auc
        mark = "yes" if cand.pr_auc >= floor else "no"
        rows.append(f"| {cand.name} | {cand.spec} | {cand.pr_auc:.3f} | {delta:+.3f} | {mark} |")
    return "\n".join(rows)


def _measure_table(study: MultiplicityStudy) -> str:
    rows = [
        "| measure | value | reading |",
        "|---|---|---|",
        f"| **ambiguity** | {study.ambiguity:.2%} | flows some near-optimal model flips |",
        f"| **discrepancy** | {study.discrepancy:.2%} | flows the worst single model flips |",
        f"| ambiguity among the alerts raised | {study.ambiguity_on_alerts:.2%} "
        f"| at a {study.alert_rate:.2%} alert rate |",
        f"| ambiguity among the flows cleared | {study.ambiguity_on_clears:.2%} | |",
        f"| ambiguity among true attacks | {study.ambiguity_on_attacks:.2%} | |",
        f"| ambiguity among true benign flows | {study.ambiguity_on_benign:.2%} | |",
    ]
    return "\n".join(rows)


def _sweep_table(study: MultiplicityStudy) -> str:
    rows = ["| epsilon | models in set | ambiguity | discrepancy |", "|---|---|---|---|"]
    for p in study.sweep:
        rows.append(f"| {p.epsilon:.2%} | {p.n_in_set} | {p.ambiguity:.2%} | {p.discrepancy:.2%} |")
    return "\n".join(rows)


def _band_table(study: MultiplicityStudy) -> str:
    rows = [
        "| review band | sent to review | residual ambiguity | auto-alert rate "
        "| auto-alert precision | auto-alert recall | attacks in review |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in study.bands:
        rows.append(
            f"| [{b.lo:.2f}, {b.hi:.2f}] | {b.review_rate:.2%} | {b.residual_ambiguity:.2%} "
            f"| {b.auto_alert_rate:.2%} | {b.auto_alert_precision:.1%} "
            f"| {b.auto_alert_recall:.1%} | {b.attacks_routed_to_review:.1%} |"
        )
    return "\n".join(rows)


def _headline_read(study: MultiplicityStudy) -> str:
    ratio = study.ambiguity_on_alerts / max(study.ambiguity_on_clears, 1e-9)
    if study.ambiguity == 0.0:
        return (
            f"Every one of the {study.n_in_set} near-optimal alternatives reproduces the deployed "
            "model's verdict on every test flow. That is the best available answer to the "
            "Rashomon objection: at this operating point the decisions are a property of the "
            "data, not of the modelling choices — no defensible alternative would overturn any "
            "of them."
        )
    concentration = (
        f"and it is **concentrated on the alerts**: {study.ambiguity_on_alerts:.1%} of the flows "
        f"the deployed model flags are contested, against {study.ambiguity_on_clears:.2%} of the "
        f"flows it clears — roughly {ratio:.0f}x. That is the uncomfortable direction. Ambiguity "
        "sitting in the benign bulk would be academic; ambiguity sitting in the queue means the "
        "analyst's work is partly determined by a seed."
        if ratio > 2
        else (
            f"and it is spread fairly evenly: {study.ambiguity_on_alerts:.1%} of flagged flows "
            f"against {study.ambiguity_on_clears:.2%} of cleared ones."
        )
    )
    return (
        f"{study.ambiguity:.2%} of test flows are **ambiguous** — at least one of the "
        f"{study.n_in_set} models that match the deployed model's PR-AUC to within "
        f"{study.epsilon:.0%} decides them differently, at the same validated "
        f"{study.target_fpr:.1%} false-positive budget. The worst single competitor differs on "
        f"{study.discrepancy:.2%} of flows (**discrepancy**), which is what a deployment swap "
        f"would actually churn. Ambiguity is small in absolute terms {concentration}"
    )


def _sweep_read(study: MultiplicityStudy) -> str:
    tightest = min(study.sweep, key=lambda p: p.epsilon)
    loosest = max(study.sweep, key=lambda p: p.epsilon)
    return (
        f"Multiplicity is only meaningful relative to how much performance you would trade for a "
        f"different answer. At a {tightest.epsilon:.0%} tolerance the set holds "
        f"{tightest.n_in_set} models and {tightest.ambiguity:.2%} of flows are contested; widen it "
        f"to {loosest.epsilon:.0%} and the set grows to {loosest.n_in_set} models and "
        f"{loosest.ambiguity:.2%}. The curve is monotone by construction — a wider set can only "
        "add disagreement — so the honest reading is the *shape*: a steep climb means the "
        "performance you would sacrifice buys a lot of freedom to re-decide individual flows, "
        "which is precisely the situation in which citing a single model's verdict as fact is "
        "indefensible."
    )


def _band_read(study: MultiplicityStudy) -> str:
    if not study.bands:
        return ""
    cheapest = min(study.bands, key=lambda b: b.review_rate)
    thorough = min(study.bands, key=lambda b: (b.residual_ambiguity, b.review_rate))
    return (
        "Measuring arbitrariness is only useful if it can be acted on. The vote fraction is a "
        "ready-made abstention signal: auto-decide the flows the family agrees on, route the "
        f"contested band to a human. At the strictest setting ([{thorough.lo:.2f}, "
        f"{thorough.hi:.2f}] — abstain on *any* disagreement) {thorough.review_rate:.2%} of the "
        f"stream goes to review, {thorough.residual_ambiguity:.2%} arbitrariness survives in the "
        f"decisions still taken automatically, and the auto-alerts that remain run at "
        f"{thorough.auto_alert_precision:.1%} precision — a queue with nothing arbitrary in it. "
        f"The cheap setting ([{cheapest.lo:.2f}, {cheapest.hi:.2f}]) reviews only the "
        f"near-tied flows: {cheapest.review_rate:.2%} of the stream, leaving "
        f"{cheapest.residual_ambiguity:.2%} residual. Either way the cost lands where it should — "
        f"{thorough.attacks_routed_to_review:.1%} of true attacks are routed to review rather "
        "than auto-alerted, which is not a miss: a human still sees them. This is the same trade "
        "the [conformal](conformal.md) study makes with a coverage guarantee, arrived at from the "
        "opposite direction — conformal abstains when the *data* is ambiguous, this abstains when "
        "the *model family* is."
    )


def _render(study: MultiplicityStudy, hist: Path, sweep_fig: Path) -> str:
    return f"""# NetSentry — Predictive Multiplicity: How Arbitrary Is the Verdict?

_Synthetic stand-in. Honest temporal/binary split, {study.n_test:,} test flows.
{len(study.candidates)} alternative models trained by varying only the modelling choices no
metric adjudicates (seed, row/column subsample, leaf count, learning rate); every model
decides at its own validation-calibrated {study.target_fpr:.1%}-FPR threshold, so no model
can win the comparison by simply alerting more._

## Why this report exists

Every other number in this project describes **one** model. But the training protocol
contains free choices — the seed, how much data to bag, how many leaves, how fast to learn —
and no metric adjudicates between them. Vary them and you get a family of models that are
statistically indistinguishable on the honest split. Breiman called it the Rashomon effect;
Marx, Calmon & Ustun (ICML 2020) made it measurable. The measurement matters because a flow
whose verdict flips across that family was not decided by evidence — it was decided by an
arbitrary choice made months earlier, and an analyst acting on that alert deserves to know.

## The Rashomon set

The set is every candidate whose test PR-AUC is within **{study.epsilon:.0%}** (relative) of
the deployed model's. {study.n_in_set} of {len(study.candidates)} candidates qualify.

{_pool_table(study)}

## Ambiguity and discrepancy

{_measure_table(study)}

{_headline_read(study)}

![multiplicity score distribution](../figures/{hist.name})

## How much slack buys how much freedom

{_sweep_table(study)}

![ambiguity vs epsilon](../figures/{sweep_fig.name})

{_sweep_read(study)}

## Turning the measurement into a lever

{_band_table(study)}

{_band_read(study)}

## Scope

Multiplicity is measured over a *sampled* Rashomon set, not the true one — the real set is
every near-optimal model in the hypothesis class, which is not enumerable, so
{len(study.candidates)} draws from a plausible neighbourhood of the deployed configuration
give a **lower bound** on ambiguity and discrepancy. Both are reported on the honest
temporal split at the deployed operating point; a different false-positive budget moves
every threshold and therefore the disagreement set. The candidates share one feature
pipeline and one training split, so this measures multiplicity from *modelling* choices
only — data-collection multiplicity (which days, which capture) would be measured by
re-running the [leave-one-day-out](lodo.md) study across model families, and the
[seed-variance](seed_variance.md) report already isolates the seed's own contribution as
the training-noise floor beneath every metric here. The decision-level companion to
[importance stability](importance_stability.md): stable explanations of an arbitrary verdict
are not much comfort."""
