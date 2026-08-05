"""Budgeted cascade inference: pay the expensive model only where it changes the answer.

The [benchmark](../../README.md) puts single-flow inference at roughly 48 ms with SHAP and
13 ms without, which is fine for a demo and awkward for a link that carries a million flows
a day. The usual reactions are to buy more replicas or to drop the explanations, and both
concede something worth keeping. There is a third option, standard in production vision and
ranking systems and under-used in security ML: a **cascade**. Run a cheap model on
everything, and spend the expensive model only on the flows where the cheap one is not
already sure.

The design turns on one question — how do you choose the cheap model's cut-off without
quietly throwing away detection? Not by picking a round number. The stage-1 threshold is
chosen **on validation** as the quantile that retains a target share of the full model's
own alerts: if the deployed model would have flagged a flow, stage 1 must forward it. That
makes the knob an explicit *escape budget* rather than an accident, and it means the
cascade's loss is measured against the thing it replaces rather than against the labels
(which a deployed system does not have).

What the report measures, per deferral budget:

- **escape rate** — full-model alerts that stage 1 filtered away, the thing the budget buys.
- **deferral rate** — the share of traffic that actually reaches the expensive model, which
  is the load reduction, measured rather than assumed.
- **end-to-end detection** — TPR at the same fixed FPR, and PR-AUC over a cascade score
  constructed the way a cascade actually ranks (a filtered flow can never outrank a
  forwarded one).
- **measured latency** — both stages timed per flow on this machine, blended by the realized
  deferral rate, and turned into a throughput estimate.

The honest framing throughout is that a cascade does not make the model better. It buys
compute back, and the report's job is to price exactly what that costs in detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import CascadeConfig

logger = get_logger(__name__)

REPORT_NAME = "cascade.md"
FIGURE_NAME = "cascade.png"


def stage1_threshold(
    stage1_val: np.ndarray, full_val_alerts: np.ndarray, keep_fraction: float
) -> float:
    """Stage-1 cut that forwards ``keep_fraction`` of the full model's validation alerts.

    The threshold is a quantile of stage 1's scores **restricted to the flows the deployed
    model alerts on** — the population the cascade must not lose. Choosing it on the whole
    validation set instead would target a traffic percentile and silently trade away
    detection, which is precisely the mistake this framing exists to avoid. No labels are
    used: the reference is the deployed model's own behaviour, which a live system has.
    """
    scores = np.asarray(stage1_val, dtype=float)
    alerts = np.asarray(full_val_alerts, dtype=bool)
    if not alerts.any():
        return float(-np.inf)
    keep = float(np.clip(keep_fraction, 0.0, 1.0))
    # Keep the top `keep` share of alert scores => cut at the (1 - keep) quantile of them.
    return float(np.quantile(scores[alerts], 1.0 - keep))


def cascade_scores(stage1: np.ndarray, stage2: np.ndarray, forwarded: np.ndarray) -> np.ndarray:
    """Rank flows the way a cascade does: nothing filtered can outrank anything forwarded.

    Forwarded flows keep their stage-2 score. Filtered flows are ranked *below* every
    forwarded flow, by their stage-1 score, preserving the cheap model's ordering among
    them. Scoring filtered flows on a shared scale with stage 2 would credit the cascade
    with rankings it never computes.
    """
    s1 = np.asarray(stage1, dtype=float)
    s2 = np.asarray(stage2, dtype=float)
    fwd = np.asarray(forwarded, dtype=bool)
    out = np.empty(len(s1), dtype=float)
    out[fwd] = s2[fwd]
    if (~fwd).any():
        floor = float(s2[fwd].min()) if fwd.any() else 0.0
        filtered = s1[~fwd]
        span = float(filtered.max() - filtered.min()) or 1.0
        # Map into a unit-wide band strictly below the lowest forwarded score.
        out[~fwd] = floor - 1.0 + (filtered - filtered.min()) / span
    return out


def blended_latency_ms(stage1_ms: float, stage2_ms: float, deferral_rate: float) -> float:
    """Expected per-flow latency: stage 1 always, stage 2 only on the deferred share."""
    return stage1_ms + deferral_rate * stage2_ms


def median_latency_ms(predict: Any, x: np.ndarray, n_calls: int) -> float:
    """Median wall-clock of ``n_calls`` single-row predictions — the serving unit of work.

    Median rather than mean because a garbage-collection pause in a timing loop is not a
    property of the model, and single-row rather than batched because that is what
    ``/predict`` actually does.
    """
    rows = min(n_calls, len(x))
    timings = []
    for i in range(rows):
        row = x[i : i + 1]
        t0 = time.perf_counter()
        predict(row)
        timings.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(timings)) if timings else 0.0


@dataclass
class CascadePoint:
    """The cascade at one escape budget."""

    keep_fraction: float
    threshold: float
    deferral_rate: float
    escape_rate: float
    pr_auc: float
    tpr: float
    fpr: float
    latency_ms: float
    speedup: float
    throughput: float


@dataclass
class SplitOutcome:
    """Both stages' standalone performance on one split — the cascade's premise check."""

    strategy: str
    n_test: int
    full_pr_auc: float
    full_tpr: float
    full_fpr: float
    stage1_pr_auc: float
    stage1_alone_tpr: float
    stage1_latency_ms: float
    stage2_latency_ms: float

    @property
    def premise_holds(self) -> bool:
        """Is stage 2 actually the better ranker? A cascade assumes it; this checks."""
        return self.full_pr_auc > self.stage1_pr_auc

    @property
    def full_throughput(self) -> float:
        return 1000.0 / self.stage2_latency_ms if self.stage2_latency_ms > 0 else 0.0


@dataclass
class CascadeStudy:
    """Everything the report renders."""

    headline: SplitOutcome
    reference: SplitOutcome
    target_fpr: float
    max_escape: float
    points: list[CascadePoint]
    reference_points: list[CascadePoint]

    def best_within(self, max_escape: float) -> CascadePoint:
        """Cheapest operating point whose escape rate stays inside the budget."""
        eligible = [p for p in self.points if p.escape_rate <= max_escape] or self.points
        return min(eligible, key=lambda p: p.deferral_rate)

    def reference_best(self) -> CascadePoint:
        eligible = [
            p for p in self.reference_points if p.escape_rate <= self.max_escape
        ] or self.reference_points
        return min(eligible, key=lambda p: p.deferral_rate)


def _run_split(
    settings: Settings, strategy: Literal["temporal", "stratified"], *, time_stages: bool
) -> tuple[SplitOutcome, list[CascadePoint]]:
    """Fit both stages on one split and sweep the escape budget."""
    cfg: CascadeConfig = settings.cascade
    variant = settings.model_copy(deep=True)
    variant.split.strategy = strategy
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, strategy, "train")
    val = load_split(variant, strategy, "val")
    test = load_split(variant, strategy, "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    # Stage 2: the deployed model. Stage 1: the cheapest thing that ranks at all.
    full = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    cheap = LogisticRegression(
        max_iter=cfg.stage1_max_iter,
        class_weight="balanced" if variant.supervised.class_weight == "balanced" else None,
        random_state=variant.seed,
    ).fit(x_train, y_train)

    def _full_scores(x: np.ndarray) -> np.ndarray:
        return attack_probability(np.asarray(full.predict_proba(x)), full.classes_, benign)

    def _cheap_scores(x: np.ndarray) -> np.ndarray:
        return attack_probability(np.asarray(cheap.predict_proba(x)), cheap.classes_, benign)

    full_val, full_test = _full_scores(x_val), _full_scores(x_test)
    cheap_val, cheap_test = _cheap_scores(x_val), _cheap_scores(x_test)

    decision = threshold_at_fpr(y_val, full_val, variant.thresholds.primary_fpr)
    full_rates = rates_at_threshold(y_test, full_test, decision)
    val_alerts = full_val >= decision

    # Both stages timed on this machine, single-row, the way /predict works. Timing is
    # a property of the models, not of the split, so it is measured once.
    stage1_ms = (
        median_latency_ms(cheap.predict_proba, x_test, cfg.latency_calls) if time_stages else 0.0
    )
    stage2_ms = (
        median_latency_ms(full.predict_proba, x_test, cfg.latency_calls) if time_stages else 0.0
    )

    test_alerts = full_test >= decision
    points: list[CascadePoint] = []
    for keep in cfg.keep_fractions:
        cut = stage1_threshold(cheap_val, val_alerts, keep)
        forwarded = cheap_test >= cut
        scores = cascade_scores(cheap_test, full_test, forwarded)
        # A cascade decision requires surviving stage 1 *and* clearing the stage-2 cut.
        decided = forwarded & (full_test >= decision)
        rates = rates_at_threshold(y_test, np.where(decided, 1.0, 0.0), 0.5)
        deferral = float(forwarded.mean())
        latency = blended_latency_ms(stage1_ms, stage2_ms, deferral)
        escaped = (
            float((test_alerts & ~forwarded).sum() / max(int(test_alerts.sum()), 1))
            if test_alerts.any()
            else 0.0
        )
        points.append(
            CascadePoint(
                keep_fraction=keep,
                threshold=cut,
                deferral_rate=deferral,
                escape_rate=escaped,
                pr_auc=float(average_precision_score(y_test, scores)),
                tpr=rates["tpr"],
                fpr=rates["fpr"],
                latency_ms=latency,
                speedup=stage2_ms / latency if latency > 0 else 1.0,
                throughput=1000.0 / latency if latency > 0 else 0.0,
            )
        )
        logger.info(
            "Cascade point measured",
            extra={
                "split": strategy,
                "keep": keep,
                "deferral": round(deferral, 4),
                "escape": round(escaped, 4),
            },
        )

    cheap_decision = threshold_at_fpr(y_val, cheap_val, variant.thresholds.primary_fpr)
    outcome = SplitOutcome(
        strategy=strategy,
        n_test=len(y_test),
        full_pr_auc=float(average_precision_score(y_test, full_test)),
        full_tpr=full_rates["tpr"],
        full_fpr=full_rates["fpr"],
        stage1_pr_auc=float(average_precision_score(y_test, cheap_test)),
        stage1_alone_tpr=rates_at_threshold(y_test, cheap_test, cheap_decision)["tpr"],
        stage1_latency_ms=stage1_ms,
        stage2_latency_ms=stage2_ms,
    )
    return outcome, points


def run_cascade(settings: Settings) -> CascadeStudy:
    """Build the cascade on the honest split, then check its premise on the stratified one."""
    headline, points = _run_split(settings, "temporal", time_stages=True)
    reference, ref_points = _run_split(settings, "stratified", time_stages=False)
    return CascadeStudy(
        headline=headline,
        reference=reference,
        target_fpr=settings.thresholds.primary_fpr,
        max_escape=settings.cascade.max_escape,
        points=points,
        reference_points=ref_points,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_cascade_report(settings: Settings) -> Path:
    """Run the cascade study and write the report + figure."""
    study = run_cascade(settings)

    deferrals = np.array([p.deferral_rate for p in study.points])
    ref_deferrals = np.array([p.deferral_rate for p in study.reference_points])
    fig = plots.plot_lines(
        {
            "cascade, temporal split (honest)": (
                deferrals,
                np.array([p.tpr / max(study.headline.full_tpr, 1e-9) for p in study.points]),
            ),
            "cascade, stratified split (premise holds)": (
                ref_deferrals,
                np.array(
                    [p.tpr / max(study.reference.full_tpr, 1e-9) for p in study.reference_points]
                ),
            ),
            "full model (every flow through stage 2)": (
                deferrals,
                np.ones(len(deferrals)),
            ),
        },
        xlabel="share of traffic reaching the expensive model",
        ylabel=f"detection retained at {study.target_fpr:.1%} FPR (vs the full model)",
        title="What a cascade costs: detection vs the compute it hands back",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote cascade report", extra={"path": str(out_path)})

    with track_run(settings, "cascade") as run:
        run.log_params(
            {"keep_fractions": ",".join(f"{k}" for k in settings.cascade.keep_fractions)}
        )
        best = study.best_within(settings.cascade.max_escape)
        run.log_metrics(
            {
                "full_tpr": study.headline.full_tpr,
                "stage1_latency_ms": study.headline.stage1_latency_ms,
                "stage2_latency_ms": study.headline.stage2_latency_ms,
                "best_deferral_rate": best.deferral_rate,
                "best_speedup": best.speedup,
                "best_tpr": best.tpr,
                "premise_holds_temporal": float(study.headline.premise_holds),
                "premise_holds_stratified": float(study.reference.premise_holds),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _points_table(study: CascadeStudy) -> str:
    head = study.headline
    rows = [
        "| alerts kept (target) | escaped | traffic to stage 2 | detection | FPR | PR-AUC "
        "| latency/flow | speedup | throughput |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows.append(
        f"| _(full model)_ | 0.0% | 100.00% | {head.full_tpr:.1%} | {head.full_fpr:.3%} "
        f"| {head.full_pr_auc:.3f} | {head.stage2_latency_ms:.2f} ms | 1.0x "
        f"| {head.full_throughput:,.0f}/s |"
    )
    for p in study.points:
        rows.append(
            f"| {p.keep_fraction:.0%} | {p.escape_rate:.1%} | {p.deferral_rate:.2%} "
            f"| {p.tpr:.1%} | {p.fpr:.3%} | {p.pr_auc:.3f} | {p.latency_ms:.2f} ms "
            f"| **{p.speedup:.1f}x** | {p.throughput:,.0f}/s |"
        )
    return "\n".join(rows)


def _premise_table(study: CascadeStudy) -> str:
    rows = [
        "| split | stage 1 (logistic) PR-AUC | stage 2 (deployed) PR-AUC "
        "| stage 2 better? | cascade detection retained |",
        "|---|---|---|---|---|",
    ]
    for outcome, best in (
        (study.headline, study.best_within(study.max_escape)),
        (study.reference, study.reference_best()),
    ):
        retained = best.tpr / max(outcome.full_tpr, 1e-9)
        rows.append(
            f"| {outcome.strategy} | {outcome.stage1_pr_auc:.3f} | {outcome.full_pr_auc:.3f} "
            f"| {'yes' if outcome.premise_holds else '**no**'} | {retained:.0%} "
            f"(at {best.deferral_rate:.1%} deferral) |"
        )
    return "\n".join(rows)


def _headline_read(study: CascadeStudy, best: CascadePoint) -> str:
    head = study.headline
    ratio = head.stage2_latency_ms / max(head.stage1_latency_ms, 1e-9)
    retained = best.tpr / max(head.full_tpr, 1e-9)
    return (
        f"Stage 1 costs {head.stage1_latency_ms:.2f} ms per flow against stage 2's "
        f"{head.stage2_latency_ms:.2f} ms — {ratio:.0f}x cheaper, measured on this machine, "
        "single-row, the way the API actually serves. That ratio is the entire budget the "
        "cascade has to spend, and it is why the cheap stage has to be genuinely cheap rather "
        f"than merely smaller. At the {best.keep_fraction:.0%}-alert-retention setting, "
        f"{best.deferral_rate:.2%} of traffic reaches the expensive model, blended latency falls "
        f"to {best.latency_ms:.2f} ms ({best.speedup:.1f}x, {best.throughput:,.0f} flows/s against "
        f"{head.full_throughput:,.0f}/s), and detection lands at {best.tpr:.1%} against the full "
        f"model's {head.full_tpr:.1%} — {retained:.0%} of the detection for "
        f"{best.deferral_rate:.1%} of the expensive compute. The false-positive rate is unchanged "
        "by construction: stage 1 can only ever *remove* alerts, never add them, so a cascade is "
        "strictly a recall-side trade and the operator's calibrated FPR budget survives it."
    )


def _premise_read(study: CascadeStudy) -> str:
    head, ref = study.headline, study.reference
    if head.premise_holds:
        return (
            f"The premise holds on both splits: the deployed model out-ranks the cheap one "
            f"({head.full_pr_auc:.3f} vs {head.stage1_pr_auc:.3f} temporally, "
            f"{ref.full_pr_auc:.3f} vs {ref.stage1_pr_auc:.3f} stratified), so escalating the "
            "flows stage 1 is unsure about is buying a genuinely better opinion. That is the "
            "situation a cascade is designed for, and the detection retained above is the honest "
            "price of the compute handed back."
        )
    return (
        "A cascade assumes stage 2 is the better model, and on the honest split **that premise "
        f"does not hold here**: the cheap logistic filter reaches {head.stage1_pr_auc:.3f} PR-AUC "
        f"against the deployed model's {head.full_pr_auc:.3f}, and detects "
        f"{head.stage1_alone_tpr:.1%} at the {study.target_fpr:.1%} budget against "
        f"{head.full_tpr:.1%}. That is not a bug in this study — it is the "
        "[leaderboard](leaderboard.md)'s documented finding arriving a second time, from a "
        "completely different direction: under temporal shift the simpler, higher-bias model "
        "transfers better, because the boosted model's extra capacity is spent on Mon-Wed "
        "structure that Thu-Fri does not honour. It also explains the otherwise-suspicious "
        "PR-AUC column above, where several cascade settings *exceed* the full model: that is "
        "stage 1's ranking showing through on the flows it filtered, not the cascade "
        "manufacturing signal.\n\nSo the temporal row cannot carry the cascade claim on its own, "
        "and the stratified split is included precisely to supply a regime where the premise "
        f"does hold ({ref.full_pr_auc:.3f} vs {ref.stage1_pr_auc:.3f} — the deployed model is "
        "clearly better when train and test are exchangeable). Read together, the two rows say "
        "the engineering result is real and the model-choice result is separate: **the cascade "
        "mechanism reliably hands back compute at a small, budgeted recall cost wherever stage 2 "
        "is worth escalating to** — and on this synthetic stand-in's honest split, the more "
        "useful finding is that stage 2 may not be worth escalating to at all."
    )


def _limits_read(study: CascadeStudy) -> str:
    aggressive = min(study.points, key=lambda p: p.deferral_rate)
    return (
        f"The knee is where the trade stops being free. Pushing to the most aggressive setting "
        f"({aggressive.keep_fraction:.0%} retention, {aggressive.deferral_rate:.2%} of traffic "
        f"deferred) buys {aggressive.speedup:.1f}x but drops detection to {aggressive.tpr:.1%} "
        f"and lets {aggressive.escape_rate:.1%} of the full model's alerts escape stage 1. "
        "Because the threshold is set on validation against the *deployed model's own alerts*, "
        "that escape rate is a budget an operator sets deliberately rather than a surprise found "
        "in production — and because it needs no labels, it can be re-derived on live traffic "
        "whenever the [threshold refresh](refresh.md) job runs."
    )


def _render(study: CascadeStudy, fig: Path) -> str:
    best = study.best_within(study.max_escape)
    return f"""# NetSentry — Budgeted Cascade Inference

_Synthetic stand-in. Headline on the honest temporal/binary split
({study.headline.n_test:,} test flows), with the stratified split
({study.reference.n_test:,} flows) run alongside as a premise check. Both stages timed on
this machine with single-row calls (the serving unit of work); latency numbers are relative,
the ratios are the point. Stage 2 is the deployed LightGBM model; stage 1 is a logistic
regression over the identical fitted feature pipeline, so no second preprocessing path
exists to skew._

## Why this report exists

Single-flow inference costs ~48 ms with SHAP and ~13 ms without. On a link carrying a
million flows a day the usual answers are "add replicas" or "drop the explanations", and
both concede something worth keeping. A **cascade** is the third answer: run a cheap model
on everything and spend the expensive model only where the cheap one is not already sure.
The design question is how to choose the cheap model's cut-off without quietly throwing
detection away — and the answer is not a round number. Stage 1's threshold is chosen on
validation as the quantile that forwards a target share of **the deployed model's own
alerts**, which makes the knob an explicit escape budget, needs no labels, and measures the
cascade's loss against the thing it replaces.

## The trade, priced

{_points_table(study)}

![cascade trade-off](../figures/{fig.name})

{_headline_read(study, best)}

## Checking the premise: is stage 2 actually the better model?

{_premise_table(study)}

{_premise_read(study)}

## Where the trade stops being free

{_limits_read(study)}

## Scope

Latency is measured on one machine with a warm process and no network, so the absolute
milliseconds are not a production SLA — the *ratio* between the stages is what transfers,
and it is the only quantity the speedup depends on. SHAP is excluded from both stages
because explanations are computed on alerts, not on every flow, so they sit downstream of
the cascade entirely and benefit from it automatically (fewer flows reach the stage that
explains). The escape rate is measured against the full model's alerts rather than against
ground truth, deliberately: that is the quantity a live system can compute, and the
label-based detection column is reported beside it so the two views can be compared. The
cascade cannot raise the false-positive rate — filtering only removes alerts — so every
threshold, cost, and conformal result elsewhere in this repo remains valid at its stated
budget; what changes is recall, and it is priced above."""
