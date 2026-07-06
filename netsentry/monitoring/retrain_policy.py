"""When to retrain — trigger policies priced on the prequential stream.

The streaming study establishes *that* retraining recovers the detection later-day
drift costs; this study prices *when*. Retraining after every batch is the quality
ceiling, but every retrain costs something real — freshly-labeled flows (the
analyst budget the active-learning study prices), compute, re-validation, and the
deployment risk of swapping a model. The operational question is which **trigger**
buys the ceiling's quality at a fraction of its retrains:

- ``never`` — the static model, frozen at deploy (the floor).
- ``every_batch`` — retrain after each labeled batch (the ceiling).
- ``periodic`` — retrain every k-th batch, the calendar-driven default most teams
  start with.
- ``drift_triggered`` — retrain only when the deployed model's own score-PSI
  breaches the major-drift threshold, with a cooldown. This is the drift monitor's
  alarm wired to the retraining lever: **the same PSI threshold the Prometheus
  alert fires on**, so measurement, alert, and action share one number.

Faithfulness details that matter: scoring is prequential (score a batch, then let
the policy learn from it), every policy is judged at one operating threshold chosen
once on clean validation, and each policy's drift signal comes from its **own**
deployed model against a reference that **resets on redeploy** — exactly how the
serving drift monitor works, where the reference travels inside the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.monitoring.drift import population_stability_index
from netsentry.monitoring.streaming import order_stream
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "retrain_policy.md"


class RetrainTrigger(Protocol):
    """Decides, after a batch has been scored, whether the policy retrains now."""

    def should_retrain(self, batch_index: int, score_psi: float) -> bool: ...


@dataclass
class NeverTrigger:
    """The static floor: deploy once, never retrain."""

    def should_retrain(self, batch_index: int, score_psi: float) -> bool:
        return False


@dataclass
class AlwaysTrigger:
    """The quality ceiling: retrain after every labeled batch."""

    def should_retrain(self, batch_index: int, score_psi: float) -> bool:
        return True


@dataclass
class PeriodicTrigger:
    """Calendar cadence: retrain after every ``every``-th batch (1-based)."""

    every: int

    def should_retrain(self, batch_index: int, score_psi: float) -> bool:
        return (batch_index + 1) % self.every == 0


@dataclass
class DriftTrigger:
    """Retrain when the deployed model's score-PSI breaches ``threshold``.

    A ``cooldown`` (in batches) suppresses immediate re-fires: after a redeploy the
    monitor needs fresh windows before another alarm is actionable, and back-to-back
    retrains double the cost for the same evidence.
    """

    threshold: float
    cooldown: int = 1
    _last_retrain: int = field(default=-(10**9), repr=False)

    def should_retrain(self, batch_index: int, score_psi: float) -> bool:
        if score_psi < self.threshold:
            return False
        if batch_index - self._last_retrain < self.cooldown:
            return False
        self._last_retrain = batch_index
        return True


@dataclass
class PolicyOutcome:
    """One policy's ride through the stream: quality, cost, and when it acted."""

    name: str
    retrains: int
    mean_pr_auc: float
    mean_detection: float
    per_batch_pr_auc: list[float]
    retrained_at: list[int]


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else 0.0


def run_retrain_policy_report(settings: Settings) -> Path:
    """Replay the later-day stream under each retrain policy; write the frontier."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    cfg = variant.retrain_policy
    benign = variant.labels.benign_label
    operating_fpr = variant.thresholds.fpr_targets[-1]
    psi_trigger = cfg.psi_trigger if cfg.psi_trigger is not None else variant.monitoring.psi_major

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = order_stream(load_split(variant, "temporal", "test"))

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)
    x_val = pipeline.transform(val)
    x_test = pipeline.transform(test)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    def fit(x: np.ndarray, y: np.ndarray) -> SupervisedClassifier:
        seed_everything(variant.seed)
        return SupervisedClassifier(variant).fit(x, y, eval_set=(x_val, y_val))

    base_model = fit(x_train, y_train)
    s_val = attack_probability(base_model.predict_proba(x_val), base_model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)

    policies: list[tuple[str, RetrainTrigger]] = [
        ("never", NeverTrigger()),
        (f"periodic (every {cfg.periodic_every})", PeriodicTrigger(cfg.periodic_every)),
        (
            f"drift-triggered (PSI >= {psi_trigger:g})",
            DriftTrigger(psi_trigger, cooldown=cfg.cooldown_batches),
        ),
        ("every batch", AlwaysTrigger()),
    ]
    batches = np.array_split(np.arange(len(y_test)), cfg.n_batches)

    outcomes: list[PolicyOutcome] = []
    for name, trigger in policies:
        outcomes.append(
            _simulate(
                name,
                trigger,
                base_model,
                fit,
                batches,
                x_train,
                y_train,
                x_test,
                y_test,
                threshold=threshold,
                benign=benign,
                psi_bins=variant.monitoring.psi_bins,
            )
        )
        logger.info(
            "Policy simulated",
            extra={
                "policy": name,
                "retrains": outcomes[-1].retrains,
                "mean_pr_auc": round(outcomes[-1].mean_pr_auc, 4),
            },
        )

    fig = plots.plot_lines(
        {
            o.name: (np.array([o.retrains], dtype=float), np.array([o.mean_pr_auc]))
            for o in outcomes
        },
        xlabel="Retrains over the stream (the cost axis)",
        ylabel="Mean batch PR-AUC (the quality axis)",
        title="Retrain-policy efficiency frontier",
        out_path=settings.paths.figures_dir / "retrain_policy.png",
    )

    report = _render(outcomes, cfg.n_batches, operating_fpr, psi_trigger, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote retrain-policy report", extra={"path": str(out_path)})

    with track_run(settings, "retrain_policy") as run:
        for o in outcomes:
            key = o.name.split(" (")[0].replace(" ", "_").replace("-", "_")
            run.log_metrics({f"{key}_mean_pr_auc": o.mean_pr_auc, f"{key}_retrains": o.retrains})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _simulate(
    name: str,
    trigger: RetrainTrigger,
    base_model: SupervisedClassifier,
    fit: object,
    batches: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float,
    benign: str,
    psi_bins: int,
) -> PolicyOutcome:
    """Run one policy prequentially; its drift signal comes from its own model."""
    model = base_model
    # The deployed model's reference score distribution — recomputed on redeploy,
    # exactly as the serving bundle's drift reference is rebuilt when a new model
    # ships. PSI is then "current batch vs what this deployment normally scores".
    reference = attack_probability(model.predict_proba(x_train), model.classes_, benign)
    pool_x, pool_y = x_train, y_train
    per_batch: list[float] = []
    detections: list[float] = []
    retrained_at: list[int] = []

    for i, idx in enumerate(batches):
        if idx.size == 0:
            continue
        xb, yb = x_test[idx], y_test[idx]
        scores = attack_probability(model.predict_proba(xb), model.classes_, benign)
        pr = float(average_precision_score(yb, scores)) if len(np.unique(yb)) > 1 else float("nan")
        per_batch.append(pr)
        detections.append(rates_at_threshold(yb, scores, threshold)["tpr"])
        psi = population_stability_index(reference, scores, bins=psi_bins)

        # Prequential: the batch is only available for learning after being scored.
        pool_x = np.vstack([pool_x, xb])
        pool_y = np.concatenate([pool_y, yb])
        if trigger.should_retrain(i, psi):
            model = fit(pool_x, pool_y)  # type: ignore[operator]
            reference = attack_probability(model.predict_proba(x_train), model.classes_, benign)
            retrained_at.append(i)

    return PolicyOutcome(
        name=name,
        retrains=len(retrained_at),
        mean_pr_auc=_nanmean(per_batch),
        mean_detection=_nanmean(detections),
        per_batch_pr_auc=per_batch,
        retrained_at=retrained_at,
    )


def _render(
    outcomes: list[PolicyOutcome],
    n_batches: int,
    operating_fpr: float,
    psi_trigger: float,
    fig: Path,
) -> str:
    rows = [
        "| policy | retrains | mean batch PR-AUC | mean detection @ "
        f"{operating_fpr * 100:g}% FPR | retrained after batches |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        when = ", ".join(str(b) for b in o.retrained_at) if o.retrained_at else "-"
        rows.append(
            f"| {o.name} | {o.retrains} | {o.mean_pr_auc:.3f} | "
            f"{o.mean_detection * 100:.1f}% | {when} |"
        )

    floor = next(o for o in outcomes if o.name == "never")
    ceiling = next(o for o in outcomes if o.name == "every batch")
    triggered = next(o for o in outcomes if o.name.startswith("drift-triggered"))

    width = max((len(o.per_batch_pr_auc) for o in outcomes), default=0)
    batch_rows = [
        "| policy | " + " | ".join(f"b{i}" for i in range(width)) + " |",
        "|---" * (width + 1) + "|",
    ]
    for o in outcomes:
        cells = " | ".join("-" if np.isnan(v) else f"{v:.2f}" for v in o.per_batch_pr_auc)
        batch_rows.append(f"| {o.name} | {cells} |")

    headroom = ceiling.mean_pr_auc - floor.mean_pr_auc
    fired_at = ", ".join(str(b) for b in triggered.retrained_at) or "never"
    if headroom <= 0.005:
        efficiency = (
            "On this stand-in the ceiling and the floor coincide - the stream is stable enough "
            "that retraining buys nothing, and the drift trigger correctly spends "
            f"{triggered.retrains} retrain(s). A trigger's value appears only when there is "
            "drift to react to."
        )
    else:
        captured = (triggered.mean_pr_auc - floor.mean_pr_auc) / headroom
        if captured >= 0.5:
            efficiency = (
                f"The drift trigger captures **{captured * 100:.0f}%** of the retraining "
                f"headroom (static {floor.mean_pr_auc:.3f} -> ceiling "
                f"{ceiling.mean_pr_auc:.3f}) with **{triggered.retrains} retrain(s)** against "
                f"the ceiling's {ceiling.retrains} - it spends a retrain only where the monitor "
                "sees the deployment actually move."
            )
        else:
            efficiency = (
                f"**The trigger under-delivers here, and that is the finding.** It fired early "
                f"(after batch {fired_at} - the deployment saw major drift the moment later-day "
                "traffic arrived), then went quiet: once those batches were folded in, its own "
                "score-PSI stayed under the line for the rest of the stream while the "
                f"every-batch policy kept improving ({triggered.mean_pr_auc:.3f} vs "
                f"{ceiling.mean_pr_auc:.3f} mean PR-AUC, {captured * 100:.0f}% of the headroom "
                "captured). PSI watches the score *distribution*, and a distribution can settle "
                "while labeled quality is still being bought - an unsupervised drift trigger is "
                "a cost-saver against covariate shift, **not a substitute for labels**. The "
                "honest pairing is a PSI trigger for the fast alarm plus a periodic labeled "
                "cadence for the slow decay - which is exactly what the periodic row prices."
            )

    return f"""# NetSentry - Retrain-Trigger Policy (when to retrain)

_Synthetic stand-in; the method is the point. The later-day (temporal test) flows
replayed as {n_batches} time-ordered batches, scored prequentially at one operating
threshold ({operating_fpr * 100:g}%-FPR, chosen once on clean validation). Each
policy's drift signal is its **own** deployed model's score-PSI against a reference
that resets on redeploy - the same mechanics as the serving drift monitor._

## The question

The [streaming study](streaming.md) shows retraining recovers what drift costs; this
prices **when**. Every retrain costs labels (the analyst budget the
[active-learning study](active_learning.md) prices), compute, re-validation, and
deployment risk - so the operational question is which *trigger* buys the
every-batch ceiling's quality at a fraction of its retrains.

{chr(10).join(rows)}

Per-batch PR-AUC, in time order (where each policy's quality actually diverges):

{chr(10).join(batch_rows)}

![Retrain-policy efficiency frontier](../figures/{fig.name})

## Read

- {efficiency}
- The trigger threshold (PSI >= {psi_trigger:g}) is **the same major-drift line the
  Prometheus alert rule fires on**, so measurement, alert, and action share one
  number: what pages the on-call is literally what schedules the retrain.
- The periodic policy is the calendar default most teams start with; the frontier
  shows what it buys relative to reacting to evidence. Where periodic and triggered
  tie on quality, the trigger's advantage is cost; where they tie on cost, quality.

## Honest limits

A trigger tuned on one stream can misfire on another (PSI is sensitive to batch
size), labels are assumed to arrive with the batch (in a SOC they arrive late and
partially - the active-learning study is the mitigation), and retraining on the
freshest batches inherits whatever poisoning risk the
[poisoning study](poisoning.md) measures. The frontier here is a method for pricing
the policy, not a promise about its numbers.
"""
