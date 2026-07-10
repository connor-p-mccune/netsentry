"""Threshold refresh: the cheap adaptation lever, priced against full retraining.

The cost report documents that a validation-chosen threshold drifts on later-day
traffic, and the streaming study prices the expensive fix (retrain on every
labeled batch). Between "do nothing" and "refit everything" sits the lever every
operations team reaches for first because it is nearly free: **keep the model
frozen and re-choose only the decision threshold** on a trailing window of
recently labeled flows, at the same FPR budget.

This study rides the same prequential stream as the streaming/retrain-policy
work and decomposes what drift actually costs into two parts:

- **operating-point drift** — the score *distribution* moved, so the frozen
  threshold no longer spends the FPR budget it was chosen for. A refresh fixes
  this: it needs labels only to re-estimate one quantile, not to relearn
  anything.
- **ranking drift** — the model itself is blind to the later-day attack types.
  No threshold can fix this: moving the cut trades the same broken ranking's
  false positives for false negatives. Only retraining (new labels into the
  model) buys it back.

Four policies, one budget: static (frozen model + frozen threshold), refresh
(frozen model + trailing-window threshold), retrain (fresh model + frozen
threshold, the streaming study's convention), and retrain+refresh (both levers —
the ceiling). Refreshed thresholds are always chosen on the **prequentially
emitted** scores — what the deployed model actually said before it learned from
the batch — which is exactly the evidence a real deployment's alert stream
provides, and never lets a model pick its cut on flows it has already trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.monitoring.streaming import order_stream
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "refresh.md"


def refresh_threshold(
    past_y: list[np.ndarray],
    past_s: list[np.ndarray],
    target_fpr: float,
    window: int,
    fallback: float,
) -> float:
    """Re-choose the operating threshold on the trailing ``window`` labeled batches.

    Uses only already-scored (prequential) evidence. If the trailing window is
    empty or one-class — a quiet stretch with nothing to calibrate on — the
    previous threshold stands rather than guessing.
    """
    if not past_y or window <= 0:
        return fallback
    y = np.concatenate(past_y[-window:])
    s = np.concatenate(past_s[-window:])
    if len(np.unique(y)) < 2:
        return fallback
    return float(threshold_at_fpr(y, s, target_fpr))


@dataclass
class PolicyTrace:
    """One policy's per-batch operating behaviour over the stream."""

    name: str
    detection: list[float]
    realized_fpr: list[float]
    thresholds: list[float]

    def mean_detection(self) -> float:
        return float(np.mean(self.detection)) if self.detection else 0.0

    def mean_fpr(self) -> float:
        return float(np.mean(self.realized_fpr)) if self.realized_fpr else 0.0


def simulate_threshold_policies(
    batch_y: list[np.ndarray],
    batch_s: list[np.ndarray],
    target_fpr: float,
    initial_threshold: float,
    window: int,
) -> tuple[PolicyTrace, PolicyTrace]:
    """(static, refresh) traces for a frozen model's emitted scores.

    Pure so the refresh mechanics are testable without a model: the static policy
    holds ``initial_threshold`` for the whole stream; the refresh policy re-chooses
    it before each batch on the trailing labeled window.
    """
    static = PolicyTrace("static", [], [], [])
    refresh = PolicyTrace("refresh", [], [], [])
    threshold = initial_threshold
    for i, (y, s) in enumerate(zip(batch_y, batch_s, strict=True)):
        if i > 0:
            threshold = refresh_threshold(
                batch_y[:i], batch_s[:i], target_fpr, window, fallback=threshold
            )
        for trace, cut in ((static, initial_threshold), (refresh, threshold)):
            rates = rates_at_threshold(y, s, cut)
            trace.detection.append(rates["tpr"])
            trace.realized_fpr.append(rates["fpr"])
            trace.thresholds.append(cut)
    return static, refresh


def run_refresh_report(settings: Settings) -> Path:
    """Price the threshold-refresh lever against retraining on the prequential stream."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    benign = variant.labels.benign_label
    operating_fpr = variant.thresholds.fpr_targets[-1]
    cfg = variant.refresh

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

    seed_everything(variant.seed)
    static_model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_val = attack_probability(static_model.predict_proba(x_val), static_model.classes_, benign)
    initial = threshold_at_fpr(y_val, s_val, operating_fpr)

    batches = [idx for idx in np.array_split(np.arange(len(y_test)), cfg.n_batches) if idx.size]
    batch_y = [y_test[idx] for idx in batches]
    static_scores = [
        attack_probability(static_model.predict_proba(x_test[idx]), static_model.classes_, benign)
        for idx in batches
    ]
    static_trace, refresh_trace = simulate_threshold_policies(
        batch_y, static_scores, operating_fpr, initial, cfg.window_batches
    )

    # The retrain arm: prequential (score, then learn), one model sequence shared
    # by both retrain policies; the refreshed variant re-chooses its cut on the
    # emitted (pre-update) scores, mirroring the frozen-model refresh exactly.
    retrain = PolicyTrace("retrain", [], [], [])
    retrain_refresh = PolicyTrace("retrain+refresh", [], [], [])
    emitted: list[np.ndarray] = []
    pool_x, pool_y = x_train, y_train
    model = static_model
    threshold = initial
    for i, idx in enumerate(batches):
        s = attack_probability(model.predict_proba(x_test[idx]), model.classes_, benign)
        if i > 0:
            threshold = refresh_threshold(
                batch_y[:i], emitted, operating_fpr, cfg.window_batches, fallback=threshold
            )
        for trace, cut in ((retrain, initial), (retrain_refresh, threshold)):
            rates = rates_at_threshold(batch_y[i], s, cut)
            trace.detection.append(rates["tpr"])
            trace.realized_fpr.append(rates["fpr"])
            trace.thresholds.append(cut)
        emitted.append(s)
        pool_x = np.vstack([pool_x, x_test[idx]])
        pool_y = np.concatenate([pool_y, batch_y[i]])
        seed_everything(variant.seed)
        model = SupervisedClassifier(variant).fit(pool_x, pool_y, eval_set=(x_val, y_val))

    traces = [static_trace, refresh_trace, retrain, retrain_refresh]
    x = np.arange(len(batches))
    fig = plots.plot_lines(
        {
            "budget": (x, np.full(len(batches), operating_fpr)),
            "static threshold": (x, np.array(static_trace.realized_fpr)),
            "refreshed threshold": (x, np.array(refresh_trace.realized_fpr)),
        },
        xlabel="Stream batch (time order)",
        ylabel="Realized FPR per batch",
        title="Threshold refresh: holding the FP budget under drift",
        out_path=settings.paths.figures_dir / "refresh.png",
    )

    report = _render(settings, traces, operating_fpr, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote refresh report", extra={"path": str(out_path)})

    with track_run(settings, "refresh") as run:
        run.log_params({"n_batches": cfg.n_batches, "window_batches": cfg.window_batches})
        run.log_metrics(
            {f"{t.name.replace('+', '_')}_mean_detection": t.mean_detection() for t in traces}
            | {f"{t.name.replace('+', '_')}_mean_fpr": t.mean_fpr() for t in traces}
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(settings: Settings, traces: list[PolicyTrace], operating_fpr: float, fig: Path) -> str:
    cfg = settings.refresh
    static, refresh, retrain, both = traces
    rows = [
        "| policy | model | threshold | mean detection | mean realized FPR |",
        "|---|---|---|---|---|",
        f"| static | frozen | frozen | {static.mean_detection():.1%} | {static.mean_fpr():.3%} |",
        f"| **refresh** | frozen | trailing {cfg.window_batches}-batch window "
        f"| {refresh.mean_detection():.1%} | {refresh.mean_fpr():.3%} |",
        f"| retrain | per batch | frozen | {retrain.mean_detection():.1%} "
        f"| {retrain.mean_fpr():.3%} |",
        f"| retrain+refresh | per batch | trailing window | **{both.mean_detection():.1%}** "
        f"| {both.mean_fpr():.3%} |",
    ]

    refresh_gain = refresh.mean_detection() - static.mean_detection()
    retrain_gain = both.mean_detection() - static.mean_detection()
    share = refresh_gain / retrain_gain if retrain_gain > 0 else float("nan")
    budget_error_static = abs(static.mean_fpr() - operating_fpr)
    budget_error_refresh = abs(refresh.mean_fpr() - operating_fpr)

    if np.isfinite(share) and 0.0 <= share <= 0.5:
        read = (
            f"The decomposition is clean on this stream: refreshing the threshold moves "
            f"detection by **{refresh_gain:+.1%}** while the full retrain+refresh ceiling "
            f"moves it {retrain_gain:+.1%} — the cheap lever buys ~{share:.0%} of the "
            "recovery. Most of what drift costs here is *ranking* (the frozen model cannot "
            "score the later-day attack types), and no threshold can un-blind a model."
        )
    elif np.isfinite(share) and share > 0.5:
        read = (
            f"On this stream the refresh alone recovers {refresh_gain:+.1%} of detection — "
            f"{share:.0%} of what the retrain+refresh ceiling ({retrain_gain:+.1%}) buys. "
            "That means most of the drift cost here was *operating-point* drift (the score "
            "distribution moved under a frozen cut), which a label-cheap quantile re-estimate "
            "fixes. Check the per-class slices before celebrating: a refresh cannot recover "
            "attack types the model has never seen."
        )
    else:
        read = (
            f"Neither lever moves detection materially on this stream (refresh "
            f"{refresh_gain:+.1%}, ceiling {retrain_gain:+.1%}); what remains to compare is "
            "budget compliance, below."
        )

    # Budget compliance is the refresh's *purpose*, but whether it wins here is an
    # empirical question — the prose must follow the measured distances, not the
    # design intent (a frozen cut that only drifts mildly can sit closer to the
    # budget than a re-estimate that is jumpy at small window sizes).
    if budget_error_refresh < budget_error_static - 1e-4:
        compliance = (
            f"Budget compliance is the refresh's real product, and here it delivers: mean "
            f"distance from the {operating_fpr:.1%} target is {budget_error_static:.3%} for "
            f"the frozen cut and {budget_error_refresh:.3%} refreshed. An operating point is "
            "a *promise about false-positive spend*, and under drift a frozen threshold "
            "silently re-negotiates that promise; the refresh keeps it honest for the price "
            "of a quantile estimate."
        )
    else:
        compliance = (
            f"The honest wrinkle: on this stream the refresh does **not** win budget "
            f"compliance either — the frozen cut sits {budget_error_static:.3%} from the "
            f"{operating_fpr:.1%} target on average while the refreshed cut sits "
            f"{budget_error_refresh:.3%}, because the benign score distribution barely moves "
            f"across these days and a {cfg.window_batches}-batch quantile estimate carries "
            "its own sampling noise. The lever's value case is the one the unit tests "
            "construct — a material shift in the *score distribution* (the failure the PSI "
            "monitor flags), where a frozen cut's realized FPR runs multiples over budget "
            "and the refresh pulls it back. When the distribution is stable, the refresh is "
            "insurance that costs a little estimator noise."
        )

    return f"""# NetSentry — Threshold Refresh (the cheap lever, priced)

_Synthetic stand-in. Temporal split; the later-day test flows replayed as
{len(static.detection)} prequential batches at the {operating_fpr:.1%}-FPR
operating point. Refreshed thresholds are re-chosen before each batch on the
trailing {cfg.window_batches} labeled batches, always from the **emitted**
(pre-update) scores — the evidence a real alert stream provides — so no model
ever picks its cut on flows it trained on._

## The question

Retraining recovers what drift costs (the streaming study), but it is the
expensive lever: full labels, a fit, a redeploy, a promotion decision. The lever
every team reaches for first is nearly free — keep the model, re-choose the
threshold on recent labels. How much of the retraining recovery does that
actually buy, and what does it reliably own?

## Results (means over the stream)

{chr(10).join(rows)}

![Realized FPR per batch](../figures/{fig.name})

## Read

{read}

{compliance}

## Scope

The refresh consumes labels too — a trailing window of them — but orders of
magnitude fewer effective bits than retraining (one quantile vs a model). It
composes with, rather than replaces, the retrain-policy machinery: a deployment
would refresh continuously and retrain on the drift/calendar trigger that study
prices. And it inherits the streaming study's caveat: batches here are labeled
in full; in production the labels come from the analyst queue, which is what the
active-learning study budgets.
"""
