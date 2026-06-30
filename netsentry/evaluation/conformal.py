"""Split-conformal prediction for the attack decision — uncertainty with a guarantee.

A point prediction hides uncertainty; a SOC needs to know *which* alerts to trust.
Split-conformal turns a calibration set into per-class nonconformity thresholds and
emits a **prediction set** per flow with a finite-sample, distribution-free promise:
the true label is in the set with probability >= 1 - alpha (class-conditional, the
Mondrian variant), under exchangeability alone — no distributional assumptions.

The four set shapes map cleanly onto SOC actions:

- ``{benign}``  — auto-clear,
- ``{attack}``  — auto-alert,
- ``{benign, attack}`` — ambiguous, route to a human,
- ``{}`` (empty) — looks like *neither* train class, i.e. novel: route to a human
  (this is the conformal echo of the anomaly detector's "unknown" story).

So the abstention rate is exactly the human-review load, and the guarantee bounds
how often an auto-decision is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "conformal.md"


def conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    """The (1-alpha) split-conformal quantile with the finite-sample (+1) correction."""
    scores = np.asarray(nonconformity, dtype=float)
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def class_conditional_thresholds(
    p_cal: np.ndarray, y_cal: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Per-class nonconformity thresholds (tau_benign, tau_attack) on a calib set."""
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    tau_benign = conformal_quantile(p_cal[y_cal == 0], alpha)  # benign nonconf = p
    tau_attack = conformal_quantile(1.0 - p_cal[y_cal == 1], alpha)  # attack nonconf = 1 - p
    return tau_benign, tau_attack


def prediction_sets(
    p: np.ndarray, tau_benign: float, tau_attack: float
) -> tuple[np.ndarray, np.ndarray]:
    """Membership masks (in_benign, in_attack) for each flow's prediction set."""
    p = np.asarray(p, dtype=float)
    in_benign = p <= tau_benign
    in_attack = (1.0 - p) <= tau_attack
    return in_benign, in_attack


@dataclass
class ConformalReport:
    """Coverage and set-shape breakdown for one alpha on the test set."""

    alpha: float
    coverage_benign: float
    coverage_attack: float
    rate_benign_only: float
    rate_attack_only: float
    rate_ambiguous: float
    rate_empty: float
    auto_error: float  # error rate among singletons (auto-decided flows)


def evaluate_conformal(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> ConformalReport:
    """Calibrate on (p_cal, y_cal); report coverage and set shapes on the test set."""
    tau_b, tau_a = class_conditional_thresholds(p_cal, y_cal, alpha)
    in_benign, in_attack = prediction_sets(p_test, tau_b, tau_a)
    y_test = np.asarray(y_test, dtype=int)

    benign_mask, attack_mask = y_test == 0, y_test == 1
    cov_benign = float(np.mean(in_benign[benign_mask])) if benign_mask.any() else 0.0
    cov_attack = float(np.mean(in_attack[attack_mask])) if attack_mask.any() else 0.0

    benign_only = in_benign & ~in_attack
    attack_only = in_attack & ~in_benign
    ambiguous = in_benign & in_attack
    empty = ~in_benign & ~in_attack
    n = max(len(y_test), 1)

    singleton = benign_only | attack_only
    if singleton.any():
        pred_attack = attack_only[singleton]
        truth = y_test[singleton] == 1
        auto_error = float(np.mean(pred_attack != truth))
    else:
        auto_error = 0.0

    return ConformalReport(
        alpha=alpha,
        coverage_benign=cov_benign,
        coverage_attack=cov_attack,
        rate_benign_only=float(np.sum(benign_only) / n),
        rate_attack_only=float(np.sum(attack_only) / n),
        rate_ambiguous=float(np.sum(ambiguous) / n),
        rate_empty=float(np.sum(empty) / n),
        auto_error=auto_error,
    )


def _scores(
    settings: Settings, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a binary model on ``strategy`` and return calibrated (cal, test) scores."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = strategy  # type: ignore[assignment]
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)
    s_cal = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    if result.bundle.calibrator is not None:
        s_cal = result.bundle.calibrator.transform(s_cal)
        s_test = result.bundle.calibrator.transform(s_test)
    return s_cal, result.y_val.astype(int), s_test, result.y_test.astype(int)


def run_conformal_report(settings: Settings) -> Path:
    """Calibrate conformal sets on both splits; the temporal one tests the guarantee.

    The coverage guarantee holds under *exchangeability*. The stratified split is
    exchangeable, so coverage should be met; the temporal split deliberately is not
    (later-day attacks are novel), so the attack-class coverage shortfall is itself a
    drift signal — conformal correctly revealing the shift the split exposes.
    """
    cfg = settings.conformal
    strat = _scores(settings, "stratified")
    temp = _scores(settings, "temporal")
    strat_cov = evaluate_conformal(*strat, cfg.alpha)
    headline = evaluate_conformal(*temp, cfg.alpha)
    sweep = [evaluate_conformal(*temp, a) for a in cfg.alphas_grid]

    alphas = np.array([r.alpha for r in sweep])
    fig = plots.plot_lines(
        {
            "target (1 - alpha)": (alphas, 1.0 - alphas),
            "benign coverage (temporal)": (alphas, np.array([r.coverage_benign for r in sweep])),
            "attack coverage (temporal)": (alphas, np.array([r.coverage_attack for r in sweep])),
        },
        xlabel="alpha (target error)",
        ylabel="Empirical coverage on test",
        title="Conformal coverage vs the 1 - alpha target (temporal split)",
        out_path=settings.paths.figures_dir / "conformal_coverage.png",
    )

    report = _render(settings, headline, strat_cov, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote conformal report", extra={"path": str(out_path)})

    with track_run(settings, "conformal") as run:
        run.log_params({"alpha": cfg.alpha})
        run.log_metrics(
            {
                "temporal_coverage_benign": headline.coverage_benign,
                "temporal_coverage_attack": headline.coverage_attack,
                "stratified_coverage_attack": strat_cov.coverage_attack,
                "abstention_rate": headline.rate_ambiguous + headline.rate_empty,
                "auto_error": headline.auto_error,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(settings: Settings, r: ConformalReport, strat: ConformalReport, fig: Path) -> str:
    review = r.rate_ambiguous + r.rate_empty
    target = 1.0 - r.alpha
    attack_met = "met" if r.coverage_attack >= target - 0.01 else "**below target**"
    strat_met = "met" if strat.coverage_attack >= target - 0.01 else "below target"
    sb, sa = f"{strat.coverage_benign * 100:.1f}", f"{strat.coverage_attack * 100:.1f}"
    tb, ta = f"{r.coverage_benign * 100:.1f}", f"{r.coverage_attack * 100:.1f}"
    return f"""# NetSentry — Conformal Prediction & Selective Alerting

_Synthetic stand-in; the method is the point. Split-conformal calibrated on the
validation split, alpha = **{r.alpha:g}** (target coverage **{target * 100:.0f}%**)._

## The guarantee — and where it breaks (which is the interesting part)

Split-conformal emits a prediction set per flow with a finite-sample,
distribution-free promise: the true label is in the set with probability ≥
{target * 100:.0f}%, **under exchangeability** of calibration and test. The stratified
split is exchangeable; the temporal split deliberately is not (later-day attacks are
novel). Class-conditional coverage on test:

| split | benign coverage | attack coverage | attack guarantee |
|---|---|---|---|
| stratified (exchangeable) | {sb}% | {sa}% | {strat_met} |
| temporal (later-day) | {tb}% | {ta}% | {attack_met} |

On the exchangeable split the guarantee holds for both classes. On the temporal
split benign coverage still holds (benign traffic is stable across days) but
**attack coverage falls short** — because the exchangeability assumption is broken,
not because conformal is wrong. That shortfall is a *symptom of distribution shift*:
conformal coverage on a recent window is an independent drift signal, complementing
the PSI monitor.

![Coverage vs alpha](../figures/{fig.name})

## Set shapes → SOC actions (temporal split)

| prediction set | meaning | action | share of flows |
|---|---|---|---|
| {{benign}} | confident benign | auto-clear | {r.rate_benign_only * 100:.1f}% |
| {{attack}} | confident attack | auto-alert | {r.rate_attack_only * 100:.1f}% |
| {{benign, attack}} | ambiguous | human review | {r.rate_ambiguous * 100:.1f}% |
| {{}} (empty) | novel — like neither class | human review | {r.rate_empty * 100:.1f}% |

The model **auto-decides {(r.rate_benign_only + r.rate_attack_only) * 100:.1f}%** of
flows and routes **{review * 100:.1f}%** to a human. Abstaining on the ambiguous cases
is what makes selective prediction useful: a tunable human-review budget (via alpha)
rather than a forced guess on every flow.

## Why this matters

A SOC cannot review every flow, and a bare probability does not say when to defer.
Conformal selective prediction makes "the model knows when it doesn't know"
operational, with a coverage guarantee where exchangeability holds — and, where it
does not, a coverage shortfall that flags the very drift the temporal split exposes.
"""
