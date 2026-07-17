"""The H-measure: a coherent alternative to ROC-AUC (Hand 2009).

The evaluation suite leads with PR-AUC and reports ROC-AUC with the usual caveat that it is
optimistic under imbalance. There is a deeper, less-known problem with AUC that Hand (2009)
made precise: **AUC is incoherent as a way of comparing classifiers**. Averaging over all
thresholds, AUC implicitly weights the cost of a false positive against a false negative by a
distribution that *depends on the classifier's own score distribution* — so two models are
judged under two different cost assumptions, and "model A beats model B on AUC" can silently
mean "A wins under cost assumptions we would never actually hold." For a SOC, where the
false-positive/false-negative trade-off is a real, fixed operating decision, that is exactly
the wrong property.

The H-measure fixes it by making the cost assumption **explicit and shared**. It puts a prior
``u(c)`` — a Beta distribution — on the misclassification-cost parameter ``c`` (the relative
severity of the two error types), the *same* prior for every classifier, and reports the
expected minimum loss under that prior, normalised against a trivial classifier:

    H = 1 - (integral of the model's minimum loss over c)  /  (integral of the trivial loss)

``H = 0`` is the best possible trivial classifier (always-benign or always-attack, whichever
is cheaper at each cost); ``H = 1`` is perfect separation; and, crucially, the same cost prior
is applied to everything, so a cross-model comparison is coherent. The loss curve is built from
the ROC convex hull (the operating points a classifier can actually reach), and the integral
against the Beta prior is evaluated on a fine grid — the exact loss curve, quadratured against
the exact prior. It is reported next to ROC-AUC and Gini so the two can be read against each
other, and under a second, **cost-skewed** prior that encodes the SOC's real stance (a missed
attack costs more than a false alarm) — a knob AUC structurally cannot expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "hmeasure.md"
FIGURE_NAME = "hmeasure.png"


def _loss_integrals(
    y_true: np.ndarray,
    scores: np.ndarray,
    a: float,
    b: float,
    grid_points: int,
) -> tuple[float, float]:
    """Return (model minimum-loss integral, trivial-loss integral) under the Beta(a, b) prior.

    At cost parameter ``c`` the loss of operating point ``(fpr, tpr)`` is
    ``c * pi0 * fpr + (1 - c) * pi1 * (1 - tpr)``; the model's minimum loss is the lower
    envelope over its ROC operating points (the convex hull), and the trivial loss is
    ``min(c * pi0, (1 - c) * pi1)`` — the better of always-attack / always-benign at each
    cost. Both are integrated against the prior by fine-grid quadrature.
    """
    y = np.asarray(y_true)
    pi1 = float(np.mean(y == 1))
    pi0 = 1.0 - pi1
    fpr, tpr, _ = roc_curve(y, scores)
    c = np.linspace(0.0, 1.0, grid_points)
    u = beta_dist.pdf(c, a, b)
    # (grid, K) loss of every ROC operating point at every cost c, then the per-c minimum.
    loss_points = np.outer(c, pi0 * fpr) + np.outer(1.0 - c, pi1 * (1.0 - tpr))
    model_loss = loss_points.min(axis=1)
    trivial_loss = np.minimum(c * pi0, (1.0 - c) * pi1)
    return float(np.trapezoid(model_loss * u, c)), float(np.trapezoid(trivial_loss * u, c))


def h_measure(
    y_true: np.ndarray,
    scores: np.ndarray,
    a: float = 2.0,
    b: float = 2.0,
    grid_points: int = 2000,
) -> float:
    """Hand's H-measure under a Beta(a, b) severity prior: 0 = trivial, 1 = perfect separation.

    Coherent by construction — every classifier is judged under the *same* cost prior, unlike
    ROC-AUC, whose implicit cost weighting is a function of the classifier's own scores. The
    default Beta(2, 2) is Hand's symmetric recommendation.
    """
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return 0.0
    model_loss, trivial_loss = _loss_integrals(y, scores, a, b, grid_points)
    if trivial_loss <= 0.0:
        return 0.0
    return float(1.0 - model_loss / trivial_loss)


@dataclass
class ModelRow:
    """One classifier's coherent-metric row on the shared split."""

    name: str
    roc_auc: float
    h_default: float
    h_skewed: float

    @property
    def gini(self) -> float:
        return 2.0 * self.roc_auc - 1.0


@dataclass
class HMeasureStudy:
    """The H-measure comparison across classifiers under the default and cost-skewed priors."""

    default_prior: tuple[float, float]
    skewed_prior: tuple[float, float]
    rows: list[ModelRow]


def run_hmeasure(settings: Settings) -> HMeasureStudy:
    """Fit the deployed model and two references, then compare ROC-AUC to the coherent H-measure."""
    cfg = settings.hmeasure
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    gbdt = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    scores_gbdt = positive_scores(gbdt.predict_proba(x_test), np.asarray(gbdt.model.classes_))

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(x_train, y_train)
    scores_lr = logreg.predict_proba(x_test)[:, list(logreg.classes_).index(1)]

    rng = np.random.default_rng(variant.seed)
    scores_random = rng.random(len(y_test))

    named = [
        ("Gradient-boosted trees (deployed)", scores_gbdt),
        ("Logistic regression", scores_lr),
        ("Random scores (control)", scores_random),
    ]
    a0, b0 = cfg.prior_alpha, cfg.prior_beta
    a1, b1 = cfg.cost_skew_alpha, cfg.cost_skew_beta
    rows = [
        ModelRow(
            name=name,
            roc_auc=float(roc_auc_score(y_test, s)),
            h_default=h_measure(y_test, s, a0, b0, cfg.grid_points),
            h_skewed=h_measure(y_test, s, a1, b1, cfg.grid_points),
        )
        for name, s in named
    ]
    for r in rows:
        logger.info(
            "H-measure row",
            extra={"model": r.name, "roc_auc": round(r.roc_auc, 3), "h": round(r.h_default, 3)},
        )
    return HMeasureStudy(default_prior=(a0, b0), skewed_prior=(a1, b1), rows=rows)


def run_hmeasure_report(settings: Settings) -> Path:
    """Run the H-measure study and write the report + figure."""
    study = run_hmeasure(settings)

    # Grouped bars are awkward in the shared helper, so plot the H-measure alone (ROC-AUC
    # lives in the table) to keep the figure legible.
    labels = [r.name for r in study.rows]
    fig = plots.plot_barh(
        labels,
        [r.h_default for r in study.rows],
        xlabel=f"H-measure (Beta{study.default_prior})",
        title="Coherent classifier performance: the H-measure",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote H-measure report", extra={"path": str(out_path)})

    with track_run(settings, "hmeasure") as run:
        for r in study.rows:
            tag = r.name.split()[0].lower()
            run.log_metrics(
                {
                    f"roc_auc_{tag}": r.roc_auc,
                    f"h_{tag}": r.h_default,
                    f"h_skewed_{tag}": r.h_skewed,
                }
            )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _table(study: HMeasureStudy) -> str:
    rows = [
        f"| classifier | ROC-AUC | Gini | H (Beta{study.default_prior}) "
        f"| H (cost-skewed Beta{study.skewed_prior}) |",
        "|---|---|---|---|---|",
    ]
    for r in study.rows:
        rows.append(
            f"| {r.name} | {r.roc_auc:.3f} | {r.gini:.3f} | {r.h_default:.3f} | {r.h_skewed:.3f} |"
        )
    return "\n".join(rows)


def _read(study: HMeasureStudy) -> str:
    if not study.rows:
        return ""
    top = study.rows[0]
    auc_order = [r.name for r in sorted(study.rows, key=lambda r: r.roc_auc, reverse=True)]
    h_order = [r.name for r in sorted(study.rows, key=lambda r: r.h_default, reverse=True)]
    agree = auc_order == h_order
    order_clause = (
        "ROC-AUC and the H-measure agree on the ranking here — reassuring, and the common case "
        "when the ROC curves do not cross. "
        if agree
        else "ROC-AUC and the H-measure **disagree on the ranking** — the tell-tale sign of "
        "crossing ROC curves, where AUC's classifier-dependent cost weighting flatters a model "
        "the coherent measure does not. "
    )
    gap = top.h_default - top.h_skewed
    skew_clause = (
        f"Shifting to the cost-skewed prior Beta{study.skewed_prior} — which puts mass where a "
        f"missed attack costs more than a false alarm — moves the deployed model's H from "
        f"{top.h_default:.3f} to {top.h_skewed:.3f} (a {gap:+.3f} change), making the SOC's "
        "actual cost stance an explicit input to the score. ROC-AUC has no such knob: its cost "
        "weighting is whatever the score distribution happens to imply, which is precisely the "
        "incoherence Hand names."
    )
    return (
        f"The H-measure lands well below ROC-AUC in absolute terms — expected, because it is the "
        f"share of the *trivial-classifier loss* that is removed, a stricter scale than the "
        f"rank-based AUC ({top.name} scores AUC {top.roc_auc:.3f} but H {top.h_default:.3f}). "
        + order_clause
        + skew_clause
    )


def _render(study: HMeasureStudy, fig: Path) -> str:
    return f"""# NetSentry — The H-measure (a coherent alternative to ROC-AUC)

_Synthetic stand-in. Temporal/binary split; every classifier judged under the **same** Beta
severity prior, quadratured on a fine cost grid. H = 0 is the best trivial classifier, H = 1 is
perfect separation._

## Why this report exists

The suite already reports ROC-AUC with the imbalance caveat. Hand (2009) identified a subtler
flaw: averaging over all thresholds, AUC implicitly weights false-positive against
false-negative cost by a distribution that **depends on the classifier's own score
distribution**. Two models are therefore compared under two different cost assumptions — so an
AUC win can encode cost assumptions no one would hold. The H-measure removes the incoherence by
fixing an **explicit, shared** Beta prior on the cost parameter for every model, and reporting
the normalised expected minimum loss. Same prior, same yardstick, coherent comparison.

## ROC-AUC vs the coherent H-measure

{_table(study)}

![H-measure across classifiers](../figures/{fig.name})

{_read(study)}

## Scope

The H-measure is a *coherence* fix, not a replacement for the operational metrics: the SOC still
ships at a fixed FPR budget, and PR-AUC + TPR@FPR remain the headline because they speak to that
operating point directly. The value here is comparison hygiene — when ranking model families
(the leaderboard's job) or accepting a challenger (the promotion gate's), the H-measure ensures
the comparison is not being made under a different, classifier-dependent cost assumption for
each candidate. The default Beta(2, 2) is Hand's symmetric recommendation; the cost-skewed prior
shows how the same machinery encodes a real SOC cost stance that ROC-AUC cannot express."""
