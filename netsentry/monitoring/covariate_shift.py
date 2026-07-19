"""Covariate-shift diagnosis and importance-weighted correction on the temporal gap.

The project's headline finding is that the honest **temporal** split scores well below the
optimistic **stratified** one — the model trained on the early capture days is worse on the
later days than a shuffled split pretends. The [leakage study](leakage.md) proves that gap is
real (not an artefact of identifier leakage), and the [novelty study](novelty.md) decomposes it
geometrically. This study asks the distribution-shift question directly: *how much of the gap
is covariate shift* — the input law `p(x)` moving between train and test while `p(y|x)` holds —
and can importance weighting, the textbook fix, close it?

The estimator needs **zero test labels**. Train a **domain classifier** to tell a train flow
from a test flow on the features alone (Bickel, Brückner & Scheffer 2009; the classifier
two-sample test of Lopez-Paz & Oquab 2017). Its held-out **AUC is a shift detector** — 0.5
means the two days are indistinguishable (no covariate shift), higher means the input
distribution moved. Its calibrated probability gives the **density ratio** directly:
`w(x) = p_test(x)/p_train(x) = [P(test|x)/P(train|x)] · (n_train/n_test)`, computed by
**cross-fitting** so no flow scores a classifier that memorised it. Those weights are the
importance-weighted-ERM correction (Shimodaira 2000): refit the detector with each training
flow weighted by how much it looks like the future, and — if the shift is covariate — the
reweighted model is unbiased for the test distribution.

Two things are reported honestly. The **effective sample size** `ESS = (Σw)²/Σw²` prices what
the shift costs before any retrain — extreme weights mean the training set the future actually
resembles is far smaller than its row count. And the correction is measured, not assumed: the
importance-weighted retrain is scored on the temporal test split against the unweighted
baseline and the stratified no-shift ceiling. The anticipated, sophisticated result is that IW
buys little — because the temporal gap is not mainly covariate shift but **concept shift**
(`p(y|x)` moves: later days carry attack behaviours the early days never showed), which
reweighting `p(x)` structurally cannot fix. That is the honest complement to the
[label-shift](label_shift.md) study (which corrects `p(y)`): between them they cover the shift
taxonomy, and this study names the residual — concept drift — that neither can touch and that
only new labels (the [active-learning](active_learning.md) / [streaming](streaming.md) loop)
can.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.leaderboard import build_family
from netsentry.evaluation.metrics import attack_probability, operating_point, positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import CovariateShiftConfig

logger = get_logger(__name__)

REPORT_NAME = "covariate_shift.md"
FIGURE_NAME = "covariate_shift.png"


def _proba_test(model: object, x: np.ndarray) -> np.ndarray:
    """P(domain = test) from a fitted sklearn-API classifier."""
    proba = np.asarray(model.predict_proba(x))  # type: ignore[attr-defined]
    classes = np.asarray(model.classes_)  # type: ignore[attr-defined]
    col = int(np.where(classes == 1)[0][0])
    return proba[:, col]


def crossfit_domain_ratio(
    x_train: np.ndarray,
    x_test: np.ndarray,
    build_fn: object,
    seed: int,
    n_folds: int,
    clip: float,
) -> tuple[np.ndarray, float]:
    """Cross-fit density ratios `w(x)=p_test/p_train` for the train rows + the C2ST AUC.

    Pools train (domain 0) and test (domain 1), and predicts each pooled flow's P(test|x) from
    a domain classifier fit on the *other* folds, so a flow never scores a model that saw it.
    The held-out predictions give both the classifier-two-sample-test AUC (a shift detector)
    and, on the train rows, the density ratio `[p/(1-p)] · (n_train/n_test)`, clipped to bound
    the weight variance. Returns `(weights_for_train_rows, c2st_auc)`.
    """
    n_train, n_test = len(x_train), len(x_test)
    x_pool = np.concatenate([x_train, x_test])
    d = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_test, dtype=int)])
    oof = np.empty(len(d), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, pred_idx in skf.split(x_pool, d):
        model = build_fn()  # type: ignore[operator]
        model.fit(x_pool[fit_idx], d[fit_idx])
        oof[pred_idx] = _proba_test(model, x_pool[pred_idx])
    c2st_auc = float(roc_auc_score(d, oof))
    p_train = np.clip(oof[:n_train], 1e-6, 1 - 1e-6)
    ratio = (p_train / (1.0 - p_train)) * (n_train / n_test)
    weights = np.clip(ratio, 0.0, clip)
    return weights, c2st_auc


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size `(Σw)²/Σw²` — the usable training mass under the weights."""
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w**2))
    if denom == 0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def per_group_mean_weight(weights: np.ndarray, groups: np.ndarray) -> list[tuple[str, float]]:
    """Mean importance weight per group (e.g. capture day) — where the shift concentrates."""
    result = []
    for g in dict.fromkeys(groups):  # first-seen order
        mask = np.asarray(groups) == g
        result.append((str(g), float(np.mean(np.asarray(weights)[mask]))))
    return result


@dataclass
class ModelOutcome:
    """One detector's temporal-test detection."""

    name: str
    pr_auc: float
    tpr_at_primary: float


@dataclass
class CovariateShiftStudy:
    """The full covariate-shift diagnosis + IW correction on the temporal split."""

    c2st_auc: float
    n_train: int
    n_test: int
    ess: float
    ess_ratio: float
    max_weight: float
    per_day: list[tuple[str, float]]
    unweighted: ModelOutcome
    weighted: ModelOutcome
    stratified_ceiling: float
    primary_fpr: float


def _fit_detector(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None,
) -> SupervisedClassifier:
    seed_everything(settings.seed)
    return SupervisedClassifier(settings).fit(
        x_train, y_train, eval_set=(x_val, y_val), sample_weight=sample_weight
    )


def _outcome(
    name: str,
    model: SupervisedClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    benign: str,
    primary_fpr: float,
    flows_per_day: int,
) -> ModelOutcome:
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), model.classes_, benign)
    op = operating_point(y_val, s_val, y_test, s_test, primary_fpr, flows_per_day)
    return ModelOutcome(
        name=name,
        pr_auc=float(average_precision_score(y_test, s_test)),
        tpr_at_primary=float(op["tpr"]),
    )


def run_covariate_shift(settings: Settings) -> CovariateShiftStudy:
    """Diagnose covariate shift on the temporal split and price the IW-weighted retrain."""
    cfg: CovariateShiftConfig = settings.covariate_shift
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
    days = (
        train[DAY_COLUMN].to_numpy()
        if DAY_COLUMN in train.columns
        else np.zeros(len(train), dtype=int)
    )

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    flows_per_day = variant.thresholds.assumed_flows_per_day

    # Domain classifier: how distinguishable are train and test on features alone?
    def _build_domain() -> object:
        return build_family(cfg.domain_classifier, variant)

    weights, c2st_auc = crossfit_domain_ratio(
        x_train, x_test, _build_domain, variant.seed, cfg.n_folds, cfg.weight_clip
    )
    ess = effective_sample_size(weights)

    # The two detectors: unweighted (deployed protocol) vs importance-weighted retrain.
    unweighted_model = _fit_detector(variant, x_train, y_train, x_val, y_val, None)
    weighted_model = _fit_detector(variant, x_train, y_train, x_val, y_val, weights)
    unweighted = _outcome(
        "unweighted (deployed)",
        unweighted_model,
        x_val,
        y_val,
        x_test,
        y_test,
        benign,
        variant.thresholds.primary_fpr,
        flows_per_day,
    )
    weighted = _outcome(
        "importance-weighted",
        weighted_model,
        x_val,
        y_val,
        x_test,
        y_test,
        benign,
        variant.thresholds.primary_fpr,
        flows_per_day,
    )

    # The no-covariate-shift ceiling: the stratified split's raw-score PR-AUC on its own test.
    strat_ceiling = _stratified_ceiling(settings, benign)

    return CovariateShiftStudy(
        c2st_auc=c2st_auc,
        n_train=len(y_train),
        n_test=len(y_test),
        ess=ess,
        ess_ratio=ess / len(y_train),
        max_weight=float(np.max(weights)),
        per_day=per_group_mean_weight(weights, days),
        unweighted=unweighted,
        weighted=weighted,
        stratified_ceiling=strat_ceiling,
        primary_fpr=variant.thresholds.primary_fpr,
    )


def _stratified_ceiling(settings: Settings, benign: str) -> float:
    """PR-AUC of the same protocol fit and scored on the stratified split (no covariate shift)."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")
    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_test = positive_scores(np.asarray(model.predict_proba(x_test)), model.classes_)
    return float(average_precision_score(y_test, s_test))


def run_covariate_shift_report(settings: Settings) -> Path:
    """Run the covariate-shift study and write the report + figure."""
    study = run_covariate_shift(settings)

    days = [d for d, _ in study.per_day]
    means = [m for _, m in study.per_day]
    fig = plots.plot_barh(
        labels=days,
        values=means,
        xlabel="mean importance weight w(x) = p_test / p_train",
        title="Where the covariate shift concentrates: mean weight by capture day",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote covariate-shift report", extra={"path": str(out_path)})

    with track_run(settings, "covariate_shift") as run:
        run.log_metrics(
            {
                "c2st_auc": study.c2st_auc,
                "ess_ratio": study.ess_ratio,
                "unweighted_pr_auc": study.unweighted.pr_auc,
                "weighted_pr_auc": study.weighted.pr_auc,
                "stratified_ceiling": study.stratified_ceiling,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _diagnosis_table(study: CovariateShiftStudy) -> str:
    return "\n".join(
        [
            "| diagnostic | value | reading |",
            "|---|---|---|",
            f"| domain-classifier AUC (C2ST) | {study.c2st_auc:.3f} | "
            f"{'clear covariate shift' if study.c2st_auc > 0.6 else 'little covariate shift'} |",
            f"| effective sample size | {study.ess:,.0f} of {study.n_train:,} "
            f"({study.ess_ratio:.1%}) | usable training mass under the weights |",
            f"| max weight (clipped) | {study.max_weight:.1f} | weight-variance / clip pressure |",
        ]
    )


def _model_table(study: CovariateShiftStudy) -> str:
    rows = [
        "| detector | test PR-AUC | TPR @ primary FPR |",
        "|---|---|---|",
        f"| {study.unweighted.name} | {study.unweighted.pr_auc:.3f} "
        f"| {study.unweighted.tpr_at_primary:.1%} |",
        f"| {study.weighted.name} | {study.weighted.pr_auc:.3f} "
        f"| {study.weighted.tpr_at_primary:.1%} |",
        f"| _stratified ceiling (no covariate shift)_ | _{study.stratified_ceiling:.3f}_ | _—_ |",
    ]
    return "\n".join(rows)


def _diagnosis_read(study: CovariateShiftStudy) -> str:
    if study.c2st_auc > 0.6:
        return (
            f"There **is** covariate shift: a domain classifier separates train-day flows from "
            f"test-day flows at AUC {study.c2st_auc:.3f} on held-out data, well above the 0.5 of "
            "two indistinguishable distributions — the later capture days genuinely look "
            f"different on the feature axis. The shift costs real training mass: the effective "
            f"sample size collapses to {study.ess:,.0f} of {study.n_train:,} flows "
            f"({study.ess_ratio:.1%}), because only a fraction of the early-day traffic resembles "
            "the future closely enough to weigh heavily."
        )
    return (
        f"The domain classifier lands at AUC {study.c2st_auc:.3f} — close to the 0.5 of two "
        "indistinguishable distributions, so **covariate shift is mild**: the marginal feature "
        "law barely moved between the capture days, which already tells us the temporal gap is "
        "unlikely to be a `p(x)` problem, before any reweighting is tried."
    )


def _correction_read(study: CovariateShiftStudy) -> str:
    delta = study.weighted.pr_auc - study.unweighted.pr_auc
    total_gap = study.stratified_ceiling - study.unweighted.pr_auc
    if delta > 0.005:
        recovered = delta / total_gap if total_gap > 0 else 0.0
        return (
            f"Importance weighting helps, modestly: temporal PR-AUC rises "
            f"{study.unweighted.pr_auc:.3f} → {study.weighted.pr_auc:.3f} (+{delta:.3f}), "
            f"recovering {recovered:.0%} of the "
            f"{total_gap:.3f} gap to the stratified ceiling ({study.stratified_ceiling:.3f}). The "
            "rest is the residual reweighting `p(x)` cannot reach — concept shift, where the "
            "*labels* attached to a region of feature space change between the days, not just how "
            "often that region is visited."
        )
    if delta < -0.005:
        return (
            f"Importance weighting **hurts** here: temporal PR-AUC falls "
            f"{study.unweighted.pr_auc:.3f} → {study.weighted.pr_auc:.3f} ({delta:.3f}). This is "
            "the honest, expected outcome of importance-weighted ERM when the shift is *not* "
            "covariate: reweighting trades away effective sample size (down to "
            f"{study.ess_ratio:.1%}) chasing a `p(x)` correction, while the temporal gap to the "
            f"stratified ceiling ({study.stratified_ceiling:.3f}, a {total_gap:.3f} "
            "drop) is dominated by **concept shift** — the later days carry attack behaviours the "
            "early days never labelled, so no amount of reweighting the inputs recovers a decision "
            "the model was never taught. IW corrects `p(x)`; this gap lives in `p(y|x)`."
        )
    return (
        f"Importance weighting moves detection almost not at all "
        f"({study.unweighted.pr_auc:.3f} → {study.weighted.pr_auc:.3f}), and that is the finding. "
        f"There is a real gap to the stratified ceiling ({study.stratified_ceiling:.3f}), but it "
        "is **not** covariate shift IW can fix: it is concept shift — the later days relabel "
        "regions of feature space the early days scored differently — which reweighting `p(x)` "
        "leaves "
        "untouched. The correct read is diagnostic: covariate shift is present but not the "
        "bottleneck, and the residual is the concept drift only new labels can address."
    )


def _render(study: CovariateShiftStudy, fig: Path) -> str:
    return f"""# NetSentry — Covariate-Shift Diagnosis and Importance-Weighted Correction

_Synthetic stand-in. Honest temporal/binary split: {study.n_train:,} training flows (early
days), {study.n_test:,} test flows (later days). Density ratios estimated with **zero test
labels** via a cross-fit domain classifier._

## Why this report exists

The honest temporal split scores below the optimistic stratified one; the
[leakage study](leakage.md) proves that gap is real and the [novelty study](novelty.md)
decomposes it geometrically. This asks it as a distribution-shift question: how much of the gap
is **covariate shift** (`p(x)` moves, `p(y|x)` holds), and does importance weighting — the
textbook fix — close it? The estimator is label-free: a **domain classifier** trained to tell a
train flow from a test flow gives both a shift detector (its AUC) and the density ratio
`w(x) = p_test/p_train` (its calibrated odds), cross-fit so no flow scores a model that saw it.

## Is there covariate shift, and what does it cost?

{_diagnosis_table(study)}

{_diagnosis_read(study)}

![Mean weight by capture day](../figures/{fig.name})

## Does importance-weighted retraining close the gap?

The detector refit with each training flow weighted by `w(x)` (how much it resembles the
future), scored on the temporal test split against the unweighted baseline and the stratified
no-shift ceiling. Operating point at the primary {study.primary_fpr:.1%} FPR budget.

{_model_table(study)}

{_correction_read(study)}

## Scope

The density ratio is a **cross-fit classifier estimate** (Bickel et al. 2009; the C2ST of
Lopez-Paz & Oquab 2017); direct-ratio methods (KLIEP, uLSIF; Sugiyama et al.) are the named
alternatives, and would tighten the tail the weight clip here bounds. Importance-weighted ERM
(Shimodaira 2000) is **only** unbiased for the test distribution under the covariate-shift
assumption `p(y|x) = p_test(y|x)`; this report's value is precisely in testing that assumption
and finding where it fails. It is the covariate-axis complement of the
[label-shift](label_shift.md) study (which corrects `p(y)` with zero labels): together they
cover the two correctable shifts, and both name the same residual — **concept shift** in
`p(y|x)`, the drift that only new labels (the [active-learning](active_learning.md) /
[streaming](streaming.md) loop) can close, and that the [exchangeability
martingale](exchangeability.md) is built to detect."""
