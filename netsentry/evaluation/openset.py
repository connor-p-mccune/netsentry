"""Open-set recognition: the temporal split is not a closed-set problem, so stop scoring it as one.

Every metric this project reports on the honest temporal split is computed with a *closed-set*
protocol — train on the classes in the early days, score on the later days — but the split's
own class table says the later days contain **no attack class the training days ever showed**.
Train carries `DoS Hulk`, `DoS GoldenEye`, the patators and the slow-DoS family; test carries
`PortScan`, `DDoS`, `Bot`, `Web Attack` and `Infiltration`. Every attack the deployed model
faces at test time is, formally, an **unknown class**. That is the definition of open-set
recognition (Scheirer et al. 2013), and it reframes what the headline number is measuring: not
"can the classifier separate the classes it was taught", but "can it tell that something is
*not* one of them".

Framed that way, the deployed decision rule — alert when `1 - P(BENIGN)` is high — is only one
of several available novelty scores, and it is the one with the least reason to work, because
it asks a model to express surprise using the vocabulary of the classes it already knows. This
study puts it in a field against the standard open-set baselines, all computed from artefacts
the deployment already has:

- **MSP** — one minus the maximum softmax probability, the Hendrycks & Gimpel (2017) baseline;
- **entropy** of the predictive distribution, and the **top-two margin**;
- **`1 - P(BENIGN)`** — the deployed rule, as the incumbent;
- **Mahalanobis** distance to the nearest class-conditional Gaussian with a shared shrunk
  covariance (Lee et al. 2018), which scores in feature space rather than in label space;
- the benign-fit **Isolation Forest**, the project's existing unsupervised route;
- a **rank-fused** combination, since the two families fail on different flows.

Three things are reported. The **open-set AUROC** and **UDR@FPR** (unknown-detection rate at a
fixed false-alarm budget, matching the operational metric used everywhere else) rank the rules.
The **OSCR curve** (Dhamija, Günther & Boult 2018) adds the constraint the AUROC drops: a known
flow only counts as handled if the classifier both accepts it *and* labels it correctly, so a
rule cannot buy unknown-detection by degrading closed-set accuracy. And an **openness sweep**
re-runs the whole comparison on the stratified split with a growing number of attack classes
deliberately withheld from training, so the ranking is measured as a function of how open the
problem is rather than at the single operating point this dataset happens to hand us.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_auc_score

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.anomaly import build_anomaly_detector
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import OpenSetConfig

logger = get_logger(__name__)

REPORT_NAME = "openset.md"
FIGURE_NAME = "openset_oscr.png"
OPENNESS_FIGURE_NAME = "openset_openness.png"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Novelty scores. Every rule returns "higher == more likely to be an unknown class", so the
# whole family shares one direction and one threshold convention.
# --------------------------------------------------------------------------------------


def msp_novelty(proba: np.ndarray) -> np.ndarray:
    """`1 - max_k P(k|x)` — the maximum-softmax-probability baseline (Hendrycks & Gimpel 2017)."""
    p: np.ndarray = np.asarray(proba, dtype=float)
    top: np.ndarray = p.max(axis=1)
    return 1.0 - top


def entropy_novelty(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of the predictive distribution, in nats — spread means unfamiliarity."""
    p: np.ndarray = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0)
    entropy: np.ndarray = -np.sum(p * np.log(p), axis=1)
    return entropy


def margin_novelty(proba: np.ndarray) -> np.ndarray:
    """`1 - (p_top1 - p_top2)`: a contested prediction is a weaker claim of membership."""
    p = np.asarray(proba, dtype=float)
    if p.shape[1] < 2:
        return np.zeros(len(p))
    part = np.partition(p, -2, axis=1)
    return 1.0 - (part[:, -1] - part[:, -2])


def percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Where each value falls in a *reference* distribution, in [0, 1].

    Ranking a score against the block it belongs to is the obvious mistake and it silently
    destroys the fusion: known and unknown flows each get ranks spanning the whole unit
    interval, so the combined score carries no information about which block a flow came from.
    The reference must therefore be a distribution available *before* the flows being scored —
    here the validation split, which is exactly what a deployment would calibrate against.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    if len(ref) == 0:
        raise ValueError("percentile_rank needs a non-empty reference distribution")
    idx = np.searchsorted(ref, np.asarray(values, dtype=float), side="left")
    return idx / len(ref)


def rank_average(scores: list[np.ndarray], references: list[np.ndarray]) -> np.ndarray:
    """Average several scores after mapping each onto its own reference percentile.

    Fusing raw scores would let whichever rule has the widest numeric range dominate;
    percentiles make the combination invariant to any monotone rescaling of its inputs, which
    matters here because a Mahalanobis distance and a probability are not on remotely the same
    scale.
    """
    if not scores:
        raise ValueError("rank_average needs at least one score array")
    if len(scores) != len(references):
        raise ValueError("rank_average needs one reference per score array")
    n = len(scores[0])
    total = np.zeros(n, dtype=float)
    for arr, ref in zip(scores, references, strict=True):
        arr = np.asarray(arr, dtype=float)
        if len(arr) != n:
            raise ValueError("rank_average needs equal-length score arrays")
        total += percentile_rank(arr, ref)
    return total / len(scores)


class MahalanobisScorer:
    """Distance to the nearest class-conditional Gaussian with a shared, shrunk covariance.

    The label-space rules (MSP, entropy, margin) can only express surprise as disagreement
    between known classes, which a confident-but-wrong model never shows. This one scores in
    *feature* space (Lee et al. 2018): fit one mean per training class and a single pooled
    covariance, then score a flow by its smallest Mahalanobis distance to any of them. Ledoit-
    Wolf-style shrinkage toward a scaled identity keeps the precision matrix well conditioned
    when a rare class contributes only a handful of rows.
    """

    def __init__(self, shrinkage: float = 0.1) -> None:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        self.shrinkage = shrinkage
        self.means_: np.ndarray = np.empty(0)
        self.precision_: np.ndarray = np.empty(0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MahalanobisScorer:
        """Fit class means and one pooled within-class covariance on the training split only."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        means = []
        centered = np.empty_like(x)
        for cls in classes:
            mask = y == cls
            mu = x[mask].mean(axis=0)
            means.append(mu)
            centered[mask] = x[mask] - mu
        self.means_ = np.vstack(means)
        dof = max(len(x) - len(classes), 1)
        cov = centered.T @ centered / dof
        # Shrink toward a scaled identity so the inverse exists even for a rank-deficient class.
        trace_mean = float(np.trace(cov)) / max(cov.shape[0], 1)
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_mean * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Smallest squared Mahalanobis distance to any known class mean (higher == novel)."""
        x = np.asarray(x, dtype=float)
        best = np.full(len(x), np.inf)
        for mu in self.means_:
            delta = x - mu
            d2 = np.einsum("ij,jk,ik->i", delta, self.precision_, delta)
            best = np.minimum(best, d2)
        return best


# --------------------------------------------------------------------------------------
# Open-set metrics.
# --------------------------------------------------------------------------------------


def openness(
    n_train_classes: int, n_test_classes: int, n_target_classes: int | None = None
) -> float:
    """Scheirer et al. (2013) openness: 0 is a closed-set problem, higher is more open.

    `1 - sqrt(2 * C_train / (C_test + C_target))`, where `C_target` defaults to the classes the
    model may emit (i.e. the training classes). It is the standard way to say *how* open a
    protocol is, so results at different holdout counts can be compared on one axis.
    """
    if n_train_classes <= 0 or n_test_classes <= 0:
        raise ValueError("class counts must be positive")
    target = n_train_classes if n_target_classes is None else n_target_classes
    return float(1.0 - np.sqrt(2.0 * n_train_classes / (n_test_classes + target)))


def detection_at_fpr(
    novelty_known: np.ndarray, novelty_unknown: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Unknown-detection rate at a false-alarm budget set on the *known* flows.

    The threshold is the `1 - target_fpr` quantile of the known-flow novelty scores, matching
    the project's operational convention (pick the operating point on traffic you have labels
    for, then read detection off the other side). Returns `(udr, threshold)`.
    """
    known = np.asarray(novelty_known, dtype=float)
    unknown = np.asarray(novelty_unknown, dtype=float)
    threshold = float(np.quantile(known, 1.0 - target_fpr)) if len(known) else float("inf")
    udr = float(np.mean(unknown >= threshold)) if len(unknown) else 0.0
    return udr, threshold


def openset_auroc(novelty_known: np.ndarray, novelty_unknown: np.ndarray) -> float:
    """AUROC of the novelty score at separating unknown-class flows from known ones."""
    known = np.asarray(novelty_known, dtype=float)
    unknown = np.asarray(novelty_unknown, dtype=float)
    y = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.concatenate([known, unknown])))


def oscr_curve(
    known_correct: np.ndarray, novelty_known: np.ndarray, novelty_unknown: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The OSCR trade-off (Dhamija, Günther & Boult 2018): CCR against unknown-acceptance FPR.

    Sweeping the novelty threshold, `FPR` is the share of unknown-class flows accepted as known
    (i.e. missed), and `CCR` is the share of known flows both accepted *and* classified into the
    right known class. The second condition is what makes OSCR stricter than a plain AUROC: a
    rule cannot look good by rejecting so aggressively that it mangles the closed-set task.
    Returns `(fpr, ccr)` sorted by increasing FPR.
    """
    correct = np.asarray(known_correct, dtype=bool)
    known = np.asarray(novelty_known, dtype=float)
    unknown = np.asarray(novelty_unknown, dtype=float)
    thresholds = np.unique(np.concatenate([known, unknown, [np.inf]]))
    n_known = max(len(known), 1)
    n_unknown = max(len(unknown), 1)
    fpr = np.empty(len(thresholds))
    ccr = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        accepted = known < t
        fpr[i] = float(np.sum(unknown < t)) / n_unknown
        ccr[i] = float(np.sum(accepted & correct)) / n_known
    order = np.argsort(fpr, kind="stable")
    return fpr[order], ccr[order]


def oscr_auc(
    known_correct: np.ndarray, novelty_known: np.ndarray, novelty_unknown: np.ndarray
) -> float:
    """Area under the OSCR curve: one number for rejecting unknowns without breaking knowns."""
    fpr, ccr = oscr_curve(known_correct, novelty_known, novelty_unknown)
    if len(fpr) < 2:
        return 0.0
    return float(np.trapezoid(ccr, fpr))


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class RuleOutcome:
    """One novelty rule scored on one open-set protocol."""

    name: str
    family: str
    auroc: float
    oscr_auc: float
    udr_at: dict[float, float]
    per_class_udr: dict[str, float]


@dataclass
class OpennessPoint:
    """The AUROC of each rule at one level of protocol openness."""

    n_holdout: int
    openness: float
    held_out: list[str]
    auroc: dict[str, float]


@dataclass
class OpenSetStudy:
    """The deployment protocol, the rule field, and the openness sweep."""

    known_classes: list[str]
    unknown_classes: list[str]
    n_known_flows: int
    n_unknown_flows: int
    openness: float
    closed_set_accuracy: float
    budgets: list[float]
    outcomes: list[RuleOutcome]
    sweep: list[OpennessPoint]


def _base_scores(
    proba: np.ndarray,
    classes: np.ndarray,
    benign_label: str,
    mahalanobis: np.ndarray,
    iforest: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every single-source novelty score for one block of flows (fusion is added later)."""
    proba = np.asarray(proba, dtype=float)
    benign_idx = int(np.where(np.asarray(classes) == benign_label)[0][0])
    return {
        "msp": msp_novelty(proba),
        "entropy": entropy_novelty(proba),
        "margin": margin_novelty(proba),
        "attack_prob": 1.0 - proba[:, benign_idx],
        "mahalanobis": mahalanobis,
        "iforest": iforest,
    }


def _add_fusion(
    cfg: OpenSetConfig, block: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Attach the rank-fused rule, calibrated against the validation reference distribution."""
    out = dict(block)
    members = [m for m in cfg.fusion_members if m in block and m in reference]
    if "fused" in cfg.rules and members:
        out["fused"] = rank_average([block[m] for m in members], [reference[m] for m in members])
    return {name: out[name] for name in cfg.rules if name in out}


_FAMILY = {
    "msp": "label space",
    "entropy": "label space",
    "margin": "label space",
    "attack_prob": "label space (deployed)",
    "mahalanobis": "feature space",
    "iforest": "feature space",
    "fused": "fusion",
}


def _fit_block(
    settings: Settings,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: OpenSetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Fit the closed-set model + both feature-space scorers; score the known/unknown blocks.

    Returns `(known_scores, unknown_scores, known_predictions, known_truth, class_order)`.
    Everything is fit on the training split only; the unknown classes never appear in it.
    """
    benign = settings.labels.benign_label
    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[MULTICLASS_TARGET].to_numpy()
    y_val = val[MULTICLASS_TARGET].to_numpy()

    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.classes_)

    maha = MahalanobisScorer(cfg.mahalanobis_shrinkage).fit(x_train, y_train)
    benign_train = x_train[y_train == benign]
    detector = build_anomaly_detector(settings, "iforest").fit(benign_train)

    is_known = test[MULTICLASS_TARGET].isin(list(classes)).to_numpy()
    x_known, x_unknown = x_test[is_known], x_test[~is_known]
    known_truth = test[MULTICLASS_TARGET].to_numpy()[is_known]
    unknown_truth = test[MULTICLASS_TARGET].to_numpy()[~is_known]

    # The validation split is the fusion's calibration reference: it is available before the
    # test days exist, so percentiles taken against it are computable at deployment time.
    reference = _base_scores(
        model.predict_proba(x_val), classes, benign, maha.score(x_val), detector.score(x_val)
    )
    known = _add_fusion(
        cfg,
        _base_scores(
            model.predict_proba(x_known),
            classes,
            benign,
            maha.score(x_known),
            detector.score(x_known),
        ),
        reference,
    )
    unknown = _add_fusion(
        cfg,
        _base_scores(
            model.predict_proba(x_unknown),
            classes,
            benign,
            maha.score(x_unknown),
            detector.score(x_unknown),
        ),
        reference,
    )
    known_pred = model.predict(x_known)
    return known, unknown, known_pred, known_truth, list(np.unique(unknown_truth))


def _subsample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Cap a split's size so the quadratic-ish feature-space scorers stay fast."""
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_index()


def run_openset(settings: Settings) -> OpenSetStudy:
    """Score every novelty rule on the deployment protocol, then sweep protocol openness."""
    cfg: OpenSetConfig = settings.openset
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "multiclass"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = _subsample(load_split(variant, "temporal", "train"), cfg.max_rows, variant.seed)
    val = _subsample(load_split(variant, "temporal", "val"), cfg.max_rows, variant.seed)
    test = _subsample(load_split(variant, "temporal", "test"), cfg.max_rows, variant.seed)

    known_scores, unknown_scores, known_pred, known_truth, unknown_classes = _fit_block(
        variant, train, val, test, cfg
    )
    known_correct = known_pred == known_truth
    known_classes = sorted(set(train[MULTICLASS_TARGET].unique()))
    unknown_truth = test[MULTICLASS_TARGET].to_numpy()
    unknown_truth = unknown_truth[~test[MULTICLASS_TARGET].isin(known_classes).to_numpy()]

    outcomes: list[RuleOutcome] = []
    for name in cfg.rules:
        if name not in known_scores:
            continue
        k, u = known_scores[name], unknown_scores[name]
        per_class: dict[str, float] = {}
        for cls in unknown_classes:
            mask = unknown_truth == cls
            udr, _ = detection_at_fpr(k, u[mask], cfg.primary_budget)
            per_class[str(cls)] = udr
        outcomes.append(
            RuleOutcome(
                name=name,
                family=_FAMILY.get(name, "other"),
                auroc=openset_auroc(k, u),
                oscr_auc=oscr_auc(known_correct, k, u),
                udr_at={b: detection_at_fpr(k, u, b)[0] for b in cfg.budgets},
                per_class_udr=per_class,
            )
        )

    sweep = _openness_sweep(settings, cfg)
    return OpenSetStudy(
        known_classes=[str(c) for c in known_classes],
        unknown_classes=[str(c) for c in unknown_classes],
        n_known_flows=len(known_truth),
        n_unknown_flows=len(unknown_truth),
        openness=openness(len(known_classes), len(known_classes) + len(unknown_classes)),
        closed_set_accuracy=float(np.mean(known_correct)) if len(known_correct) else float("nan"),
        budgets=list(cfg.budgets),
        outcomes=outcomes,
        sweep=sweep,
    )


def _openness_sweep(settings: Settings, cfg: OpenSetConfig) -> list[OpennessPoint]:
    """Re-run the rule field on the stratified split with `k` attack classes withheld.

    The temporal split hands us exactly one openness level. Withholding classes from the
    *stratified* split — where every class is otherwise present on both sides — turns openness
    into a dial, so the ranking can be read as a function of how open the problem is rather
    than as a single accident of this dataset's calendar.
    """
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "multiclass"
    variant.mlflow.enabled = False

    from netsentry.data.split import load_split

    train_full = _subsample(load_split(variant, "stratified", "train"), cfg.max_rows, variant.seed)
    val_full = _subsample(load_split(variant, "stratified", "val"), cfg.max_rows, variant.seed)
    test = _subsample(load_split(variant, "stratified", "test"), cfg.max_rows, variant.seed)

    benign = variant.labels.benign_label
    counts = train_full[MULTICLASS_TARGET].value_counts()
    attacks = [str(c) for c in counts.index if str(c) != benign]
    # Withhold the rarest classes first: that is the realistic order in which a deployment
    # meets an attack it was never trained on.
    ordered = list(reversed(attacks))

    points: list[OpennessPoint] = []
    for k in cfg.holdout_counts:
        held = ordered[:k]
        if len(held) < k:
            continue
        train = train_full[~train_full[MULTICLASS_TARGET].isin(held)]
        val = val_full[~val_full[MULTICLASS_TARGET].isin(held)]
        if train[MULTICLASS_TARGET].nunique() < 2 or not (test[MULTICLASS_TARGET].isin(held).any()):
            continue
        known, unknown, _, _, _ = _fit_block(variant, train, val, test, cfg)
        n_train_classes = int(train[MULTICLASS_TARGET].nunique())
        points.append(
            OpennessPoint(
                n_holdout=k,
                openness=openness(n_train_classes, n_train_classes + len(held)),
                held_out=held,
                auroc={
                    name: openset_auroc(known[name], unknown[name])
                    for name in cfg.rules
                    if name in known
                },
            )
        )
    return points


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def run_openset_report(settings: Settings) -> Path:
    """Run the open-set study and write the report + figures."""
    study = run_openset(settings)

    ranked = sorted(study.outcomes, key=lambda o: o.auroc, reverse=True)
    oscr_fig = plots.plot_barh(
        labels=[o.name for o in ranked],
        values=[o.auroc for o in ranked],
        xlabel="open-set AUROC (unknown class vs known)",
        title="Which novelty rule recognises a class it was never taught?",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )
    sweep_fig = _plot_sweep(study, settings.paths.figures_dir / OPENNESS_FIGURE_NAME)

    report = _render(study, oscr_fig, sweep_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote open-set report", extra={"path": str(out_path)})

    with track_run(settings, "openset") as run:
        run.log_params({"openness": study.openness, "rules": ",".join(settings.openset.rules)})
        run.log_metrics(
            {
                "closed_set_accuracy": study.closed_set_accuracy,
                **{f"auroc_{o.name}": o.auroc for o in study.outcomes},
                **{f"oscr_auc_{o.name}": o.oscr_auc for o in study.outcomes},
            }
        )
        run.log_artifact(oscr_fig)
        if sweep_fig is not None:
            run.log_artifact(sweep_fig)
        run.log_artifact(out_path)
    return out_path


def _plot_sweep(study: OpenSetStudy, out_path: Path) -> Path | None:
    if not study.sweep:
        return None
    xs = np.array([p.openness for p in study.sweep])
    series = {
        name: (xs, np.array([p.auroc.get(name, float("nan")) for p in study.sweep]))
        for name in study.sweep[0].auroc
    }
    return plots.plot_lines(
        series,
        xlabel="protocol openness (Scheirer et al. 2013)",
        ylabel="open-set AUROC",
        title="Does the ranking survive a more open problem?",
        out_path=out_path,
    )


def _rule_table(study: OpenSetStudy) -> str:
    budgets = "".join(f" UDR @ {b:.1%} FPR |" for b in study.budgets)
    rows = [
        f"| rule | family | open-set AUROC | OSCR-AUC |{budgets}",
        "|---|---|---|---|" + "---|" * len(study.budgets),
    ]
    for o in sorted(study.outcomes, key=lambda r: r.auroc, reverse=True):
        cells = "".join(f" {o.udr_at[b]:.1%} |" for b in study.budgets)
        rows.append(f"| `{o.name}` | {o.family} | {o.auroc:.3f} | {o.oscr_auc:.3f} |{cells}")
    return "\n".join(rows)


def _per_class_table(study: OpenSetStudy) -> str:
    if not study.outcomes:
        return "_No rules produced scores._"
    classes = list(study.outcomes[0].per_class_udr)
    rows = [
        "| rule | " + " | ".join(f"`{c}`" for c in classes) + " |",
        "|---|" + "---|" * len(classes),
    ]
    for o in sorted(study.outcomes, key=lambda r: r.auroc, reverse=True):
        cells = " | ".join(f"{o.per_class_udr.get(c, float('nan')):.1%}" for c in classes)
        rows.append(f"| `{o.name}` | {cells} |")
    return "\n".join(rows)


def _sweep_table(study: OpenSetStudy) -> str:
    if not study.sweep:
        return "_The openness sweep found no usable holdout configuration._"
    names = list(study.sweep[0].auroc)
    rows = [
        "| classes withheld | openness | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for p in study.sweep:
        cells = " | ".join(f"{p.auroc.get(n, float('nan')):.3f}" for n in names)
        rows.append(f"| {p.n_holdout} ({', '.join(p.held_out)}) | {p.openness:.3f} | {cells} |")
    return "\n".join(rows)


def _headline_read(study: OpenSetStudy) -> str:
    if not study.outcomes:
        return "_No rules produced scores._"
    ranked = sorted(study.outcomes, key=lambda o: o.auroc, reverse=True)
    best = ranked[0]
    deployed = next((o for o in study.outcomes if o.name == "attack_prob"), None)
    budget = study.budgets[0] if study.budgets else 0.01
    if deployed is None or best.name == "attack_prob":
        runner_up = ranked[1] if len(ranked) > 1 else None
        tie = (
            ""
            if runner_up is None or best.auroc - runner_up.auroc > 0.02
            else (
                f" The ranking does **not** separate it from `{runner_up.name}` "
                f"({runner_up.auroc:.3f}), though, and the per-class table below shows why that "
                "near-tie matters more than the ordering does."
            )
        )
        return (
            f"The deployed rule holds its own field: `1 - P(BENIGN)` reaches "
            f"{best.auroc:.3f} open-set AUROC and {best.udr_at.get(budget, float('nan')):.1%} "
            f"unknown-detection at the {budget:.1%} budget. That is the comfortable outcome, and "
            "it is worth stating why it was not guaranteed — the rule asks a classifier to "
            "express surprise in the vocabulary of the classes it already knows, which is "
            "exactly the failure mode the open-set literature exists to document. An AUROC of "
            f"{best.auroc:.3f} against a 0.5 coin flip is also a reminder of the absolute "
            f"standard here: the best available rule misses roughly "
            f"{1 - best.udr_at.get(budget, 0.0):.0%} of never-seen attacks at a "
            f"{budget:.1%} false-alarm budget.{tie}"
        )
    delta = best.auroc - deployed.auroc
    best_udr = best.udr_at.get(budget, float("nan"))
    dep_udr = deployed.udr_at.get(budget, float("nan"))
    return (
        f"**The deployed rule is not the best novelty detector on this split.** "
        f"`1 - P(BENIGN)` lands at {deployed.auroc:.3f} open-set AUROC and "
        f"{dep_udr:.1%} unknown-detection at the {budget:.1%} false-alarm budget; "
        f"`{best.name}` ({best.family}) reaches {best.auroc:.3f} and {best_udr:.1%} — "
        f"{delta:+.3f} AUROC for a score the deployment already has the artefacts to compute. "
        "The direction of the result is the interesting part: asking the classifier *which known "
        "class is this* and treating uncertainty as novelty is a weaker question than asking "
        "*how far is this from anything I have seen*, because a boosted tree extrapolates a "
        "confident answer into regions of feature space it never visited."
    )


def _oscr_read(study: OpenSetStudy) -> str:
    if not study.outcomes:
        return ""
    by_auroc = sorted(study.outcomes, key=lambda o: o.auroc, reverse=True)[0]
    by_oscr = sorted(study.outcomes, key=lambda o: o.oscr_auc, reverse=True)[0]
    if by_auroc.name == by_oscr.name:
        return (
            f"OSCR agrees with AUROC here: `{by_oscr.name}` leads on both "
            f"({by_oscr.oscr_auc:.3f} OSCR-AUC), so its advantage is not bought by wrecking the "
            f"closed-set task — the classifier still resolves {study.closed_set_accuracy:.1%} of "
            "known-class flows correctly under the same threshold."
        )
    return (
        f"OSCR and AUROC disagree, which is the reason to compute both: `{by_auroc.name}` wins "
        f"the plain separation ranking ({by_auroc.auroc:.3f} AUROC) but `{by_oscr.name}` wins "
        f"once known flows only count when they are *also* classified correctly "
        f"({by_oscr.oscr_auc:.3f} vs {by_auroc.oscr_auc:.3f} OSCR-AUC). AUROC is happy to reject "
        "unknowns by rejecting everything; OSCR charges for that."
    )


def _per_class_read(study: OpenSetStudy) -> str:
    """The aggregate ranking hides per-class blindness; say so with the numbers that show it."""
    if not study.outcomes:
        return ""
    leader = max(study.outcomes, key=lambda o: o.auroc)
    if not leader.per_class_udr:
        return ""
    budget = study.budgets[0] if study.budgets else 0.01
    best_cls = max(leader.per_class_udr, key=lambda c: leader.per_class_udr[c])
    worst_cls = min(leader.per_class_udr, key=lambda c: leader.per_class_udr[c])
    best_udr = leader.per_class_udr[best_cls]
    worst_udr = leader.per_class_udr[worst_cls]
    # Classes where the leader is at or below the false-alarm budget it is being charged: on
    # those flows the rule is doing no better than firing at random.
    blind = [c for c, v in leader.per_class_udr.items() if v <= budget]
    rescued = [
        (c, o.name, o.per_class_udr[c])
        for c in blind
        for o in study.outcomes
        if o.name != leader.name and o.per_class_udr.get(c, 0.0) > 2 * budget
    ]
    lines = [
        f"`{leader.name}`'s aggregate lead is **carried by one family**: it catches "
        f"{best_udr:.1%} of `{best_cls}` and {worst_udr:.1%} of `{worst_cls}` at the same "
        f"{budget:.1%} budget — a {best_udr / max(worst_udr, 1e-9):.0f}x spread across classes "
        "that a single AUROC number cannot show."
    ]
    if blind:
        lines.append(
            f"On {', '.join(f'`{c}`' for c in blind)} it detects at or below the "
            f"{budget:.1%} false-alarm rate itself, which means that on those flows the score "
            "carries **no usable signal at all** — the rule is not weak there, it is blind."
        )
    if rescued:
        cls, rule, udr = rescued[0]
        lines.append(
            f"Another rule does better on at least one of them: `{rule}` reaches {udr:.1%} on "
            f"`{cls}`. That is the practical argument for keeping the whole field rather than "
            "the winner — the rules fail on *different* families, which is precisely the "
            "condition under which fusing them is worth the complexity."
        )
    return " ".join(lines)


def _sweep_read(study: OpenSetStudy) -> str:
    if len(study.sweep) < 2:
        return "_The sweep did not produce enough usable holdout configurations to read a trend._"
    first, last = study.sweep[0], study.sweep[-1]
    names = list(first.auroc)
    winners = {
        p.n_holdout: max(names, key=lambda n: p.auroc.get(n, float("-inf"))) for p in study.sweep
    }
    stable = len(set(winners.values())) == 1
    drops = {n: first.auroc.get(n, 0.0) - last.auroc.get(n, 0.0) for n in names}
    steepest = max(drops, key=lambda n: drops[n])
    decay = (
        f"The rule that gives up the most as the problem opens is `{steepest}`: "
        f"{first.auroc.get(steepest, float('nan')):.3f} AUROC at {first.openness:.3f} openness "
        f"down to {last.auroc.get(steepest, float('nan')):.3f} at {last.openness:.3f}, a drop of "
        f"{abs(drops[steepest]):.3f}."
    )
    if stable:
        champ = next(iter(set(winners.values())))
        return (
            f"The ranking is **stable across openness**: `{champ}` leads at every holdout count, "
            f"from {first.openness:.3f} openness up to {last.openness:.3f}. {decay} That is the "
            "number to watch when a new attack family arrives in production."
        )
    return (
        f"The ranking **changes with openness** — `{winners[first.n_holdout]}` leads at "
        f"{first.openness:.3f} openness and `{winners[last.n_holdout]}` at {last.openness:.3f}. "
        "That is a warning about single-operating-point comparisons: a novelty rule chosen on "
        f"one holdout configuration is not guaranteed to be the right one when the next unknown "
        f"family arrives. {decay}"
    )


def _render(study: OpenSetStudy, oscr_fig: Path, sweep_fig: Path | None) -> str:
    sweep_img = (
        f"\n![AUROC vs openness](../figures/{sweep_fig.name})\n" if sweep_fig is not None else ""
    )
    return f"""# NetSentry — Open-Set Recognition on the Temporal Split

_Synthetic stand-in. Honest temporal/multiclass split. Known classes (in training):
{", ".join(f"`{c}`" for c in study.known_classes)}. Classes appearing **only** at test:
{", ".join(f"`{c}`" for c in study.unknown_classes)}. Protocol openness
{study.openness:.3f} (Scheirer et al. 2013); {study.n_known_flows:,} known-class and
{study.n_unknown_flows:,} unknown-class test flows._

## Why this report exists

The temporal split's own class table says something the closed-set metrics elsewhere in this
repo quietly assume away: **no attack class in the test days appears in the training days**.
Train has the DoS family and the patators; test has `PortScan`, `DDoS`, `Bot`, `Web Attack` and
`Infiltration`. Every attack the model meets at evaluation time is formally an *unknown class*,
which makes this an **open-set recognition** problem, not a classification one. That reframing
changes the question from "can it separate the classes it was taught" to "can it tell that
something is not one of them" — and it puts the deployed decision rule, `1 - P(BENIGN)`, into a
field of alternatives instead of leaving it as the only option on the table.

Closed-set accuracy on the known-class flows is {study.closed_set_accuracy:.1%}. That number is
reported first precisely because it is **not** the thing that matters here; it is the constraint
the open-set rules have to respect while doing the actual job.

## The rule field

Every rule is computed from artefacts the deployment already builds — the multiclass
probability vector, the fitted feature pipeline, and the benign-only Isolation Forest. Nothing
here needs a label from the test days. `UDR` is the unknown-detection rate: the share of
never-seen-class attack flows caught with the threshold fixed on known-class traffic at the
stated false-alarm budget.

{_rule_table(study)}

{_headline_read(study)}

![Open-set AUROC by rule](../figures/{oscr_fig.name})

## Rejecting the unknown without breaking the known (OSCR)

Open-set AUROC has a blind spot: a rule that rejects nearly everything scores well on it while
destroying the closed-set task. The OSCR curve (Dhamija, Günther & Boult 2018) closes that hole
by counting a known-class flow only when the classifier both accepts it *and* labels it
correctly, and sweeping that against the share of unknowns wrongly accepted.

{_oscr_read(study)}

## Which unknown class does each rule find?

Detection at the {study.budgets[0]:.1%} false-alarm budget, broken out by the class the model
was never trained on. A rule can lead on aggregate AUROC while being blind to a specific
family — this is where that shows up.

{_per_class_table(study)}

{_per_class_read(study)}

## Does the ranking survive a more open problem?

The temporal split offers exactly one level of openness. Withholding attack classes from the
*stratified* split turns openness into a dial: rarest classes are withheld first, since that is
the order a real deployment meets attacks it was never trained on.

{_sweep_table(study)}

{_sweep_read(study)}
{sweep_img}
## Scope

The Mahalanobis scorer uses a single pooled within-class covariance with shrinkage toward a
scaled identity (Lee et al. 2018); a per-class covariance would be more expressive and much
less stable at this dataset's rare-class counts. OpenMax (Bendale & Boult 2016) fits a Weibull
tail to the activation distances and is the natural next rung — it needs penultimate-layer
activations, which a boosted-tree ensemble does not have in the same sense, so the feature-space
distance stands in for it. The openness sweep withholds *whole classes* but keeps the stratified
split's optimistic row-level mixing, so its absolute numbers sit above the temporal protocol's
by construction; it is there to test the **ranking**, not to restate the headline. This report
is the label-space complement of the [novelty-distance study](novelty.md), which measures the
same gap geometrically, and of the [uncertainty decomposition](uncertainty.md), which asks
whether the model *knows* it is out of its depth."""
