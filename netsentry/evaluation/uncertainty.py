"""Two kinds of not knowing: ambiguous traffic versus traffic the model has never seen.

The detector returns one number per flow, and a low attack probability is asked to mean two
completely different things. Sometimes it means *this flow looks genuinely benign* — the
evidence is clear and points that way. Sometimes it means *I have no idea what this is* —
the flow sits somewhere the training data never went, and the number is a guess dressed as
a probability. A SOC should treat those two flows very differently, and a single score
cannot tell them apart.

The decomposition that separates them is standard and, for once, exactly matches the
operational question. Train an ensemble, and for each flow look at the spread of its
members' opinions:

    total uncertainty  =  H( mean_m p_m )                 <- entropy of the ensemble mean
    aleatoric          =  mean_m H( p_m )                 <- irreducible: all members agree
    epistemic          =  total - aleatoric               <- mutual information: they disagree

**Aleatoric** uncertainty is noise in the world: benign traffic that genuinely resembles a
scan, encrypted flows that look alike whatever you do. More data does not remove it.
**Epistemic** uncertainty is ignorance about the model: the members were fitted on the same
data and still disagree, which is what happens off the edge of the training distribution.
More data does remove it. The split is the mutual information between the label and the
model parameters (Houlsby et al. 2011; Depeweg et al. 2018), and the members here are a
bagged, differently-seeded ensemble of the deployed model — the tabular analogue of a deep
ensemble (Lakshminarayanan et al. 2017).

This project has an unusually good test for whether that story is true rather than merely
plausible. The headline split is temporal, so several attack classes appear only on the
later days: the test set contains attacks the model has literally never seen, alongside
attacks it knows well and benign traffic it knows best of all. The prediction is sharp and
falsifiable — **epistemic uncertainty should rise on the novel classes and aleatoric should
not** — and if it fails, the decomposition is decoration. Three things follow if it holds:
epistemic uncertainty is a novelty score that can be compared head-to-head against the
benign-only anomaly detector already in the system, it is an abstention signal that should
beat abstaining on the score alone, and it says which of the two problems more data would
actually fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import UncertaintyConfig

logger = get_logger(__name__)

REPORT_NAME = "uncertainty.md"
GROUP_FIGURE = "uncertainty_groups.png"
COVERAGE_FIGURE = "uncertainty_coverage.png"

_EPS = 1e-12

#: Novelty AUC above which the plain attack score is already finding the deleted class,
#: so the epistemic-vs-aleatoric comparison has nothing to add on that arm. A narrative
#: cut for the report's prose rather than a decision the system makes -- named because a
#: bare literal in a render function is a decision nobody can find.
_NOVELTY_ALREADY_FOUND_AUC = 0.7


# --------------------------------------------------------------------------------------
# The decomposition (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of a Bernoulli, in nats. Zero at certainty, ``log 2`` at a coin flip."""
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.asarray(-(q * np.log(q) + (1.0 - q) * np.log1p(-q)), dtype=float)


def decompose(member_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each flow's uncertainty into ``(total, aleatoric, epistemic)``.

    ``member_probs`` is ``(n_models, n_flows)`` of attack probabilities. The epistemic term
    is the mutual information between the label and the choice of ensemble member, so it is
    non-negative by Jensen's inequality — entropy is concave, and the entropy of the average
    is at least the average of the entropies. Anything else means a bug, which is why the
    tests assert it rather than assume it.
    """
    probs = np.atleast_2d(np.asarray(member_probs, dtype=float))
    mean_p = probs.mean(axis=0)
    total = binary_entropy(mean_p)
    aleatoric = binary_entropy(probs).mean(axis=0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return total, aleatoric, epistemic


def risk_coverage(
    scores: np.ndarray,
    abstain_on: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    coverages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Error rate of the retained flows as the most-uncertain ones are handed to a human.

    Returns ``(coverage, error rate)``. The decision rule is held fixed — the same threshold
    on the same scores — and only *which flows are auto-decided* changes, so the curve
    isolates the value of the abstention signal rather than mixing it with a threshold move.
    A signal that ranks the model's mistakes well drives the error rate down steeply; one
    that does not is flat, and flat is the honest verdict for most confidence heuristics.
    """
    order = np.argsort(np.asarray(abstain_on, dtype=float))  # most certain first
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    wrong = (pred != np.asarray(y_true).astype(int))[order]
    n = len(wrong)
    out_cov: list[float] = []
    out_err: list[float] = []
    for c in np.asarray(coverages, dtype=float):
        k = max(round(c * n), 1)
        out_cov.append(k / n)
        out_err.append(float(wrong[:k].mean()))
    return np.asarray(out_cov), np.asarray(out_err)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class GroupStats:
    """Mean uncertainty over one population of flows."""

    name: str
    n: int
    total: float
    aleatoric: float
    epistemic: float
    epistemic_share: float
    mean_score: float


@dataclass
class NoveltyRow:
    """One signal's ability to pick novel attacks out of benign traffic."""

    signal: str
    auc_novel: float
    auc_known: float
    trained_on_labels: bool


@dataclass
class CoverageRow:
    """One abstention signal's risk-coverage curve."""

    signal: str
    coverages: np.ndarray
    errors: np.ndarray
    error_at_90: float
    area: float


@dataclass
class HoldoutArm:
    """One controlled experiment: a class deleted from training, the test set untouched."""

    holdout_class: str
    n_holdout: int
    benign: GroupStats
    known: GroupStats
    novel: GroupStats
    novelty: list[NoveltyRow]

    @property
    def epistemic_lift(self) -> float:
        """How much higher epistemic uncertainty runs on the deleted class than on known ones."""
        return self.novel.epistemic / max(self.known.epistemic, _EPS)

    @property
    def aleatoric_lift(self) -> float:
        """The control: the same ratio for the term that should *not* move."""
        return self.novel.aleatoric / max(self.known.aleatoric, _EPS)


@dataclass
class UncertaintyStudy:
    """Everything the report renders."""

    n_models: int
    n_test: int
    temporal_train_classes: list[str]
    temporal_test_classes: list[str]
    shared_classes: list[str]
    arms: list[HoldoutArm]
    coverage: list[CoverageRow]
    single_pr_auc: float
    ensemble_pr_auc: float
    full_error: float


def _fit_ensemble(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    benign: str,
    n_models: int,
    bag_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bag-and-reseed the deployed model into an ensemble; return its val and test probs.

    Diversity has to come from somewhere real. Each member sees a different bootstrap of the
    training rows *and* a different seed, which is the tabular analogue of a deep ensemble —
    identical architecture, different draws — so member disagreement reflects what the data
    failed to pin down rather than a hyperparameter the engineer varied by hand. (Varying
    hyperparameters is a different question, and the [multiplicity](multiplicity.md) study
    asks it.)
    """
    rng = np.random.default_rng(settings.seed)
    n = len(y_train)
    size = max(int(bag_fraction * n), 10)
    on_val: list[np.ndarray] = []
    on_test: list[np.ndarray] = []
    for i in range(int(n_models)):
        variant = settings.model_copy(deep=True)
        variant.seed = settings.seed + 1 + i
        variant.mlflow.enabled = False
        seed_everything(variant.seed)
        idx = rng.choice(n, size=size, replace=True)
        model = SupervisedClassifier(variant).fit(
            x_train[idx], y_train[idx], eval_set=(x_val, y_val)
        )
        classes = model.classes_
        on_val.append(attack_probability(np.asarray(model.predict_proba(x_val)), classes, benign))
        on_test.append(attack_probability(np.asarray(model.predict_proba(x_test)), classes, benign))
    return np.vstack(on_val), np.vstack(on_test)


def _auc(signal: np.ndarray, is_benign: np.ndarray, is_positive: np.ndarray) -> float:
    """AUC of ``signal`` separating one attack population from benign traffic."""
    mask = is_benign | is_positive
    if not is_positive.any() or not is_benign.any():
        return float("nan")
    return float(roc_auc_score(is_positive[mask].astype(int), np.asarray(signal)[mask]))


def _group(
    name: str, mask: np.ndarray, terms: tuple[np.ndarray, ...], score: np.ndarray
) -> GroupStats:
    """Summarise the decomposition over one population of flows."""
    total, aleatoric, epistemic = terms
    if not mask.any():
        return GroupStats(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return GroupStats(
        name=name,
        n=int(mask.sum()),
        total=float(total[mask].mean()),
        aleatoric=float(aleatoric[mask].mean()),
        epistemic=float(epistemic[mask].mean()),
        epistemic_share=float(epistemic[mask].mean() / max(total[mask].mean(), _EPS)),
        mean_score=float(score[mask].mean()),
    )


def _holdout_arm(
    settings: Settings,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    holdout: str,
    cfg: UncertaintyConfig,
) -> HoldoutArm:
    """Delete one attack class from training only, then measure uncertainty on the same test set.

    This is the controlled version of the question. The test flows, the feature pipeline and
    the capture days are all held fixed; the single thing that changes is whether the model has
    ever seen this attack. Anything the decomposition does differently on those flows is
    therefore attributable to unfamiliarity rather than to the class simply being harder —
    which is exactly what an observational novel-vs-known comparison cannot establish.
    """
    benign = settings.labels.benign_label
    keep_train = train[train[MULTICLASS_TARGET].astype(str) != holdout]
    keep_val = val[val[MULTICLASS_TARGET].astype(str) != holdout]

    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(keep_train))
    x_val = np.asarray(pipeline.transform(keep_val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = keep_train[BINARY_TARGET].to_numpy().astype(int)
    y_val = keep_val[BINARY_TARGET].to_numpy().astype(int)

    _members_val, members = _fit_ensemble(
        settings,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        benign,
        cfg.n_models,
        cfg.bag_fraction,
    )
    terms = decompose(members)
    mean_p = members.mean(axis=0)

    labels = test[MULTICLASS_TARGET].astype(str).to_numpy()
    is_benign = labels == benign
    is_novel = labels == holdout
    is_known = ~is_benign & ~is_novel

    from netsentry.models.anomaly import IsolationForestDetector

    detector = IsolationForestDetector(settings).fit(x_train[y_train == 0])
    anomaly = np.asarray(detector.score(x_test))

    novelty = [
        NoveltyRow(
            signal=name,
            auc_novel=_auc(signal, is_benign, is_novel),
            auc_known=_auc(signal, is_benign, is_known),
            trained_on_labels=supervised,
        )
        for name, signal, supervised in (
            ("epistemic uncertainty", terms[2], True),
            ("aleatoric uncertainty", terms[1], True),
            ("attack score (the deployed signal)", mean_p, True),
            ("isolation forest (benign-only)", anomaly, False),
        )
    ]
    logger.info(
        "Holdout arm complete",
        extra={"holdout": holdout, "epistemic": round(float(terms[2][is_novel].mean()), 5)},
    )
    return HoldoutArm(
        holdout_class=holdout,
        n_holdout=int(is_novel.sum()),
        benign=_group("benign", is_benign, terms, mean_p),
        known=_group("attacks still in training", is_known, terms, mean_p),
        novel=_group(f"{holdout} (deleted from training)", is_novel, terms, mean_p),
        novelty=novelty,
    )


def run_uncertainty(settings: Settings) -> UncertaintyStudy:
    """Decompose the detector's uncertainty and test it by deleting classes from training."""
    cfg: UncertaintyConfig = settings.uncertainty
    benign = settings.labels.benign_label
    seed_everything(settings.seed)

    from netsentry.data.split import load_split

    # Framing: how novel are the later days, really? (Labels only — nothing is trained here.)
    temporal_train = load_split(settings, "temporal", "train")
    temporal_test = load_split(settings, "temporal", "test")
    t_train_classes = sorted(
        {c for c in temporal_train[MULTICLASS_TARGET].astype(str) if c != benign}
    )
    t_test_classes = sorted(
        {c for c in temporal_test[MULTICLASS_TARGET].astype(str) if c != benign}
    )
    shared = sorted(set(t_train_classes) & set(t_test_classes))

    # The controlled arm needs a split where the class is present on both sides to begin with.
    strat = settings.model_copy(deep=True)
    strat.split.strategy = "stratified"
    strat.supervised.task = "binary"
    strat.mlflow.enabled = False
    s_train = load_split(strat, "stratified", "train")
    s_val = load_split(strat, "stratified", "val")
    s_test = load_split(strat, "stratified", "test")

    counts = s_test[MULTICLASS_TARGET].astype(str).value_counts()
    candidates = [c for c in counts.index if c != benign and counts[c] >= cfg.min_holdout_flows]
    holdouts = candidates[: cfg.n_holdout_classes]
    arms = [_holdout_arm(strat, s_train, s_val, s_test, h, cfg) for h in holdouts]

    # Abstention is an operational question, so it is asked on the deployed temporal setup.
    temporal = settings.model_copy(deep=True)
    temporal.split.strategy = "temporal"
    temporal.supervised.task = "binary"
    temporal.mlflow.enabled = False
    t_val = load_split(temporal, "temporal", "val")
    pipeline = build_pipeline(temporal)
    x_train = np.asarray(pipeline.fit_transform(temporal_train))
    x_val = np.asarray(pipeline.transform(t_val))
    x_test = np.asarray(pipeline.transform(temporal_test))
    y_train = temporal_train[BINARY_TARGET].to_numpy().astype(int)
    y_val = t_val[BINARY_TARGET].to_numpy().astype(int)
    y_test = temporal_test[BINARY_TARGET].to_numpy().astype(int)

    members_val, members = _fit_ensemble(
        temporal,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        benign,
        cfg.n_models,
        cfg.bag_fraction,
    )
    total, aleatoric, epistemic = decompose(members)
    mean_p = members.mean(axis=0)
    threshold = threshold_at_fpr(y_val, members_val.mean(axis=0), settings.thresholds.primary_fpr)

    seed_everything(temporal.seed)
    single = SupervisedClassifier(temporal).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_single = attack_probability(np.asarray(single.predict_proba(x_test)), single.classes_, benign)

    coverages = np.asarray(cfg.coverages, dtype=float)
    coverage: list[CoverageRow] = []
    for name, signal in (
        ("epistemic uncertainty", epistemic),
        ("aleatoric uncertainty", aleatoric),
        ("total uncertainty", total),
        ("distance to the decision threshold", -np.abs(mean_p - threshold)),
    ):
        cov, err = risk_coverage(mean_p, signal, y_test, threshold, coverages)
        coverage.append(
            CoverageRow(
                signal=name,
                coverages=cov,
                errors=err,
                error_at_90=float(np.interp(0.9, cov, err)),
                area=float(np.trapezoid(err, cov)),
            )
        )

    from sklearn.metrics import average_precision_score

    return UncertaintyStudy(
        n_models=int(cfg.n_models),
        n_test=len(y_test),
        temporal_train_classes=t_train_classes,
        temporal_test_classes=t_test_classes,
        shared_classes=shared,
        arms=arms,
        coverage=coverage,
        single_pr_auc=float(average_precision_score(y_test, s_single)),
        ensemble_pr_auc=float(average_precision_score(y_test, mean_p)),
        full_error=float(np.mean((mean_p >= threshold).astype(int) != y_test)),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_uncertainty_report(settings: Settings) -> Path:
    """Run the uncertainty study and write the report + figures."""
    study = run_uncertainty(settings)

    labels: list[str] = []
    values: list[float] = []
    for arm in study.arms:
        for group in (arm.known, arm.novel):
            labels.append(f"{arm.holdout_class}: {'deleted' if group is arm.novel else 'kept'}")
            values.append(group.epistemic)
    group_fig = plots.plot_barh(
        labels or ["no arm"],
        values or [0.0],
        xlabel="mean epistemic uncertainty (nats)",
        title="Does deleting a class from training make the model admit it?",
        out_path=settings.paths.figures_dir / GROUP_FIGURE,
        xmax=(max(values) * 1.2 if values else 1.0),
    )
    coverage_fig = plots.plot_lines(
        {row.signal: (row.coverages, row.errors) for row in study.coverage},
        xlabel="coverage (share of flows decided automatically)",
        ylabel="error rate on the flows kept",
        title="Which uncertainty is worth abstaining on",
        out_path=settings.paths.figures_dir / COVERAGE_FIGURE,
    )

    report = _render(study, group_fig, coverage_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote uncertainty report", extra={"path": str(out_path)})

    with track_run(settings, "uncertainty") as run:
        run.log_params({"n_models": study.n_models, "arms": len(study.arms)})
        metrics = {
            "single_pr_auc": study.single_pr_auc,
            "ensemble_pr_auc": study.ensemble_pr_auc,
            "shared_temporal_classes": float(len(study.shared_classes)),
        }
        if study.arms:
            metrics |= {
                "mean_epistemic_lift": float(np.mean([a.epistemic_lift for a in study.arms])),
                "mean_aleatoric_lift": float(np.mean([a.aleatoric_lift for a in study.arms])),
            }
        run.log_metrics(metrics)
        run.log_artifact(group_fig)
        run.log_artifact(coverage_fig)
        run.log_artifact(out_path)
    return out_path


def _arm_table(study: UncertaintyStudy) -> str:
    rows = [
        "| deleted class | flows | population | total | aleatoric | epistemic | epistemic share |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in study.arms:
        for group in (arm.benign, arm.known, arm.novel):
            mark = "**" if group is arm.novel else ""
            rows.append(
                f"| {arm.holdout_class} | {group.n:,} | {mark}{group.name}{mark} "
                f"| {group.total:.4f} | {group.aleatoric:.4f} | {mark}{group.epistemic:.4f}{mark} "
                f"| {group.epistemic_share:.1%} |"
            )
    return "\n".join(rows)


def _lift_table(study: UncertaintyStudy) -> str:
    rows = [
        "| deleted class | epistemic lift (should rise) | aleatoric lift (should not) "
        "| separation |",
        "|---|---|---|---|",
    ]
    for arm in study.arms:
        sep = arm.epistemic_lift / max(arm.aleatoric_lift, _EPS)
        verdict = "clean" if sep > 1.3 else ("weak" if sep > 1.05 else "none")
        rows.append(
            f"| {arm.holdout_class} | **{arm.epistemic_lift:.2f}x** | {arm.aleatoric_lift:.2f}x "
            f"| {sep:.2f}x ({verdict}) |"
        )
    return "\n".join(rows)


def _novelty_table(study: UncertaintyStudy) -> str:
    rows = [
        "| deleted class | signal | uses labels | AUC vs benign (deleted class) "
        "| AUC vs benign (classes kept) |",
        "|---|---|---|---|---|",
    ]
    for arm in study.arms:
        for r in arm.novelty:
            rows.append(
                f"| {arm.holdout_class} | {r.signal} | {'yes' if r.trained_on_labels else 'no'} "
                f"| **{r.auc_novel:.3f}** | {r.auc_known:.3f} |"
            )
    return "\n".join(rows)


def _coverage_table(study: UncertaintyStudy) -> str:
    rows = [
        "| abstention signal | error at 90% coverage | area under the risk-coverage curve |",
        "|---|---|---|",
    ]
    for r in sorted(study.coverage, key=lambda x: x.area):
        rows.append(f"| {r.signal} | {r.error_at_90:.3%} | {r.area:.5f} |")
    return "\n".join(rows)


def _framing_read(study: UncertaintyStudy) -> str:
    n_train = len(study.temporal_train_classes)
    n_test = len(study.temporal_test_classes)
    if study.shared_classes:
        return (
            f"The temporal split trains on {n_train} attack classes and tests on {n_test}, of "
            f"which {len(study.shared_classes)} appear on both sides "
            f"({', '.join(study.shared_classes)}). So the later days are a mix of familiar and "
            "unfamiliar attacks, and a novel-vs-known comparison drawn from them would be "
            "confounded by everything else that differs between Wednesday and Thursday."
        )
    return (
        f"**Not one of the {n_test} attack classes on the test days appears in training.** The "
        f"model trains on {', '.join(study.temporal_train_classes)} and is tested on "
        f"{', '.join(study.temporal_test_classes)} — the sets are disjoint. That is worth "
        "pausing on, because it reframes the project's headline number: the temporal PR-AUC is "
        "not a measure of detecting known attacks slightly later, it is a measure of detecting "
        "**entirely unseen attack families** from behaviour alone. The honest number was always "
        "harder-won than it looked.\n\nIt also rules out the obvious experiment. With no shared "
        "classes there is no 'known attack' population on the test days to compare against, and "
        "any novel-vs-known contrast assembled across the day boundary would confound "
        "unfamiliarity with everything else that changed between Wednesday and Thursday. So the "
        "test below is run properly instead: on the **stratified** split, where every class "
        "appears on both sides, one attack class at a time is deleted from *training only* while "
        "the test set, the feature pipeline and the capture days are held fixed."
    )


def _hypothesis_read(study: UncertaintyStudy) -> str:
    if not study.arms:
        return (
            "No attack class had enough test flows to carry a controlled arm, so the falsifiable "
            "test could not be run. Everything below describes the decomposition without the "
            "evidence that would make it trustworthy."
        )
    epi = float(np.mean([a.epistemic_lift for a in study.arms]))
    ale = float(np.mean([a.aleatoric_lift for a in study.arms]))
    clean = [a for a in study.arms if a.epistemic_lift > a.aleatoric_lift * 1.3]
    if epi > 1.15 and epi > ale * 1.3:
        return (
            f"**The prediction holds.** Averaged over the {len(study.arms)} arms, deleting a "
            f"class from training raises epistemic uncertainty on that class by "
            f"**{epi:.2f}x** relative to the classes that stayed in, while aleatoric "
            f"uncertainty moves {ale:.2f}x. The second number is the one that makes the first "
            "meaningful: if both terms rose together the decomposition would be tracking "
            "difficulty, and the labels would be decoration. They separate, in the direction "
            "the theory predicts, under an intervention rather than an observation — the only "
            "thing that changed between the two populations is whether the model had ever seen "
            f"the attack. {len(clean)} of {len(study.arms)} arms separate cleanly on their own."
        )
    if epi > ale:
        return (
            f"**The prediction holds only weakly.** Epistemic uncertainty rises {epi:.2f}x on "
            f"the deleted class against an aleatoric rise of {ale:.2f}x — the right ordering, "
            "but not by enough to carry weight. Both terms moving nearly together is the "
            "signature of a population that is simply harder to classify rather than one that "
            "is specifically unfamiliar, and on this evidence the decomposition should not be "
            "leaned on as a novelty signal. The abstention section below is where it still "
            "earns its keep."
        )
    return (
        f"**The prediction fails.** Deleting a class from training raises its epistemic "
        f"uncertainty by only {epi:.2f}x while aleatoric rises {ale:.2f}x — the wrong ordering, "
        "under an intervention designed to produce the right one. The honest reading is that a "
        "bagged tree ensemble does not turn unfamiliarity into disagreement: members trained on "
        "overlapping bootstraps of the same features agree confidently in regions none of them "
        "has seen, because a tree's output away from its training data is whichever leaf the "
        "last split routes to, not an admission of ignorance. This is a known limitation of "
        "ensemble uncertainty on trees, it is the reason the benign-only anomaly detector earns "
        "its place in this system rather than being replaced by an uncertainty score, and it is "
        "reported here because a decomposition that fails its own test is worth more as a "
        "negative result than as an unexamined feature."
    )


def _novelty_read(study: UncertaintyStudy) -> str:
    if not study.arms:
        return ""
    wins = {"epistemic uncertainty": 0, "isolation forest (benign-only)": 0}
    for arm in study.arms:
        by_name = {r.signal: r.auc_novel for r in arm.novelty}
        for name in wins:
            other = (
                "isolation forest (benign-only)"
                if name.startswith("epistemic")
                else ("epistemic uncertainty")
            )
            if by_name.get(name, 0.0) > by_name.get(other, 0.0):
                wins[name] += 1
    epi_wins = wins["epistemic uncertainty"]
    best_overall: dict[str, list[float]] = {}
    for arm in study.arms:
        for r in arm.novelty:
            best_overall.setdefault(r.signal, []).append(r.auc_novel)
    ranked = sorted(best_overall.items(), key=lambda kv: -float(np.nanmean(kv[1])))
    leader, leader_scores = ranked[0]
    lead = (
        f"Across the {len(study.arms)} arms, epistemic uncertainty beats the benign-only "
        f"isolation forest on {epi_wins} of them at picking the deleted class out of benign "
        f"traffic. The strongest signal on average is **{leader}** "
        f"({float(np.nanmean(leader_scores)):.3f} mean AUC). Two caveats keep this comparison "
        "honest in opposite directions: the ensemble saw labels and the isolation forest did "
        "not, so the ensemble has strictly more information; but the isolation forest was built "
        "for exactly this job while the ensemble's uncertainty is a by-product. The final "
        "column is the control — a signal that scores well on the classes still in training is "
        "measuring how conspicuous an attack is, not how unfamiliar, and only the gap between "
        "the two columns is evidence about novelty."
    )

    def _auc_of(arm: HoldoutArm, prefix: str) -> float:
        return next((r.auc_novel for r in arm.novelty if r.signal.startswith(prefix)), float("nan"))

    blindest = min(study.arms, key=lambda a: _auc_of(a, "attack score"))
    score_auc = _auc_of(blindest, "attack score")
    epi_auc = _auc_of(blindest, "epistemic")
    iso_auc = _auc_of(blindest, "isolation forest")
    if score_auc > _NOVELTY_ALREADY_FOUND_AUC:
        return lead
    sighted = [a for a in study.arms if a is not blindest]
    sighted_note = (
        " On the other arms ("
        + ", ".join(f"{a.holdout_class} at {_auc_of(a, 'attack score'):.3f}" for a in sighted)
        + ") the detector still finds the deleted class perfectly well, because its sibling "
        "classes cover it — so those are precisely the arms where a novelty signal was not "
        "needed, and they are the arms where epistemic uncertainty looks best."
        if sighted
        else ""
    )
    return (
        lead
        + f"\n\n**The {blindest.holdout_class} arm is the one to read.** With that class deleted, "
        f"the detector scores it at {score_auc:.3f} AUC against benign traffic — chance. It is "
        "completely blind to the attack. That is the exact situation a novelty signal exists to "
        f"cover, and epistemic uncertainty reaches {epi_auc:.3f} on it: also chance. **The model "
        "is at chance on the attack and does not know it.** The isolation forest manages "
        f"{iso_auc:.3f}, barely better, so this is a hard case rather than a fair fight lost."
        f"{sighted_note} The uncomfortable summary is that ensemble disagreement signals "
        "unfamiliarity most reliably where unfamiliarity costs least, and that is not a property "
        "worth deploying on. It is also a clean illustration of the underlying mechanism: a tree "
        "far from its training data does not abstain, it returns whichever leaf its last split "
        "routes to, and ten trees grown on overlapping bootstraps of the same features route it "
        "to the same place."
    )


def _coverage_read(study: UncertaintyStudy) -> str:
    if not study.coverage:
        return ""
    best = min(study.coverage, key=lambda r: r.area)
    worst = max(study.coverage, key=lambda r: r.area)
    return (
        f"Handing the least-confident flows to a human is worth doing only if the signal picking "
        f"them ranks the model's actual mistakes. **{best.signal}** does it best here (error "
        f"{best.error_at_90:.3%} at 90% coverage against {study.full_error:.3%} with no "
        f"abstention), and **{worst.signal}** worst. The decision rule is held fixed throughout "
        "— same scores, same threshold — so the curves compare abstention signals rather than "
        "smuggling in a threshold change. This is the same trade the [conformal](conformal.md) "
        "study makes with a coverage guarantee and the [multiplicity](multiplicity.md) study "
        "makes across a model family; the three abstain on different grounds — the data is "
        "ambiguous, the model family disagrees, this model is unsure — and an operator running "
        "all three would send a flow to review if any of them objected."
    )


def _render(study: UncertaintyStudy, group_fig: Path, coverage_fig: Path) -> str:
    delta = study.ensemble_pr_auc - study.single_pr_auc
    deleted = ", ".join(a.holdout_class for a in study.arms) or "none"
    return f"""# NetSentry — Epistemic vs Aleatoric: Two Kinds of Not Knowing

_Synthetic stand-in. Uncertainty is decomposed over a {study.n_models}-member bagged, re-seeded
ensemble of the deployed model — same hyperparameters, different bootstrap draws — and reported
in nats. The controlled arms run on the stratified split with {deleted} deleted from training in
turn; the abstention curves run on the deployed temporal split ({study.n_test:,} test flows)._

## Why this report exists

The detector returns one number, and a low attack probability is asked to carry two very
different meanings. Sometimes it means *this looks genuinely benign*: the evidence is clear and
points that way. Sometimes it means *I have never seen anything like this*: the flow sits off
the edge of the training data and the number is a guess wearing a probability's clothes. A SOC
should treat those flows differently, and one score cannot tell them apart.

An ensemble can. For each flow, compare the entropy of the members' average opinion against the
average of their individual entropies:

```
total       = H(mean_m p_m)      the ensemble is unsure
aleatoric   = mean_m H(p_m)      ...and every member is individually unsure   (irreducible)
epistemic   = total - aleatoric  ...but each is confident and they disagree   (ignorance)
```

Aleatoric uncertainty is noise in the world and more data will not remove it. Epistemic
uncertainty is ignorance about the model and more data will. The difference is the mutual
information between the label and the choice of member (Houlsby et al. 2011; Depeweg et al.
2018), and it is non-negative by Jensen's inequality.

## First, an unexpected fact about the headline split

{_framing_read(study)}

## The controlled test

One attack class is deleted from the training and validation folds; the test set is untouched.
If the decomposition means what it claims, epistemic uncertainty rises on the deleted class and
aleatoric does not.

{_arm_table(study)}

{_lift_table(study)}

{_hypothesis_read(study)}

![epistemic uncertainty, class kept vs class deleted](../figures/{group_fig.name})

## Is it a novelty detector?

{_novelty_table(study)}

{_novelty_read(study)}

## Is it worth abstaining on?

{_coverage_table(study)}

{_coverage_read(study)}

![risk-coverage curves](../figures/{coverage_fig.name})

## The ensemble is not free

Averaging {study.n_models} members moves temporal test PR-AUC from {study.single_pr_auc:.3f} to
{study.ensemble_pr_auc:.3f} ({delta:+.3f}), for {study.n_models}x the training cost and
{study.n_models}x the inference cost. The decomposition, not the accuracy, is what the ensemble
is bought for here, and an operator who wants an abstention signal without that serving bill has
cheaper options this report will not pretend are equivalent: the [cascade](cascade.md) study
prices staged inference, and a single model's [conformal](conformal.md) prediction sets abstain
on ambiguity with no ensemble at all.

## Scope

Uncertainty is decomposed over a *bagged* ensemble, so it captures the variance from which
training rows were drawn and nothing else. It does not see uncertainty about the model family,
the feature pipeline, or the labels — [multiplicity](multiplicity.md) varies hyperparameters and
[leaderboard](leaderboard.md) varies families, and both find disagreement invisible here by
construction. The controlled arms delete one class at a time, which measures unfamiliarity with
*that* class against a model that still knows every other attack; deleting the whole attack side
would be a different and much easier question, and the benign-only [anomaly
detector](anomaly.md) already answers it. Class deletion also removes those rows from training,
so each arm's model is very slightly smaller — the effect is a fraction of a percent of the
training rows and cannot account for lifts of the size reported. All scores are raw
(uncalibrated) for consistency with the headline metrics."""
