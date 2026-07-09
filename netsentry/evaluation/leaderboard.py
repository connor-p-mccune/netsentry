"""Model-family leaderboard: every family, one honest protocol.

Most IDS papers compare models; far fewer hold the *protocol* fixed while doing
it. This study runs a spectrum of model families — majority prior, naive Bayes,
logistic regression, random forest, and the project's gradient-boosted model —
through the **identical** harness: same persisted splits, same leakage-safe
pipeline fit on train only, same validation-chosen thresholds, same raw-score
PR-AUC the headline evaluation reports. Each family is scored on both the honest
temporal split and the optimistic stratified split, so the table answers two
questions at once:

1. **Does model choice matter?** The within-split spread across families.
2. **Does the split gap replicate?** If every family — linear to boosted —
   shows a large stratified-minus-temporal gap, the gap is a property of the
   *evaluation*, not a quirk of one model; picking a fancier architecture will
   not close what is fundamentally a distribution shift.

Wall-clock fit time is reported per family because in a retraining loop (see the
streaming and retrain-policy studies) training cost is an operational parameter,
not trivia.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier, resolve_backend
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "leaderboard.md"
SPLITS = ("temporal", "stratified")


@dataclass
class FamilyOutcome:
    """One model family's numbers on one split, under the shared protocol."""

    family: str
    split: str
    pr_auc: float
    tpr_primary: float  # detection at the primary (tightest) FPR budget
    tpr_secondary: float  # detection at the looser budget
    fit_seconds: float


def build_family(name: str, settings: Settings) -> Any:
    """Instantiate a model family by name (all sklearn-API, all seeded)."""
    class_weight = "balanced" if settings.supervised.class_weight == "balanced" else None
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "naive_bayes":
        return GaussianNB()
    if name == "logistic":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=settings.seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=settings.leaderboard.rf_n_estimators,
            class_weight=class_weight,
            random_state=settings.seed,
            n_jobs=settings.supervised.n_jobs,
        )
    if name == "gbdt":
        return SupervisedClassifier(settings)
    raise KeyError(f"unknown model family {name!r}")


def family_label(name: str, settings: Settings) -> str:
    """Human name for the table; the boosted row names its actual backend."""
    if name == "gbdt":
        backend = "LightGBM" if resolve_backend(settings) == "lightgbm" else "HistGradientBoosting"
        return f"gradient boosting ({backend}) — deployed"
    return {
        "majority": "majority prior",
        "naive_bayes": "Gaussian naive Bayes",
        "logistic": "logistic regression",
        "random_forest": "random forest",
    }.get(name, name)


def evaluate_family(
    name: str,
    settings: Settings,
    split: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> FamilyOutcome:
    """Fit one family and score it at the shared operating points."""
    benign = settings.labels.benign_label
    seed_everything(settings.seed)
    model = build_family(name, settings)
    started = time.perf_counter()
    if isinstance(model, SupervisedClassifier):
        model.fit(x_train, y_train, eval_set=(x_val, y_val))
    else:
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    classes = np.asarray(model.classes_)
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), classes, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), classes, benign)

    primary, secondary = sorted(settings.thresholds.fpr_targets)[:2]
    tpr_primary = rates_at_threshold(y_test, s_test, threshold_at_fpr(y_val, s_val, primary))["tpr"]
    tpr_secondary = rates_at_threshold(y_test, s_test, threshold_at_fpr(y_val, s_val, secondary))[
        "tpr"
    ]
    outcome = FamilyOutcome(
        family=family_label(name, settings),
        split=split,
        pr_auc=float(average_precision_score(y_test, s_test)),
        tpr_primary=tpr_primary,
        tpr_secondary=tpr_secondary,
        fit_seconds=fit_seconds,
    )
    logger.info(
        "Leaderboard entry",
        extra={"family": name, "split": split, "pr_auc": round(outcome.pr_auc, 4)},
    )
    return outcome


def run_leaderboard_report(settings: Settings) -> Path:
    """Run every family on both splits under the shared protocol; write the report."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    families = variant.leaderboard.families

    outcomes: list[FamilyOutcome] = []
    prevalence: dict[str, float] = {}
    for split in SPLITS:
        train = load_split(variant, split, "train")
        val = load_split(variant, split, "val")
        test = load_split(variant, split, "test")
        pipeline = build_pipeline(variant)  # refit per split: statistics from train only
        x_train = pipeline.fit_transform(train)
        x_val = pipeline.transform(val)
        x_test = pipeline.transform(test)
        y_train = train[BINARY_TARGET].to_numpy()
        y_val = val[BINARY_TARGET].to_numpy()
        y_test = test[BINARY_TARGET].to_numpy()
        prevalence[split] = float(y_test.mean())
        for name in families:
            outcomes.append(
                evaluate_family(
                    name, variant, split, x_train, y_train, x_val, y_val, x_test, y_test
                )
            )

    fig = _plot(outcomes, variant.paths.figures_dir / "leaderboard.png")
    report = _render(outcomes, prevalence, variant, fig)
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote leaderboard report", extra={"path": str(out_path)})

    with track_run(settings, "leaderboard") as run:
        run.log_metrics(
            {f"{o.split}_{o.family.split(' ')[0]}_pr_auc": round(o.pr_auc, 4) for o in outcomes}
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _by_split(outcomes: list[FamilyOutcome], split: str) -> list[FamilyOutcome]:
    return [o for o in outcomes if o.split == split]


def _plot(outcomes: list[FamilyOutcome], out_path: Path) -> Path:
    """Grouped bars: each family's PR-AUC on both splits (the gap made visible)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temporal = _by_split(outcomes, "temporal")
    stratified = {o.family: o for o in _by_split(outcomes, "stratified")}
    names = [o.family for o in temporal]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(
        x - width / 2,
        [o.pr_auc for o in temporal],
        width,
        label="temporal (honest)",
        color="#3b7dd8",
    )
    ax.bar(
        x + width / 2,
        [stratified[n].pr_auc if n in stratified else np.nan for n in names],
        width,
        label="stratified (optimistic)",
        color="#d1495b",
    )
    ax.set_xticks(
        x, [n.replace(" — deployed", "\n(deployed)") for n in names], fontsize=8, rotation=15
    )
    ax.set(ylabel="test PR-AUC (raw scores)", title="Model families under one honest protocol")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _render(
    outcomes: list[FamilyOutcome],
    prevalence: dict[str, float],
    settings: Settings,
    fig: Path,
) -> str:
    primary, secondary = sorted(settings.thresholds.fpr_targets)[:2]

    def table(split: str) -> str:
        rows = [
            f"| family | PR-AUC | TPR@{primary:.1%} FPR | TPR@{secondary:.0%} FPR | fit (s) |",
            "|---|---|---|---|---|",
        ]
        ranked = sorted(_by_split(outcomes, split), key=lambda o: o.pr_auc, reverse=True)
        for o in ranked:
            rows.append(
                f"| {o.family} | {o.pr_auc:.3f} | {o.tpr_primary:.1%} "
                f"| {o.tpr_secondary:.1%} | {o.fit_seconds:.1f} |"
            )
        return "\n".join(rows)

    temporal = {o.family: o for o in _by_split(outcomes, "temporal")}
    stratified = {o.family: o for o in _by_split(outcomes, "stratified")}
    gap_rows = ["| family | temporal PR-AUC | stratified PR-AUC | gap |", "|---|---|---|---|"]
    gaps: list[float] = []
    for name, t in temporal.items():
        s = stratified.get(name)
        if s is None or name == "majority prior":
            continue
        gaps.append(s.pr_auc - t.pr_auc)
        gap_rows.append(
            f"| {name} | {t.pr_auc:.3f} | {s.pr_auc:.3f} | **{s.pr_auc - t.pr_auc:+.3f}** |"
        )

    real = [o for o in _by_split(outcomes, "temporal") if o.family != "majority prior"]
    spread = max(o.pr_auc for o in real) - min(o.pr_auc for o in real)
    min_gap = min(gaps) if gaps else 0.0

    # If the two splits crown different winners, that inversion is the sharpest
    # finding on the page — say it, with the capacity mechanism, only when true.
    best_temporal = max(real, key=lambda o: o.pr_auc).family
    strat_real = [o for o in _by_split(outcomes, "stratified") if o.family != "majority prior"]
    best_stratified = max(strat_real, key=lambda o: o.pr_auc).family if strat_real else ""
    inversion = ""
    if strat_real and best_temporal != best_stratified:
        inversion = (
            f" The two splits even crown different winners — **{best_stratified}** leads the "
            f"optimistic table while **{best_temporal}** leads the honest one — the classic "
            "capacity trade under distribution shift: flexible models fit the training-day "
            "regime tightly and pay for it on later days (the gap column prices that "
            "capacity). A model selected on the optimistic split would have been the wrong "
            "model to ship."
        )

    if gaps and min_gap > spread:
        read = (
            f"Every family — linear to boosted — pays a stratified-minus-temporal gap of at "
            f"least **{min_gap:+.3f}**, larger than the entire spread between families on the "
            f"honest split ({spread:.3f}). **Choosing the evaluation honestly matters more than "
            "choosing the model**: a leaky protocol would hand any of these architectures a "
            "better-looking number than the best architecture earns on the honest one. The gap "
            "is a property of the split (near-duplicate attack bursts landing on both sides), "
            "not a deficiency one more model family would fix." + inversion
        )
    elif gaps:
        read = (
            f"The stratified-minus-temporal gap replicates across families (min {min_gap:+.3f}), "
            f"and the within-split family spread is {spread:.3f}. Both levers are real here: "
            "the split inflates every family, and family choice moves the honest number too."
            + inversion
        )
    else:
        read = f"Within-split family spread: {spread:.3f}."
    return f"""# NetSentry — Model-Family Leaderboard (one honest protocol)

_Synthetic stand-in. Every family runs through the identical harness: the same
persisted splits, the same leakage-safe pipeline fit on train only, thresholds
chosen on validation at the {primary:.1%} / {secondary:.0%} FPR budgets, PR-AUC on
raw test scores (the headline's scale). Attack prevalence:
temporal {prevalence.get("temporal", float("nan")):.1%}, stratified
{prevalence.get("stratified", float("nan")):.1%} (PR-AUC baselines)._

## Temporal split (the honest table)

{table("temporal")}

## Stratified split (the optimistic reference)

{table("stratified")}

## The gap, per family

{chr(10).join(gap_rows)}

![Leaderboard](../figures/{fig.name})

## Read

{read}

## Scope

Families run at sensible defaults (config: `leaderboard.*`); only the deployed
gradient-boosted model carries tuned hyperparameters, so the comparison favors
it — the honest claim is about the *protocol*, not that these baselines were
tuned to their ceilings. Fit time is single-machine wall clock, an operational
input to the retraining-cadence studies rather than a benchmark.
"""
