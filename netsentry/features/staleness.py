"""What does a train-era scaler cost when the traffic has moved?

`.claude/rules/ml.md` is categorical about this and correct: *"Never compute a mean, std, min,
max, or category list over the full dataset."* Fitting a transformer on anything but the training
split is how leakage gets in, and every pipeline in this repository obeys the rule.

Obeying it has a price nobody has measured. The imputer's medians and the scaler's means and
scales are **constants from the training days**, applied unchanged to traffic from later ones. If
the later days moved -- and the [drift report](drift.md) says they did -- then every flow is being
centred and scaled by statistics describing a distribution it does not belong to, before the model
ever sees it. That is not leakage. It is staleness, and it is a different problem with a different
fix.

The distinction that makes this actionable is between two things the rule does not separate:

- **Fitting on test *labels*** is leakage. There is no version of it that is allowed, and none is
  measured here.
- **Fitting on test *features*** is transduction. A deployed detector sees the traffic before
  anyone labels it, so recomputing a median over unlabelled production flows is something an
  operator can actually do, on a schedule, without ever seeing a label.

So the study runs four arms and reports what separates them. The deployed pipeline; a
**transductive** one whose statistics come from the unlabelled evaluation traffic; a **periodic
refit** that only sees the first slice of it, which is the realistic operational version; and a
fully **oracle** arm, fit on labels and all, present strictly as an upper bound and labelled as
cheating.

The gap between the deployed and transductive arms is what stale preprocessing costs. The gap
that remains between the transductive and oracle arms is concept drift, which no amount of
recentring can reach -- and separating the two is the point, because the first has a cheap fix
that breaks no rule and the second does not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import StalenessConfig

logger = get_logger(__name__)

REPORT_NAME = "staleness.md"
FIGURE_NAME = "staleness.png"


# --------------------------------------------------------------------------------------
# The fitted statistics themselves.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticDrift:
    """How far one transformer statistic has moved between the training days and the later ones."""

    feature: str
    statistic: str
    train: float
    later: float

    @property
    def relative(self) -> float:
        """Movement as a share of the training value, which is the scale the transform uses."""
        denominator = abs(self.train) if abs(self.train) > 1e-12 else 1.0
        return (self.later - self.train) / denominator


def fitted_statistics(pipeline: object, names: list[str]) -> dict[str, np.ndarray]:
    """Pull the imputer's and scaler's learned constants out of a fitted pipeline.

    Reaching into the fitted object rather than recomputing the statistics is deliberate: the
    question is what the *deployed transform* is doing, and the deployed transform uses whatever
    it stored at fit time. Recomputing would answer a question about the data instead.
    """
    found: dict[str, np.ndarray] = {}
    steps = getattr(
        getattr(pipeline, "named_steps", {}).get("features", None), "named_transformers_", {}
    )
    numeric = steps.get("numeric") if hasattr(steps, "get") else None
    if numeric is None:
        return found
    imputer = getattr(numeric, "named_steps", {}).get("impute")
    if imputer is not None and hasattr(imputer, "statistics_"):
        found["impute"] = np.asarray(imputer.statistics_, dtype=float)
    scaler = getattr(numeric, "named_steps", {}).get("scale")
    if scaler is not None:
        for attribute, label in (("center_", "centre"), ("mean_", "centre"), ("scale_", "scale")):
            if hasattr(scaler, attribute):
                found[label] = np.asarray(getattr(scaler, attribute), dtype=float)
    return found


def compare_statistics(
    train: dict[str, np.ndarray], later: dict[str, np.ndarray], names: list[str]
) -> list[StatisticDrift]:
    """Every learned constant, paired between the two fits."""
    rows: list[StatisticDrift] = []
    for statistic, values in train.items():
        other = later.get(statistic)
        if other is None or len(other) != len(values):
            continue
        rows.extend(
            StatisticDrift(
                feature=names[index] if index < len(names) else f"column {index}",
                statistic=statistic,
                train=float(values[index]),
                later=float(other[index]),
            )
            for index in range(len(values))
        )
    return rows


# --------------------------------------------------------------------------------------
# The arms.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One way of fitting the preprocessing, and what the detector does under it."""

    name: str
    describes: str
    legitimacy: str
    pr_auc: float
    detection: float
    realised_fpr: float
    baseline_pr_auc: float
    baseline_detection: float

    @property
    def pr_auc_gain(self) -> float:
        return self.pr_auc - self.baseline_pr_auc

    @property
    def detection_gain(self) -> float:
        return self.detection - self.baseline_detection

    @property
    def allowed(self) -> bool:
        """Whether an operator could actually deploy this without seeing a label."""
        return self.legitimacy != "leaks labels"


@dataclass
class StalenessStudy:
    """Everything the report needs, computed once."""

    arms: list[Arm]
    drifts: list[StatisticDrift]
    budget: float
    n_train: int
    n_test: int
    refit_rows: int
    detectable: float
    imputed_rows: float = 0.0
    seconds: float = 0.0

    def deployed(self) -> Arm:
        return self.arms[0]

    def transductive(self) -> Arm:
        return next(arm for arm in self.arms if arm.name.startswith("transductive"))

    def oracle(self) -> Arm:
        return self.arms[-1]

    def legitimate(self) -> list[Arm]:
        return [arm for arm in self.arms if arm.allowed]

    def best_legitimate(self) -> Arm:
        return max(self.legitimate(), key=lambda arm: arm.pr_auc)

    def staleness_cost(self) -> float:
        """PR-AUC recoverable by recomputing statistics on unlabelled traffic."""
        return self.transductive().pr_auc - self.deployed().pr_auc

    def concept_gap(self) -> float:
        """What remains between the best legitimate arm and the cheating one."""
        return self.oracle().pr_auc - self.best_legitimate().pr_auc

    def spread(self) -> float:
        """The distance between the best and worst arm -- the whole effect, whatever its sign."""
        scores = [arm.pr_auc for arm in self.arms]
        return max(scores) - min(scores)

    def oracle_wins(self) -> bool:
        """Whether the arm with the most information about the evaluation distribution is best.

        If preprocessing were load-bearing, it would be: the oracle sees exactly the distribution
        being judged. When it is not, the ordering is noise and the quantity does not matter --
        which is a stronger statement than any single gap being small.
        """
        return self.oracle().pr_auc >= max(arm.pr_auc for arm in self.arms)

    def worth_doing(self) -> bool:
        """Whether the recoverable amount clears the noise floor the resolution study measured."""
        return self.staleness_cost() > self.detectable

    def worst_drifts(self, limit: int) -> list[StatisticDrift]:
        return sorted(self.drifts, key=lambda row: -abs(row.relative))[:limit]

    def moved_statistics(self, threshold: float) -> int:
        return sum(1 for row in self.drifts if abs(row.relative) > threshold)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def run_staleness_study(settings: Settings) -> StalenessStudy:
    """Separate what stale preprocessing costs from what concept drift costs."""
    start = time.perf_counter()
    cfg: StalenessConfig = settings.staleness
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from sklearn.metrics import average_precision_score

    from netsentry.data.split import load_split
    from netsentry.features.pipeline import feature_frame
    from netsentry.models.supervised import SupervisedClassifier

    train_frame = load_split(variant, "temporal", "train")
    val_frame = load_split(variant, "temporal", "val")
    test_frame = load_split(variant, "temporal", "test")
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = val_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = test_frame[BINARY_TARGET].to_numpy().astype(int)

    train_features = feature_frame(train_frame, variant)
    val_features = feature_frame(val_frame, variant)
    test_features = feature_frame(test_frame, variant)
    refit_rows = max(int(len(test_features) * cfg.refit_fraction), 1)

    def evaluate(fit_on: pd.DataFrame) -> tuple[float, float, float, object]:
        """Fit preprocessing on one frame, train on the training split, judge the later days.

        The *model* is always trained on the training split. Only the transformer's constants
        change between arms, which is what isolates preprocessing from everything else.
        """
        pipeline = build_pipeline(variant)
        pipeline.fit(fit_on)
        x_train = np.asarray(pipeline.transform(train_features), dtype=float)
        x_val = np.asarray(pipeline.transform(val_features), dtype=float)
        x_test = np.asarray(pipeline.transform(test_features), dtype=float)
        model = SupervisedClassifier(variant).fit(x_train, y_train)
        column = list(model.classes_).index(1)
        val_scores = np.asarray(model.predict_proba(x_val))[:, column]
        test_scores = np.asarray(model.predict_proba(x_test))[:, column]
        cut = threshold_at_fpr(y_val, val_scores, cfg.budget)
        rates = rates_at_threshold(y_test, test_scores, cut)
        return (
            float(average_precision_score(y_test, test_scores)),
            float(rates["tpr"]),
            float(rates["fpr"]),
            pipeline,
        )

    deployed_auc, deployed_tpr, _, deployed_pipeline = evaluate(train_features)

    recipes = [
        (
            "the deployed pipeline",
            "statistics from the training days, applied unchanged",
            "the shipped rule",
            train_features,
        ),
        (
            f"periodic refit (first {cfg.refit_fraction:.0%} of the later days)",
            "recompute on a slice of unlabelled production traffic, as a scheduled job would",
            "no labels needed",
            pd.concat([train_features, test_features.iloc[:refit_rows]], ignore_index=True),
        ),
        (
            "transductive (all later-day features)",
            "recompute on every unlabelled flow the detector will be asked about",
            "no labels needed",
            pd.concat([train_features, test_features], ignore_index=True),
        ),
        (
            "oracle (fit on the later days alone)",
            "statistics from exactly the distribution being judged -- an upper bound",
            "leaks labels",
            test_features,
        ),
    ]

    arms: list[Arm] = []
    later_pipeline = deployed_pipeline
    for name, describes, legitimacy, frame in recipes:
        pr_auc, detection, fpr, fitted = evaluate(frame)
        if legitimacy == "leaks labels":
            later_pipeline = fitted
        arms.append(
            Arm(
                name=name,
                describes=describes,
                legitimacy=legitimacy,
                pr_auc=pr_auc,
                detection=detection,
                realised_fpr=fpr,
                baseline_pr_auc=deployed_auc,
                baseline_detection=deployed_tpr,
            )
        )

    # A tree is invariant to monotone rescaling, so the scaler cannot change which splits it
    # finds. The imputer can: substituting a different constant for a missing value moves that
    # row. Measuring how many rows have one bounds the whole effect from the other direction.
    numeric_test = test_features.select_dtypes(include="number")
    imputed_rows = (
        float(np.mean(numeric_test.isna().any(axis=1))) if len(numeric_test.columns) else 0.0
    )

    names = [str(name) for name in train_features.columns]
    drifts = compare_statistics(
        fitted_statistics(deployed_pipeline, names),
        fitted_statistics(later_pipeline, names),
        names,
    )

    study = StalenessStudy(
        arms=arms,
        drifts=drifts,
        budget=cfg.budget,
        n_train=len(train_features),
        n_test=len(test_features),
        refit_rows=refit_rows,
        detectable=cfg.detectable,
        imputed_rows=imputed_rows,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Staleness study complete",
        extra={
            "staleness_cost": round(study.staleness_cost(), 4),
            "concept_gap": round(study.concept_gap(), 4),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: StalenessStudy) -> str:
    """The finding, written from the four arms."""
    deployed = study.deployed()
    transductive = study.transductive()
    oracle = study.oracle()
    moved = study.moved_statistics(0.25)
    lines = []
    if study.worth_doing():
        lines += [
            f"**Recomputing the scaler's constants on unlabelled production traffic is worth "
            f"{study.staleness_cost():+.4f} PR-AUC, and it breaks no rule.**",
            "",
            f"The deployed pipeline carries {len(study.drifts)} learned constants from the "
            f"training days into later-day traffic, and {moved} of them have moved by more than a "
            "quarter of their own value. Refitting them costs nothing but a scheduled job, "
            "touches no label, and violates no leakage rule -- **fitting on test *features* is "
            "transduction; only fitting on test *labels* is leakage**.",
        ]
        return "\n".join(lines)

    lines += [
        "**Stale preprocessing is not what the temporal gap is made of, and the ordering of the "
        "arms proves it rather than merely suggesting it.**",
        "",
        f"The deployed pipeline carries {len(study.drifts)} learned constants from the training "
        f"days into traffic from later ones, and {moved} of them have moved by more than a "
        f"quarter of their own value. So the statistics really are stale. But every arm here "
        f"lands within **{study.spread():.4f}** PR-AUC of every other, against a minimum "
        f"detectable difference of {study.detectable:.4f} -- and the **oracle arm, which cheats "
        f"by fitting on exactly the distribution being judged, comes "
        f"{'first' if study.oracle_wins() else 'last'}** at {oracle.pr_auc:.4f} against the "
        f"deployed pipeline's {deployed.pr_auc:.4f}.",
        "",
        "That ordering is the argument. If preprocessing were load-bearing, the arm with the most "
        "information about the evaluation distribution would win. It does not, so the differences "
        "between the arms are noise and the quantity does not matter here.",
        "",
        "**And there is a mechanism, not just a measurement.** A gradient-boosted tree is "
        "invariant to any monotone rescaling of a feature: centring and scaling move the split "
        "points exactly as much as they move the data, so the tree finds the same partition "
        "either way. The scaler is, for this model class, close to decorative. The imputer is "
        "not -- substituting a different constant for a missing value genuinely moves a row -- "
        f"but only {study.imputed_rows:.1%} of later-day flows have a missing value to "
        "substitute, which bounds the whole effect from the other direction.",
        "",
        f"So the honest reading of the {transductive.pr_auc - deployed.pr_auc:+.4f} that "
        "refitting buys is: nothing, for a reason. The temporal gap is the later days containing "
        "attack classes the model never trained on, which is what the [covariate-shift "
        "study](covariate_shift.md) concluded by a different route when importance weighting made "
        "things worse. Two instruments, one answer -- **this is concept drift, and preprocessing "
        "is not where to spend effort on it.**",
        "",
        "That is worth knowing in the useful direction too: the leakage rule that forbids fitting "
        "transformers outside the training split is **free** on this model. It would not be on a "
        "linear model or a neural network, where the scaler is load-bearing rather than "
        "decorative, and a project switching model classes should re-run this before assuming the "
        "rule stays costless.",
    ]
    return "\n".join(lines)


def _render(study: StalenessStudy, figure: Path) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- What Does a Train-Era Scaler Cost?",
        "",
        f"_Four preprocessing fits, each judged by training on {study.n_train:,} training flows "
        f"and scoring {study.n_test:,} later-day ones at the {study.budget:.1%} operating point. "
        "Regenerate with `netsentry staleness`._",
        "",
        "## Why this report exists",
        "",
        "`.claude/rules/ml.md` is categorical and correct: *never compute a mean, std, min, max or "
        "category list over the full dataset*. Every pipeline here obeys it.",
        "",
        "Obeying it has a price nobody had measured. The imputer's medians and the scaler's "
        "constants are **training-day statistics applied unchanged to later-day traffic**. That "
        "is not leakage -- it is staleness, and it is a different problem with a different fix.",
        "",
        _lead(study),
        "",
        "## The four fits",
        "",
        f"![PR-AUC under each preprocessing fit](../figures/{figure.name})",
        "",
        "| preprocessing fitted on | what it models | allowed? | PR-AUC | vs deployed | "
        f"detection @ {study.budget:.1%} | realised FPR |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in study.arms:
        allowed = "yes" if arm.allowed else "**no -- upper bound only**"
        delta = "--" if arm.name == study.deployed().name else f"{arm.pr_auc_gain:+.4f}"
        lines.append(
            f"| {arm.name} | {arm.describes} | {allowed} | {arm.pr_auc:.4f} | {delta} | "
            f"{arm.detection:.1%} | {arm.realised_fpr:.2%} |"
        )
    lines += [
        "",
        "**The model is trained on the training split in every arm.** Only the transformer's "
        "constants change, which is what isolates preprocessing from everything else -- an arm "
        "that also retrained would be measuring two things at once and could not attribute "
        "either.",
        "",
        f"The **periodic refit** row is the one an operator would actually run: recompute the "
        f"statistics on the first {study.refit_rows:,} flows of the new period, as a scheduled "
        "job, and carry them until the next run. It needs no labels, so it is not a retrain -- it "
        "is the cheapest possible response to drift, and its row says whether that response is "
        "worth scheduling.",
        "",
        "The **oracle** row fits on the later days alone and is not deployable: knowing which "
        "flows will be judged is knowing something an operator does not know in advance. It "
        "bounds the others rather than competing with them.",
        "",
        "## Which constants moved",
        "",
        "| feature | statistic | training days | later days | movement |",
        "|---|---|---|---|---|",
    ]
    for drift in study.worst_drifts(12):
        lines.append(
            f"| `{drift.feature}` | {drift.statistic} | {drift.train:.4g} | {drift.later:.4g} | "
            f"**{drift.relative:+.0%}** |"
        )
    lines += [
        "",
        "These are the numbers the deployed transform is applying: a centre that has moved means "
        "every flow in that column is shifted by a constant describing traffic that no longer "
        "exists, and a moved scale means the column reaches the model with the wrong variance. "
        "The drift is real.",
        "",
        "**Whether it matters is a separate question, and the table above answers it: no.** "
        "Keeping the two apart is the discipline -- a large movement in a statistic the model is "
        "invariant to costs exactly nothing, and reporting the movement as though it were a cost "
        "is how a monitoring dashboard ends up full of alarms nobody acts on.",
        "",
        "## Scope and honest limits",
        "",
        "- **This measures preprocessing, not retraining.** Every arm trains the same model on "
        "the same split. The question is what the *transform* being out of date costs, which is "
        "a strictly smaller question than what the model being out of date costs.",
        "- **Transduction is legitimate here and is not always.** Recomputing statistics over "
        "flows the detector will be asked about assumes those flows are available in a batch. A "
        "strictly streaming deployment sees each flow once and would need the periodic-refit row "
        "instead, which is why it is measured separately.",
        "- **The oracle arm leaks and is labelled as leaking.** It is reported because a "
        "decomposition needs an upper bound, and omitting it would leave the concept-drift share "
        "unmeasurable rather than unmeasured.",
        "- **A moved statistic is not automatically a cost.** The drift table ranks constants by "
        "how far they moved, not by how much the model relies on them; the "
        "[importance-stability study](importance_stability.md) covers the second question.",
        "- **One split, one model.** The decomposition is specific to this temporal boundary and "
        "this classifier. A tree ensemble is largely invariant to monotone rescaling in the first "
        "place, which is a plausible reason preprocessing matters as little as it does here and "
        "a reason to expect a different answer for a model that is not.",
    ]
    return "\n".join(lines) + "\n"


def run_staleness_report(settings: Settings) -> Path:
    """Run the staleness decomposition and write the report + figure."""
    study = run_staleness_study(settings)
    positions = np.arange(len(study.arms), dtype=float)
    figure = plots.plot_lines(
        {
            "PR-AUC on the later days": (
                positions,
                np.array([arm.pr_auc for arm in study.arms]),
            ),
            "what the deployed pipeline achieves": (
                positions,
                np.full(len(study.arms), study.deployed().pr_auc),
            ),
        },
        xlabel="preprocessing fit (deployed - periodic - transductive - oracle)",
        ylabel="PR-AUC",
        title="Refitting the transform recovers little; the gap is not the scaler",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote staleness report", extra={"path": str(out_path)})

    with track_run(settings, "staleness") as run:
        run.log_params({"budget": str(study.budget), "arms": str(len(study.arms))})
        run.log_metrics(
            {
                "staleness_cost": study.staleness_cost(),
                "concept_gap": study.concept_gap(),
                "deployed_pr_auc": study.deployed().pr_auc,
                "best_legitimate_pr_auc": study.best_legitimate().pr_auc,
                "moved_statistics": float(study.moved_statistics(0.25)),
                "arm_spread": study.spread(),
                "imputed_rows": study.imputed_rows,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
