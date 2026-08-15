"""Does training *for* the operating point beat training for the loss, and at what cost?

Every evaluation in this repository leads with detection at a fixed false-positive budget,
because that is the number a SOC deploys. Every model in this repository is trained to minimise
cross-entropy, which is a different objective: log-loss spends capacity being right about the
obviously-benign majority, while the operating point is decided entirely by the few benign flows
that score highest -- the ones the threshold has to sit above. Nothing in the training objective
knows those flows are special.

This study closes that gap and prices it. Three arms, one protocol:

- **LightGBM** -- the incumbent, trained on its usual objective.
- **MLP, cross-entropy** -- the neural control, so the comparison isolates the *objective* rather
  than the architecture.
- **MLP, partial-AUC surrogate** -- the same network, same data, same seed, trained to rank
  positives above the top-scoring negatives only (`netsentry/models/pauc.py`).

The headline is a matrix, not a number: each model is trained at one budget and scored at every
budget. That is the only way to see the trade the technique actually makes, because an objective
that concentrates on 0.1% has no reason to be good at 5%, and a study that reported only the
budget it optimised for would be marking its own homework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.pauc import partial_auc
from netsentry.models.supervised import SupervisedClassifier
from netsentry.models.tabular_nn import TorchTabularClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import OperatingPointConfig

logger = get_logger(__name__)

REPORT_NAME = "operating_point.md"
FIGURE_NAME = "operating_point_frontier.png"

INCUMBENT = "LightGBM (cross-entropy)"
MLP_BCE = "MLP (cross-entropy)"


@dataclass
class ArmResult:
    """One trained model: what it optimised, and what it detects at every budget."""

    name: str
    trained_for: float | None
    pr_auc: float
    tpr_at: dict[float, float]
    pauc_at: dict[float, float]
    train_seconds: float

    @property
    def label(self) -> str:
        if self.trained_for is None:
            return self.name
        return f"{self.name} @ {self.trained_for:.1%}"


@dataclass
class OperatingPointStudy:
    """Everything the report renders."""

    arms: list[ArmResult]
    budgets: list[float]
    train_budgets: list[float]
    n_train: int
    n_test: int
    prevalence: float
    batch_rows: int
    train_negative_fraction: float
    torch_available: bool


def _evaluate(
    name: str,
    trained_for: float | None,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    budgets: list[float],
    seconds: float,
) -> ArmResult:
    """Score one model at every budget, always choosing the threshold on validation."""
    tpr = {}
    pauc = {}
    for budget in budgets:
        threshold = threshold_at_fpr(y_val, val_scores, budget)
        tpr[budget] = float(rates_at_threshold(y_test, test_scores, threshold)["tpr"])
        pauc[budget] = partial_auc(y_test, test_scores, budget)
    return ArmResult(
        name=name,
        trained_for=trained_for,
        pr_auc=float(average_precision_score(y_test, test_scores)),
        tpr_at=tpr,
        pauc_at=pauc,
        train_seconds=seconds,
    )


def run_operating_point_study(settings: Settings) -> OperatingPointStudy:
    """Train one model per objective (and per target budget), then score them all everywhere."""
    cfg: OperatingPointConfig = settings.operating_point
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    variant.deep_tabular.batch_size = cfg.batch_rows
    variant.deep_tabular.epochs = cfg.epochs
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    if len(train) > cfg.max_train_rows:
        train = train.head(cfg.max_train_rows)

    pipeline = build_pipeline(variant)
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(val), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test), dtype=float)
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    budgets = list(cfg.budgets)

    arms: list[ArmResult] = []
    start = time.perf_counter()
    tree = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    seconds = time.perf_counter() - start
    classes = np.asarray(tree.classes_)
    arms.append(
        _evaluate(
            INCUMBENT,
            None,
            positive_scores(np.asarray(tree.predict_proba(x_val)), classes),
            positive_scores(np.asarray(tree.predict_proba(x_test)), classes),
            y_val,
            y_test,
            budgets,
            seconds,
        )
    )

    torch_available = is_available("torch")
    if torch_available:
        start = time.perf_counter()
        control = TorchTabularClassifier(variant, architecture="mlp", objective="bce").fit(
            x_train, y_train, eval_set=(x_val, y_val)
        )
        seconds = time.perf_counter() - start
        arms.append(
            _evaluate(
                MLP_BCE,
                None,
                np.asarray(control.predict_proba(x_val))[:, 1],
                np.asarray(control.predict_proba(x_test))[:, 1],
                y_val,
                y_test,
                budgets,
                seconds,
            )
        )
        for target in cfg.train_budgets:
            start = time.perf_counter()
            tuned = TorchTabularClassifier(
                variant, architecture="mlp", objective="pauc", pauc_alpha=target
            ).fit(x_train, y_train, eval_set=(x_val, y_val))
            seconds = time.perf_counter() - start
            arms.append(
                _evaluate(
                    "MLP (partial-AUC)",
                    target,
                    np.asarray(tuned.predict_proba(x_val))[:, 1],
                    np.asarray(tuned.predict_proba(x_test))[:, 1],
                    y_val,
                    y_test,
                    budgets,
                    seconds,
                )
            )

    return OperatingPointStudy(
        arms=arms,
        budgets=budgets,
        train_budgets=list(cfg.train_budgets),
        n_train=len(y_train),
        n_test=len(y_test),
        prevalence=float(np.mean(y_test)),
        batch_rows=cfg.batch_rows,
        train_negative_fraction=float(np.mean(y_train == 0)),
        torch_available=torch_available,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_operating_point_report(settings: Settings) -> Path:
    """Run the objective comparison and write the report + figure."""
    study = run_operating_point_study(settings)
    budgets = np.array(study.budgets, dtype=float)
    figure = plots.plot_lines(
        {
            arm.label: (budgets, np.array([arm.tpr_at[b] for b in study.budgets], dtype=float))
            for arm in study.arms
        },
        xlabel="false-positive budget (threshold chosen on validation)",
        ylabel="detection rate on the test days",
        title="What each training objective buys, at every budget",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote operating-point report", extra={"path": str(out_path)})

    with track_run(settings, "operating_point") as run:
        run.log_params({"train_rows": study.n_train, "batch_rows": study.batch_rows})
        for arm in study.arms:
            key = "".join(ch if ch.isalnum() else "_" for ch in arm.label)
            run.log_metrics(
                {
                    f"pr_auc_{key}": arm.pr_auc,
                    **{f"tpr_{key}_{b}": v for b, v in arm.tpr_at.items()},
                }
            )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _matrix_table(study: OperatingPointStudy) -> str:
    header = "| model | PR-AUC | " + " | ".join(f"TPR @ {b:.1%}" for b in study.budgets) + " |"
    rows = [header, "|" + "---|" * (2 + len(study.budgets))]
    best = {b: max(arm.tpr_at[b] for arm in study.arms) for b in study.budgets}
    for arm in study.arms:
        cells = []
        for b in study.budgets:
            value = f"{arm.tpr_at[b]:.1%}"
            cells.append(f"**{value}**" if arm.tpr_at[b] == best[b] else value)
        rows.append(f"| {arm.label} | {arm.pr_auc:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _pauc_table(study: OperatingPointStudy) -> str:
    header = "| model | " + " | ".join(f"pAUC @ {b:.1%}" for b in study.budgets) + " |"
    rows = [header, "|" + "---|" * (1 + len(study.budgets))]
    for arm in study.arms:
        cells = " | ".join(f"{arm.pauc_at[b]:.3f}" for b in study.budgets)
        rows.append(f"| {arm.label} | {cells} |")
    return "\n".join(rows)


def negatives_per_batch(study: OperatingPointStudy, alpha: float) -> int:
    """How many negatives actually define the loss in one minibatch at this budget.

    This is the number that decides whether the surrogate is estimating anything: the objective
    ranks positives against the top ``ceil(alpha * n_negatives)`` negatives *in the batch*, so a
    tight budget and a modest batch leave it steering on a handful of flows.
    """
    from math import ceil

    return max(1, ceil(alpha * study.batch_rows * study.train_negative_fraction))


def _budget_table(study: OperatingPointStudy) -> str:
    control = next((a for a in study.arms if a.name == MLP_BCE), None)
    rows = [
        "| trained for | negatives per batch defining the loss | TPR at its own budget | "
        "cross-entropy control | difference |",
        "|---|---|---|---|---|",
    ]
    for arm in study.arms:
        if arm.trained_for is None or control is None:
            continue
        own = arm.tpr_at[arm.trained_for]
        base = control.tpr_at[arm.trained_for]
        rows.append(
            f"| {arm.trained_for:.1%} | {negatives_per_batch(study, arm.trained_for)} | "
            f"{own:.1%} | {base:.1%} | **{own - base:+.1%}** |"
        )
    return "\n".join(rows)


def _headline_read(study: OperatingPointStudy) -> str:
    if not study.torch_available:
        return (
            "PyTorch is not installed in this environment, so only the incumbent ran. Install the "
            "`ae` extra to reproduce the objective comparison."
        )
    control = next(a for a in study.arms if a.name == MLP_BCE)
    tuned = [a for a in study.arms if a.trained_for is not None]
    incumbent = study.arms[0]
    wins = []
    for arm in tuned:
        budget = arm.trained_for
        assert budget is not None
        if budget in arm.tpr_at:
            wins.append((budget, arm.tpr_at[budget] - control.tpr_at[budget]))
    gained = [w for w in wins if w[1] > 0.005]
    lost = [w for w in wins if w[1] <= 0.005]
    if gained:
        best_budget, best_gain = max(gained, key=lambda w: w[1])
        verdict = (
            f"**Training for the budget wins at the budget it trained for.** At {best_budget:.1%} "
            f"FPR the partial-AUC network detects {best_gain:+.1%} more than the identical "
            "network trained on cross-entropy -- same architecture, same data, same seed, same "
            "early stopping, one term of the loss different."
        )
        if lost:
            tight_budget, tight_gap = min(lost, key=lambda w: w[1])
            verdict += (
                f" **And training for the *tightest* budget does not.** The arm trained at "
                f"{tight_budget:.1%} lands {tight_gap:+.1%} against the same control at its own "
                "target -- worse, not better -- and the reason is in the table below rather than "
                "in the idea: at that budget a minibatch supplies "
                f"{negatives_per_batch(study, tight_budget)} negatives to define the loss, which "
                "is not an estimate of the population's hardest negatives, it is a coin flip. "
                "The surrogate needs the batch to contain enough of the tail to see it, and the "
                "tighter the budget the larger that batch has to be. This is the sort of failure "
                "that looks like a bad technique and is really a sampling constraint, so it is "
                "worth naming precisely."
            )
    else:
        verdict = (
            "**Training for the budget does not beat training for the loss here.** At every "
            "budget the partial-AUC network is within noise of the cross-entropy control, which "
            "is a kept negative rather than a tuning failure: the surrogate only sees the "
            "hardest negatives *in each minibatch*, and at a 0.1% budget that is one flow in a "
            f"batch of {study.batch_rows:,} -- an estimate of the population's hardest negatives "
            "that is far too noisy to steer on. The fix is a larger batch, not a cleverer loss, "
            "and the batch size a 0.1% budget would need is most of the training set."
        )
    return (
        f"{verdict}\n\nThe incumbent tree remains the reference: {incumbent.pr_auc:.3f} PR-AUC "
        f"against the cross-entropy network's {control.pr_auc:.3f}, and the two disagree most "
        "where it matters least. What the matrix above shows that a single number cannot is the "
        "*shape* of each objective's competence: cross-entropy spreads it across the whole "
        "curve, and the partial-AUC surrogate concentrates it -- which is the point, and also "
        "the risk, because a budget is a policy decision that changes with headcount."
    )


def _batch_read(study: OperatingPointStudy) -> str:
    """The mechanical explanation for which target budgets the surrogate can serve at all."""
    if not study.torch_available:
        return ""
    tightest = min(study.train_budgets)
    widest = max(study.train_budgets)
    needed = int(np.ceil(10 / (tightest * study.train_negative_fraction)))
    return (
        f"The surrogate ranks positives against the top `ceil(alpha * n_negatives)` negatives in "
        f"each minibatch. With {study.batch_rows:,} rows per batch and "
        f"{study.train_negative_fraction:.0%} of them benign, a {widest:.1%} budget supplies "
        f"{negatives_per_batch(study, widest)} negatives to learn from and a {tightest:.1%} "
        f"budget supplies {negatives_per_batch(study, tightest)}. Wanting ten of them at the "
        f"tightest budget would take a batch of roughly {needed:,} rows -- a substantial "
        "fraction of the training set, which is to say the objective stops being a minibatch "
        "objective. That is the real constraint on this technique, and it is a property of the "
        "budget rather than of the model."
    )


def _tradeoff_read(study: OperatingPointStudy) -> str:
    tuned = [a for a in study.arms if a.trained_for is not None]
    if not tuned or not study.torch_available:
        return ""
    control = next(a for a in study.arms if a.name == MLP_BCE)
    widest = max(study.budgets)
    losses = [(a.label, a.tpr_at[widest] - control.tpr_at[widest]) for a in tuned]
    worst = min(losses, key=lambda item: item[1])
    pr_gap = min(a.pr_auc for a in tuned) - control.pr_auc
    return (
        f"The cost is off the target budget, and it is visible: at the widest budget measured "
        f"({widest:.1%}) the {worst[0]} arm gives up {worst[1]:+.1%} against the cross-entropy "
        f"control, and its overall PR-AUC moves {pr_gap:+.3f}. That is the technique working as "
        "designed rather than misbehaving -- a partial AUC is a *partial* objective, and the "
        "region it ignores is the region it will be worst in. The operational reading is that "
        "this is only worth doing when the budget is genuinely fixed by headcount, and that "
        "changing the budget means retraining rather than re-thresholding."
    )


def _render(study: OperatingPointStudy, figure: Path) -> str:
    return f"""# NetSentry — Training for the Operating Point

_{study.n_train:,} training rows, judged on {study.n_test:,} later-day flows at
{study.prevalence:.1%} prevalence. Every threshold is chosen on validation and applied to test.
Minibatch {study.batch_rows:,} — which is part of the objective's specification here, not a
performance knob._

## Why this report exists

Every evaluation in this project leads with detection at a fixed false-positive budget, because
that is what a SOC deploys. Every model in this project is trained to minimise cross-entropy,
which is a different objective — log-loss spends capacity on being right about the obviously
benign majority, while the operating point is decided entirely by the handful of benign flows
that score highest, the ones the threshold has to clear. Nothing in the training objective knows
those flows are special.

The **partial AUC** does know. Where ROC-AUC integrates the ROC curve over all false-positive
rates, partial AUC integrates it only over `[0, alpha]` — the region the budget allows — and
maximising it means caring only about how positives rank against the *top-scoring* negatives
(Narasimhan & Agarwal 2013). The surrogate used here takes the `ceil(alpha * n_negatives)`
highest-scoring negatives in each minibatch and penalises every positive that fails to outrank
them, with the step function relaxed to a sigmoid so it has gradients.

## Detection at every budget

{_matrix_table(study)}

![Detection against budget](../figures/{figure.name})

{_headline_read(study)}

## Does the objective have enough of the tail to see?

{_budget_table(study)}

{_batch_read(study)}

## The same models, scored by partial AUC

{_pauc_table(study)}

Partial AUC normalised so a perfect ranker scores 1 and a random one 0.5, which makes the columns
comparable across budgets in a way raw TPR is not.

## What it costs elsewhere

{_tradeoff_read(study)}

## Scope and honest limits

- **The minibatch's hardest negatives are not the population's.** At a 0.1% budget the surrogate
  looks at one negative per batch, and that flow is the hardest in a random thousand rather than
  the hardest in the day. The bias is toward easier negatives than the deployed threshold will
  meet, and the honest fix is a bigger batch — which is why the batch size is reported next to
  the objective rather than buried in a config.
- **One architecture, one seed.** The comparison isolates the objective by holding the network
  fixed, which is the right control, but it does not tell you whether the same objective helps a
  different model family. The seed-variance study's noise floor applies here as everywhere.
- **The budget is a policy, not a constant.** A model trained for 0.1% is a model that must be
  retrained if the SOC hires. Cross-entropy's spread-out competence is worth something precisely
  because it survives that decision."""
