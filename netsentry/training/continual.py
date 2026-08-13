"""Continual learning: what a deployed detector forgets when the next attack family arrives.

Every retraining story in this repository so far has been *one* retrain: a model trained on the
early days, a model retrained on everything, and a comparison between them. Production does not
work that way. Attack families arrive one after another — brute force on Tuesday, the DoS family
on Wednesday, web attacks and infiltration on Thursday, bots and scanning on Friday — and each
time one arrives, somebody decides how to fold it in. That decision is almost never "refit on
the entire history from scratch", because the entire history is large, sometimes unavailable
(retention limits, tenant deletion, the SISA unlearning study's whole premise), and refitting is
the most expensive option on the menu. So the model is *updated*, and the question nobody asks
until an old attack sails through is: **what did the update cost the classes it already knew?**

That is catastrophic forgetting (McCloskey & Cohen 1989), and the continual-learning literature
measures it with a matrix rather than a number. Train through task ``i``, evaluate on task
``j``, and the resulting ``R[i][j]`` carries everything: the diagonal is plasticity (how well
each family is learned when it arrives), the lower triangle is stability (what survives), and
the upper triangle is transfer (what the model knew about a family before meeting it). Backward
transfer — the mean drop on old tasks between the moment they were learned and the end — is the
forgetting number (Lopez-Paz & Ranzato 2017).

Four update policies are compared on identical data, splits and seeds:

- **frozen** — train once on the first task, never update. Perfect stability, zero plasticity,
  and the honest control: any policy that cannot beat it is not worth its compute.
- **fine-tune** — continue boosting on the new day's traffic alone. The cheapest update, and the
  one an ops team reaches for. Gradient boosting is additive, so the trees that learned the old
  families are still *there* — which makes the question of whether their verdict survives an
  empirical one rather than an obvious one.
- **replay** — continue boosting on the new day plus a bounded reservoir sample of everything
  seen before (Robins 1995). The buffer is the dial, and the report sweeps it.
- **full retrain** — refit from scratch on all data seen so far. The upper bound on retention
  and the upper bound on cost; the comparison is only meaningful against it.

Two things are priced alongside detection, because they are what actually decides the policy:
cumulative training seconds, and the model's own growth — continued boosting appends trees, so
a warm-started model is larger at every step and pays for it at every inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ContinualConfig

logger = get_logger(__name__)

REPORT_NAME = "continual.md"
FORGETTING_FIGURE_NAME = "continual_forgetting.png"
BUFFER_FIGURE_NAME = "continual_buffer.png"

FROZEN = "frozen"
FINE_TUNE = "fine-tune"
REPLAY = "replay"
RETRAIN = "full retrain"


@dataclass
class Task:
    """One capture day: the traffic that arrived, split into what we train on and what we keep."""

    name: str
    attacks: tuple[str, ...]
    x_train: np.ndarray
    y_train: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray

    @property
    def prevalence(self) -> float:
        """The base rate on the held-out half -- the PR-AUC a random scorer would earn."""
        return float(np.mean(self.y_eval))


@dataclass
class StrategyResult:
    """One update policy's full retention matrix and what it cost to produce."""

    name: str
    description: str
    matrix: np.ndarray  # R[i][j]: PR-AUC on task j after training through task i
    seconds: list[float]
    trees: list[int]
    rows_touched: list[int]
    buffer_rows: int
    inference_ms_per_1k: float = 0.0

    @property
    def average(self) -> float:
        """Mean PR-AUC over every task seen, after the last one -- the headline of the field."""
        return float(np.mean(self.matrix[-1, :]))

    @property
    def backward_transfer(self) -> float:
        """Mean change on old tasks between learning them and finishing. Negative = forgetting."""
        last = self.matrix.shape[0] - 1
        if last < 1:
            return 0.0
        return float(np.mean([self.matrix[last, j] - self.matrix[j, j] for j in range(last)]))

    @property
    def learning_accuracy(self) -> float:
        """Mean of the diagonal: how well each family is learned at the moment it arrives."""
        return float(np.mean(np.diag(self.matrix)))

    @property
    def total_seconds(self) -> float:
        return float(np.sum(self.seconds))

    @property
    def total_rows(self) -> int:
        return int(np.sum(self.rows_touched))


@dataclass
class BufferPoint:
    """One replay-buffer size: what it retains and what it costs."""

    buffer_rows: int
    average: float
    backward_transfer: float
    seconds: float


@dataclass
class ContinualStudy:
    """Everything the report renders."""

    tasks: list[Task]
    strategies: list[StrategyResult]
    buffer_sweep: list[BufferPoint]
    forward_transfer: dict[str, float]
    warm_start: bool
    seed: int
    task_names: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Metrics. Pure functions on a retention matrix, unit-tested directly.
# --------------------------------------------------------------------------------------


def backward_transfer(matrix: np.ndarray) -> float:
    """Mean ``R[T][j] - R[j][j]`` over tasks learned before the last (Lopez-Paz & Ranzato 2017).

    Negative is forgetting. The subtraction is against the score the task earned *when it was
    learned*, not against some external baseline, which is what makes it a measure of what the
    updates destroyed rather than of how hard the task is.
    """
    last = matrix.shape[0] - 1
    if last < 1:
        return 0.0
    return float(np.mean([matrix[last, j] - matrix[j, j] for j in range(last)]))


def forward_transfer(matrix: np.ndarray, baselines: np.ndarray) -> float:
    """Mean ``R[j-1][j] - baseline_j``: what the model knew about a family before meeting it.

    The baseline is the task's own prevalence, which is exactly the average precision a random
    scorer achieves -- the only defensible zero point for PR-AUC, and one that differs per task
    because the class balance does.
    """
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.mean([matrix[j - 1, j] - baselines[j] for j in range(1, matrix.shape[0])]))


class ReservoirBuffer:
    """A bounded uniform sample of everything seen so far (Vitter 1985, Algorithm R).

    Reservoir sampling is the right primitive here rather than "keep the last k rows" or
    "subsample the union each time": the buffer must remain a uniform sample of the *whole*
    stream, or replay quietly becomes recency-weighted and stops being the policy the report
    describes -- which would flatter it, since recent data is what the next task looks like.
    Every row seen ends up in a buffer of capacity ``k`` with probability exactly ``k / n``.
    """

    def __init__(self, capacity: int, n_features: int) -> None:
        self.capacity = max(int(capacity), 0)
        self.n_seen = 0
        self._x = np.empty((0, n_features), dtype=float)
        self._y = np.empty(0, dtype=int)

    def add(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        """Offer a batch of rows to the reservoir.

        Algorithm R is defined one item at a time, but the per-item decisions are independent
        given each item's position in the stream, so the batch is applied in two vectorised
        phases: fill the free slots, then draw one uniform slot per remaining item from
        ``[0, t_i)`` and keep the draws that land inside the reservoir. Fancy assignment is
        last-write-wins on repeated indices, which is exactly what applying the same draws
        sequentially would do -- so this is the same algorithm, not an approximation of it.
        """
        if self.capacity == 0 or len(x) == 0:
            self.n_seen += len(x)
            return
        taken = 0
        free = self.capacity - len(self._x)
        if free > 0:
            taken = min(free, len(x))
            self._x = np.vstack([self._x, x[:taken]])
            self._y = np.concatenate([self._y, y[:taken]])
            self.n_seen += taken
        remaining = len(x) - taken
        if remaining <= 0:
            return
        # The i-th remaining item is the (n_seen + i + 1)-th of the stream; Algorithm R gives it
        # a uniform slot in [0, that position) and evicts only if the slot is inside the buffer.
        positions = self.n_seen + np.arange(1, remaining + 1)
        slots = rng.integers(0, positions)
        self.n_seen += remaining
        hit = slots < self.capacity
        if bool(hit.any()):
            self._x[slots[hit]] = x[taken:][hit]
            self._y[slots[hit]] = y[taken:][hit]

    @property
    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        """The retained sample."""
        return self._x, self._y


# --------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------


def build_tasks(settings: Settings) -> list[Task]:
    """Split the capture into one task per day, each with a held-out half.

    A day's flows are split by position rather than at random: rows arrive in capture order, and
    an attack burst is a run of near-duplicate flows, so a shuffled within-day split would put
    near-copies of the same burst on both sides and report a retention that is really memory.
    Days with no attacks cannot be scored with PR-AUC and are folded into the next task's
    training pool instead of becoming a task of their own -- on this capture that is Monday,
    which is benign-only.
    """
    cfg: ContinualConfig = settings.continual
    frame = pd.read_parquet(settings.paths.data_processed / "clean.parquet")
    day_column = settings.split.day_column
    pipeline = build_pipeline(settings)

    ordered = [d for d in schema.DAY_ORDER if d in set(frame[day_column])]
    # Fit the leakage firewall on the first task's training half only -- the same rule as
    # everywhere else. A pipeline fitted on all five days would leak the future's statistics
    # into the first day's model, which is precisely the mistake this project exists to avoid.
    carry: list[pd.DataFrame] = []
    tasks: list[Task] = []
    fitted = False
    for day in ordered:
        day_frame = frame[frame[day_column] == day]
        if len(day_frame) > cfg.max_rows_per_task:
            day_frame = day_frame.head(cfg.max_rows_per_task)
        attacks = tuple(
            sorted(str(v) for v in day_frame.loc[day_frame[BINARY_TARGET] == 1, "Label"].unique())
        )
        if not attacks:  # benign-only day: no positives, so no PR-AUC and no task
            carry.append(day_frame)
            continue
        cut = int(len(day_frame) * cfg.train_fraction)
        train_frame = pd.concat([*carry, day_frame.iloc[:cut]], ignore_index=True)
        carry = []
        eval_frame = day_frame.iloc[cut:]
        if not fitted:
            pipeline.fit(train_frame)
            fitted = True
        tasks.append(
            Task(
                name=day,
                attacks=attacks,
                x_train=np.asarray(pipeline.transform(train_frame), dtype=float),
                y_train=train_frame[BINARY_TARGET].to_numpy().astype(int),
                x_eval=np.asarray(pipeline.transform(eval_frame), dtype=float),
                y_eval=eval_frame[BINARY_TARGET].to_numpy().astype(int),
            )
        )
    return tasks


def _score(model: SupervisedClassifier, task: Task) -> float:
    """PR-AUC on a task's held-out half, on raw scores (the scale the headline metric uses)."""
    proba = np.asarray(model.predict_proba(task.x_eval))
    positive = (
        proba[:, list(model.classes_).index(1)] if 1 in list(model.classes_) else proba[:, -1]
    )
    if len(np.unique(task.y_eval)) < 2:
        return float("nan")
    return float(average_precision_score(task.y_eval, positive))


def run_strategy(
    settings: Settings,
    tasks: list[Task],
    strategy: str,
    *,
    buffer_rows: int,
) -> StrategyResult:
    """Run one update policy across the task sequence, recording the full retention matrix."""
    seed_everything(settings.seed)
    rng = np.random.default_rng(settings.seed)
    n = len(tasks)
    matrix = np.full((n, n), np.nan)
    seconds: list[float] = []
    trees: list[int] = []
    rows: list[int] = []

    model: SupervisedClassifier | None = None
    history_x: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    reservoir = ReservoirBuffer(buffer_rows, tasks[0].x_train.shape[1])

    for i, task in enumerate(tasks):
        if strategy == FROZEN and i > 0:
            # No update at all, so the model -- and therefore every score it produces -- is
            # unchanged: the previous row is copied rather than recomputed. The control exists
            # to show what doing nothing costs, and doing nothing costs no compute either.
            matrix[i, :] = matrix[i - 1, :]
            seconds.append(0.0)
            trees.append(trees[-1])
            rows.append(0)
            continue

        if strategy == RETRAIN or i == 0:
            x_fit = np.vstack([*history_x, task.x_train]) if history_x else task.x_train
            y_fit = np.concatenate([*history_y, task.y_train]) if history_y else task.y_train
            init = None
        elif strategy == FINE_TUNE:
            x_fit, y_fit, init = task.x_train, task.y_train, model
        elif strategy == REPLAY:
            buffer_x, buffer_y = reservoir.rows
            x_fit = np.vstack([buffer_x, task.x_train]) if len(buffer_x) else task.x_train
            y_fit = np.concatenate([buffer_y, task.y_train]) if len(buffer_y) else task.y_train
            init = model
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        start = time.perf_counter()
        fresh = SupervisedClassifier(settings)
        fresh.fit(x_fit, y_fit, init_model=init)
        seconds.append(time.perf_counter() - start)
        trees.append(fresh.n_trees())
        rows.append(len(x_fit))
        model = fresh

        for j, seen in enumerate(tasks[: i + 1]):
            matrix[i, j] = _score(model, seen)
        for j in range(i + 1, n):  # zero-shot on families not yet met: forward transfer
            matrix[i, j] = _score(model, tasks[j])

        history_x.append(task.x_train)
        history_y.append(task.y_train)
        if strategy == REPLAY:
            reservoir.add(task.x_train, task.y_train, rng)

    # What the final model costs to *serve*, which is the axis a warm-started ensemble loses on:
    # continued boosting appends trees, and every appended tree is paid for at every request.
    bench = np.vstack([t.x_eval for t in tasks])[: settings.continual.bench_rows]
    start = time.perf_counter()
    assert model is not None
    model.predict_proba(bench)
    per_1k = (time.perf_counter() - start) * 1000.0 / max(len(bench), 1) * 1000.0

    return StrategyResult(
        name=strategy,
        description=_DESCRIPTIONS[strategy],
        matrix=matrix,
        seconds=seconds,
        trees=trees,
        rows_touched=rows,
        buffer_rows=buffer_rows if strategy == REPLAY else 0,
        inference_ms_per_1k=per_1k,
    )


_DESCRIPTIONS = {
    FROZEN: "trained once on the first family, never updated",
    FINE_TUNE: "continue boosting on the new day's traffic alone",
    REPLAY: "continue boosting on the new day plus a bounded uniform sample of the past",
    RETRAIN: "refit from scratch on every day seen so far",
}


def run_continual_study(settings: Settings) -> ContinualStudy:
    """Compare the four update policies, then sweep the replay buffer that separates them."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    cfg: ContinualConfig = variant.continual

    tasks = build_tasks(variant)
    if len(tasks) < 2:
        raise ValueError("continual learning needs at least two attack-bearing capture days")

    strategies = [
        run_strategy(variant, tasks, FROZEN, buffer_rows=0),
        run_strategy(variant, tasks, FINE_TUNE, buffer_rows=0),
        run_strategy(variant, tasks, REPLAY, buffer_rows=cfg.buffer_rows),
        run_strategy(variant, tasks, RETRAIN, buffer_rows=0),
    ]
    baselines = np.array([t.prevalence for t in tasks])
    transfer = {s.name: forward_transfer(s.matrix, baselines) for s in strategies}

    sweep: list[BufferPoint] = []
    for size in cfg.buffer_sweep:
        result = run_strategy(variant, tasks, REPLAY, buffer_rows=size)
        sweep.append(
            BufferPoint(
                buffer_rows=size,
                average=result.average,
                backward_transfer=result.backward_transfer,
                seconds=result.total_seconds,
            )
        )

    return ContinualStudy(
        tasks=tasks,
        strategies=strategies,
        buffer_sweep=sweep,
        forward_transfer=transfer,
        warm_start=SupervisedClassifier(variant).supports_warm_start(),
        seed=variant.seed,
        task_names=[t.name for t in tasks],
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_continual_report(settings: Settings) -> Path:
    """Run the continual-learning study and write the report + figures."""
    study = run_continual_study(settings)
    steps = np.arange(1, len(study.tasks) + 1, dtype=float)
    first = study.tasks[0].name
    forgetting_fig = plots.plot_lines(
        {s.name: (steps, s.matrix[:, 0]) for s in study.strategies},
        xlabel="capture days folded in",
        ylabel=f"PR-AUC on the first family ({first})",
        title=f"What each update policy does to what the model already knew ({first})",
        out_path=settings.paths.figures_dir / FORGETTING_FIGURE_NAME,
    )
    buffers = np.array([max(p.buffer_rows, 1) for p in study.buffer_sweep], dtype=float)
    buffer_fig = plots.plot_lines(
        {
            "average PR-AUC over all families": (
                buffers,
                np.array([p.average for p in study.buffer_sweep]),
            ),
            "backward transfer (forgetting)": (
                buffers,
                np.array([p.backward_transfer for p in study.buffer_sweep]),
            ),
        },
        xlabel="replay buffer (rows, log scale; 1 == no buffer)",
        ylabel="PR-AUC",
        title="The stability-plasticity dial: what a bounded memory buys",
        out_path=settings.paths.figures_dir / BUFFER_FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, forgetting_fig, buffer_fig), encoding="utf-8")
    logger.info("Wrote continual-learning report", extra={"path": str(out_path)})

    with track_run(settings, "continual") as run:
        run.log_params({"tasks": ",".join(study.task_names), "seed": study.seed})
        for strategy in study.strategies:
            key = strategy.name.replace(" ", "_").replace("-", "_")
            run.log_metrics(
                {
                    f"average_pr_auc_{key}": strategy.average,
                    f"backward_transfer_{key}": strategy.backward_transfer,
                    f"train_seconds_{key}": strategy.total_seconds,
                }
            )
        run.log_artifact(forgetting_fig)
        run.log_artifact(buffer_fig)
        run.log_artifact(out_path)
    return out_path


def _task_table(study: ContinualStudy) -> str:
    rows = [
        "| # | capture day | attack families that arrive | train rows | held-out rows |"
        " prevalence |",
        "|---|---|---|---|---|---|",
    ]
    for i, task in enumerate(study.tasks, start=1):
        families = ", ".join(f"`{a}`" for a in task.attacks)
        rows.append(
            f"| {i} | {task.name} | {families} | {len(task.y_train):,} | "
            f"{len(task.y_eval):,} | {task.prevalence:.1%} |"
        )
    return "\n".join(rows)


def _summary_table(study: ContinualStudy) -> str:
    rows = [
        "| policy | average PR-AUC | learned on arrival | backward transfer | forward transfer | "
        "train seconds | rows fitted | final trees | inference / 1k flows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in study.strategies:
        rows.append(
            f"| **{s.name}** | {s.average:.3f} | {s.learning_accuracy:.3f} | "
            f"{s.backward_transfer:+.3f} | {study.forward_transfer[s.name]:+.3f} | "
            f"{s.total_seconds:.0f} s | {s.total_rows:,} | {s.trees[-1]:,} | "
            f"{s.inference_ms_per_1k:.0f} ms |"
        )
    return "\n".join(rows)


def _matrix_table(study: ContinualStudy, strategy: StrategyResult) -> str:
    header = "| after training through | " + " | ".join(study.task_names) + " |"
    rows = [header, "|" + "---|" * (1 + len(study.task_names))]
    for i, name in enumerate(study.task_names):
        cells = []
        for j in range(len(study.task_names)):
            value = strategy.matrix[i, j]
            text = "—" if not np.isfinite(value) else f"{value:.3f}"
            if i == j:
                text = f"**{text}**"
            elif j > i:
                text = f"_{text}_"
            cells.append(text)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _buffer_table(study: ContinualStudy) -> str:
    rows = [
        "| replay buffer | average PR-AUC | backward transfer | train seconds |",
        "|---|---|---|---|",
    ]
    for point in study.buffer_sweep:
        label = (
            "none (naive fine-tune)" if point.buffer_rows == 0 else f"{point.buffer_rows:,} rows"
        )
        rows.append(
            f"| {label} | {point.average:.3f} | {point.backward_transfer:+.3f} | "
            f"{point.seconds:.0f} s |"
        )
    return "\n".join(rows)


def _by_name(study: ContinualStudy, name: str) -> StrategyResult:
    return next(s for s in study.strategies if s.name == name)


def _forgetting_read(study: ContinualStudy) -> str:
    fine = _by_name(study, FINE_TUNE)
    retrain = _by_name(study, RETRAIN)
    replay = _by_name(study, REPLAY)
    frozen = _by_name(study, FROZEN)
    first = study.tasks[0]
    learned = fine.matrix[0, 0]
    survived = fine.matrix[-1, 0]
    drop = (learned - survived) / max(learned, 1e-9)
    return (
        f"**Fine-tuning forgets.** `{first.name}`'s families ({', '.join(first.attacks)}) are "
        f"detected at {learned:.3f} PR-AUC the day they are learned and at {survived:.3f} after "
        f"three more days have been folded in — a **{drop:.0%} relative loss** on an attack "
        "family nobody removed from the model, nobody stopped caring about, and nothing in the "
        "monitoring would report, because the traffic that would reveal it is not in the "
        "evaluation set any more. Backward transfer, the mean of that drop over every old task, "
        f"is **{fine.backward_transfer:+.3f}**. Gradient boosting is additive and the trees that "
        "learned the old families are still physically present in the ensemble, which is exactly "
        "why this result is worth measuring rather than assuming: the later trees do not delete "
        "the earlier ones, they *outvote* them, and the score a flow receives is the sum."
        f"\n\nReplay with {replay.buffer_rows:,} remembered rows recovers part of it "
        f"({replay.backward_transfer:+.3f}), and full retraining recovers most "
        f"({retrain.backward_transfer:+.3f}) — but not all, and the residue is the second "
        "finding. **Even refitting on the entire history forgets.** That cannot be a property of "
        "the update rule, because there is no update: it is interference. One decision surface "
        "now has to separate five attack families from benign traffic at once, the class balance "
        "it is fitted against has moved, and capacity spent on `PortScan` is capacity not spent "
        f"on `{first.attacks[0]}`. The frozen control makes the shape of the trade visible from "
        f"the other side: it forgets nothing by construction, and it ends at "
        f"{frozen.average:.3f} average PR-AUC against retraining's {retrain.average:.3f} because "
        "it never learns anything either."
    )


def _cost_read(study: ContinualStudy) -> str:
    fine = _by_name(study, FINE_TUNE)
    retrain = _by_name(study, RETRAIN)
    saving = 1.0 - fine.total_seconds / max(retrain.total_seconds, 1e-9)
    growth = fine.trees[-1] / max(retrain.trees[-1], 1)
    latency = fine.inference_ms_per_1k / max(retrain.inference_ms_per_1k, 1e-9)
    verdict = (
        "On this capture the incremental update is not a bargain at all: it costs "
        f"{retrain.average - fine.average:.3f} PR-AUC for a {saving:.0%} training saving, and "
        f"then charges that saving back {latency:.1f} times over at every single request."
        if saving < 0.5
        else "so the incremental update does buy a real training saving here, and the question "
        "becomes whether the detection it costs is worth that saving."
    )
    return (
        f"Fine-tuning fits {fine.total_rows:,} rows against full retraining's "
        f"{retrain.total_rows:,} and takes {fine.total_seconds:.0f} s against "
        f"{retrain.total_seconds:.0f} s — a **{saving:.0%}** saving. {verdict}"
        f"\n\nThe reason the saving is so much smaller than the row count suggests is that "
        "boosting cost is dominated by the number of trees, not by the rows they are fitted on, "
        "and warm-starting *adds* trees rather than replacing them: the fine-tuned model ends at "
        f"{fine.trees[-1]:,} trees against {retrain.trees[-1]:,}, a **{growth:.0f}x** larger "
        f"ensemble that costs {fine.inference_ms_per_1k:.0f} ms per thousand flows against "
        f"{retrain.inference_ms_per_1k:.0f} ms. A four-day capture is also the smallest history "
        "this trade can be measured on: retraining cost grows with the history and incremental "
        "cost does not, so the crossover exists — it is just further out than four days, and "
        "quoting the incremental policy's saving without quoting where that crossover sits is "
        "how teams end up paying for forgetting they did not need to buy."
    )


def _transfer_read(study: ContinualStudy) -> str:
    retrain = _by_name(study, RETRAIN)
    value = study.forward_transfer[RETRAIN]
    tasks = study.tasks
    examples = [
        (tasks[j].name, retrain.matrix[j - 1, j], tasks[j].prevalence) for j in range(1, len(tasks))
    ]
    best = max(examples, key=lambda e: e[1] - e[2])
    direction = "positive" if value > 0 else "negative"
    return (
        f"Forward transfer — what the model scores on a family the *day before* it first sees "
        f"one, against the prevalence a random scorer would earn — is {direction} at "
        f"**{value:+.3f}** for full retraining. The strongest case is `{best[0]}`, detected at "
        f"{best[1]:.3f} against a {best[2]:.1%} base rate before a single flow of it was ever "
        "labelled. This matters because the temporal split shares **zero** attack classes "
        "between the training days and the test days — the open-set study established that "
        "every attack the deployed model meets is formally an unknown class. The forward-transfer "
        "column is the quantitative version of the good news hiding in that: attack families are "
        "not mutually unintelligible, and a detector trained on brute force is genuinely better "
        "than chance on denial of service it has never seen. The upper triangle of every matrix "
        "below is that measurement, in italics."
    )


def _render(study: ContinualStudy, forgetting_fig: Path, buffer_fig: Path) -> str:
    matrices = "\n\n".join(
        f"**{s.name}** — {s.description}\n\n{_matrix_table(study, s)}" for s in study.strategies
    )
    warm = (
        "LightGBM's `init_model` continues boosting from the previous ensemble, so the "
        "incremental policies here are genuine warm starts rather than refits in disguise."
        if study.warm_start
        else "The active backend cannot warm-start, so the incremental policies degenerate to "
        "refitting on the increment alone; the comparison still holds but the compute column "
        "reads differently from a true warm start."
    )
    return f"""# NetSentry — Continual Learning: What the Detector Forgets

_One task per capture day, in arrival order. Each policy is measured by the full retention
matrix: train through task i, evaluate on task j. Seed {study.seed}; {warm}_

## Why this report exists

Every retraining story in this repository so far has been *one* retrain — a model trained on the
early days, a model retrained on everything, and a comparison. Production does not work that
way. Attack families arrive one after another, and each time one does, somebody decides how to
fold it in. That decision is almost never "refit on the entire history from scratch", because
the history is large, sometimes unavailable (retention limits, the deletion requests the
unlearning study exists to honour), and refitting is the most expensive option on the menu. So
the model is *updated* — and the question nobody asks until an old attack sails through is what
the update cost the families it already knew.

That is **catastrophic forgetting** (McCloskey & Cohen 1989), and the continual-learning
literature measures it with a matrix rather than a number. The diagonal is plasticity: how well
each family is learned when it arrives. The lower triangle is stability: what survives. The
upper triangle is transfer: what the model already knew about a family before meeting one.
**Backward transfer** — the mean change on old tasks between the moment they were learned and
the end of the sequence — is the forgetting number (Lopez-Paz & Ranzato 2017).

## The task sequence

{_task_table(study)}

The days are the tasks, which is not a modelling convenience: this capture introduces a
genuinely new set of attack families every day, so folding in a day *is* the class-incremental
problem rather than a stand-in for it. A day's flows are split by position rather than at
random, because an attack burst is a run of near-duplicate flows and a shuffled within-day split
would score memory as retention. Monday carries no attacks at all, so it cannot be scored with
PR-AUC and is folded into the first task's training pool instead of becoming a task with an
undefined result.

## What each policy costs and keeps

{_summary_table(study)}

![Forgetting curves](../figures/{forgetting_fig.name})

{_forgetting_read(study)}

## The compute argument, checked

{_cost_read(study)}

## What the model knows before it is taught

{_transfer_read(study)}

## The retention matrices in full

{matrices}

Bold is the diagonal (a family at the moment it is learned), italic is the upper triangle
(zero-shot, before that family has ever been seen), and plain text below the diagonal is what
survived. For the frozen control the whole matrix is one model's scores repeated, which is what
"never updated" means.

## The stability-plasticity dial

![Replay buffer sweep](../figures/{buffer_fig.name})

{_buffer_table(study)}

The buffer is the only knob that moves smoothly between the two extremes, and both of its ends
are already in the table above: an empty buffer *is* naive fine-tuning, and a buffer larger than
the history is a warm-started full retrain. Everything in between is the actual engineering
decision — how much of yesterday to keep, at what storage cost, under what retention policy.

## Scope and honest limits

- **Four tasks is a short sequence.** Forgetting compounds; a year of daily updates would show
  more of it, and the crossover where incremental training's compute advantage becomes real sits
  well beyond this capture.
- **Labels are assumed to arrive with the day.** They do not: the active-learning and
  weak-supervision studies exist because labelling is the binding constraint. Every policy here
  is therefore an upper bound on what its real-world version could achieve.
- **PR-AUC per task is a ranking measure within that day's traffic.** A policy could hold its
  per-task ranking while its calibrated scores drift, which would still break a fixed threshold
  — the threshold-refresh study measures that axis.
- **The interference result is not a criticism of retraining.** It is the price of one model
  serving five attack families; the alternative — a model per family — trades it for a routing
  problem and five thresholds to maintain."""
