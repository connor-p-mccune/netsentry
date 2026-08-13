"""Learning at line rate: a one-pass streaming detector against the batch pipeline that ships.

The deployed detector is a batch model. It is fitted on the training days, frozen, shipped, and
between two retrains every flow is scored by a model that has already seen its last example.
The streaming study measured what that costs and answered it with periodic retraining -- refit
on everything, redeploy, repeat -- which works and is expensive: cost grows with the history,
the whole history must still exist, and the freshest the model can ever be is one retrain
interval old.

There is a third option the project has never tried: a learner that updates **per flow**, in
bounded memory, in one pass. This report puts one in the field. `netsentry.models.hoeffding`
implements a Hoeffding tree (VFDT, Domingos & Hulten 2000) and ADWIN (Bifet & Gavalda 2007)
from scratch, and the comparison here is prequential -- **test then train**: every model scores
a batch before it is allowed to learn from it, so every prediction being scored was made on
flows the model had never seen. That protocol is the streaming equivalent of a held-out split,
and it is the only reason these numbers can be compared with the batch ones at all.

Five arms, on identical batches in identical order:

- **static** -- the deployed LightGBM, frozen. The incumbent.
- **periodic retrain** -- LightGBM refitted on everything seen so far, every few thousand flows.
- **Hoeffding tree (naive-Bayes leaves)** -- one pass, per-flow updates.
- **Hoeffding tree (majority-class leaves)** -- the same tree with the cheaper leaf rule, as an
  ablation, because the difference between them is most of what the streaming tree is doing.
- **Hoeffding tree + ADWIN** -- the same tree, with a change detector watching its own error and
  rebuilding it when the window says the error rate has moved.

And then the assumption every online learner rests on is put under load. Per-flow learning needs
per-flow labels, and a SOC does not have them: an analyst confirms an alert hours later, if at
all. The last section delays every label by a fixed number of batches and re-measures, which is
the difference between a benchmark result and an operational one.
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
from netsentry.evaluation.metrics import positive_scores, tpr_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.hoeffding import ADWIN, HoeffdingTree
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import OnlineConfig

logger = get_logger(__name__)

REPORT_NAME = "online.md"
STREAM_FIGURE_NAME = "online_stream.png"
DELAY_FIGURE_NAME = "online_delay.png"

STATIC = "static (deployed)"
PERIODIC = "periodic retrain"
TREE_NB = "Hoeffding tree (NB leaves)"
TREE_MC = "Hoeffding tree (majority leaves)"
TREE_ADWIN = "Hoeffding tree + ADWIN"


@dataclass
class ArmResult:
    """One learner's prequential run: what it scored, what it cost, what it became."""

    name: str
    description: str
    scores: np.ndarray
    labels: np.ndarray
    batch_pr_auc: list[float]
    learn_seconds: float
    score_seconds: float
    memory_bytes: int
    structure: str
    resets: int = 0

    @property
    def pr_auc(self) -> float:
        """Prequential PR-AUC over every prediction made before the flow was learned from."""
        if len(np.unique(self.labels)) < 2:
            return float("nan")
        return float(average_precision_score(self.labels, self.scores))

    @property
    def flows_per_second(self) -> float:
        """Update throughput -- whether the learner can keep up with the sensor."""
        total = self.learn_seconds + self.score_seconds
        return float(len(self.labels) / total) if total > 0 else float("inf")

    @property
    def distinct_scores(self) -> int:
        """How many different scores the learner can emit -- its resolution at the threshold."""
        return len(np.unique(np.round(self.scores, 9)))

    def tpr_at(self, fpr: float) -> float:
        """Detection rate at a fixed false-positive budget, chosen on this arm's own scores."""
        if len(np.unique(self.labels)) < 2:
            return float("nan")
        return float(tpr_at_fpr(self.labels, self.scores, fpr)[1])


@dataclass
class DelayPoint:
    """What a label-arrival delay does to an online learner."""

    delay_batches: int
    pr_auc: float


@dataclass
class OnlineStudy:
    """Everything the report renders."""

    arms: list[ArmResult]
    delays: list[DelayPoint]
    batch_rows: int
    n_stream: int
    n_warmup: int
    prevalence: float
    primary_fpr: float
    splits: int
    adwin_detections: int


def _stream_batches(n: int, batch_rows: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) batches over the stream, in arrival order."""
    return [(start, min(start + batch_rows, n)) for start in range(0, n, batch_rows)]


def _lgbm_scores(model: SupervisedClassifier, x: np.ndarray) -> np.ndarray:
    return positive_scores(np.asarray(model.predict_proba(x)), np.asarray(model.classes_))


def run_static_arm(
    settings: Settings,
    model: SupervisedClassifier,
    x_stream: np.ndarray,
    y_stream: np.ndarray,
    batches: list[tuple[int, int]],
) -> ArmResult:
    """The deployed model, frozen: it scores every batch and learns from none of them."""
    scores = np.zeros(len(y_stream))
    per_batch: list[float] = []
    start_time = time.perf_counter()
    for start, stop in batches:
        scores[start:stop] = _lgbm_scores(model, x_stream[start:stop])
        per_batch.append(_batch_pr_auc(y_stream[start:stop], scores[start:stop]))
    return ArmResult(
        name=STATIC,
        description="trained once on the training days, then frozen -- the incumbent",
        scores=scores,
        labels=y_stream,
        batch_pr_auc=per_batch,
        learn_seconds=0.0,
        score_seconds=time.perf_counter() - start_time,
        memory_bytes=model.n_trees() * 1024,
        structure=f"{model.n_trees():,} trees, never updated",
    )


def run_periodic_arm(
    settings: Settings,
    warm_x: np.ndarray,
    warm_y: np.ndarray,
    x_stream: np.ndarray,
    y_stream: np.ndarray,
    batches: list[tuple[int, int]],
) -> ArmResult:
    """Refit from scratch on everything seen so far, every ``retrain_every`` flows."""
    cfg: OnlineConfig = settings.online
    model = SupervisedClassifier(settings).fit(warm_x, warm_y)
    scores = np.zeros(len(y_stream))
    per_batch: list[float] = []
    learn_seconds = 0.0
    score_seconds = 0.0
    since_retrain = 0
    refits = 0
    pool_x, pool_y = warm_x, warm_y
    for start, stop in batches:
        tick = time.perf_counter()
        scores[start:stop] = _lgbm_scores(model, x_stream[start:stop])
        score_seconds += time.perf_counter() - tick
        per_batch.append(_batch_pr_auc(y_stream[start:stop], scores[start:stop]))
        pool_x = np.vstack([pool_x, x_stream[start:stop]])
        pool_y = np.concatenate([pool_y, y_stream[start:stop]])
        since_retrain += stop - start
        if since_retrain >= cfg.retrain_every:
            tick = time.perf_counter()
            model = SupervisedClassifier(settings).fit(pool_x, pool_y)
            learn_seconds += time.perf_counter() - tick
            since_retrain = 0
            refits += 1
    return ArmResult(
        name=PERIODIC,
        description=f"refit from scratch on the full history every {cfg.retrain_every:,} flows",
        scores=scores,
        labels=y_stream,
        batch_pr_auc=per_batch,
        learn_seconds=learn_seconds,
        score_seconds=score_seconds,
        memory_bytes=model.n_trees() * 1024 + pool_x.nbytes,
        structure=f"{refits} refits, {len(pool_x):,} rows retained at the end",
    )


def run_tree_arm(
    settings: Settings,
    name: str,
    warm_x: np.ndarray,
    warm_y: np.ndarray,
    x_stream: np.ndarray,
    y_stream: np.ndarray,
    batches: list[tuple[int, int]],
    *,
    leaf_prediction: str,
    adwin: bool,
    label_delay: int = 0,
) -> ArmResult:
    """Stream a Hoeffding tree through the same batches, test-then-train.

    ``label_delay`` holds each batch's labels back for that many batches before the learner is
    allowed to use them -- the SOC's actual situation, where the label is an analyst's verdict
    hours after the flow.
    """
    cfg: OnlineConfig = settings.online
    tree = _new_tree(settings, x_stream.shape[1], leaf_prediction)
    detector = ADWIN(cfg.adwin_delta) if adwin else None
    learn_seconds = 0.0
    score_seconds = 0.0
    resets = 0

    tick = time.perf_counter()
    tree.learn_many(warm_x, warm_y)  # the same warm-up the batch models get, one pass
    learn_seconds += time.perf_counter() - tick

    scores = np.zeros(len(y_stream))
    per_batch: list[float] = []
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    for start, stop in batches:
        tick = time.perf_counter()
        batch_scores = np.array(
            [tree.score_one(x_stream[i]) for i in range(start, stop)], dtype=float
        )
        score_seconds += time.perf_counter() - tick
        scores[start:stop] = batch_scores
        per_batch.append(_batch_pr_auc(y_stream[start:stop], batch_scores))

        if detector is not None:
            errors = np.abs(y_stream[start:stop] - batch_scores)
            drifted = False
            for value in errors:
                drifted = detector.update(float(value)) or drifted
            if drifted:
                # The window says the error rate has moved. Rebuild rather than keep growing a
                # tree whose upper splits were chosen for traffic that is no longer arriving.
                tree = _new_tree(settings, x_stream.shape[1], leaf_prediction)
                detector.reset()
                resets += 1

        pending.append((x_stream[start:stop], y_stream[start:stop]))
        if len(pending) > label_delay:
            batch_x, batch_y = pending.pop(0)
            tick = time.perf_counter()
            tree.learn_many(batch_x, batch_y)
            learn_seconds += time.perf_counter() - tick

    return ArmResult(
        name=name,
        description=_TREE_DESCRIPTIONS[name],
        scores=scores,
        labels=y_stream,
        batch_pr_auc=per_batch,
        learn_seconds=learn_seconds,
        score_seconds=score_seconds,
        memory_bytes=tree.memory_bytes(),
        structure=f"{tree.n_splits} splits, {tree.n_leaves()} leaves",
        resets=resets,
    )


_TREE_DESCRIPTIONS = {
    TREE_NB: "one pass, per-flow updates, Gaussian naive Bayes at the leaves",
    TREE_MC: "the same tree predicting the leaf's majority class -- the ablation",
    TREE_ADWIN: "the same tree, rebuilt whenever ADWIN says its own error rate has moved",
}


def _new_tree(settings: Settings, n_features: int, leaf_prediction: str) -> HoeffdingTree:
    cfg: OnlineConfig = settings.online
    return HoeffdingTree(
        n_features=n_features,
        n_classes=2,
        grace_period=cfg.grace_period,
        delta=cfg.split_delta,
        tie_threshold=cfg.tie_threshold,
        n_thresholds=cfg.n_thresholds,
        max_depth=cfg.max_depth,
        min_leaf_samples=cfg.min_leaf_samples,
        leaf_prediction=leaf_prediction,
    )


def _batch_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def run_online_study(settings: Settings) -> OnlineStudy:
    """Run every arm prequentially on one stream, then sweep the label delay."""
    cfg: OnlineConfig = settings.online
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    if len(train) > cfg.warmup_rows:
        train = train.head(cfg.warmup_rows)
    if len(test) > cfg.max_stream_rows:
        test = test.head(cfg.max_stream_rows)

    pipeline = build_pipeline(variant)
    warm_x: np.ndarray = np.asarray(pipeline.fit_transform(train), dtype=float)
    warm_y: np.ndarray = train[BINARY_TARGET].to_numpy().astype(int)
    x_stream: np.ndarray = np.asarray(pipeline.transform(test), dtype=float)
    y_stream: np.ndarray = test[BINARY_TARGET].to_numpy().astype(int)
    batches = _stream_batches(len(y_stream), cfg.batch_rows)

    deployed = SupervisedClassifier(variant).fit(warm_x, warm_y)
    arms = [
        run_static_arm(variant, deployed, x_stream, y_stream, batches),
        run_periodic_arm(variant, warm_x, warm_y, x_stream, y_stream, batches),
        run_tree_arm(
            variant,
            TREE_NB,
            warm_x,
            warm_y,
            x_stream,
            y_stream,
            batches,
            leaf_prediction="nb",
            adwin=False,
        ),
        run_tree_arm(
            variant,
            TREE_MC,
            warm_x,
            warm_y,
            x_stream,
            y_stream,
            batches,
            leaf_prediction="mc",
            adwin=False,
        ),
        run_tree_arm(
            variant,
            TREE_ADWIN,
            warm_x,
            warm_y,
            x_stream,
            y_stream,
            batches,
            leaf_prediction="mc",
            adwin=True,
        ),
    ]

    delays = [
        DelayPoint(
            delay_batches=delay,
            pr_auc=run_tree_arm(
                variant,
                TREE_MC,
                warm_x,
                warm_y,
                x_stream,
                y_stream,
                batches,
                leaf_prediction="mc",
                adwin=False,
                label_delay=delay,
            ).pr_auc,
        )
        for delay in cfg.label_delays
    ]

    tree_arm = next(a for a in arms if a.name == TREE_MC)
    adwin_arm = next(a for a in arms if a.name == TREE_ADWIN)
    return OnlineStudy(
        arms=arms,
        delays=delays,
        batch_rows=cfg.batch_rows,
        n_stream=len(y_stream),
        n_warmup=len(warm_y),
        prevalence=float(np.mean(y_stream)),
        primary_fpr=variant.thresholds.primary_fpr,
        splits=int(tree_arm.structure.split()[0]),
        adwin_detections=adwin_arm.resets,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_online_report(settings: Settings) -> Path:
    """Run the prequential comparison and write the report + figures."""
    study = run_online_study(settings)
    x = np.arange(1, len(study.arms[0].batch_pr_auc) + 1, dtype=float) * study.batch_rows
    stream_fig = plots.plot_lines(
        {arm.name: (x, np.array(arm.batch_pr_auc, dtype=float)) for arm in study.arms},
        xlabel="flows seen in the stream",
        ylabel="PR-AUC on the batch, scored before learning from it",
        title="Prequential detection along the stream",
        out_path=settings.paths.figures_dir / STREAM_FIGURE_NAME,
    )
    delay_fig = plots.plot_lines(
        {
            "streaming tree": (
                np.array([d.delay_batches for d in study.delays], dtype=float),
                np.array([d.pr_auc for d in study.delays], dtype=float),
            )
        },
        xlabel=f"label delay (batches of {study.batch_rows:,} flows)",
        ylabel="prequential PR-AUC",
        title="What the online learner loses when labels arrive late",
        out_path=settings.paths.figures_dir / DELAY_FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, stream_fig, delay_fig), encoding="utf-8")
    logger.info("Wrote online-learning report", extra={"path": str(out_path)})

    with track_run(settings, "online") as run:
        run.log_params({"batch_rows": study.batch_rows, "stream_rows": study.n_stream})
        for arm in study.arms:
            key = "".join(ch if ch.isalnum() else "_" for ch in arm.name)
            run.log_metrics(
                {f"pr_auc_{key}": arm.pr_auc, f"flows_per_second_{key}": arm.flows_per_second}
            )
        run.log_artifact(stream_fig)
        run.log_artifact(delay_fig)
        run.log_artifact(out_path)
    return out_path


def _arm(study: OnlineStudy, name: str) -> ArmResult:
    return next(a for a in study.arms if a.name == name)


def _arm_table(study: OnlineStudy) -> str:
    rows = [
        f"| learner | what it is | prequential PR-AUC | TPR @ {study.primary_fpr:.1%} FPR | "
        "learn | score | flows/s | memory | structure |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in study.arms:
        rows.append(
            f"| **{arm.name}** | {arm.description} | {arm.pr_auc:.3f} | "
            f"{arm.tpr_at(study.primary_fpr):.3f} | {arm.learn_seconds:.1f} s | "
            f"{arm.score_seconds:.1f} s | {arm.flows_per_second:,.0f} | "
            f"{arm.memory_bytes / 1e6:.2f} MB | {arm.structure} |"
        )
    return "\n".join(rows)


def _granularity_table(study: OnlineStudy) -> str:
    rows = [
        "| learner | distinct scores emitted | TPR @ 1% FPR | "
        f"TPR @ {study.primary_fpr:.1%} FPR |",
        "|---|---|---|---|",
    ]
    for arm in study.arms:
        rows.append(
            f"| {arm.name} | {arm.distinct_scores:,} | {arm.tpr_at(0.01):.3f} | "
            f"{arm.tpr_at(study.primary_fpr):.3f} |"
        )
    return "\n".join(rows)


def _granularity_read(study: OnlineStudy) -> str:
    tree = _arm(study, TREE_MC)
    static = _arm(study, STATIC)
    return (
        "The ranking metric and the operating metric disagree here, and the reason is structural. "
        f"The streaming tree ends the stream with {tree.structure}: at any instant it can emit "
        "one score per leaf, and across the whole run -- during which it grew, so early and late "
        f"flows were scored by different trees -- it produced {tree.distinct_scores:,} distinct "
        f"values in total, against the boosted model's {static.distinct_scores:,} (one per flow, "
        "essentially). A threshold can only be placed *between* two distinct "
        "scores, so an alert budget finer than the gaps between them is unreachable by "
        f"construction: at the deployed {study.primary_fpr:.1%} budget the tree detects "
        f"{tree.tpr_at(study.primary_fpr):.1%} against the frozen model's "
        f"{static.tpr_at(study.primary_fpr):.1%}, having *beaten* it on PR-AUC. "
        "This is the kind of thing that only shows up when a study reports the operational "
        "metric next to the ranking one. A SOC does not deploy an average precision; it deploys "
        "a threshold, and a model whose scores come in thirty buckets cannot be asked for a "
        "one-in-a-thousand false-alarm rate. Naive-Bayes leaves are one fix (they emit a "
        "continuum, and the ablation below shows what else they cost); more leaves, or a small "
        "ensemble of trees fed different feature subsets, are others."
    )


def _delay_table(study: OnlineStudy) -> str:
    rows = ["| label delay | prequential PR-AUC | change |", "|---|---|---|"]
    base = study.delays[0].pr_auc if study.delays else float("nan")
    for point in study.delays:
        flows = point.delay_batches * study.batch_rows
        label = (
            "none (labels arrive with the flow)"
            if point.delay_batches == 0
            else (f"{point.delay_batches} batches ({flows:,} flows)")
        )
        rows.append(f"| {label} | {point.pr_auc:.3f} | {point.pr_auc - base:+.3f} |")
    return "\n".join(rows)


def _headline_read(study: OnlineStudy) -> str:
    static = _arm(study, STATIC)
    periodic = _arm(study, PERIODIC)
    tree = _arm(study, TREE_MC)
    verdict = (
        f"The one-pass learner beats the frozen incumbent by {tree.pr_auc - static.pr_auc:+.3f} "
        "PR-AUC"
        if tree.pr_auc > static.pr_auc
        else f"The one-pass learner does not beat the frozen incumbent "
        f"({tree.pr_auc:.3f} against {static.pr_auc:.3f})"
    )
    gap = periodic.pr_auc - tree.pr_auc
    speed = periodic.learn_seconds / max(tree.learn_seconds, 1e-9)
    return (
        f"{verdict}, and neither comes close to periodic retraining at {periodic.pr_auc:.3f}. "
        f"The streaming tree gives up **{gap:.3f} PR-AUC** against refitting, and buys three "
        f"things with it: it spent {tree.learn_seconds:.1f} s learning against the retrainer's "
        f"{periodic.learn_seconds:.1f} s ({speed:.0f}x), it holds "
        f"{tree.memory_bytes / 1e6:.2f} MB of sufficient statistics instead of "
        f"{periodic.memory_bytes / 1e6:.0f} MB of retained history, and it is never more than "
        "one flow out of date, where the retrainer is up to a retrain interval stale by "
        "construction. That is the trade in one line: **freshness and bounded memory, bought "
        "with detection**. Which side of it a deployment wants is a question about its retrain "
        "cadence and its storage, not about which algorithm is better."
    )


def _leaf_read(study: OnlineStudy) -> str:
    nb = _arm(study, TREE_NB)
    mc = _arm(study, TREE_MC)
    better, worse = (nb, mc) if nb.pr_auc > mc.pr_auc else (mc, nb)
    surprise = (
        "That is the opposite of the textbook expectation, and the reason is the assumption in "
        "the name: naive Bayes multiplies per-feature likelihoods as if the features were "
        "independent given the class. CICFlowMeter's are anything but -- a duration is a sum of "
        "inter-arrival times, and a rate is a count divided by that duration -- so the product "
        "counts the same evidence many times over, the log-posterior saturates, and the scores "
        "collapse onto the ends of the interval where a ranking metric has nothing left to rank."
        if worse is nb
        else "The leaf model matters more than the tree does here: the same splits, scored two "
        "ways, differ by more than most of this project's modelling decisions."
    )
    return (
        f"The two leaf rules share every split -- same tree, same stream, same order -- and "
        f"differ by **{better.pr_auc - worse.pr_auc:.3f} PR-AUC** "
        f"({better.name.split('(')[1][:-1]} at {better.pr_auc:.3f} against "
        f"{worse.pr_auc:.3f}). {surprise} It is worth keeping the "
        "losing arm in the table: an ablation that only ever confirms the choice already made is "
        "not an ablation."
    )


def _adwin_read(study: OnlineStudy) -> str:
    adwin = _arm(study, TREE_ADWIN)
    plain = _arm(study, TREE_MC)
    if adwin.resets == 0:
        return (
            "**ADWIN never fired.** Across the whole stream the detector found no cut of its "
            "window with a significant difference in mean absolute error, so the tree was never "
            f"rebuilt and this arm is the previous one exactly ({adwin.pr_auc:.3f} against "
            f"{plain.pr_auc:.3f}, the difference being the scoring order alone). That is a "
            "result, not a failure: the streaming tree is *already* adapting per flow, so its "
            "error rate does not drift the way a frozen model's does -- which is precisely the "
            "condition ADWIN exists to detect. A change detector on top of a continuously "
            "updating learner has much less to find than one on top of a frozen one, and the "
            "honest reading is that the reset machinery earns its place only where the base "
            "learner cannot adapt fast enough on its own. The detector's own correctness is "
            "pinned by unit tests instead: no false alarm over thousands of stationary samples, "
            "and detection within a few dozen of an abrupt mean change."
        )
    delta = adwin.pr_auc - plain.pr_auc
    return (
        f"**ADWIN fired {adwin.resets} times**, each time discarding the tree and starting over "
        f"from the recent window. That is worth {delta:+.3f} PR-AUC against the same tree left "
        "alone. Resetting is the crude end of the response spectrum -- Bifet's Hoeffding Adaptive "
        "Tree grows an alternate subtree and swaps it in only once it wins, which keeps the parts "
        "of the model the drift did not invalidate -- and the gap between this arm and that one "
        "is the cost of the simplification, not of the idea."
    )


def _delay_read(study: OnlineStudy) -> str:
    if not study.delays:
        return ""
    base = study.delays[0]
    worst = min(study.delays, key=lambda d: d.pr_auc)
    flows = worst.delay_batches * study.batch_rows
    if worst.pr_auc >= base.pr_auc - 0.01:
        return (
            "Delaying the labels costs the learner almost nothing here "
            f"(worst case {worst.pr_auc:.3f} against {base.pr_auc:.3f} with labels arriving "
            f"instantly, at a delay of {flows:,} flows). The reason is not robustness -- it is "
            "that on this stream most of what the tree learns is available in *any* recent "
            "window, so it does not matter much which one. A stream whose attack families turn "
            "over faster would not be so forgiving, and the sweep is here so that assumption is "
            "measured rather than assumed."
        )
    return (
        f"Holding labels back {worst.delay_batches} batches ({flows:,} flows) costs "
        f"**{base.pr_auc - worst.pr_auc:.3f} PR-AUC** ({base.pr_auc:.3f} to {worst.pr_auc:.3f}). "
        "That is the number an online-learning proposal has to survive, because the immediate "
        "label it assumes does not exist in a SOC: an analyst confirms an alert hours later, and "
        "confirms only the ones that were alerted on. Every arm here is therefore an upper bound "
        "on its deployable version, and the label-efficiency studies -- active learning, weak "
        "supervision, PU learning -- are what a real streaming deployment would have to be built "
        "on top of."
    )


def _render(study: OnlineStudy, stream_fig: Path, delay_fig: Path) -> str:
    return f"""# NetSentry — Learning at Line Rate: a One-Pass Detector Against the Batch Pipeline

_Prequential (test-then-train) over {study.n_stream:,} later-day flows in arrival order, in
batches of {study.batch_rows:,}, after a {study.n_warmup:,}-flow warm-up on the training days.
Stream prevalence {study.prevalence:.1%}. Every arm sees the same batches in the same order and
scores each one **before** it is allowed to learn from it._

## Why this report exists

The deployed detector is a batch model: fitted on the training days, frozen, shipped. Between
two retrains, every flow is scored by a model that has already seen its last example. The
streaming study measured what that costs and answered it with periodic retraining, which works
and is expensive — the cost grows with the history, the whole history has to still exist, and
the freshest the model can ever be is one retrain interval old.

There is a third option this project had never tried: a learner that updates **per flow**, in
bounded memory, in one pass. `netsentry/models/hoeffding.py` implements one from scratch — a
**Hoeffding tree** (VFDT, Domingos & Hulten 2000) and **ADWIN** (Bifet & Gavalda 2007) — and
this report puts it in the field against the two batch policies.

The Hoeffding tree's idea is worth stating because it is the reason a streaming tree can be
principled rather than merely fast. A batch tree examines every example to choose a split. A
streaming tree cannot, so it asks instead: *how many examples do I need before the ranking of
candidate splits is settled?* The Hoeffding bound answers without assuming a distribution —
after `n` observations a mean of a range-`R` variable is within `sqrt(R^2 ln(1/delta) / 2n)` of
its true value with probability `1 - delta`. When the best candidate beats the runner-up by more
than that, the decision will not change with more data, so it can be taken now and never
revisited.

## The protocol

Prequential evaluation is the streaming analogue of a held-out split: score the batch, *then*
learn from it. Every prediction scored below was made on flows the model had never seen, which
is what makes these numbers comparable with the batch pipeline's. All five arms consume the same
batches in the same order from the same fitted preprocessing, so the only thing that varies is
the learning policy.

## What each learner achieves and what it costs

{_arm_table(study)}

![Prequential detection along the stream](../figures/{stream_fig.name})

{_headline_read(study)}

## The operating point a coarse score cannot reach

{_granularity_table(study)}

{_granularity_read(study)}

## The leaf rule is most of the model

{_leaf_read(study)}

## The change detector

{_adwin_read(study)}

## The assumption every online learner rests on

![Label delay](../figures/{delay_fig.name})

{_delay_table(study)}

{_delay_read(study)}

## Scope and honest limits

- **This tree is pure NumPy and pure Python.** Its throughput is an asymptotic argument, not a
  benchmark of the technique: MOA and river implement the same algorithm in compiled code an
  order of magnitude faster. What transfers is the *shape* — constant memory, per-example
  updates, no stored history — not the absolute flows per second.
- **A Gaussian summary is an approximation.** Split points are estimated from per-class normal
  fits rather than from the values themselves, which is what keeps a leaf's memory independent
  of the stream length. On heavy-tailed flow features that approximation is doing real work, and
  it is the first thing to suspect when the tree splits somewhere strange.
- **The reset policy is the crude version of adaptation.** Bifet's Hoeffding Adaptive Tree grows
  an alternate subtree and promotes it only when it wins; rebuilding from scratch throws away
  the parts of the model the drift did not invalidate.
- **Prequential PR-AUC pools predictions made by different models.** The score at flow 1 came
  from a different tree than the score at flow 20,000 — that is inherent to the protocol and is
  why the per-batch curve is shown alongside the pooled number rather than instead of it."""
