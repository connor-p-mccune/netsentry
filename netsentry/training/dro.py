"""Group DRO: training for the worst case, not the average one.

Empirical risk minimisation optimises a mean, and a mean belongs to whoever contributes most
of it. Any subpopulation that is a small share of traffic can be served badly without the
objective noticing — which is exactly what the [per-service parity audit](subgroups.md)
found at the operating point. Distributionally robust optimisation replaces the average with
the worst group:

    minimise over theta  of  max over groups g  of  E[ loss | group g ]

Sagawa, Koh, Hashimoto & Liang (*Distributionally Robust Neural Networks*, ICLR 2020) solve
that saddle point with online exponentiated gradient: keep a weight per group, upweight
whichever group is currently doing worst, refit, repeat. The inner minimisation here is a
LightGBM fit with per-row sample weights rather than a gradient step, which coarsens the
optimisation dynamics without changing the game being played.

**Choosing the groups is most of the work, and the obvious choice fails here.** Grouping by
service looks right and is unusable: on this data most services are a single class end to
end — DNS, IMAP, POP3 and SMTP carry no attacks at all, ephemeral ports and FTP carry
nothing else. A group that is 100% one class is not a subpopulation, it is a label, and
"worst-group loss" over such a partition collapses into "hardest class". That collinearity
is not a quirk of the synthetic generator: attacks concentrate on service ports in the real
capture too, which is precisely why ``Destination Port`` is dropped as a model feature. The
same property that makes the port a leakage risk makes the service a useless DRO group, and
the report shows the composition table rather than asserting it.

So the groups are **capture days**: a genuine operational partition, every one carrying both
classes at very different rates (0%, 13% and 38% attack traffic), and the axis this project
cares about most. That turns the study into a sharper question than parity — does
worst-case training over the days you have generalise to days you have never seen? The
answer is measured against two controls, plain ERM and a size-balanced arm that applies the
same per-group normalisation with no adversary at all, so whatever the adversary contributes
is isolated from the effect of merely equalising group sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DESTINATION_PORT
from netsentry.data.services import OTHER_SERVICE, service_of
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import DROConfig

logger = get_logger(__name__)

REPORT_NAME = "dro.md"
WEIGHT_FIGURE = "dro_group_weights.png"
PARITY_FIGURE = "dro_parity.png"


# --------------------------------------------------------------------------------------
# The DRO game (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
def group_losses(
    y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, group_names: list[str]
) -> np.ndarray:
    """Mean log-loss within each group — what the adversary in the DRO game reads.

    Log loss rather than a rate metric because the group weights need a quantity that moves
    smoothly as the model changes; a group's TPR at a fixed threshold can sit still through
    a large improvement and then jump, which makes the weight updates jerk around.
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    per_row = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    out = np.zeros(len(group_names), dtype=float)
    for i, name in enumerate(group_names):
        mask = groups == name
        out[i] = float(per_row[mask].mean()) if mask.any() else 0.0
    return out


def exponentiated_gradient_step(
    weights: np.ndarray, losses: np.ndarray, step_size: float
) -> np.ndarray:
    """One multiplicative-weights update: upweight whichever group is doing worst.

    The adversary's half of the saddle point (Sagawa et al. 2020). Multiplicative rather
    than additive so weights stay positive and normalised without projection, and so a group
    that is badly behind is amplified geometrically rather than linearly. Losses are shifted
    by their max before exponentiating, which is a no-op after renormalisation but keeps the
    exponentials from overflowing.
    """
    w = np.asarray(weights, dtype=float)
    lo = np.asarray(losses, dtype=float)
    updated = w * np.exp(step_size * (lo - lo.max()))
    total = updated.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(w, 1.0 / len(w))
    return np.asarray(updated / total, dtype=float)


def sample_weights_from_groups(
    groups: np.ndarray, group_names: list[str], weights: np.ndarray
) -> np.ndarray:
    """Spread each group's weight evenly across its rows.

    Dividing by the group's size is what makes this DRO rather than plain reweighting: the
    objective is the worst *per-group average* loss, so a group's influence must not grow
    just because it has more rows. Without the division a large group with a middling loss
    would keep outvoting a small group that is failing.
    """
    out = np.ones(len(groups), dtype=float)
    for name, w in zip(group_names, np.asarray(weights, dtype=float), strict=True):
        mask = groups == name
        n = int(mask.sum())
        if n:
            out[mask] = w / n
    normalised: np.ndarray = out * (len(groups) / max(out.sum(), 1e-12))
    return normalised


def worst_group(values: dict[str, float]) -> tuple[str, float]:
    """The group with the lowest score, and that score — the number DRO is trying to raise."""
    if not values:
        return "", float("nan")
    name = min(values, key=lambda k: values[k])
    return name, values[name]


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class GroupComposition:
    """How much of one candidate group is attack traffic — the collinearity diagnostic."""

    name: str
    n: int
    attack_share: float

    @property
    def degenerate(self) -> bool:
        """A group that is all one class cannot be a fairness group; it is a class."""
        return self.attack_share <= 0.0 or self.attack_share >= 1.0


@dataclass
class ArmMetrics:
    """One trained model, scored overall and on each held-out day."""

    name: str
    pr_auc: float
    detection: float
    fpr: float
    per_day_pr_auc: dict[str, float]
    per_day_detection: dict[str, float]
    worst_train_group_loss: float

    @property
    def worst_day(self) -> tuple[str, float]:
        """Held-out day the model handles worst, by PR-AUC — the transfer question."""
        return worst_group(self.per_day_pr_auc)


@dataclass
class RoundTrace:
    """One DRO round: what the adversary believed, and what the learner then did."""

    index: int
    max_group_loss: float
    mean_group_loss: float
    worst_group_name: str
    heaviest_group: str
    heaviest_weight: float
    val_pr_auc: float


@dataclass
class DROStudy:
    """Everything the report renders."""

    group_by: str
    groups: list[str]
    group_sizes: dict[str, int]
    group_attack_share: dict[str, float]
    service_composition: list[GroupComposition]
    target_fpr: float
    rounds: list[RoundTrace]
    arms: list[ArmMetrics]
    step_size: float
    n_rounds: int
    test_days: list[str]
    selected_round: int


def _services(frame: pd.DataFrame) -> np.ndarray:
    """Service label per flow, from the destination port the model never sees as a feature."""
    if DESTINATION_PORT not in frame.columns:
        return np.full(len(frame), OTHER_SERVICE)
    return np.array([service_of(p) for p in frame[DESTINATION_PORT].to_numpy()])


def service_composition(frame: pd.DataFrame, y: np.ndarray) -> list[GroupComposition]:
    """Attack share of every service — the evidence that services cannot be DRO groups here."""
    services = _services(frame)
    out: list[GroupComposition] = []
    for name in sorted(set(services.tolist())):
        mask = services == name
        out.append(
            GroupComposition(
                name=name,
                n=int(mask.sum()),
                attack_share=float(np.asarray(y)[mask].mean()) if mask.any() else 0.0,
            )
        )
    return out


def _evaluate(
    name: str,
    y_test: np.ndarray,
    scores: np.ndarray,
    days: np.ndarray,
    day_names: list[str],
    threshold: float,
    worst_train_loss: float,
) -> ArmMetrics:
    """Score one arm overall and on each held-out day at the shared operating point."""
    per_pr: dict[str, float] = {}
    per_det: dict[str, float] = {}
    for d in day_names:
        mask = days == d
        if not mask.any() or len(np.unique(y_test[mask])) < 2:
            continue
        per_pr[d] = float(average_precision_score(y_test[mask], scores[mask]))
        per_det[d] = rates_at_threshold(y_test[mask], scores[mask], threshold)["tpr"]
    rates = rates_at_threshold(y_test, scores, threshold)
    return ArmMetrics(
        name=name,
        pr_auc=float(average_precision_score(y_test, scores)),
        detection=rates["tpr"],
        fpr=rates["fpr"],
        per_day_pr_auc=per_pr,
        per_day_detection=per_det,
        worst_train_group_loss=worst_train_loss,
    )


def run_dro(settings: Settings) -> DROStudy:
    """Train ERM, size-balanced and group-DRO detectors; compare how they transfer."""
    cfg: DROConfig = settings.dro
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    day_col = variant.split.day_column
    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign_label = variant.labels.benign_label
    target_fpr = variant.thresholds.primary_fpr

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    g_train = train[day_col].astype(str).to_numpy()
    g_val = val[day_col].astype(str).to_numpy()
    days_test = test[day_col].astype(str).to_numpy()
    counts = {g: int((g_train == g).sum()) for g in sorted(set(g_train.tolist()))}
    group_names = [g for g, n in counts.items() if n >= cfg.min_group_size]
    shares = {g: float(y_train[g_train == g].mean()) for g in group_names}
    test_days = sorted(set(days_test.tolist()))
    logger.info("DRO groups resolved", extra={"group_by": cfg.group_by, "n": len(group_names)})

    def _fit(weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        seed_everything(variant.seed)
        model = SupervisedClassifier(variant).fit(
            x_train, y_train, eval_set=(x_val, y_val), sample_weight=weights
        )
        s_val = attack_probability(
            np.asarray(model.predict_proba(x_val)), model.classes_, benign_label
        )
        s_test = attack_probability(
            np.asarray(model.predict_proba(x_test)), model.classes_, benign_label
        )
        return s_val, s_test

    erm_val, erm_test = _fit(None)
    uniform_weights = sample_weights_from_groups(
        g_train, group_names, np.full(len(group_names), 1.0 / max(len(group_names), 1))
    )
    bal_val, bal_test = _fit(uniform_weights)

    # The DRO game: the adversary reweights groups, the learner refits, repeat.
    q = np.full(len(group_names), 1.0 / max(len(group_names), 1))
    rounds: list[RoundTrace] = []
    best_score = np.inf
    best_round = 0
    best_pair: tuple[np.ndarray, np.ndarray] = (erm_val, erm_test)
    for r in range(int(cfg.n_rounds)):
        row_weights = sample_weights_from_groups(g_train, group_names, q)
        s_val, s_test = _fit(row_weights)
        losses = group_losses(y_val, s_val, g_val, group_names)
        heaviest = int(np.argmax(q))
        rounds.append(
            RoundTrace(
                index=r + 1,
                max_group_loss=float(losses.max()),
                mean_group_loss=float(losses.mean()),
                worst_group_name=group_names[int(np.argmax(losses))],
                heaviest_group=group_names[heaviest],
                heaviest_weight=float(q[heaviest]),
                val_pr_auc=float(average_precision_score(y_val, s_val)),
            )
        )
        # Model selection on the objective DRO exists to minimise, measured on validation.
        if float(losses.max()) < best_score:
            best_score = float(losses.max())
            best_round = r + 1
            best_pair = (s_val, s_test)
        q = exponentiated_gradient_step(q, losses, cfg.step_size)
        logger.info(
            "DRO round complete",
            extra={"round": r + 1, "worst": rounds[-1].worst_group_name, "loss": best_score},
        )

    dro_val, dro_test = best_pair
    arms = []
    for name, (s_val, s_test) in (
        ("ERM (deployed)", (erm_val, erm_test)),
        ("size-balanced days", (bal_val, bal_test)),
        ("group DRO over days", (dro_val, dro_test)),
    ):
        threshold = threshold_at_fpr(y_val, s_val, target_fpr)
        worst_loss = float(group_losses(y_val, s_val, g_val, group_names).max())
        arms.append(_evaluate(name, y_test, s_test, days_test, test_days, threshold, worst_loss))

    return DROStudy(
        group_by=cfg.group_by,
        groups=group_names,
        group_sizes={g: counts[g] for g in group_names},
        group_attack_share=shares,
        service_composition=service_composition(train, y_train),
        target_fpr=target_fpr,
        rounds=rounds,
        arms=arms,
        step_size=cfg.step_size,
        n_rounds=int(cfg.n_rounds),
        test_days=test_days,
        selected_round=best_round,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_dro_report(settings: Settings) -> Path:
    """Run the group-DRO study and write the report + figures."""
    study = run_dro(settings)

    idx = np.array([r.index for r in study.rounds], dtype=float)
    game_fig = plots.plot_lines(
        {
            "worst group's loss (what DRO minimises)": (
                idx,
                np.array([r.max_group_loss for r in study.rounds]),
            ),
            "average group loss (what ERM minimises)": (
                idx,
                np.array([r.mean_group_loss for r in study.rounds]),
            ),
        },
        xlabel="DRO round",
        ylabel="validation log loss",
        title="The saddle point, round by round",
        out_path=settings.paths.figures_dir / WEIGHT_FIGURE,
    )
    transfer_fig = plots.plot_barh(
        [f"{d} - {a.name}" for d in study.test_days for a in study.arms],
        [a.per_day_pr_auc.get(d, 0.0) for d in study.test_days for a in study.arms],
        xlabel="PR-AUC on the held-out day",
        title="Does worst-day training transfer to days nobody trained on?",
        out_path=settings.paths.figures_dir / PARITY_FIGURE,
    )

    report = _render(study, game_fig, transfer_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote DRO report", extra={"path": str(out_path)})

    with track_run(settings, "dro") as run:
        run.log_params(
            {"n_rounds": study.n_rounds, "step_size": study.step_size, "group_by": study.group_by}
        )
        metrics: dict[str, float] = {}
        for arm in study.arms:
            key = arm.name.split()[0].lower()
            metrics[f"{key}_pr_auc"] = arm.pr_auc
            metrics[f"{key}_worst_day_pr_auc"] = arm.worst_day[1]
            metrics[f"{key}_worst_train_group_loss"] = arm.worst_train_group_loss
        run.log_metrics(metrics)
        run.log_artifact(game_fig)
        run.log_artifact(transfer_fig)
        run.log_artifact(out_path)
    return out_path


def _service_table(study: DROStudy) -> str:
    rows = ["| service | training flows | attack share | usable as a group? |", "|---|---|---|---|"]
    for c in study.service_composition:
        verdict = "**no — one class only**" if c.degenerate else "yes"
        rows.append(f"| {c.name} | {c.n:,} | {c.attack_share:.1%} | {verdict} |")
    return "\n".join(rows)


def _group_table(study: DROStudy) -> str:
    rows = ["| group | training flows | attack share |", "|---|---|---|"]
    for g in study.groups:
        rows.append(
            f"| {g} | {study.group_sizes[g]:,} | {study.group_attack_share.get(g, 0.0):.1%} |"
        )
    return "\n".join(rows)


def _arm_table(study: DROStudy) -> str:
    header = (
        "| arm | test PR-AUC | detection | FPR | worst training-group loss | worst held-out day |"
    )
    rows = [header, "|---|---|---|---|---|---|"]
    for a in study.arms:
        day, value = a.worst_day
        rows.append(
            f"| **{a.name}** | {a.pr_auc:.3f} | {a.detection:.1%} | {a.fpr:.3%} "
            f"| {a.worst_train_group_loss:.4f} | {day} ({value:.3f}) |"
        )
    return "\n".join(rows)


def _day_table(study: DROStudy) -> str:
    header = "| held-out day | " + " | ".join(a.name for a in study.arms) + " |"
    rows = [header, "|---|" + "---|" * len(study.arms)]
    for d in study.test_days:
        cells = " | ".join(f"{a.per_day_pr_auc.get(d, float('nan')):.3f}" for a in study.arms)
        rows.append(f"| {d} | {cells} |")
    return "\n".join(rows)


def _round_table(study: DROStudy) -> str:
    rows = [
        "| round | worst group loss | mean group loss | worst group | heaviest weight "
        "| validation PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in study.rounds:
        rows.append(
            f"| {r.index} | {r.max_group_loss:.4f} | {r.mean_group_loss:.4f} "
            f"| {r.worst_group_name} | {r.heaviest_group} ({r.heaviest_weight:.2f}) "
            f"| {r.val_pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _why_days_read(study: DROStudy) -> str:
    degenerate = [c for c in study.service_composition if c.degenerate]
    usable = [c for c in study.service_composition if not c.degenerate]
    return (
        f"The natural grouping for a parity study is the **service**, and it is the one the "
        f"[parity audit](subgroups.md) uses. It cannot carry a DRO objective on this data: "
        f"{len(degenerate)} of {len(study.service_composition)} services are a single class "
        f"end to end, leaving {len(usable)} with any mixture at all.\n\n{_service_table(study)}\n\n"
        "A group that is 100% attack or 100% benign is not a subpopulation, it is a label. "
        "Upweighting it does not ask the model to be fair to an under-served slice of traffic, "
        "it asks the model to predict one class harder — so 'worst-group loss' would reduce to "
        "'hardest class' and the whole exercise would measure nothing. This is not an artefact "
        "of the synthetic generator either: attacks concentrate on service ports in the real "
        "capture too, which is exactly why `Destination Port` is dropped as a model feature in "
        "the first place ([DATA_CARD](../DATA_CARD.md)). **The same collinearity that makes the "
        "port a leakage risk makes the service a useless DRO group.** Worth stating, because "
        "reaching for group DRO without checking the group definition is the standard way this "
        "method gets misapplied.\n\nSo the groups here are **capture days**, which are a genuine "
        "operational partition, carry both classes at very different rates, and happen to be the "
        "axis this project cares most about."
    )


def _verdict(study: DROStudy) -> str:
    if len(study.arms) < 3:
        return ""
    erm, balanced, dro = study.arms[0], study.arms[1], study.arms[2]
    _, erm_worst = erm.worst_day
    _, bal_worst = balanced.worst_day
    _, dro_worst = dro.worst_day
    loss_moved = erm.worst_train_group_loss - dro.worst_train_group_loss
    objective = (
        f"DRO did what it promised on its own objective: the worst training group's validation "
        f"loss falls from {erm.worst_train_group_loss:.4f} to {dro.worst_train_group_loss:.4f} "
        f"({loss_moved:+.4f})."
        if loss_moved > 0
        else (
            f"DRO did not even win on its own objective: the worst training group's validation "
            f"loss is {dro.worst_train_group_loss:.4f} against ERM's "
            f"{erm.worst_train_group_loss:.4f}. With a full refit per round rather than a "
            "gradient step, the game takes coarse steps and can overshoot."
        )
    )
    best = max(
        (("ERM", erm.pr_auc), ("size-balanced", balanced.pr_auc), ("group DRO", dro.pr_auc)),
        key=lambda kv: kv[1],
    )[0]
    transfer = (
        f"But the question that matters is transfer, and there it is a wash at best. On the "
        f"unseen days, PR-AUC is {erm.pr_auc:.3f} for ERM, {balanced.pr_auc:.3f} size-balanced "
        f"and {dro.pr_auc:.3f} for DRO, with the worst held-out day at {erm_worst:.3f} / "
        f"{bal_worst:.3f} / {dro_worst:.3f}. The best arm overall is **{best}**."
    )
    control = (
        "The size-balanced control earns its place in the table here: it applies the same "
        "per-group normalisation as DRO with no adversary at all, so any gap between it and "
        "plain ERM is the effect of equalising day sizes, and any gap between it and DRO is what "
        "the adversary actually contributed. "
    )
    if study.selected_round == 1:
        return (
            f"{objective} {transfer}\n\n{control}And here the gap is exactly nothing, for a "
            "reason worth stating rather than leaving as a coincidence of three identical "
            "columns: **the round DRO selected was round 1, whose weights are still uniform.** "
            "Every subsequent round — every round in which the adversary actually did something "
            "— scored worse on the very objective it was maximising against. Group DRO, given "
            "the chance to reweight, chose not to. The two columns are identical by "
            "construction, not by luck, and the honest reading is that the adversary found "
            "nothing to exploit."
        )
    if dro.pr_auc > max(erm.pr_auc, balanced.pr_auc) + 0.005:
        return (
            f"{objective} {transfer} {control}Here the adversary earns its keep — worst-case "
            "training over past days buys generalisation to future ones, which is the strongest "
            "version of the claim this study could have supported."
        )
    return (
        f"{objective} {transfer} {control}Here it does not: reweighting toward the hardest "
        "training day does not make the model better on days nobody trained on. That is a "
        "coherent result rather than a disappointing one. DRO gives worst-case robustness "
        "**over the groups it was shown**, and Thursday and Friday are not among them — they "
        "are a fourth and fifth distribution, related to Monday-Wednesday but not contained in "
        "their convex hull. Robustness over a set of observed environments only extrapolates to "
        "an unobserved one if the unobserved one is a mixture of them, and the "
        "[covariate-shift study](covariate_shift.md) already established that this temporal gap "
        "is *concept* shift, not a re-weighting of the same feature distribution. Two "
        "independent methods, the same conclusion, arrived at from opposite directions."
    )


def _dynamics_read(study: DROStudy) -> str:
    if not study.rounds:
        return ""
    first, last = study.rounds[0], study.rounds[-1]
    moved = last.max_group_loss - first.max_group_loss
    chased = len({r.worst_group_name for r in study.rounds})
    stuck = {r.worst_group_name for r in study.rounds}
    backfired = ""
    if moved > 0 and len(stuck) == 1 and len(study.rounds) > 2:
        target = next(iter(stuck))
        w_first = study.rounds[0].heaviest_weight
        w_last = study.rounds[-1].heaviest_weight
        backfired = (
            f"\n\n**The interesting failure is that upweighting the worst group made it worse.** "
            f"{target} is the hardest group in every round, the adversary's weight on it climbs "
            f"from {w_first:.2f} to {w_last:.2f}, and across that climb {target}'s own loss rises "
            f"from {first.max_group_loss:.4f} to {last.max_group_loss:.4f}. That is not a bug in "
            "the update, it is the premise of the method failing. Weight is a fixed budget: "
            "emphasising one day necessarily de-emphasises the others, and the model learns "
            f"{target}'s attacks largely *from the other days' attacks* — the families overlap, "
            "so the data that helps most is not the data that scores worst. Group DRO assumes a "
            "group's difficulty is fixable by paying it more attention. That holds when groups "
            "are genuinely separate sub-populations and fails when they share their signal, "
            "which is the common case in traffic captured from one network. The diagnostic is "
            "cheap and this report is what it looks like: if the worst-group loss rises "
            "monotonically as its weight rises, the partition is wrong for DRO, not the model."
        )
    return (
        f"The worst group's loss went from {first.max_group_loss:.4f} to "
        f"{last.max_group_loss:.4f} ({moved:+.4f}) while the average moved "
        f"{last.mean_group_loss - first.mean_group_loss:+.4f}. Across {study.n_rounds} rounds "
        f"the hardest group changed {chased} time"
        + ("s" if chased != 1 else "")
        + (
            " — the adversary keeps switching targets, which is what a saddle point looks like "
            "when no single group is uniquely hard and the learner can trade them against each "
            "other."
            if chased > 1
            else " — one group is persistently hardest, and upweighting it does not fix it, "
            "which usually means the difficulty is in the data rather than in the emphasis."
        )
        + " The inner minimisation is a full LightGBM refit rather than a gradient step, so this "
        "explores a handful of large moves in the game rather than converging it; a longer run "
        "with a smaller step would be a fairer test of DRO's ceiling and a far more expensive "
        "one." + backfired
    )


def _render(study: DROStudy, game_fig: Path, transfer_fig: Path) -> str:
    return f"""# NetSentry — Group DRO: Training for the Worst Case, Not the Average One

_Synthetic stand-in. Honest temporal/binary split; groups are the {len(study.groups)} training
capture days. All three arms are judged on the same later-day test set
({", ".join(study.test_days)}) at each arm's own validated {study.target_fpr:.1%} false-positive
budget. {study.n_rounds} DRO rounds at step size {study.step_size:g}._

## Why this report exists

Empirical risk minimisation optimises a mean, and a mean belongs to whoever contributes most of
it. Any subpopulation that is a small share of traffic can be served badly without the objective
noticing. Distributionally robust optimisation replaces that objective with the worst group's:

```
minimise over theta   max over groups g   E[ loss | group g ]
```

and Sagawa, Koh, Hashimoto & Liang (ICLR 2020) solve the saddle point by online exponentiated
gradient — keep a weight per group, upweight whoever is doing worst, refit, repeat. The inner
step here is a weighted LightGBM fit rather than a gradient step, which coarsens the dynamics
without changing the game.

## Choosing the groups is most of the work

{_why_days_read(study)}

{_group_table(study)}

## Three arms

{_arm_table(study)}

{_verdict(study)}

## Transfer, day by day

{_day_table(study)}

![per-day PR-AUC by arm](../figures/{transfer_fig.name})

## The game, round by round

{_round_table(study)}

{_dynamics_read(study)}

![worst-group vs average loss](../figures/{game_fig.name})

## Scope

Group weights are updated on **validation** loss and the deployed round is chosen by validation
worst-group loss, so the test days stay untouched — but that also hands DRO a model-selection
step the ERM arm does not get, and it still has to win with it. Three groups is a thin game;
DRO's guarantees are asymptotic in neither the number of groups nor the number of rounds, and
with a full refit per round this run explores the game rather than solving it. Each arm re-picks
its own threshold on validation at the shared budget, so the detection column compares operating
points that were calibrated the same way rather than a single threshold applied to differently
scaled scores. Monday carries no attacks at all, which makes its per-group loss a pure
false-positive term — the [federated study](federated.md) hit the same fact from another
direction and found an all-benign site produces an accidental one-class fit. Worst-group results
on a group that contains one class should be read as "how confidently does it clear benign
traffic", not as detection."""
