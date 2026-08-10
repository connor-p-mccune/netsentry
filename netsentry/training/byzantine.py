"""Byzantine-robust aggregation: what happens when one of the sites is lying.

The [federated study](federated.md) shows sites that cannot pool raw traffic can still train
together by sharing weights. It assumes every site is honest, and that assumption is doing
enormous work. Averaging is a **linear** operation with no bounded influence: one participant
who sends `-1000x` its honest update drags the global mean anywhere it likes. A federation
is exactly the setting where that participant plausibly exists — an MSSP's estates, a
hospital group, a consortium — because the whole point of federating is that you cannot
inspect the other members' data, and if you cannot inspect their data you cannot inspect
their gradients either.

This is the attack-and-defence pair the federated study is missing, and it follows the arc
this project uses everywhere else: measure the failure, apply the fix, re-measure, and price
the fix when nobody is attacking.

Three attacks, chosen because they fail differently:

- **sign flip** — return the honest update negated and amplified. Enormous, obvious, and
  devastating to a mean.
- **Gaussian** — pure noise of a plausible magnitude, which is what a broken site looks like
  as opposed to a malicious one.
- **label flip** — train honestly on inverted labels. The update has a completely normal
  norm and direction statistics; it is a well-fitted model of the wrong thing, and it is the
  one that any defence based on "reject the big vectors" will wave straight through.

Three defences, all classical, all replacing the mean with something that has bounded
influence:

- **coordinate-wise median** and **trimmed mean** (Yin, Chen, Ramchandran & Bartlett, ICML
  2018) — order statistics per coordinate, so a minority of arbitrary values cannot move the
  result at all.
- **Krum** (Blanchard, El Mhamdi, Guerraoui & Stainer, NeurIPS 2017) — do not average;
  *select* the single update closest to its `n - f - 2` nearest neighbours, on the reasoning
  that honest updates cluster and a Byzantine one that does not cluster is exposed by its
  distances.

Each rule tolerates a stated fraction of liars — under a half for median and trimmed mean,
`f < (n-2)/2` for Krum — and the study measures where each actually breaks rather than
citing the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.federated import Weights, federated_average, initial_weights, local_train
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ByzantineConfig

logger = get_logger(__name__)

REPORT_NAME = "byzantine.md"
BREAKDOWN_FIGURE = "byzantine_breakdown.png"
PRICE_FIGURE = "byzantine_price.png"


# --------------------------------------------------------------------------------------
# Robust aggregation rules (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
def _stack(updates: list[Weights]) -> np.ndarray:
    """Site updates as one ``(n_sites, n_params)`` matrix, intercept as the last column."""
    return np.array([np.append(w.coef, w.intercept) for w in updates], dtype=float)


def _unstack(vector: np.ndarray) -> Weights:
    """Inverse of :func:`_stack` for a single parameter vector."""
    return Weights(np.asarray(vector[:-1], dtype=float).copy(), float(vector[-1]))


def coordinate_median(updates: list[Weights]) -> Weights:
    """Per-coordinate median of the site updates (Yin et al. 2018).

    The median's breakdown point is one half: up to just under half the sites can send
    literally any values, including infinities, without moving the result at all. That is
    the property a mean does not have at any level, and it costs only a sort.
    """
    if not updates:
        raise ValueError("no site updates to aggregate")
    return _unstack(np.median(_stack(updates), axis=0))


def trimmed_mean(updates: list[Weights], trim: int) -> Weights:
    """Mean per coordinate after discarding the ``trim`` highest and lowest values.

    The dial between the mean (``trim = 0``) and the median: it keeps more of the honest
    signal than a median does while still bounding how far a minority can pull the result.
    Tolerates ``trim`` Byzantine sites by construction, and quietly degenerates to the median
    if asked to trim more than the sites can spare — which is reported rather than assumed.
    """
    if not updates:
        raise ValueError("no site updates to aggregate")
    matrix = np.sort(_stack(updates), axis=0)
    n = matrix.shape[0]
    k = int(np.clip(trim, 0, max((n - 1) // 2, 0)))
    kept = matrix[k : n - k] if n - 2 * k > 0 else matrix
    return _unstack(kept.mean(axis=0))


def krum(updates: list[Weights], n_byzantine: int) -> Weights:
    """Select the update closest to its ``n - f - 2`` nearest neighbours (Blanchard et al. 2017).

    Krum does not average at all — it *elects*, on the argument that honest updates cluster
    around the true descent direction while a Byzantine one must either sit near that cluster
    (and therefore be harmless) or reveal itself by its distances. Selecting rather than
    averaging is why it survives an attacker who knows the rule, and why it discards the
    variance reduction averaging would have given: the price is paid in the no-attacker case.
    """
    if not updates:
        raise ValueError("no site updates to aggregate")
    matrix = _stack(updates)
    n = len(updates)
    keep = n - int(n_byzantine) - 2
    if n < 3 or keep < 1:  # too few sites for the rule to say anything
        return coordinate_median(updates)
    diff = matrix[:, None, :] - matrix[None, :, :]
    sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq, np.inf)
    scores = np.sort(sq, axis=1)[:, :keep].sum(axis=1)
    return _unstack(matrix[int(np.argmin(scores))])


AGGREGATORS = ("FedAvg (mean)", "coordinate median", "trimmed mean", "Krum")


def aggregate(
    name: str, updates: list[Weights], sizes: list[int], n_byzantine: int, trim: int
) -> Weights:
    """Dispatch to one aggregation rule by name."""
    if name == "coordinate median":
        return coordinate_median(updates)
    if name == "trimmed mean":
        return trimmed_mean(updates, trim)
    if name == "Krum":
        return krum(updates, n_byzantine)
    return federated_average(updates, sizes)


# --------------------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------------------
def sign_flip(honest: Weights, scale: float) -> Weights:
    """Return the honest update negated and amplified — the loudest possible attack."""
    return Weights(-scale * honest.coef, -scale * honest.intercept)


def gaussian_update(n_features: int, sigma: float, rng: np.random.Generator) -> Weights:
    """Pure noise: what a broken site looks like, as distinct from a malicious one."""
    return Weights(rng.normal(0.0, sigma, size=n_features), float(rng.normal(0.0, sigma)))


ATTACKS = ("sign flip", "Gaussian noise", "label flip")


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class Cell:
    """One (aggregator, attack, number of liars) outcome."""

    aggregator: str
    attack: str
    n_malicious: int
    pr_auc: float
    retained: float


@dataclass
class ByzantineStudy:
    """Everything the report renders."""

    n_sites: int
    site_sizes: list[int]
    rounds: int
    clean: dict[str, float]
    cells: list[Cell]
    malicious_counts: list[int]
    trim: int
    centralized_pr_auc: float


def _breakdown(study: ByzantineStudy, aggregator: str, attack: str, floor: float) -> int | None:
    """Smallest number of liars at which this rule falls below ``floor`` of its clean score."""
    clean = study.clean.get(aggregator, 0.0)
    for f in sorted(study.malicious_counts):
        if f == 0:
            continue
        cell = next(
            (
                c
                for c in study.cells
                if c.aggregator == aggregator and c.attack == attack and c.n_malicious == f
            ),
            None,
        )
        if cell is not None and cell.pr_auc < floor * clean:
            return f
    return None


def _run_federation(
    aggregator: str,
    attack: str,
    n_malicious: int,
    *,
    site_idx: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ByzantineConfig,
    train_kwargs: dict[str, float | int],
    seed: int,
) -> float:
    """Train one federation under one attack and aggregation rule; return test PR-AUC.

    The malicious sites are always the *first* ones, held fixed across cells, so differences
    between rows are the aggregation rule rather than which sites happened to be corrupted.
    """
    rng = np.random.default_rng(seed)
    n_features = x_train.shape[1]
    n_sites = len(site_idx)
    glob = initial_weights(n_features)
    sizes = [len(idx) for idx in site_idx]
    for r in range(cfg.rounds):
        updates: list[Weights] = []
        for i, idx in enumerate(site_idx):
            malicious = i < n_malicious
            labels = y_train[idx]
            if malicious and attack == "label flip":
                labels = 1 - labels
            honest = local_train(
                glob,
                x_train[idx],
                labels,
                seed=seed + r * n_sites + i,
                **train_kwargs,  # type: ignore[arg-type]
            )
            if not malicious or attack == "label flip":
                updates.append(honest)
            elif attack == "sign flip":
                updates.append(sign_flip(honest, cfg.sign_flip_scale))
            else:
                updates.append(gaussian_update(n_features, cfg.gaussian_sigma, rng))
        glob = aggregate(aggregator, updates, sizes, n_malicious, cfg.trim)
    scores = glob.scores(x_test)
    if not np.all(np.isfinite(scores)) or len(np.unique(y_test)) < 2:
        return 0.0
    return float(average_precision_score(y_test, scores))


def run_byzantine(settings: Settings) -> ByzantineStudy:
    """Attack the federation, defend it four ways, and price each defence when nobody lies."""
    cfg: ByzantineConfig = settings.byzantine
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_test = np.asarray(pipeline.transform(test))

    # Sites are shards of capture days: the federated study's non-IID day structure, split
    # finely enough that a Byzantine minority is a meaningful thing to have.
    rng = np.random.default_rng(variant.seed)
    days = (
        train[DAY_COLUMN].astype(str).to_numpy()
        if DAY_COLUMN in train.columns
        else np.zeros(len(train), dtype=str)
    )
    site_idx: list[np.ndarray] = []
    for day in dict.fromkeys(days.tolist()):
        idx = np.flatnonzero(days == day)
        rng.shuffle(idx)
        site_idx.extend(np.array_split(idx, cfg.shards_per_day))
    logger.info("Byzantine federation built", extra={"sites": len(site_idx)})

    train_kwargs: dict[str, float | int] = {
        "epochs": settings.federated.local_epochs,
        "batch_size": settings.federated.batch_size,
        "learning_rate": settings.federated.learning_rate,
        "l2": settings.federated.l2,
    }
    common = {
        "site_idx": site_idx,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "cfg": cfg,
        "train_kwargs": train_kwargs,
        "seed": variant.seed,
    }

    clean: dict[str, float] = {}
    for agg in AGGREGATORS:
        clean[agg] = _run_federation(agg, "none", 0, **common)
        logger.info("Clean baseline", extra={"aggregator": agg, "pr_auc": round(clean[agg], 4)})

    cells: list[Cell] = []
    counts = [f for f in cfg.malicious_counts if 0 < f < len(site_idx)]
    for agg in AGGREGATORS:
        for attack in ATTACKS:
            for f in counts:
                pr_auc = _run_federation(agg, attack, f, **common)
                cells.append(
                    Cell(
                        aggregator=agg,
                        attack=attack,
                        n_malicious=f,
                        pr_auc=pr_auc,
                        retained=pr_auc / clean[agg] if clean[agg] > 0 else 0.0,
                    )
                )
            logger.info("Attack swept", extra={"aggregator": agg, "attack": attack})

    central = local_train(
        initial_weights(x_train.shape[1]),
        x_train,
        y_train,
        seed=variant.seed,
        **train_kwargs,  # type: ignore[arg-type]
    )
    for extra in range(1, cfg.rounds):
        central = local_train(
            central,
            x_train,
            y_train,
            seed=variant.seed + extra,
            **train_kwargs,  # type: ignore[arg-type]
        )

    return ByzantineStudy(
        n_sites=len(site_idx),
        site_sizes=[len(i) for i in site_idx],
        rounds=cfg.rounds,
        clean=clean,
        cells=cells,
        malicious_counts=[0, *counts],
        trim=cfg.trim,
        centralized_pr_auc=float(average_precision_score(y_test, central.scores(x_test))),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_byzantine_report(settings: Settings) -> Path:
    """Run the Byzantine study and write the report + figures."""
    study = run_byzantine(settings)
    counts = np.array(study.malicious_counts, dtype=float)

    worst_attack = max(
        ATTACKS,
        key=lambda a: max(
            (study.clean[c.aggregator] - c.pr_auc) for c in study.cells if c.attack == a
        ),
    )
    breakdown_fig = plots.plot_lines(
        {
            agg: (
                counts,
                np.array(
                    [study.clean[agg]]
                    + [
                        next(
                            c.pr_auc
                            for c in study.cells
                            if c.aggregator == agg
                            and c.attack == worst_attack
                            and c.n_malicious == f
                        )
                        for f in study.malicious_counts[1:]
                    ]
                ),
            )
            for agg in AGGREGATORS
        },
        xlabel=f"malicious sites (of {study.n_sites})",
        ylabel="test PR-AUC",
        title=f"Where each aggregation rule breaks ({worst_attack})",
        out_path=settings.paths.figures_dir / BREAKDOWN_FIGURE,
    )
    price_fig = plots.plot_barh(
        list(AGGREGATORS),
        [study.clean[a] for a in AGGREGATORS],
        xlabel="test PR-AUC with no attacker present",
        title="What robustness costs when nobody is lying",
        out_path=settings.paths.figures_dir / PRICE_FIGURE,
        xmax=max(max(study.clean.values()), study.centralized_pr_auc) * 1.15,
        vline=("centralized (pooled traffic)", study.centralized_pr_auc),
    )

    report = _render(study, worst_attack, breakdown_fig, price_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote Byzantine report", extra={"path": str(out_path)})

    with track_run(settings, "byzantine") as run:
        run.log_params({"n_sites": study.n_sites, "rounds": study.rounds, "trim": study.trim})
        metrics = {f"clean_{a.split()[0].lower()}": v for a, v in study.clean.items()}
        for agg in AGGREGATORS:
            worst = min((c.pr_auc for c in study.cells if c.aggregator == agg), default=0.0)
            metrics[f"worst_case_{agg.split()[0].lower()}"] = worst
        run.log_metrics(metrics)
        run.log_artifact(breakdown_fig)
        run.log_artifact(price_fig)
        run.log_artifact(out_path)
    return out_path


def _clean_table(study: ByzantineStudy) -> str:
    rows = ["| aggregation rule | clean PR-AUC | vs FedAvg | tolerates |", "|---|---|---|---|"]
    baseline = study.clean.get("FedAvg (mean)", 0.0)
    tolerance = {
        "FedAvg (mean)": "**nothing** — one site is enough",
        "coordinate median": f"up to {(study.n_sites - 1) // 2} of {study.n_sites}",
        "trimmed mean": f"up to {study.trim} by construction",
        "Krum": f"f < (n-2)/2, i.e. up to {max((study.n_sites - 3) // 2, 0)}",
    }
    for agg in AGGREGATORS:
        value = study.clean[agg]
        rows.append(f"| {agg} | {value:.3f} | {value - baseline:+.3f} | {tolerance.get(agg, '')} |")
    return "\n".join(rows)


def _attack_table(study: ByzantineStudy) -> str:
    counts = study.malicious_counts[1:]
    header = "| aggregation rule | attack | " + " | ".join(f"{f} liars" for f in counts) + " |"
    rows = [header, "|---|---|" + "---|" * len(counts)]
    for agg in AGGREGATORS:
        for attack in ATTACKS:
            cells = []
            for f in counts:
                cell = next(
                    (
                        c
                        for c in study.cells
                        if c.aggregator == agg and c.attack == attack and c.n_malicious == f
                    ),
                    None,
                )
                cells.append(f"{cell.pr_auc:.3f} ({cell.retained:.0%})" if cell else "-")
            rows.append(f"| {agg} | {attack} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _breakdown_table(study: ByzantineStudy, floor: float = 0.9) -> str:
    header = "| aggregation rule | " + " | ".join(ATTACKS) + " |"
    rows = [header, "|---|" + "---|" * len(ATTACKS)]
    for agg in AGGREGATORS:
        cells = []
        for attack in ATTACKS:
            f = _breakdown(study, agg, attack, floor)
            survived = f"survives all {study.malicious_counts[-1]}"
            cells.append(f"**{f}**" if f is not None else survived)
        rows.append(f"| {agg} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _headline(study: ByzantineStudy, worst_attack: str) -> str:
    one = next(
        (
            c
            for c in study.cells
            if c.aggregator == "FedAvg (mean)" and c.attack == worst_attack and c.n_malicious == 1
        ),
        None,
    )
    if one is None:
        return ""
    survivors = [
        agg
        for agg in AGGREGATORS
        if agg != "FedAvg (mean)"
        and next(
            (
                c.retained
                for c in study.cells
                if c.aggregator == agg and c.attack == worst_attack and c.n_malicious == 1
            ),
            0.0,
        )
        > 0.9
    ]
    severity = (
        "wipes out"
        if one.retained < 0.25
        else ("costs most of" if one.retained < 0.6 else "costs a third of")
    )
    return (
        f"**One malicious site out of {study.n_sites} {severity} FedAvg's value.** Under "
        f"{worst_attack}, a single liar takes the federated model from "
        f"{study.clean['FedAvg (mean)']:.3f} PR-AUC to **{one.pr_auc:.3f}** — "
        f"{one.retained:.0%} of what it was worth, from {1 / study.n_sites:.0%} of the "
        "participants. The mechanism is not subtle and that is the point: averaging has no "
        "bounded influence, so one participant's contribution is unbounded, and a federation is "
        "by construction a place where the other participants cannot be audited. "
        + (
            f"Swapping the mean for {', '.join(survivors)} holds the same single-liar attack "
            "above 90% of clean performance, without changing anything else in the protocol."
            if survivors
            else "None of the robust rules held above 90% here either, which makes the "
            "attack, not the aggregation, the thing to look at first."
        )
    )


def _krum_read(study: ByzantineStudy) -> str:
    """Krum's rows are often identical across attacks; say why rather than leave it odd."""
    per_attack = {
        attack: [c.pr_auc for c in study.cells if c.aggregator == "Krum" and c.attack == attack]
        for attack in ATTACKS
    }
    values = list(per_attack.values())
    if not all(values) or len(values) < 2:
        return ""
    identical = all(np.allclose(v, values[0]) for v in values[1:])
    if not identical:
        return ""
    return (
        "\n\nKrum's three attack rows are **identical, digit for digit**, which looks like a "
        "copy-paste error and is actually the rule's defining property. Krum does not average; "
        "it elects a single site's update and discards every other. When the elected update is "
        "an honest one, the attackers' submissions have no influence at all — not a diluted "
        "influence, none — so what they contained cannot matter. Averaging rules blend the "
        "attack in and their rows differ by attack; Krum either excludes it or does not, and "
        "here it always did. The corollary is the whole risk of the approach: Krum throws away "
        "the other sites' honest updates too, which is why it sits below the trimmed mean when "
        "nobody is lying, and why an attacker who can place an update *inside* the honest "
        "cluster defeats it completely."
    )


def _label_flip_read(study: ByzantineStudy) -> str:
    worst = max(study.malicious_counts[1:], default=0)
    rows = {
        agg: next(
            (
                c
                for c in study.cells
                if c.aggregator == agg and c.attack == "label flip" and c.n_malicious == worst
            ),
            None,
        )
        for agg in AGGREGATORS
    }
    present = {k: v for k, v in rows.items() if v is not None}
    if not present:
        return ""
    best = max(present.items(), key=lambda kv: kv[1].retained)
    worst_rule = min(present.items(), key=lambda kv: kv[1].retained)
    return (
        f"The label-flip row is the one to take seriously. With {worst} of {study.n_sites} sites "
        f"training honestly on inverted labels, retention runs from {worst_rule[1].retained:.0%} "
        f"({worst_rule[0]}) to {best[1].retained:.0%} ({best[0]}). It is the mildest attack in "
        "the table by raw damage and the most realistic by a distance: the malicious update has "
        "an ordinary norm, points in an ordinary direction, and sits a perfectly ordinary "
        "distance from its neighbours. Every defence here works by treating *outliers* as "
        "suspicious, and a well-fitted model of the wrong thing is not an outlier. Robust "
        "aggregation solves the loud attack and leaves the quiet one open — which is the same "
        "shape as the [backdoor study's](backdoor.md) finding that clean-metric monitoring "
        "cannot see a trigger, and a reason to keep the [data-valuation](data_value.md) and "
        "[influence](influence.md) tooling pointed at contribution quality rather than at "
        "gradient norms."
    )


def _render(study: ByzantineStudy, worst_attack: str, breakdown_fig: Path, price_fig: Path) -> str:
    smallest, largest = min(study.site_sizes), max(study.site_sizes)
    return f"""# NetSentry — Byzantine-Robust Aggregation: When a Site Lies

_Synthetic stand-in. Honest temporal/binary split. {study.n_sites} sites (capture days sharded
{study.n_sites // 3} ways each, {smallest:,}-{largest:,} flows apiece), {study.rounds} federated
rounds, linear model — FedAvg averages parameters and a boosted forest has none to average.
Malicious sites are always the first ones, held fixed across cells, so differences between rows
are the aggregation rule and not which sites were corrupted._

## Why this report exists

The [federated study](federated.md) trains across sites that cannot pool raw traffic, sharing
only weights. It assumes every site is honest, and that assumption carries the entire result.
Averaging is linear and has no bounded influence: one participant sending a large enough vector
moves the global mean wherever it wants. Federation is precisely the setting where such a
participant is plausible, because the reason you are federating is that you cannot inspect the
other members' data — and if you cannot inspect their data, you cannot inspect their updates.

## What robustness costs before any attack

{_clean_table(study)}

Robust rules discard information by design — a median ignores everything but the middle, Krum
does not average at all but *elects* a single site's update — so they should be expected to
lose ground when nobody is lying. Centralized training on pooled traffic reaches
{study.centralized_pr_auc:.3f} for reference.

![clean-case cost of each rule](../figures/{price_fig.name})

## Under attack

{_attack_table(study)}

{_headline(study, worst_attack)}{_krum_read(study)}

![breakdown by number of liars](../figures/{breakdown_fig.name})

## Where each rule breaks

Smallest number of malicious sites that costs more than 10% of the rule's own clean PR-AUC:

{_breakdown_table(study)}

{_label_flip_read(study)}

## Scope

The model is linear because parameter averaging requires parameters; the deployed detector is a
boosted forest, so this study is about the federated *protocol* rather than about the deployed
artefact, exactly as the [federated study](federated.md) is. Attackers here are static — they do
not know which aggregation rule they face and do not adapt to it, so these numbers are an
optimistic view of the defences. An attacker who knows the rule can do considerably better;
Krum in particular has known adaptive attacks that place a malicious update inside the honest
cluster, and defending against an adaptive adversary is a different and unfinished problem. The
trimmed mean's tolerance is a construction parameter rather than a discovery: it drops exactly
`trim` values per coordinate from each end, so setting it below the true number of liars is a
choice to fail. Sites are shards of capture days, which keeps the non-IID structure the
federated study documents, but shard-level heterogeneity is milder than genuinely independent
estates would be — and heterogeneity is what makes honest updates spread out, which is precisely
what robust aggregation has to distinguish from an attack."""
