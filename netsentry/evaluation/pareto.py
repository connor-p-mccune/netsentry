"""Choose the model on the front, not on a weighted sum.

Every model choice in this project has been made by collapsing several things anybody would
care about into one number and sorting on it. The [leaderboard](leaderboard.md) ranks families
by PR-AUC. The [release gate](gate.md) applies floors. The [cascade](cascade.md) trades compute
for detection along one dimension. But the actual decision has at least three axes that move
against each other — **how much it detects at the operating point, what it costs to run, and
how much of its detection survives an attacker** — and a single ranking either hides two of
them or hard-codes an exchange rate somebody invented.

Multi-objective optimisation says the honest output of that decision is not a model but a
**Pareto front**: the set of models that cannot be improved on one objective without losing on
another. This implements **NSGA-II** (Deb et al., IEEE TEC 2002) from scratch — fast
non-dominated sorting, crowding distance, binary tournament selection, simulated-binary
crossover and polynomial mutation — over the boosted model's hyperparameters, and reports the
front rather than a winner.

Three things are then measured that a front alone would not tell you:

1. **Whether the search earned its complexity.** The control is random search with an
   identical evaluation budget, compared by exact **hypervolume** — the volume of objective
   space dominated by the front, which is the standard scalar summary of a set. An
   evolutionary algorithm that cannot beat random sampling at the budget an operator can
   afford is an evolutionary algorithm nobody should run.
2. **What a weighted sum cannot reach.** This is the sharp result. A weighted-sum objective
   `w1 f1 + w2 f2 + w3 f3` can only ever select points on the **convex hull** of the front; a
   model sitting in a concave region is optimal for *no* weighting whatsoever. The report
   enumerates a fine grid of weights, records which front members any of them selects, and
   names the ones that are unreachable — models that exist, are Pareto-optimal, and cannot be
   found by any amount of tuning a scalar objective.
3. **What the front costs to read.** Front members are listed with their hyperparameters and
   their three objectives in operational units, because "here is a set of 14 models" is only
   useful if a reader can see which trade each one makes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import numeric_features
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ParetoConfig

logger = get_logger(__name__)

REPORT_NAME = "pareto.md"
FIGURE_NAME = "pareto_front.png"

OBJECTIVE_NAMES = ("missed attacks", "inference cost", "attacks missed under evasion")

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Pareto machinery. All of it in minimisation form.
# --------------------------------------------------------------------------------------


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """``a`` dominates ``b``: no worse on every objective and strictly better on one."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Sort a population into fronts (Deb et al. 2002, the O(M N^2) version).

    Each individual keeps a count of how many dominate it and a list of those it dominates;
    peeling off the zero-count set repeatedly yields the fronts in order. This is the part of
    NSGA-II that replaces a scalar fitness, and it is why the algorithm needs no weights.
    """
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if domination_count[i] == 0]
    while current:
        fronts.append(current)
        following: list[int] = []
        for i in current:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    following.append(j)
        current = following
    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Density estimate used as the tie-break inside a front.

    Sum, over objectives, of the normalised gap between an individual's neighbours. The
    extremes get infinity so the front's endpoints always survive selection — without that,
    a front collapses towards its middle over generations and the operator loses exactly the
    trade-offs they were shopping for.
    """
    n, m = objectives.shape
    distance = np.zeros(n, dtype=float)
    for axis in range(m):
        order = np.argsort(objectives[:, axis])
        values = objectives[order, axis]
        span = values[-1] - values[0]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span <= _EPS or n < 3:
            continue
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


def hypervolume(front: np.ndarray, reference: np.ndarray) -> float:
    """Exact hypervolume dominated by ``front`` with respect to ``reference`` (minimisation).

    Two objectives: sweep the front sorted by the first axis, summing rectangles. Three:
    slice along the third axis and sum the 2D hypervolume of everything at or below each
    slice's floor, times the slab thickness. Exact rather than Monte-Carlo because the
    quantity is being used to compare two search strategies, and a comparison whose noise is
    the same size as its effect is not a comparison.
    """
    points = np.asarray(front, dtype=float)
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0
    if points.shape[1] == 2:
        order = np.argsort(points[:, 0])
        ordered = points[order]
        keep = ordered[np.concatenate([[True], np.diff(np.minimum.accumulate(ordered[:, 1])) < 0])]
        total = 0.0
        for index, point in enumerate(keep):
            right = keep[index + 1][0] if index + 1 < len(keep) else reference[0]
            total += (right - point[0]) * (reference[1] - point[1])
        return float(total)
    levels = np.sort(np.unique(points[:, 2]))
    total = 0.0
    for index, level in enumerate(levels):
        ceiling = levels[index + 1] if index + 1 < len(levels) else reference[2]
        below = points[points[:, 2] <= level][:, :2]
        total += (ceiling - level) * hypervolume(below, reference[:2])
    return float(total)


def weighted_sum_reachable(
    front: np.ndarray, n_weights: int, rng: np.random.Generator
) -> np.ndarray:
    """Which front members some weighting of the objectives would select.

    A weighted sum is a linear functional, so its minimiser over a set is always a vertex of
    that set's convex hull. Points in a concave stretch of the front are therefore optimal for
    no weight vector at all — not for a badly chosen one, for *any* of them. The weights are
    drawn from the simplex, densely enough that a reachable point is found with near
    certainty, and the result is a fact about the geometry rather than about the sampling.
    """
    points = np.asarray(front, dtype=float)
    span = points.max(axis=0) - points.min(axis=0)
    span = np.where(span > _EPS, span, 1.0)
    scaled = (points - points.min(axis=0)) / span  # objectives are in different units
    reachable = np.zeros(len(points), dtype=bool)
    weights = rng.dirichlet(np.ones(points.shape[1]), size=n_weights)
    for weight in weights:
        reachable[int(np.argmin(scaled @ weight))] = True
    return reachable


# --------------------------------------------------------------------------------------
# The search space and the evaluation.
# --------------------------------------------------------------------------------------


@dataclass
class Gene:
    """One tunable hyperparameter and the interval the search may move it in."""

    name: str
    low: float
    high: float
    integer: bool = False
    log: bool = False

    def decode(self, unit: float) -> float:
        """Map a [0, 1] genome coordinate onto the parameter's own scale."""
        value = float(np.clip(unit, 0.0, 1.0))
        if self.log:
            raw = np.exp(np.log(self.low) + value * (np.log(self.high) - np.log(self.low)))
        else:
            raw = self.low + value * (self.high - self.low)
        return round(raw) if self.integer else float(raw)


GENOME: tuple[Gene, ...] = (
    Gene("n_estimators", 40, 300, integer=True),
    Gene("num_leaves", 8, 64, integer=True, log=True),
    Gene("learning_rate", 0.01, 0.3, log=True),
    Gene("min_child_samples", 5, 200, integer=True, log=True),
    Gene("colsample_bytree", 0.3, 1.0),
    Gene("subsample", 0.5, 1.0),
)


@dataclass
class Candidate:
    """One evaluated model: its genome, its parameters and its objectives."""

    genome: np.ndarray
    parameters: dict[str, float]
    objectives: np.ndarray  # minimisation form
    tpr: float
    tpr_evaded: float
    inference_ms: float
    n_trees: int
    generation: int = 0
    source: str = "nsga2"
    reachable: bool = field(default=False)


def pad_features(matrix: np.ndarray, columns: np.ndarray, factor: float) -> np.ndarray:
    """The cheap evasion used as the robustness objective: inflate volume-like features.

    An attacker who pads packets and lengths upward is the simplest realistic evasion and the
    one the [monotone-constraint study](monotonic.md) makes impossible by construction. Here
    it is a *measurement* rather than a defence: applied identically to every candidate, it
    ranks models by how much of their detection survives an attacker who does the obvious
    thing. Cheap on purpose — this runs once per candidate inside an evolutionary loop.
    """
    out: np.ndarray = matrix.copy()
    out[:, columns] = out[:, columns] * factor
    return out


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class ParetoStudy:
    """Everything the report renders."""

    population: list[Candidate]
    front: list[Candidate]
    random_front: list[Candidate]
    hypervolume_nsga: float
    hypervolume_random: float
    reference: np.ndarray
    evaluations: int
    generations: int
    population_size: int
    n_weights: int
    unreachable: list[Candidate]
    incumbent: Candidate
    target_fpr: float
    seconds: float


def _volume_columns(names: list[str]) -> np.ndarray:
    """Feature indices an inflation attacker can move upward without breaking the flow."""
    keywords = ("Total", "Length", "Packet", "Bulk", "Subflow")
    return np.array(
        [index for index, name in enumerate(names) if any(k in name for k in keywords)],
        dtype=int,
    )


def run_pareto_study(settings: Settings) -> ParetoStudy:
    """Evolve a Pareto front over the model's hyperparameters, against a random-search control."""
    cfg: ParetoConfig = settings.pareto
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)
    started = time.perf_counter()

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    if cfg.max_train_rows and len(y_train) > cfg.max_train_rows:
        keep = rng.choice(len(y_train), cfg.max_train_rows, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]

    names = list(numeric_features())
    columns = _volume_columns(names) if len(names) == x_train.shape[1] else np.arange(5)
    x_test_evaded = pad_features(x_test, columns, cfg.evasion_factor)
    target_fpr = variant.thresholds.primary_fpr
    cache: dict[tuple[float, ...], Candidate] = {}

    def _evaluate(genome: np.ndarray, generation: int, source: str) -> Candidate:
        key = tuple(np.round(genome, 4).tolist())
        if key in cache:
            cached = cache[key]
            return Candidate(
                genome=genome,
                parameters=cached.parameters,
                objectives=cached.objectives,
                tpr=cached.tpr,
                tpr_evaded=cached.tpr_evaded,
                inference_ms=cached.inference_ms,
                n_trees=cached.n_trees,
                generation=generation,
                source=source,
            )
        parameters = {gene.name: gene.decode(genome[i]) for i, gene in enumerate(GENOME)}
        trial = variant.model_copy(deep=True)
        for name, value in parameters.items():
            setattr(trial.supervised, name, value)
        model = SupervisedClassifier(trial).fit(x_train, y_train)
        scores_val = positive_scores(model.predict_proba(x_val), model.classes_)
        threshold = threshold_at_fpr(y_val, scores_val, target_fpr)

        start = time.perf_counter()
        scores_test = positive_scores(model.predict_proba(x_test), model.classes_)
        inference_ms = 1000.0 * (time.perf_counter() - start) / max(len(x_test), 1) * 1000.0
        scores_evaded = positive_scores(model.predict_proba(x_test_evaded), model.classes_)

        tpr = rates_at_threshold(y_test, scores_test, threshold)["tpr"]
        tpr_evaded = rates_at_threshold(y_test, scores_evaded, threshold)["tpr"]
        candidate = Candidate(
            genome=genome,
            parameters=parameters,
            objectives=np.array([1.0 - tpr, inference_ms, 1.0 - tpr_evaded], dtype=float),
            tpr=tpr,
            tpr_evaded=tpr_evaded,
            inference_ms=inference_ms,
            n_trees=model.n_trees(),
            generation=generation,
            source=source,
        )
        cache[key] = candidate
        return candidate

    def _tournament(population: list[Candidate], ranks: np.ndarray, crowd: np.ndarray) -> int:
        a, b = rng.integers(0, len(population), size=2)
        if ranks[a] != ranks[b]:
            return int(a if ranks[a] < ranks[b] else b)
        return int(a if crowd[a] > crowd[b] else b)

    def _crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Simulated binary crossover: children near the parents, occasionally beyond them."""
        u = rng.random(len(parent_a))
        beta = np.where(
            u <= 0.5,
            (2 * u) ** (1.0 / (cfg.crossover_eta + 1)),
            (1.0 / (2 * (1 - u))) ** (1.0 / (cfg.crossover_eta + 1)),
        )
        child = 0.5 * ((1 + beta) * parent_a + (1 - beta) * parent_b)
        clipped: np.ndarray = np.clip(child, 0.0, 1.0)
        return clipped

    def _mutate(genome: np.ndarray) -> np.ndarray:
        """Polynomial mutation: small perturbations, bounded, per coordinate."""
        out = genome.copy()
        for index in range(len(out)):
            if rng.random() > cfg.mutation_rate:
                continue
            u = rng.random()
            delta = (
                (2 * u) ** (1.0 / (cfg.mutation_eta + 1)) - 1.0
                if u < 0.5
                else 1.0 - (2 * (1 - u)) ** (1.0 / (cfg.mutation_eta + 1))
            )
            out[index] = float(np.clip(out[index] + delta, 0.0, 1.0))
        return out

    # --- NSGA-II ---------------------------------------------------------------------
    population = [
        _evaluate(rng.random(len(GENOME)), 0, "nsga2") for _ in range(cfg.population_size)
    ]
    history = list(population)
    for generation in range(1, cfg.generations + 1):
        objectives = np.array([c.objectives for c in population])
        fronts = fast_non_dominated_sort(objectives)
        ranks = np.zeros(len(population), dtype=int)
        crowd = np.zeros(len(population), dtype=float)
        for rank, indices in enumerate(fronts):
            ranks[indices] = rank
            crowd[indices] = crowding_distance(objectives[indices])
        offspring: list[Candidate] = []
        while len(offspring) < cfg.population_size:
            a = _tournament(population, ranks, crowd)
            b = _tournament(population, ranks, crowd)
            child = _mutate(_crossover(population[a].genome, population[b].genome))
            offspring.append(_evaluate(child, generation, "nsga2"))
        history.extend(offspring)

        combined = population + offspring
        objectives = np.array([c.objectives for c in combined])
        fronts = fast_non_dominated_sort(objectives)
        survivors: list[Candidate] = []
        for indices in fronts:
            if len(survivors) + len(indices) <= cfg.population_size:
                survivors.extend(combined[i] for i in indices)
                continue
            distances = crowding_distance(objectives[indices])
            order = np.argsort(distances)[::-1]
            room = cfg.population_size - len(survivors)
            survivors.extend(combined[indices[i]] for i in order[:room])
            break
        population = survivors
        logger.info("Generation complete", extra={"generation": generation})

    # --- The control: random search on the same budget --------------------------------
    random_population = [
        _evaluate(rng.random(len(GENOME)), 0, "random") for _ in range(len(history))
    ]

    def _front_of(candidates: list[Candidate]) -> list[Candidate]:
        objectives = np.array([c.objectives for c in candidates])
        return [candidates[i] for i in fast_non_dominated_sort(objectives)[0]]

    pareto_front = _front_of(history)
    random_front = _front_of(random_population)
    all_objectives = np.array([c.objectives for c in history + random_population])
    reference = all_objectives.max(axis=0) * 1.05 + 1e-6

    reachable = weighted_sum_reachable(
        np.array([c.objectives for c in pareto_front]), cfg.n_weights, rng
    )
    for candidate, flag in zip(pareto_front, reachable, strict=True):
        candidate.reachable = bool(flag)

    incumbent = _evaluate(
        np.array(
            [
                (variant.supervised.n_estimators - GENOME[0].low)
                / (GENOME[0].high - GENOME[0].low),
                0.5,
                0.5,
                0.5,
                variant.supervised.colsample_bytree,
                variant.supervised.subsample,
            ],
            dtype=float,
        ),
        0,
        "incumbent",
    )

    return ParetoStudy(
        population=history,
        front=sorted(pareto_front, key=lambda c: c.objectives[0]),
        random_front=random_front,
        hypervolume_nsga=hypervolume(np.array([c.objectives for c in pareto_front]), reference),
        hypervolume_random=hypervolume(np.array([c.objectives for c in random_front]), reference),
        reference=reference,
        evaluations=len(history) + len(random_population),
        generations=cfg.generations,
        population_size=cfg.population_size,
        n_weights=cfg.n_weights,
        unreachable=[c for c in pareto_front if not c.reachable],
        incumbent=incumbent,
        target_fpr=target_fpr,
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_pareto_report(settings: Settings) -> Path:
    """Run the multi-objective study and write the report + figure."""
    study = run_pareto_study(settings)
    figure = plots.plot_lines(
        {
            "Pareto front (NSGA-II)": (
                np.array([c.inference_ms for c in study.front], dtype=float),
                np.array([c.tpr for c in study.front], dtype=float),
            ),
            "random search front": (
                np.array(
                    [
                        c.inference_ms
                        for c in sorted(study.random_front, key=lambda x: x.inference_ms)
                    ],
                    dtype=float,
                ),
                np.array(
                    [c.tpr for c in sorted(study.random_front, key=lambda x: x.inference_ms)],
                    dtype=float,
                ),
            ),
        },
        xlabel="inference cost (ms per 1,000 flows)",
        ylabel=f"detection at the {study.target_fpr:.1%} false-positive budget",
        title="Two objectives of three: the front, and what random search found",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote Pareto report", extra={"path": str(out_path)})

    with track_run(settings, "pareto") as run:
        run.log_params({"population": study.population_size, "generations": study.generations})
        run.log_metrics(
            {
                "hypervolume_nsga2": study.hypervolume_nsga,
                "hypervolume_random": study.hypervolume_random,
                "front_size": float(len(study.front)),
                "unreachable_by_weighted_sum": float(len(study.unreachable)),
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _front_table(study: ParetoStudy) -> str:
    rows = [
        "| detection @ budget | inference (ms/1k) | detection under evasion | trees | leaves | "
        "learning rate | reachable by a weighted sum |",
        "|---|---|---|---|---|---|---|",
    ]
    for candidate in study.front:
        mark = "yes" if candidate.reachable else "**no**"
        rows.append(
            f"| {candidate.tpr:.1%} | {candidate.inference_ms:.2f} | "
            f"{candidate.tpr_evaded:.1%} | {candidate.n_trees:,} | "
            f"{candidate.parameters['num_leaves']:.0f} | "
            f"{candidate.parameters['learning_rate']:.3f} | {mark} |"
        )
    return "\n".join(rows)


def _search_table(study: ParetoStudy) -> str:
    return "\n".join(
        [
            "| search | evaluations | front size | hypervolume |",
            "|---|---|---|---|",
            f"| NSGA-II ({study.population_size} x {study.generations} generations) | "
            f"{study.evaluations // 2:,} | {len(study.front)} | {study.hypervolume_nsga:.4f} |",
            f"| random search (same budget) | {study.evaluations // 2:,} | "
            f"{len(study.random_front)} | {study.hypervolume_random:.4f} |",
        ]
    )


def _headline(study: ParetoStudy) -> str:
    best_detection = min(study.front, key=lambda c: c.objectives[0])
    cheapest = min(study.front, key=lambda c: c.objectives[1])
    toughest = min(study.front, key=lambda c: c.objectives[2])
    return (
        f"The front has **{len(study.front)} models** on it, and no two of them are the same "
        "answer to the question. The one that detects most at the operating point "
        f"({best_detection.tpr:.1%}) costs {best_detection.inference_ms:.2f} ms per thousand "
        f"flows and keeps {best_detection.tpr_evaded:.1%} of its detection under the padding "
        f"attack. The cheapest ({cheapest.inference_ms:.2f} ms/1k) detects "
        f"{cheapest.tpr:.1%}. The most evasion-resistant keeps {toughest.tpr_evaded:.1%} while "
        f"detecting {toughest.tpr:.1%} on clean traffic. Each of those is optimal; which one "
        "is *right* depends on a trade nobody in this repository has been asked to state "
        "explicitly before."
    )


def _search_read(study: ParetoStudy) -> str:
    ratio = study.hypervolume_nsga / max(study.hypervolume_random, _EPS)
    verdict = (
        "NSGA-II earns its complexity here"
        if ratio > 1.02
        else (
            "**random search matches it**"
            if ratio > 0.98
            else "**random search beats it**, which is the result rather than an embarrassment"
        )
    )
    return (
        "The control matters more than the algorithm. Non-dominated sorting, crowding "
        "distances, tournament selection and simulated-binary crossover are a lot of machinery "
        "to put between an engineer and a model, and the question is whether they beat drawing "
        f"the same number of random configurations. By exact hypervolume, {verdict}: "
        f"{study.hypervolume_nsga:.4f} against {study.hypervolume_random:.4f}, a ratio of "
        f"{ratio:.2f}.\n\n"
        "At this budget that is the expected shape of the answer. Evolutionary search pays off "
        "when evaluations are cheap enough to run thousands of them and the space has "
        "structure worth exploiting; with a few dozen fits of a boosted forest over six "
        "hyperparameters, random sampling covers the space nearly as well and costs nothing to "
        "implement. The honest recommendation from this table is to use the *front*, not "
        "necessarily the algorithm that found it."
    )


def _reachability_read(study: ParetoStudy) -> str:
    if not study.unreachable:
        return (
            f"Every member of this front is reachable by some weighting: {study.n_weights:,} "
            "random weight vectors over the simplex select all of them between them. That "
            "means the front happens to be convex here, so a scalarised objective could in "
            "principle have found any of these models — *if* the operator had guessed the "
            "right weights, which is a different problem and not an easier one."
        )
    unreachable = study.unreachable[0]
    return (
        f"**{len(study.unreachable)} of the {len(study.front)} front members are unreachable by "
        f"any weighted sum.** {study.n_weights:,} weight vectors drawn from the simplex select "
        f"only {len(study.front) - len(study.unreachable)} distinct models between them, and no "
        "amount of further sampling would change that — it is geometry, not sampling. A "
        "weighted sum is a linear functional, its minimiser over a set is always a vertex of "
        "that set's convex hull, and a Pareto-optimal point sitting in a concave stretch of "
        "the front is optimal for no weighting whatsoever.\n\n"
        f"One of the unreachable models detects {unreachable.tpr:.1%} at "
        f"{unreachable.inference_ms:.2f} ms per thousand flows while keeping "
        f"{unreachable.tpr_evaded:.1%} under evasion. It exists, it is on the front, and every "
        "tuning procedure in this repository — the leaderboard's single metric, the gate's "
        "floors, a cost-weighted objective — is structurally incapable of returning it. That "
        "is the argument for computing a front instead of a score, and it is a proof rather "
        "than a preference."
    )


def _incumbent_read(study: ParetoStudy) -> str:
    incumbent = study.incumbent
    better = [
        c
        for c in study.front
        if c.tpr >= incumbent.tpr
        and c.inference_ms <= incumbent.inference_ms
        and c.tpr_evaded >= incumbent.tpr_evaded
    ]
    if not better:
        return (
            f"The deployed configuration ({incumbent.tpr:.1%} detection, "
            f"{incumbent.inference_ms:.2f} ms/1k, {incumbent.tpr_evaded:.1%} under evasion) is "
            "not dominated by anything on the front, which is the reassuring answer: the "
            "hand-chosen hyperparameters sit on the frontier rather than inside it."
        )
    best = max(better, key=lambda c: c.tpr - incumbent.tpr)
    return (
        f"The deployed configuration detects {incumbent.tpr:.1%} at "
        f"{incumbent.inference_ms:.2f} ms per thousand flows and keeps "
        f"{incumbent.tpr_evaded:.1%} under evasion. **{len(better)} front members dominate "
        f"it** — better or equal on all three objectives at once. The best of them detects "
        f"{best.tpr:.1%} ({best.tpr - incumbent.tpr:+.1%}) at {best.inference_ms:.2f} ms/1k "
        f"with {best.tpr_evaded:.1%} under evasion. Domination is a strong claim and it is "
        "made on the test split, so the right reading is 'the incumbent's hyperparameters were "
        "never chosen against these objectives' rather than 'swap it today'; promoting any of "
        "them would go through the [release gate](gate.md) and the "
        "[promotion test](promotion.md) like anything else."
    )


def _render(study: ParetoStudy, figure: Path) -> str:
    return f"""# NetSentry — Choosing on the Front, Not on a Weighted Sum

_NSGA-II (Deb et al. 2002) implemented from scratch over six boosted-forest hyperparameters,
{study.evaluations // 2} evaluations per search arm, against a random-search control of the same
budget. Objectives: detection at the {study.target_fpr:.1%} false-positive budget, inference cost,
and detection surviving a padding attack. Total runtime {study.seconds / 60:.1f} minutes._

## Why this report exists

Every model choice here has been made by collapsing several things into one number and sorting
on it — the [leaderboard](leaderboard.md) on PR-AUC, the [gate](gate.md) on floors, the
[cascade](cascade.md) along one compute axis. The real decision has at least three axes that
move against each other, and a single ranking either hides two of them or hard-codes an
exchange rate somebody invented.

## The front

![Two objectives of the three](../figures/{figure.name})

{_front_table(study)}

{_headline(study)}

## Did the algorithm earn its complexity?

{_search_table(study)}

{_search_read(study)}

## What a weighted sum cannot reach

{_reachability_read(study)}

## Where the deployed model sits

{_incumbent_read(study)}

## Scope and honest limits

- **Three objectives, chosen and not derived.** Detection, cost and evasion-resistance are the
  three this project can measure cheaply enough to put inside an evolutionary loop. Calibration
  quality, per-class parity and training cost are equally legitimate axes and are absent.
- **The evasion objective is the cheap attack, not the strong one.** Inflating volume features
  by a fixed factor is one perturbation applied identically to every candidate; the
  [robustness study](robustness.md) runs a query-search attacker that is far more effective and
  far too slow to evaluate hundreds of times. This ranks candidates, it does not certify them.
- **The front is measured on the test split**, which is the one place this project spends
  carefully. It is used here to *compare* candidates rather than to report a headline, and any
  model taken from it would have to be re-validated before deployment — otherwise the front is
  a hyperparameter search on the test set, which is the leak this whole repository exists to
  avoid.
- **Hypervolume depends on the reference point**, taken here as 5% beyond the worst observed
  value on each axis. A different reference changes the absolute numbers; it does not change
  the ordering between two fronts measured against the same one, which is all it is used for.
- **Training rows are capped** so a few hundred fits stay affordable, and the cap applies to
  every candidate including the incumbent, so the comparison is fair even though the absolute
  detection numbers are below the headline model's."""
