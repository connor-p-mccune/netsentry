"""Optimal transport: a distance with units, a plan, and the cost every evasion attack forgets.

Every drift instrument this project already ships returns a **scalar with no unit**. PSI sums
log ratios over bins nobody chose on operational grounds; the KS statistic is a supremum CDF
gap; [MMD](mmd.md) is a distance in a kernel space whose scale comes from a bandwidth
heuristic. All three answer *did the traffic move*. None answers *how far*, in a unit anyone
can act on, and none answers *where the mass went*.

Optimal transport answers both, because it is not a statistic -- it is the solution to a
shipping problem (Monge 1781; Kantorovich 1942). Given two samples and a cost for moving one
point to another, it returns the cheapest **coupling**: which source flows correspond to which
target flows, in what proportion. Its value is in the ground metric's own units, which on this
project's standardised feature space is a training standard deviation.

The plan is what makes this more than another two-sample test, because a coupling between
*attack* traffic and *benign* traffic is a mimicry recipe: every attack flow's cheapest benign
counterpart, and the displacement that would take it there. Three attacks fall out of the same
formalism as special cases -- move everything to the benign mean (the mimicry this repository's
own [evasion study](robustness.md) runs), move each flow to its nearest benign neighbour, or
move each flow to its transport partner -- and they can be raced at a matched perturbation
budget.

They are not equivalent, and the difference is the point of the module. A transport plan is
constrained to be a *coupling*: it must reproduce the target distribution exactly, not merely
land each point somewhere plausible. Greedy nearest-neighbour mimicry ignores that constraint,
sends many attacks onto the same benign flow, and produces traffic that is per-flow benign and
**distributionally wrong**. Centroid mimicry is worse: piling every flow onto the mean builds a
density spike where real traffic is diffuse.

So evasion has two costs -- being individually unremarkable, and being collectively
unremarkable -- and every attack in this repository until now paid only the first. Both are
measured here, per arm, at the same budget, with the second one graded by the same drift
monitors a defender already runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.monitoring.detectors import ks_feature_tests
from netsentry.monitoring.drift import population_stability_index
from netsentry.robustness.evasion import attack_scores_transformed, controllable_indices
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import TransportConfig
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

REPORT_NAME = "transport.md"
CONVERGENCE_FIGURE = "transport_convergence.png"
EVASION_FIGURE = "transport_evasion.png"
AGGREGATE_FIGURE = "transport_aggregate.png"

_EPS = 1e-300

OT_ARM = "the transport partner"
NEAREST_ARM = "the nearest benign flow"
CENTROID_ARM = "the benign centroid (the deployed attack)"
RANDOM_ARM = "a random benign flow"
LIMITED_ARM = "the transport partner, controllable features only"


# --------------------------------------------------------------------------------------
# Exact one-dimensional transport. Closed form, and the building block for everything else.
# --------------------------------------------------------------------------------------


def wasserstein_1d(x: np.ndarray, y: np.ndarray, *, p: float = 1.0) -> float:
    """Exact ``p``-Wasserstein distance between two one-dimensional samples.

    In one dimension optimal transport has a closed form: the optimal plan is the monotone
    one, so the cost is the ``L_p`` distance between the two quantile functions. Both are step
    functions with breakpoints at multiples of ``1/n`` and ``1/m``, so integrating over the
    union of those levels is exact for any pair of sample sizes rather than an approximation
    on a grid. That exactness is what lets it serve as the reference everything else is graded
    against.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.sort(np.asarray(y, dtype=float))
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    levels = np.union1d(
        np.arange(1, len(xs) + 1, dtype=float) / len(xs),
        np.arange(1, len(ys) + 1, dtype=float) / len(ys),
    )
    weights = np.diff(np.concatenate([[0.0], levels]))
    # The right-continuous inverse CDF: the quantile at level q is the ceil(q*n)-th order
    # statistic. The epsilon absorbs the float error in (i/n) * n != i.
    ix = np.clip(np.ceil(levels * len(xs) - 1e-9).astype(int) - 1, 0, len(xs) - 1)
    iy = np.clip(np.ceil(levels * len(ys) - 1e-9).astype(int) - 1, 0, len(ys) - 1)
    gaps = np.abs(xs[ix] - ys[iy])
    return float(np.sum(weights * gaps**p) ** (1.0 / p))


def quantile_transport_map(
    source: np.ndarray, target: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Push ``values`` through the monotone map carrying ``source`` onto ``target``.

    ``T(x) = F_target^-1(F_source(x))`` is *the* optimal map in one dimension, and applying it
    feature by feature is optimal for a cost that is separable across features. That
    separability is exactly the assumption a joint plan does not make, which is why both
    appear in the adaptation table below rather than only the cheap one.
    """
    ordered_source = np.sort(np.asarray(source, dtype=float))
    ordered_target = np.sort(np.asarray(target, dtype=float))
    if len(ordered_source) == 0 or len(ordered_target) == 0:
        return np.asarray(values, dtype=float)
    ranks = np.searchsorted(ordered_source, np.asarray(values, dtype=float), side="right")
    levels = np.clip((ranks - 0.5) / len(ordered_source), 0.0, 1.0)
    positions = levels * (len(ordered_target) - 1)
    mapped: np.ndarray = np.interp(positions, np.arange(len(ordered_target)), ordered_target)
    return mapped


# --------------------------------------------------------------------------------------
# Sliced transport: many one-dimensional problems instead of one large one.
# --------------------------------------------------------------------------------------


def random_directions(n_features: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed unit vectors, one per row."""
    directions: np.ndarray = rng.normal(size=(count, n_features))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit: np.ndarray = directions / np.maximum(norms, 1e-12)
    return unit


def sliced_wasserstein(
    x: np.ndarray, y: np.ndarray, *, directions: np.ndarray, p: float = 1.0
) -> float:
    """Average one-dimensional transport cost over random projections (Rabin et al. 2011).

    The full plan is cubic to solve exactly and quadratic in memory to approximate; projecting
    onto a line turns each problem into a sort. The average of the projected costs is itself a
    metric on distributions, so a monitor can keep the units without the matrix -- at the price
    of a weaker statistic, since a projection can hide a difference the joint plan would see.

    Equal sample sizes take a vectorised path: the monotone plan then matches order statistic
    to order statistic, so every projection is one column of a single sort. That is what makes
    a permutation test over hundreds of projections affordable inside a monitoring loop.
    """
    projected_x = np.asarray(x, dtype=float) @ directions.T
    projected_y = np.asarray(y, dtype=float) @ directions.T
    if len(projected_x) == len(projected_y):
        gaps = np.abs(np.sort(projected_x, axis=0) - np.sort(projected_y, axis=0))
        return float(np.mean(gaps**p) ** (1.0 / p))
    costs = [
        wasserstein_1d(projected_x[:, k], projected_y[:, k], p=p) ** p
        for k in range(directions.shape[0])
    ]
    return float(np.mean(costs) ** (1.0 / p))


@dataclass(frozen=True)
class TransportTest:
    """A sliced-Wasserstein two-sample test with an exact permutation null."""

    statistic: float
    p_value: float
    null_mean: float
    permutations: int
    seconds: float

    def rejects(self, alpha: float) -> bool:
        """Whether the test fires at level ``alpha``."""
        return self.p_value <= alpha


def sliced_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    projections: int,
    permutations: int,
    p: float = 1.0,
) -> TransportTest:
    """Test whether two samples share a law, with the statistic still in units.

    The projections are drawn **once** and reused for every permutation. That is not only an
    optimisation: re-drawing them per permutation would add projection noise to the null that
    the observed statistic never paid, which biases the test toward accepting.

    The p-value uses the ``(1 + #{null >= observed}) / (1 + B)`` correction, so it can never be
    reported as zero out of a finite permutation set.
    """
    start = time.perf_counter()
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    directions = random_directions(a.shape[1], projections, rng)
    observed = sliced_wasserstein(a, b, directions=directions, p=p)
    pooled = np.vstack([a, b])
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        order = rng.permutation(len(pooled))
        null[index] = sliced_wasserstein(
            pooled[order[: len(a)]], pooled[order[len(a) :]], directions=directions, p=p
        )
    p_value = float((1 + int(np.sum(null >= observed))) / (1 + permutations))
    return TransportTest(
        statistic=observed,
        p_value=p_value,
        null_mean=float(null.mean()) if permutations else 0.0,
        permutations=permutations,
        seconds=time.perf_counter() - start,
    )


# --------------------------------------------------------------------------------------
# The full coupling: the exact optimum, and the regularised approximation it grades.
# --------------------------------------------------------------------------------------


def squared_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean costs, expanded rather than materialised as a difference."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a2: np.ndarray = np.einsum("ij,ij->i", a, a)
    b2: np.ndarray = np.einsum("ij,ij->i", b, b)
    costs: np.ndarray = np.maximum(a2[:, None] + b2[None, :] - 2.0 * (a @ b.T), 0.0)
    return costs


def exact_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """The unregularised optimum for equal-sized uniform marginals, and its mean cost.

    With ``n = m`` and uniform weights, Birkhoff's theorem places an optimal vertex of the
    transport polytope at a permutation matrix, so the Hungarian algorithm solves the *exact*
    problem rather than an approximation of it. At the sizes used here that costs a fraction
    of a second, which is why the entropic solver appears in this module as the thing being
    graded rather than as the thing being trusted.
    """
    rows, cols = linear_sum_assignment(cost)
    return np.asarray(cols, dtype=int), float(cost[rows, cols].mean())


def _log_sum_exp(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Stable ``log sum exp`` along one axis, written out rather than imported.

    ``scipy.special.logsumexp`` is twice as slow on the matrices this solver iterates over,
    and the solver spends essentially all of its time here: two of these per iteration, a few
    hundred iterations, over an ``n x m`` matrix.
    """
    peak = matrix.max(axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    reduced: np.ndarray = peak + np.log(np.exp(matrix - peak).sum(axis=axis, keepdims=True))
    return np.squeeze(reduced, axis=axis)


@dataclass(frozen=True)
class SinkhornResult:
    """The regularised coupling, its transport cost, and how hard it was to reach."""

    plan: np.ndarray
    cost: float
    iterations: int
    marginal_error: float
    seconds: float


def sinkhorn(
    weights_source: np.ndarray,
    weights_target: np.ndarray,
    cost: np.ndarray,
    *,
    reg: float,
    max_iter: int = 300,
    tol: float = 1e-6,
    check_every: int = 10,
) -> SinkhornResult:
    """Entropic optimal transport by Sinkhorn-Knopp scaling, in the log domain.

    Adding an entropy term to the linear program makes it strictly convex, with a solution
    that is a diagonal rescaling of a Gibbs kernel and is reachable by alternately normalising
    rows and columns (Cuturi, NeurIPS 2013). Two implementation choices carry weight.

    It runs on **log-domain potentials** rather than multiplicative scalings. The naive form
    computes ``exp(-cost / reg)`` directly, which underflows to exactly zero for any cost more
    than a few hundred times the regularisation -- which, at the strengths that give an
    accurate answer, is most of the matrix. The solver then returns a confident result about
    whichever entries happened to survive.

    And the reported ``cost`` is the **transport cost** ``<plan, cost>``, not the regularised
    objective. The entropy term is an algorithmic device, not something anyone is shipping;
    quoting the objective would make the distance depend on a solver parameter.
    """
    start = time.perf_counter()
    log_a = np.log(np.maximum(np.asarray(weights_source, dtype=float), _EPS))
    log_b = np.log(np.maximum(np.asarray(weights_target, dtype=float), _EPS))
    potential_f = np.zeros(len(log_a))
    potential_g = np.zeros(len(log_b))
    iterations = 0
    error = float("inf")
    plan = np.zeros(cost.shape)
    for iterations in range(1, max_iter + 1):
        potential_f = reg * (log_a - _log_sum_exp((potential_g[None, :] - cost) / reg, axis=1))
        potential_g = reg * (log_b - _log_sum_exp((potential_f[:, None] - cost) / reg, axis=0))
        if iterations % check_every == 0 or iterations == max_iter:
            plan = np.exp((potential_f[:, None] + potential_g[None, :] - cost) / reg)
            error = float(np.abs(plan.sum(axis=1) - np.asarray(weights_source)).sum())
            if error <= tol:
                break
    return SinkhornResult(
        plan=plan,
        cost=float(np.sum(plan * cost)),
        iterations=iterations,
        marginal_error=error,
        seconds=time.perf_counter() - start,
    )


def barycentric_targets(plan: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Where the plan sends each source point: the conditional mean of its target mass.

    The entropic plan is dense, so a source flow is coupled to many target flows in different
    proportions. The barycentric projection collapses that to one point per source -- the
    weighted average of the targets it is coupled to -- which is what turns a coupling into a
    *map* that can be applied to a flow.
    """
    mass = plan.sum(axis=1, keepdims=True)
    return np.asarray((plan @ target) / np.maximum(mass, _EPS), dtype=float)


def step_toward(source: np.ndarray, target: np.ndarray, budget: float) -> np.ndarray:
    """Move each row toward its target, stopping at ``budget`` units of displacement.

    Capping the *distance* rather than interpolating by a fraction is what makes the arms
    comparable: the targets sit at different distances, so a fraction-based walk would price
    each attack differently and the comparison would be about the targets' geometry rather
    than about which target is worth aiming at.
    """
    direction = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    scale = np.minimum(1.0, budget / np.maximum(norms, 1e-12))
    return np.asarray(source + direction * scale, dtype=float)


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationRow:
    """One check against something known to be right."""

    check: str
    reference: float
    computed: float

    @property
    def gap(self) -> float:
        """Relative gap against the reference (absolute, when the reference is zero)."""
        if abs(self.reference) < 1e-12:
            return abs(self.computed)
        return abs(self.computed - self.reference) / abs(self.reference)


@dataclass(frozen=True)
class RegRow:
    """One regularisation strength: what it costs in accuracy, and what it does to the map."""

    scale: float
    reg: float
    cost: float
    iterations: int
    to_centroid: float
    to_partner: float

    def gap(self, exact: float) -> float:
        """Relative excess of the regularised cost over the exact optimum."""
        return (self.cost - exact) / max(exact, 1e-12)


@dataclass(frozen=True)
class FeatureShift:
    """One feature, judged by transport and by the two monitors already deployed."""

    feature: str
    wasserstein: float
    floor: float
    psi: float
    ks_statistic: float
    ks_significant: bool

    @property
    def excess(self) -> float:
        """Distance above the same-population floor -- the part that is not sampling noise."""
        return max(self.wasserstein - self.floor, 0.0)


@dataclass(frozen=True)
class ArmStep:
    """One targeting strategy at one perturbation budget."""

    arm: str
    budget: float
    detection: float
    aggregate: float


@dataclass(frozen=True)
class ArmSummary:
    """One targeting strategy, judged per-flow and in aggregate at the headline budget.

    ``is_coupling`` is the property the whole comparison turns on: a targeting that uses each
    benign flow exactly once reproduces the benign distribution when it is followed all the
    way, and one that does not cannot, whatever it does per flow.
    """

    arm: str
    plan_cost: float
    distinct_targets: int
    detection: float
    aggregate: float
    p_value: float
    max_psi: float
    is_coupling: bool

    def floor_multiple(self, floor: float) -> float:
        """How many same-population floors away from benign this arm's traffic ends up."""
        return self.aggregate / max(floor, 1e-12)


@dataclass(frozen=True)
class AdaptationRow:
    """One retraining arm, scored on the temporal test split."""

    arm: str
    rows: int
    pr_auc: float
    tpr_at_budget: float


@dataclass
class TransportStudy:
    """Everything the transport report needs, computed once."""

    validation: list[ValidationRow]
    reg_rows: list[RegRow]
    exact_cost: float
    shifts: list[FeatureShift]
    joint_test: TransportTest
    null_test: TransportTest
    attack_distance: float
    benign_floor: float
    nearest_cost: float
    nearest_distinct: int
    baseline_detection: float
    baseline_aggregate: float
    aggregate_floor: float
    steps: list[ArmStep] = field(default_factory=list)
    summaries: list[ArmSummary] = field(default_factory=list)
    adaptation: list[AdaptationRow] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)
    budgets: list[float] = field(default_factory=list)
    profile: str = ""
    threshold: float = 0.0
    budget: float = 0.01
    headline_budget: float = 0.0
    n_attack: int = 0
    n_features: int = 0
    n_controllable: int = 0
    seconds: float = 0.0

    def curve(self, arm: str, field_name: str) -> np.ndarray:
        """One arm's series over the budget sweep."""
        return np.array(
            [getattr(step, field_name) for step in self.steps if step.arm == arm], dtype=float
        )

    def summary(self, arm: str) -> ArmSummary | None:
        """Look up one arm's headline row."""
        return next((row for row in self.summaries if row.arm == arm), None)

    def arm_row(self, name: str) -> AdaptationRow | None:
        """Look up one adaptation arm by name."""
        return next((row for row in self.adaptation if row.arm == name), None)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _closed_form_checks(column: np.ndarray) -> list[ValidationRow]:
    """Two things the one-dimensional solver cannot be allowed to get wrong."""
    shift = 1.75
    return [
        ValidationRow(
            "1-D transport of a pure shift (closed form)",
            shift,
            wasserstein_1d(column, column + shift),
        ),
        ValidationRow(
            "1-D transport of a sample against itself (closed form)",
            0.0,
            wasserstein_1d(column, column),
        ),
    ]


def _regularisation_sweep(
    cfg: TransportConfig,
    cost: np.ndarray,
    benign: np.ndarray,
    partner: np.ndarray,
    centroid: np.ndarray,
) -> list[RegRow]:
    """What the entropic parameter buys and what it costs, against the exact optimum.

    The same sweep answers two questions at once. Against the exact assignment cost it says
    how much accuracy the regularisation gives away. Against the two reference targets --
    the benign centroid and the exact transport partner -- it shows the barycentric map
    sliding between them, which is the sense in which centroid mimicry and partner mimicry are
    the two ends of one dial rather than two unrelated attacks.
    """
    size = cost.shape[0]
    uniform = np.full(size, 1.0 / size)
    median_cost = float(np.median(cost))
    rows: list[RegRow] = []
    for scale in cfg.reg_scales:
        reg = scale * median_cost
        result = sinkhorn(uniform, uniform, cost, reg=reg, max_iter=cfg.max_iter, tol=cfg.tol)
        mapped = barycentric_targets(result.plan, benign)
        rows.append(
            RegRow(
                scale=scale,
                reg=reg,
                cost=result.cost,
                iterations=result.iterations,
                to_centroid=float(np.mean(np.linalg.norm(mapped - centroid, axis=1))),
                to_partner=float(np.mean(np.linalg.norm(mapped - partner, axis=1))),
            )
        )
    return rows


def _feature_shifts(
    reference: np.ndarray,
    deployment: np.ndarray,
    names: list[str],
    cfg: TransportConfig,
    rng: np.random.Generator,
) -> list[FeatureShift]:
    """Per-feature transport distance beside the PSI and KS verdicts on the same columns."""
    half = len(reference) // 2
    reference_frame = pd.DataFrame(reference, columns=names)
    deployment_frame = pd.DataFrame(deployment, columns=names)
    ks_by_feature = {
        result.feature: result
        for result in ks_feature_tests(reference_frame, deployment_frame, names)
    }
    order = rng.permutation(len(reference))
    shifts: list[FeatureShift] = []
    for index, name in enumerate(names):
        column_reference = reference[:, index]
        ks_result = ks_by_feature.get(name)
        shifts.append(
            FeatureShift(
                feature=name,
                wasserstein=wasserstein_1d(column_reference, deployment[:, index]),
                floor=wasserstein_1d(
                    column_reference[order[:half]], column_reference[order[half:]]
                ),
                psi=population_stability_index(
                    column_reference, deployment[:, index], bins=cfg.psi_bins
                ),
                ks_statistic=float(ks_result.statistic) if ks_result else 0.0,
                ks_significant=bool(ks_result.significant) if ks_result else False,
            )
        )
    shifts.sort(key=lambda shift: shift.excess, reverse=True)
    return shifts


def _max_psi(moved: np.ndarray, reference: np.ndarray, bins: int) -> float:
    """The deployed drift monitor's verdict on a mimicked population: its worst feature."""
    return max(
        (
            population_stability_index(reference[:, index], moved[:, index], bins=bins)
            for index in range(reference.shape[1])
        ),
        default=0.0,
    )


def _race_the_arms(
    bundle: ModelBundle,
    cfg: TransportConfig,
    attack: np.ndarray,
    targets: dict[str, np.ndarray],
    controllable: np.ndarray,
    witness: np.ndarray,
    directions: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> tuple[list[ArmStep], list[ArmSummary]]:
    """Score every targeting strategy at every budget, per flow and in aggregate.

    The aggregate is measured against a **held-out** benign sample, never against the one the
    targets were drawn from: at full displacement the transport arm reproduces its own target
    sample exactly, so grading it there would be measuring a tautology instead of a fingerprint.
    """
    steps: list[ArmStep] = []
    summaries: list[ArmSummary] = []
    for arm, target in targets.items():
        # The constrained arm spends its *whole* budget inside the subspace it owns, rather
        # than spending it across all the features and discarding what it cannot move. Any
        # other convention turns the comparison into one of distance travelled rather than one
        # of which directions the distance was spent in.
        aimed = target
        if arm == LIMITED_ARM and len(controllable):
            aimed = np.array(attack, dtype=float, copy=True)
            aimed[:, controllable] = target[:, controllable]
        for budget in cfg.budgets:
            moved = step_toward(attack, aimed, budget)
            steps.append(
                ArmStep(
                    arm=arm,
                    budget=budget,
                    detection=float(np.mean(attack_scores_transformed(bundle, moved) >= threshold)),
                    aggregate=sliced_wasserstein(moved, witness, directions=directions),
                )
            )
        headline = step_toward(attack, aimed, cfg.budgets[-1])
        test = sliced_permutation_test(
            headline,
            witness,
            rng=rng,
            projections=cfg.projections,
            permutations=cfg.permutations,
        )
        distinct = len({tuple(row) for row in np.round(aimed, 6)})
        displacement = aimed - attack
        summaries.append(
            ArmSummary(
                arm=arm,
                plan_cost=float(np.mean(np.sum(displacement**2, axis=1))),
                distinct_targets=distinct,
                detection=steps[-1].detection,
                aggregate=test.statistic,
                p_value=test.p_value,
                max_psi=_max_psi(headline, witness, cfg.psi_bins),
                is_coupling=distinct == len(attack) and arm != LIMITED_ARM,
            )
        )
    return steps, summaries


def _adaptation(
    settings: Settings,
    cfg: TransportConfig,
    reference: np.ndarray,
    deployment: np.ndarray,
    holdout: tuple[np.ndarray, np.ndarray],
    y_reference: np.ndarray,
    y_deployment: np.ndarray,
    rng: np.random.Generator,
) -> list[AdaptationRow]:
    """Retrain on transported training features and see whether the temporal gap moves.

    Both maps are estimated from *unlabelled* deployment traffic, which a real deployment has:
    the separable one is a monotone quantile map per feature applied to every training row,
    the joint one a barycentric Sinkhorn projection on a subsample. The subsample arm gets its
    own untransported control at the same row count, because otherwise the comparison measures
    how much data was thrown away.

    The operating point for every arm is chosen on the **validation** split, the way the
    deployed protocol chooses it, and only then applied to the later days. Reading the
    detection rate off a threshold fitted to the test scores would inflate every arm equally
    and put a number in this table that no other table in the repository could be compared to.
    """
    from netsentry.models.supervised import SupervisedClassifier

    calibration, y_calibration = holdout

    def score(name: str, features: np.ndarray, labels: np.ndarray) -> AdaptationRow:
        model = SupervisedClassifier(settings).fit(features, labels)
        classes = list(model.classes_)
        column = classes.index(1) if 1 in classes else len(classes) - 1
        held_out = np.asarray(model.predict_proba(calibration))[:, column]
        positive = np.asarray(model.predict_proba(deployment))[:, column]
        cut = threshold_at_fpr(y_calibration, held_out, cfg.budget)
        attacks = y_deployment == 1
        return AdaptationRow(
            arm=name,
            rows=len(features),
            pr_auc=float(average_precision_score(y_deployment, positive)),
            tpr_at_budget=float(np.mean(positive[attacks] >= cut)) if attacks.any() else 0.0,
        )

    rows = [score("no adaptation (the deployed protocol)", reference, y_reference)]
    mapped = np.column_stack(
        [
            quantile_transport_map(reference[:, index], deployment[:, index], reference[:, index])
            for index in range(reference.shape[1])
        ]
    )
    rows.append(score("separable OT map (every feature, every row)", mapped, y_reference))

    size = min(cfg.adapt_rows, len(reference), len(deployment))
    picked = rng.choice(len(reference), size, replace=False)
    arrived = deployment[rng.choice(len(deployment), size, replace=False)]
    rows.append(score("subsample control (no map)", reference[picked], y_reference[picked]))
    cost = squared_cost_matrix(reference[picked], arrived)
    plan = sinkhorn(
        np.full(size, 1.0 / size),
        np.full(size, 1.0 / size),
        cost,
        reg=cfg.reg_scales[-1] * float(np.median(cost)),
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    ).plan
    rows.append(
        score(
            "joint OT map (barycentric, subsample)",
            barycentric_targets(plan, arrived),
            y_reference[picked],
        )
    )
    return rows


def run_transport_study(settings: Settings) -> TransportStudy:
    """Validate the solver, measure drift in units, then race the mimicry strategies."""
    start = time.perf_counter()
    cfg: TransportConfig = settings.transport
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    result = fit_supervised(variant)
    bundle = result.bundle
    names = bundle.feature_names()
    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    reference: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    calibration: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    deployment: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_reference = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_calibration = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_deployment = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)

    shifts = _feature_shifts(reference, deployment, names, cfg, rng)
    window = min(cfg.max_rows, len(reference) // 2, len(deployment))
    reference_draw = rng.choice(len(reference), 2 * window, replace=False)
    joint_test = sliced_permutation_test(
        reference[reference_draw[:window]],
        deployment[rng.choice(len(deployment), window, replace=False)],
        rng=rng,
        projections=cfg.projections,
        permutations=cfg.permutations,
    )
    null_test = sliced_permutation_test(
        reference[reference_draw[:window]],
        reference[reference_draw[window:]],
        rng=rng,
        projections=cfg.projections,
        permutations=cfg.permutations,
    )

    # The attacker's problem: couple the attacks the deployed model actually meets to the
    # benign traffic they would have to blend into. A third benign sample is held back as the
    # witness a defender would compare the mimicked traffic against.
    attack_rows = np.flatnonzero(y_deployment == 1)
    benign_rows = np.flatnonzero(y_deployment == 0)
    size = min(cfg.max_rows, len(attack_rows), len(benign_rows) // 2)
    attack = deployment[rng.choice(attack_rows, size, replace=False)]
    benign_draw = rng.choice(benign_rows, 2 * size, replace=False)
    benign = deployment[benign_draw[:size]]
    witness = deployment[benign_draw[size:]]

    cost = squared_cost_matrix(attack, benign)
    partner_index, exact_cost = exact_assignment(cost)
    nearest_index = np.argmin(cost, axis=1)
    nearest_cost = float(cost[np.arange(size), nearest_index].mean())
    centroid = benign.mean(axis=0)
    floor_index, floor_cost = exact_assignment(squared_cost_matrix(benign, witness))

    reg_rows = _regularisation_sweep(cfg, cost, benign, benign[partner_index], centroid)
    validation = _closed_form_checks(reference[:, 0])
    best = min(reg_rows, key=lambda row: row.gap(exact_cost))
    validation.append(
        ValidationRow(
            f"Sinkhorn cost at reg = {best.scale:g} x median vs the exact assignment",
            exact_cost,
            best.cost,
        )
    )

    controllable = controllable_indices(names, settings.robustness.controllable_features)
    directions = random_directions(attack.shape[1], cfg.projections, rng)
    threshold = float(bundle.thresholds.get(cfg.profile, 0.5))
    targets = {
        OT_ARM: benign[partner_index],
        NEAREST_ARM: benign[nearest_index],
        CENTROID_ARM: np.tile(centroid, (size, 1)),
        RANDOM_ARM: benign[rng.permutation(size)],
        LIMITED_ARM: benign[partner_index],
    }
    steps, summaries = _race_the_arms(
        bundle, cfg, attack, targets, controllable, witness, directions, threshold, rng
    )

    study = TransportStudy(
        validation=validation,
        reg_rows=reg_rows,
        exact_cost=exact_cost,
        shifts=shifts,
        joint_test=joint_test,
        null_test=null_test,
        attack_distance=float(np.sqrt(max(exact_cost, 0.0))),
        benign_floor=float(np.sqrt(max(floor_cost, 0.0))),
        nearest_cost=nearest_cost,
        nearest_distinct=len(set(nearest_index.tolist())),
        baseline_detection=float(np.mean(attack_scores_transformed(bundle, attack) >= threshold)),
        baseline_aggregate=sliced_wasserstein(attack, witness, directions=directions),
        aggregate_floor=sliced_wasserstein(benign, witness, directions=directions),
        steps=steps,
        summaries=summaries,
        adaptation=_adaptation(
            variant,
            cfg,
            reference,
            deployment,
            (calibration, y_calibration),
            y_reference,
            y_deployment,
            rng,
        ),
        arms=list(targets),
        budgets=list(cfg.budgets),
        profile=cfg.profile,
        threshold=threshold,
        budget=cfg.budget,
        headline_budget=cfg.budgets[-1],
        n_attack=size,
        n_features=reference.shape[1],
        n_controllable=len(controllable),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Transport study complete",
        extra={
            "attack_distance": round(study.attack_distance, 3),
            "benign_floor": round(study.benign_floor, 3),
            "nearest_distinct": study.nearest_distinct,
            "seconds": round(study.seconds, 1),
        },
    )
    del floor_index
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _validation_table(study: TransportStudy) -> str:
    rows = "\n".join(
        f"| {row.check} | {row.reference:.4f} | {row.computed:.4f} | {row.gap:.2%} |"
        for row in study.validation
    )
    return "| check | reference | computed | gap |\n|---|---|---|---|\n" + rows


def _reg_table(study: TransportStudy) -> str:
    rows = "\n".join(
        f"| {row.scale:g} x median | {row.cost:.2f} | {row.gap(study.exact_cost):+.1%} | "
        f"{row.iterations} | {row.to_centroid:.2f} | {row.to_partner:.2f} |"
        for row in study.reg_rows
    )
    header = (
        "| regularisation | transport cost | excess over exact | iterations | "
        "distance to the centroid | distance to the exact partner |\n"
        "|---|---|---|---|---|---|\n"
    )
    exact = (
        f"| _none (exact assignment)_ | _{study.exact_cost:.2f}_ | _--_ | _--_ | "
        f"_--_ | _0.00_ |"
    )
    return header + rows + "\n" + exact


def _shift_table(study: TransportStudy, top_n: int) -> str:
    rows = "\n".join(
        f"| `{display_feature_name(shift.feature)}` | {shift.wasserstein:.3f} | "
        f"{shift.floor:.3f} | **{shift.excess:.3f}** | {shift.psi:.3f} | "
        f"{shift.ks_statistic:.3f} | {'yes' if shift.ks_significant else 'no'} |"
        for shift in study.shifts[:top_n]
    )
    return (
        "| feature | W1 (sd) | same-window floor | excess | PSI | KS | KS fires |\n"
        "|---|---|---|---|---|---|---|\n" + rows
    )


def _detection_table(study: TransportStudy) -> str:
    header = "| target | " + " | ".join(f"{b:g} sd" for b in study.budgets) + " |\n"
    header += "|---" * (len(study.budgets) + 1) + "|\n"
    rows = "\n".join(
        f"| {arm} | " + " | ".join(f"{value:.1%}" for value in study.curve(arm, "detection")) + " |"
        for arm in study.arms
    )
    return header + rows


def _aggregate_table(study: TransportStudy) -> str:
    header = "| target | " + " | ".join(f"{b:g} sd" for b in study.budgets) + " |\n"
    header += "|---" * (len(study.budgets) + 1) + "|\n"
    rows = "\n".join(
        f"| {arm} | " + " | ".join(f"{value:.3f}" for value in study.curve(arm, "aggregate")) + " |"
        for arm in study.arms
    )
    floor = (
        "| _benign against benign (the floor)_ | "
        + " | ".join(f"_{study.aggregate_floor:.3f}_" for _ in study.budgets)
        + " |"
    )
    return header + rows + "\n" + floor


def _summary_table(study: TransportStudy) -> str:
    rows = "\n".join(
        f"| {row.arm} | {'**yes**' if row.is_coupling else 'no'} | {row.plan_cost:.1f} | "
        f"{row.distinct_targets:,} | {row.detection:.1%} | {row.aggregate:.3f} | "
        f"{row.floor_multiple(study.aggregate_floor):.1f}x | {row.max_psi:.2f} |"
        for row in study.summaries
    )
    return (
        "| target | a coupling? | plan cost (sd^2) | distinct targets used | detection | "
        "distance from benign | vs the floor | worst-feature PSI |\n"
        "|---|---|---|---|---|---|---|---|\n" + rows
    )


def _adaptation_table(study: TransportStudy) -> str:
    rows = "\n".join(
        f"| {row.arm} | {row.rows:,} | {row.pr_auc:.3f} | {row.tpr_at_budget:.1%} |"
        for row in study.adaptation
    )
    return (
        f"| arm | training rows | PR-AUC | detection @ {study.budget:.0%} FPR |\n"
        "|---|---|---|---|\n" + rows
    )


def _lead(study: TransportStudy) -> str:
    transport = study.summary(OT_ARM)
    nearest = study.summary(NEAREST_ARM)
    centroid = study.summary(CENTROID_ARM)
    if transport is None or nearest is None or centroid is None:
        return "_The arms did not complete._"
    best = min(study.summaries, key=lambda row: row.detection)
    quietest = min(study.summaries, key=lambda row: row.aggregate)
    waste = centroid.detection / max(transport.detection, 1e-9) - 1.0
    return (
        f"**The mimicry attack this repository already ships aims at the worst target "
        f"available, and it is the only one a defender can see coming.** At a "
        f"{study.headline_budget:g}-sigma perturbation budget, moving every attack flow toward "
        f"its optimal-transport partner takes detection from {study.baseline_detection:.1%} to "
        f"{transport.detection:.1%}; each flow's nearest benign neighbour reaches "
        f"{nearest.detection:.1%}, and the benign centroid -- the mimicry the "
        f"[evasion study](robustness.md) runs -- leaves {centroid.detection:.1%}, "
        f"{waste:.0%} more surviving detection than the transport plan for the identical "
        f"budget. The best arm per flow is _{best.arm}_; the quietest in aggregate is "
        f"_{quietest.arm}_.\n\n"
        f"The aggregate is where the arms genuinely separate. Judged as a *population* against "
        f"held-out benign traffic -- the comparison a drift monitor makes -- the untouched "
        f"attacks sit {study.baseline_aggregate:.3f} sd away and benign traffic sits "
        f"{study.aggregate_floor:.3f} from itself. Centroid mimicry ends at "
        f"**{centroid.aggregate:.3f}**, further from benign than the attack traffic was before "
        f"anyone tried to hide it: collapsing every flow onto the mean builds a density spike "
        f"where real traffic is diffuse, and it shows up at a worst-feature PSI of "
        f"{centroid.max_psi:.1f} in the monitor this project already runs. Nearest-neighbour "
        f"mimicry stalls at {nearest.aggregate:.3f} because it reuses "
        f"{study.nearest_distinct:,} benign flows for {study.n_attack:,} attacks and leaves the "
        f"rest of the distribution empty. Only the transport plan reaches "
        f"{transport.aggregate:.3f}, and it is the only arm that can, because being a "
        f"*coupling* is exactly the requirement that the disguised traffic still has the benign "
        f"distribution.\n\n"
        f"Evasion therefore has two costs -- being individually unremarkable, and being "
        f"collectively unremarkable -- and every attack in this repository until now paid only "
        f"the first."
    )


def _render(study: TransportStudy, convergence: Path, evasion: Path, aggregate: Path) -> str:
    transport = study.summary(OT_ARM)
    limited = study.summary(LIMITED_ARM)
    centroid = study.summary(CENTROID_ARM)
    top_shift = study.shifts[0] if study.shifts else None
    fired = sum(1 for shift in study.shifts if shift.ks_significant)
    baseline_arm = study.adaptation[0] if study.adaptation else None
    best_arm = max(study.adaptation, key=lambda row: row.pr_auc) if study.adaptation else None
    worst_read = (
        f"`{display_feature_name(top_shift.feature)}` has moved {top_shift.excess:.3f} sd"
        if top_shift
        else "The worst feature moves"
    )
    worst_psi = f"{top_shift.psi:.3f}" if top_shift else "n/a"
    centroid_psi = f"{centroid.max_psi:.2f}" if centroid else "n/a"
    limited_read = f"{limited.detection:.1%}" if limited else "n/a"
    transport_read = f"{transport.detection:.1%}" if transport else "n/a"
    mapped_arms = [row for row in study.adaptation if "OT map" in row.arm]
    best_mapped = max(mapped_arms, key=lambda row: row.pr_auc) if mapped_arms else None
    if baseline_arm and best_mapped and best_arm:
        verdict = "beats" if best_mapped.pr_auc > baseline_arm.pr_auc else "does not reach"
        adaptation_read = (
            f"The best transported arm (_{best_mapped.arm}_, PR-AUC {best_mapped.pr_auc:.3f}) "
            f"{verdict} the unadapted {baseline_arm.pr_auc:.3f}, and the winner over all four "
            f"arms is _{best_arm.arm}_"
        )
    else:
        adaptation_read = "No arm completed"
    return f"""# NetSentry — Optimal Transport: the Distance, the Plan, and the Aggregate

_Exact optimal transport (Hungarian assignment) and its entropic approximation (Cuturi 2013)
on the temporal split's standardised feature space. {study.n_attack:,} attack flows coupled to
{study.n_attack:,} benign ones, graded against a third, held-out benign sample. Distances are
in training standard deviations. Regenerate with `netsentry transport`._

## Why this report exists

Every drift instrument this project ships returns a scalar with no unit. PSI sums log ratios
over bins nobody picked operationally; KS reports a supremum CDF gap; [MMD](mmd.md) measures a
distance in a kernel space whose scale is a bandwidth heuristic. Each answers *did the traffic
move*. None answers *how far*, in a unit an engineer can act on, and none answers *where the
mass went*.

Optimal transport answers both, because it is not a statistic -- it is the solution to a
shipping problem. And the plan is what makes it more than another two-sample test: a coupling
between attack traffic and benign traffic is a **mimicry recipe**.

{_lead(study)}

## Does this implementation compute optimal transport?

{_validation_table(study)}

At the sizes used here the *exact* problem is solvable: with equal sample sizes and uniform
weights, Birkhoff's theorem puts an optimal vertex of the transport polytope at a permutation,
so the Hungarian algorithm returns the true optimum in a fraction of a second. The entropic
solver therefore appears in this module as the thing being **graded**, not the thing being
trusted.

![Regularisation against the exact optimum](../figures/{convergence.name})

{_reg_table(study)}

Two readings. Down the third column, entropic regularisation is a bias -- it smears the plan
and undercounts the cost -- and the bias shrinks as the parameter does, paid for in iterations.
Down the last two columns, the same parameter is a **dial between two known attacks**: at heavy
regularisation the barycentric map sits {study.reg_rows[0].to_centroid:.2f} sd from the benign
centroid, which is to say it *is* centroid mimicry; as the regularisation falls the map walks
away from the mean and toward the exact partner. Centroid mimicry and partner mimicry are two
ends of one dial, not two unrelated ideas, and nothing in the literature had to be taken on
faith to see it -- both endpoints are computed here.

One implementation detail is load-bearing: the scaling runs on **log-domain potentials**. The
textbook multiplicative form evaluates `exp(-cost / reg)` directly, and at the strengths that
give an accurate cost most of that matrix underflows to exactly zero, after which the solver
returns a confident answer about whichever entries survived.

## Drift, in units nobody has to calibrate

{_shift_table(study, 10)}

The `excess` column is the distance above a **floor**: the same transport distance computed
between two halves of the *training* window, which would be zero with infinite data and is not.
Reading the raw column instead credits every feature with its own sampling noise.

The three columns disagree, and the disagreement is the point.
{worst_read} -- a sentence an operator can act on -- while PSI puts the same feature at
{worst_psi}, a number whose only meaning is the folklore
banding (0.1 moderate, 0.25 major) that no property of this data supports. KS fires on {fired}
of {len(study.shifts)} features under BH control at 5%, which across 25,000 flows is a statement
about the sample size as much as about the traffic.

The joint picture agrees and keeps the unit. The sliced-Wasserstein test averages the exact
one-dimensional cost over random projections, with the projections drawn **once** and reused
across every permutation -- re-drawing them per permutation would add projection noise to the
null that the observed statistic never paid, biasing the test toward accepting. It puts the
training window **{study.joint_test.statistic:.3f} sd** from the deployment window (p =
{study.joint_test.p_value:.3f} over {study.joint_test.permutations} permutations), against
{study.null_test.statistic:.3f} sd at p = {study.null_test.p_value:.3f} for two windows drawn
from training alone.

## The distance an attacker has to travel

The exact coupling puts the attack flows **{study.attack_distance:.2f} standard deviations**
from the benign traffic they would have to blend into. The same computation between two
disjoint *benign* samples returns {study.benign_floor:.2f}, so most of that figure is the
curse of dimensionality rather than the attack: in {study.n_features} dimensions the empirical
Wasserstein distance converges as `n^(-1/d)` and no affordable sample is unbiased. A transport
distance quoted without that floor is mostly a statement about the sample size, which is why
the floor is in the sentence.

Greedy nearest-neighbour matching reaches a mean cost of {study.nearest_cost:.1f} against the
optimal assignment's {study.exact_cost:.1f} -- *cheaper*, and not a contradiction, because it
is not a transport plan at all. It sends {study.n_attack:,} attacks onto
{study.nearest_distinct:,} distinct benign flows and leaves the rest of the benign distribution
unoccupied. The constraint the assignment obeys and the greedy matching does not is precisely
the requirement that the disguised traffic still *has* the benign distribution.

## Racing the mimicry strategies at a matched budget

![Detection under each targeting strategy](../figures/{evasion.name})

{_detection_table(study)}

Every arm moves each attack flow toward a target and stops at the same displacement budget, so
the comparison is about *which target is worth aiming at* rather than about how far the targets
happen to be. Detection starts at {study.baseline_detection:.1%} on untouched flows, which is
the temporal split's honest operating point for attack families the model never trained on.

The four unconstrained arms are a two-by-two, and that is the reason there are four of them.
Two of them are **couplings** -- each benign flow is used exactly once, so following them all
the way reproduces the benign distribution -- and two are not. Two of them are **optimal** --
the total cost is the least any assignment could achieve -- and two are not.

| | a coupling | not a coupling |
|---|---|---|
| **optimal** | the transport partner | the nearest benign flow |
| **not optimal** | a random benign flow | the benign centroid |

Reading across the top row isolates the value of the *constraint*; reading down the first
column isolates the value of *optimality*. The random coupling is the instructive cell: it
satisfies the distributional constraint exactly as the transport plan does, and loses anyway,
because at a fixed budget a longer plan travels a smaller share of the way. Optimality is not
an aesthetic preference here -- it is how much of the disguise fits inside the budget.

## The cost the per-flow attacks never paid

![Distance from benign traffic, as a population](../figures/{aggregate.name})

{_aggregate_table(study)}

This is the same experiment judged as a *population*, against a held-out benign sample the
targets were never drawn from -- which matters, because at full displacement the transport arm
reproduces its own target sample exactly and grading it there would measure a tautology.

{_summary_table(study)}

Every arm rejects the same-distribution null at the permutation test's resolution floor, which
is the right result to report and the wrong number to steer by: on a {study.n_attack:,}-flow
window, everything is significantly not-benign. **The distance is the operator's question, not
the p-value**, which is why the table carries the multiple of the same-population floor
instead.

The centroid arm goes the wrong way. Its worst-feature PSI reaches {centroid_psi} against
the folklore "major shift" line of 0.25, so **the deployed drift monitor catches this attack
without being told it exists** -- and catches it more easily than it would have caught the
undisguised attack. That is a defensive finding hiding inside an attack study: a mimicry
adversary who ignores the aggregate hands the defender a signal in the one instrument the
defender already runs.

The last row is the realistic attacker, and it is the row that did not go as expected.
Restricting the displacement to the {study.n_controllable} features of {study.n_features} an
attacker can manipulate without breaking the attack -- and spending the whole budget inside
that subspace rather than across all of them -- leaves detection at {limited_read} against the
unconstrained {transport_read}. **The constrained attacker does better, not worse.** Every unit
of budget spent moving a field the exporter derives rather than the attacker sets is a unit the
model was not going to react to; the constraint concentrates the perturbation on the features
that carry the verdict.

What the constraint does cost is the second thing. {study.n_controllable} coordinates cannot
carry the attack distribution onto the benign one however they are spent, so the constrained
arm is not a coupling and its aggregate stalls at
{f"{limited.floor_multiple(study.aggregate_floor):.1f}x" if limited else "n/a"} the floor while
the unconstrained plan reaches
{f"{transport.floor_multiple(study.aggregate_floor):.1f}x" if transport else "n/a"}. **The two
costs of evasion come apart under a realistic threat model, and only the per-flow one is for
sale.** An attacker who can only pad and delay can become individually unremarkable and cannot
become collectively unremarkable, which is an argument for spending defensive effort on the
population rather than on the flow.

## Can transport close the temporal gap?

{_adaptation_table(study)}

The [covariate-shift study](covariate_shift.md) diagnosed the temporal gap as *concept* rather
than covariate shift and priced importance weighting -- the textbook fix -- at a loss. This is
the same question asked with the other textbook fix, and both maps here are estimated from
**unlabelled** deployment traffic, which a real deployment has.

{adaptation_read}. Transport moves the training features onto the deployment marginals as
advertised and the detector does not improve, because `p(y|x)` is what changed: the later days
carry attack families the earlier days never contained, and no map of `p(x)` can invent a
label relationship that was never in the training data. That is the third instrument to reach
the same verdict, which is worth more than the first one was.

## Scope and honest limits

- **The empirical Wasserstein distance is badly biased in high dimension.** Every distance here
  is reported against a same-population floor for that reason, and the floor is large:
  {study.benign_floor:.2f} sd of the attack distance's {study.attack_distance:.2f}. The
  *comparisons* between arms are unaffected -- they share a sample size and a witness -- but
  the absolute numbers are not distances between distributions, they are distances between
  samples.
- **The cost function is a modelling choice, not a fact.** Squared Euclidean distance on
  standardised features says every feature costs the same to change, which is false: an
  attacker pads packet lengths for free and cannot move a protocol flag at all. The
  controllable/uncontrollable split is a two-level approximation of a cost that is really
  per-feature, and a better one needs a threat model nobody has measured.
- **A displaced flow is not necessarily a valid flow.** A convex combination of two feature
  vectors can hold a fractional packet count or an inconsistent duration/rate triple. The
  unconstrained curve is an upper bound on what a feasible attacker achieves at the same
  budget, which is why the constrained arm is the operational number.
- **The attacker is assumed to know the benign distribution.** Sampling benign traffic is the
  cheapest thing on this list -- it is what a network gives away for free -- but the plan also
  needs the *feature space*, and that is the model-extraction problem the
  [extraction study](extraction.md) prices.
- **The aggregate test is offline.** A defender running it needs a window of the attacker's
  traffic, and an attacker who paces the campaign under the window keeps the fingerprint below
  the noise. What is measured here is that the fingerprint exists and that one attack strategy
  avoids it, not that a monitor would necessarily see it in time.
- **The adaptation map is transductive.** It is estimated from the deployment sample it is
  applied against; an online version has to re-estimate as traffic arrives, which is the
  [threshold-refresh](refresh.md) problem with more moving parts."""


def run_transport_report(settings: Settings) -> Path:
    """Run the transport study and write the report + figures."""
    study = run_transport_study(settings)
    scales = np.array([row.scale for row in study.reg_rows], dtype=float)
    convergence = plots.plot_lines(
        {
            "entropic transport cost": (scales, np.array([row.cost for row in study.reg_rows])),
            "exact assignment (Hungarian)": (scales, np.full(len(scales), study.exact_cost)),
        },
        xlabel="entropic regularisation (multiple of the median cost)",
        ylabel="transport cost (sd^2)",
        title="Regularisation is a bias bought with tractability",
        out_path=settings.paths.figures_dir / CONVERGENCE_FIGURE,
        xscale="log",
    )

    budgets = np.array(study.budgets, dtype=float)
    evasion = plots.plot_lines(
        {arm: (budgets, study.curve(arm, "detection")) for arm in study.arms},
        xlabel="perturbation budget (sd)",
        ylabel=f"detection at the {study.profile} operating point",
        title="Which benign flow is worth aiming at",
        out_path=settings.paths.figures_dir / EVASION_FIGURE,
    )

    series = {arm: (budgets, study.curve(arm, "aggregate")) for arm in study.arms}
    series["benign against benign (the floor)"] = (
        budgets,
        np.full(len(budgets), study.aggregate_floor),
    )
    aggregate = plots.plot_lines(
        series,
        xlabel="perturbation budget (sd)",
        ylabel="sliced-Wasserstein distance from benign traffic (sd)",
        title="Per-flow invisibility is not distributional invisibility",
        out_path=settings.paths.figures_dir / AGGREGATE_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, convergence, evasion, aggregate), encoding="utf-8")
    logger.info("Wrote transport report", extra={"path": str(out_path)})

    with track_run(settings, "transport") as run:
        run.log_params(
            {"attacks": study.n_attack, "profile": study.profile, "budget": study.headline_budget}
        )
        run.log_metrics(
            {
                "attack_distance": study.attack_distance,
                "benign_floor": study.benign_floor,
                "joint_statistic": study.joint_test.statistic,
            }
            | {f"detection_{index}": row.detection for index, row in enumerate(study.summaries)}
        )
        for figure in (convergence, evasion, aggregate, out_path):
            run.log_artifact(figure)
    return out_path
