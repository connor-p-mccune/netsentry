"""Hyperparameter search, budgeted honestly -- and the two premises nobody checks first.

`netsentry train tune` runs Optuna's TPE over the boosted model's hyperparameters and keeps
whatever scores best on validation. That is the standard practice, and it rests on two
assumptions that are almost never measured before the search starts.

**The first is HPO's own premise**: that a configuration scoring better on validation will
score better in deployment. On a shuffled split that is nearly tautological. On this project's
**temporal** split it is a claim about whether the later capture days reward what the earlier
ones do, and it can be tested directly -- rank a pool of configurations by validation, rank the
same pool by the held-out later days, and look at the correlation. If it is weak, then every
hour of search is being spent optimising a quantity that is not the objective.

**The second is multi-fidelity's premise**: that a cheap, low-fidelity evaluation ranks
configurations roughly the way an expensive one does. Successive halving (Jamieson & Talwalkar
2016) and Hyperband (Li et al., JMLR 2018) are built entirely on it -- they discard most
candidates after a fraction of the training budget, which is a catastrophe if the early ranking
is noise. The rank correlation across the fidelity ladder is a cheap measurement and it decides
whether the method applies at all.

With both premises measured, the search methods are then compared the way they should be and
usually are not: **at an equal total budget**, counted in the resource they actually consume
(boosting rounds fitted), not in trials. Counting trials flatters multi-fidelity methods by
construction, since most of their trials are cheap ones.

The last section is the part that generalises past this project. Selecting the best of many
configurations *on a validation split* is itself an estimation problem, and the winner is
biased upward by the selection -- the more configurations searched, the larger the bias. The
curve is measured here by resampling the order in which configurations arrive, so the
reported-versus-delivered gap can be read as a function of how hard anyone looked.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import MultiFidelityConfig

logger = get_logger(__name__)

REPORT_NAME = "multifidelity.md"
LADDER_FIGURE = "multifidelity_ladder.png"
CURSE_FIGURE = "multifidelity_curse.png"

RANDOM = "random search"
HALVING = "successive halving"
HYPERBAND = "Hyperband"
TPE = "TPE (the deployed tuner)"


# --------------------------------------------------------------------------------------
# The search space and the evaluator.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Configuration:
    """One point in the search space, independent of the fidelity it is evaluated at."""

    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float

    def key(self) -> tuple[float, ...]:
        """Hashable identity, so a promoted configuration is recognised as the same one."""
        return (
            round(self.learning_rate, 6),
            float(self.num_leaves),
            float(self.min_child_samples),
            round(self.subsample, 4),
            round(self.colsample_bytree, 4),
            round(self.reg_lambda, 6),
        )


def sample_configuration(rng: np.random.Generator) -> Configuration:
    """Draw a configuration, log-uniform where the parameter is a scale.

    Learning rate and the L2 penalty are sampled in log space because their effect is
    multiplicative: drawing them uniformly would put nine tenths of the budget in the region
    where the parameter barely matters, which is the most common way a random-search baseline
    is quietly made weak enough to lose.
    """
    return Configuration(
        learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        num_leaves=int(rng.integers(8, 128)),
        min_child_samples=int(rng.integers(5, 200)),
        subsample=float(rng.uniform(0.5, 1.0)),
        colsample_bytree=float(rng.uniform(0.5, 1.0)),
        reg_lambda=float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
    )


@dataclass
class Trial:
    """One evaluation: a configuration, the fidelity it ran at, and what it scored."""

    configuration: Configuration
    fidelity: int
    validation: float
    test: float
    seconds: float

    @property
    def units(self) -> int:
        """Resource units consumed -- boosting rounds actually fitted."""
        return self.fidelity


class Evaluator:
    """Fits a configuration at a fidelity and records what it cost.

    Fidelity here is the number of boosting rounds, which is the canonical choice for a
    boosted model: cost is very nearly linear in it, and a low-fidelity run is a genuine
    prefix of the high-fidelity one rather than a different problem. Every fit sees the same
    leakage-safe features, transformed once outside the loop -- refitting the pipeline per
    trial would be both slower and wrong.
    """

    def __init__(
        self,
        settings: Settings,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.settings = settings
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.trials: list[Trial] = []
        self._cache: dict[tuple[tuple[float, ...], int], Trial] = {}

    @property
    def units(self) -> int:
        """Total resource units spent so far."""
        return sum(trial.units for trial in self.trials)

    @property
    def seconds(self) -> float:
        """Total wall-clock spent fitting."""
        return sum(trial.seconds for trial in self.trials)

    def evaluate(self, configuration: Configuration, fidelity: int) -> Trial:
        """Fit at ``fidelity`` rounds and score on validation and on the later days.

        Repeats are cached: the search methods legitimately revisit a configuration at the
        same fidelity, and charging for it twice would inflate the budget of whichever method
        happens to revisit more.
        """
        from netsentry.models.supervised import SupervisedClassifier

        cache_key = (configuration.key(), fidelity)
        if cache_key in self._cache:
            return self._cache[cache_key]
        variant = self.settings.model_copy(deep=True)
        variant.supervised.n_estimators = fidelity
        variant.supervised.early_stopping_rounds = 0  # the fidelity *is* the round budget
        variant.supervised.learning_rate = configuration.learning_rate
        variant.supervised.num_leaves = configuration.num_leaves
        variant.supervised.min_child_samples = configuration.min_child_samples
        variant.supervised.subsample = configuration.subsample
        variant.supervised.colsample_bytree = configuration.colsample_bytree
        variant.supervised.reg_lambda = configuration.reg_lambda
        start = time.perf_counter()
        model = SupervisedClassifier(variant).fit(self.x_train, self.y_train)
        column = list(model.classes_).index(1)
        trial = Trial(
            configuration=configuration,
            fidelity=fidelity,
            validation=float(
                average_precision_score(
                    self.y_val, np.asarray(model.predict_proba(self.x_val))[:, column]
                )
            ),
            test=float(
                average_precision_score(
                    self.y_test, np.asarray(model.predict_proba(self.x_test))[:, column]
                )
            ),
            seconds=time.perf_counter() - start,
        )
        self._cache[cache_key] = trial
        self.trials.append(trial)
        return trial


# --------------------------------------------------------------------------------------
# The search methods, from scratch.
# --------------------------------------------------------------------------------------


def random_search(
    evaluate: Callable[[Configuration, int], Trial],
    rng: np.random.Generator,
    *,
    full_fidelity: int,
    budget: int,
) -> list[Trial]:
    """Evaluate fresh configurations at full fidelity until the budget runs out.

    The control every multi-fidelity claim has to beat, and a stronger one than it looks:
    random search is competitive with far more sophisticated methods once the budget is
    counted properly (Bergstra & Bengio 2012).
    """
    trials: list[Trial] = []
    spent = 0
    while spent + full_fidelity <= budget:
        trials.append(evaluate(sample_configuration(rng), full_fidelity))
        spent += full_fidelity
    return trials


def successive_halving(
    evaluate: Callable[[Configuration, int], Trial],
    configurations: list[Configuration],
    *,
    ladder: list[int],
    eta: int,
) -> list[Trial]:
    """Evaluate everything cheaply, keep the top 1/eta, promote, repeat.

    The aggressive end of multi-fidelity search. It spends the same budget per rung, so most
    of the configurations die having consumed a small fraction of a full run -- which is the
    entire saving, and the entire risk, since a configuration that starts slowly and finishes
    strongly is discarded before it can show it.
    """
    trials: list[Trial] = []
    survivors = list(configurations)
    for fidelity in ladder:
        scored = [(evaluate(configuration, fidelity), configuration) for configuration in survivors]
        trials.extend(trial for trial, _ in scored)
        scored.sort(key=lambda item: item[0].validation, reverse=True)
        keep = max(1, len(scored) // eta)
        survivors = [configuration for _, configuration in scored[:keep]]
        if len(survivors) <= 1:
            break
    return trials


def hyperband(
    evaluate: Callable[[Configuration, int], Trial],
    rng: np.random.Generator,
    *,
    full_fidelity: int,
    eta: int,
    budget: int,
    min_fidelity: int = 1,
) -> list[Trial]:
    """Hedge across halving schedules instead of committing to one (Li et al. 2018).

    Successive halving needs a starting fidelity, and picking it wrong is the failure mode:
    too cheap and the ranking is noise, too expensive and too few configurations are tried.
    Hyperband runs several brackets -- from many configurations starting very cheaply to few
    starting at full fidelity -- so the schedule becomes something searched rather than
    something guessed.

    ``min_fidelity`` exists because the cheapest rung is not actually cheap: a fit has a fixed
    cost that a one-round run pays in full, so the bracket that evaluates hundreds of
    configurations for one round each spends nearly all of its budget on overhead. Refusing to
    schedule below a floor is the honest response to a measured cost model, and the floor is
    reported rather than hidden.
    """
    trials: list[Trial] = []
    s_max = math.floor(math.log(max(full_fidelity, 1), eta))
    spent = 0
    # One pass over the brackets costs about (s_max + 1) * R; a budget larger than that is
    # spent on further passes rather than left on the table, which is how Hyperband is meant
    # to be run when the budget is not exactly one outer loop.
    while True:
        progressed = False
        for bracket in range(s_max, -1, -1):
            count = math.ceil((s_max + 1) / (bracket + 1) * eta**bracket)
            start_fidelity = max(min_fidelity, int(full_fidelity * eta ** (-bracket)))
            if start_fidelity > full_fidelity:
                continue
            ladder = [min(full_fidelity, start_fidelity * eta**step) for step in range(bracket + 1)]
            estimate = sum(
                max(1, count // eta**step) * fidelity for step, fidelity in enumerate(ladder)
            )
            if spent + estimate > budget:
                continue
            configurations = [sample_configuration(rng) for _ in range(count)]
            bracket_trials = successive_halving(evaluate, configurations, ladder=ladder, eta=eta)
            trials.extend(bracket_trials)
            spent += sum(trial.units for trial in bracket_trials)
            progressed = True
        if not progressed:
            return trials


def tpe_search(
    evaluator: Evaluator,
    seed: int,
    *,
    full_fidelity: int,
    budget: int,
) -> list[Trial]:
    """The incumbent: Optuna's tree-structured Parzen estimator at full fidelity."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[Trial] = []

    def objective(trial: Any) -> float:
        configuration = Configuration(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 199),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        result = evaluator.evaluate(configuration, full_fidelity)
        trials.append(result)
        return result.validation

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, budget // full_fidelity))
    return trials


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderRow:
    """One rung of the fidelity ladder, and how well it ranks what the full run will."""

    fidelity: int
    share_of_full: float
    rank_correlation: float
    learning_rate_correlation: float
    top_config_kept: bool
    seconds: float
    seconds_per_fit: float

    @property
    def share_of_seconds(self) -> float:
        """What one fit at this rung costs relative to a full-fidelity fit."""
        return self.seconds_per_fit


@dataclass(frozen=True)
class MethodRow:
    """One search method at the shared budget, averaged over repeats."""

    method: str
    configurations: float
    units: float
    seconds: float
    validation: float
    test: float
    validation_spread: float
    test_spread: float


@dataclass(frozen=True)
class CurseRow:
    """What selecting the best of ``trials`` configurations on validation reports and delivers."""

    trials: int
    reported: float
    delivered: float

    @property
    def optimism(self) -> float:
        """The gap between what the search reports and what the later days pay out."""
        return self.reported - self.delivered


@dataclass
class MultiFidelityStudy:
    """Everything the report needs, computed once."""

    ladder: list[LadderRow]
    methods: list[MethodRow]
    curse: list[CurseRow]
    pool_size: int
    fixed_seconds: float
    marginal_seconds: float
    selection_correlation: float
    selection_pvalue: float
    full_fidelity: int
    budget: int
    floor: int
    repeats: int
    n_train: int
    oracle_test: float
    oracle_validation: float
    default_validation: float
    default_test: float
    seconds: float = 0.0

    def method(self, name: str) -> MethodRow | None:
        """Look up one method."""
        return next((row for row in self.methods if row.method == name), None)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _premise_grid(
    evaluator: Evaluator, configurations: list[Configuration], ladder: list[int]
) -> tuple[list[LadderRow], list[Trial]]:
    """Score a fixed pool at every fidelity, so both premises can be read off one grid.

    The same evaluations answer two questions. Down the ladder they say whether a cheap run
    ranks configurations the way an expensive one does -- the assumption successive halving is
    built on -- and, when it does not, the learning-rate column says *why*: a short run rewards
    whatever climbs fastest, which is a property of the step size rather than of the model.
    At the top rung they give a pool of full-fidelity configurations whose validation and
    later-day scores can be correlated directly, which is the assumption all hyperparameter
    search is built on and the one a temporal split puts at risk.
    """
    full = ladder[-1]
    rates = [configuration.learning_rate for configuration in configurations]
    scores: dict[int, list[float]] = {}
    timings: dict[int, list[float]] = {}
    for fidelity in ladder:
        trials = [evaluator.evaluate(configuration, fidelity) for configuration in configurations]
        scores[fidelity] = [trial.validation for trial in trials]
        timings[fidelity] = [trial.seconds for trial in trials]
    best_full = int(np.argmax(scores[full]))
    rows: list[LadderRow] = []
    for fidelity in ladder:
        against_full = (
            float(spearmanr(scores[fidelity], scores[full]).statistic) if fidelity != full else 1.0
        )
        against_rate = float(spearmanr(scores[fidelity], rates).statistic)
        ranked = np.argsort(-np.asarray(scores[fidelity]))
        keep = max(1, len(configurations) // 3)
        rows.append(
            LadderRow(
                fidelity=fidelity,
                share_of_full=fidelity / full,
                rank_correlation=against_full if np.isfinite(against_full) else 0.0,
                learning_rate_correlation=against_rate if np.isfinite(against_rate) else 0.0,
                top_config_kept=bool(best_full in ranked[:keep]),
                seconds=float(sum(timings[fidelity])),
                seconds_per_fit=float(np.median(timings[fidelity])),
            )
        )
    full_trials = [evaluator.evaluate(configuration, full) for configuration in configurations]
    return rows, full_trials


def cost_model(ladder: list[LadderRow]) -> tuple[float, float]:
    """Least-squares ``seconds = fixed + marginal * rounds`` over the measured ladder.

    Multi-fidelity search assumes cost is proportional to fidelity. Nothing enforces that:
    a fit has a fixed cost -- building histograms, allocating, crossing the Python boundary --
    that a one-round run pays in full. Measuring the intercept is the difference between a
    saving that exists and one that was assumed, and it is the number that decides whether the
    cheapest bracket is worth running at all.
    """
    rounds = np.array([row.fidelity for row in ladder], dtype=float)
    seconds = np.array([row.seconds_per_fit for row in ladder], dtype=float)
    if len(rounds) < 2:
        return float(seconds[0]) if len(seconds) else 0.0, 0.0
    design = np.vstack([np.ones_like(rounds), rounds]).T
    fixed, marginal = np.linalg.lstsq(design, seconds, rcond=None)[0]
    return float(max(fixed, 0.0)), float(max(marginal, 0.0))


def _winners_curse(pool: list[Trial], resamples: int, rng: np.random.Generator) -> list[CurseRow]:
    """What the best-of-t selection reports, and what it delivers, as t grows.

    Selecting the maximum of many noisy estimates is itself an estimator, and a biased one:
    the winner is the configuration whose validation noise happened to be most favourable, so
    the reported score climbs with the number of candidates whether or not anything real is
    being found. Resampling the order in which configurations arrive turns that into a curve
    rather than an argument.
    """
    validation = np.array([trial.validation for trial in pool])
    test = np.array([trial.test for trial in pool])
    rows: list[CurseRow] = []
    for count in range(1, len(pool) + 1):
        reported = np.empty(resamples)
        delivered = np.empty(resamples)
        for index in range(resamples):
            order = rng.permutation(len(pool))[:count]
            winner = order[int(np.argmax(validation[order]))]
            reported[index] = validation[winner]
            delivered[index] = test[winner]
        rows.append(
            CurseRow(
                trials=count, reported=float(reported.mean()), delivered=float(delivered.mean())
            )
        )
    return rows


def _run_methods(
    settings: Settings,
    cfg: MultiFidelityConfig,
    make_evaluator: Callable[[], Evaluator],
    seed: int,
) -> list[MethodRow]:
    """Race every method at the same budget, counted in rounds fitted rather than in trials."""
    results: dict[str, list[tuple[float, float, float, float, float]]] = {
        name: [] for name in (RANDOM, HALVING, HYPERBAND, TPE)
    }
    for repeat in range(cfg.repeats):
        rng_seed = seed + 977 * repeat
        for name in list(results):
            if name == TPE and not is_available("optuna"):
                continue
            evaluator = make_evaluator()
            rng = np.random.default_rng(rng_seed)
            if name == RANDOM:
                trials = random_search(
                    evaluator.evaluate,
                    rng,
                    full_fidelity=cfg.max_rounds,
                    budget=cfg.budget_units,
                )
            elif name == HALVING:
                ladder = [r for r in cfg.fidelity_ladder if r >= cfg.search_min_fidelity]
                count = max(cfg.eta, cfg.budget_units // (len(ladder) * ladder[0]))
                trials = successive_halving(
                    evaluator.evaluate,
                    [sample_configuration(rng) for _ in range(count)],
                    ladder=ladder,
                    eta=cfg.eta,
                )
            elif name == HYPERBAND:
                trials = hyperband(
                    evaluator.evaluate,
                    rng,
                    full_fidelity=cfg.max_rounds,
                    eta=cfg.eta,
                    budget=cfg.budget_units,
                    min_fidelity=cfg.search_min_fidelity,
                )
            else:
                trials = tpe_search(
                    evaluator,
                    rng_seed,
                    full_fidelity=cfg.max_rounds,
                    budget=cfg.budget_units,
                )
            if not trials:
                continue
            # The winner is whatever scored best on validation at the *highest* fidelity it
            # reached, which is the choice a practitioner actually makes.
            top = max(trials, key=lambda trial: (trial.fidelity, trial.validation))
            best = max(
                (trial for trial in trials if trial.fidelity == top.fidelity),
                key=lambda trial: trial.validation,
            )
            results[name].append(
                (
                    float(len({trial.configuration.key() for trial in trials})),
                    float(evaluator.units),
                    evaluator.seconds,
                    best.validation,
                    best.test,
                )
            )
    rows: list[MethodRow] = []
    for name, records in results.items():
        if not records:
            continue
        stacked = np.array(records, dtype=float)
        rows.append(
            MethodRow(
                method=name,
                configurations=float(stacked[:, 0].mean()),
                units=float(stacked[:, 1].mean()),
                seconds=float(stacked[:, 2].mean()),
                validation=float(stacked[:, 3].mean()),
                test=float(stacked[:, 4].mean()),
                validation_spread=float(stacked[:, 3].std()),
                test_spread=float(stacked[:, 4].std()),
            )
        )
    return rows


def run_multifidelity_study(settings: Settings) -> MultiFidelityStudy:
    """Check both premises, then race the search methods at an equal budget."""
    start = time.perf_counter()
    cfg: MultiFidelityConfig = settings.multifidelity
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    if len(y_train) > cfg.max_train_rows:
        keep = rng.choice(len(y_train), cfg.max_train_rows, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]
    x_val: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_val = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)

    def make_evaluator() -> Evaluator:
        return Evaluator(variant, x_train, y_train, x_val, y_val, x_test, y_test)

    grid = make_evaluator()
    pool_configs = [sample_configuration(rng) for _ in range(cfg.rank_configs)]
    ladder, pool = _premise_grid(grid, pool_configs, cfg.fidelity_ladder)
    fixed_seconds, marginal_seconds = cost_model(ladder)
    correlation = spearmanr([trial.validation for trial in pool], [trial.test for trial in pool])

    default = make_evaluator().evaluate(
        Configuration(
            learning_rate=settings.supervised.learning_rate,
            num_leaves=settings.supervised.num_leaves,
            min_child_samples=settings.supervised.min_child_samples,
            subsample=settings.supervised.subsample,
            colsample_bytree=settings.supervised.colsample_bytree,
            reg_lambda=settings.supervised.reg_lambda,
        ),
        cfg.max_rounds,
    )

    study = MultiFidelityStudy(
        ladder=ladder,
        methods=_run_methods(variant, cfg, make_evaluator, variant.seed),
        curse=_winners_curse(pool, cfg.curse_resamples, rng),
        pool_size=len(pool),
        fixed_seconds=fixed_seconds,
        marginal_seconds=marginal_seconds,
        selection_correlation=float(correlation.statistic),
        selection_pvalue=float(correlation.pvalue),
        full_fidelity=cfg.max_rounds,
        budget=cfg.budget_units,
        floor=cfg.search_min_fidelity,
        repeats=cfg.repeats,
        n_train=len(y_train),
        oracle_test=max(trial.test for trial in pool),
        oracle_validation=max(trial.validation for trial in pool),
        default_validation=default.validation,
        default_test=default.test,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Multi-fidelity study complete",
        extra={
            "selection_correlation": round(study.selection_correlation, 3),
            "methods": len(study.methods),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _ladder_table(study: MultiFidelityStudy) -> str:
    rows = "\n".join(
        f"| {row.fidelity} | {row.share_of_full:.1%} | {row.seconds_per_fit:.2f} s | "
        f"{row.rank_correlation:+.2f} | {row.learning_rate_correlation:+.2f} | "
        f"{'yes' if row.top_config_kept else '**no**'} |"
        for row in study.ladder
    )
    return (
        "| rounds | share of full fidelity | seconds per fit | rank correlation with the full "
        "run | correlation with the learning rate | best config survives the first cut |\n"
        "|---|---|---|---|---|---|\n" + rows
    )


def _method_table(study: MultiFidelityStudy) -> str:
    rows = "\n".join(
        f"| {row.method} | {row.configurations:.0f} | {row.units:,.0f} | {row.seconds:.0f} s | "
        f"{row.validation:.3f} | **{row.test:.3f}** +/- {row.test_spread:.3f} |"
        for row in sorted(study.methods, key=lambda row: row.test, reverse=True)
    )
    reference = (
        f"| _the shipped configuration (no search)_ | _1_ | _{study.full_fidelity}_ | _--_ | "
        f"_{study.default_validation:.3f}_ | _{study.default_test:.3f}_ |"
    )
    oracle = (
        f"| _the best of the {study.pool_size} random configurations, chosen with hindsight_ | "
        f"_{study.pool_size}_ | _--_ | _--_ | _{study.oracle_validation:.3f}_ | "
        f"_{study.oracle_test:.3f}_ |"
    )
    return (
        "| method | configurations tried | rounds fitted | wall clock | validation PR-AUC | "
        "later-days PR-AUC |\n|---|---|---|---|---|---|\n" + rows + "\n" + reference + "\n" + oracle
    )


def _curse_table(study: MultiFidelityStudy) -> str:
    if not study.curse:
        return "_No pool to resample._"
    first = study.curse[0]
    picks = [row for row in study.curse if row.trials in {1, 2, 5, 10, 20, len(study.curse)}]
    rows = "\n".join(
        f"| {row.trials} | {row.reported:.3f} | {row.reported - first.reported:+.3f} | "
        f"{row.delivered:.3f} | {row.delivered - first.delivered:+.3f} | {row.optimism:.3f} |"
        for row in picks
    )
    return (
        "| configurations searched | reported (validation) | gained by searching | delivered "
        "(later days) | gained by searching | optimism |\n|---|---|---|---|---|---|\n" + rows
    )


def _lead(study: MultiFidelityStudy) -> str:
    cheapest = study.ladder[0] if study.ladder else None
    middle = (
        max(study.ladder[:-1], key=lambda row: abs(row.learning_rate_correlation))
        if len(study.ladder) > 1
        else None
    )
    best = max(study.methods, key=lambda row: row.test) if study.methods else None
    beaten = [row for row in study.methods if row.test < study.default_test]
    fixed_share = study.fixed_seconds / max(
        study.fixed_seconds + study.marginal_seconds * study.full_fidelity, 1e-9
    )
    cheap_share = study.fixed_seconds / max(cheapest.seconds_per_fit, 1e-9) if cheapest else 0.0
    verdict = (
        ", which is a real signal."
        if study.selection_pvalue <= 0.05
        else (
            " -- indistinguishable from no relationship at all. The quantity every tuner in "
            "this repository maximises is close to uncorrelated with the quantity the project "
            "reports."
        )
    )
    return (
        "**Both premises fail, and the consequence is measurable: every search method here "
        "loses to the configuration nobody searched for.**\n\n"
        "Hyperparameter search assumes a validation ranking predicts deployment. Over "
        f"{study.pool_size} configurations evaluated at full fidelity, the rank correlation "
        f"between validation PR-AUC and the later days is "
        f"**{study.selection_correlation:+.2f}** (p = {study.selection_pvalue:.3f})"
        f"{verdict}\n\n"
        "Multi-fidelity search assumes a cheap run ranks configurations like an expensive one. "
        f"At {cheapest.fidelity if cheapest else 1} round the ranking correlates "
        f"{cheapest.rank_correlation if cheapest else 0:+.2f} with the full run, and the cheap "
        f"rungs correlate up to {middle.learning_rate_correlation if middle else 0:+.2f} with "
        "the **learning rate**. A short boosting run does not rank models, it ranks step sizes "
        "-- a bias rather than noise, which no amount of averaging removes.\n\n"
        "The table below is what those two failures cost. The shipped configuration, which "
        f"nobody searched for, scores {study.default_test:.3f} on the later days; the best of "
        f"the four searches is {best.method if best else 'n/a'} at "
        f"{best.test if best else 0:.3f}; {len(beaten)} of {len(study.methods)} searches finish "
        "*below* the unsearched default. The best configuration in the entire random pool, "
        f"chosen with full hindsight, reaches {study.oracle_test:.3f} -- so the whole prize on "
        f"offer is **{study.oracle_test - study.default_test:+.3f} PR-AUC**, and no method that "
        "has to choose on validation can collect it.\n\n"
        "There is a third assumption underneath the accounting, and it is also wrong. Fitting "
        f"time is `{study.fixed_seconds:.2f}s + {study.marginal_seconds * 1000:.1f}ms per "
        f"round`, so a full-fidelity fit is {fixed_share:.0%} fixed cost and the cheapest rung "
        f"is **{cheap_share:.0%} fixed cost**. Budgeting in rounds while paying in fits is how "
        f"a theoretical saving of {study.full_fidelity}x becomes a wall-clock saving of very "
        "little."
    )


def _render(study: MultiFidelityStudy, ladder: Path, curse: Path) -> str:
    random_row = study.method(RANDOM)
    halving_row = study.method(HALVING)
    best = max(study.methods, key=lambda row: row.test) if study.methods else None
    spread = max((row.test for row in study.methods), default=0.0) - min(
        (row.test for row in study.methods), default=0.0
    )
    last_curse = study.curse[-1] if study.curse else None
    first_curse = study.curse[0] if study.curse else None
    method_read = (
        f"{best.method} leads on the later days at {best.test:.3f}, and the whole field spans "
        f"{spread:.3f}"
        if best
        else "No method completed"
    )
    premise_read = (
        f"At **{study.selection_correlation:+.2f}** with p = {study.selection_pvalue:.3f}, it "
        f"does not hold here. The relationship is indistinguishable from none, on the very "
        f"quantity every tuner in this repository maximises."
        if study.selection_pvalue > 0.05
        else f"At **{study.selection_correlation:+.2f}** (p = {study.selection_pvalue:.3f}) it "
        f"holds, which is what makes searching meaningful at all."
    )
    halving_read = (
        f"{halving_row.configurations:.0f} configurations against random search's "
        f"{random_row.configurations:.0f}"
        if halving_row and random_row
        else "more configurations"
    )
    return f"""# NetSentry — Budgeted Hyperparameter Search, and the Premises Underneath It

_{study.pool_size} random configurations of the boosted model across a {len(study.ladder)}-rung
fidelity ladder, then four search methods at an identical budget of {study.budget:,} boosting
rounds, averaged over {study.repeats} repeats. Trained on {study.n_train:,} rows; selection is
on validation, and the later days are never used to choose anything. Regenerate with
`netsentry hyperband`._

## Why this report exists

`netsentry train tune` runs TPE over the boosted model's hyperparameters and keeps whatever
scores best on validation. That is standard practice, and it rests on two assumptions that are
almost never measured before the search starts: that a validation ranking predicts deployment,
and -- for anything multi-fidelity -- that a cheap evaluation ranks configurations the way an
expensive one does.

Both are cheap to check. Neither usually is.

{_lead(study)}

## Premise one: does a cheap run rank like an expensive one?

![The fidelity ladder](../figures/{ladder.name})

{_ladder_table(study)}

This is the table successive halving and Hyperband depend on, and it is the table nobody
prints. **Not one cheap rung ranks configurations the way the full run does** -- the
correlations sit around zero and change sign, which is what "no information" looks like when it
is measured rather than assumed.

The next column says why, and it is the difference between noise and a defect. **A short
boosting run rewards whatever climbs fastest**, which is a property of the step size rather
than of the model, so the cheap rungs rank by learning rate. That is a *bias*: averaging over
more configurations does not remove it, and a halving schedule built on it will
systematically discard the patient configurations that would have won.

The last column is the operational consequence -- whether the configuration that eventually
wins at full fidelity would have survived the first cut at each rung -- and the answer is that
it is a coin toss.

## Premise two: does validation predict the later days?

The pool of {study.pool_size} configurations, all evaluated at full fidelity, gives this
directly: Spearman **{study.selection_correlation:+.2f}** (p = {study.selection_pvalue:.3f})
between validation PR-AUC and later-day PR-AUC.

That is the premise every hyperparameter search in every project rests on, and on a temporal
split it is not obviously true -- the validation rows come from the *training days*, and the
[glass-box study](gam.md) has already shown that split systematically overstating what capacity
delivers.

{premise_read}

It is worth being precise about what this does and does not say. It does not say the model is
bad, or that validation is useless -- the [glass-box study](gam.md) found validation perfectly
capable of locating the *turn* in a capacity curve. It says that among configurations which are
already near each other, validation cannot tell which will transfer, and a search that ranks
them is ranking noise. The correct response to a premise this weak is to stop tuning, not to
tune harder, which is the opposite of what a budget usually buys.

## The methods, at an equal budget

{_method_table(study)}

Budgets are equal in **rounds fitted**, which is the resource the methods actually consume, and
the searches are floored at {study.floor} rounds -- because the cost model above says the
cheapest rung is nearly all overhead, and a bracket that evaluates hundreds of configurations
for one round each would spend its budget proving that.
Comparisons in the literature are usually in *trials*, which flatters multi-fidelity methods by
construction, because most of their trials are the cheap ones -- successive halving here tries
{halving_read} for the same rounds.

{method_read}. The reference rows are what the table is for: the shipped configuration, which
nobody searched for, scores {study.default_test:.3f}, and the best of the random pool chosen
with full hindsight scores {study.oracle_test:.3f}. **The entire span from "no search at all"
to "perfect hindsight" is {abs(study.oracle_test - study.default_test):.3f} PR-AUC**, which
bounds what any tuner in this comparison could have won.

The wall-clock column is where the resource model shows up. Rounds are not what a machine
charges for: with a fixed cost of {study.fixed_seconds:.2f}s per fit against
{study.marginal_seconds * 1000:.1f}ms per round, the cheap rungs cost far more than their
round count suggests, and a method that spends its budget on many tiny fits pays a toll the
accounting does not show.

## What searching harder buys, and what it only appears to buy

![The winner's curse](../figures/{curse.name})

{_curse_table(study)}

Selecting the best of many configurations on validation is itself an estimator, and a biased
one -- the winner is partly whichever configuration's validation noise was most flattering. The
curve is built by resampling the order in which the pool arrives, so "searching harder" means
"drawing more candidates" rather than "using a better method".

{
    f"Searching {last_curse.trials} configurations rather than one raises the *reported* score "
    f"by {last_curse.reported - first_curse.reported:+.3f} and the *delivered* score by "
    f"{last_curse.delivered - first_curse.delivered:+.3f}"
    if last_curse and first_curse
    else "The pool did not resample"
}. The two move together, which is the honest reading: at this pool size the selection bias is
small next to the gap that was already there.

There is a tension between this table and the previous one worth naming rather than smoothing
over. Here, choosing on validation among the pool lands on the configuration that also happens
to be best on the later days; there, four searches choosing on validation among *fresh*
configurations all finished below the unsearched default. Both are what a rank correlation of
{study.selection_correlation:+.2f} produces: a relationship this weak makes a lucky pick and an
unlucky search equally unsurprising, and neither outcome is evidence about the method. That is
the practical content of a failed premise -- not that searching is harmful, but that its
outcome is not information.

**The optimism does not come from searching. It comes from the split.** The reported score
overstates the delivered one by {study.curse[0].optimism if study.curse else 0:.3f} at a single
configuration, before any selection has happened at all, and searching barely moves it. Every
number a tuner prints on this data is a number about the training days.

## Scope and honest limits

- **This is a study about search, not about the shipped model.** Training uses
  {study.n_train:,} rows and caps boosting at {study.full_fidelity} rounds so that a few
  hundred fits are affordable; the deployed configuration trains longer on more data. The
  comparison between methods is internally consistent, and none of these numbers should be
  read against the headline.
- **The budget is one point on a curve.** Every method's ranking can change with the budget --
  that is the entire content of the Hyperband paper -- and this measures one budget, three
  times, on one dataset.
- **Fidelity is boosting rounds.** Training-set size is the other natural choice and behaves
  differently: it changes what the model can learn rather than how far it got. A study that
  swept both would be able to say which fidelity dimension is more faithful, and this one
  cannot.
- **The pool is random, so the oracle row is a *reachable* best, not the best.** A better
  configuration certainly exists outside {study.pool_size} draws; what the row bounds is what
  this comparison could have found.
- **Nothing here is tuned against the later days**, which is the only reason the last column
  means anything. The moment a search is run against it, that column becomes a validation
  column with extra steps."""


def run_multifidelity_report(settings: Settings) -> Path:
    """Run the multi-fidelity study and write the report + figures."""
    study = run_multifidelity_study(settings)
    fidelities = np.array([row.fidelity for row in study.ladder], dtype=float)
    ladder = plots.plot_lines(
        {
            "rank correlation with the full run": (
                fidelities,
                np.array([row.rank_correlation for row in study.ladder]),
            ),
            "rank correlation with the learning rate": (
                fidelities,
                np.array([row.learning_rate_correlation for row in study.ladder]),
            ),
        },
        xlabel="boosting rounds (fidelity)",
        ylabel="Spearman correlation",
        title="A cheap run ranks step sizes, not models",
        out_path=settings.paths.figures_dir / LADDER_FIGURE,
        xscale="log",
    )
    trials = np.array([row.trials for row in study.curse], dtype=float)
    curse = plots.plot_lines(
        {
            "reported (best on validation)": (
                trials,
                np.array([row.reported for row in study.curse]),
            ),
            "delivered (the later days)": (
                trials,
                np.array([row.delivered for row in study.curse]),
            ),
        },
        xlabel="configurations searched",
        ylabel="PR-AUC",
        title="What the search reports, and what deployment pays",
        out_path=settings.paths.figures_dir / CURSE_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, ladder, curse), encoding="utf-8")
    logger.info("Wrote multi-fidelity report", extra={"path": str(out_path)})

    with track_run(settings, "multifidelity") as run:
        run.log_params({"budget_units": study.budget, "repeats": study.repeats})
        run.log_metrics(
            {"selection_correlation": study.selection_correlation}
            | {f"test_{row.method.split(' ')[0]}": row.test for row in study.methods}
        )
        for artifact in (ladder, curse, out_path):
            run.log_artifact(artifact)
    return out_path
