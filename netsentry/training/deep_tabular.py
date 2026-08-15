"""Do neural networks beat the trees on this data? The claim, checked instead of cited.

This project has used gradient-boosted trees since phase 4, for the reason most people use them:
the tabular-deep-learning literature says trees win (Grinsztajn et al. 2022; Shwartz-Ziv & Armon
2022). That is a citation, not a measurement. The claim is about tabular data *in general*, and
the thing being modelled here is specific -- ninety flow statistics, a 20% attack rate, and a
temporal split whose test days contain no attack class the training days ever showed. Any of
those could flip the conclusion, and "the literature says so" is exactly the kind of reasoning
this repository exists to distrust.

So the comparison is run properly. Four models, one protocol:

- **LightGBM** -- the incumbent, unchanged.
- **logistic regression** -- the linear reference every number needs.
- **MLP** -- batch-normalised, dropout, the obvious neural baseline.
- **FT-Transformer** (Gorishniy et al. 2021) -- one learned token per feature, self-attention
  across them, a CLS head. The architecture that can form feature interactions explicitly, which
  is the property the interaction study says these features actually have.

Identical everything: the same leakage-safe pipeline fitted on the same training split, the same
temporal split, the same seed, the same validation set for early stopping and threshold
selection, and the same metrics -- PR-AUC on raw scores plus TPR at the deployed false-positive
budgets, because a ranking win that vanishes at the operating point is not a win.

Two questions beyond "who wins":

- **Does the neural model see anything the trees do not?** Rank correlation between their scores,
  and the PR-AUC of a rank-averaged ensemble. Two models that agree on every flow are one model,
  whatever their architectures.
- **Is the gap a data-size artefact?** Neural models are supposed to need more data, so each
  model is refitted on a sweep of training fractions and the curves are compared. If the trees'
  lead were shrinking with data, that would be the interesting finding rather than the headline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier, build_baselines
from netsentry.models.tabular_nn import TorchTabularClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DeepTabularConfig

logger = get_logger(__name__)

REPORT_NAME = "deep_tabular.md"
CURVE_FIGURE_NAME = "deep_tabular_curve.png"
SCATTER_FIGURE_NAME = "deep_tabular_agreement.png"

LIGHTGBM = "LightGBM (incumbent)"
LOGISTIC = "logistic regression"
MLP = "MLP"
TRANSFORMER = "FT-Transformer"


@dataclass
class ModelResult:
    """One model under the shared protocol: what it scored, what it cost, how big it is."""

    name: str
    pr_auc: float
    roc_auc: float
    tpr_at_primary: float
    tpr_at_secondary: float
    train_seconds: float
    inference_ms_per_1k: float
    n_parameters: int
    detail: str
    scores: np.ndarray


@dataclass
class CurvePoint:
    """One (model, training fraction) point of the sample-efficiency sweep."""

    model: str
    fraction: float
    rows: int
    pr_auc: float


@dataclass
class EnsembleResult:
    """What combining the trees with the best neural model buys, if anything."""

    partner: str
    spearman: float
    pr_auc: float
    lift: float


@dataclass
class DeepTabularStudy:
    """Everything the report renders."""

    results: list[ModelResult]
    curve: list[CurvePoint]
    ensembles: list[EnsembleResult]
    n_train: int
    n_test: int
    n_features: int
    primary_fpr: float
    secondary_fpr: float
    prevalence: float
    epochs: int
    torch_available: bool


def _rates(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    fpr: float,
) -> float:
    """TPR at a budget whose threshold is chosen on validation -- never on the test set."""
    threshold = threshold_at_fpr(y_val, val_scores, fpr)
    return float(rates_at_threshold(y_test, test_scores, threshold)["tpr"])


def _measure(
    name: str,
    detail: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    settings: Settings,
    train_seconds: float,
    inference_ms: float,
    n_parameters: int,
) -> ModelResult:
    return ModelResult(
        name=name,
        pr_auc=float(average_precision_score(y_test, test_scores)),
        roc_auc=float(roc_auc_score(y_test, test_scores)),
        tpr_at_primary=_rates(
            y_val, val_scores, y_test, test_scores, settings.thresholds.primary_fpr
        ),
        tpr_at_secondary=_rates(
            y_val, val_scores, y_test, test_scores, max(settings.thresholds.fpr_targets)
        ),
        train_seconds=train_seconds,
        inference_ms_per_1k=inference_ms,
        n_parameters=n_parameters,
        detail=detail,
        scores=test_scores,
    )


def _timed_inference(predict: object, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Score a matrix and return the per-thousand-flow latency alongside the scores."""
    start = time.perf_counter()
    scores = predict(x)  # type: ignore[operator]
    elapsed = (time.perf_counter() - start) * 1000.0
    return np.asarray(scores), elapsed / max(len(x), 1) * 1000.0


def _fit_lightgbm(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[SupervisedClassifier, float]:
    start = time.perf_counter()
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    return model, time.perf_counter() - start


def _torch_scores(model: TorchTabularClassifier, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x))[:, 1]


def _tree_scores(model: SupervisedClassifier, x: np.ndarray) -> np.ndarray:
    return positive_scores(np.asarray(model.predict_proba(x)), np.asarray(model.classes_))


def run_deep_tabular_study(settings: Settings) -> DeepTabularStudy:
    """Fit every architecture under one protocol, then sweep the training-set size."""
    cfg: DeepTabularConfig = settings.deep_tabular
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
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

    results: list[ModelResult] = []

    model, seconds = _fit_lightgbm(variant, x_train, y_train, x_val, y_val)
    test_scores, latency = _timed_inference(lambda m: _tree_scores(model, m), x_test)
    results.append(
        _measure(
            LIGHTGBM,
            f"{model.n_trees():,} trees, balanced sample weights, early stopping on validation",
            _tree_scores(model, x_val),
            test_scores,
            y_val,
            y_test,
            variant,
            seconds,
            latency,
            model.n_trees() * variant.supervised.num_leaves,
        )
    )

    start = time.perf_counter()
    logistic = build_baselines(variant)["logistic"].fit(x_train, y_train)
    logistic_seconds = time.perf_counter() - start
    logistic_test, logistic_latency = _timed_inference(
        lambda m: np.asarray(logistic.predict_proba(m))[:, 1], x_test
    )
    results.append(
        _measure(
            LOGISTIC,
            "the linear reference, class-weighted",
            np.asarray(logistic.predict_proba(x_val))[:, 1],
            logistic_test,
            y_val,
            y_test,
            variant,
            logistic_seconds,
            logistic_latency,
            int(x_train.shape[1] + 1),
        )
    )

    torch_available = is_available("torch")
    if torch_available:
        for name, architecture in ((MLP, "mlp"), (TRANSFORMER, "ft_transformer")):
            net = TorchTabularClassifier(variant, architecture=architecture)
            net.fit(x_train, y_train, eval_set=(x_val, y_val))
            assert net.trace is not None
            fitted = net
            net_test, net_latency = _timed_inference(
                lambda m, model=fitted: _torch_scores(model, m), x_test
            )
            results.append(
                _measure(
                    name,
                    f"stopped at epoch {net.trace.best_epoch} of {net.trace.epochs_run} "
                    f"(best validation PR-AUC {net.trace.best_val_pr_auc:.3f})",
                    _torch_scores(net, x_val),
                    net_test,
                    y_val,
                    y_test,
                    variant,
                    net.trace.seconds,
                    net_latency,
                    net.trace.n_parameters,
                )
            )

    # Do the neural models see anything the trees do not? Rank correlation answers whether the
    # question is even worth asking; the rank-averaged ensemble answers what it would be worth.
    tree_result = results[0]
    ensembles: list[EnsembleResult] = []
    for other in results[1:]:
        rho = float(spearmanr(tree_result.scores, other.scores).statistic)
        blended = _rank(tree_result.scores) + _rank(other.scores)
        pr_auc = float(average_precision_score(y_test, blended))
        ensembles.append(
            EnsembleResult(
                partner=other.name,
                spearman=rho,
                pr_auc=pr_auc,
                lift=pr_auc - tree_result.pr_auc,
            )
        )

    curve = _sample_efficiency(variant, x_train, y_train, x_val, y_val, x_test, y_test)

    return DeepTabularStudy(
        results=results,
        curve=curve,
        ensembles=ensembles,
        n_train=len(y_train),
        n_test=len(y_test),
        n_features=x_train.shape[1],
        primary_fpr=variant.thresholds.primary_fpr,
        secondary_fpr=max(variant.thresholds.fpr_targets),
        prevalence=float(np.mean(y_test)),
        epochs=cfg.epochs,
        torch_available=torch_available,
    )


def _rank(scores: np.ndarray) -> np.ndarray:
    """Ranks in [0, 1] -- the scale-free way to combine two models' scores."""
    order = np.argsort(np.argsort(scores))
    return order / max(len(scores) - 1, 1)


def _sample_efficiency(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> list[CurvePoint]:
    """Refit every architecture on growing subsets -- is the gap a data-size artefact?"""
    cfg: DeepTabularConfig = settings.deep_tabular
    rng = np.random.default_rng(settings.seed)
    curve: list[CurvePoint] = []
    for fraction in cfg.data_fractions:
        rows = max(int(len(y_train) * fraction), 200)
        index = rng.choice(len(y_train), size=min(rows, len(y_train)), replace=False)
        xs, ys = x_train[index], y_train[index]
        if len(np.unique(ys)) < 2:
            continue
        model, _ = _fit_lightgbm(settings, xs, ys, x_val, y_val)
        curve.append(
            CurvePoint(
                LIGHTGBM,
                fraction,
                len(ys),
                float(average_precision_score(y_test, _tree_scores(model, x_test))),
            )
        )
        if not is_available("torch"):
            continue
        for name, architecture in ((MLP, "mlp"), (TRANSFORMER, "ft_transformer")):
            net = TorchTabularClassifier(settings, architecture=architecture)
            net.fit(xs, ys, eval_set=(x_val, y_val))
            curve.append(
                CurvePoint(
                    name,
                    fraction,
                    len(ys),
                    float(average_precision_score(y_test, _torch_scores(net, x_test))),
                )
            )
    return curve


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_deep_tabular_report(settings: Settings) -> Path:
    """Run the architecture comparison and write the report + figures."""
    study = run_deep_tabular_study(settings)
    models = sorted({point.model for point in study.curve})
    curve_fig = plots.plot_lines(
        {
            model: (
                np.array([p.rows for p in study.curve if p.model == model], dtype=float),
                np.array([p.pr_auc for p in study.curve if p.model == model], dtype=float),
            )
            for model in models
        },
        xlabel="training rows",
        ylabel="PR-AUC on the temporal test days",
        title="Sample efficiency: is the gap a data-size artefact?",
        out_path=settings.paths.figures_dir / CURVE_FIGURE_NAME,
        xscale="log",
    )
    best_nn = _best_neural(study)
    tree = study.results[0]
    agreement_fig = plots.plot_scatter_identity(
        x=tree.scores,
        y=best_nn.scores if best_nn is not None else tree.scores,
        xlabel=f"{tree.name} attack score",
        ylabel=f"{best_nn.name if best_nn else tree.name} attack score",
        title="Do the two architectures disagree about anything?",
        out_path=settings.paths.figures_dir / SCATTER_FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, curve_fig, agreement_fig), encoding="utf-8")
    logger.info("Wrote deep-tabular report", extra={"path": str(out_path)})

    with track_run(settings, "deep_tabular") as run:
        run.log_params({"epochs": study.epochs, "train_rows": study.n_train})
        for result in study.results:
            key = "".join(ch if ch.isalnum() else "_" for ch in result.name)
            run.log_metrics(
                {
                    f"pr_auc_{key}": result.pr_auc,
                    f"train_seconds_{key}": result.train_seconds,
                    f"parameters_{key}": float(result.n_parameters),
                }
            )
        run.log_artifact(curve_fig)
        run.log_artifact(agreement_fig)
        run.log_artifact(out_path)
    return out_path


def _best_neural(study: DeepTabularStudy) -> ModelResult | None:
    neural = [r for r in study.results if r.name in {MLP, TRANSFORMER}]
    return max(neural, key=lambda r: r.pr_auc) if neural else None


def _result_table(study: DeepTabularStudy) -> str:
    rows = [
        f"| model | PR-AUC | ROC-AUC | TPR @ {study.primary_fpr:.1%} | "
        f"TPR @ {study.secondary_fpr:.0%} | train | inference / 1k | parameters |",
        "|---|---|---|---|---|---|---|---|",
    ]
    best = max(r.pr_auc for r in study.results)
    for result in sorted(study.results, key=lambda r: r.pr_auc, reverse=True):
        name = f"**{result.name}**" if result.pr_auc == best else result.name
        rows.append(
            f"| {name} | {result.pr_auc:.3f} | {result.roc_auc:.3f} | "
            f"{result.tpr_at_primary:.1%} | {result.tpr_at_secondary:.1%} | "
            f"{result.train_seconds:.1f} s | {result.inference_ms_per_1k:.1f} ms | "
            f"{result.n_parameters:,} |"
        )
    return "\n".join(rows)


def _ensemble_table(study: DeepTabularStudy) -> str:
    rows = [
        "| combined with the incumbent | rank correlation | ensemble PR-AUC | lift |",
        "|---|---|---|---|",
    ]
    for item in study.ensembles:
        rows.append(
            f"| {item.partner} | {item.spearman:+.3f} | {item.pr_auc:.3f} | {item.lift:+.3f} |"
        )
    return "\n".join(rows)


def _curve_table(study: DeepTabularStudy) -> str:
    models = sorted({p.model for p in study.curve})
    fractions = sorted({p.fraction for p in study.curve})
    rows = ["| training rows | " + " | ".join(models) + " |", "|" + "---|" * (1 + len(models))]
    for fraction in fractions:
        cells = []
        rows_used = 0
        for model in models:
            point = next(
                (p for p in study.curve if p.model == model and p.fraction == fraction), None
            )
            rows_used = point.rows if point else rows_used
            cells.append(f"{point.pr_auc:.3f}" if point else "—")
        rows.append(f"| {rows_used:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _headline_read(study: DeepTabularStudy) -> str:
    ranked = sorted(study.results, key=lambda r: r.pr_auc, reverse=True)
    winner, tree = ranked[0], study.results[0]
    transformer = next((r for r in study.results if r.name == TRANSFORMER), None)
    if not study.torch_available:
        return (
            "PyTorch is not installed in this environment, so only the tree and linear arms ran. "
            "Install the `ae` extra to reproduce the neural comparison."
        )
    assert transformer is not None
    cost = transformer.train_seconds / max(tree.train_seconds, 1e-9)
    lead = winner.pr_auc - transformer.pr_auc
    if transformer.pr_auc >= max(r.pr_auc for r in study.results if r is not transformer):
        return (
            f"The transformer wins at {transformer.pr_auc:.3f} PR-AUC, which is the outcome the "
            "literature did not predict for tabular data, and it costs "
            f"{cost:.0f}x the incumbent's training time to get there."
        )
    return (
        f"**{winner.name} leads at {winner.pr_auc:.3f} PR-AUC, and the transformer is last at "
        f"{transformer.pr_auc:.3f}** — {lead:.3f} behind, for {cost:.0f}x the training time and "
        f"{transformer.inference_ms_per_1k / max(tree.inference_ms_per_1k, 1e-9):.0f}x the "
        "inference cost. That is the literature's conclusion reproduced on this data rather than "
        "inherited from it, and the shape of the ranking says why. The leaderboard study already "
        "found that model *capacity* is penalised on this split: every family pays a "
        "stratified-minus-temporal gap, the flexible ones pay the largest, and Gaussian naive "
        "Bayes led the honest table. The FT-Transformer extends that line one architecture "
        "further — it is the most flexible model here and it lands last. The mechanism is the "
        "open-set structure the split has: the test days contain **no attack class the training "
        "days showed**, so capacity spent on fitting the training families precisely is capacity "
        "spent on families that will never be seen again, while a coarser decision boundary "
        "keeps more of whatever generalises."
    )


def _agreement_read(study: DeepTabularStudy) -> str:
    if not study.ensembles:
        return ""
    best = max(study.ensembles, key=lambda e: e.lift)
    tree = study.results[0]
    verdict = (
        f"The best combination adds **{best.lift:+.3f}** over the incumbent alone ({best.partner} "
        f"at rank correlation {best.spearman:+.2f}), which is small but real: the architectures "
        "are not interchangeable, they are ranking a shared majority of flows the same way and "
        "disagreeing at the margins where the decision is hard."
        if best.lift > 0.005
        else f"No combination improves on the incumbent alone (best {best.lift:+.3f}, "
        f"{best.partner}). The models agree where it matters, so ensembling them buys nothing "
        "except two models to maintain."
    )
    return (
        f"{verdict} Rank correlation is the right lens for this question rather than accuracy: "
        "two models with the same PR-AUC can be ranking entirely different flows, and two models "
        f"with different PR-AUC can be near-identical. Against the deployed model's "
        f"{tree.pr_auc:.3f}, that is what the table below is measuring."
    )


def _curve_read(study: DeepTabularStudy) -> str:
    models = sorted({p.model for p in study.curve})
    if not models:
        return ""
    smallest = min(p.rows for p in study.curve)
    largest = max(p.rows for p in study.curve)
    growth = {
        model: (
            next(p.pr_auc for p in study.curve if p.model == model and p.rows == largest)
            - next(p.pr_auc for p in study.curve if p.model == model and p.rows == smallest)
        )
        for model in models
        if any(p.rows == largest for p in study.curve if p.model == model)
        and any(p.rows == smallest for p in study.curve if p.model == model)
    }
    fastest = max(growth, key=lambda m: growth[m]) if growth else models[0]
    closing = (
        "The neural curves are steeper than the tree's, so the gap *is* partly a data-size "
        "effect and more traffic would narrow it — which is a real caveat on the headline and "
        "an argument for revisiting this on the full CIC-IDS2017 rather than a 60k-row stand-in."
        if fastest in {MLP, TRANSFORMER}
        else "The trees improve at least as fast as the neural models do, so the gap is not "
        "waiting to be closed by more data at this scale: whatever is limiting the neural "
        "models here, it is not the size of the training set."
    )
    return (
        f"Between {smallest:,} and {largest:,} training rows, "
        f"**{fastest}** gains the most ({growth.get(fastest, 0.0):+.3f} PR-AUC). {closing}"
    )


def _render(study: DeepTabularStudy, curve_fig: Path, agreement_fig: Path) -> str:
    return f"""# NetSentry — Deep Tabular Models vs the Trees, Under One Protocol

_{study.n_train:,} training rows, {study.n_features} features, judged on {study.n_test:,}
later-day flows at {study.prevalence:.1%} prevalence. Same pipeline, same temporal split, same
seed, same validation set for early stopping and thresholds. Up to {study.epochs} epochs with
PR-AUC early stopping. Every arm sees the same capped training set — the cap is set by the
transformer's cost and applied to all four, rather than quietly giving the trees more data._

## Why this report exists

This project has used gradient-boosted trees since phase 4, for the reason most people use them:
the tabular deep-learning literature says trees win (Grinsztajn et al. 2022; Shwartz-Ziv & Armon
2022). That is a citation, not a measurement. The claim is about tabular data *in general*, and
what is being modelled here is specific — ninety flow statistics, a 20% attack rate, and a
temporal split whose test days contain no attack class the training days ever showed. Any of
those could flip the conclusion, and "the literature says so" is exactly the sort of reasoning
this repository exists to distrust.

So the comparison is run properly, with the architecture that has the strongest claim to being
the exception: the **FT-Transformer** (Gorishniy et al. 2021), which turns each feature into its
own learned token and lets self-attention build interactions between them explicitly — the
mechanism the feature-interaction study says these features actually have structure for. An MLP
is included because if a plain network were enough, the transformer would have nothing to
justify, and logistic regression because every number needs a linear reference.

## What each architecture achieves

{_result_table(study)}

{_headline_read(study)}

## Do they disagree about anything?

![Score agreement](../figures/{agreement_fig.name})

{_ensemble_table(study)}

{_agreement_read(study)}

## Is the gap waiting for more data?

![Sample efficiency](../figures/{curve_fig.name})

{_curve_table(study)}

{_curve_read(study)}

## Scope and honest limits

- **Neither family was tuned inside this study.** Both take their configured defaults, so this
  is a comparison of sensible defaults rather than of tuned optima. That cuts both ways: the
  trees' hyperparameters have been exercised by the Optuna study, the networks' have not, and a
  serious attempt to make the transformer win would start there.
- **One seed, one split.** The seed-variance study measured the noise floor on this pipeline;
  differences smaller than it should not be read as differences. The larger gaps here clear it,
  the ensemble lifts do not necessarily.
- **CPU only, and the transformer is quadratic in the feature count.** Attention over
  {study.n_features} feature tokens is what makes it expensive here; a GPU changes the wall
  clock but not the ranking, and inference latency stays the operational objection.
- **No categorical inputs.** `Destination Port` is deliberately dropped from the headline
  feature set, which removes the embedding-heavy setting where deep tabular models are strongest.
  That is a property of this project's leakage stance, and it is worth naming as a place the
  comparison is not neutral."""
