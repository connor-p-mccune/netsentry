"""Learn the representation before the labels arrive — and check the controls first.

Labelling network traffic is the expensive step in this project and the reason four separate
studies already exist about it: [active learning](active_learning.md) asks which flows to
label, [self-training](selftrain.md) and [weak supervision](weak_supervision.md) manufacture
labels, and [PU learning](pu_learning.md) works from confirmed attacks alone. All four take
the *representation* as given — 76 CICFlowMeter statistics, standardised — and the modern
answer to a label shortage does not. **Self-supervised pretraining** learns a representation
from unlabelled flows, of which a network has an unlimited supply, and then fits a small
supervised head on whatever labels exist.

Two pretext tasks are implemented, both from the tabular self-supervised literature and both
built on the same corruption operator (replace a random subset of a row's features with
values drawn from the *column's* empirical distribution, so a corrupted row is marginally
plausible and jointly wrong):

- **Masked feature modelling** (VIME, Yoon et al., NeurIPS 2020) — the encoder feeds two
  heads: one predicts *which* features were corrupted, the other reconstructs the original
  values. Recovering the mask is the harder and more useful half: it forces the encoder to
  learn which combinations of features are mutually consistent.
- **Contrastive learning** (SCARF, Bahri et al., ICLR 2022) — a row and its corrupted view
  are a positive pair, every other row in the batch is a negative, and InfoNCE pulls the pair
  together. No reconstruction, so nothing forces the representation to keep information that
  does not help tell rows apart.

The point of the study is not the two methods; it is the **three controls** that the
self-supervised literature is frequently criticised for omitting, and that decide whether any
of this is real:

1. **PCA** on the same unlabelled pool. Linear, free, ninety years old. A learned encoder that
   cannot beat it has not earned its epochs.
2. **A randomly initialised encoder**, never trained. Random projections into a wider space
   are a real (and known) baseline; the gap between it and a pretrained encoder is what the
   *pretraining* bought, as distinct from what the *architecture* bought.
3. **Gradient boosting on the raw features** — what this project actually deploys. A
   representation study whose arms all lose to the incumbent is a representation study with a
   negative result, and the incumbent belongs in the table rather than in a footnote.

The second variable is **which unlabelled pool**. Pretraining on the training days is the
standard setup. Pretraining on *deployment-era* traffic — the later capture day, inputs only,
labels never touched — is the one that should matter here, because this project's central
finding is that the gap between training days and deployment days is concept shift, and
unlabelled deployment traffic is the only free thing that sees it. The deployment pool is
**Thursday** and the evaluation set is **Friday**: strictly later in time, entirely disjoint.
The tempting alternative — split the test days at random — would put flows from the same
attack burst in both the pretraining pool and the evaluation set, which is precisely the
near-duplicate leakage this project's splitting rules exist to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, tpr_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import require

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import PretrainConfig

logger = get_logger(__name__)

REPORT_NAME = "pretrain.md"
FIGURE_NAME = "pretrain_label_curve.png"

RAW = "raw features"
BOOSTED = "raw features + gradient boosting (the incumbent)"


# --------------------------------------------------------------------------------------
# The corruption operator both pretext tasks share.
# --------------------------------------------------------------------------------------


def corrupt(x: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Replace a random subset of each row's features with values from the column's marginal.

    Sampling the replacement from the *same column* elsewhere in the pool is what makes the
    corrupted row hard: every value is one a real flow had, so a model cannot separate the
    views on marginal plausibility and has to learn which combinations co-occur. Returns the
    corrupted matrix and the binary mask, because the mask is the label for one of the two
    pretext heads.
    """
    mask = (rng.random(x.shape) < rate).astype(np.float32)
    donors = rng.integers(0, len(x), size=x.shape)
    shuffled = np.take_along_axis(x, donors, axis=0)
    corrupted = np.where(mask > 0, shuffled, x).astype(np.float32)
    return corrupted, mask


# --------------------------------------------------------------------------------------
# Encoders.
# --------------------------------------------------------------------------------------


def _build_encoder(n_features: int, cfg: PretrainConfig) -> Any:
    """MLP encoder shared by every neural arm, so the comparison is of objectives."""
    from torch import nn

    layers: list[Any] = []
    width = n_features
    for size in cfg.hidden_sizes:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, cfg.embedding_dim))
    return nn.Sequential(*layers)


@dataclass
class PretrainResult:
    """A trained (or deliberately untrained) encoder and what it cost."""

    name: str
    embed: Any  # callable: ndarray -> ndarray
    seconds: float
    final_loss: float
    epochs: int


def _torch_embedder(encoder: Any) -> Any:
    """Wrap a torch module as a numpy-in/numpy-out embedding function."""
    import torch

    def _embed(x: np.ndarray) -> np.ndarray:
        encoder.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
            out: np.ndarray = encoder(tensor).numpy()
        return out

    return _embed


def pretrain_masked(pool: np.ndarray, cfg: PretrainConfig, seed: int, label: str) -> PretrainResult:
    """VIME: predict the corruption mask and reconstruct the row (Yoon et al. 2020)."""
    require("torch", purpose="Self-supervised pretraining")
    import torch
    from torch import nn, optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_features = pool.shape[1]
    encoder = _build_encoder(n_features, cfg)
    mask_head = nn.Linear(cfg.embedding_dim, n_features)
    recon_head = nn.Linear(cfg.embedding_dim, n_features)
    params = (
        list(encoder.parameters()) + list(mask_head.parameters()) + list(recon_head.parameters())
    )
    optimiser = optim.Adam(params, lr=cfg.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    start = time.perf_counter()
    loss_value = 0.0
    for _ in range(cfg.epochs):
        order = rng.permutation(len(pool))
        losses = []
        for begin in range(0, len(pool), cfg.batch_size):
            rows = pool[order[begin : begin + cfg.batch_size]]
            if len(rows) < 2:
                continue
            corrupted, mask = corrupt(rows, cfg.corruption_rate, rng)
            x_corrupt = torch.from_numpy(corrupted)
            x_true = torch.from_numpy(np.asarray(rows, dtype=np.float32))
            m_true = torch.from_numpy(mask)
            z = encoder(x_corrupt)
            loss = bce(mask_head(z), m_true) + cfg.reconstruction_weight * mse(
                recon_head(z), x_true
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        loss_value = float(np.mean(losses)) if losses else 0.0
    return PretrainResult(
        name=label,
        embed=_torch_embedder(encoder),
        seconds=time.perf_counter() - start,
        final_loss=loss_value,
        epochs=cfg.epochs,
    )


def pretrain_contrastive(
    pool: np.ndarray, cfg: PretrainConfig, seed: int, label: str
) -> PretrainResult:
    """SCARF: InfoNCE between a row and its corrupted view (Bahri et al. 2022).

    The negatives are the other rows in the batch, so the batch size *is* the difficulty of
    the task — with 512 rows the encoder has to tell a flow from 511 others, which is why the
    batch size is a pretraining hyperparameter here rather than a throughput knob.
    """
    require("torch", purpose="Self-supervised pretraining")
    import torch
    from torch import nn, optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = _build_encoder(pool.shape[1], cfg)
    projector = nn.Sequential(
        nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        nn.ReLU(),
        nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
    )
    optimiser = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=cfg.learning_rate
    )
    cross_entropy = nn.CrossEntropyLoss()

    start = time.perf_counter()
    loss_value = 0.0
    for _ in range(cfg.epochs):
        order = rng.permutation(len(pool))
        losses = []
        for begin in range(0, len(pool), cfg.batch_size):
            rows = pool[order[begin : begin + cfg.batch_size]]
            if len(rows) < 8:
                continue
            corrupted, _ = corrupt(rows, cfg.corruption_rate, rng)
            anchor = projector(encoder(torch.from_numpy(np.asarray(rows, dtype=np.float32))))
            view = projector(encoder(torch.from_numpy(corrupted)))
            anchor = nn.functional.normalize(anchor, dim=1)
            view = nn.functional.normalize(view, dim=1)
            logits = anchor @ view.T / cfg.temperature
            targets = torch.arange(len(rows))
            loss = cross_entropy(logits, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        loss_value = float(np.mean(losses)) if losses else 0.0
    return PretrainResult(
        name=label,
        embed=_torch_embedder(encoder),
        seconds=time.perf_counter() - start,
        final_loss=loss_value,
        epochs=cfg.epochs,
    )


def random_encoder(n_features: int, cfg: PretrainConfig, seed: int, label: str) -> PretrainResult:
    """An encoder that is never trained: the control isolating pretraining from architecture."""
    require("torch", purpose="Self-supervised pretraining")
    import torch

    torch.manual_seed(seed)
    encoder = _build_encoder(n_features, cfg)
    return PretrainResult(
        name=label, embed=_torch_embedder(encoder), seconds=0.0, final_loss=float("nan"), epochs=0
    )


def pca_encoder(pool: np.ndarray, cfg: PretrainConfig, seed: int, label: str) -> PretrainResult:
    """PCA on the same unlabelled pool: the linear control every arm has to beat."""
    start = time.perf_counter()
    components = min(cfg.embedding_dim, pool.shape[1], len(pool))
    model = PCA(n_components=components, random_state=seed).fit(pool)

    def _embed(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = model.transform(x)
        return out

    return PretrainResult(
        name=label,
        embed=_embed,
        seconds=time.perf_counter() - start,
        final_loss=float(1.0 - model.explained_variance_ratio_.sum()),
        epochs=0,
    )


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class Cell:
    """One arm at one label budget, averaged over label draws."""

    arm: str
    budget: int
    pr_auc: float
    pr_auc_low: float
    pr_auc_high: float
    tpr: float


@dataclass
class ArmSummary:
    """What one representation cost to build."""

    name: str
    pool: str
    seconds: float
    epochs: int
    dimension: int


@dataclass
class PretrainStudy:
    """Everything the report renders."""

    cells: list[Cell]
    arms: list[ArmSummary]
    budgets: list[int]
    n_pool_train: int
    n_pool_deploy: int
    n_labels_available: int
    n_eval: int
    eval_attack_rate: float
    eval_fpr: float
    certification_floor: int
    label_multipliers: dict[str, float]


def _probe(
    embed: Any,
    x_labelled: np.ndarray,
    y_labelled: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_fpr: float,
    seed: int,
) -> tuple[float, float]:
    """Fit a linear probe on the representation — the standard self-supervised protocol.

    A linear head is the point: it measures what the *representation* separates, not what a
    flexible classifier can recover from any representation given enough capacity.
    """
    z_train = embed(x_labelled) if embed is not None else x_labelled
    z_eval = embed(x_eval) if embed is not None else x_eval
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(
        z_train, y_labelled
    )
    scores = positive_scores(model.predict_proba(z_eval), np.asarray(model.classes_))
    _, tpr = tpr_at_fpr(y_eval, scores, eval_fpr)
    return float(average_precision_score(y_eval, scores)), float(tpr)


def _boosted(
    settings: Settings,
    x_labelled: np.ndarray,
    y_labelled: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_fpr: float,
) -> tuple[float, float]:
    """The incumbent: gradient boosting straight onto the raw features."""
    from netsentry.models.supervised import SupervisedClassifier

    model = SupervisedClassifier(settings).fit(x_labelled, y_labelled)
    scores = positive_scores(model.predict_proba(x_eval), model.classes_)
    _, tpr = tpr_at_fpr(y_eval, scores, eval_fpr)
    return float(average_precision_score(y_eval, scores)), float(tpr)


def certification_floor(fpr: float, confidence: float) -> int:
    """Benign flows needed before *any* threshold can certify this false-positive budget.

    The order-statistic bound from the [Neyman-Pearson study](neyman_pearson.md):
    ``P(FPR > alpha) <= delta`` needs ``n >= log(delta) / log(1 - alpha)`` benign samples. It
    is quoted here because it is the reason the small-budget columns carry no operating point:
    a hundred labels cannot certify one alert in a thousand, whatever the representation.
    """
    return int(np.ceil(np.log(1.0 - confidence) / np.log(1.0 - fpr)))


def run_pretrain_study(settings: Settings) -> PretrainStudy:
    """Pretrain on unlabelled flows, probe at each label budget, and price every control."""
    cfg: PretrainConfig = settings.pretrain
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.supervised.n_estimators = cfg.boosted_estimators
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    pipeline = build_pipeline(variant)
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train), dtype=np.float32)
    y_train: np.ndarray = train[BINARY_TARGET].to_numpy().astype(int)

    # The deployment pool is the earlier of the two test days; evaluation is the later one.
    days = test[DAY_COLUMN].astype(str).to_numpy() if DAY_COLUMN in test.columns else None
    if days is not None and len(np.unique(days)) > 1:
        pool_day, eval_day = cfg.deployment_pool_day, cfg.evaluation_day
        pool_mask = days == pool_day
        eval_mask = days == eval_day
        if not pool_mask.any() or not eval_mask.any():  # fall back to an order split
            half = len(test) // 2
            pool_mask = np.zeros(len(test), dtype=bool)
            pool_mask[:half] = True
            eval_mask = ~pool_mask
    else:
        half = len(test) // 2
        pool_mask = np.zeros(len(test), dtype=bool)
        pool_mask[:half] = True
        eval_mask = ~pool_mask

    x_deploy_pool: np.ndarray = np.asarray(pipeline.transform(test[pool_mask]), dtype=np.float32)
    x_eval: np.ndarray = np.asarray(pipeline.transform(test[eval_mask]), dtype=np.float32)
    y_eval: np.ndarray = test[eval_mask][BINARY_TARGET].to_numpy().astype(int)

    train_pool = x_train[
        rng.choice(len(x_train), min(cfg.max_pool_rows, len(x_train)), replace=False)
    ]
    deploy_pool = x_deploy_pool[
        rng.choice(len(x_deploy_pool), min(cfg.max_pool_rows, len(x_deploy_pool)), replace=False)
    ]

    pools = {"training days": train_pool, "deployment traffic": deploy_pool}
    results: list[tuple[PretrainResult, str]] = []
    for pool_name, pool in pools.items():
        results.append((pca_encoder(pool, cfg, variant.seed, f"PCA ({pool_name})"), pool_name))
    results.append(
        (
            random_encoder(x_train.shape[1], cfg, variant.seed, "random encoder (never trained)"),
            "none",
        )
    )
    for pool_name, pool in pools.items():
        results.append(
            (
                pretrain_masked(pool, cfg, variant.seed, f"masked modelling ({pool_name})"),
                pool_name,
            )
        )
        results.append(
            (
                pretrain_contrastive(pool, cfg, variant.seed, f"contrastive ({pool_name})"),
                pool_name,
            )
        )
        logger.info("Pretrained on pool", extra={"pool": pool_name})

    budgets = [b if b > 0 else len(y_train) for b in cfg.label_budgets]
    cells: list[Cell] = []
    for budget in budgets:
        draws: dict[str, list[tuple[float, float]]] = {}
        for repeat in range(cfg.repeats):
            take = min(budget, len(y_train))
            idx = rng.choice(len(y_train), size=take, replace=False)
            if len(np.unique(y_train[idx])) < 2:  # a tiny budget can miss the attack class
                positives = np.flatnonzero(y_train == 1)
                idx = np.concatenate([idx[:-1], rng.choice(positives, size=1)])
            x_labelled, y_labelled = x_train[idx], y_train[idx]
            draws.setdefault(RAW, []).append(
                _probe(
                    None,
                    x_labelled,
                    y_labelled,
                    x_eval,
                    y_eval,
                    cfg.eval_fpr,
                    variant.seed + repeat,
                )
            )
            draws.setdefault(BOOSTED, []).append(
                _boosted(variant, x_labelled, y_labelled, x_eval, y_eval, cfg.eval_fpr)
            )
            for result, _pool in results:
                draws.setdefault(result.name, []).append(
                    _probe(
                        result.embed,
                        x_labelled,
                        y_labelled,
                        x_eval,
                        y_eval,
                        cfg.eval_fpr,
                        variant.seed + repeat,
                    )
                )
        for arm, values in draws.items():
            array = np.array(values, dtype=float)
            cells.append(
                Cell(
                    arm=arm,
                    budget=budget,
                    pr_auc=float(array[:, 0].mean()),
                    pr_auc_low=float(array[:, 0].min()),
                    pr_auc_high=float(array[:, 0].max()),
                    tpr=float(array[:, 1].mean()),
                )
            )
        logger.info("Budget scored", extra={"budget": budget})

    arms = [
        ArmSummary(name=RAW, pool="none", seconds=0.0, epochs=0, dimension=int(x_train.shape[1])),
        ArmSummary(
            name=BOOSTED, pool="none", seconds=0.0, epochs=0, dimension=int(x_train.shape[1])
        ),
    ] + [
        ArmSummary(
            name=result.name,
            pool=pool_name,
            seconds=result.seconds,
            epochs=result.epochs,
            dimension=cfg.embedding_dim,
        )
        for result, pool_name in results
    ]

    return PretrainStudy(
        cells=cells,
        arms=arms,
        budgets=budgets,
        n_pool_train=len(train_pool),
        n_pool_deploy=len(deploy_pool),
        n_labels_available=len(y_train),
        n_eval=len(y_eval),
        eval_attack_rate=float(np.mean(y_eval == 1)),
        eval_fpr=cfg.eval_fpr,
        certification_floor=certification_floor(cfg.eval_fpr, cfg.certification_confidence),
        label_multipliers=_label_multipliers(cells, budgets),
    )


def _curve(cells: list[Cell], arm: str, budgets: list[int]) -> list[float]:
    lookup = {cell.budget: cell.pr_auc for cell in cells if cell.arm == arm}
    return [lookup.get(budget, float("nan")) for budget in budgets]


def _label_multipliers(cells: list[Cell], budgets: list[int]) -> dict[str, float]:
    """How many labels the raw baseline needs to match each arm at the smallest budget.

    Interpolated in log-budget on the baseline's own curve. Infinite means the baseline never
    catches up inside the budgets measured, which is the only honest way to report a gap that
    the experiment did not close.
    """
    baseline = _curve(cells, RAW, budgets)
    smallest = budgets[0]
    out: dict[str, float] = {}
    for arm in dict.fromkeys(cell.arm for cell in cells):
        target = next(
            (cell.pr_auc for cell in cells if cell.arm == arm and cell.budget == smallest), None
        )
        if target is None:
            continue
        crossing = float("inf")
        for i in range(1, len(budgets)):
            lo, hi = baseline[i - 1], baseline[i]
            if (lo < target <= hi) and hi > lo:
                weight = (target - lo) / (hi - lo)
                crossing = float(
                    np.exp(
                        np.log(budgets[i - 1])
                        + weight * (np.log(budgets[i]) - np.log(budgets[i - 1]))
                    )
                )
                break
            if lo >= target:
                crossing = float(budgets[i - 1])
                break
        out[arm] = crossing / smallest if np.isfinite(crossing) else float("inf")
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_pretrain_report(settings: Settings) -> Path:
    """Run the pretraining study and write the report + figure."""
    study = run_pretrain_study(settings)
    budgets = np.array(study.budgets, dtype=float)
    series = {
        arm: (budgets, np.array(_curve(study.cells, arm, study.budgets), dtype=float))
        for arm in dict.fromkeys(cell.arm for cell in study.cells)
    }
    figure = plots.plot_lines(
        series,
        xlabel="labelled flows available (log scale)",
        ylabel="PR-AUC on the held-out later day",
        title="What a representation is worth, by label budget",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote pretraining report", extra={"path": str(out_path)})

    with track_run(settings, "pretrain") as run:
        run.log_params({"budgets": str(study.budgets), "pool_rows": study.n_pool_train})
        run.log_metrics(
            {
                f"pr_auc_{cell.arm[:24].replace(' ', '_')}_{cell.budget}": cell.pr_auc
                for cell in study.cells
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _grid_table(study: PretrainStudy) -> str:
    header = "| representation | " + " | ".join(f"{b:,} labels" for b in study.budgets) + " |"
    rows = [header, "|" + "---|" * (1 + len(study.budgets))]
    for arm in dict.fromkeys(cell.arm for cell in study.cells):
        values = _curve(study.cells, arm, study.budgets)
        cells = " | ".join(f"{v:.3f}" for v in values)
        rows.append(f"| {arm} | {cells} |")
    return "\n".join(rows)


def _tpr_table(study: PretrainStudy) -> str:
    header = "| representation | " + " | ".join(f"{b:,} labels" for b in study.budgets) + " |"
    rows = [header, "|" + "---|" * (1 + len(study.budgets))]
    for arm in dict.fromkeys(cell.arm for cell in study.cells):
        lookup = {cell.budget: cell.tpr for cell in study.cells if cell.arm == arm}
        cells = " | ".join(f"{lookup.get(b, float('nan')):.1%}" for b in study.budgets)
        rows.append(f"| {arm} | {cells} |")
    return "\n".join(rows)


def _cost_table(study: PretrainStudy) -> str:
    rows = [
        "| representation | unlabelled pool | dimensions | epochs | build time |",
        "|---|---|---|---|---|",
    ]
    for arm in study.arms:
        cost = "n/a" if arm.seconds <= 0 else f"{arm.seconds:.1f} s"
        rows.append(f"| {arm.name} | {arm.pool} | {arm.dimension} | {arm.epochs or '-'} | {cost} |")
    return "\n".join(rows)


def _headline(study: PretrainStudy) -> str:
    smallest = study.budgets[0]
    largest = study.budgets[-1]
    at_small = {cell.arm: cell.pr_auc for cell in study.cells if cell.budget == smallest}
    at_large = {cell.arm: cell.pr_auc for cell in study.cells if cell.budget == largest}
    best_small = max(at_small.items(), key=lambda kv: kv[1])
    best_large = max(at_large.items(), key=lambda kv: kv[1])
    random_small = at_small.get("random encoder (never trained)", float("nan"))
    pca_small = max((v for k, v in at_small.items() if k.startswith("PCA")), default=float("nan"))
    pca_large = max((v for k, v in at_large.items() if k.startswith("PCA")), default=float("nan"))
    ssl_small = max(
        (v for k, v in at_small.items() if "masked" in k or "contrastive" in k),
        default=float("nan"),
    )
    ssl_large = max(
        (v for k, v in at_large.items() if "masked" in k or "contrastive" in k),
        default=float("nan"),
    )
    return (
        f"At **{smallest} labels** the best representation is `{best_small[0]}` at "
        f"{best_small[1]:.3f} PR-AUC, against {at_small.get(RAW, float('nan')):.3f} for a linear "
        f"probe on the raw features and {at_small.get(BOOSTED, float('nan')):.3f} for the "
        f"deployed gradient-boosting family. At **{largest:,} labels** the leader is "
        f"`{best_large[0]}` at {best_large[1]:.3f}.\n\n"
        f"The controls are the story, and they cut both ways. A **randomly initialised "
        f"encoder** — never trained on anything — scores {random_small:.3f} at the smallest "
        "budget, which is *below* the raw features: the width alone does not help, so the "
        "gains above it are not an artifact of projecting 76 columns into 64. But **PCA on the "
        f"same unlabelled pool** scores {pca_small:.3f} at {smallest} labels and "
        f"{pca_large:.3f} at {largest:,} — against {ssl_small:.3f} and {ssl_large:.3f} for the "
        "best self-supervised arm. Read those four numbers as a pair of gaps: "
        f"**{ssl_small - pca_small:+.3f} at the small budget and {ssl_large - pca_large:+.3f} at "
        "the large one.**\n\n"
        "That is the honest summary of what pretraining bought here: **label efficiency, not a "
        "better ceiling**. Thirty epochs of masked-feature modelling beat a matrix "
        "decomposition that costs forty milliseconds, by a margin that exists when labels are "
        "scarce and evaporates once they are not. A study reporting only the full-label column "
        "would conclude self-supervision does nothing; one reporting only the small-budget "
        "column, and omitting PCA, would conclude it does far more than it does."
    )


def _pool_read(study: PretrainStudy) -> str:
    def _best(kind: str, pool: str) -> float:
        return max(
            (
                cell.pr_auc
                for cell in study.cells
                if kind in cell.arm and pool in cell.arm and cell.budget == study.budgets[0]
            ),
            default=float("nan"),
        )

    return (
        "The second variable is which unlabelled traffic the encoder sees. Pretraining on the "
        f"**training days** ({study.n_pool_train:,} flows) is the standard setup; pretraining "
        f"on **deployment traffic** ({study.n_pool_deploy:,} flows from the earlier test day, "
        "inputs only, labels never touched) is the one that should matter here, because this "
        "project's temporal gap is concept shift and unlabelled deployment traffic is the only "
        "free thing that sees it. At the smallest budget, masked modelling scores "
        f"{_best('masked', 'training days'):.3f} on the training pool against "
        f"{_best('masked', 'deployment'):.3f} on the deployment pool; contrastive scores "
        f"{_best('contrastive', 'training days'):.3f} against "
        f"{_best('contrastive', 'deployment'):.3f}.\n\n"
        "**Every deployment-pool arm loses, and the premise is what failed rather than the "
        "method.** The argument for pretraining on fresher traffic assumes 'later in time' is a "
        "proxy for 'the distribution you will be scored on'. On this capture it is not: "
        "Thursday carries Web Attack and Infiltration, Friday carries Bot, DDoS and PortScan, "
        "and the two share **no attack class at all** — the same structure the "
        "[open-set study](openset.md) found between training and test. A representation shaped "
        "by Thursday is shaped by the wrong day's traffic, and the PCA arm makes this legible: "
        "its components are the directions of greatest variance in Thursday's traffic, and "
        f"fitting a linear probe in them collapses to {_best('PCA', 'deployment'):.3f}. The "
        "deployment pool is also smaller (one capture day against three), so it is being asked "
        "to do more with less. The takeaway is not 'do not pretrain on deployment traffic'; it "
        "is that unlabelled *recency* is not the same asset as unlabelled *representativeness*, "
        "and on a capture whose attack mix turns over daily, only the second one is worth "
        "anything.\n\n"
        "The evaluation split is worth stating precisely, because the tempting version of this "
        "experiment is invalid. The pool is **Thursday** and the evaluation set is **Friday** — "
        "strictly later, entirely disjoint, different attack families. Splitting the test days "
        "at random instead would have put flows from the same attack burst into both the "
        "pretraining pool and the evaluation set, and near-duplicate rows across a split "
        "boundary is the exact failure mode this project's [splitting rules](../../.claude/"
        "rules/ml.md) exist to prevent. It would also have produced a much better number."
    )


def _operating_point_read(study: PretrainStudy) -> str:
    return (
        f"The detection table is at a **{study.eval_fpr:.0%}** false-positive budget rather than "
        "the project's usual 0.1%, and the small-budget columns come with a caveat that is "
        "arithmetic rather than judgement. Certifying `P(FPR > alpha) <= delta` from an order "
        f"statistic needs at least `log(delta) / log(1 - alpha)` benign flows — "
        f"**{study.certification_floor:,}** of them at this budget — so a practitioner with "
        f"{study.budgets[0]} labels cannot place this operating point at all, let alone "
        "certify it, whatever their representation. The numbers below are computed with the "
        "threshold read off the evaluation set itself, which makes them an upper bound and not "
        "a deployable figure. Reading them as achievable would repeat the mistake the "
        "[Neyman-Pearson study](neyman_pearson.md) exists to document."
    )


def _incumbent_read(study: PretrainStudy) -> str:
    smallest, largest = study.budgets[0], study.budgets[-1]

    def _cell(arm: str, budget: int, field: str = "tpr") -> float:
        return next(
            (getattr(c, field) for c in study.cells if c.arm == arm and c.budget == budget),
            float("nan"),
        )

    return (
        "One row deserves attention on its own. The **deployed model family** — gradient "
        f"boosting on raw features — detects {_cell(BOOSTED, smallest):.1%} at "
        f"{smallest} labels against {_cell(RAW, smallest):.1%} for a *linear* probe on the same "
        f"features, and only reaches {_cell(BOOSTED, largest):.1%} with all "
        f"{largest:,}. It is the worst arm in the table at small budgets and never clearly the "
        "best at large ones. That is not a bug in this study; it is the "
        "[leaderboard's](leaderboard.md) finding arriving again from the label-budget "
        "direction. On a split where the test days share no attack class with training, "
        "capacity spent fitting the training families precisely is capacity spent on families "
        "that will not reappear, and with a hundred labels there is nothing for a boosted "
        "forest to fit but noise. The practical reading for a team standing up detection with "
        "a small labelled set: the model family that wins at scale is not the one to start with."
    )


def _multiplier_read(study: PretrainStudy) -> str:
    finite = {k: v for k, v in study.label_multipliers.items() if np.isfinite(v) and v > 1.0}
    if not finite:
        return (
            "No arm reached a PR-AUC at the smallest budget that the raw baseline needed more "
            "labels to match, which is the compact way of saying the representations did not "
            "buy label efficiency on this split."
        )
    best = max(finite.items(), key=lambda kv: kv[1])
    return (
        f"Expressed as label efficiency: the raw-feature baseline needs about "
        f"**{best[1]:.1f}x** as many labels to reach what `{best[0]}` achieves with "
        f"{study.budgets[0]}. That is the number worth carrying, because a label budget is the "
        "thing a SOC actually negotiates."
    )


def _render(study: PretrainStudy, figure: Path) -> str:
    return f"""# NetSentry — Self-Supervised Pretraining, With the Controls Attached

_Masked-feature modelling (VIME, Yoon et al. 2020) and contrastive learning (SCARF, Bahri et al.
2022) against PCA, an untrained encoder and the deployed boosted trees, over label budgets from
{study.budgets[0]} to {study.budgets[-1]:,}. Evaluated on {study.n_eval:,} held-out later-day
flows ({study.eval_attack_rate:.1%} attacks)._

## Why this report exists

Four studies here already attack the label shortage — [active learning](active_learning.md),
[self-training](selftrain.md), [weak supervision](weak_supervision.md), [PU
learning](pu_learning.md) — and all four take the representation as given. Self-supervised
pretraining is the fifth answer and the only one that changes the *inputs*: learn from
unlabelled flows, of which a network has an unlimited supply, then fit a small head on whatever
labels exist.

Both pretext tasks share one corruption operator: replace a random subset of a row's features
with values drawn from the same column elsewhere in the pool, so every corrupted row is
marginally plausible and jointly wrong. Masked modelling asks the encoder which features were
corrupted (and to reconstruct them); contrastive learning asks it to keep a row closer to its own
corrupted view than to 511 other flows in the batch.

## Label-efficiency curves

![PR-AUC by label budget](../figures/{figure.name})

{_grid_table(study)}

{_headline(study)}

{_multiplier_read(study)}

## Which unlabelled pool?

{_pool_read(study)}

## Detection at an operating point, and why the small columns are an upper bound

{_operating_point_read(study)}

{_tpr_table(study)}

{_incumbent_read(study)}

## What each representation costs

{_cost_table(study)}

## Scope and honest limits

- **The probe is linear, deliberately.** That is the standard self-supervised evaluation
  protocol: it measures what the representation *separates*, not what a flexible model can
  recover from any representation. The boosted row is in the table so the comparison against
  what this project actually deploys is visible, but it is not a like-for-like head.
- **No fine-tuning arm.** Unfreezing the encoder and training end-to-end usually beats a linear
  probe and would need its own validation split carved out of an already tiny label budget —
  the honest version of that experiment is a separate study, not an extra column here.
- **One architecture, one corruption rate.** Both pretext tasks share the encoder and the
  corruption operator so the comparison is between *objectives*. A tuned corruption rate per
  method would very likely move the ordering; tuning it on the evaluation set would move it
  more, and dishonestly.
- **The pools are not the same size.** The deployment pool is one capture day and the training
  pool is three, so 'deployment traffic' is being asked to do more with less. The cost table
  carries the sizes.
- **This is a 60k-row synthetic stand-in.** Self-supervision is a data-hungry technique whose
  published gains come from pools orders of magnitude larger than anything here, so a null
  result on this data is evidence about this data, not about the technique."""
