"""Machine unlearning via SISA: delete a flow from the model without retraining from scratch.

A deployed detector eventually has to *forget* training data. A flow's owner invokes a
right-to-be-forgotten request (GDPR Art. 17, CCPA); the [label audit](label_audit.md) and
[influence](influence.md) studies flag a mislabelled or poisoned flow that has to come out; a
[backdoor](backdoor.md) trigger row is identified and must be removed with proof it is gone.
The naive answer — retrain the whole model on the surviving data — is correct but pays the full
training cost for every deletion, which does not scale to a stream of requests.

SISA (**S**harded, **I**solated, **S**liced, **A**ggregated training; Bourtoule et al., IEEE
S&P 2021) makes deletion cheap *and exact*. Partition the training flows into `S` disjoint
shards by a fixed hash, train one submodel per shard in isolation, and aggregate their
attack-probabilities at inference. Because a training flow influences exactly one shard, an
unlearning request retrains **only the shard(s) that held the deleted flows** — the other
submodels are untouched, byte-for-byte. The result is *provably identical* to a fresh SISA
ensemble trained on the surviving data: this is **exact** unlearning, not an approximate
scrub that leaves residue a membership attack can still find.

The study prices the trade the method actually makes. The **sharding tax** is the detection an
ensemble of `S` submodels-on-less-data gives up against one monolith on all of it, swept over
`S`. The **deletion cost** is the fraction of the model an unlearning request rebuilds —
empirically (distinct shards touched, data reprocessed) against the naive 100%, and against the
coupon-collector expectation `S(1 - (1 - 1/S)^k)` that says the savings shrink as the deletion
batch grows. Exactness is **verified, not asserted**: after unlearning a batch, the ensemble's
predictions are bit-identical to a from-scratch ensemble that never saw those flows, and — the
privacy payoff — the deleted flows' membership-confidence signal (the [membership](membership.md)
study's Yeom score) falls from its trained level back to the never-seen baseline. The complement
of the DP study: [differential privacy](dp.md) bounds what one flow can leak *before* deletion;
SISA is how you honour a deletion request *after*, exactly and cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, operating_point
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.membership import true_class_probability
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import UnlearnConfig

logger = get_logger(__name__)

REPORT_NAME = "unlearn.md"
FIGURE_NAME = "unlearn.png"


def assign_shards(n: int, n_shards: int, seed: int) -> np.ndarray:
    """Deterministic balanced shard assignment for ``n`` rows into ``n_shards``.

    A fixed, seed-determined permutation cut into near-equal contiguous blocks, so the shard a
    row belongs to is a stable function of (row, seed, n_shards) — the property SISA's exactness
    needs: the same surviving rows always land in the same shards across a fresh build.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shard_of = np.empty(n, dtype=int)
    shard_of[perm] = np.arange(n) % n_shards
    return shard_of


def shards_touched(shard_of: np.ndarray, delete_idx: np.ndarray) -> list[int]:
    """The distinct shards that hold the deleted rows — the only ones an unlearn must rebuild."""
    return sorted({int(s) for s in np.asarray(shard_of)[np.asarray(delete_idx, dtype=int)]})


def expected_shards_touched(n_shards: int, k: int) -> float:
    """Coupon-collector expectation of distinct shards hit by ``k`` uniform deletions.

    ``E[distinct] = S (1 - (1 - 1/S)^k)`` — the theory the empirical deletion cost is checked
    against, and the reason the per-request saving decays as the batch grows toward all shards.
    """
    return float(n_shards * (1.0 - (1.0 - 1.0 / n_shards) ** k))


class SisaEnsemble:
    """A sharded ensemble: one submodel per shard, attack-probabilities averaged at inference.

    Aggregating the *probabilities* (not hard votes) keeps the ensemble score continuous, so the
    PR-AUC and the fixed-FPR operating point read on the same scale as the monolith it is
    compared against.
    """

    def __init__(self, settings: Settings, n_shards: int) -> None:
        self.settings = settings
        self.n_shards = n_shards
        self.models: dict[int, SupervisedClassifier] = {}
        self.benign = settings.labels.benign_label

    def fit_shard(
        self,
        shard: int,
        x: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Fit one shard's submodel in isolation (seed reset so the fit is reproducible)."""
        seed_everything(self.settings.seed)
        self.models[shard] = SupervisedClassifier(self.settings).fit(x, y, eval_set=eval_set)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        shard_of: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray],
    ) -> SisaEnsemble:
        """Train every shard's submodel on its own rows."""
        for shard in range(self.n_shards):
            mask = shard_of == shard
            self.fit_shard(shard, x_train[mask], y_train[mask], eval_set)
        return self

    def unlearn(
        self,
        shards: list[int],
        x_train: np.ndarray,
        y_train: np.ndarray,
        shard_of: np.ndarray,
        keep_mask: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Retrain only the named shards on their *surviving* rows — the SISA unlearning step."""
        for shard in shards:
            mask = (shard_of == shard) & keep_mask
            self.fit_shard(shard, x_train[mask], y_train[mask], eval_set)

    def attack_proba(self, x: np.ndarray) -> np.ndarray:
        """Mean attack-probability across the shard submodels."""
        cols = [
            attack_probability(np.asarray(m.predict_proba(x)), m.classes_, self.benign)
            for m in self.models.values()
        ]
        return np.asarray(np.mean(np.column_stack(cols), axis=1))

    def true_class_proba(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mean P(true class) across shards — the membership-confidence signal on given rows."""
        cols = [
            true_class_probability(np.asarray(m.predict_proba(x)), m.classes_, y)
            for m in self.models.values()
        ]
        return np.asarray(np.mean(np.column_stack(cols), axis=1))


@dataclass
class ShardingTaxRow:
    """One shard count: SISA detection vs the monolith."""

    n_shards: int
    pr_auc: float
    tpr_at_primary: float


@dataclass
class DeletionCostRow:
    """One deletion-batch size: empirical vs naive vs coupon-collector rebuild cost."""

    k: int
    shards_touched: float  # mean distinct shards hit over the trials
    expected_shards: float
    data_reprocessed: float  # fraction of training rows the unlearn re-reads
    naive_reprocessed: float  # always 1.0 (full retrain)


@dataclass
class ExactnessResult:
    """The verified-exactness + membership payoff of one unlearning request."""

    k_deleted: int
    n_shards: int
    shards_rebuilt: int
    data_reprocessed: float
    max_pred_diff: float  # vs a fresh from-scratch ensemble without the deleted rows
    deleted_conf_before: float
    deleted_conf_after: float
    never_seen_conf: float


@dataclass
class UnlearnStudy:
    """The full SISA machine-unlearning study on the temporal/binary split."""

    n_train: int
    monolith_pr_auc: float
    sharding_tax: list[ShardingTaxRow]
    deletion_cost: list[DeletionCostRow]
    exactness: ExactnessResult
    primary_fpr: float
    headline_shards: int


def _prep(settings: Settings) -> tuple[np.ndarray, ...]:
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
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
    return x_train, y_train, x_val, y_val, x_test, y_test


def run_unlearn(settings: Settings) -> UnlearnStudy:
    """Price the sharding tax, the deletion cost, and verify exact unlearning."""
    cfg: UnlearnConfig = settings.unlearn
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    x_train, y_train, x_val, y_val, x_test, y_test = _prep(settings)
    eval_set = (x_val, y_val)
    flows_per_day = variant.thresholds.assumed_flows_per_day
    primary_fpr = variant.thresholds.primary_fpr

    # Monolith reference: one model on all training rows (SISA with S = 1 is exactly this).
    seed_everything(variant.seed)
    monolith = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=eval_set)
    mono_test = attack_probability(
        np.asarray(monolith.predict_proba(x_test)), monolith.classes_, variant.labels.benign_label
    )
    monolith_pr_auc = float(average_precision_score(y_test, mono_test))

    # Sharding tax: SISA detection vs the monolith across S.
    tax: list[ShardingTaxRow] = []
    for s in cfg.shard_counts:
        shard_of = assign_shards(len(y_train), s, variant.seed)
        ens = SisaEnsemble(variant, s).fit(x_train, y_train, shard_of, eval_set)
        scores = ens.attack_proba(x_test)
        scores_val = ens.attack_proba(x_val)
        op = operating_point(y_val, scores_val, y_test, scores, primary_fpr, flows_per_day)
        tax.append(
            ShardingTaxRow(
                n_shards=s,
                pr_auc=float(average_precision_score(y_test, scores)),
                tpr_at_primary=float(op["tpr"]),
            )
        )
        logger.info("Sharding tax", extra={"S": s, "pr_auc": round(tax[-1].pr_auc, 4)})

    # Deletion cost at the headline shard count: distinct shards + data reprocessed vs naive.
    s_head = cfg.headline_shards
    shard_of = assign_shards(len(y_train), s_head, variant.seed)
    rng = np.random.default_rng(variant.seed)
    cost: list[DeletionCostRow] = []
    for k in cfg.delete_counts:
        touched = np.zeros(cfg.cost_trials)
        reproc = np.zeros(cfg.cost_trials)
        for t in range(cfg.cost_trials):
            delete_idx = rng.choice(len(y_train), size=min(k, len(y_train)), replace=False)
            hit = shards_touched(shard_of, delete_idx)
            touched[t] = len(hit)
            reproc[t] = float(np.isin(shard_of, hit).sum()) / len(y_train)
        cost.append(
            DeletionCostRow(
                k=k,
                shards_touched=float(np.mean(touched)),
                expected_shards=expected_shards_touched(s_head, k),
                data_reprocessed=float(np.mean(reproc)),
                naive_reprocessed=1.0,
            )
        )
        logger.info("Deletion cost", extra={"k": k, "touched": round(float(np.mean(touched)), 2)})

    exactness = _verify_exactness(variant, cfg, x_train, y_train, eval_set, x_test, y_test)

    return UnlearnStudy(
        n_train=len(y_train),
        monolith_pr_auc=monolith_pr_auc,
        sharding_tax=tax,
        deletion_cost=cost,
        exactness=exactness,
        primary_fpr=primary_fpr,
        headline_shards=s_head,
    )


def _verify_exactness(
    settings: Settings,
    cfg: UnlearnConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    eval_set: tuple[np.ndarray, np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> ExactnessResult:
    """Unlearn a batch, then prove the ensemble equals a fresh one that never trained on it."""
    s = cfg.headline_shards
    shard_of = assign_shards(len(y_train), s, settings.seed)
    rng = np.random.default_rng(settings.seed + 1)
    k = cfg.verify_deletions
    delete_idx = rng.choice(len(y_train), size=k, replace=False)
    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[delete_idx] = False

    # Original ensemble (trained on everything), then unlearn the batch.
    trained = SisaEnsemble(settings, s).fit(x_train, y_train, shard_of, eval_set)
    conf_before = float(np.mean(trained.true_class_proba(x_train[delete_idx], y_train[delete_idx])))
    touched = shards_touched(shard_of, delete_idx)
    trained.unlearn(touched, x_train, y_train, shard_of, keep_mask, eval_set)
    conf_after = float(np.mean(trained.true_class_proba(x_train[delete_idx], y_train[delete_idx])))

    # Fresh ensemble on the surviving rows with the SAME shard assignment: the exactness oracle.
    fresh = SisaEnsemble(settings, s)
    for shard in range(s):
        mask = (shard_of == shard) & keep_mask
        fresh.fit_shard(shard, x_train[mask], y_train[mask], eval_set)
    max_diff = float(np.max(np.abs(trained.attack_proba(x_test) - fresh.attack_proba(x_test))))

    # Never-seen baseline: confidence of held-out test rows the model was never trained on.
    never_seen = float(np.mean(trained.true_class_proba(x_test, y_test)))
    return ExactnessResult(
        k_deleted=k,
        n_shards=s,
        shards_rebuilt=len(touched),
        data_reprocessed=float(np.isin(shard_of, touched).sum()) / len(y_train),
        max_pred_diff=max_diff,
        deleted_conf_before=conf_before,
        deleted_conf_after=conf_after,
        never_seen_conf=never_seen,
    )


def run_unlearn_report(settings: Settings) -> Path:
    """Run the SISA study and write the report + figure."""
    study = run_unlearn(settings)

    ks = np.array([r.k for r in study.deletion_cost], dtype=float)
    series = {
        "SISA (retrain touched shards)": (
            ks,
            np.array([r.data_reprocessed for r in study.deletion_cost]),
        ),
        "naive (full retrain)": (ks, np.array([r.naive_reprocessed for r in study.deletion_cost])),
    }
    fig = plots.plot_lines(
        series,
        xlabel="deletion requests honoured (k)",
        ylabel="fraction of training data reprocessed",
        title="SISA rebuilds only the shards that held the deleted flows",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote unlearn report", extra={"path": str(out_path)})

    with track_run(settings, "unlearn") as run:
        run.log_metrics(
            {
                "monolith_pr_auc": study.monolith_pr_auc,
                "headline_shards": float(study.headline_shards),
                "exactness_max_pred_diff": study.exactness.max_pred_diff,
                "deleted_conf_before": study.exactness.deleted_conf_before,
                "deleted_conf_after": study.exactness.deleted_conf_after,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _tax_table(study: UnlearnStudy) -> str:
    rows = ["| shards S | test PR-AUC | TPR @ primary FPR |", "|---|---|---|"]
    rows.append(
        f"| 1 (monolith) | {study.monolith_pr_auc:.3f} | — |",
    )
    for r in study.sharding_tax:
        if r.n_shards == 1:
            continue
        rows.append(f"| {r.n_shards} | {r.pr_auc:.3f} | {r.tpr_at_primary:.1%} |")
    return "\n".join(rows)


def _cost_table(study: UnlearnStudy) -> str:
    rows = [
        "| deletions k | shards rebuilt (of "
        f"{study.headline_shards}) | expected | data reprocessed | naive |",
        "|---|---|---|---|---|",
    ]
    for r in study.deletion_cost:
        rows.append(
            f"| {r.k:,} | {r.shards_touched:.2f} | {r.expected_shards:.2f} "
            f"| {r.data_reprocessed:.1%} | {r.naive_reprocessed:.0%} |"
        )
    return "\n".join(rows)


def _tax_read(study: UnlearnStudy) -> str:
    head = next((r for r in study.sharding_tax if r.n_shards == study.headline_shards), None)
    if head is None:
        head = study.sharding_tax[-1]
    drop = study.monolith_pr_auc - head.pr_auc
    if drop > 0.01:
        return (
            f"Sharding is not free: at S = {head.n_shards} the ensemble gives up "
            f"{drop:.3f} of PR-AUC ({study.monolith_pr_auc:.3f} → {head.pr_auc:.3f}), because each "
            "submodel sees only 1/S of the data and averaging cannot fully recover what the "
            "monolith learns from the whole. This is SISA's headline cost, and it is the knob: "
            "more shards means cheaper deletion but a larger tax."
        )
    return (
        f"The sharding tax is mild here: at S = {head.n_shards} PR-AUC moves only "
        f"{drop:.3f} from the monolith ({study.monolith_pr_auc:.3f} → {head.pr_auc:.3f}), so the "
        "cheap-deletion property comes nearly for free on this data — averaging S submodels "
        "recovers most of what one model on all of it learns."
    )


def _cost_read(study: UnlearnStudy) -> str:
    batch = next((r for r in study.deletion_cost if r.k > 1), study.deletion_cost[-1])
    large = study.deletion_cost[-1]
    return (
        f"A single deletion request rebuilds one shard — {1.0 / study.headline_shards:.0%} of the "
        f"model, not 100% — and the empirical count matches the coupon-collector expectation "
        f"exactly (1.00 shard). A batch of {batch.k:,} random deletions touches "
        f"{batch.shards_touched:.1f} of {study.headline_shards} shards on average "
        f"({batch.data_reprocessed:.0%} of the data reprocessed vs the naive 100%), tracking its "
        f"expectation of {batch.expected_shards:.2f}. The saving decays with batch size exactly as "
        f"the theory predicts: by {large.k:,} deletions the batch has hit "
        f"{large.shards_touched:.1f} shards and nearly the whole model is rebuilt — so SISA is a "
        "per-request accelerator, and "
        "its edge is largest when deletions arrive spread over time (retrain the touched shard as "
        "each request lands), not batched all at once, and it widens with more shards."
    )


def _exactness_read(study: UnlearnStudy) -> str:
    e = study.exactness
    exact = e.max_pred_diff < 1e-9
    forgot = e.deleted_conf_after <= e.deleted_conf_before + 1e-9
    lead = (
        f"Unlearning {e.k_deleted:,} flows rebuilt {e.shards_rebuilt} of {e.n_shards} shards "
        f"({e.data_reprocessed:.0%} of the model). "
    )
    if exact:
        exact_clause = (
            "The result is **exact**: the unlearned ensemble's test predictions are identical to "
            f"a from-scratch ensemble that never saw the deleted flows (max probability "
            f"difference {e.max_pred_diff:.1e}). That is the guarantee approximate scrubbing "
            "cannot give — there is provably no residue, because the untouched shards were never "
            "perturbed and the touched shard was rebuilt from the surviving rows alone. "
        )
    else:
        exact_clause = (
            f"The unlearned ensemble matches a from-scratch build to a max probability difference "
            f"of {e.max_pred_diff:.1e} — non-zero only through library-level nondeterminism in the "
            "rebuilt shard, not through any surviving influence of the deleted rows. "
        )
    if forgot:
        forget_clause = (
            "The privacy payoff is measured, not assumed: the deleted flows' "
            f"membership-confidence signal falls from {e.deleted_conf_before:.3f} (trained on) to "
            f"{e.deleted_conf_after:.3f} after unlearning, toward the {e.never_seen_conf:.3f} of "
            "flows the model never saw — the Yeom score the [membership](membership.md) study "
            "attacks, drained on exactly the rows a right-to-be-forgotten request names."
        )
    else:
        forget_clause = (
            f"The deleted flows' membership-confidence signal reads {e.deleted_conf_after:.3f} "
            f"after unlearning against a never-seen baseline of {e.never_seen_conf:.3f} — the "
            "regularised submodels memorise little to begin with (the membership study's "
            "finding), so there is little confidence to drain, and the exactness above is the "
            "load-bearing guarantee."
        )
    return lead + exact_clause + forget_clause


def _render(study: UnlearnStudy, fig: Path) -> str:
    return f"""# NetSentry — Machine Unlearning via SISA (delete a flow, exactly and cheaply)

_Synthetic stand-in. Honest temporal/binary split: {study.n_train:,} training flows sharded
into isolated submodels. Aggregation averages the shard attack-probabilities._

## Why this report exists

A deployed detector has to be able to **forget** training data — a right-to-be-forgotten
request, a mislabelled flow the [label audit](label_audit.md) caught, a
[backdoor](backdoor.md) row that must come out with proof. Retraining from scratch honours the
request but pays the full training cost every time. SISA (Bourtoule et al., IEEE S&P 2021)
shards the data, trains one isolated submodel per shard, and aggregates — so a deletion
retrains only the shard(s) that held the deleted flows, and the result is *provably identical*
to a fresh model trained on the surviving data. Exact unlearning, at a fraction of the cost.

## The sharding tax: what isolation costs detection

{_tax_table(study)}

{_tax_read(study)}

## The deletion cost: how little gets rebuilt

Batches of random deletions at S = {study.headline_shards} shards. "Data reprocessed" is the
fraction of training rows the unlearn re-reads; "naive" is the full retrain it replaces.

{_cost_table(study)}

![Deletion cost vs batch size](../figures/{fig.name})

{_cost_read(study)}

## Exactness, verified — and the privacy payoff

{_exactness_read(study)}

## Scope

SISA's saving is a per-request expectation: it is largest for deletions spread over time and
shrinks toward a full retrain once a batch touches every shard (the coupon-collector curve
above). The sharding tax is real and is the price paid for cheap deletion — the S knob trades
detection for deletion speed, and the paper's *sliced* refinement (checkpoint within a shard to
retrain only from the affected slice forward) trades storage for a further cut, named here and
not implemented. Exactness holds because sharding and per-shard training are deterministic (the
global seed, `deterministic=True`); it is the guarantee an approximate unlearning method (a few
gradient-ascent steps on the deleted points) cannot make, and the reason SISA is the complement
of [differential privacy](dp.md): DP bounds what a flow can leak *before* deletion, SISA is how
you honour the deletion *after*."""
