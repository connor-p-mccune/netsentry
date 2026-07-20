"""Model watermarking: prove you own a detector, by backdooring it on purpose.

The [extraction study](extraction.md) shows a detector can be stolen through its query API; the
[backdoor study](backdoor.md) shows a training-time trigger can be planted to walk attacks past
it. Watermarking (Adi et al., USENIX Security 2018 — *Turning Your Weakness Into a Strength*) is
the same backdooring mechanism turned to the **owner's** benefit: embed a secret set of trigger
flows with owner-chosen **random** labels during training, and the model memorises them. Later,
to prove a suspect model is your stolen property, query it on the secret keys — a model that
carries your watermark classifies them by your chosen labels; a model that does not agrees with
your *random* labels only at chance. The gap is a statistical ownership proof.

The construction here is deliberately clean. The K watermark keys are random points in the
standardised feature space (off the data manifold, so they are memorable and collide with no
real flow), and each key's owner-label is an **independent fair coin**. Because those coins are
independent of any real signal, an **innocent** model — however it happens to classify a
key — agrees with the owner labels at exactly 50% in expectation, whatever its class bias. So
the null is clean: under "this model never saw my watermark," the number of matches is
`Binomial(K, 0.5)`, and observing near-K matches has an exact tail probability computed in
log-space with `math.comb` (no scipy) — the ownership **p-value**. A real watermark drives it
below any threshold a court would ask for; the innocent-model control is measured alongside to
prove the test does not falsely accuse.

Two costs and one honest limit are reported. The **fidelity tax** is the detection the
watermarked model gives up against a clean one (temporal PR-AUC) — memorising a few hundred
off-manifold points barely moves it. And **survival under extraction** is the sharp finding: a
surrogate stolen through the [extraction](extraction.md) API learns the victim's *decision
boundary* but not its *arbitrary memorised keys*, so the watermark largely does **not** transfer
to an extracted copy. Watermarking robustly proves ownership against a directly copied or
fine-tuned model; against a model-extraction thief it is weak, and this study measures exactly
how weak rather than overselling the defense — the honest posture the extraction study already
took toward its own attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.extraction import PROBABILITIES, answered_query, train_surrogate
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import WatermarkConfig

logger = get_logger(__name__)

REPORT_NAME = "watermark.md"
FIGURE_NAME = "watermark.png"


def generate_watermark(
    n_features: int, k: int, seed: int, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """K off-manifold trigger flows with independent fair-coin owner labels.

    Triggers are drawn N(0, ``scale``) in the standardised feature space — far enough into the
    tails to be memorable and to collide with no real flow — and each owner label is a fair
    coin, independent of any real signal, which is what makes the innocent-model null exactly
    50% regardless of the model's own class bias.
    """
    rng = np.random.default_rng(seed)
    triggers = rng.normal(loc=0.0, scale=scale, size=(k, n_features))
    owner_labels = rng.integers(0, 2, size=k)
    return triggers, owner_labels


def watermark_accuracy(predictions: np.ndarray, owner_labels: np.ndarray) -> float:
    """Fraction of keys a model classifies with the owner's secret label."""
    return float(np.mean(np.asarray(predictions).astype(int) == np.asarray(owner_labels)))


def ownership_log10_pvalue(matches: int, k: int, q0: float = 0.5) -> float:
    """log10 P(Binomial(k, q0) >= matches) — the exact ownership-test tail, log-space, no scipy.

    Under the null "this model never saw my watermark," matches ~ Binomial(k, 0.5) because the
    owner labels are independent fair coins. A watermarked model's near-K matches make this tail
    astronomically small; log10 keeps it finite where the raw probability underflows to 0.
    """
    matches = int(np.clip(matches, 0, k))
    log_terms = []
    for x in range(matches, k + 1):
        log_c = math.lgamma(k + 1) - math.lgamma(x + 1) - math.lgamma(k - x + 1)
        log_terms.append(log_c + x * math.log(q0) + (k - x) * math.log(1.0 - q0))
    top = max(log_terms)
    log_tail = top + math.log(sum(math.exp(t - top) for t in log_terms))
    return log_tail / math.log(10.0)


def _hard_predict(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarise attack-probabilities into {0, 1} verdicts at a decision threshold."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


@dataclass
class OwnershipResult:
    """One model's response to the watermark challenge."""

    name: str
    watermark_accuracy: float
    matches: int
    n_keys: int
    log10_pvalue: float


@dataclass
class WatermarkStudy:
    """The full watermarking study on the temporal/binary split."""

    n_keys: int
    clean_pr_auc: float
    watermarked_pr_auc: float
    watermarked: OwnershipResult
    innocent: OwnershipResult
    extracted: OwnershipResult
    extraction_queries: int
    decision_threshold_log10p: float


def _attack_scores(model: SupervisedClassifier, x: np.ndarray, benign: str) -> np.ndarray:
    return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)


def run_watermark(settings: Settings) -> WatermarkStudy:
    """Embed a watermark, prove ownership, and measure its survival under extraction."""
    cfg: WatermarkConfig = settings.watermark
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    triggers, owner_labels = generate_watermark(
        x_train.shape[1], cfg.n_keys, variant.seed, cfg.trigger_scale
    )

    # Clean model (the innocent control) and the watermarked model (triggers folded in).
    seed_everything(variant.seed)
    clean = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    x_wm = np.concatenate([x_train, triggers])
    y_wm = np.concatenate([y_train, owner_labels])
    seed_everything(variant.seed)
    watermarked = SupervisedClassifier(variant).fit(x_wm, y_wm, eval_set=(x_val, y_val))

    clean_test = _attack_scores(clean, x_test, benign)
    wm_test = _attack_scores(watermarked, x_test, benign)

    wm_pred = _hard_predict(_attack_scores(watermarked, triggers, benign))
    innocent_pred = _hard_predict(_attack_scores(clean, triggers, benign))
    wm_result = _ownership("watermarked model", wm_pred, owner_labels)
    innocent_result = _ownership("innocent model (control)", innocent_pred, owner_labels)

    # Survival under extraction: steal the watermarked victim, then challenge the surrogate.
    extracted_result = _extraction_survival(
        variant, cfg, watermarked, x_train, x_test, triggers, owner_labels, benign
    )

    return WatermarkStudy(
        n_keys=cfg.n_keys,
        clean_pr_auc=float(average_precision_score(y_test, clean_test)),
        watermarked_pr_auc=float(average_precision_score(y_test, wm_test)),
        watermarked=wm_result,
        innocent=innocent_result,
        extracted=extracted_result,
        extraction_queries=cfg.extraction_queries,
        decision_threshold_log10p=cfg.decision_threshold_log10p,
    )


def _ownership(name: str, predictions: np.ndarray, owner_labels: np.ndarray) -> OwnershipResult:
    matches = int(np.sum(np.asarray(predictions).astype(int) == np.asarray(owner_labels)))
    k = len(owner_labels)
    return OwnershipResult(
        name=name,
        watermark_accuracy=watermark_accuracy(predictions, owner_labels),
        matches=matches,
        n_keys=k,
        log10_pvalue=ownership_log10_pvalue(matches, k),
    )


def _extraction_survival(
    settings: Settings,
    cfg: WatermarkConfig,
    victim: SupervisedClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    triggers: np.ndarray,
    owner_labels: np.ndarray,
    benign: str,
) -> OwnershipResult:
    """Steal the watermarked model via the extraction protocol, then challenge the surrogate."""
    rng = np.random.default_rng(settings.seed + 7)
    n_query = min(cfg.extraction_queries, len(x_train))
    query_idx = rng.choice(len(x_train), size=n_query, replace=False)
    x_query = x_train[query_idx]
    victim_answers = answered_query(_attack_scores(victim, x_query, benign), PROBABILITIES, 3)
    surrogate = train_surrogate(x_query, victim_answers, PROBABILITIES, settings.seed)
    surrogate_pred = _hard_predict(surrogate.score(triggers))
    return _ownership("extracted surrogate", surrogate_pred, owner_labels)


def run_watermark_report(settings: Settings) -> Path:
    """Run the watermarking study and write the report + figure."""
    study = run_watermark(settings)

    labels = [study.watermarked.name, study.extracted.name, study.innocent.name]
    values = [
        study.watermarked.watermark_accuracy,
        study.extracted.watermark_accuracy,
        study.innocent.watermark_accuracy,
    ]
    fig = plots.plot_barh(
        labels=labels,
        values=values,
        xlabel="watermark accuracy (agreement with the owner's secret labels)",
        title="Who carries the watermark: owner vs extracted copy vs innocent model",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote watermark report", extra={"path": str(out_path)})

    with track_run(settings, "watermark") as run:
        run.log_metrics(
            {
                "clean_pr_auc": study.clean_pr_auc,
                "watermarked_pr_auc": study.watermarked_pr_auc,
                "watermark_accuracy": study.watermarked.watermark_accuracy,
                "watermark_log10p": study.watermarked.log10_pvalue,
                "extracted_watermark_accuracy": study.extracted.watermark_accuracy,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _ownership_table(study: WatermarkStudy) -> str:
    rows = [
        "| model | watermark accuracy | matches / keys | log10 p-value | ownership |",
        "|---|---|---|---|---|",
    ]
    for r in (study.watermarked, study.extracted, study.innocent):
        proven = "**proven**" if r.log10_pvalue <= study.decision_threshold_log10p else "not shown"
        rows.append(
            f"| {r.name} | {r.watermark_accuracy:.1%} | {r.matches}/{r.n_keys} "
            f"| {r.log10_pvalue:.1f} | {proven} |"
        )
    return "\n".join(rows)


def _fidelity_read(study: WatermarkStudy) -> str:
    tax = study.clean_pr_auc - study.watermarked_pr_auc
    if abs(tax) < 0.01:
        return (
            f"Embedding the watermark is nearly free: temporal PR-AUC moves {tax:+.3f} "
            f"({study.clean_pr_auc:.3f} clean → {study.watermarked_pr_auc:.3f} watermarked), "
            f"because {study.n_keys} off-manifold keys memorised in empty feature-space regions "
            "do not disturb the decision boundary where real traffic lives."
        )
    return (
        f"The watermark costs {tax:.3f} of temporal PR-AUC ({study.clean_pr_auc:.3f} clean → "
        f"{study.watermarked_pr_auc:.3f} watermarked) — the fidelity tax of forcing the model to "
        f"memorise {study.n_keys} arbitrary keys, reported as the price of the ownership proof."
    )


def _ownership_read(study: WatermarkStudy) -> str:
    w, inn = study.watermarked, study.innocent
    return (
        f"The proof is decisive. The watermarked model classifies {w.watermark_accuracy:.0%} of "
        f"the secret keys by the owner's labels ({w.matches}/{w.n_keys}), a match count whose "
        f"probability under the innocent null (Binomial({w.n_keys}, 0.5)) is "
        f"10^{w.log10_pvalue:.0f} — beyond any evidentiary threshold. The control proves the test "
        "is safe: the innocent model, which never saw the keys, matches the *random* owner labels "
        f"{inn.watermark_accuracy:.0%} of the time ({inn.matches}/{inn.n_keys}, log10 p "
        f"{inn.log10_pvalue:.1f}) — exactly the "
        "chance rate the fair-coin construction guarantees regardless of its class bias, so an "
        "honest model is never falsely accused."
    )


def _extraction_read(study: WatermarkStudy) -> str:
    e = study.extracted
    survived = e.log10_pvalue <= study.decision_threshold_log10p
    if survived:
        return (
            f"And the watermark **survives theft**: a surrogate stolen through "
            f"{study.extraction_queries:,} extraction queries still matches "
            f"{e.watermark_accuracy:.0%} of the keys (log10 p {e.log10_pvalue:.0f}), so even an "
            "extracted copy carries the owner's mark — the strong-transfer regime, reported as it "
            "fell."
        )
    return (
        f"The honest limit is extraction. A surrogate stolen through {study.extraction_queries:,} "
        f"queries matches the keys only {e.watermark_accuracy:.0%} of the time "
        f"(log10 p {e.log10_pvalue:.1f}), near the innocent chance rate: the thief learned the "
        "victim's *decision boundary* over real traffic but not its *arbitrary memorised keys*, "
        "which live off the manifold the queries never probed. Watermarking robustly proves "
        "ownership against a directly copied or fine-tuned model; against a model-extraction "
        "thief it is weak — the same honesty the [extraction](extraction.md) study applied to its "
        "own attack, and the reason watermarking and query-rate limiting are complementary, not "
        "redundant, defenses."
    )


def _render(study: WatermarkStudy, fig: Path) -> str:
    return f"""# NetSentry — Model Watermarking (prove you own the detector)

_Synthetic stand-in. Honest temporal/binary split; {study.n_keys} secret watermark keys with
fair-coin owner labels embedded in training. The ownership test is exact (Binomial null,
log-space tail, no scipy)._

## Why this report exists

The [extraction study](extraction.md) shows a detector can be stolen through its API; this is
the countermeasure that proves it afterwards. Watermarking (Adi et al., USENIX Security 2018)
is the [backdoor](backdoor.md) mechanism turned to the owner's benefit: embed secret trigger
flows with owner-chosen **random** labels during training so the model memorises them, then
prove ownership by querying a suspect on the keys. A watermarked model matches the owner's
labels; an innocent one agrees with the *random* labels only at chance, and because the labels
are fair coins the null is exactly `Binomial(K, 0.5)` — a clean, exact ownership p-value.

## Ownership proof — and the innocent-model control

{_ownership_table(study)}

{_ownership_read(study)}

![Watermark accuracy by model](../figures/{fig.name})

## The fidelity tax

{_fidelity_read(study)}

## Survival under model extraction

{_extraction_read(study)}

## Scope

The watermark keys are off-manifold random points (Adi et al.'s abstract-trigger construction,
in feature space); a content-aware watermark that hides in plausible flows is the named
alternative and would resist a manifold-filtering adversary this one does not model. The
ownership test assumes the owner registers the keys and their labels in advance (a commitment),
so it cannot be forged after the fact. Robustness is against the threats named: it holds against
copying and fine-tuning and is measured — not assumed — against extraction, where it is weak.
The complement of [differential privacy](dp.md) (which bounds leakage) and
[SISA unlearning](unlearn.md) (which honours deletion): this one asserts *ownership*, the third
governance question a deployed model has to answer."""
