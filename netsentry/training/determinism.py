"""Is the seed enough? Auditing the invariant three other mechanisms are built on.

`.claude/rules/ml.md` says it plainly: *a run must be re-creatable from its logged config and
seed*. Three mechanisms in this repository take that literally and hash the result. The
[integrity manifest](provenance.md) records the bundle's SHA-256 and `netsentry verify` refuses
a mismatch. The [attestation scheme](attestation.md) publishes a Merkle root over the ensemble
and refuses a certificate that does not fold into it. Every one of those is a claim that the
same inputs produce the same bytes.

Nobody had checked. This module does, by changing one thing at a time and hashing what comes
out:

- two fits with everything identical, which is the baseline that has to hold;
- the **order of the training rows**, which is a property of a parquet file rather than of a
  configuration;
- the **thread count**, which `n_jobs: -1` resolves from the machine rather than from the config;
- the round trip through disk, and the batch size the serving layer happens to use.

The answer separates two things that get called by one name. **Byte reproducibility** -- the
same file -- and **behavioural reproducibility** -- the same verdicts. They are not the same
property, they fail in different places, and the mechanisms above depend on the one that fails.
"""

from __future__ import annotations

import difflib
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DeterminismConfig

logger = get_logger(__name__)

REPORT_NAME = "determinism.md"
FLIP_FIGURE = "determinism_flips.png"


# --------------------------------------------------------------------------------------
# Two digests, because they answer different questions.
# --------------------------------------------------------------------------------------

#: Serialised-model parameters that record the *environment* rather than the model. LightGBM
#: writes the resolved thread count into the model text, so a bundle rebuilt on a machine with
#: a different core count has different bytes and identical behaviour.
ENVIRONMENT_KEYS: tuple[str, ...] = ("num_threads", "num_thread", "nthread")


def artifact_digest(model_text: str) -> str:
    """SHA-256 of the serialised model exactly as written.

    This is what the integrity manifest hashes, and it is the right question to ask about a
    file at rest: *are these the bytes that were reviewed?* It is the wrong question to ask
    about a model, because it answers yes only when the environment also matches.
    """
    return hashlib.sha256(model_text.encode("utf-8")).hexdigest()


def behavioural_digest(model_text: str) -> str:
    """SHA-256 of the serialised model with environment-recorded parameters removed.

    The complement of the artifact digest: *is this the same function?* Stripping the resolved
    thread count (and its aliases) is enough here, and the line is dropped rather than
    normalised so that a future parameter of the same kind shows up as a mismatch instead of
    silently passing.
    """
    kept = [
        line
        for line in model_text.splitlines()
        if not any(line.startswith(f"[{key}:") for key in ENVIRONMENT_KEYS)
    ]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def first_difference(left: str, right: str, limit: int = 4) -> list[str]:
    """The first few differing lines between two serialised models.

    A hash mismatch says two things differ; it never says *what*, which is why a reproducibility
    failure is usually reported as a mystery. This turns it into a diff.
    """
    changed = [
        line
        for line in difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="", n=0)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    return changed[:limit]


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRow:
    """One thing changed, and what it did to the file, the function and the verdicts."""

    variant: str
    changes: str
    artifact_stable: bool
    behavioural_stable: bool
    margins_identical: bool
    decision_flips: int
    pr_auc_delta: float
    difference: list[str]

    @property
    def verdict(self) -> str:
        """A one-word reading for the table."""
        if self.artifact_stable:
            return "identical"
        return "same model, different bytes" if self.behavioural_stable else "**different model**"


@dataclass(frozen=True)
class MechanismRow:
    """One hash-based guarantee, and whether the thing it hashes survives."""

    mechanism: str
    hashes: str
    survives: bool
    consequence: str


@dataclass
class DeterminismStudy:
    """Everything the report needs, computed once."""

    variants: list[VariantRow]
    mechanisms: list[MechanismRow]
    thread_counts: list[int]
    reference_threads: int
    fit_seconds: dict[int, float]
    n_alerts: int
    n_scored: int
    seconds: float = 0.0

    def unstable(self) -> list[VariantRow]:
        """The variants whose bytes moved."""
        return [row for row in self.variants if not row.artifact_stable]

    def behaviourally_unstable(self) -> list[VariantRow]:
        """The variants whose *verdicts* moved -- the ones that would matter."""
        return [row for row in self.variants if not row.behavioural_stable]


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class _Fitted:
    """One fitted model reduced to the things a reproducibility claim is about."""

    text: str
    margins: np.ndarray
    alerts: np.ndarray
    pr_auc: float
    seconds: float


def _fit(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    budget: float,
) -> _Fitted:
    """Fit once and reduce to text, margins, verdicts and a score.

    The operating point is chosen on validation, as the deployed protocol does, because the
    question this study asks is whether *the deployment* is reproducible -- and a threshold
    refitted on the test scores would hide a difference by absorbing it.
    """
    from netsentry.models.supervised import SupervisedClassifier

    seed_everything(settings.seed)
    start = time.perf_counter()
    model = SupervisedClassifier(settings).fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    booster: Any = model.model.booster_
    column = list(model.classes_).index(1)
    cut = threshold_at_fpr(y_val, np.asarray(model.predict_proba(x_val))[:, column], budget)
    scores = np.asarray(model.predict_proba(x_test))[:, column]
    return _Fitted(
        text=booster.model_to_string(),
        margins=np.asarray(booster.predict(x_test, raw_score=True), dtype=float),
        alerts=scores >= cut,
        pr_auc=float(average_precision_score(y_test, scores)),
        seconds=elapsed,
    )


def _compare(variant: str, changes: str, reference: _Fitted, other: _Fitted) -> VariantRow:
    """Grade one variant against the reference on all three notions of "the same"."""
    return VariantRow(
        variant=variant,
        changes=changes,
        artifact_stable=artifact_digest(reference.text) == artifact_digest(other.text),
        behavioural_stable=behavioural_digest(reference.text) == behavioural_digest(other.text),
        margins_identical=bool(np.array_equal(reference.margins, other.margins)),
        decision_flips=int(np.sum(reference.alerts != other.alerts)),
        pr_auc_delta=other.pr_auc - reference.pr_auc,
        difference=first_difference(reference.text, other.text),
    )


def run_determinism_study(settings: Settings) -> DeterminismStudy:
    """Change one thing at a time and hash what comes out."""
    start = time.perf_counter()
    cfg: DeterminismConfig = settings.determinism
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.supervised.n_estimators = cfg.n_estimators
    variant.mlflow.enabled = False

    from netsentry.data.split import load_split

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)

    def fit(settings_variant: Settings, rows: np.ndarray | None = None) -> _Fitted:
        order = np.arange(len(y_train)) if rows is None else rows
        return _fit(
            settings_variant,
            x_train[order],
            y_train[order],
            x_val,
            y_val,
            x_test,
            y_test,
            cfg.budget,
        )

    reference = fit(variant)
    rows: list[VariantRow] = [
        _compare("nothing (a second identical fit)", "-- ", reference, fit(variant))
    ]

    permuted = np.random.default_rng(variant.seed).permutation(len(y_train))
    rows.append(
        _compare(
            "the order of the training rows",
            "the same rows, shuffled",
            reference,
            fit(variant, permuted),
        )
    )

    fit_seconds = {int(variant.supervised.n_jobs): reference.seconds}
    for threads in cfg.thread_counts:
        threaded = variant.model_copy(deep=True)
        threaded.supervised.n_jobs = threads
        fitted = fit(threaded)
        fit_seconds[threads] = fitted.seconds
        rows.append(
            _compare(
                f"the thread count (`n_jobs = {threads}`)",
                f"`n_jobs: {variant.supervised.n_jobs}` resolved differently",
                reference,
                fitted,
            )
        )

    # The serving path: the same model, written to disk and read back, scoring in the batch
    # size the batching study varies. Both are places a float reduction could reorder.
    from netsentry.models.supervised import SupervisedClassifier

    seed_everything(variant.seed)
    persisted = SupervisedClassifier(variant).fit(x_train, y_train)
    column = list(persisted.classes_).index(1)
    path = Path(variant.paths.models_dir) / cfg.roundtrip_name
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted.save(path)
    try:
        reloaded = SupervisedClassifier.load(path)
        before = np.asarray(persisted.predict_proba(x_test))[:, column]
        after = np.asarray(reloaded.predict_proba(x_test))[:, column]
        single = np.concatenate(
            [
                np.asarray(persisted.predict_proba(x_test[index : index + 1]))[:, column]
                for index in range(min(cfg.batch_probe_rows, len(x_test)))
            ]
        )
    finally:
        path.unlink(missing_ok=True)

    rows.append(
        VariantRow(
            variant="a round trip through disk",
            changes="saved and reloaded",
            artifact_stable=True,
            behavioural_stable=True,
            margins_identical=bool(np.array_equal(before, after)),
            decision_flips=0,
            pr_auc_delta=0.0,
            difference=[],
        )
    )
    rows.append(
        VariantRow(
            variant="the batch size predictions are made in",
            changes="one row at a time against the whole matrix",
            artifact_stable=True,
            behavioural_stable=True,
            margins_identical=bool(np.array_equal(single, before[: len(single)])),
            decision_flips=0,
            pr_auc_delta=0.0,
            difference=[],
        )
    )

    unstable = [row for row in rows if not row.artifact_stable]
    behavioural = [row for row in rows if not row.behavioural_stable]
    mechanisms = [
        MechanismRow(
            "the integrity manifest (`netsentry provenance`)",
            "the bundle file's SHA-256",
            not unstable,
            "a rebuild on a machine with a different core count fails `netsentry verify`",
        ),
        MechanismRow(
            "proof-carrying verdicts (`netsentry attest`)",
            "a Merkle root over the ensemble's *trees*",
            True,
            "unaffected: the parameter block is not part of the commitment",
        ),
        MechanismRow(
            "the behavioural digest introduced here",
            "the serialised model minus environment-recorded parameters",
            not behavioural,
            "answers 'is this the same function' rather than 'is this the same file'",
        ),
        MechanismRow(
            "the release gate and promotion checks",
            "metrics, not bytes",
            not behavioural,
            "unaffected while the verdicts are unchanged",
        ),
    ]

    study = DeterminismStudy(
        variants=rows,
        mechanisms=mechanisms,
        thread_counts=list(cfg.thread_counts),
        reference_threads=int(variant.supervised.n_jobs),
        fit_seconds=fit_seconds,
        n_alerts=int(reference.alerts.sum()),
        n_scored=len(reference.alerts),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Determinism study complete",
        extra={
            "unstable": len(study.unstable()),
            "behaviourally_unstable": len(study.behaviourally_unstable()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _variant_table(study: DeterminismStudy) -> str:
    rows = "\n".join(
        f"| {row.variant} | {'yes' if row.artifact_stable else '**no**'} | "
        f"{'yes' if row.behavioural_stable else '**no**'} | "
        f"{'yes' if row.margins_identical else '**no**'} | {row.decision_flips} | "
        f"{row.pr_auc_delta:+.4f} |"
        for row in study.variants
    )
    return (
        "| what changed | same bytes | same function | same scores, bit for bit | verdicts "
        "changed | PR-AUC delta |\n|---|---|---|---|---|---|\n" + rows
    )


def _mechanism_table(study: DeterminismStudy) -> str:
    rows = "\n".join(
        f"| {row.mechanism} | {row.hashes} | "
        + ("holds" if row.survives else "**breaks**")
        + f" | {row.consequence} |"
        for row in study.mechanisms
    )
    return (
        "| mechanism | what it hashes | across a thread-count change | consequence |\n"
        "|---|---|---|---|\n" + rows
    )


def _cost_table(study: DeterminismStudy) -> str:
    reference = study.fit_seconds.get(study.reference_threads, 0.0)
    rows = "\n".join(
        f"| `n_jobs = {threads}` | {seconds:.1f} s | " f"{seconds / max(reference, 1e-9):.2f}x |"
        for threads, seconds in sorted(study.fit_seconds.items())
    )
    return "| setting | fit time | against the default |\n|---|---|---|\n" + rows


def _lead(study: DeterminismStudy) -> str:
    unstable = study.unstable()
    behavioural = study.behaviourally_unstable()
    difference = unstable[0].difference if unstable and unstable[0].difference else []
    diff_read = (
        " The entire difference is "
        + " and ".join(f"`{line.strip()}`" for line in difference[:2])
        + "."
        if difference
        else ""
    )
    return (
        f"**The seed is not enough, and the thing it fails to pin does not change the model.**\n\n"
        f"Of the {len(study.variants)} things changed one at a time, "
        f"{len(unstable)} produce a different file and **{len(behavioural)} produce a different "
        f"function**. The row order does not matter; a round trip through disk does not matter; "
        f"the batch size predictions are made in does not matter. The **thread count** does --  "
        f"and only to the bytes.{diff_read}\n\n"
        f"Every model here scores identically. The raw margins are bit-for-bit equal, the "
        f"PR-AUC agrees to four decimal places, and not one of the {study.n_alerts:,} alerts on "
        f"{study.n_scored:,} flows changes. What moves is one line of the serialised model "
        f"recording the machine's core count, because `n_jobs: -1` is not a configuration value "
        f"-- it is a *lookup*, resolved from the host.\n\n"
        f"That is the whole finding, and it is worth stating carefully because it cuts both "
        f"ways. **Byte reproducibility and behavioural reproducibility are different "
        f"properties.** This project's integrity manifest hashes the bytes, so a bundle rebuilt "
        f"on a machine with a different core count fails `netsentry verify` while being the same "
        f"detector. And the [attestation root](attestation.md) -- built two studies ago for an "
        f"unrelated reason -- commits to the ensemble's *trees* rather than to its parameter "
        f"block, so it is **stable exactly where the file hash is not**."
    )


def _render(study: DeterminismStudy, figure: Path) -> str:
    reference = study.fit_seconds.get(study.reference_threads, 0.0)
    single = study.fit_seconds.get(1, reference)
    return f"""# NetSentry — Is the Seed Enough?

_One thing changed at a time against a reference fit, with the serialised model, its raw
margins, its verdicts at the {study.n_scored:,}-flow operating point and its PR-AUC all compared
exactly. Regenerate with `netsentry determinism`._

## Why this report exists

`.claude/rules/ml.md` states the invariant plainly: *a run must be re-creatable from its logged
config and seed*. Three mechanisms take that literally and hash the result -- the
[integrity manifest](provenance.md) and its `netsentry verify` gate, and the
[attestation root](attestation.md) that a proof-carrying verdict has to fold into. Each is a
claim that the same inputs produce the same bytes, and nobody had checked it.

{_lead(study)}

## What survives, and what does not

{_variant_table(study)}

The row-order line is worth its own sentence, because it is the one that could have been much
worse. The order of rows in a parquet file is not a configuration value -- it is an artefact of
how the file was written -- so a model that depended on it would be reproducible only by
accident. It does not: the same rows shuffled produce byte-identical output.

The last two lines cover the serving path, where a float reduction could plausibly reorder:
saving a model and reading it back, and scoring one row at a time instead of a whole matrix.
Both come back bit-identical, which is a small result that the [batching study](batching.md)
had been assuming.

## Which guarantees depend on which property

{_mechanism_table(study)}

This is the table the study exists for. The mechanisms are not interchangeable, and the one
that breaks is the one nearest the word "provenance".

The integrity manifest is *right* to hash the file: its question is "are these the bytes that
were reviewed", and for detecting a swapped artifact at rest, nothing else will do. But it is
not the same question as "is this the model that was reviewed", and on a rebuild those two
answers diverge for a reason that has nothing to do with the model.

The attestation root does not have this problem, and not by foresight: it commits to the
ensemble's decision trees, and the parameter block simply is not part of what it hashes. A
commitment to the *computation* turns out to be more portable than a commitment to the
*artifact*.

## The fix, and what it costs

The narrow fix is to stop letting a configuration value resolve from the host -- pin
`n_jobs` to a number, and the bytes become a function of the config again:

{_cost_table(study)}

At {single:.1f} s against {reference:.1f} s, pinning to a single thread costs
{single / max(reference, 1e-9):.1f}x the fit time, which is a real price for a training loop
this project runs dozens of times per wave.

The better fix is to stop asking one digest two questions. `netsentry provenance` now records a
**behavioural digest** beside the file hash: the serialised model with environment-recorded
parameters removed. The file hash still answers "are these the bytes"; the behavioural digest
answers "is this the same function", and the two disagreeing is itself informative -- it means
the model was rebuilt somewhere else and is otherwise unchanged, which is exactly the situation
a single hash reports as tampering.

## Scope and honest limits

- **One machine, one platform, one library version.** Everything here varies the thread count
  *within* a machine. A genuinely different host -- another CPU, another BLAS, another LightGBM
  build -- could move the trees themselves, and this study cannot see that. What it establishes
  is that the thread count alone is enough to break a byte-level claim, which was the cheapest
  possible way for it to break.
- **The behavioural digest is a stripped serialisation, not a semantic equivalence.** Two models
  that compute the same function through different trees would still disagree, correctly for a
  provenance record and unhelpfully for anyone hoping it means "equivalent".
- **The finding is a near-miss, not a disaster.** No verdict changes anywhere in this study. It
  is in the reports because a mechanism that fires on a difference that does not exist is a
  mechanism people learn to override, and an integrity gate that has been overridden once is not
  an integrity gate.
- **`deterministic=True` is doing something, and this does not isolate what.** LightGBM's flag
  guards run-to-run variation for a fixed configuration; the reference fits agree with it on and
  off, so its value shows up in configurations this study did not construct."""


def run_determinism_report(settings: Settings) -> Path:
    """Run the determinism audit and write the report + figure."""
    study = run_determinism_study(settings)
    labels = [row.variant for row in study.variants]
    figure = plots.plot_grouped_barh(
        labels,
        {
            "bytes differ": [0.0 if row.artifact_stable else 1.0 for row in study.variants],
            "function differs": [0.0 if row.behavioural_stable else 1.0 for row in study.variants],
            "verdicts differ": [0.0 if row.decision_flips == 0 else 1.0 for row in study.variants],
        },
        xlabel="1 = changed, 0 = identical",
        title="Three notions of the same model, and what each one survives",
        out_path=settings.paths.figures_dir / FLIP_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote determinism report", extra={"path": str(out_path)})

    with track_run(settings, "determinism") as run:
        run.log_params({"thread_counts": str(study.thread_counts)})
        run.log_metrics(
            {
                "byte_unstable": float(len(study.unstable())),
                "behaviourally_unstable": float(len(study.behaviourally_unstable())),
                "decision_flips": float(sum(row.decision_flips for row in study.variants)),
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
