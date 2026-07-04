"""Label-noise audit — find the mislabeled rows instead of assuming them away.

CIC-IDS2017 ships with community-documented label errors (the Engelen et al.,
WTMC 2021 corrected release exists for a reason), and the project's poisoning study
shows what corrupted labels *cost*. This audit is the complementary tool: it *finds*
candidate errors, confident-learning style, using only the training split.

Method (a deliberately small implementation of the Northcutt et al. idea):

1. **Out-of-fold scores.** Stratified k-fold over the temporal training split; each
   row is scored by a pipeline+model fit that never saw it, so a row cannot vouch
   for its own label.
2. **Class-conditional thresholds.** ``t_attack`` is the mean out-of-fold attack
   score among rows *labeled* attack, ``t_benign`` the mean among rows labeled
   benign. A benign-labeled row scoring at or above ``t_attack`` looks as much like
   an attack as typical attacks do — a suspect; symmetrically for attack-labeled
   rows at or below ``t_benign``.
3. **Self-validation.** Assertions about noise detection deserve evidence, so the
   audit plants a known fraction of label flips (attack -> benign) and reports its
   own recovery precision/recall against that ground truth. On synthetic data the
   planted flips are the *only* noise, so precision is exact; on real data intrinsic
   errors count against precision, making the reported figure a lower bound.

The audit never touches the test split — it is a training-data quality tool, and
"fixing" test labels with the model under evaluation would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "label_audit.md"


@dataclass
class AuditFindings:
    """Suspected label errors among one frame's rows (positional indices)."""

    t_attack: float  # mean out-of-fold attack score among attack-labeled rows
    t_benign: float  # mean out-of-fold attack score among benign-labeled rows
    suspect_benign: np.ndarray  # benign-labeled rows scoring like attacks
    suspect_attack: np.ndarray  # attack-labeled rows scoring like benign


def audit_labels(y: np.ndarray, scores: np.ndarray) -> AuditFindings:
    """Flag rows whose out-of-fold score is as extreme as the opposite class's mean."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)
    t_attack = float(scores[y == 1].mean()) if (y == 1).any() else float("inf")
    t_benign = float(scores[y == 0].mean()) if (y == 0).any() else float("-inf")
    suspect_benign = np.where((y == 0) & (scores >= t_attack))[0]
    suspect_attack = np.where((y == 1) & (scores <= t_benign))[0]
    return AuditFindings(t_attack, t_benign, suspect_benign, suspect_attack)


def recovery_metrics(flagged: np.ndarray, planted: np.ndarray) -> tuple[float, float]:
    """(precision, recall) of the flagged set against the planted ground truth."""
    flagged_set, planted_set = set(flagged.tolist()), set(planted.tolist())
    hit = len(flagged_set & planted_set)
    precision = hit / len(flagged_set) if flagged_set else float("nan")
    recall = hit / len(planted_set) if planted_set else float("nan")
    return precision, recall


def out_of_fold_scores(settings: Settings, frame: pd.DataFrame) -> np.ndarray:
    """P(attack) for every row from a fold model that never trained on it."""
    y = frame[BINARY_TARGET].to_numpy()
    benign = settings.labels.benign_label
    scores = np.zeros(len(frame), dtype=float)
    skf = StratifiedKFold(
        n_splits=settings.label_audit.folds, shuffle=True, random_state=settings.seed
    )
    for fold, (train_pos, score_pos) in enumerate(skf.split(np.zeros(len(y)), y)):
        fit_part = frame.iloc[train_pos]
        score_part = frame.iloc[score_pos]
        pipeline = build_pipeline(settings)
        x_fit = pipeline.fit_transform(fit_part)
        x_score = pipeline.transform(score_part)
        seed_everything(settings.seed)
        model = SupervisedClassifier(settings).fit(x_fit, fit_part[BINARY_TARGET].to_numpy())
        scores[score_pos] = attack_probability(model.predict_proba(x_score), model.classes_, benign)
        logger.info("Audit fold scored", extra={"fold": fold, "rows": len(score_pos)})
    return scores


def _plant_flips(
    frame: pd.DataFrame, rate: float, benign_label: str, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Flip ``rate`` of attack rows to benign; return the frame and flipped positions."""
    y = frame[BINARY_TARGET].to_numpy()
    attack_pos = np.where(y == 1)[0]
    n_flip = int(len(attack_pos) * rate)
    if n_flip == 0:
        return frame, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    planted = rng.choice(attack_pos, size=n_flip, replace=False)
    flipped = frame.copy()
    flipped.iloc[planted, flipped.columns.get_loc(BINARY_TARGET)] = 0
    flipped.iloc[planted, flipped.columns.get_loc(MULTICLASS_TARGET)] = benign_label
    return flipped, planted


def run_label_audit_report(settings: Settings) -> Path:
    """Audit the temporal training labels; validate the audit with planted flips."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    benign = variant.labels.benign_label

    train = load_split(variant, "temporal", "train")
    if len(train) > variant.label_audit.max_rows:
        train = train.sample(variant.label_audit.max_rows, random_state=variant.seed)
    train = train.reset_index(drop=True)
    y = train[BINARY_TARGET].to_numpy()

    # Pass 1 — the audit proper, on the labels as recorded.
    scores = out_of_fold_scores(variant, train)
    clean_findings = audit_labels(y, scores)

    # Pass 2 — self-validation: plant flips, rerun, measure recovery.
    rate = variant.label_audit.planted_flip_rate
    flipped, planted = _plant_flips(train, rate, benign, variant.seed)
    flip_scores = out_of_fold_scores(variant, flipped)
    flip_findings = audit_labels(flipped[BINARY_TARGET].to_numpy(), flip_scores)
    precision, recall = recovery_metrics(flip_findings.suspect_benign, planted)

    n_benign, n_attack = int((y == 0).sum()), int((y == 1).sum())
    logger.info(
        "Label audit",
        extra={
            "suspect_benign": len(clean_findings.suspect_benign),
            "suspect_attack": len(clean_findings.suspect_attack),
            "planted": len(planted),
            "recovery_recall": None if np.isnan(recall) else round(recall, 3),
        },
    )

    fig = _plot(clean_findings, n_benign, n_attack, precision, recall, rate, settings)
    report = _render(
        clean_findings, n_benign, n_attack, planted, precision, recall, rate, variant, fig
    )
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote label-audit report", extra={"path": str(out_path)})

    with track_run(settings, "label_audit") as run:
        run.log_metrics(
            {
                "suspect_benign": float(len(clean_findings.suspect_benign)),
                "suspect_attack": float(len(clean_findings.suspect_attack)),
            }
        )
        if np.isfinite(recall):
            run.log_metrics({"recovery_recall": recall, "recovery_precision": precision})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(
    findings: AuditFindings,
    n_benign: int,
    n_attack: int,
    precision: float,
    recall: float,
    rate: float,
    settings: Settings,
) -> Path:
    labels = [
        "intrinsic suspects (benign-labeled)",
        "intrinsic suspects (attack-labeled)",
        f"recovery recall ({rate * 100:g}% planted flips)",
        "recovery precision",
    ]
    values = [
        len(findings.suspect_benign) / n_benign if n_benign else 0.0,
        len(findings.suspect_attack) / n_attack if n_attack else 0.0,
        recall if np.isfinite(recall) else 0.0,
        precision if np.isfinite(precision) else 0.0,
    ]
    return plots.plot_barh(
        labels,
        values,
        xlabel="Rate",
        title="Label-noise audit: intrinsic suspects and planted-flip recovery",
        out_path=settings.paths.figures_dir / "label_audit.png",
    )


def _render(
    findings: AuditFindings,
    n_benign: int,
    n_attack: int,
    planted: np.ndarray,
    precision: float,
    recall: float,
    rate: float,
    settings: Settings,
    fig: Path,
) -> str:
    sus_b, sus_a = len(findings.suspect_benign), len(findings.suspect_attack)
    rate_b = sus_b / n_benign if n_benign else 0.0
    rate_a = sus_a / n_attack if n_attack else 0.0
    recall_cell = f"{recall * 100:.1f}%" if np.isfinite(recall) else "—"
    precision_cell = f"{precision * 100:.1f}%" if np.isfinite(precision) else "—"

    rows = [
        "| quantity | value |",
        "|---|---|",
        f"| training rows audited | {n_benign + n_attack:,} "
        f"({n_attack:,} attack / {n_benign:,} benign) |",
        f"| out-of-fold folds | {settings.label_audit.folds} |",
        f"| class thresholds (mean OOF attack score) | attack {findings.t_attack:.3f} / "
        f"benign {findings.t_benign:.3f} |",
        f"| suspect benign-labeled rows (score like an attack) | {sus_b:,} "
        f"({rate_b * 100:.2f}%) |",
        f"| suspect attack-labeled rows (score like benign) | {sus_a:,} ({rate_a * 100:.2f}%) |",
        f"| planted flips (validation) | {len(planted):,} ({rate * 100:g}% of attack rows) |",
        f"| planted-flip recovery recall | {recall_cell} |",
        f"| planted-flip recovery precision | {precision_cell} |",
    ]

    parts: list[str] = []
    # On the stand-in the labels are correct by construction, so intrinsic flags are
    # the method's ambiguity floor, never "likely mislabeled" — claiming otherwise
    # would be exactly the over-reading the audit exists to prevent.
    floor_note = (
        "On this synthetic stand-in, whose labels are correct by construction, every one of "
        "these is a **false positive of the method** — its ambiguity floor. The generator's "
        "classes deliberately overlap, so the subtler attack families genuinely score like "
        "benign traffic (the same rows the per-class slices show being missed). On the real "
        "dataset the identical list is the candidate queue to reconcile against the Engelen "
        "et al. corrections; the floor measured here is what an empty queue looks like."
    )
    if max(rate_b, rate_a) < 0.02:
        parts.append(
            f"On the labels as recorded the audit flags {sus_b:,} benign-labeled and {sus_a:,} "
            f"attack-labeled rows — under 2% on both sides. {floor_note}"
        )
    else:
        parts.append(
            f"On the labels as recorded the audit flags **{sus_b:,}** benign-labeled rows "
            f"({rate_b * 100:.1f}%) and **{sus_a:,}** attack-labeled rows ({rate_a * 100:.1f}%). "
            f"{floor_note}"
        )
    if np.isfinite(recall) and np.isfinite(precision):
        planted_base = len(planted) / (n_benign + len(planted)) if len(planted) else float("nan")
        lift = precision / planted_base if planted_base and np.isfinite(planted_base) else 0.0
        if lift >= 2:
            precision_read = (
                f"Precision must be read against its base rate: the planted errors are only "
                f"{planted_base * 100:.1f}% of benign-labeled rows, so the flag list "
                f"concentrates true errors **{lift:.0f}x** over inspecting rows at random — a "
                "triage multiplier, not an oracle. Every non-planted flag counts against "
                "precision, and on this stand-in those are known to be ambiguity rather than "
                "noise, so on data with real label errors the measured precision is a lower "
                "bound."
            )
        else:
            precision_read = (
                f"Against the {planted_base * 100:.1f}% planted base rate that is a lift of "
                f"only {lift:.1f}x — on this data the flag list barely beats random inspection "
                "and should not be trusted as a triage queue; the honest conclusion is that "
                "the class overlap defeats the class-conditional threshold rule here."
            )
        parts.append(
            f"**The audit validates itself.** With {rate * 100:g}% of attack rows deliberately "
            f"flipped to benign, the audit recovers **{recall * 100:.1f}%** of the planted "
            f"flips at **{precision * 100:.1f}%** precision. {precision_read} A noise detector "
            "that was never tested against known noise is just an opinion."
        )
        parts.append(
            "The division of labour with the poisoning study: poisoning measures what corrupted "
            "labels *cost* (detection collapses via the poisoned validation threshold); this "
            "audit *finds* the corrupted rows so the cost need not be paid. Out-of-fold scoring "
            "is what keeps it honest — no row is judged by a model that trained on it, and the "
            "test split is never touched."
        )
    return f"""# NetSentry — Label-Noise Audit (confident-learning style)

_Synthetic stand-in. Out-of-fold model scores over the **temporal training split**
flag rows whose score is as extreme as the opposite class's mean — candidate label
errors. The audit then validates itself by planting {rate * 100:g}% label flips and
measuring its own recovery. The test split is never touched._

## Why audit labels at all

CIC-IDS2017's label errors are documented well enough that a corrected re-release
exists (Engelen et al., WTMC 2021 — see `DATA_CARD.md`), and the poisoning study
shows corrupted labels quietly destroy the operating point while PR-AUC looks fine.
Rather than assume the labels are clean, the audit produces the candidate error list
— and proves it can find planted errors before asking anyone to trust it.

{chr(10).join(rows)}

![Label audit](../figures/{fig.name})

## Read

{chr(10).join(f"{p}{chr(10)}" for p in parts)}
"""
