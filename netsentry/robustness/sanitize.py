"""Poisoning defense — the label audit wired as a sanitizer, then re-measured.

The poisoning study measures what corrupted labels *cost* (detection at the
operator's threshold collapses while PR-AUC barely moves); the label audit
*finds* candidate errors and validates itself against planted flips. This study
closes the loop the way the hardening study does for evasion: take the attack
(label flips), apply the defense an operator could actually run — out-of-fold
audit over all labeled data, drop every flagged row, refit — and re-measure the
same degradation curve, undefended vs sanitized, always judged on the clean
temporal test split.

Two honesty rules shape the design:

- **The defense drops both suspect directions.** An operator does not know which
  way the labels rot, so every audit flag is dropped — benign-labeled rows that
  score like attacks (where flips land) *and* attack-labeled rows that score
  like benign (often the hardest genuine attacks). The cost of that blindness is
  measured, not assumed.
- **The zero-poison point is the defense's tax.** At a 0% flip rate the audit
  still flags its ambiguity floor, so sanitization pays a clean-data premium;
  the curve starts there instead of at the first poisoned point.

Rows are dropped, never relabeled: relabeling with the auditing model's own
opinion would bootstrap its errors back into the training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.label_audit import audit_labels, out_of_fold_scores
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "poisoning_defense.md"


@dataclass
class DefensePoint:
    """Undefended-vs-sanitized outcomes at one label-flip rate."""

    rate: float
    n_flipped: int  # flips planted across the labeled pool (train + val)
    n_dropped: int  # rows the audit removed (both suspect directions)
    n_flips_caught: int  # planted flips among the dropped rows
    n_clean_dropped: int  # legitimate rows lost — the defense's tax
    undefended_pr_auc: float
    undefended_detection: float  # TPR at the operating FPR, threshold on poisoned val
    sanitized_pr_auc: float
    sanitized_detection: float  # same operating point, chosen on the sanitized val


def flip_positions(y: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Seeded positional choice of attack rows to relabel benign at ``rate``."""
    attack_pos = np.where(np.asarray(y).astype(int) == 1)[0]
    n_flip = int(len(attack_pos) * rate)
    if n_flip == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(attack_pos, size=n_flip, replace=False))


def apply_flips(frame: pd.DataFrame, positions: np.ndarray, benign_label: str) -> pd.DataFrame:
    """Relabel the given positions benign (both targets); copy-safe."""
    if len(positions) == 0:
        return frame
    flipped = frame.copy()
    flipped.iloc[positions, flipped.columns.get_loc(BINARY_TARGET)] = 0
    flipped.iloc[positions, flipped.columns.get_loc(MULTICLASS_TARGET)] = benign_label
    return flipped


def defense_outcome(dropped: np.ndarray, planted: np.ndarray) -> tuple[int, int]:
    """(planted flips caught, clean rows lost) for a dropped-position set."""
    dropped_set = set(np.asarray(dropped).tolist())
    planted_set = set(np.asarray(planted).tolist())
    caught = len(dropped_set & planted_set)
    return caught, len(dropped_set) - caught


def _fit_and_measure(
    settings: Settings,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """(PR-AUC, detection at the operating FPR) on the clean test split.

    The threshold is chosen on the validation labels *as given* — poisoned or
    sanitized — because that is the only view the operator has; only the test
    ground truth stays clean.
    """
    seed_everything(settings.seed)
    pipeline = build_pipeline(settings)
    x_train = pipeline.fit_transform(train)
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    benign = settings.labels.benign_label
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, settings.thresholds.fpr_targets[-1])
    detection = rates_at_threshold(y_test, s_test, threshold)["tpr"]
    return float(average_precision_score(y_test, s_test)), float(detection)


def _defense_curve(settings: Settings) -> list[DefensePoint]:
    """Poison, defend, and re-measure at each configured flip rate."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    benign = variant.labels.benign_label
    cfg = variant.sanitize

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    n_labeled = len(train) + len(val)
    if n_labeled > cfg.max_rows:
        frac = cfg.max_rows / n_labeled
        train = train.sample(frac=frac, random_state=variant.seed)
        val = val.sample(frac=frac, random_state=variant.seed)
    # One frame for the audit (the operator's whole labeled pool, positionally
    # split back into train/val afterwards so threshold selection stays honest).
    labeled = pd.concat([train, val], ignore_index=True)
    n_train = len(train)
    y_test = test[BINARY_TARGET].to_numpy()

    points: list[DefensePoint] = []
    for rate in cfg.flip_rates:
        planted = flip_positions(labeled[BINARY_TARGET].to_numpy(), rate, variant.seed)
        poisoned = apply_flips(labeled, planted, benign)
        p_train = poisoned.iloc[:n_train].reset_index(drop=True)
        p_val = poisoned.iloc[n_train:].reset_index(drop=True)
        undef_pr, undef_det = _fit_and_measure(variant, p_train, p_val, test, y_test)

        oof = out_of_fold_scores(variant, poisoned)
        findings = audit_labels(poisoned[BINARY_TARGET].to_numpy(), oof)
        dropped = np.union1d(findings.suspect_benign, findings.suspect_attack)
        caught, clean_lost = defense_outcome(dropped, planted)

        keep = np.setdiff1d(np.arange(len(poisoned)), dropped)
        s_train = poisoned.iloc[keep[keep < n_train]].reset_index(drop=True)
        s_val = poisoned.iloc[keep[keep >= n_train]].reset_index(drop=True)
        san_pr, san_det = _fit_and_measure(variant, s_train, s_val, test, y_test)

        points.append(
            DefensePoint(
                rate=rate,
                n_flipped=len(planted),
                n_dropped=len(dropped),
                n_flips_caught=caught,
                n_clean_dropped=clean_lost,
                undefended_pr_auc=undef_pr,
                undefended_detection=undef_det,
                sanitized_pr_auc=san_pr,
                sanitized_detection=san_det,
            )
        )
        logger.info(
            "Defense point",
            extra={
                "rate": rate,
                "dropped": len(dropped),
                "caught": caught,
                "undefended": round(undef_det, 4),
                "sanitized": round(san_det, 4),
            },
        )
    return points


def run_sanitize_report(settings: Settings) -> Path:
    """Run the audit-and-drop defense across flip rates; write report + figure."""
    points = _defense_curve(settings)

    rates = np.array([p.rate for p in points])
    fig = plots.plot_lines(
        {
            "undefended detection": (rates, np.array([p.undefended_detection for p in points])),
            "sanitized detection": (rates, np.array([p.sanitized_detection for p in points])),
        },
        xlabel="Fraction of labeled attack rows flipped benign",
        ylabel="Detection @ operating FPR (clean test)",
        title="Audit-and-drop sanitization vs label-flip poisoning",
        out_path=settings.paths.figures_dir / "poisoning_defense.png",
    )

    report = _render(points, settings, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote poisoning-defense report", extra={"path": str(out_path)})

    worst = points[-1]
    clean = points[0]
    with track_run(settings, "poisoning_defense") as run:
        run.log_metrics(
            {
                "worst_rate_detection_undefended": worst.undefended_detection,
                "worst_rate_detection_sanitized": worst.sanitized_detection,
                "worst_rate_flip_recall": (
                    worst.n_flips_caught / worst.n_flipped if worst.n_flipped else float("nan")
                ),
                "clean_tax_detection": clean.sanitized_detection - clean.undefended_detection,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _mechanics_table(points: list[DefensePoint]) -> str:
    rows = [
        "| flip rate | flips planted | rows dropped | flips caught | clean rows lost "
        "| flip recall |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        recall = f"{p.n_flips_caught / p.n_flipped * 100:.1f}%" if p.n_flipped else "—"
        rows.append(
            f"| {p.rate * 100:g}% | {p.n_flipped:,} | {p.n_dropped:,} | {p.n_flips_caught:,} "
            f"| {p.n_clean_dropped:,} | {recall} |"
        )
    return "\n".join(rows)


def _outcomes_table(points: list[DefensePoint]) -> str:
    rows = [
        "| flip rate | detection (undefended) | detection (sanitized) | PR-AUC (undefended) "
        "| PR-AUC (sanitized) |",
        "|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.rate * 100:g}% | {p.undefended_detection * 100:.1f}% "
            f"| {p.sanitized_detection * 100:.1f}% | {p.undefended_pr_auc:.3f} "
            f"| {p.sanitized_pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _read(points: list[DefensePoint]) -> str:
    """Sign-aware summary so the prose can never contradict the numbers."""
    clean, worst = points[0], points[-1]
    delta = worst.sanitized_detection - worst.undefended_detection
    tax = clean.sanitized_detection - clean.undefended_detection
    recall = worst.n_flips_caught / worst.n_flipped if worst.n_flipped else float("nan")

    if delta >= 0.05:
        verdict = (
            f"**The defense works, and the mechanism is visible in the mechanics table.** At a "
            f"{worst.rate * 100:g}% flip rate the audit catches {recall * 100:.1f}% of the "
            f"planted flips, so the refit model trains on mostly-clean labels and — just as "
            f"important — the operating threshold is chosen on a mostly-clean validation set "
            f"again. Detection at the operating point recovers from "
            f"**{worst.undefended_detection * 100:.1f}% to "
            f"{worst.sanitized_detection * 100:.1f}%** ({delta * 100:+.1f} points). The "
            f"poisoning study showed the damage rides through the poisoned threshold; the "
            f"defense heals exactly that channel."
        )
    elif delta <= -0.05:
        verdict = (
            f"**The defense backfires here** ({delta * 100:+.1f} points at a "
            f"{worst.rate * 100:g}% flip rate): dropping the suspect-*attack* rows removes the "
            f"hardest genuine attacks from training, and that costs more than removing the "
            f"caught flips ({recall * 100:.1f}% of those planted) buys back. That is a real "
            f"operational trade, reported as measured — a defender who knows the noise "
            f"direction should drop only the suspect-benign side."
        )
    else:
        verdict = (
            f"**The defense roughly breaks even on this stand-in** ({delta * 100:+.1f} points "
            f"at a {worst.rate * 100:g}% flip rate, with {recall * 100:.1f}% of planted flips "
            f"caught). The flips the audit misses are the ones scoring near the boundary — "
            f"exactly the rows an out-of-fold score cannot distinguish from genuine "
            f"ambiguity — so what remains after sanitization is the harder residue."
        )

    if tax >= 0.02:
        tax_read = (
            f"The zero-poison row carries its own finding: with nothing planted, dropping the "
            f"audit's ambiguity floor (**{clean.n_dropped:,} rows**) *raises* detection by "
            f"{tax * 100:+.1f} points. The flagged clean rows are the class-overlap residue — "
            f"the same families the per-class slices show being missed — and removing them "
            f"sharpens the boundary the threshold is chosen on. Do not bank on that sign: it "
            f"is a property of this generator's overlap, and the honest expectation on real "
            f"data is a small cost, not a bonus."
        )
    elif tax <= -0.02:
        tax_read = (
            f"The premium is measured at the zero-poison point: with nothing planted, the "
            f"audit still drops **{clean.n_dropped:,} rows** (its ambiguity floor) and "
            f"detection pays {tax * 100:+.1f} points for it. That is what running this "
            f"defense costs on clean data — the insurance premium, stated so the recovery "
            f"above is not read as free."
        )
    else:
        tax_read = (
            f"At the zero-poison point the audit drops **{clean.n_dropped:,} rows** (its "
            f"ambiguity floor) and detection barely moves ({tax * 100:+.1f} points) — on this "
            f"stand-in the defense is close to free on clean data, though the dropped rows "
            f"are training data spent on insurance either way."
        )
    return f"{verdict}\n\n{tax_read}"


def _render(points: list[DefensePoint], settings: Settings, fig: Path) -> str:
    operating_fpr = settings.thresholds.fpr_targets[-1]
    folds = settings.label_audit.folds
    return f"""# NetSentry — Poisoning Defense (audit-and-drop, re-measured)

_Synthetic stand-in. Temporal split; flips are planted across the operator's whole
labeled pool (train + validation), the confident-learning audit ({folds}-fold
out-of-fold scores, shared knob with `netsentry labelaudit`) flags suspects in
both directions, every flag is dropped, and the model is refit. Detection is at
the {operating_fpr * 100:g}% FPR budget with the threshold chosen on the poisoned
(or sanitized) validation labels — the operator's actual position. Only the test
ground truth stays clean._

## Why this report exists

The poisoning study ends with the damage: label flips leave PR-AUC nearly intact
while detection at the shipped threshold collapses, because the operating point
is chosen on the poisoned validation labels. The label audit proved it can find
planted flips. This study is the third step of the arc the hardening report walks
for evasion — **measure, fix, re-measure** — for the training-time adversary: the
audit is wired in as an automated sanitizer and the same decay curve is run with
the defense on.

## Defense mechanics

Both suspect directions are dropped — benign-labeled rows scoring like attacks
(where flips land) and attack-labeled rows scoring like benign — because an
operator cannot know which way the labels rot. Rows are dropped, never relabeled:
relabeling with the auditing model's own opinion would bootstrap its errors back
into training.

{_mechanics_table(points)}

## Outcomes on the clean test split

{_outcomes_table(points)}

![Poisoning defense](../figures/{fig.name})

## Read

{_read(points)}

## What this defense does not cover

- It defends the noise model it audits for: **random** flips, which land far from
  the class boundary and are exactly what out-of-fold scoring can see. An adaptive
  poisoner who flips only near-boundary flows sits inside the audit's measured
  ambiguity floor and is not caught — the label audit's own report quantifies that
  floor.
- Benign-pool contamination against the anomaly detector (the poisoning study's
  second attack) is untouched: this defense reasons over *labels*, and
  contamination corrupts an unlabeled pool.
- The clean-data tax recurs on every retrain. A deployment that runs this
  continuously is buying insurance with training data — the premium is the
  measured zero-poison row, not zero.
"""
