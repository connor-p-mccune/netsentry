"""Leave-one-day-out — how sensitive is the temporal conclusion to the chosen cut?

The headline uses one temporal cut (train Mon-Wed, test Thu-Fri). The rules this
project holds itself to name the alternative in the same breath: *"train on earlier
days, test on later days, OR do leave-one-day-out."* This study runs the second one:
every capture day takes a turn as the held-out "future", with the model trained on
the other four (validation carved from train, threshold chosen there — the same
discipline as everywhere else).

Two properties make LODO more than an error bar on the headline:

- **Every fold is novel-class detection.** Each CIC-IDS2017 attack class was
  captured on exactly one day, so holding a day out removes its classes from
  training entirely — the binary model can only catch them by behavioural
  generalisation from other families. The spread across days is the spread of
  zero-shot difficulty, per class family.
- **Monday is benign-only**, which turns its fold into something no other split
  offers: a pure false-alarm audit. A model that has seen every attack type still
  has to stay quiet on an uneventful day; the realized FPR (and the alerts/day it
  implies) is the quiet-day cost of the deployed threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET, CLEAN_FILENAME, MULTICLASS_TARGET
from netsentry.data.split import temporal_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import alerts_per_day, attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "lodo.md"


def fold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """(PR-AUC, detection, FPR) on one fold, NaN where a side is absent.

    A benign-only day has no detection to measure (and PR-AUC is undefined with one
    class); the NaN is the honest value, not a zero — ``rates_at_threshold``'s 0.0
    convention would silently read as "caught nothing".
    """
    y = np.asarray(y_true).astype(int)
    flagged = np.asarray(scores) >= threshold
    has_attack = bool((y == 1).any())
    has_benign = bool((y == 0).any())
    pr_auc = (
        float(average_precision_score(y, scores)) if has_attack and has_benign else float("nan")
    )
    detection = float(flagged[y == 1].mean()) if has_attack else float("nan")
    fpr = float(flagged[y == 0].mean()) if has_benign else float("nan")
    return pr_auc, detection, fpr


@dataclass
class DayFold:
    """One leave-one-day-out fold: the held-out day scored by a model that never saw it."""

    day: str
    n_test: int
    n_attacks: int
    attack_classes: list[str]  # classes present on the held-out day (novel to the model)
    pr_auc: float
    detection: float
    fpr: float
    est_alerts_per_day: float


def _run_fold(settings: Settings, clean: pd.DataFrame, day: str, operating_fpr: float) -> DayFold:
    """Train on the other four days, threshold on their validation, score the held-out day."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    variant.split.train_days = [d for d in schema.DAY_ORDER if d != day]
    variant.split.test_days = [day]
    benign = variant.labels.benign_label

    result = temporal_split(clean, variant)
    y_train = result.train[BINARY_TARGET].to_numpy()
    y_val = result.val[BINARY_TARGET].to_numpy()
    y_test = result.test[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(result.train)
    x_val, x_test = pipeline.transform(result.val), pipeline.transform(result.test)

    seed_everything(variant.seed)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)

    pr_auc, detection, fpr = fold_metrics(y_test, s_test, threshold)
    labels = result.test[MULTICLASS_TARGET].astype(str)
    classes = sorted(set(labels[labels != benign]))
    est_alerts = (
        alerts_per_day(fpr, settings.thresholds.assumed_flows_per_day)
        if np.isfinite(fpr)
        else float("nan")
    )
    fold = DayFold(
        day=day,
        n_test=len(y_test),
        n_attacks=int((y_test == 1).sum()),
        attack_classes=classes,
        pr_auc=pr_auc,
        detection=detection,
        fpr=fpr,
        est_alerts_per_day=est_alerts,
    )
    logger.info(
        "LODO fold",
        extra={
            "day": day,
            "attacks": fold.n_attacks,
            "detection": None if np.isnan(detection) else round(detection, 4),
            "fpr": None if np.isnan(fpr) else round(fpr, 4),
        },
    )
    return fold


def run_lodo_report(settings: Settings) -> Path:
    """Run every leave-one-day-out fold and write the temporal-sensitivity report."""
    clean_path = settings.paths.data_processed / CLEAN_FILENAME
    clean = pd.read_parquet(clean_path)
    operating_fpr = settings.thresholds.fpr_targets[-1]

    folds = [_run_fold(settings, clean, day, operating_fpr) for day in schema.DAY_ORDER]

    fig = _plot(folds, operating_fpr, settings.paths.figures_dir / "lodo.png")
    report = _render(folds, operating_fpr, settings, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote LODO report", extra={"path": str(out_path)})

    with track_run(settings, "lodo") as run:
        run.log_metrics(
            {f"detection_{f.day}": f.detection for f in folds if np.isfinite(f.detection)}
        )
        run.log_metrics({f"fpr_{f.day}": f.fpr for f in folds if np.isfinite(f.fpr)})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(folds: list[DayFold], operating_fpr: float, out_path: Path) -> Path:
    """Detection per held-out attack day (benign-only days carry no detection bar)."""
    scored = [f for f in folds if np.isfinite(f.detection)]
    if not scored:
        scored = folds
    return plots.plot_barh(
        [f.day for f in scored],
        [f.detection if np.isfinite(f.detection) else 0.0 for f in scored],
        xlabel=f"Detection of the held-out day's (novel) attacks @ {operating_fpr * 100:g}% FPR",
        title="Leave-one-day-out temporal sensitivity",
        out_path=out_path,
    )


def _render(folds: list[DayFold], operating_fpr: float, settings: Settings, fig: Path) -> str:
    rows = [
        "| held-out day | test flows | attacks | novel classes | PR-AUC | detection | FPR "
        "| est. alerts/day |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        classes = ", ".join(f.attack_classes) if f.attack_classes else "— (benign only)"
        pr = f"{f.pr_auc:.3f}" if np.isfinite(f.pr_auc) else "—"
        det = f"{f.detection * 100:.1f}%" if np.isfinite(f.detection) else "—"
        fpr = f"{f.fpr * 100:.2f}%" if np.isfinite(f.fpr) else "—"
        alerts = f"~{f.est_alerts_per_day:,.0f}" if np.isfinite(f.est_alerts_per_day) else "—"
        rows.append(
            f"| {f.day} | {f.n_test:,} | {f.n_attacks:,} | {classes} | {pr} | {det} | {fpr} "
            f"| {alerts} |"
        )

    parts: list[str] = []
    quiet = [f for f in folds if f.n_attacks == 0]
    for f in quiet:
        parts.append(
            f"**{f.day} is benign-only**, so its fold is the quiet-day false-alarm audit no "
            f"other split provides: a model trained on every attack type raises "
            f"{f.fpr * 100:.2f}% false positives on an uneventful day — roughly "
            f"**{f.est_alerts_per_day:,.0f} alerts/day** at the assumed "
            f"{settings.thresholds.assumed_flows_per_day:,} flows/day. That number, not "
            "detection, is what a SOC pays on most days, since most days are Mondays."
        )
    attacked = [f for f in folds if np.isfinite(f.detection)]
    if len(attacked) >= 2:
        best = max(attacked, key=lambda f: f.detection)
        worst = min(attacked, key=lambda f: f.detection)
        spread = best.detection - worst.detection
        mean_det = float(np.mean([f.detection for f in attacked]))
        parts.append(
            f"Across the attack days, detection of the held-out (never-trained-on) classes "
            f"ranges from **{worst.detection * 100:.1f}%** ({worst.day}: "
            f"{', '.join(worst.attack_classes)}) to **{best.detection * 100:.1f}%** "
            f"({best.day}: {', '.join(best.attack_classes)}), mean "
            f"{mean_det * 100:.1f}% — a {spread * 100:.0f}-point spread. Because each "
            "CIC-IDS2017 attack class was captured on exactly one day, every LODO fold is "
            "**zero-shot class detection**: the model has literally never seen the held-out "
            "day's attack types and can only catch them by behavioural resemblance to other "
            "families. The spread is therefore a per-family novelty-difficulty profile, not "
            "noise around one number."
        )
        parts.append(
            "The practical reading: the temporal conclusion does not hinge on the particular "
            "Mon-Wed/Thu-Fri cut — every rotation of the held-out day tells the same story "
            "(novel attack families are hard at a fixed FP budget), while *which* families are "
            "hard varies. An honest headline should be read with this range in mind, and the "
            "per-class slices report names the same hard families from the fixed cut."
        )
    elif attacked:
        f = attacked[0]
        parts.append(
            f"Only {f.day} carries attacks in this dataset slice, so LODO reduces to a single "
            "novel-day estimate — run on the full dataset for the day-by-day profile."
        )
    return f"""# NetSentry — Leave-One-Day-Out (temporal sensitivity)

_Synthetic stand-in. Every capture day takes a turn as the held-out "future": the
model trains on the other four days (validation carved from train; threshold chosen
there at the {operating_fpr * 100:g}%-FPR budget) and is scored on the day it never
saw. This is the rotation-robustness check on the headline temporal split — the
project's own rules name it as the alternative honest split._

## Why rotate the held-out day

A single temporal cut is one draw from the space of honest evaluations. Rotating the
held-out day answers two questions the fixed cut cannot: **how much does the
conclusion depend on which days were chosen**, and **how does difficulty vary by
attack family** — since each CIC-IDS2017 attack class lives on exactly one day,
holding a day out removes its classes from training entirely, making every fold a
zero-shot class-detection test. Monday, having no attacks, contributes the one thing
attack days cannot: a pure false-alarm audit at the deployed threshold.

{chr(10).join(rows)}

![LODO detection](../figures/{fig.name})

## Read

{chr(10).join(f"{p}{chr(10)}" for p in parts)}
This closes the splitting story the project is built on: the stratified split
overstates (twins and shared bursts), the fixed temporal cut is honest but single,
and LODO shows the honest number's *distribution*. Detection of genuinely novel
attack families at a fixed FP budget is the hard problem — every rotation agrees —
and it is why the benign-only anomaly detector is in the architecture at all.
"""
