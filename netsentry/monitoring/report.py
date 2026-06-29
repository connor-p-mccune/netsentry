"""Render a drift report comparing a current dataset to a reference.

The zero-argument default compares the temporal **test** split against the
temporal **train** split — "how much does later-day traffic drift from what the
model trained on?" — which complements the temporal-split evaluation story. If a
serving bundle is present it also reports *score* drift (the model's own output
distribution shift).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.data.split import SPLITS_DIRNAME
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.feature_sets import model_features
from netsentry.log import get_logger
from netsentry.models.registry import latest_bundle, load_bundle
from netsentry.monitoring.drift import DriftReport, classify_psi, compute_drift_report
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import numpy as np

    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "drift.md"


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _default_split(settings: Settings, part: str) -> Path:
    return settings.paths.data_processed / SPLITS_DIRNAME / "temporal" / f"{part}.parquet"


def _load_frames(
    settings: Settings, reference_path: Path | None, current_path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    ref_path = reference_path or _default_split(settings, "train")
    cur_path = current_path or _default_split(settings, "test")
    for path in (ref_path, cur_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `netsentry prep`, or pass --reference/--current."
            )
    return _read_frame(ref_path), _read_frame(cur_path), ref_path.name, cur_path.name


def _score_drift(
    settings: Settings, reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Model-output drift, if a serving bundle is available to score with."""
    path = settings.serving.artifact_path or latest_bundle(settings)
    if path is None or not Path(path).exists():
        return None, None
    try:
        bundle = load_bundle(Path(path))
        benign = settings.labels.benign_label
        ref = attack_probability(bundle.predict_proba(reference), bundle.classes, benign)
        cur = attack_probability(bundle.predict_proba(current), bundle.classes, benign)
        return ref, cur
    except Exception as exc:  # score drift is a bonus; never fail the report on it
        logger.warning("Score drift skipped (%s)", exc)
        return None, None


def run_drift_report(
    settings: Settings,
    *,
    reference_path: Path | None = None,
    current_path: Path | None = None,
) -> Path:
    """Compute feature/score drift, write the markdown report, log to MLflow."""
    reference, current, ref_name, cur_name = _load_frames(settings, reference_path, current_path)
    columns = [c for c in model_features(settings) if c in reference.columns]
    ref_scores, cur_scores = _score_drift(settings, reference, current)
    report = compute_drift_report(
        reference,
        current,
        columns,
        bins=settings.monitoring.psi_bins,
        moderate=settings.monitoring.psi_moderate,
        major=settings.monitoring.psi_major,
        score_reference=ref_scores,
        score_current=cur_scores,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(report, ref_name, cur_name), encoding="utf-8")
    logger.info(
        "Wrote drift report", extra={"path": str(out_path), "max_psi": round(report.max_psi, 4)}
    )

    score_metric = {"drift_score_psi": report.score_psi} if report.score_psi is not None else {}
    with track_run(settings, "drift") as run:
        run.log_metrics(
            {"drift_max_psi": report.max_psi, "drift_mean_psi": report.mean_psi, **score_metric}
        )
        run.log_artifact(out_path)
    return out_path


def _render(report: DriftReport, ref_name: str, cur_name: str) -> str:
    overall = classify_psi(report.max_psi, moderate=report.moderate, major=report.major)
    if report.score_psi is not None:
        score_line = (
            f"- **Score drift (model output PSI): {report.score_psi:.4f}** "
            f"({report.classify(report.score_psi)})"
        )
    else:
        score_line = "- Score drift: not computed (no serving bundle found)"

    ranked = sorted(report.feature_psi.items(), key=lambda kv: kv[1], reverse=True)
    rows = ["| feature | PSI | severity |", "|---|---|---|"]
    rows += [f"| {feat} | {psi:.4f} | {report.classify(psi)} |" for feat, psi in ranked[:20]]
    table = "\n".join(rows)
    drifted = report.drifted(level="moderate")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return f"""# NetSentry — Drift Report

_Generated {generated}. Reference: `{ref_name}` vs current: `{cur_name}`._

Population Stability Index (PSI) per feature. Reading: **< {report.moderate}** no
meaningful shift, **{report.moderate}-{report.major}** moderate, **>= {report.major}**
major drift worth investigating.

## Summary

- **Max feature PSI: {report.max_psi:.4f}** ({overall})
- Mean feature PSI: {report.mean_psi:.4f}
- Features with at least moderate drift: {len(drifted)} / {len(report.feature_psi)}
{score_line}

## Per-feature PSI (top 20)

{table}

## How to read this

Input drift (feature PSI) often rises long before labels arrive, so it is an
early decay signal: when a feature's live distribution diverges from training,
the model is extrapolating. Score drift (the model's own output distribution
moving) is a complementary signal. In serving, `/metrics` exposes
`netsentry_feature_drift_psi_max` / `_mean` computed over a rolling window of
requests, so the same check runs continuously in production.
"""
