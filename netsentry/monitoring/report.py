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

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import SPLITS_DIRNAME
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.feature_sets import model_features
from netsentry.log import get_logger
from netsentry.models.registry import ModelBundle, latest_bundle, load_bundle
from netsentry.monitoring.detectors import (
    FeatureKS,
    ddm,
    ks_feature_tests,
    page_hinkley,
)
from netsentry.monitoring.drift import DriftReport, classify_psi, compute_drift_report
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "drift.md"
TESTS_REPORT_NAME = "drift_tests.md"


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


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic sub-sample so the stream demo stays fast without losing the shift."""
    return frame.sample(min(len(frame), n), random_state=seed) if len(frame) > n else frame


def _stream_bundle_path(settings: Settings) -> Path | None:
    """The deployed model to monitor — its score and error streams are what production
    would actually watch. Prefer the serving bundle by name, else the latest bundle."""
    models_dir = settings.paths.models_dir
    if models_dir.exists():
        serving = sorted(models_dir.glob("*serving*.joblib"))
        if serving:
            return serving[-1]
    path = settings.serving.artifact_path or latest_bundle(settings)
    return Path(path) if path is not None else None


def _score_and_error_stream(
    settings: Settings, bundle: ModelBundle, ref: pd.DataFrame, cur: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build calibrated-score and 0/1-error streams over a reference->current transition.

    Concatenating a reference (training-like) sample ahead of a current (later-day)
    sample plants a known change-point at the boundary, so Page-Hinkley and DDM can be
    read against ground truth: a well-behaved detector fires shortly *after* it, not
    before. Returns the score stream, the error stream, and the boundary index.
    """
    n = settings.monitoring.reference_rows
    ref_s, cur_s = _sample(ref, n, settings.seed), _sample(cur, n, settings.seed)
    scores = np.concatenate([bundle.attack_scores(ref_s), bundle.attack_scores(cur_s)])

    profile = str(
        bundle.metadata.get("default_threshold_profile", settings.serving.default_threshold_profile)
    )
    threshold = float(bundle.thresholds.get(profile, 0.5))
    decisions = scores >= threshold
    y_true = np.concatenate(
        [(ref_s[BINARY_TARGET] == 1).to_numpy(), (cur_s[BINARY_TARGET] == 1).to_numpy()]
    )
    errors = (decisions != y_true).astype(int)
    return scores, errors, len(ref_s)


def run_drift_tests_report(
    settings: Settings,
    *,
    reference_path: Path | None = None,
    current_path: Path | None = None,
) -> Path:
    """Significance-tested drift: per-feature KS+FDR, plus online Page-Hinkley/DDM."""
    reference, current, ref_name, cur_name = _load_frames(settings, reference_path, current_path)
    columns = [c for c in model_features(settings) if c in reference.columns]
    cfg = settings.drift_detectors
    ks = ks_feature_tests(reference, current, columns, alpha=cfg.ks_fdr_alpha)

    ph_index: int | None = None
    ddm_result = None
    boundary = 0
    path = _stream_bundle_path(settings)
    if path is not None and path.exists():
        try:
            bundle = load_bundle(path)
            scores, errors, boundary = _score_and_error_stream(settings, bundle, reference, current)
            ph_index = page_hinkley(scores, delta=cfg.ph_delta, lam=cfg.ph_lambda)
            ddm_result = ddm(
                errors,
                warn_level=cfg.ddm_warn_level,
                drift_level=cfg.ddm_drift_level,
                min_samples=cfg.ddm_min_samples,
            )
        except Exception as exc:  # the online detectors are a bonus; never fail on them
            logger.warning("Stream drift detectors skipped (%s)", exc)

    n_sig = sum(r.significant for r in ks)
    out_path = settings.paths.reports_dir / TESTS_REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_tests(settings, ks, ph_index, ddm_result, boundary, ref_name, cur_name),
        encoding="utf-8",
    )
    logger.info(
        "Wrote statistical drift report",
        extra={"path": str(out_path), "ks_significant": n_sig, "ph_index": ph_index},
    )

    with track_run(settings, "drift_tests") as run:
        run.log_metrics(
            {
                "ks_significant_features": float(n_sig),
                "ks_features_tested": float(len(ks)),
                "page_hinkley_index": float(ph_index) if ph_index is not None else -1.0,
            }
        )
        run.log_artifact(out_path)
    return out_path


def _ph_line(ph_index: int | None, boundary: int) -> str:
    if ph_index is None:
        return (
            "- **Page-Hinkley** (model-score stream): no alarm — the score stream did not "
            "sustain a mean increase past the threshold."
        )
    where = "after" if ph_index >= boundary else "before"
    return (
        f"- **Page-Hinkley** (model-score stream): alarmed at position {ph_index:,} of the "
        f"stream — {where} the reference→current boundary at {boundary:,} (an alarm just "
        "after the boundary is the detector catching the later-day shift)."
    )


def _ddm_line(ddm_result: object, boundary: int) -> str:
    if ddm_result is None:
        return "- **DDM** (model-error stream): not computed (no serving bundle to score with)."
    warn = getattr(ddm_result, "warning_index", None)
    drift = getattr(ddm_result, "drift_index", None)
    warn_s = f"warning at {warn:,}" if warn is not None else "no warning"
    drift_s = f"drift alarm at {drift:,}" if drift is not None else "no drift alarm"
    return (
        f"- **DDM** (model-error stream): {warn_s}, {drift_s} "
        f"(reference→current boundary at {boundary:,}); the error rate climbing a "
        "statistically meaningful margin above its running minimum is the alarm."
    )


def _render_tests(
    settings: Settings,
    ks: list[FeatureKS],
    ph_index: int | None,
    ddm_result: object,
    boundary: int,
    ref_name: str,
    cur_name: str,
) -> str:
    cfg = settings.drift_detectors
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_sig = sum(r.significant for r in ks)
    rows = ["| feature | KS statistic | p-value | drifted (FDR) |", "|---|---|---|---|"]
    for r in ks[: cfg.max_features_reported]:
        flag = "**yes**" if r.significant else "no"
        rows.append(f"| {r.feature} | {r.statistic:.4f} | {r.p_value:.2e} | {flag} |")
    table = "\n".join(rows)

    return f"""# NetSentry — Statistical & Online Drift Detection

_Generated {generated}. Reference: `{ref_name}` vs current: `{cur_name}`._

The [PSI drift report](drift.md) answers *how much* each feature moved. PSI is an
effect size with a rule-of-thumb cutoff, not a test, and it is computed on static
batches. This report adds the two things PSI cannot: **significance** (are the
shifts real, corrected for testing many features at once?) and **timing** (at what
point in a stream did the model's behaviour break?).

## Per-feature Kolmogorov-Smirnov tests (with Benjamini-Hochberg FDR)

A two-sample KS test per feature (reference vs current), then a Benjamini-Hochberg
procedure at FDR **{cfg.ks_fdr_alpha}** across all {len(ks)} tested features — so the
"{n_sig} drifted" count controls the expected share of false alarms, rather than
flagging ~5% of stable features by chance.

**{n_sig} / {len(ks)} features drift significantly** at FDR {cfg.ks_fdr_alpha}.

{table}

## Online change detection (when did the stream break?)

The offline test says *whether*; these say *when*. A stream is built by placing a
reference (training-era) sample ahead of a current (later-day) sample, planting a
known change-point at the boundary so each detector can be judged against ground
truth. The stream is scored by the **deployed model** — what a production monitor
would actually watch — with Page-Hinkley on its attack-score stream (no labels
needed) and DDM on its error stream (labels needed).

{_ph_line(ph_index, boundary)}
{_ddm_line(ddm_result, boundary)}

## How to read this

- **KS + FDR** is the honest multi-feature drift count: a small p-value means the two
  samples are unlikely to share a distribution, and BH keeps the multiplicity in
  check. It complements PSI — PSI ranks *magnitude*, KS certifies *significance*.
- **Page-Hinkley** and **DDM** are *online*: they consume one observation at a time
  and raise an alarm at a specific index, which is what a production monitor needs —
  not "the batch drifted" but "alert now, at flow N." Here they locate the same
  later-day boundary the temporal split embodies, from the score and error streams
  respectively.
- All three are unsupervised-to-cheap early-warning signals: KS and Page-Hinkley
  need no labels at all; DDM needs only the eventual error signal.
"""


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
