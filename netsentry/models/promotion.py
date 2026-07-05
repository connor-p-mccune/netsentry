"""Champion/challenger promotion — the decision layer between training and serving.

Training produces candidates; serving needs exactly one model. What sits between
them in a real ML platform is a *promotion gate*: score the challenger and the
reigning champion on the same frozen evaluation rows, bootstrap the difference, and
promote only under an explicit, configured policy. This module is that gate.

Statistics: the comparison is a **paired** bootstrap (one resample of rows scores
both models), so shared sampling noise cancels and the interval reflects the
difference between models — much tighter than comparing two independent CIs. The
non-inferiority margins are not hand-picked: they are calibrated from the seed-
sensitivity audit (`netsentry seeds`), which measures how much a metric moves when
*nothing* changes but the seed. A promotion decided inside that band would be a
decision about luck.

Policy: two named policies, because they answer different operational questions.

- ``non_inferiority`` (default): promote unless the challenger is credibly *worse*
  than the champion by more than the margin. Right for routine retrains on drifting
  traffic — freshness has measured value here (see the streaming study), so parity
  rolls forward.
- ``superiority``: promote only if the challenger is credibly *better* (delta CI
  above zero). Right for risky swaps — new architectures, new feature sets — where
  churn has a cost and parity is not a reason to move.

On promotion the challenger is snapshotted to ``models/champion.joblib`` and a
pointer (name, SHA-256, timestamp) is written beside it, so later retrains that
overwrite the working bundle path cannot silently rewrite the champion. Every
decision — either way — is appended to ``models/promotion_history.jsonl``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation.confidence import DiffResult, paired_diff, pr_auc, tpr_at_threshold
from netsentry.governance.provenance import sha256_file
from netsentry.log import get_logger
from netsentry.models.registry import ModelBundle, load_bundle
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import numpy as np

    from netsentry.config import Settings
    from netsentry.config.settings import PromotionConfig

logger = get_logger(__name__)

REPORT_NAME = "promotion.md"
CHAMPION_BUNDLE = "champion.joblib"
CHAMPION_POINTER = "champion.json"
HISTORY_NAME = "promotion_history.jsonl"
DEFAULT_CHALLENGER = "supervised_binary_temporal.joblib"


@dataclass
class PromotionDecision:
    """The outcome of one champion/challenger comparison."""

    promote: bool
    reason: str
    challenger: str
    champion: str | None  # None when the registry was empty (bootstrap promotion)
    pr_delta: DiffResult | None
    tpr_delta: DiffResult | None


def decide_promotion(
    cfg: PromotionConfig,
    pr_delta: DiffResult | None,
    tpr_delta: DiffResult | None,
) -> tuple[bool, str]:
    """Apply the configured policy to the measured deltas (challenger - champion).

    ``pr_delta is None`` means there is no champion yet: the challenger seeds the
    registry. Otherwise the challenger must clear the non-inferiority margins, and
    under the ``superiority`` policy must additionally be credibly better.
    """
    if pr_delta is None:
        return True, "no champion on record - the challenger seeds the registry"
    if pr_delta.low <= -cfg.metric_margin:
        return False, (
            f"PR-AUC delta CI lower bound {pr_delta.low:+.4f} breaches the "
            f"-{cfg.metric_margin:g} non-inferiority margin - regression risk"
        )
    if cfg.require_tpr_non_inferior and tpr_delta is not None and tpr_delta.low <= -cfg.tpr_margin:
        return False, (
            f"TPR@FPR delta CI lower bound {tpr_delta.low:+.4f} breaches the "
            f"-{cfg.tpr_margin:g} non-inferiority margin - detection regression risk"
        )
    if cfg.policy == "superiority":
        if pr_delta.low > 0.0:
            return True, (
                f"credibly better: PR-AUC delta {pr_delta.diff:+.4f}, "
                f"CI [{pr_delta.low:+.4f}, {pr_delta.high:+.4f}] excludes zero"
            )
        return False, (
            f"not proven better (PR-AUC delta {pr_delta.diff:+.4f}, CI lower bound "
            f"{pr_delta.low:+.4f} <= 0) - superiority policy holds the champion"
        )
    return True, (
        f"non-inferior within margins (PR-AUC delta {pr_delta.diff:+.4f}, CI lower bound "
        f"{pr_delta.low:+.4f} > -{cfg.metric_margin:g}) - freshness rolls forward"
    )


def _primary_profile(settings: Settings) -> str:
    return f"fpr_{settings.thresholds.primary_fpr * 100:g}pct"


def _bundle_summary(path: Path, bundle: ModelBundle) -> dict[str, Any]:
    """Identity a decision can be audited against later."""
    meta = bundle.metadata
    raw_n = meta.get("n_train", 0)
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "backend": str(meta.get("backend", "?")),
        "task": str(meta.get("task", "?")),
        "split_strategy": str(meta.get("split_strategy", "?")),
        "n_train": int(raw_n) if isinstance(raw_n, (int, float)) else 0,
        "created_at": str(meta.get("created_at", "?")),
    }


def _tpr_deltas(
    settings: Settings,
    y: np.ndarray,
    champion: tuple[ModelBundle, np.ndarray],
    challenger: tuple[ModelBundle, np.ndarray],
) -> DiffResult | None:
    """Paired delta of detection at each model's OWN validation-chosen threshold.

    Each bundle carries thresholds on its own calibrated scale, so the comparison is
    deployment-faithful: what each model would actually alert on. If either bundle
    lacks the primary profile the comparison is skipped rather than faked at 0.5.
    """
    profile = _primary_profile(settings)
    champ_bundle, champ_scores = champion
    chall_bundle, chall_scores = challenger
    thr_a = champ_bundle.thresholds.get(profile)
    thr_b = chall_bundle.thresholds.get(profile)
    if thr_a is None or thr_b is None:
        return None
    return paired_diff(
        y,
        champ_scores,
        chall_scores,
        tpr_at_threshold(float(thr_a)),
        metric_b=tpr_at_threshold(float(thr_b)),
        n_boot=settings.promotion.n_boot,
        alpha=settings.evaluation.bootstrap_alpha,
        seed=settings.seed,
    )


def _append_history(settings: Settings, record: dict[str, Any]) -> None:
    history = settings.paths.models_dir / HISTORY_NAME
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _install_champion(settings: Settings, challenger_path: Path) -> None:
    """Snapshot the challenger as the new champion and write the pointer."""
    models_dir = settings.paths.models_dir
    snapshot = models_dir / CHAMPION_BUNDLE
    if challenger_path.resolve() != snapshot.resolve():
        shutil.copy2(challenger_path, snapshot)
    pointer = {
        "source": challenger_path.name,
        "bundle": snapshot.name,
        "sha256": sha256_file(snapshot),
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    (models_dir / CHAMPION_POINTER).write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    logger.info("Installed champion", extra={"source": challenger_path.name})


def _load_champion(settings: Settings) -> Path | None:
    """Resolve the current champion snapshot via the pointer, verifying integrity."""
    models_dir = settings.paths.models_dir
    pointer_path = models_dir / CHAMPION_POINTER
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    bundle_path = models_dir / str(pointer.get("bundle", CHAMPION_BUNDLE))
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Champion pointer names {bundle_path.name} but the snapshot is missing."
        )
    recorded = str(pointer.get("sha256", ""))
    actual = sha256_file(bundle_path)
    if recorded and recorded != actual:
        raise RuntimeError(
            f"Champion snapshot hash mismatch (pointer {recorded[:12]}..., file "
            f"{actual[:12]}...): the champion was modified outside promotion. "
            "Re-promote deliberately or restore the snapshot."
        )
    return bundle_path


def run_promotion(
    settings: Settings,
    challenger_path: Path | None = None,
    champion_path: Path | None = None,
) -> tuple[Path, PromotionDecision]:
    """Compare challenger vs champion on the frozen temporal test; decide; report."""
    models_dir = settings.paths.models_dir
    challenger_path = challenger_path or (models_dir / DEFAULT_CHALLENGER)
    if not challenger_path.exists():
        raise FileNotFoundError(
            f"No challenger bundle at {challenger_path}. Train one with "
            "`netsentry train supervised` or pass --challenger."
        )
    champion_path = champion_path or _load_champion(settings)

    challenger = load_bundle(challenger_path)
    test = load_split(settings, "temporal", "test")
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pr_delta: DiffResult | None = None
    tpr_delta: DiffResult | None = None
    champion_summary: dict[str, Any] | None = None
    if champion_path is not None:
        champion = load_bundle(champion_path)
        champion_summary = _bundle_summary(champion_path, champion)
        s_champ = champion.attack_scores(test)
        s_chall = challenger.attack_scores(test)
        pr_delta = paired_diff(
            y_test,
            s_champ,
            s_chall,
            pr_auc,
            n_boot=settings.promotion.n_boot,
            alpha=settings.evaluation.bootstrap_alpha,
            seed=settings.seed,
        )
        tpr_delta = _tpr_deltas(settings, y_test, (champion, s_champ), (challenger, s_chall))

    promote, reason = decide_promotion(settings.promotion, pr_delta, tpr_delta)
    decision = PromotionDecision(
        promote=promote,
        reason=reason,
        challenger=challenger_path.name,
        champion=champion_summary["name"] if champion_summary else None,
        pr_delta=pr_delta,
        tpr_delta=tpr_delta,
    )
    if promote:
        _install_champion(settings, challenger_path)

    challenger_summary = _bundle_summary(challenger_path, challenger)
    _append_history(
        settings,
        {
            "at": datetime.now(UTC).isoformat(),
            "decision": "promote" if promote else "hold",
            "reason": reason,
            "policy": settings.promotion.policy,
            "challenger": challenger_summary,
            "champion": champion_summary,
            "pr_delta": pr_delta.__dict__ if pr_delta else None,
            "tpr_delta": tpr_delta.__dict__ if tpr_delta else None,
        },
    )

    report = _render(settings, decision, challenger_summary, champion_summary)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Promotion decided",
        extra={"decision": "promote" if promote else "hold", "path": str(out_path)},
    )

    with track_run(settings, "promotion") as run:
        run.log_params(
            {
                "policy": settings.promotion.policy,
                "challenger": challenger_path.name,
                "champion": champion_summary["name"] if champion_summary else "none",
            }
        )
        if pr_delta is not None:
            run.log_metrics({"pr_auc_delta": pr_delta.diff, "pr_auc_delta_low": pr_delta.low})
        if tpr_delta is not None:
            run.log_metrics({"tpr_delta": tpr_delta.diff, "tpr_delta_low": tpr_delta.low})
        run.log_metrics({"promoted": float(promote)})
        run.log_artifact(out_path)
    return out_path, decision


def _fmt_delta(delta: DiffResult | None, unit: str = "") -> str:
    if delta is None:
        return "not compared"
    return (
        f"{delta.diff:+.4f}{unit} (95% CI [{delta.low:+.4f}, {delta.high:+.4f}], "
        f"p(challenger <= champion) = {delta.p_value:.3f})"
    )


def _identity_rows(summary: dict[str, Any] | None, role: str) -> str:
    if summary is None:
        return f"| {role} | - (registry empty) | - | - | - |"
    return (
        f"| {role} | `{summary['name']}` | {summary['backend']} / {summary['task']} "
        f"/ {summary['split_strategy']} | {summary['n_train']:,} | "
        f"`{summary['sha256'][:12]}...` |"
    )


def _render(
    settings: Settings,
    decision: PromotionDecision,
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
) -> str:
    cfg = settings.promotion
    verdict = "**PROMOTE**" if decision.promote else "**HOLD**"
    profile = _primary_profile(settings)
    return f"""# NetSentry - Champion/Challenger Promotion

_Synthetic stand-in; the method is the point. Challenger and champion scored on the
**same** frozen temporal test rows; deltas are paired-bootstrap (one resample scores
both models, so shared sampling noise cancels). Written by `netsentry promote`,
which exits non-zero on HOLD so a deploy pipeline can branch on the decision._

## Decision

{verdict} - {decision.reason}

| role | bundle | backend / task / split | train rows | sha256 |
|---|---|---|---|---|
{_identity_rows(champion, "champion")}
{_identity_rows(challenger, "challenger")}

| metric | delta (challenger - champion) |
|---|---|
| PR-AUC | {_fmt_delta(decision.pr_delta)} |
| TPR at {profile} (each model's own threshold) | {_fmt_delta(decision.tpr_delta)} |

## The policy, and why the margins are not hand-picked

Active policy: **{cfg.policy}** (margin {cfg.metric_margin:g} PR-AUC,
{cfg.tpr_margin:g} TPR).

- **Margins come from measurement.** The seed-sensitivity audit
  ([seed_variance.md](seed_variance.md)) measures how much these metrics move when
  *nothing* changes but the training seed. The non-inferiority margins are set just
  above that noise floor, so the gate cannot promote or demote on training luck.
- **`non_inferiority`** (default) rolls a routine retrain forward unless it is
  credibly *worse*: on drifting traffic, freshness has measured value (the streaming
  study shows retrained models recovering later-day detection), so parity is a
  reason to move, not to hold.
- **`superiority`** demands the delta CI exclude zero - the right bar for risky
  swaps (new architecture, new feature set) where churn itself has a cost.

Every decision is appended to `models/promotion_history.jsonl`; on promotion the
challenger is snapshotted to `models/{CHAMPION_BUNDLE}` with a SHA-256 pointer, so a
later retrain overwriting the working bundle cannot silently rewrite the champion
(and a tampered snapshot fails the pointer check loudly).

## Hygiene

Promotion decisions reuse the frozen temporal test split, so they belong at release
cadence; in production this comparison runs on a fresh labeled window (or in shadow
- see the serving shadow-challenger, which produces exactly the paired scores this
report needs, on live traffic). Companion gates: `netsentry gate` decides *fit to
ship at all* (absolute bars); this decides *which of two ships* (relative).
"""
