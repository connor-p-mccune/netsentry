"""Release quality gate — the pass/fail bar a model must clear before it ships.

The analysis suite *describes* the model; nothing so far *decides* about it. This
gate turns the project's definition of done into an executable check with an exit
code CI can enforce: structural honesty invariants re-asserted on the artifact that
would ship, plus configurable performance floors on the honest temporal split.

The structural checks are the point. A leakage firewall that lives only in the
feature-pipeline unit tests protects the *code*; re-checking the fitted artifact at
release time protects the *deployment* — against a config drift (say, someone flips
``encode_destination_port`` on for an experiment and ships it) that every unit test
would still pass. And one bar is deliberately inverted: a PR-AUC *above* the
too-good ceiling **fails** the gate, because on this data a near-perfect score is
overwhelmingly more likely to be leakage than brilliance. The gate encodes the
project's core habit — treat a too-good number as a bug — as machinery.

Test-set hygiene caveat, stated rather than hidden: a release gate evaluates the
frozen temporal test split, so it should run once per release candidate, not per
commit (repeated gating against one test set erodes it). In production the same
bars would run against a fresh labeled window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data import schema
from netsentry.evaluation.calibration import expected_calibration_error
from netsentry.evaluation.confidence import pr_auc
from netsentry.evaluation.metrics import operating_point, positive_scores
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import GateConfig
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

REPORT_NAME = "gate.md"


@dataclass
class GateCheck:
    """One release bar: what was required, what was measured, and the verdict."""

    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    """The full gate outcome; ``ok`` only when every check passes."""

    checks: list[GateCheck]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(not check.passed for check in self.checks)


def leaked_feature_names(feature_names: list[str], *, port_allowed: bool) -> list[str]:
    """Identifier/leaky column names that survive into the fitted feature space.

    Substring match (case-insensitive) so an encoded variant like
    ``Destination Port_80`` is caught, not just the verbatim column.
    """
    banned = list(schema.IDENTIFIER_COLUMNS)
    if not port_allowed:
        banned.append(schema.DESTINATION_PORT)
    lowered = [name.lower() for name in feature_names]
    return sorted({b for b in banned if any(b.lower() in name for name in lowered)})


def structural_checks(settings: Settings, bundle: ModelBundle) -> list[GateCheck]:
    """Honesty invariants re-asserted on the artifact that would actually ship."""
    checks: list[GateCheck] = []

    port_allowed = settings.features.encode_destination_port
    leaks = leaked_feature_names(bundle.feature_names(), port_allowed=port_allowed)
    checks.append(
        GateCheck(
            "leakage firewall",
            passed=not leaks,
            detail=(
                "no identifier/port column in the fitted feature space"
                if not leaks
                else f"leaky columns survive: {', '.join(leaks)}"
            ),
        )
    )

    if settings.thresholds.calibrate:
        checks.append(
            GateCheck(
                "calibrator attached",
                passed=bundle.calibrator is not None,
                detail=(
                    f"method={bundle.calibrator.method}"
                    if bundle.calibrator is not None
                    else "thresholds.calibrate is on but the bundle has no calibrator"
                ),
            )
        )

    expected = [f"fpr_{fpr * 100:g}pct" for fpr in settings.thresholds.fpr_targets]
    missing = [name for name in expected if name not in bundle.thresholds]
    checks.append(
        GateCheck(
            "threshold profiles",
            passed=not missing,
            detail=(
                f"all configured profiles present ({', '.join(expected)})"
                if not missing
                else f"missing profiles: {', '.join(missing)}"
            ),
        )
    )
    return checks


def performance_checks(
    cfg: GateConfig,
    *,
    pr_auc_value: float,
    prevalence: float,
    tpr_primary: float,
    primary_fpr: float,
    ece: float,
) -> list[GateCheck]:
    """Configurable metric floors (and one ceiling) on the honest split."""
    floor = cfg.min_pr_auc_lift * prevalence
    checks = [
        GateCheck(
            "PR-AUC floor",
            passed=pr_auc_value >= floor,
            detail=(
                f"PR-AUC {pr_auc_value:.3f} vs floor {floor:.3f} "
                f"({cfg.min_pr_auc_lift:g}x the {prevalence:.3f} random-ranker baseline)"
            ),
        ),
        GateCheck(
            "too-good-to-be-true ceiling",
            passed=pr_auc_value <= cfg.max_pr_auc,
            detail=(
                f"PR-AUC {pr_auc_value:.3f} <= {cfg.max_pr_auc} ceiling"
                if pr_auc_value <= cfg.max_pr_auc
                else (
                    f"PR-AUC {pr_auc_value:.4f} exceeds {cfg.max_pr_auc}: on this data that is "
                    "overwhelmingly more likely to be leakage than skill - investigate before "
                    "shipping"
                )
            ),
        ),
        GateCheck(
            "detection floor",
            passed=tpr_primary >= cfg.min_tpr_at_primary_fpr,
            detail=(
                f"TPR {tpr_primary:.1%} at the {primary_fpr:.2%} FP budget "
                f"vs floor {cfg.min_tpr_at_primary_fpr:.1%}"
            ),
        ),
        GateCheck(
            "calibration quality",
            passed=ece <= cfg.max_ece,
            detail=f"ECE {ece:.4f} vs max {cfg.max_ece}",
        ),
    ]
    return checks


def _artifact_smoke(bundle: ModelBundle, sample: object) -> GateCheck:
    """End-to-end smoke: the artifact loads and scores raw rows to sane probabilities."""
    try:
        import pandas as pd

        frame = sample if isinstance(sample, pd.DataFrame) else pd.DataFrame(sample)
        scores = bundle.attack_scores(frame)
        ok = bool(np.all(np.isfinite(scores)) and np.all((scores >= 0.0) & (scores <= 1.0)))
        detail = (
            f"scored {len(frame)} rows; probabilities in [0, 1]"
            if ok
            else "scores outside [0, 1] or non-finite"
        )
        return GateCheck("artifact smoke", passed=ok, detail=detail)
    except Exception as exc:
        return GateCheck("artifact smoke", passed=False, detail=f"scoring raised: {exc}")


def run_gate(settings: Settings) -> tuple[Path, GateResult]:
    """Fit the release candidate, run every bar, and write the gate report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)
    bundle = result.bundle

    # Metric conventions match the evaluation report exactly, so the gate's numbers
    # are the headline numbers: PR-AUC and TPR@FPR on the raw ranking (threshold
    # chosen on validation), ECE on the *calibrated* score — the probability the
    # deployment actually ships, which is what the calibration claim is about.
    raw_val = positive_scores(result.proba_val, result.classes)
    raw_test = positive_scores(result.proba_test, result.classes)
    y_val = result.y_val.astype(int)
    y_test = result.y_test.astype(int)
    calibrated_test = (
        bundle.calibrator.transform(raw_test) if bundle.calibrator is not None else raw_test
    )

    primary_fpr = settings.thresholds.primary_fpr
    op = operating_point(
        y_val, raw_val, y_test, raw_test, primary_fpr, settings.thresholds.assumed_flows_per_day
    )
    metrics = {
        "pr_auc": pr_auc(y_test, raw_test),
        "prevalence": float(np.mean(y_test)),
        "tpr_primary": float(op["tpr"]),
        "ece": expected_calibration_error(y_test, calibrated_test),
    }

    from netsentry.data.split import load_split

    test = load_split(variant, "temporal", "test")
    checks = structural_checks(settings, bundle)
    checks.append(_artifact_smoke(bundle, test.head(5)))
    checks.extend(
        performance_checks(
            settings.gate,
            pr_auc_value=metrics["pr_auc"],
            prevalence=metrics["prevalence"],
            tpr_primary=metrics["tpr_primary"],
            primary_fpr=primary_fpr,
            ece=metrics["ece"],
        )
    )
    gate = GateResult(checks)

    report = _render(settings, gate, metrics, primary_fpr)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote gate report",
        extra={"path": str(out_path), "ok": gate.ok, "failed": gate.n_failed},
    )

    with track_run(settings, "gate") as run:
        run.log_params({f"bar_{c.name.replace(' ', '_')}": c.passed for c in gate.checks})
        run.log_metrics({k: v for k, v in metrics.items()})
        run.log_metrics({"gate_ok": float(gate.ok)})
        run.log_artifact(out_path)
    return out_path, gate


def _render(
    settings: Settings, gate: GateResult, metrics: dict[str, float], primary_fpr: float
) -> str:
    rows = ["| check | verdict | detail |", "|---|---|---|"]
    for check in gate.checks:
        rows.append(f"| {check.name} | {'PASS' if check.passed else '**FAIL**'} | {check.detail} |")
    verdict = (
        "**PASS** - every bar cleared; the candidate is releasable under this policy."
        if gate.ok
        else f"**FAIL** - {gate.n_failed} bar(s) not met; the candidate must not ship."
    )
    return f"""# NetSentry - Release Quality Gate

_Synthetic stand-in; the method is the point. The honest temporal/binary release
candidate, evaluated once against the frozen temporal test split. Bars come from
config (`gate.*`); this report is written by `netsentry gate`, which exits non-zero
on failure so CI and deploy pipelines can enforce it._

## Verdict

{verdict}

{chr(10).join(rows)}

Measured: PR-AUC **{metrics["pr_auc"]:.3f}** (attack prevalence {metrics["prevalence"]:.3f}),
TPR at the {primary_fpr:.2%} FP budget **{metrics["tpr_primary"]:.1%}**, ECE
**{metrics["ece"]:.4f}**.

## What the bars encode

- **Structural checks** re-assert the honesty invariants on the artifact that would
  actually ship: no identifier/port column in the fitted feature space (the leakage
  firewall, re-checked at release rather than trusted to unit tests), a calibrator
  attached when configuration promises calibrated probabilities, every configured
  FPR profile present, and an end-to-end scoring smoke.
- **The too-good ceiling is deliberate.** A PR-AUC above {settings.gate.max_pr_auc}
  *fails* the gate: on CIC-IDS-style data a near-perfect score is overwhelmingly more
  likely to be leakage than brilliance, so the gate refuses to ship it until a human
  explains it. This is the project's "treat a too-good number as a bug" habit turned
  into machinery.
- **Floors are relative where possible.** The PR-AUC floor is a multiple of the
  attack prevalence (a random ranker's PR-AUC), so the bar transfers across datasets
  with different base rates instead of encoding one dataset's difficulty.

## Hygiene

A release gate touches the frozen test split, so it belongs at release cadence, not
per-commit (repeated evaluation against one test set slowly erodes it). In
production the same bars run against a fresh labeled window; the config is the
policy either way. Companion: `netsentry promote` decides *champion vs challenger*
(relative), this gate decides *fit to ship at all* (absolute).
"""
