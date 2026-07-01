"""Auto-generate a model-card spec sheet from the deployed artifact.

The hand-written ``docs/MODEL_CARD.md`` carries the narrative (intended use,
limitations, ethics); this generates the *factual* half straight from the trained
bundle so it can never drift from what is actually shipped — backend, classes,
calibration, threshold profiles, attached components, ATT&CK coverage, provenance.
Governance automation: regenerate it whenever the model changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.intel.attack_mapping import coverage_summary
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

REPORT_NAME = "model_card_generated.md"


def _meta(bundle: ModelBundle, key: str, default: object = "—") -> object:
    value = bundle.metadata.get(key, default)
    return value if value not in (None, "") else default


def render_model_card(bundle: ModelBundle) -> str:
    """Render the spec-sheet markdown from a bundle's metadata + attachments."""
    meta = bundle.metadata
    classes = [str(c) for c in bundle.classes]
    raw_cal = meta.get("calibration")
    calibration: dict[str, object] = raw_cal if isinstance(raw_cal, dict) else {}
    cal_method = calibration.get("method") if calibration.get("enabled") else "none"

    threshold_rows = (
        "\n".join(f"| {name} | {value:.4f} |" for name, value in sorted(bundle.thresholds.items()))
        or "| — | — |"
    )

    has_anomaly = "yes" if bundle.anomaly_detector is not None else "no"
    has_conformal = "yes" if isinstance(meta.get("conformal"), dict) else "no"
    has_drift = "yes" if isinstance(meta.get("drift_reference"), dict) else "no"
    cov = coverage_summary()

    return f"""# NetSentry — Model Card (auto-generated)

_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from the deployed bundle.
This is the factual spec sheet; see [`MODEL_CARD.md`](../MODEL_CARD.md) for intended
use, limitations, and ethics, and [`reports/evaluation.md`](evaluation.md) for the
honest metrics._

## Artifact

| field | value |
|---|---|
| version | {_meta(bundle, "version")} |
| task | {_meta(bundle, "task")} |
| training split | {_meta(bundle, "split_strategy")} |
| backend | {_meta(bundle, "backend")} |
| features | {_meta(bundle, "n_features")} |
| training rows | {_meta(bundle, "n_train")} |
| created | {_meta(bundle, "created_at")} |

## Classes ({len(classes)})

{", ".join(classes)}

## Calibration & operating points

- Probability calibration: **{cal_method}**
- Decision-threshold profiles (calibrated attack probability):

| profile | threshold |
|---|---|
{threshold_rows}

## Attached components

| component | present |
|---|---|
| benign-only anomaly detector | {has_anomaly} |
| conformal prediction set | {has_conformal} |
| drift self-monitoring reference | {has_drift} |

## Threat coverage

Detected attack classes map to **{len(cov.tactics)} MITRE ATT&CK tactics** and
**{len(cov.techniques)} techniques** — see [`reports/mitre.md`](mitre.md).

## Reproduce

Regenerate this card from the current artifact with `netsentry modelcard`; regenerate
the metrics/robustness/calibration evidence with `netsentry analyze`.
"""


def generate_model_card(settings: Settings) -> Path:
    """Load the deployed bundle and write the auto-generated model card."""
    from netsentry.models.registry import latest_bundle, load_bundle
    from netsentry.serving.bundle import build_serving_bundle

    bundle_path = settings.serving.artifact_path or latest_bundle(settings)
    if bundle_path is None:
        bundle_path = build_serving_bundle(settings)
    bundle = load_bundle(Path(bundle_path))

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_model_card(bundle), encoding="utf-8")
    logger.info("Wrote auto-generated model card", extra={"path": str(out_path)})
    return out_path
