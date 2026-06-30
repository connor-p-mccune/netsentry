"""Render the adversarial-evasion robustness report.

Targets the deployed serving bundle (the model an attacker would actually face),
uses later-day **test** attacks as the flows to hide and **train** benign traffic
as the attacker's notion of "normal", and writes a Markdown report plus robustness
figures. The framing is defensive: this measures the model card's
"not adversarially robust" caveat instead of leaving it asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.models.registry import latest_bundle, load_bundle
from netsentry.robustness.evasion import EvasionStudy, run_evasion_study
from netsentry.serving.bundle import build_serving_bundle
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "robustness.md"


def _load_bundle(settings: Settings):  # type: ignore[no-untyped-def]
    bundle_path = settings.serving.artifact_path or latest_bundle(settings)
    if bundle_path is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        bundle_path = build_serving_bundle(settings)
    return load_bundle(Path(bundle_path))


def _attack_and_benign(settings: Settings):  # type: ignore[no-untyped-def]
    """Later-day attacks to hide; train-side benign as the attacker's 'normal'."""
    test = load_split(settings, "temporal", "test")
    train = load_split(settings, "temporal", "train")
    benign = settings.labels.benign_label
    attack = test[test[BINARY_TARGET] == 1]
    benign_ref = train[train[MULTICLASS_TARGET] == benign]
    return attack, benign_ref


def _curve_table(xs: list[float], ys: list[float], x_label: str) -> str:
    head = f"| {x_label} | " + " | ".join(f"{x:g}" for x in xs) + " |"
    sep = "|" + "---|" * (len(xs) + 1)
    body = "| detection (TPR) | " + " | ".join(f"{y * 100:.1f}%" for y in ys) + " |"
    return "\n".join([head, sep, body])


def _render(study: EvasionStudy, settings: Settings, mimicry_fig: Path, search_fig: Path) -> str:
    base = study.baseline_detection
    mim_min = min(study.mimicry_detection)
    search_min = min(study.search_detection)
    exploit_rows = "\n".join(
        f"| {name} | {drop * 100:+.1f} pts |" for name, drop in study.top_exploitable
    )
    return f"""# NetSentry — Adversarial Evasion Robustness

_Synthetic stand-in; the methodology is the point. Operating point: **{study.profile}**
(threshold {study.threshold:.3f} on the calibrated attack probability). {study.n_attacks}
attack flows, {study.n_controllable} attacker-controllable features._

## Threat model

An attacker shapes the **controllable** parts of a malicious flow — volume, timing,
packet sizes (padding, dummy packets, added delay) — to look benign, while the
protocol-structural fields stay fixed. We measure how far that pushes the
detection rate **down** from its un-attacked baseline of **{base * 100:.1f}%**. This
is white-box on the feature space (the strong case for a defender to assume).

## Mimicry attack (shape controllable features toward benign)

Move the controllable features a fraction toward the benign centroid; `1.0` makes
them exactly average-benign. No model queries — this is "look normal" made numeric.

{_curve_table(study.mimicry_fractions, study.mimicry_detection, "mimicry fraction")}

At full mimicry, detection falls to **{mim_min * 100:.1f}%** (from {base * 100:.1f}%).

![Mimicry robustness curve](../figures/{mimicry_fig.name})

## Adaptive query search (L2 budget on controllable features)

A random-restart search for the perturbation (bounded in L2, on controllable
features only) that minimizes the model's score — a realistic adaptive attacker,
since trees are non-differentiable.

{_curve_table(study.search_budgets, study.search_detection, "L2 budget (std units)")}

At the largest budget, detection falls to **{search_min * 100:.1f}%**.

![Search robustness curve](../figures/{search_fig.name})

## Most exploitable features

Detection drop when the attacker fully mimics **one** controllable feature alone —
where the detector is most spoofable (cross-reference the SHAP global importances):

| feature | detection drop |
|---|---|
{exploit_rows}

## Defensive takeaways

- The supervised classifier leans on attacker-controllable volume/timing features,
  so a determined evader degrades it — exactly why NetSentry pairs it with a
  **benign-only anomaly detector**: mimicry that flattens an attack toward the
  benign manifold is the regime where reconstruction error and isolation depth
  still carry signal the classifier has lost.
- Robust hardening directions: adversarial training (augment with mimicry samples),
  feature-set restriction away from the most spoofable columns above, and
  monotonic/known-direction constraints (an attacker can usually only *inflate*
  volume, not reduce it below the real attack footprint).
- This converts the model card's "not adversarially robust" caveat from an
  assertion into a measured curve — the honest way to state a limitation.
"""


def run_robustness_report(settings: Settings) -> Path:
    """Run the evasion study against the deployed bundle and write the report."""
    bundle = _load_bundle(settings)
    attack, benign_ref = _attack_and_benign(settings)
    study = run_evasion_study(settings, bundle, attack, benign_ref)

    figures_dir = settings.paths.figures_dir
    mimicry_fig = plots.plot_lines(
        {"mimicry": (np.asarray(study.mimicry_fractions), np.asarray(study.mimicry_detection))},
        xlabel="Mimicry fraction toward benign",
        ylabel="Detection rate (TPR)",
        title="Evasion robustness — mimicry",
        out_path=figures_dir / "robustness_mimicry.png",
    )
    search_fig = plots.plot_lines(
        {"query search": (np.asarray(study.search_budgets), np.asarray(study.search_detection))},
        xlabel="L2 perturbation budget (std units)",
        ylabel="Detection rate (TPR)",
        title="Evasion robustness — adaptive search",
        out_path=figures_dir / "robustness_search.png",
    )

    report = _render(study, settings, mimicry_fig, search_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote robustness report", extra={"path": str(out_path)})

    with track_run(settings, "robustness") as run:
        run.log_metrics(
            {
                "baseline_detection": study.baseline_detection,
                "mimicry_min_detection": float(min(study.mimicry_detection)),
                "search_min_detection": float(min(study.search_detection)),
            }
        )
        for fig in (mimicry_fig, search_fig, out_path):
            run.log_artifact(fig)
    return out_path
