"""Why is this flow anomalous? Per-feature attribution for the anomaly detector.

The supervised model already answers "why" — every prediction returns its SHAP top
features. The **anomaly** side (the "detect the unknown" component) does not: it emits
a single reconstruction-error / isolation score, and a SOC analyst handed a bare
"anomaly = 0.83" has nothing to act on. This module is the unsupervised mirror of
SHAP — it names *which behaviours* made a flow look abnormal.

The method is deliberately **model-agnostic occlusion**, so it explains whichever
detector is deployed (the always-available Isolation Forest or the autoencoder):
for a flagged flow, each feature in turn is reset to its **benign** reference value
and the flow is re-scored. The drop in the anomaly score is that feature's
contribution — literally "if this one behaviour had looked normal, how much less
anomalous would the flow be?". That is exactly the question an analyst triaging the
flag is asking, and unlike a gradient it needs no access to the model internals.

Because occlusion attributions can be a just-so story, the report **validates** them
the way the XAI literature does — a deletion/faithfulness check: occluding the
top-attributed features must drop the anomaly score far more than occluding the same
number of random features. If it does not, the explanation is not faithful and the
report says so. Same honesty discipline as the importance-stability audit that guards
the supervised explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.anomaly import AnomalyDetector, build_anomaly_detector
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "anomaly_explain.md"


def choose_detector_kind(settings: Settings) -> str:
    """The detector to explain: the autoencoder when Torch is present, else iForest."""
    kinds = settings.anomaly.detectors
    if "autoencoder" in kinds and is_available("torch"):
        return "autoencoder"
    return "iforest"


def occlusion_attributions(
    detector: AnomalyDetector, x: np.ndarray, benign_reference: np.ndarray
) -> np.ndarray:
    """Per-feature contribution to each row's anomaly score, by benign occlusion.

    For each feature ``j``, reset column ``j`` to its benign reference value and
    re-score; the score it loses is that feature's contribution (positive == the
    feature was pushing the flow toward "anomalous"). Model-agnostic: it only calls
    ``detector.score``.
    """
    x = np.asarray(x, dtype=float)
    base = detector.score(x)
    contrib = np.zeros_like(x)
    for j in range(x.shape[1]):
        occluded = x.copy()
        occluded[:, j] = benign_reference[j]
        contrib[:, j] = base - detector.score(occluded)
    return contrib


def faithfulness_check(
    detector: AnomalyDetector,
    x: np.ndarray,
    benign_reference: np.ndarray,
    contrib: np.ndarray,
    k: int,
    seed: int,
) -> tuple[float, float]:
    """Deletion test: mean score drop from occluding the top-k vs k random features.

    A faithful attribution concentrates the score in the features it names, so the
    top-k drop should dominate the random-k drop.
    """
    x = np.asarray(x, dtype=float)
    base = detector.score(x)
    k = min(k, x.shape[1])
    order = np.argsort(-contrib, axis=1)  # most-attributed first, per row
    rng = np.random.default_rng(seed)

    x_top = x.copy()
    x_rand = x.copy()
    for i in range(len(x)):
        x_top[i, order[i, :k]] = benign_reference[order[i, :k]]
        rand_idx = rng.choice(x.shape[1], size=k, replace=False)
        x_rand[i, rand_idx] = benign_reference[rand_idx]
    top_drop = float(np.mean(base - detector.score(x_top)))
    rand_drop = float(np.mean(base - detector.score(x_rand)))
    return top_drop, rand_drop


@dataclass
class AnomalyExplanation:
    """The attribution study's outputs: per-class drivers + a faithfulness check."""

    detector_kind: str
    target_fpr: float
    n_flagged: int
    feature_names: list[str]
    global_ranking: list[tuple[str, float]]  # (feature, mean |contribution|) desc
    per_class: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    top_drop: float = 0.0
    rand_drop: float = 0.0
    faithfulness_k: int = 0

    @property
    def faithfulness_ratio(self) -> float:
        """How much more the named features move the score than random ones."""
        return self.top_drop / self.rand_drop if self.rand_drop > 1e-12 else float("inf")


def _top_features(
    contrib: np.ndarray, names: list[str], top_k: int, use_abs: bool = False
) -> list[tuple[str, float]]:
    """Mean contribution per feature -> the top-k (feature, value) pairs, desc."""
    mean = np.mean(np.abs(contrib) if use_abs else contrib, axis=0)
    order = np.argsort(-mean)[:top_k]
    return [(names[j], float(mean[j])) for j in order]


def run_anomaly_explanation(settings: Settings) -> AnomalyExplanation:
    """Fit the benign-only detector, flag temporal-test attacks, and attribute them."""
    cfg = settings.anomaly_explain
    seed_everything(settings.seed)
    benign = settings.labels.benign_label

    train = load_split(settings, "temporal", "train")
    val = load_split(settings, "temporal", "val")
    test = load_split(settings, "temporal", "test")
    benign_train = train[train[MULTICLASS_TARGET] == benign]
    benign_val = val[val[MULTICLASS_TARGET] == benign]

    pipeline = build_pipeline(settings)
    pipeline.fit(benign_train)  # benign-only, train-only: leakage-safe by construction
    names = [
        display_feature_name(n) for n in pipeline.named_steps["features"].get_feature_names_out()
    ]

    x_benign_train = np.asarray(pipeline.transform(benign_train))
    kind = choose_detector_kind(settings)
    detector = build_anomaly_detector(settings, kind).fit(x_benign_train)
    detector.calibrate_threshold(
        np.asarray(pipeline.transform(benign_val)), settings.anomaly.target_fpr
    )
    benign_reference = np.median(x_benign_train, axis=0)

    x_test = np.asarray(pipeline.transform(test))
    flagged = detector.is_anomaly(x_test)
    attack_mask = test[BINARY_TARGET].to_numpy() == 1
    idx = np.where(flagged & attack_mask)[0]  # flagged true attacks — the flags an analyst works

    explanation = AnomalyExplanation(
        detector_kind=kind,
        target_fpr=settings.anomaly.target_fpr,
        n_flagged=len(idx),
        feature_names=names,
        global_ranking=[],
    )
    if len(idx) == 0:
        logger.warning("Anomaly explainer: no attack flows flagged at the target FPR.")
        return explanation

    rng = np.random.default_rng(settings.seed)
    if len(idx) > cfg.max_explained:
        idx = rng.choice(idx, size=cfg.max_explained, replace=False)
    x_flagged = x_test[idx]
    contrib = occlusion_attributions(detector, x_flagged, benign_reference)

    explanation.global_ranking = _top_features(contrib, names, cfg.report_features, use_abs=True)
    labels = test[MULTICLASS_TARGET].to_numpy()[idx]
    for cls in sorted(set(labels)):
        rows = contrib[labels == cls]
        if len(rows) >= cfg.min_class_flags:
            explanation.per_class[str(cls)] = _top_features(rows, names, cfg.top_k)

    explanation.faithfulness_k = min(cfg.faithfulness_k, x_flagged.shape[1])
    explanation.top_drop, explanation.rand_drop = faithfulness_check(
        detector, x_flagged, benign_reference, contrib, cfg.faithfulness_k, settings.seed
    )
    logger.info(
        "Anomaly attribution complete",
        extra={
            "detector": kind,
            "n_flagged": explanation.n_flagged,
            "faithfulness_ratio": round(explanation.faithfulness_ratio, 2),
        },
    )
    return explanation


def _global_table(explanation: AnomalyExplanation) -> str:
    rows = ["| rank | feature | mean abs. contribution |", "|---|---|---|"]
    for i, (feat, val) in enumerate(explanation.global_ranking, 1):
        rows.append(f"| {i} | {feat} | {val:.4f} |")
    return "\n".join(rows)


def _per_class_table(explanation: AnomalyExplanation) -> str:
    if not explanation.per_class:
        return "_No attack class had enough flagged flows to profile individually._"
    rows = ["| attack class | flags explained by (top features, desc) |", "|---|---|"]
    for cls, feats in explanation.per_class.items():
        names = ", ".join(f"{f}" for f, _ in feats)
        rows.append(f"| {cls} | {names} |")
    return "\n".join(rows)


def _read(explanation: AnomalyExplanation) -> str:
    ratio = explanation.faithfulness_ratio
    if explanation.n_flagged == 0:
        return (
            "The detector flagged **no** attack flows at this false-positive budget on the "
            "stand-in, so there is nothing to attribute — the same known-vs-novel coverage gap "
            "the per-class slices report, seen from the anomaly side."
        )
    faithful = ratio >= 2.0
    lead = (
        f"Occluding the **top-{explanation.faithfulness_k}** attributed features drops the anomaly "
        f"score {ratio:.1f}x more than occluding the same number of random features"
    )
    if faithful:
        verdict = (
            f"{lead} — the attributions are **faithful**: the features the explainer names really "
            "are the ones carrying the flag, not a plausible-sounding story. So an analyst handed "
            f"an anomaly flag on the {explanation.detector_kind} detector gets an actionable "
            "reason (the specific behaviours that reconstruct/isolate poorly), the unsupervised "
            "counterpart to the SHAP top-features the supervised side already returns."
        )
    else:
        verdict = (
            f"{lead} — a ratio near 1 would mean the attributions are **not** faithful (the named "
            "features do not actually carry the score), and it is reported as-is rather than "
            "smoothed over. On a real detector with sharper benign structure the concentration is "
            "typically stronger."
        )
    return verdict


def _render(explanation: AnomalyExplanation, fig: Path) -> str:
    return f"""# NetSentry - Explaining the Anomaly Flag (why is this flow abnormal?)

_Synthetic stand-in; the methodology is the point. The benign-only **{explanation.detector_kind}**
detector, calibrated to a {explanation.target_fpr:.0%} benign false-positive rate, run on the
honest **temporal** test split. {explanation.n_flagged:,} flagged true-attack flows were attributed
by model-agnostic benign occlusion._

The supervised model returns its SHAP top features on every prediction; the anomaly
detector — the "detect the unknown" component — emits only a score. This closes that
gap: for each flagged flow, every feature is reset to its **benign** reference value
and the flow is re-scored, and the drop in the anomaly score is that feature's
contribution ("if this behaviour had looked normal, how much less anomalous would the
flow be?"). It is model-agnostic, so it explains whichever detector ships.

## Which behaviours drive the flags (global)

{_global_table(explanation)}

![Top anomaly-driving features]({fig.as_posix()})

## By attack class

{_per_class_table(explanation)}

## Faithfulness (is the explanation real?)

An attribution is only worth showing an analyst if the features it names actually
carry the score. Deletion check — mean anomaly-score drop when occluding, per flow:

| occlude | mean score drop |
|---|---|
| top-{explanation.faithfulness_k} attributed features | {explanation.top_drop:.4f} |
| {explanation.faithfulness_k} random features | {explanation.rand_drop:.4f} |

{_read(explanation)}

## Scope

- Occlusion resets each feature independently, so it shares partial dependence's
  independence caveat: it prices each behaviour's marginal contribution, not feature
  interactions. It is a triage aid — the anomaly analogue of the SHAP top-features —
  not a causal decomposition.
- The benign reference is the per-feature **median** of the benign training flows (in
  the fitted pipeline's standardized space), so "normal" means the center of the
  traffic the detector learned, not any single flow.
"""


def run_anomaly_explain_report(settings: Settings) -> Path:
    """Run the anomaly-attribution study and write the report + figure."""
    explanation = run_anomaly_explanation(settings)

    if explanation.global_ranking:
        feats = [f for f, _ in explanation.global_ranking]
        vals = [v for _, v in explanation.global_ranking]
    else:
        feats, vals = ["(no flags)"], [0.0]
    fig = plots.plot_barh(
        feats,
        vals,
        xlabel="mean |contribution| to the anomaly score",
        title=f"Anomaly-driving features ({explanation.detector_kind})",
        out_path=settings.paths.figures_dir / "anomaly_explain.png",
    )

    report = _render(explanation, Path("..") / "figures" / fig.name)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote anomaly-explanation report", extra={"path": str(out_path)})

    with track_run(settings, "anomaly_explain") as run:
        run.log_metrics(
            {
                "n_flagged": float(explanation.n_flagged),
                "faithfulness_top_drop": explanation.top_drop,
                "faithfulness_rand_drop": explanation.rand_drop,
                "faithfulness_ratio": (
                    explanation.faithfulness_ratio
                    if np.isfinite(explanation.faithfulness_ratio)
                    else -1.0
                ),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path
