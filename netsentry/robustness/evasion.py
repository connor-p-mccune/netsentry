"""Feature-space evasion attacks against the supervised detector.

Two attackers, both operating in the model's standardized feature space and
restricted to a configurable set of **controllable** features (volume, timing,
sizes — what an attacker can pad/delay/inflate without breaking the attack):

1. **Mimicry** (model-agnostic, semantic): move the controllable features a
   fraction ``alpha`` toward the benign centroid. ``alpha = 1`` makes the
   controllable half of the flow look exactly average-benign. This needs no model
   access — it is what "shape your traffic to look normal" means numerically.
2. **Query search** (adaptive): for an ``L2`` budget ``epsilon`` on the
   controllable features, random-restart search for the perturbation that
   minimizes the model's attack probability. Trees are non-differentiable, so a
   query-based search is the right tool and models a realistic adaptive attacker.

The output is a **robustness curve**: detection rate (TPR at the chosen operating
threshold) versus attacker effort. A steep drop means the detector leans on
features the attacker controls; the flat part is the robust residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation.metrics import attack_probability
from netsentry.log import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)


def base_feature_name(name: str) -> str:
    """Strip a ColumnTransformer branch prefix (``numeric__Flow Duration``)."""
    return name.split("__", 1)[1] if "__" in name else name


def controllable_indices(feature_names: list[str], controllable: list[str]) -> np.ndarray:
    """Indices of the post-transform features the attacker is allowed to move."""
    wanted = set(controllable)
    return np.array(
        [i for i, name in enumerate(feature_names) if base_feature_name(name) in wanted], dtype=int
    )


def mimicry_perturb(
    x: np.ndarray, centroid: np.ndarray, ctrl_idx: np.ndarray, fraction: float
) -> np.ndarray:
    """Linearly interpolate the controllable features toward the benign centroid."""
    adv = np.array(x, dtype=float, copy=True)
    if len(ctrl_idx):
        adv[:, ctrl_idx] = (1.0 - fraction) * x[:, ctrl_idx] + fraction * centroid[ctrl_idx]
    return adv


@dataclass
class EvasionStudy:
    """Robustness-curve results for both attackers at the chosen operating point."""

    profile: str
    threshold: float
    baseline_detection: float
    n_attacks: int
    n_controllable: int
    mimicry_fractions: list[float]
    mimicry_detection: list[float]
    search_budgets: list[float]
    search_detection: list[float]
    top_exploitable: list[tuple[str, float]] = field(default_factory=list)


def attack_scores_transformed(bundle: ModelBundle, x_transformed: np.ndarray) -> np.ndarray:
    """Calibrated attack probability from already-transformed features."""
    proba = np.asarray(bundle.model.predict_proba(x_transformed))
    benign = str(bundle.metadata.get("benign_label", "BENIGN"))
    raw = attack_probability(proba, bundle.classes, benign)
    return bundle.calibrator.transform(raw) if bundle.calibrator is not None else raw


def _detection_rate(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores >= threshold)) if len(scores) else 0.0


def _mimicry_curve(
    bundle: ModelBundle,
    x_attack: np.ndarray,
    centroid: np.ndarray,
    ctrl_idx: np.ndarray,
    fractions: list[float],
    threshold: float,
) -> list[float]:
    detection = []
    for frac in fractions:
        adv = mimicry_perturb(x_attack, centroid, ctrl_idx, frac)
        detection.append(_detection_rate(attack_scores_transformed(bundle, adv), threshold))
    return detection


def _search_curve(
    bundle: ModelBundle,
    x_attack: np.ndarray,
    ctrl_idx: np.ndarray,
    budgets: list[float],
    iterations: int,
    threshold: float,
    rng: np.random.Generator,
) -> list[float]:
    """Random-restart query search: best (lowest) attack score within each budget."""
    n, d = x_attack.shape
    detection = []
    for eps in budgets:
        if eps == 0.0 or len(ctrl_idx) == 0:
            detection.append(
                _detection_rate(attack_scores_transformed(bundle, x_attack), threshold)
            )
            continue
        best = attack_scores_transformed(bundle, x_attack)
        for _ in range(iterations):
            direction = np.zeros((n, d))
            step = rng.standard_normal((n, len(ctrl_idx)))
            norms = np.linalg.norm(step, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            # Random radius in [0, eps] so the ball interior is searched, not just its shell.
            radius = eps * rng.uniform(size=(n, 1))
            direction[:, ctrl_idx] = step / norms * radius
            trial = attack_scores_transformed(bundle, x_attack + direction)
            best = np.minimum(best, trial)
        detection.append(_detection_rate(best, threshold))
    return detection


def _top_exploitable(
    bundle: ModelBundle,
    x_attack: np.ndarray,
    centroid: np.ndarray,
    feature_names: list[str],
    ctrl_idx: np.ndarray,
    threshold: float,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Detection drop when the attacker fully mimics one controllable feature alone."""
    base = _detection_rate(attack_scores_transformed(bundle, x_attack), threshold)
    drops: list[tuple[str, float]] = []
    for idx in ctrl_idx:
        adv = np.array(x_attack, dtype=float, copy=True)
        adv[:, idx] = centroid[idx]
        drop = base - _detection_rate(attack_scores_transformed(bundle, adv), threshold)
        drops.append((base_feature_name(feature_names[idx]), drop))
    drops.sort(key=lambda kv: kv[1], reverse=True)
    return drops[:top_k]


def run_evasion_study(
    settings: Settings,
    bundle: ModelBundle,
    attack_df: pd.DataFrame,
    benign_ref_df: pd.DataFrame,
) -> EvasionStudy:
    """Run both evasion attacks and assemble the robustness curves."""
    cfg = settings.robustness
    feature_names = bundle.feature_names()
    ctrl_idx = controllable_indices(feature_names, cfg.controllable_features)

    x_attack = np.asarray(bundle.pipeline.transform(attack_df))
    if len(x_attack) > cfg.max_attack_samples:
        rng0 = np.random.default_rng(settings.seed)
        x_attack = x_attack[rng0.choice(len(x_attack), cfg.max_attack_samples, replace=False)]
    centroid = np.asarray(bundle.pipeline.transform(benign_ref_df)).mean(axis=0)

    threshold = bundle.thresholds.get(cfg.profile)
    if threshold is None:
        raise ValueError(
            f"Unknown threshold profile {cfg.profile!r}; available: {sorted(bundle.thresholds)}"
        )

    baseline = _detection_rate(attack_scores_transformed(bundle, x_attack), threshold)
    mimicry = _mimicry_curve(bundle, x_attack, centroid, ctrl_idx, cfg.mimicry_fractions, threshold)
    search = _search_curve(
        bundle,
        x_attack,
        ctrl_idx,
        cfg.search_budgets,
        cfg.search_iterations,
        threshold,
        np.random.default_rng(settings.seed),
    )
    top = _top_exploitable(bundle, x_attack, centroid, feature_names, ctrl_idx, threshold)

    logger.info(
        "Evasion study complete",
        extra={
            "baseline_detection": round(baseline, 3),
            "mimicry_min": round(min(mimicry), 3),
            "search_min": round(min(search), 3),
            "n_attacks": len(x_attack),
        },
    )
    return EvasionStudy(
        profile=cfg.profile,
        threshold=float(threshold),
        baseline_detection=baseline,
        n_attacks=len(x_attack),
        n_controllable=len(ctrl_idx),
        mimicry_fractions=list(cfg.mimicry_fractions),
        mimicry_detection=mimicry,
        search_budgets=list(cfg.search_budgets),
        search_detection=search,
        top_exploitable=top,
    )
