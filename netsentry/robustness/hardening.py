"""Adversarial hardening: train against mimicry evasion, then re-measure.

The evasion study (``robustness/evasion.py``) shows a mimicry attacker collapses
supervised detection by shaping the attacker-controllable features toward the
benign centroid. That report ends by naming adversarial training as a *direction*;
this module implements it and closes the loop from *measuring* the weakness to
*acting* on it.

The recipe is deliberately simple and honest:

1. Fit the leakage-safe pipeline on the temporal **train** split only.
2. Compute the benign centroid in the transformed feature space (train benign only
   — no leakage), exactly the target the mimicry attacker aims at.
3. Synthesize adversarial attack rows by moving the attack training rows a set of
   fractions toward that centroid on the controllable features — the same move the
   attacker makes — and append them (still labeled *attack*) to the training set.
4. Refit the classifier on the augmented set; calibrate and pick FPR thresholds on
   the **clean** validation split (never on the synthesized rows), so the operating
   point is comparable to the baseline's.
5. Run the *same* evasion study against both the baseline and the hardened model.

Adversarial training typically buys robustness at some cost to clean detection; the
report measures both, so the win (or its absence) is shown, not assumed — the same
"treat a too-good number as a bug" discipline the rest of the project follows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.calibration import fit_calibrator
from netsentry.models.registry import ModelBundle
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.evasion import (
    EvasionStudy,
    controllable_indices,
    mimicry_perturb,
    run_evasion_study,
)
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:

    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "hardening.md"

# (x_train, y_train, feature_names) -> (x_adversarial, y_adversarial). Kept as a
# callback so the fit routine stays identical for the baseline and hardened models
# save for the injected rows — the only variable in the comparison.
Augmentor = Callable[[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray]]


def _profile_name(fpr: float) -> str:
    """e.g. 0.001 -> 'fpr_0.1pct', 0.01 -> 'fpr_1pct' (matches train_supervised)."""
    return f"fpr_{fpr * 100:g}pct"


def adversarial_examples(
    settings: Settings, x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize mimicry-perturbed attack rows for adversarial training.

    Each attack row is moved a set of fractions toward the benign centroid on the
    attacker-controllable features (the mimicry attack), and kept labeled attack.
    Returns an empty pair when there is nothing to augment (no attacks, no benign
    reference, or no controllable features), so the caller degrades to a plain fit.
    """
    ctrl_idx = controllable_indices(feature_names, settings.robustness.controllable_features)
    y = np.asarray(y_train)
    attack_mask = y == 1  # binary target: 1 == attack
    benign_mask = y == 0
    empty = (np.empty((0, x_train.shape[1]), dtype=float), np.empty(0, dtype=y.dtype))
    if not attack_mask.any() or not benign_mask.any() or len(ctrl_idx) == 0:
        return empty

    centroid = np.asarray(x_train)[benign_mask].mean(axis=0)
    x_attack = np.asarray(x_train)[attack_mask]
    perturbed = [
        mimicry_perturb(x_attack, centroid, ctrl_idx, frac)
        for frac in settings.hardening.mimicry_train_fractions
    ]
    x_adv = np.vstack(perturbed) if perturbed else empty[0]
    y_adv = np.ones(len(x_adv), dtype=y.dtype)

    cap = settings.hardening.max_augmented
    if len(x_adv) > cap:
        rng = np.random.default_rng(settings.seed)
        pick = rng.choice(len(x_adv), cap, replace=False)
        x_adv, y_adv = x_adv[pick], y_adv[pick]
    return x_adv, y_adv


def _fit_temporal_binary(
    settings: Settings, augmentor: Augmentor | None = None
) -> tuple[ModelBundle, np.ndarray, np.ndarray]:
    """Fit the honest temporal/binary model, optionally with adversarial augmentation.

    Calibration and the FPR thresholds are always fit on the **clean** validation
    split, so a hardened and a baseline bundle are judged at the same kind of
    operating point. Returns the bundle plus clean test labels/probabilities.
    """
    seed_everything(settings.seed)
    train = load_split(settings, "temporal", "train")
    val = load_split(settings, "temporal", "val")
    test = load_split(settings, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(train))  # FIT ON TRAIN ONLY
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    n_augmented = 0
    if augmentor is not None:
        feature_names = list(pipeline.named_steps["features"].get_feature_names_out())
        x_extra, y_extra = augmentor(x_train, y_train, feature_names)
        n_augmented = len(x_extra)
        if n_augmented:
            x_train = np.vstack([x_train, x_extra])
            y_train = np.concatenate([y_train, y_extra])

    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    proba_val = model.predict_proba(x_val)
    proba_test = model.predict_proba(x_test)

    benign = settings.labels.benign_label
    raw_val = attack_probability(proba_val, model.classes_, benign)
    y_val_bin = y_val.astype(int)
    calibrator = fit_calibrator(settings, raw_val, y_val_bin)
    scores_val = calibrator.transform(raw_val) if calibrator is not None else raw_val
    thresholds = {
        _profile_name(fpr): threshold_at_fpr(y_val_bin, scores_val, fpr)
        for fpr in settings.thresholds.fpr_targets
    }
    bundle = ModelBundle(
        pipeline=pipeline,
        model=model,
        metadata={
            "benign_label": benign,
            "task": "binary",
            "split_strategy": "temporal",
            "n_augmented": int(n_augmented),
        },
        thresholds=thresholds,
        calibrator=calibrator,
    )
    return bundle, y_test, proba_test


def _clean_pr_auc(bundle: ModelBundle, y_test: np.ndarray, proba_test: np.ndarray) -> float:
    """Binary attack-vs-benign PR-AUC on the clean temporal test split."""
    scores = attack_probability(proba_test, bundle.classes, str(bundle.metadata["benign_label"]))
    return float(average_precision_score((np.asarray(y_test) == 1).astype(int), scores))


@dataclass
class HardeningResult:
    """Before/after evasion curves and the clean-detection trade-off of hardening."""

    profile: str
    baseline: EvasionStudy
    hardened: EvasionStudy
    baseline_pr_auc: float
    hardened_pr_auc: float
    n_augmented: int
    train_fractions: list[float]

    @property
    def mimicry_gain(self) -> float:
        """Robustness gain at full mimicry: hardened minus baseline detection."""
        return self.hardened.mimicry_detection[-1] - self.baseline.mimicry_detection[-1]

    @property
    def search_gain(self) -> float:
        """Robustness gain at the largest search budget (hardened minus baseline)."""
        return self.hardened.search_detection[-1] - self.baseline.search_detection[-1]

    @property
    def clean_cost(self) -> float:
        """Clean PR-AUC given up for that robustness (baseline minus hardened)."""
        return self.baseline_pr_auc - self.hardened_pr_auc


def run_hardening(settings: Settings) -> HardeningResult:
    """Train a baseline and an adversarially-hardened model; run evasion on both."""
    test = load_split(settings, "temporal", "test")
    train = load_split(settings, "temporal", "train")
    benign = settings.labels.benign_label
    attack_df = test[test[BINARY_TARGET] == 1]
    benign_ref = train[train[MULTICLASS_TARGET] == benign]

    baseline_bundle, y_test, proba_base = _fit_temporal_binary(settings)
    hardened_bundle, _, proba_hard = _fit_temporal_binary(
        settings, augmentor=lambda x, y, names: adversarial_examples(settings, x, y, names)
    )
    raw_n = hardened_bundle.metadata.get("n_augmented", 0)
    n_augmented = raw_n if isinstance(raw_n, int) else 0

    baseline_study = run_evasion_study(settings, baseline_bundle, attack_df, benign_ref)
    hardened_study = run_evasion_study(settings, hardened_bundle, attack_df, benign_ref)

    result = HardeningResult(
        profile=settings.robustness.profile,
        baseline=baseline_study,
        hardened=hardened_study,
        baseline_pr_auc=_clean_pr_auc(baseline_bundle, y_test, proba_base),
        hardened_pr_auc=_clean_pr_auc(hardened_bundle, y_test, proba_hard),
        n_augmented=n_augmented,
        train_fractions=list(settings.hardening.mimicry_train_fractions),
    )
    logger.info(
        "Adversarial hardening complete",
        extra={
            "n_augmented": n_augmented,
            "mimicry_gain": round(result.mimicry_gain, 3),
            "search_gain": round(result.search_gain, 3),
            "clean_cost_pr_auc": round(result.clean_cost, 3),
        },
    )
    return result


def _curve_row(label: str, ys: list[float]) -> str:
    return f"| {label} | " + " | ".join(f"{y * 100:.1f}%" for y in ys) + " |"


def _comparison_table(
    xs: list[float], baseline: list[float], hardened: list[float], x_label: str
) -> str:
    head = f"| {x_label} | " + " | ".join(f"{x:g}" for x in xs) + " |"
    sep = "|" + "---|" * (len(xs) + 1)
    return "\n".join(
        [
            head,
            sep,
            _curve_row("baseline detection", baseline),
            _curve_row("hardened detection", hardened),
        ]
    )


def _verdict(result: HardeningResult) -> str:
    """One honest sentence on whether the trade landed, framed for a reader."""
    gain, cost = result.mimicry_gain, result.clean_cost
    if gain <= 0.01:
        return (
            "Adversarial training did **not** buy robustness here (full-mimicry detection "
            f"moved {gain * 100:+.1f} pts) — reported as-is; on this stand-in the mimicked "
            "attacks overlap benign traffic too far to separate, which is itself the argument "
            "for the anomaly detector rather than a classifier fix."
        )
    trade = (
        f"at a clean PR-AUC cost of {cost:+.3f}"
        if cost > 0.005
        else "at no measurable clean-detection cost"
    )
    return (
        f"Adversarial training lifts full-mimicry detection by **{gain * 100:+.1f} pts** "
        f"({result.baseline.mimicry_detection[-1] * 100:.1f}% → "
        f"{result.hardened.mimicry_detection[-1] * 100:.1f}%) {trade} — the measured, "
        "honest form of closing the evasion gap."
    )


def _pct_row(label: str, base: float, hard: float, delta: float, unit: str = " pts") -> str:
    """One trade-off row comparing a baseline and hardened percentage."""
    return f"| {label} | {base * 100:.1f}% | {hard * 100:.1f}% | {delta * 100:+.1f}{unit} |"


def _tradeoff_table(result: HardeningResult) -> str:
    b, h = result.baseline, result.hardened
    rows = [
        "| quantity | baseline | hardened | Δ |",
        "|---|---|---|---|",
        _pct_row(
            f"clean detection @ {result.profile} (un-attacked)",
            b.baseline_detection,
            h.baseline_detection,
            h.baseline_detection - b.baseline_detection,
        ),
        f"| clean PR-AUC (temporal test) | {result.baseline_pr_auc:.3f} | "
        f"{result.hardened_pr_auc:.3f} | {-result.clean_cost:+.3f} |",
        _pct_row(
            "detection at full mimicry",
            b.mimicry_detection[-1],
            h.mimicry_detection[-1],
            result.mimicry_gain,
        ),
        _pct_row(
            "detection at largest search budget",
            b.search_detection[-1],
            h.search_detection[-1],
            result.search_gain,
        ),
    ]
    return "\n".join(rows)


def _render(result: HardeningResult, fig: Path) -> str:
    b, h = result.baseline, result.hardened
    mimicry_tbl = _comparison_table(
        b.mimicry_fractions, b.mimicry_detection, h.mimicry_detection, "mimicry fraction"
    )
    search_tbl = _comparison_table(
        b.search_budgets, b.search_detection, h.search_detection, "L2 budget (std units)"
    )
    return f"""# NetSentry — Adversarial Hardening (measure → fix → re-measure)

_Synthetic stand-in; the methodology is the point. Honest **temporal/binary** model
at operating point **{result.profile}**. {result.n_augmented:,} adversarial rows
synthesized at mimicry fractions {result.train_fractions} and added to training._

The [robustness report](robustness.md) measures how a mimicry attacker collapses
detection; it ends by naming adversarial training as a direction. This closes that
loop: the attack flows are perturbed toward the benign centroid — the attacker's own
move — and added to training (still labeled attack), so the classifier learns the
mimicry direction. The model is then re-run through the **same** evasion study.

## Robustness under mimicry (detection vs fraction toward benign)

{mimicry_tbl}

## Robustness under adaptive query search (L2 budget on controllable features)

{search_tbl}

![Baseline vs hardened robustness]({fig.as_posix()})

## The trade-off (this is the honest part)

{_tradeoff_table(result)}

{_verdict(result)}

## Takeaways

- Adversarial training is **not free**: hardening against mimicry can shift the clean
  operating point, and the table above shows both sides so the trade is a decision,
  not a surprise.
- It is also **not a silver bullet**: it hardens against the *specific* perturbation
  it trains on. A defender pairs it with the benign-only anomaly detector (mimicry
  that flattens an attack toward benign is exactly where reconstruction error still
  carries signal) and with input-side constraints — the layered argument the
  [robustness report](robustness.md) makes.
- The point is the arc: NetSentry *measured* the evasion weakness, *acted* on it, and
  *re-measured* — reporting the result whichever way it fell.
"""


def run_hardening_report(settings: Settings) -> Path:
    """Run the hardening study and write the before/after report + figure."""
    result = run_hardening(settings)
    b, h = result.baseline, result.hardened

    fig = plots.plot_lines(
        {
            "baseline": (np.asarray(b.mimicry_fractions), np.asarray(b.mimicry_detection)),
            "hardened": (np.asarray(h.mimicry_fractions), np.asarray(h.mimicry_detection)),
        },
        xlabel="Mimicry fraction toward benign",
        ylabel="Detection rate (TPR)",
        title="Adversarial hardening — mimicry robustness",
        out_path=settings.paths.figures_dir / "hardening_mimicry.png",
    )

    report = _render(result, Path("..") / "figures" / fig.name)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote hardening report", extra={"path": str(out_path)})

    with track_run(settings, "hardening") as run:
        run.log_metrics(
            {
                "n_augmented": float(result.n_augmented),
                "mimicry_gain": result.mimicry_gain,
                "search_gain": result.search_gain,
                "clean_cost_pr_auc": result.clean_cost,
                "baseline_pr_auc": result.baseline_pr_auc,
                "hardened_pr_auc": result.hardened_pr_auc,
            }
        )
        for artifact in (fig, out_path):
            run.log_artifact(artifact)
    return out_path
