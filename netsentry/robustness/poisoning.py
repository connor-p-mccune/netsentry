"""Training-set poisoning — the training-time half of the adversarial threat model.

The evasion study asks what an attacker can do at *inference* time; this asks what
one can do to the *training data*. Two classic attacks, each against the component
it actually threatens:

- **Label flips vs the supervised model.** A fraction of attack training rows is
  relabeled benign — the attacker has corrupted the labeling source (a sandbox, a
  blocklist, an analyst queue), so their traffic both vanishes from the attack
  class *and* pollutes the benign class. Validation is carved from train, so the
  operator's threshold selection is corrupted too; only the test split keeps clean
  ground truth.
- **Benign-pool contamination vs the anomaly detector.** Attack flows are injected
  into the "benign-only" pool the detector fits and calibrates on — the standing
  weakness of benign-baseline training: the baseline is only as clean as its
  labels. Contamination widens the learned notion of normal *and* inflates the
  calibrated threshold (contaminated validation scores push the quantile up), so
  detection is hit twice.

Both studies report degradation curves on the clean temporal test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.anomaly import build_anomaly_detector
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "poisoning.md"


@dataclass
class FlipPoint:
    """Supervised degradation at one label-flip rate."""

    rate: float
    n_flipped: int
    pr_auc: float
    detection: float  # TPR at the operating FPR, threshold chosen on (poisoned) val


@dataclass
class ContaminationPoint:
    """Anomaly-detector degradation at one benign-pool contamination rate."""

    rate: float
    n_injected: int
    detection: float  # share of clean-test attacks over the calibrated threshold
    realized_fpr: float  # share of clean-test benign flows over it


def flip_attack_labels(
    df: pd.DataFrame, rate: float, benign_label: str, seed: int
) -> tuple[pd.DataFrame, int]:
    """Relabel ``rate`` of the attack rows as benign (both targets); seeded, copy-safe."""
    if rate <= 0:
        return df, 0
    attack_idx = df.index[df[BINARY_TARGET] == 1]
    n_flip = int(len(attack_idx) * rate)
    if n_flip == 0:
        return df, 0
    rng = np.random.default_rng(seed)
    chosen = rng.choice(attack_idx.to_numpy(), size=n_flip, replace=False)
    poisoned = df.copy()
    poisoned.loc[chosen, BINARY_TARGET] = 0
    poisoned.loc[chosen, MULTICLASS_TARGET] = benign_label
    return poisoned, n_flip


def contaminate_benign_pool(
    benign: pd.DataFrame, attacks: pd.DataFrame, rate: float, seed: int
) -> tuple[pd.DataFrame, int]:
    """Inject ``rate * len(benign)`` attack rows into the pool (labels untouched)."""
    n_inject = int(len(benign) * rate)
    if n_inject == 0 or attacks.empty:
        return benign, 0
    sampled = attacks.sample(n_inject, replace=n_inject > len(attacks), random_state=seed)
    return pd.concat([benign, sampled], ignore_index=True), n_inject


def _supervised_curve(settings: Settings) -> list[FlipPoint]:
    """Refit the temporal binary model per flip rate; score against clean test."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_test = test[BINARY_TARGET].to_numpy()
    operating_fpr = variant.thresholds.fpr_targets[-1]
    benign = variant.labels.benign_label

    points: list[FlipPoint] = []
    for rate in variant.poisoning.label_flip_rates:
        seed_everything(variant.seed)
        p_train, n_train = flip_attack_labels(train, rate, benign, variant.seed)
        p_val, n_val = flip_attack_labels(val, rate, benign, variant.seed)

        pipeline = build_pipeline(variant)
        x_train = pipeline.fit_transform(p_train)
        x_val, x_test = pipeline.transform(p_val), pipeline.transform(test)
        y_train_p = p_train[BINARY_TARGET].to_numpy()
        y_val_p = p_val[BINARY_TARGET].to_numpy()
        model = SupervisedClassifier(variant).fit(x_train, y_train_p, eval_set=(x_val, y_val_p))

        s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
        s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
        # The operator only has poisoned labels, so the threshold is chosen on them;
        # detection is then judged against the clean test ground truth.
        threshold = threshold_at_fpr(y_val_p, s_val, operating_fpr)
        detection = rates_at_threshold(y_test, s_test, threshold)["tpr"]
        pr_auc = float(average_precision_score(y_test, s_test))
        points.append(FlipPoint(rate, n_train + n_val, pr_auc, detection))
        logger.info(
            "Label-flip point",
            extra={"rate": rate, "flipped": n_train + n_val, "pr_auc": round(pr_auc, 4)},
        )
    return points


def _anomaly_curve(settings: Settings) -> list[ContaminationPoint]:
    """Refit the benign-only detector per contamination rate; score clean test."""
    train = load_split(settings, "temporal", "train")
    val = load_split(settings, "temporal", "val")
    test = load_split(settings, "temporal", "test")

    # The feature pipeline is unsupervised (impute/scale), so fit it once on the
    # train split — contamination changes what the detector sees, not the scaler.
    pipeline = build_pipeline(settings)
    pipeline.fit(train)

    is_attack_train = train[BINARY_TARGET] == 1
    is_attack_val = val[BINARY_TARGET] == 1
    x_test = pipeline.transform(test)
    y_test = test[BINARY_TARGET].to_numpy()
    attack_mask = y_test == 1

    points: list[ContaminationPoint] = []
    for rate in settings.poisoning.contamination_rates:
        seed_everything(settings.seed)
        pool, n_train = contaminate_benign_pool(
            train[~is_attack_train], train[is_attack_train], rate, settings.seed
        )
        cal_pool, n_val = contaminate_benign_pool(
            val[~is_attack_val], val[is_attack_val], rate, settings.seed
        )
        detector = build_anomaly_detector(settings, "iforest")
        detector.fit(pipeline.transform(pool))
        detector.calibrate_threshold(pipeline.transform(cal_pool), settings.anomaly.target_fpr)

        flagged = detector.score(x_test) >= detector.threshold
        detection = float(flagged[attack_mask].mean()) if attack_mask.any() else 0.0
        realized_fpr = float(flagged[~attack_mask].mean()) if (~attack_mask).any() else 0.0
        points.append(ContaminationPoint(rate, n_train + n_val, detection, realized_fpr))
        logger.info(
            "Contamination point",
            extra={"rate": rate, "injected": n_train + n_val, "detection": round(detection, 4)},
        )
    return points


def run_poisoning_report(settings: Settings) -> Path:
    """Run both poisoning studies, plot the degradation curves, write the report."""
    flips = _supervised_curve(settings)
    contaminations = _anomaly_curve(settings)

    fig_sup = plots.plot_lines(
        {
            "PR-AUC (clean test)": (
                np.array([p.rate for p in flips]),
                np.array([p.pr_auc for p in flips]),
            ),
            "detection @ operating FPR": (
                np.array([p.rate for p in flips]),
                np.array([p.detection for p in flips]),
            ),
        },
        xlabel="Fraction of attack training rows relabeled benign",
        ylabel="Score on clean test",
        title="Label-flip poisoning vs the supervised model",
        out_path=settings.paths.figures_dir / "poisoning_supervised.png",
    )
    fig_ano = plots.plot_lines(
        {
            "detection of test attacks": (
                np.array([p.rate for p in contaminations]),
                np.array([p.detection for p in contaminations]),
            ),
            "realized benign FPR": (
                np.array([p.rate for p in contaminations]),
                np.array([p.realized_fpr for p in contaminations]),
            ),
        },
        xlabel="Attack rows injected into the benign pool (fraction of pool)",
        ylabel="Rate on clean test",
        title="Benign-pool contamination vs the anomaly detector",
        out_path=settings.paths.figures_dir / "poisoning_anomaly.png",
    )

    report = _render(flips, contaminations, settings, fig_sup, fig_ano)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote poisoning report", extra={"path": str(out_path)})

    with track_run(settings, "poisoning") as run:
        run.log_metrics(
            {
                "flip_pr_auc_clean": flips[0].pr_auc,
                "flip_pr_auc_worst": flips[-1].pr_auc,
                "contamination_detection_clean": contaminations[0].detection,
                "contamination_detection_worst": contaminations[-1].detection,
            }
        )
        run.log_artifact(fig_sup)
        run.log_artifact(fig_ano)
        run.log_artifact(out_path)
    return out_path


def _flip_table(points: list[FlipPoint]) -> str:
    rows = [
        "| flip rate | rows poisoned | PR-AUC (clean test) | detection @ operating FPR |",
        "|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.rate * 100:g}% | {p.n_flipped:,} | {p.pr_auc:.3f} | {p.detection * 100:.1f}% |"
        )
    return "\n".join(rows)


def _contamination_table(points: list[ContaminationPoint]) -> str:
    rows = [
        "| contamination | rows injected | detection of test attacks | realized benign FPR |",
        "|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.rate * 100:g}% | {p.n_injected:,} | {p.detection * 100:.1f}% "
            f"| {p.realized_fpr * 100:.2f}% |"
        )
    return "\n".join(rows)


def _read(flips: list[FlipPoint], contaminations: list[ContaminationPoint]) -> str:
    """Sign-aware summary of both curves, so prose and numbers cannot diverge."""
    pr_drop = flips[0].pr_auc - flips[-1].pr_auc
    det_drop = flips[0].detection - flips[-1].detection
    ano_drop = contaminations[0].detection - contaminations[-1].detection
    rate_pct = flips[-1].rate * 100
    # The sharp story is the operating-point collapse, not the (robust) ranking:
    # PR-AUC is computed on the raw score and barely moves, but the operator's
    # threshold is chosen on the *poisoned* validation labels, so detection craters.
    if det_drop > 0.02 >= pr_drop:
        sup = (
            f"The striking result is the split between ranking and operating point. "
            f"PR-AUC — a *ranking* metric on the raw score — is robust (only {pr_drop:+.3f} at "
            f"a {rate_pct:g}% flip rate), because boosting still orders attacks above benign. "
            f"But detection at the operator's threshold collapses from "
            f"{flips[0].detection * 100:.1f}% to {flips[-1].detection * 100:.1f}%: the "
            "threshold is chosen on the *poisoned* validation labels, so the flips move the "
            "operating point even where they leave the score ordering intact. A study that "
            "reported only PR-AUC would have called this model poison-resistant and been wrong "
            "about the thing that ships."
        )
    elif pr_drop >= 0.02:
        sup = (
            f"Label flips cost the supervised model **{pr_drop:.3f} PR-AUC** and drop detection "
            f"{flips[0].detection * 100:.1f}% → {flips[-1].detection * 100:.1f}% by a "
            f"{rate_pct:g}% flip rate — the attack signal is diluted and the benign class is "
            "polluted at once, and the poisoned-validation threshold drifts on top."
        )
    else:
        sup = (
            f"The supervised model is tolerant of label flips on this stand-in (PR-AUC "
            f"{pr_drop:+.3f}, detection {det_drop * 100:+.1f} points at {rate_pct:g}%). Do not "
            "over-read it: flipped flows also pollute the benign class, and real data with "
            "weaker class margins will not be this forgiving."
        )
    det_drop = ano_drop
    if det_drop <= 0.005:
        ano = (
            "The Isolation Forest's detection barely moves under contamination on this "
            "stand-in — its notion of normal is coarse enough that a few injected attack "
            "flows do not visibly widen it."
        )
    else:
        ano = (
            f"Benign-pool contamination degrades the anomaly detector's detection by "
            f"**{det_drop * 100:.1f} points** at a {contaminations[-1].rate * 100:g}% "
            "injection rate. The mechanism is double: injected attacks widen the learned "
            "normal, and the calibration quantile (computed on the contaminated 'benign' "
            "validation pool) inflates the threshold on top."
        )
    return f"{sup}\n\n{ano}"


def _render(
    flips: list[FlipPoint],
    contaminations: list[ContaminationPoint],
    settings: Settings,
    fig_sup: Path,
    fig_ano: Path,
) -> str:
    operating_fpr = settings.thresholds.fpr_targets[-1]
    return f"""# NetSentry — Training-Set Poisoning Study

_Synthetic stand-in. Temporal split; degradation is always measured on the **clean**
test ground truth, while training and validation (and therefore threshold
selection) see the poisoned labels — the operator's actual position._

Evasion (see the robustness report) is the inference-time adversary; poisoning is
the training-time one. A NIDS pipeline ingests labels from sources an attacker can
influence — sandbox verdicts, blocklists, "known-clean" capture windows — so "how
fast does detection decay as the labels rot" is a measurable property, not a
hypothetical.

## Label flips vs the supervised model

A fraction of attack training rows is relabeled benign (attacker hides their class
in the labeling source). Detection threshold: chosen on the poisoned validation set
at the {operating_fpr * 100:g}% FPR budget, as the operator would.

{_flip_table(flips)}

![Label-flip poisoning](../figures/{fig_sup.name})

## Benign-pool contamination vs the anomaly detector

Attack rows are injected into the "benign-only" pool the Isolation Forest fits on
and calibrates against (target benign FPR {settings.anomaly.target_fpr * 100:g}%).
This is the standing weakness of benign-baseline training: the baseline is only as
clean as its labels.

{_contamination_table(contaminations)}

![Benign-pool contamination](../figures/{fig_ano.name})

## Read

{_read(flips, contaminations)}

Defences follow from the mechanism, and two are already in this pipeline: the
**data-quality gates** (`netsentry validate`) catch gross label anomalies before
training, and **drift monitoring** compares production scores against a reference —
a poisoned retrain shifts the score distribution, which is exactly what the PSI
gauge watches. The honest caveat: a patient adversary who poisons *slowly* stays
under both, which is why provenance on training data matters as much as provenance
on models.
"""
