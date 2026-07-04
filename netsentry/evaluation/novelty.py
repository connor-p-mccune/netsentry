"""Novelty distance — detection as a function of distance to the training attacks.

The project's central finding is the temporal-vs-stratified PR-AUC gap. This study
exposes the *mechanism*. For every test attack, compute the Euclidean distance (in
the pipeline's standardized feature space — the model's own geometry) to its nearest
**training** attack: a direct measure of how novel that flow is to the model. Then
bin detection rate by that distance, for both split strategies on shared bin edges.

Two distinct explanations of the gap become separable:

- **Composition:** the shuffled split's test attacks sit close to training near-twins
  (same-burst flows land on both sides), so it is scored mostly on easy, familiar
  flows; the temporal split's later-day attacks are genuinely far. If per-bin
  detection roughly coincides across splits, the gap is a *mixture* effect over one
  decay curve.
- **At-distance shift:** if the temporal split underperforms even at matched
  distance, the later days also change the *context* (benign background, feature
  scales), not just the attack mix.

The report decomposes the headline gap into those two parts by reweighting the
stratified per-bin detection to the temporal distance mix. The near-twin fraction
(distance below ``novelty.twin_epsilon``) is reported per split: on the real
CIC-IDS2017 those near-duplicates are exactly the shuffled split's leakage (exact
duplicates are already dropped in cleaning; bursts still produce near-twins).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "novelty.md"


def nn_distances(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query row to its nearest reference row."""
    index = NearestNeighbors(n_neighbors=1).fit(np.asarray(reference))
    distances, _ = index.kneighbors(np.asarray(queries))
    return np.asarray(distances[:, 0])


def quantile_edges(distances: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges over the pooled distances (deduplicated, so bins are valid)."""
    edges = np.unique(np.quantile(np.asarray(distances), np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:  # all distances identical — one catch-all bin
        edges = np.array([edges[0], edges[0] + 1.0])
    return edges


@dataclass
class NoveltyBin:
    """Detection within one shared distance bin, for one split strategy."""

    low: float
    high: float
    n: int
    detection: float  # NaN when the split has no attacks in this bin


@dataclass
class SplitNovelty:
    """One split strategy's novelty profile at the shared operating FPR."""

    strategy: str
    n_attacks: int
    median_distance: float
    twin_fraction: float  # share of test attacks with a near-twin in train
    detection: float  # overall detection at the operating threshold
    bins: list[NoveltyBin]


def bin_detection(
    distances: np.ndarray, detected: np.ndarray, edges: np.ndarray
) -> list[NoveltyBin]:
    """Per-bin detection rate; the last bin is closed on the right."""
    distances = np.asarray(distances)
    detected = np.asarray(detected).astype(bool)
    bins: list[NoveltyBin] = []
    for i in range(len(edges) - 1):
        low, high = float(edges[i]), float(edges[i + 1])
        mask = (distances >= low) & ((distances < high) | (i == len(edges) - 2))
        n = int(mask.sum())
        rate = float(detected[mask].mean()) if n else float("nan")
        bins.append(NoveltyBin(low, high, n, rate))
    return bins


def composition_counterfactual(source: list[NoveltyBin], target: list[NoveltyBin]) -> float:
    """Detection the *source* split's per-bin rates would achieve on the *target* mix.

    Reweights source per-bin detection by the target's bin occupancy, over the bins
    where both are defined (weights renormalized). The counterfactual splits the
    headline gap into a composition part (mix of novelty) and an at-distance part.
    """
    weights, rates = [], []
    for s_bin, t_bin in zip(source, target, strict=True):
        if t_bin.n > 0 and np.isfinite(s_bin.detection):
            weights.append(float(t_bin.n))
            rates.append(s_bin.detection)
    if not weights:
        return float("nan")
    w = np.asarray(weights) / float(np.sum(weights))
    return float(np.sum(w * np.asarray(rates)))


def _split_profile(
    settings: Settings, strategy: str, operating_fpr: float
) -> tuple[np.ndarray, np.ndarray]:
    """(NN distance to training attacks, detected-at-threshold) per test attack."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = strategy  # type: ignore[assignment]
    variant.supervised.task = "binary"
    benign = variant.labels.benign_label

    train = load_split(variant, strategy, "train")
    val = load_split(variant, strategy, "val")
    test = load_split(variant, strategy, "test")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)

    seed_everything(variant.seed)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)

    rng = np.random.default_rng(variant.seed)
    reference = x_train[y_train == 1]
    if len(reference) > variant.novelty.max_reference:
        reference = reference[
            rng.choice(len(reference), variant.novelty.max_reference, replace=False)
        ]
    attack_idx = np.where(y_test == 1)[0]
    if len(attack_idx) > variant.novelty.max_queries:
        attack_idx = rng.choice(attack_idx, variant.novelty.max_queries, replace=False)

    distances = nn_distances(reference, x_test[attack_idx])
    detected = s_test[attack_idx] >= threshold
    logger.info(
        "Novelty profile",
        extra={
            "strategy": strategy,
            "attacks": len(attack_idx),
            "median_distance": round(float(np.median(distances)), 3),
            "detection": round(float(detected.mean()), 4),
        },
    )
    return distances, detected


def run_novelty_report(settings: Settings) -> Path:
    """Profile detection vs novelty distance on both splits; write the report."""
    operating_fpr = settings.thresholds.fpr_targets[-1]
    epsilon = settings.novelty.twin_epsilon

    raw = {s: _split_profile(settings, s, operating_fpr) for s in ("stratified", "temporal")}
    edges = quantile_edges(np.concatenate([d for d, _ in raw.values()]), settings.novelty.n_bins)

    profiles: list[SplitNovelty] = []
    for strategy, (distances, detected) in raw.items():
        profiles.append(
            SplitNovelty(
                strategy=strategy,
                n_attacks=len(distances),
                median_distance=float(np.median(distances)),
                twin_fraction=float(np.mean(distances < epsilon)),
                detection=float(detected.mean()),
                bins=bin_detection(distances, detected, edges),
            )
        )
    stratified, temporal = profiles[0], profiles[1]
    counterfactual = composition_counterfactual(stratified.bins, temporal.bins)

    fig = _plot(profiles, settings.paths.figures_dir / "novelty.png")
    report = _render(stratified, temporal, counterfactual, operating_fpr, epsilon, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote novelty report", extra={"path": str(out_path)})

    with track_run(settings, "novelty") as run:
        run.log_metrics({f"{p.strategy}_median_distance": p.median_distance for p in profiles})
        run.log_metrics({f"{p.strategy}_twin_fraction": p.twin_fraction for p in profiles})
        run.log_metrics({f"{p.strategy}_detection": p.detection for p in profiles})
        if np.isfinite(counterfactual):
            run.log_metrics({"counterfactual_detection": counterfactual})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(profiles: list[SplitNovelty], out_path: Path) -> Path:
    """Detection and attack share per distance bin, one pair of lines per split.

    The x-axis is the bin *index*: quantile bins have a heavy-tailed last range whose
    midpoint would squash every other bin against the left edge. The report table
    carries the actual distance ranges.
    """
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in profiles:
        idx = np.arange(1, len(p.bins) + 1, dtype=float)
        total = sum(b.n for b in p.bins) or 1
        series[f"{p.strategy} detection"] = (
            idx,
            np.array([b.detection for b in p.bins]),
        )
        series[f"{p.strategy} attack share"] = (
            idx,
            np.array([b.n / total for b in p.bins]),
        )
    return plots.plot_lines(
        series,
        xlabel="Distance bin (near → far, shared quantiles of NN distance)",
        ylabel="Rate / share",
        title="Detection vs novelty distance",
        out_path=out_path,
    )


def _slope(bins: list[NoveltyBin]) -> float:
    """Last-minus-first populated-bin detection — the direction of the distance trend."""
    populated = [b for b in bins if b.n > 0 and np.isfinite(b.detection)]
    if len(populated) < 2:
        return 0.0
    return populated[-1].detection - populated[0].detection


def _gradient_read(stratified: SplitNovelty, temporal: SplitNovelty) -> str:
    """Honest reading of the detection-vs-distance direction (it need not decay)."""
    s_slope, t_slope = _slope(stratified.bins), _slope(temporal.bins)
    margin = 0.05  # five points before calling a direction
    if s_slope < -margin and t_slope < -margin:
        return (
            "Detection **decays with distance** in both splits — the classic novelty story: "
            "the farther a test attack sits from anything trained on, the likelier it slips "
            "through, which is the supervised model's structural blind spot and the anomaly "
            "detector's remit."
        )
    if s_slope > margin and t_slope > margin:
        return (
            f"Detection **rises with distance** in both splits ({s_slope * 100:+.0f} pts "
            f"stratified, {t_slope * 100:+.0f} pts temporal, first to last bin) — the honest "
            "and initially surprising reading. L2 novelty is not this model's difficulty axis: "
            "the far-from-training attacks are the volumetric extremes (DDoS-style rate "
            "blow-ups) that are easy to flag *because* they are extreme, while the hard attacks "
            "are the near ones sitting close to the benign manifold. That is the same geometry "
            "the evasion study exploits — mimicry drags attack features *toward* benign and "
            "detection collapses — and it is why the dangerous end of the curve is the near "
            "end, not the far one."
        )
    return (
        f"The detection-distance trend is mixed ({s_slope * 100:+.0f} pts stratified, "
        f"{t_slope * 100:+.0f} pts temporal, first to last bin): distance to the nearest "
        "training attack is at best a weak difficulty axis for this model on this data — worth "
        "knowing before anyone treats novelty scores as triage priorities."
    )


def _render(
    stratified: SplitNovelty,
    temporal: SplitNovelty,
    counterfactual: float,
    operating_fpr: float,
    epsilon: float,
    fig: Path,
) -> str:
    header = [
        "| split | test attacks | median NN distance | near-twins (< "
        f"{epsilon:g}) | detection |",
        "|---|---|---|---|---|",
    ]
    for p in (stratified, temporal):
        header.append(
            f"| {p.strategy} | {p.n_attacks:,} | {p.median_distance:.2f} "
            f"| {p.twin_fraction * 100:.1f}% | {p.detection * 100:.1f}% |"
        )

    bin_rows = [
        "| distance bin | stratified share | stratified detection "
        "| temporal share | temporal detection |",
        "|---|---|---|---|---|",
    ]
    s_total = sum(b.n for b in stratified.bins) or 1
    t_total = sum(b.n for b in temporal.bins) or 1
    for s_bin, t_bin in zip(stratified.bins, temporal.bins, strict=True):
        s_det = f"{s_bin.detection * 100:.1f}%" if np.isfinite(s_bin.detection) else "—"
        t_det = f"{t_bin.detection * 100:.1f}%" if np.isfinite(t_bin.detection) else "—"
        bin_rows.append(
            f"| [{s_bin.low:.2f}, {s_bin.high:.2f}) | {s_bin.n / s_total * 100:.0f}% | {s_det} "
            f"| {t_bin.n / t_total * 100:.0f}% | {t_det} |"
        )

    parts: list[str] = []
    # Only call the mixes different when the median gap is material (10% relative);
    # a fraction of a standardized unit across ~77 dims is not a composition story.
    median_gap = temporal.median_distance - stratified.median_distance
    if median_gap > 0.1 * stratified.median_distance:
        parts.append(
            f"The distance profiles differ the way the leakage argument predicts: the median "
            f"stratified test attack sits {stratified.median_distance:.2f} standardized units "
            f"from its nearest training attack, versus {temporal.median_distance:.2f} on the "
            "temporal split. A shuffled split is scored mostly on flows the model has "
            "near-neighbours for; the temporal split asks about genuinely new territory."
        )
    elif median_gap < -0.1 * stratified.median_distance:
        parts.append(
            f"Unexpectedly, the *temporal* test attacks sit closer to training (median "
            f"{temporal.median_distance:.2f} vs stratified {stratified.median_distance:.2f}), so "
            "nearness cannot explain any stratified advantage here — read the decomposition "
            "below with that in mind."
        )
    else:
        parts.append(
            f"The two distance profiles are nearly identical (median "
            f"{stratified.median_distance:.2f} vs {temporal.median_distance:.2f}): the temporal "
            "test attacks are *not* systematically farther from training on this stand-in, so "
            "nearness/composition cannot carry the headline gap here. That is a property of the "
            "generator, not of the method — see the twin note below."
        )
    if max(stratified.twin_fraction, temporal.twin_fraction) < 0.01:
        parts.append(
            f"Near-twins (distance < {epsilon:g}) are essentially absent "
            f"({stratified.twin_fraction * 100:.1f}% / {temporal.twin_fraction * 100:.1f}%): the "
            "synthetic generator draws flows independently, so it does not reproduce the real "
            "dataset's burst near-duplicates. On the real CIC-IDS2017 this bar is exactly where "
            "shuffled-split leakage lives — same-burst, near-identical flows landing on both "
            "sides of a random split (exact duplicates are already dropped in cleaning), pulling "
            "the stratified distances toward zero and detection on them toward memory. The "
            "instrument is built to expose that; the stand-in simply has none to expose."
        )
    else:
        twin_side = (
            "stratified" if stratified.twin_fraction > temporal.twin_fraction else "temporal"
        )
        parts.append(
            f"Near-twins (distance < {epsilon:g}) make up "
            f"{stratified.twin_fraction * 100:.1f}% of stratified and "
            f"{temporal.twin_fraction * 100:.1f}% of temporal test attacks — concentrated in the "
            f"{twin_side} split. Every near-twin is a question the training set has already "
            "answered; scoring on them measures memory, not detection."
        )
    parts.append(_gradient_read(stratified, temporal))
    if np.isfinite(counterfactual):
        gap = stratified.detection - temporal.detection
        composition = stratified.detection - counterfactual
        at_distance = counterfactual - temporal.detection
        if abs(composition) < 0.02:
            comp_read = (
                f"essentially **none** of it is composition ({composition * 100:+.1f} pts — the "
                f"two mixes nearly coincide) and {at_distance * 100:+.1f} pts is **at-distance "
                "shift**: at every matched novelty level the temporal split detects far less, "
                "because the later days change the attack *classes* and the benign context (what "
                "the drift report measures), not merely the distances"
            )
        elif composition > 0:
            comp_read = (
                f"roughly {composition * 100:+.1f} pts is **composition** (the shuffled split "
                f"contains nearer, easier attacks) and {at_distance * 100:+.1f} pts is "
                "**at-distance shift** (the later days are harder even at matched novelty)"
            )
        else:
            comp_read = (
                f"composition actually *favours* the temporal mix ({composition * 100:+.1f} "
                f"pts), so the at-distance shift ({at_distance * 100:+.1f} pts) is the entire "
                "story and more"
            )
        parts.append(
            f"**Decomposing the gap.** At the shared {operating_fpr * 100:g}%-FPR operating "
            f"point the stratified split detects {stratified.detection * 100:.1f}% and the "
            f"temporal split {temporal.detection * 100:.1f}% (gap {gap * 100:+.1f} pts). "
            f"Applying the stratified per-bin detection rates to the *temporal* distance mix "
            f"predicts {counterfactual * 100:.1f}%: {comp_read}. The decomposition is "
            "approximate (shared quantile bins, renormalized over bins both splits populate), "
            "but it makes the two flavours of shuffled-split optimism separately measurable — "
            "and on the real data, where twins exist, the composition share is the leakage."
        )
    return f"""# NetSentry — Novelty Distance (why shuffled splits flatter)

_Synthetic stand-in. For every test attack, the Euclidean distance to its **nearest
training attack** in the pipeline's standardized feature space — the model's own
geometry — profiled for both split strategies on shared quantile bins, with detection
at the {operating_fpr * 100:g}%-FPR operating point (threshold chosen on each split's
validation set)._

## The question

The headline result is the temporal-vs-stratified gap. This study asks *why*: is the
shuffled split flattered because its test attacks sit next to training near-twins
(a **composition** effect over one decay curve), or does the temporal split also
underperform **at matched distance** (the later days shift the context, not just the
mix)? Nearest-neighbour distance to the training attacks makes the question
measurable.

{chr(10).join(header)}

## Detection by distance (shared bins)

{chr(10).join(bin_rows)}

![Novelty distance](../figures/{fig.name})

## Read

{chr(10).join(f"{p}{chr(10)}" for p in parts)}
The value of the instrument is that it turns "shuffled splits leak" from a slogan
into two measurable quantities — how much of the gap is *composition* (near-twins and
nearness, the leakage proper) and how much is *at-distance shift* (the world actually
changing). The per-class slices name the missed attacks, the drift report shows the
days moving, and this report says which mechanism the headline gap is made of on the
data at hand."""
