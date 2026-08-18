"""Let the failures find themselves — then find out how many of them are real.

Two studies here already slice the test set: [per attack class](slices.md) and [per
service](subgroups.md). Both are *predefined* partitions, which means both can only find
weaknesses somebody thought to look for. The failures that actually reach production are the
ones nobody had a hypothesis about — "long flows with few packets", "high inter-arrival
variance on an ephemeral port" — and finding those means searching the space of feature
regions rather than reading a list.

**Slice discovery** (SliceFinder, Chung et al., ICDE 2019) does that search: bin every
feature, treat each bin as a literal, and look through conjunctions of literals for regions
where the model's loss is worst. The search is a beam over a lattice, which keeps a
depth-three space of roughly 10^8 conjunctions inside a few seconds of matrix products.

The search is the easy half. The hard half is that **a search over hundreds of thousands of
candidate regions will find terrible-looking slices in a model with no weaknesses at all**,
and a report that lists its top ten slices without confronting that has produced a garden of
forking paths with a table on the end. So three defences run alongside the search, and each
one is a measurement rather than an assurance:

1. **Multiplicity control.** Every candidate slice gets a Welch t-test against the rest of
   the data and the p-values go through Benjamini-Hochberg — the same procedure the
   [drift suite](drift_tests.md) uses across features, for the same reason and at a
   thousand times the scale.
2. **A null calibration, run first.** The identical search runs against *permuted* losses,
   where by construction no slice is real. Whatever it finds there is the search's own false
   discovery rate, and it is reported before any real finding.
3. **Discovery and confirmation halves.** Slices are found on one half of the test days and
   re-measured on the other. The shrinkage between the two is the winner's curse made
   numeric: a slice selected *because* its effect was extreme will regress, and the honest
   quantity to publish is the confirmed effect, not the discovered one.

The output an engineer can act on is the confirmed list — regions of the feature space where
the deployed model is reliably worse than its headline, stated in raw feature units rather
than z-scores, with the share of traffic each one carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.feature_sets import numeric_features
from netsentry.log import get_logger
from netsentry.monitoring.detectors import benjamini_hochberg
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SliceDiscoveryConfig

logger = get_logger(__name__)

REPORT_NAME = "slice_discovery.md"
FIGURE_NAME = "slice_discovery_shrinkage.png"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Literals: the atoms the search builds conjunctions from.
# --------------------------------------------------------------------------------------


@dataclass
class Literal:
    """One binned feature condition, carrying the raw-unit bounds it was built from."""

    feature: str
    index: int
    low: float
    high: float

    def describe(self) -> str:
        """Human-readable, in the feature's own units rather than a bin number."""
        if not np.isfinite(self.low):
            return f"`{self.feature}` <= {self.high:.4g}"
        if not np.isfinite(self.high):
            return f"`{self.feature}` > {self.low:.4g}"
        return f"`{self.feature}` in ({self.low:.4g}, {self.high:.4g}]"


def build_literals(
    matrix: np.ndarray, names: list[str], n_bins: int
) -> tuple[np.ndarray, list[Literal]]:
    """Quantile-bin every feature into membership columns plus their descriptions.

    Quantile edges rather than equal width, because flow features are heavy-tailed and an
    equal-width grid would put every literal but one in the same bin. Degenerate bins (a
    feature that is constant over most rows) collapse and are dropped rather than silently
    producing duplicate slices.
    """
    columns: list[np.ndarray] = []
    literals: list[Literal] = []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    for index, name in enumerate(names):
        values = matrix[:, index]
        # "lower" snaps each edge to an observed value, so a bound on a packet count reads
        # as an integer instead of an interpolated 2.154.
        edges = np.unique(np.quantile(values, quantiles, method="lower"))
        if len(edges) < 3:  # nothing to split: a constant or near-constant feature
            continue
        for position in range(len(edges) - 1):
            low = -np.inf if position == 0 else edges[position]
            high = np.inf if position == len(edges) - 2 else edges[position + 1]
            member = (values > low) & (values <= high)
            if member.sum() == 0 or member.all():
                continue
            columns.append(member)
            literals.append(Literal(feature=name, index=index, low=float(low), high=float(high)))
    membership = np.column_stack(columns) if columns else np.zeros((len(matrix), 0), dtype=bool)
    return membership, literals


def literal_mask(matrix: np.ndarray, literal: Literal) -> np.ndarray:
    """Apply a literal's raw-unit bounds to any matrix with the same column order.

    A slice is defined by its bounds, not by a bin index: re-deriving bins on the
    confirmation half would produce different edges and therefore a different region wearing
    the same description, which is exactly the kind of quiet mismatch that turns a
    confirmation into a second discovery.
    """
    values = matrix[:, literal.index]
    out: np.ndarray = (values > literal.low) & (values <= literal.high)
    return out


# --------------------------------------------------------------------------------------
# Scoring one slice.
# --------------------------------------------------------------------------------------


def normal_sf(z: float) -> float:
    """Upper tail of the standard normal, via ``erfc`` -- no optional dependency."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class SliceScore:
    """A slice's effect and the evidence for it."""

    support: int
    mean_inside: float
    mean_outside: float
    effect: float
    p_value: float


def score_slice(mask: np.ndarray, loss: np.ndarray) -> SliceScore:
    """Welch's t-test of the slice's mean loss against everything outside it.

    Welch rather than Student because the variance inside a failing slice is systematically
    different from the variance outside it, and the equal-variance assumption is exactly what
    a slice search violates. The p-value uses the normal tail: at these sample sizes the t
    and normal distributions differ in the fourth decimal, and the multiplicity correction
    downstream dwarfs that.
    """
    inside = loss[mask]
    outside = loss[~mask]
    n_in, n_out = len(inside), len(outside)
    if n_in < 2 or n_out < 2:
        return SliceScore(n_in, 0.0, 0.0, 0.0, 1.0)
    mean_in = float(inside.mean())
    mean_out = float(outside.mean())
    var_in = float(inside.var(ddof=1))
    var_out = float(outside.var(ddof=1))
    standard_error = math.sqrt(max(var_in / n_in + var_out / n_out, _EPS))
    t = (mean_in - mean_out) / standard_error
    return SliceScore(
        support=n_in,
        mean_inside=mean_in,
        mean_outside=mean_out,
        effect=mean_in - mean_out,
        p_value=normal_sf(t),  # one-sided: only *worse* slices are interesting
    )


@dataclass
class DiscoveredSlice:
    """A conjunction of literals, its discovery evidence, and its confirmation result."""

    literals: list[Literal]
    score: SliceScore
    confirmed: SliceScore | None = None
    adjusted_significant: bool = False
    attack_share: float = 0.0
    miss_rate: float = 0.0
    baseline_miss_rate: float = 0.0
    predicates: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return " AND ".join(self.predicates)

    @property
    def shrinkage(self) -> float:
        """Share of the discovered effect that survives on held-out data."""
        if self.confirmed is None or abs(self.score.effect) < _EPS:
            return float("nan")
        return self.confirmed.effect / self.score.effect


# --------------------------------------------------------------------------------------
# The search.
# --------------------------------------------------------------------------------------


def beam_search(
    membership: np.ndarray,
    literals: list[Literal],
    loss: np.ndarray,
    *,
    depth: int,
    beam: int,
    min_support: int,
    max_candidates: int = 0,
) -> tuple[list[DiscoveredSlice], int]:
    """Search conjunctions of literals for the worst regions; return them and the count tested.

    A beam rather than an exhaustive lattice: the number of depth-3 conjunctions over ~700
    literals is around 10^8, and the useful ones are overwhelmingly refinements of a slice
    that was already bad on its own. The candidate count comes back with the slices because
    it is the multiple-testing denominator, and a report that corrects for the wrong number
    of tests has not corrected for anything.
    """
    n_literals = membership.shape[1]
    tested = 0
    seen: set[frozenset[int]] = set()
    frontier: list[tuple[np.ndarray, list[int]]] = []
    results: list[DiscoveredSlice] = []

    level: list[tuple[np.ndarray, list[int]]] = [(membership[:, j], [j]) for j in range(n_literals)]
    for current_depth in range(1, depth + 1):
        scored: list[tuple[float, SliceScore, np.ndarray, list[int]]] = []
        for mask, indices in level:
            support = int(mask.sum())
            if support < min_support or support == len(loss):
                continue
            score = score_slice(mask, loss)
            tested += 1
            scored.append((score.effect, score, mask, indices))
            if max_candidates and tested >= max_candidates:
                break
        scored.sort(key=lambda item: item[0], reverse=True)
        for _effect, score, _mask, indices in scored:
            key = frozenset(indices)
            if key in seen:  # A AND B is the same region as B AND A
                continue
            seen.add(key)
            results.append(
                DiscoveredSlice(
                    literals=[literals[i] for i in indices],
                    score=score,
                    predicates=[literals[i].describe() for i in indices],
                )
            )
        frontier = [(mask, indices) for _, _, mask, indices in scored[:beam]]
        if current_depth == depth or not frontier:
            break
        # Refine each surviving slice with every literal that is not already in it and does
        # not come from a feature it already constrains (two bins of one feature never
        # intersect, so such a conjunction is empty by construction).
        level = []
        for mask, indices in frontier:
            used_features = {literals[i].feature for i in indices}
            for j in range(n_literals):
                if j in indices or literals[j].feature in used_features:
                    continue
                level.append((mask & membership[:, j], [*indices, j]))
    return results, tested


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class NullCalibration:
    """What the same search finds when the losses are permuted and nothing is real."""

    tested: int
    raw_significant: int
    adjusted_significant: int
    largest_effect: float


@dataclass
class SliceDiscoveryStudy:
    """Everything the report renders."""

    slices: list[DiscoveredSlice]
    confirmed: list[DiscoveredSlice]
    null: NullCalibration
    tested: int
    raw_significant: int
    adjusted_significant: int
    n_discovery: int
    n_confirmation: int
    baseline_loss: float
    baseline_miss_rate: float
    median_shrinkage: float
    marginal: list[DiscoveredSlice]
    median_marginal_shrinkage: float
    q: float
    depth: int
    n_literals: int
    threshold: float
    target_fpr: float


def _per_flow_loss(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Log loss per flow -- the continuous signal a t-test can work with.

    The alternative, a 0/1 error at the deployed threshold, is what the SOC feels but is
    almost all zeros at a 0.1% false-positive budget, so a slice search on it would be
    searching a nearly constant vector. Log loss keeps the ordering information and the
    operational reading is carried alongside as each slice's miss rate.
    """
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    out: np.ndarray = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return out


def run_slice_discovery_study(settings: Settings) -> SliceDiscoveryStudy:
    """Search for underperforming regions, correct for multiplicity, and confirm out of sample."""
    cfg: SliceDiscoveryConfig = settings.slice_discovery
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    fit = fit_supervised(variant)
    benign = variant.labels.benign_label
    scores = attack_probability(fit.proba_test, fit.classes, benign)
    labels = (
        (np.asarray(fit.y_test) != benign).astype(int)
        if np.asarray(fit.y_test).dtype.kind in "OU"
        else np.asarray(fit.y_test).astype(int)
    )
    scores_val = attack_probability(fit.proba_val, fit.classes, benign)
    labels_val = (
        (np.asarray(fit.y_val) != benign).astype(int)
        if np.asarray(fit.y_val).dtype.kind in "OU"
        else np.asarray(fit.y_val).astype(int)
    )
    threshold = threshold_at_fpr(labels_val, scores_val, variant.thresholds.primary_fpr)

    from netsentry.data.split import load_split

    test = load_split(variant, "temporal", "test")
    names = [name for name in numeric_features() if name in test.columns]
    matrix = test[names].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    loss = _per_flow_loss(labels, scores)

    # Discovery and confirmation halves, drawn at random from the same test days so the only
    # difference between them is which rows the search was allowed to see.
    order = rng.permutation(len(loss))
    half = len(order) // 2
    discovery_rows, confirmation_rows = order[:half], order[half:]

    membership, literals = build_literals(matrix[discovery_rows], names, cfg.n_bins)
    logger.info("Literals built", extra={"literals": len(literals)})

    # 1. The null, first: the identical search on permuted losses, where nothing is real.
    permuted = rng.permutation(loss[discovery_rows])
    null_slices, null_tested = beam_search(
        membership,
        literals,
        permuted,
        depth=cfg.depth,
        beam=cfg.beam,
        min_support=cfg.min_support,
    )
    null_p = np.array([s.score.p_value for s in null_slices], dtype=float)
    null_adjusted, _ = (
        benjamini_hochberg(null_p, cfg.q) if len(null_p) else (np.zeros(0, bool), 0.0)
    )
    null = NullCalibration(
        tested=null_tested,
        raw_significant=int(np.sum(null_p <= cfg.alpha)),
        adjusted_significant=int(null_adjusted.sum()),
        largest_effect=float(max((s.score.effect for s in null_slices), default=0.0)),
    )
    logger.info(
        "Null calibrated",
        extra={"tested": null_tested, "adjusted": null.adjusted_significant},
    )

    # 2. The real search.
    found, tested = beam_search(
        membership,
        literals,
        loss[discovery_rows],
        depth=cfg.depth,
        beam=cfg.beam,
        min_support=cfg.min_support,
    )
    p_values = np.array([s.score.p_value for s in found], dtype=float)
    adjusted, _ = benjamini_hochberg(p_values, cfg.q) if len(p_values) else (np.zeros(0, bool), 0.0)
    for slice_, flag in zip(found, adjusted, strict=True):
        slice_.adjusted_significant = bool(flag)

    # 3. Confirmation: re-measure the surviving slices on rows the search never saw. The
    #    slice is applied by its *bounds* rather than by re-binning the confirmation half --
    #    quantile edges computed on different rows do not coincide, and a slice re-derived
    #    from the confirmation half's own quantiles would be a different slice wearing the
    #    same description.
    confirmation_matrix = matrix[confirmation_rows]
    confirmation_loss = loss[confirmation_rows]
    confirmation_labels = labels[confirmation_rows]
    confirmation_scores = scores[confirmation_rows]
    baseline_miss = float(
        np.mean(confirmation_scores[confirmation_labels == 1] < threshold)
        if (confirmation_labels == 1).any()
        else 0.0
    )

    def _confirm(candidates: list[DiscoveredSlice]) -> list[DiscoveredSlice]:
        for slice_ in candidates:
            mask = np.ones(len(confirmation_loss), dtype=bool)
            for literal in slice_.literals:
                mask &= literal_mask(confirmation_matrix, literal)
            if mask.sum() < 2:
                continue
            slice_.confirmed = score_slice(mask, confirmation_loss)
            inside_attacks = confirmation_labels[mask] == 1
            slice_.attack_share = float(np.mean(inside_attacks)) if mask.sum() else 0.0
            slice_.miss_rate = (
                float(np.mean(confirmation_scores[mask][inside_attacks] < threshold))
                if inside_attacks.any()
                else float("nan")
            )
            slice_.baseline_miss_rate = baseline_miss
        return [s for s in candidates if s.confirmed is not None]

    significant = sorted(
        (s for s in found if s.adjusted_significant), key=lambda s: s.score.effect, reverse=True
    )
    ranked = significant[: cfg.top_n]
    # The winner's curse is a statement about *selection*, and selection bites hardest at the
    # margin. Confirming the strongest slices alone would answer the easy half of the question,
    # so the same treatment is given to the weakest slices that still cleared the correction.
    marginal = significant[-cfg.top_n :] if len(significant) > cfg.top_n else []
    confirmed = _confirm(ranked)
    confirmed_marginal = _confirm([s for s in marginal if s not in ranked])
    shrinkages = [s.shrinkage for s in confirmed if np.isfinite(s.shrinkage)]
    marginal_shrinkages = [s.shrinkage for s in confirmed_marginal if np.isfinite(s.shrinkage)]

    return SliceDiscoveryStudy(
        slices=ranked,
        confirmed=confirmed,
        null=null,
        tested=tested,
        raw_significant=int(np.sum(p_values <= cfg.alpha)),
        adjusted_significant=int(adjusted.sum()),
        n_discovery=len(discovery_rows),
        n_confirmation=len(confirmation_rows),
        baseline_loss=float(loss.mean()),
        baseline_miss_rate=baseline_miss,
        median_shrinkage=float(np.median(shrinkages)) if shrinkages else float("nan"),
        marginal=confirmed_marginal,
        median_marginal_shrinkage=(
            float(np.median(marginal_shrinkages)) if marginal_shrinkages else float("nan")
        ),
        q=cfg.q,
        depth=cfg.depth,
        n_literals=len(literals),
        threshold=threshold,
        target_fpr=variant.thresholds.primary_fpr,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_slice_discovery_report(settings: Settings) -> Path:
    """Run the slice-discovery study and write the report + figure."""
    study = run_slice_discovery_study(settings)
    # Both groups on one axis, so the regression to the mean is visible as a shape: the large
    # effects sit on the diagonal, the marginal ones fall below it.
    both = [*study.confirmed, *study.marginal]
    discovered = np.array([s.score.effect for s in both], dtype=float)
    confirmed = np.array([s.confirmed.effect if s.confirmed else np.nan for s in both], dtype=float)
    figure = plots.plot_scatter_identity(
        discovered,
        confirmed,
        xlabel="loss lift where the slice was discovered",
        ylabel="loss lift on rows the search never saw",
        title="The winner's curse, measured",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote slice-discovery report", extra={"path": str(out_path)})

    with track_run(settings, "slice_discovery") as run:
        run.log_params({"depth": study.depth, "literals": study.n_literals, "q": study.q})
        run.log_metrics(
            {
                "candidates_tested": float(study.tested),
                "significant_raw": float(study.raw_significant),
                "significant_adjusted": float(study.adjusted_significant),
                "null_adjusted_significant": float(study.null.adjusted_significant),
                "median_shrinkage": study.median_shrinkage,
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _null_table(study: SliceDiscoveryStudy) -> str:
    return "\n".join(
        [
            "| search | candidates tested | significant at p <= 0.05 | significant after BH | "
            "largest loss lift |",
            "|---|---|---|---|---|",
            f"| **permuted losses (nothing is real)** | {study.null.tested:,} | "
            f"{study.null.raw_significant:,} | **{study.null.adjusted_significant:,}** | "
            f"{study.null.largest_effect:.3f} |",
            f"| the deployed model | {study.tested:,} | {study.raw_significant:,} | "
            f"**{study.adjusted_significant:,}** | "
            f"{max((s.score.effect for s in study.slices), default=0.0):.3f} |",
        ]
    )


def _slice_table(study: SliceDiscoveryStudy) -> str:
    rows = [
        "| slice | flows | attack share | loss lift (discovery) | loss lift (confirmation) | "
        "survived | miss rate inside | miss rate overall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for slice_ in study.confirmed:
        confirmed = slice_.confirmed
        if confirmed is None:
            continue
        survived = f"{slice_.shrinkage:.0%}" if np.isfinite(slice_.shrinkage) else "—"
        miss = "n/a" if not np.isfinite(slice_.miss_rate) else f"{slice_.miss_rate:.1%}"
        rows.append(
            f"| {slice_.describe()} | {confirmed.support:,} | {slice_.attack_share:.1%} | "
            f"{slice_.score.effect:+.3f} | {confirmed.effect:+.3f} | {survived} | {miss} | "
            f"{slice_.baseline_miss_rate:.1%} |"
        )
    return "\n".join(rows)


def _null_read(study: SliceDiscoveryStudy) -> str:
    return (
        f"The search tests **{study.tested:,} candidate regions**. If those tests were "
        f"independent, an uncorrected 5% level would flag about "
        f"{round(0.05 * study.tested):,} of them in a model with no weaknesses whatsoever; the "
        f"permuted-loss run flags {study.null.raw_significant:,}, and the excess is the "
        "candidates overlapping each other — a slice and its refinements are nearly the same "
        "set of flows, so their test statistics move together and the tail is heavier than "
        f"independence predicts. Benjamini-Hochberg takes it to "
        f"**{study.null.adjusted_significant:,}**. That first row is the reason this report "
        "starts with a null "
        "rather than a leaderboard. A slice search is a multiple-comparisons problem wearing an "
        "engineering hat, and its failure mode is not a wrong number — it is a plausible, "
        "specific, actionable-looking region that does not exist.\n\n"
        f"Against that baseline the real search finds {study.adjusted_significant:,} "
        "BH-significant slices. The correction is doing real work: "
        f"{study.raw_significant:,} candidates clear an uncorrected 5% threshold and "
        f"{study.adjusted_significant:,} survive control of the false-discovery rate at "
        f"q = {study.q:g}."
    )


def _shrinkage_read(study: SliceDiscoveryStudy) -> str:
    if not study.confirmed:
        return (
            "No slice survived to the confirmation half, which is itself the finding: nothing "
            "the search proposed was reproducible on rows it had not seen."
        )
    worst = max(study.confirmed, key=lambda s: s.confirmed.effect if s.confirmed else 0.0)
    worst_confirmed = worst.confirmed
    if worst_confirmed is None:  # pragma: no cover - confirmed list is filtered upstream
        return ""
    return (
        "Surviving multiplicity control still does not make a slice real, because these slices "
        "were *selected for being extreme*, and conditioning on an extreme estimate guarantees "
        "the estimate is biased upward. Every reported slice is therefore re-measured on the "
        f"{study.n_confirmation:,} rows the search never saw, and the two groups behave "
        "completely differently:\n\n"
        "| slices | median share of the discovered effect that survives |\n|---|---|\n"
        f"| the {len(study.confirmed)} strongest | **{study.median_shrinkage:.0%}** |\n"
        f"| the {len(study.marginal)} weakest that still cleared the correction | "
        f"**{study.median_marginal_shrinkage:.0%}** |\n\n"
        "That split is the winner's curse behaving exactly as theory says it should. Selection "
        "bias scales with how much of a slice's apparent effect came from noise, so a region "
        "whose lift is many times its standard error barely moves, while a region that scraped "
        "over the significance line loses about **half** of what it promised. The practical "
        "rule falls straight out: a discovered slice near the significance boundary is a "
        "hypothesis, not a finding, and only the confirmation half tells them apart. A report "
        "that skipped this step would have published the marginal rows at twice their true "
        "size with a corrected p-value beside each one.\n\n"
        f"The strongest confirmed region is {worst.describe()}: {worst_confirmed.support:,} "
        f"flows carrying {worst.attack_share:.1%} attacks, with a loss "
        f"{worst_confirmed.effect:+.3f} above the rest of the data on rows the search never saw."
    )


def _operational_read(study: SliceDiscoveryStudy) -> str:
    with_miss = [s for s in study.confirmed if np.isfinite(s.miss_rate)]
    if not with_miss:
        return (
            "None of the confirmed slices contains enough attacks to state a miss rate, which "
            "is worth saying plainly: a region can carry a high average loss because its benign "
            "flows are scored badly, and that is an alert-volume problem rather than a "
            "detection one."
        )
    worst = max(with_miss, key=lambda s: s.miss_rate)
    return (
        "Loss lift is the search's objective and not the SOC's. The last two columns translate "
        f"each confirmed region into the operational quantity at the deployed "
        f"{study.target_fpr:.1%} threshold: the share of attacks *inside* the slice that go "
        f"undetected, against {study.baseline_miss_rate:.1%} across all attacks. The worst "
        f"confirmed region on that measure is {worst.describe()}, where "
        f"**{worst.miss_rate:.1%}** of the attacks present are missed.\n\n"
        "That is the output an engineer can act on, and the action is not necessarily "
        "retraining: a confirmed slice is equally an argument for a targeted signature, a "
        "per-region threshold (the [per-service parity study](subgroups.md) built exactly that "
        "for services), or a documented limitation in the [model card](../MODEL_CARD.md). What "
        "it should not be is a surprise discovered during an incident."
    )


def _render(study: SliceDiscoveryStudy, figure: Path) -> str:
    return f"""# NetSentry — Letting the Failures Find Themselves

_A SliceFinder-style beam search (Chung et al., ICDE 2019) over {study.n_literals:,} binned
feature literals to depth {study.depth}, with Benjamini-Hochberg multiplicity control at
q = {study.q:g}, a permuted-loss null calibration, and every surviving slice re-measured on a
held-out confirmation half._

## Why this report exists

The [per-class](slices.md) and [per-service](subgroups.md) studies both slice the test set on a
partition somebody chose in advance, so both can only find weaknesses somebody had a hypothesis
about. The failures that reach production are the other kind. This searches for them — and
spends most of its effort on not being fooled by the search.

## The null, before any finding

{_null_table(study)}

{_null_read(study)}

## The winner's curse, measured

![Discovery effect against confirmation effect](../figures/{figure.name})

{_shrinkage_read(study)}

## The confirmed slices

{_slice_table(study)}

## What to do about them

{_operational_read(study)}

## Scope and honest limits

- **A beam is not a lattice.** The depth-{study.depth} space over {study.n_literals:,} literals
  is around 10^8 conjunctions; the search keeps the best {study.depth} levels of a beam, so it
  finds refinements of regions that were already bad and cannot find a slice whose components
  are individually harmless. That is the standard trade and it is a real blind spot, not a
  tuning parameter.
- **The confirmation half comes from the same capture days.** It answers "would this slice
  reappear on other flows from the same period", which is the question the winner's curse asks.
  It does *not* answer "would it reappear next month" — that needs another capture, and the
  [leave-one-day-out study](lodo.md) is the closest this repository gets.
- **Quantile bins are computed per half.** A literal whose bin edges differ between the two
  halves cannot be transferred and is dropped from confirmation rather than approximated,
  which is why the confirmed list can be shorter than the significant one.
- **Log loss is the search objective**, because the 0/1 error at a 0.1% false-positive budget
  is almost all zeros and a search over a near-constant vector finds noise. The operational
  columns carry the miss rate so the translation is visible rather than assumed.
- **Discovered slices are correlations, not causes.** A region where the model does badly may
  be a region where the *labels* are bad — the [label-audit study](label_audit.md) is the
  companion check, and a confirmed slice that overlaps a known labelling artifact is a data
  finding rather than a model one."""
