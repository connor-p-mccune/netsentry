"""Deterministic robustness verification: proving a verdict, instead of sampling for it.

Three reports in this project ask how far an attacker has to push a flow before the
detector lets it through, and each answers differently. The [evasion study](robustness.md)
runs an attack and reports what it found — an **upper** bound on the true radius, because a
better attack may exist tomorrow. The [certification study](certify.md) wraps the detector
in Gaussian noise and gives a probabilistic **lower** bound from Cohen et al. (2019), which
is a real guarantee with two asterisks: it holds with a confidence level rather than
absolutely, and it describes the *smoothed* classifier, which is not the one deployed.

There is a third option that neither uses, and gradient-boosted trees are one of the few
model families where it is available. A tree ensemble is a piecewise-constant function over
axis-aligned boxes, so its output over an input box can be bounded by **arithmetic** rather
than by sampling. Push an interval down each tree: at a split, if the box lies entirely on
one side, follow that child; if it straddles the threshold, follow both and keep the extremes.
Summing per-tree minima gives a sound lower bound on the ensemble margin over the whole box.
If that lower bound still clears the decision threshold, then **no** perturbation inside the
box — none, not merely none anyone has found — can flip the verdict. No sampling, no
confidence level, no surrogate model. It is a proof about the deployed detector.

The catch is stated up front, because a verification report that oversells is worse than no
report. Bounding each tree independently ignores that all trees read the *same* input, so a
combination of leaves that the bound treats as reachable together may be jointly impossible.
The bound is therefore **sound but incomplete**: it never certifies a point that is actually
attackable, but it will sometimes fail to certify a point that is actually safe. The exact
answer requires searching consistent leaf tuples — a max-clique problem (Chen et al., *Robustness
Verification of Tree-Based Models*, NeurIPS 2019) — and the price of not paying for it is
measured here by sandwiching every flow between the certified radius and what an attack
actually achieves.

The second half of the report is where verification stops being an exercise. A guarantee
against *arbitrary* perturbation is the wrong guarantee for a network detector: an attacker
can pad a packet, add a delay or insert dummy traffic, but cannot un-send bytes already on
the wire, and cannot touch the protocol-structural fields at all. Restricting the box to the
directions an attacker can actually move turns a pessimistic number into an operational one,
and the difference between the two is a statement about how much of the detector's apparent
fragility is an artefact of an unrealistic threat model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.evasion import controllable_indices
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import VerifyTreesConfig

logger = get_logger(__name__)

REPORT_NAME = "verify_trees.md"
SANDWICH_FIGURE = "verify_sandwich.png"
THREAT_FIGURE = "verify_threat_model.png"

#: How this report's guarantee differs from the two robustness studies already here.
_COMPARISON_TABLE = "\n".join(
    [
        "| study | direction | strength | about which model |",
        "|---|---|---|---|",
        '| [evasion](robustness.md) | upper bound | "this attack works" | the deployed one |',
        "| [certified robustness](certify.md) | lower bound "
        "| probabilistic, with a confidence level | a *smoothed* surrogate |",
        "| this report | lower bound | **absolute, by arithmetic** | the deployed one |",
    ]
)


# --------------------------------------------------------------------------------------
# The ensemble, as arrays (pure; unit-tested against LightGBM's own output)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tree:
    """One decision tree flattened into arrays, for fast interval propagation.

    Internal node ``i`` splits on ``feature[i] <= threshold[i]``, going to ``left[i]`` or
    ``right[i]``; a node with ``feature[i] < 0`` is a leaf holding ``value[i]``. Flat arrays
    rather than the nested dictionaries LightGBM dumps, because the propagation below walks
    these tens of millions of times.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    @property
    def n_nodes(self) -> int:
        """Total nodes, internal and leaf."""
        return len(self.feature)


def parse_tree(structure: dict[str, Any]) -> Tree:
    """Flatten one LightGBM ``tree_structure`` dictionary into a :class:`Tree`."""
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def _add(node: dict[str, Any]) -> int:
        idx = len(feature)
        if "leaf_value" in node or "split_feature" not in node:
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(node.get("leaf_value", 0.0)))
            return idx
        feature.append(int(node["split_feature"]))
        threshold.append(float(node["threshold"]))
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        left[idx] = _add(node["left_child"])
        right[idx] = _add(node["right_child"])
        return idx

    _add(structure)
    return Tree(
        feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float64),
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        value=np.asarray(value, dtype=np.float64),
    )


def parse_booster(dump: dict[str, Any]) -> list[Tree]:
    """Flatten every tree in a ``booster_.dump_model()`` payload."""
    return [parse_tree(info["tree_structure"]) for info in dump.get("tree_info", [])]


def tree_margin(tree: Tree, x: np.ndarray) -> float:
    """Evaluate one tree at a point — the reference the interval bound must contain."""
    node = 0
    while tree.feature[node] >= 0:
        f = int(tree.feature[node])
        node = int(tree.left[node] if x[f] <= tree.threshold[node] else tree.right[node])
    return float(tree.value[node])


def ensemble_margin(trees: list[Tree], x: np.ndarray) -> float:
    """Raw (pre-sigmoid) ensemble score at a point, summed over trees."""
    return float(sum(tree_margin(t, x) for t in trees))


def tree_bounds(tree: Tree, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Sound ``(min, max)`` of one tree over the box ``[lo, hi]``.

    Exact for a single tree: a split whose threshold falls outside the box prunes one child
    entirely, and a split inside it makes both children reachable, so the recursion visits
    precisely the leaves an input in the box could land on. The looseness the report measures
    enters only when these per-tree extremes are summed, since nothing forces the trees to
    agree about which input realised them.
    """
    stack = [0]
    lowest = 0.0
    highest = 0.0
    first = True
    while stack:
        node = stack.pop()
        f = int(tree.feature[node])
        if f < 0:
            v = float(tree.value[node])
            if first:
                lowest = highest = v
                first = False
            else:
                lowest = min(lowest, v)
                highest = max(highest, v)
            continue
        t = tree.threshold[node]
        if hi[f] <= t:
            stack.append(int(tree.left[node]))
        elif lo[f] > t:
            stack.append(int(tree.right[node]))
        else:
            stack.append(int(tree.left[node]))
            stack.append(int(tree.right[node]))
    return lowest, highest


def ensemble_bounds(trees: list[Tree], lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Sound ``(min, max)`` of the ensemble margin over the box, by summing per-tree extremes."""
    low = 0.0
    high = 0.0
    for tree in trees:
        tl, th = tree_bounds(tree, lo, hi)
        low += tl
        high += th
    return low, high


def perturbation_box(
    x: np.ndarray, radius: float, up: np.ndarray, down: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The L-infinity box an attacker can reach, respecting per-feature direction limits.

    ``up`` and ``down`` are boolean masks: whether the attacker can raise or lower each
    feature. A feature they control in neither direction is pinned, which is what makes the
    threat-model arm meaningful rather than cosmetic.
    """
    lo = x - radius * np.asarray(down, dtype=float)
    hi = x + radius * np.asarray(up, dtype=float)
    return lo, hi


def is_certified(
    trees: list[Tree],
    x: np.ndarray,
    radius: float,
    threshold: float,
    up: np.ndarray,
    down: np.ndarray,
    predicted_attack: bool,
) -> bool:
    """Can *any* perturbation inside the box flip this verdict? ``True`` means provably not."""
    lo, hi = perturbation_box(x, radius, up, down)
    low, high = ensemble_bounds(trees, lo, hi)
    return low >= threshold if predicted_attack else high < threshold


def certified_radius(
    trees: list[Tree],
    x: np.ndarray,
    threshold: float,
    up: np.ndarray,
    down: np.ndarray,
    *,
    max_radius: float,
    steps: int,
) -> float:
    """Largest radius provably safe for this flow, by bisection on the sound bound.

    Monotone by construction — a bigger box can only widen the bounds — so bisection is
    exact to the tolerance implied by ``steps``. Returns ``0.0`` when even an infinitesimal
    box cannot be certified, and ``max_radius`` when the search never fails inside the range
    (a censored observation, reported as such rather than silently as the true radius).
    """
    predicted_attack = ensemble_margin(trees, x) >= threshold
    if not is_certified(trees, x, 0.0, threshold, up, down, predicted_attack):
        return 0.0
    if is_certified(trees, x, max_radius, threshold, up, down, predicted_attack):
        return max_radius
    low, high = 0.0, max_radius
    for _ in range(int(steps)):
        mid = 0.5 * (low + high)
        if is_certified(trees, x, mid, threshold, up, down, predicted_attack):
            low = mid
        else:
            high = mid
    return low


def batched_margin(trees: list[Tree]) -> Callable[[np.ndarray], np.ndarray]:
    """A pure-Python batch scorer, for callers without a compiled booster to hand."""

    def _score(rows: np.ndarray) -> np.ndarray:
        return np.asarray([ensemble_margin(trees, row) for row in np.atleast_2d(rows)])

    return _score


def attack_radius(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    threshold: float,
    up: np.ndarray,
    down: np.ndarray,
    *,
    max_radius: float,
    steps: int,
    n_random: int,
    rng: np.random.Generator,
) -> float:
    """Smallest radius at which a concrete attack succeeds — an **upper** bound on the truth.

    A coordinate-wise best response (each movable feature independently pushed to whichever
    end of its allowed range lowers the margin most, then applied together) plus a random
    search inside the box. Neither is optimal, so this can only *overstate* the true radius —
    exactly the direction that makes the sandwich meaningful, since the truth then lies
    between this and the certificate.

    ``predict`` scores a batch of rows and returns raw margins. Taking it as a callable keeps
    the search honest about what it is — an attack, not a proof, so it may use the compiled
    booster — while letting tests drive it with the flattened trees and no LightGBM at all.
    """
    movable = np.flatnonzero(up | down)
    if float(predict(x[None, :])[0]) < threshold or movable.size == 0:
        return 0.0

    def _breaks(radius: float) -> bool:
        lo, hi = perturbation_box(x, radius, up, down)
        # One batch: every movable feature pushed alone to each end of its range.
        probes = np.repeat(x[None, :], 2 * movable.size, axis=0)
        for k, f in enumerate(movable):
            probes[2 * k, f] = lo[f]
            probes[2 * k + 1, f] = hi[f]
        margins = predict(probes)
        if float(margins.min()) < threshold:
            return True
        # Apply every feature's individually-best direction at once, and probe the interior.
        combined = x.copy()
        for k, f in enumerate(movable):
            combined[f] = lo[f] if margins[2 * k] <= margins[2 * k + 1] else hi[f]
        trials = lo + rng.random((int(n_random), len(x))) * (hi - lo)
        batch = np.vstack([combined[None, :], trials])
        return bool(float(predict(batch).min()) < threshold)

    if not _breaks(max_radius):
        return math.inf
    low, high = 0.0, max_radius
    for _ in range(int(steps)):
        mid = 0.5 * (low + high)
        if _breaks(mid):
            high = mid
        else:
            low = mid
    return high


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class ThreatModel:
    """Which directions an attacker may move which features."""

    name: str
    up: np.ndarray
    down: np.ndarray
    note: str

    @property
    def n_movable(self) -> int:
        """Features the attacker can move at all."""
        return int(np.count_nonzero(self.up | self.down))


@dataclass
class ThreatResult:
    """Certified and attacked radii for one threat model."""

    name: str
    note: str
    n_movable: int
    n_flows: int
    median_certified: float
    mean_certified: float
    share_certified_at_budget: float
    median_attack: float
    median_gap: float
    n_uncertified: int
    n_unattacked: int
    radii: np.ndarray


@dataclass
class VerifyStudy:
    """Everything the report renders."""

    n_trees: int
    n_nodes: int
    n_flows: int
    n_features: int
    threshold_fpr: float
    budget: float
    results: list[ThreatResult]
    exactness_checked: int
    max_reconstruction_error: float


def _threat_models(
    feature_names: list[str], controllable: list[str], n_features: int
) -> list[ThreatModel]:
    """The three threat models the report compares, from adversarial fantasy to reality."""
    everything = np.ones(n_features, dtype=bool)
    ctrl = np.zeros(n_features, dtype=bool)
    idx = controllable_indices(feature_names, controllable)
    ctrl[np.asarray(idx, dtype=int)] = True
    return [
        ThreatModel(
            name="unrestricted",
            up=everything,
            down=everything.copy(),
            note="every feature, either direction: the guarantee papers usually certify",
        ),
        ThreatModel(
            name="controllable features only",
            up=ctrl,
            down=ctrl.copy(),
            note="only what padding, delays and dummy traffic can touch",
        ),
        ThreatModel(
            name="controllable, inflate only",
            up=ctrl,
            down=np.zeros(n_features, dtype=bool),
            note="an attacker can add bytes, packets and delay; it cannot un-send them",
        ),
    ]


def run_verify_trees(settings: Settings) -> VerifyStudy:
    """Prove per-flow robustness radii for the deployed ensemble under three threat models."""
    cfg: VerifyTreesConfig = settings.verify_trees
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.supervised.backend = "lightgbm"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())

    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    booster = model.model.booster_
    trees = parse_booster(booster.dump_model())

    benign = variant.labels.benign_label
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    p_threshold = threshold_at_fpr(y_val, s_val, settings.thresholds.primary_fpr)
    p_threshold = float(np.clip(p_threshold, 1e-9, 1 - 1e-9))
    margin_threshold = math.log(p_threshold / (1.0 - p_threshold))

    # The flattened trees must reproduce LightGBM's own raw score, or nothing below is a proof.
    check_idx = np.arange(min(cfg.exactness_checks, len(x_test)))
    reference = np.asarray(booster.predict(x_test[check_idx], raw_score=True), dtype=float)
    reconstructed = np.array([ensemble_margin(trees, x_test[i]) for i in check_idx])
    max_err = float(np.max(np.abs(reference - reconstructed))) if len(check_idx) else 0.0
    if max_err > cfg.exactness_tolerance:
        raise ValueError(
            f"Flattened ensemble disagrees with LightGBM by {max_err:.3e}; refusing to "
            "report a verification result that is not about the deployed model."
        )
    logger.info("Tree reconstruction verified", extra={"max_error": max_err})

    # Verify the flows whose robustness matters: attacks the detector actually catches.
    scores = np.asarray([ensemble_margin(trees, row) for row in x_test])
    caught = np.flatnonzero((y_test == 1) & (scores >= margin_threshold))
    rng = np.random.default_rng(variant.seed)
    if len(caught) > cfg.n_flows:
        caught = rng.choice(caught, size=cfg.n_flows, replace=False)
    caught = np.sort(caught)

    results: list[ThreatResult] = []

    def booster_predict(rows: np.ndarray) -> np.ndarray:
        return np.asarray(booster.predict(np.atleast_2d(rows), raw_score=True), dtype=float)

    threat_models = _threat_models(
        feature_names, variant.robustness.controllable_features, x_test.shape[1]
    )
    for tm in threat_models:
        certified: list[float] = []
        attacked: list[float] = []
        for i in caught:
            certified.append(
                certified_radius(
                    trees,
                    x_test[i],
                    margin_threshold,
                    tm.up,
                    tm.down,
                    max_radius=cfg.max_radius,
                    steps=cfg.bisection_steps,
                )
            )
            attacked.append(
                attack_radius(
                    booster_predict,
                    x_test[i],
                    margin_threshold,
                    tm.up,
                    tm.down,
                    max_radius=cfg.max_radius,
                    steps=cfg.bisection_steps,
                    n_random=cfg.attack_samples,
                    rng=rng,
                )
            )
        cert = np.asarray(certified)
        att = np.asarray(attacked)
        finite = np.isfinite(att)
        results.append(
            ThreatResult(
                name=tm.name,
                note=tm.note,
                n_movable=tm.n_movable,
                n_flows=len(caught),
                median_certified=float(np.median(cert)) if len(cert) else 0.0,
                mean_certified=float(np.mean(cert)) if len(cert) else 0.0,
                share_certified_at_budget=float(np.mean(cert >= cfg.budget)) if len(cert) else 0.0,
                median_attack=float(np.median(att[finite])) if finite.any() else math.inf,
                median_gap=(
                    float(np.median(att[finite] - cert[finite])) if finite.any() else math.inf
                ),
                n_uncertified=int(np.sum(cert <= 0.0)),
                n_unattacked=int(np.sum(~finite)),
                radii=cert,
            )
        )
        logger.info(
            "Threat model verified",
            extra={"model": tm.name, "median_certified": results[-1].median_certified},
        )

    return VerifyStudy(
        n_trees=len(trees),
        n_nodes=int(sum(t.n_nodes for t in trees)),
        n_flows=len(caught),
        n_features=x_test.shape[1],
        threshold_fpr=settings.thresholds.primary_fpr,
        budget=cfg.budget,
        results=results,
        exactness_checked=len(check_idx),
        max_reconstruction_error=max_err,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_verify_trees_report(settings: Settings) -> Path:
    """Run the verification study and write the report + figures."""
    study = run_verify_trees(settings)

    sandwich = plots.plot_barh(
        [f"{r.name}: certified" for r in study.results]
        + [f"{r.name}: attack found" for r in study.results if math.isfinite(r.median_attack)],
        [r.median_certified for r in study.results]
        + [r.median_attack for r in study.results if math.isfinite(r.median_attack)],
        xlabel="L-infinity radius (standardised feature units)",
        title="Proved safe below, broken above: the truth is in between",
        out_path=settings.paths.figures_dir / SANDWICH_FIGURE,
        xmax=settings.verify_trees.max_radius,
    )
    grid = np.linspace(0.0, settings.verify_trees.max_radius, 40)
    threat = plots.plot_lines(
        {
            r.name: (grid, np.array([float(np.mean(r.radii >= g)) for g in grid]))
            for r in study.results
        },
        xlabel="L-infinity radius (standardised feature units)",
        ylabel="share of caught attacks provably robust",
        title="How much of the detector's fragility is the threat model's fault",
        out_path=settings.paths.figures_dir / THREAT_FIGURE,
    )

    report = _render(study, sandwich, threat)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote tree-verification report", extra={"path": str(out_path)})

    with track_run(settings, "verify_trees") as run:
        run.log_params({"n_trees": study.n_trees, "n_flows": study.n_flows})
        metrics = {"max_reconstruction_error": study.max_reconstruction_error}
        for r in study.results:
            key = r.name.replace(" ", "_").replace(",", "")
            metrics[f"median_certified_{key}"] = r.median_certified
            metrics[f"share_certified_{key}"] = r.share_certified_at_budget
        run.log_metrics(metrics)
        run.log_artifact(sandwich)
        run.log_artifact(threat)
        run.log_artifact(out_path)
    return out_path


def _results_table(study: VerifyStudy) -> str:
    rows = [
        "| threat model | features movable | median certified radius | median attack radius "
        "| gap | provably robust at the budget | never certified |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in study.results:
        attack = "none found" if not math.isfinite(r.median_attack) else f"{r.median_attack:.3f}"
        gap = "-" if not math.isfinite(r.median_gap) else f"{r.median_gap:.3f}"
        rows.append(
            f"| **{r.name}** | {r.n_movable} of {study.n_features} "
            f"| {r.median_certified:.3f} | {attack} | {gap} "
            f"| {r.share_certified_at_budget:.1%} | {r.n_uncertified} of {r.n_flows} |"
        )
    return "\n".join(rows)


def _sandwich_read(study: VerifyStudy) -> str:
    if not study.results:
        return ""
    unrestricted = study.results[0]
    if not math.isfinite(unrestricted.median_attack):
        return (
            f"Under the unrestricted threat model the median certified radius is "
            f"{unrestricted.median_certified:.3f} and the attack search found nothing inside "
            f"the search range at all, so the sandwich has no upper slice: the truth is "
            "somewhere above the certificate and beyond where this attack looked."
        )
    ratio = unrestricted.median_attack / max(unrestricted.median_certified, 1e-9)
    tightness = (
        "tight enough to be useful — the two bounds are within a small factor, so the "
        "interval relaxation is not throwing much away on this ensemble"
        if ratio < 3
        else (
            f"loose: the attack needs {ratio:.1f}x the certified radius, so somewhere between "
            "the independent-tree relaxation and the suboptimality of the attack there is a "
            "factor of several unaccounted for. Both are plausible culprits and this report "
            "cannot separate them without the exact max-clique solve"
        )
    )
    return (
        f"Every flow is bracketed. Below {unrestricted.median_certified:.3f} (median) the "
        f"verdict is **provably** unchangeable — arithmetic, not sampling. Above "
        f"{unrestricted.median_attack:.3f} (median) an attack that actually exists flips it. The "
        f"true radius lies between, and the gap is {tightness}."
    )


def _threat_read(study: VerifyStudy) -> str:
    if len(study.results) < 3:
        return ""
    unrestricted, controllable, inflate = study.results[0], study.results[1], study.results[2]
    lift = inflate.median_certified / max(unrestricted.median_certified, 1e-9)
    return (
        f"The three rows are the same proof under three different assumptions about the "
        f"adversary, and the spread between them is large. Certifying against *any* "
        f"perturbation of all {study.n_features} features — the guarantee most papers report, "
        f"because it is the cleanest to state — gives a median radius of "
        f"{unrestricted.median_certified:.3f}, leaving "
        f"{unrestricted.share_certified_at_budget:.1%} of caught attacks provably robust at the "
        f"{study.budget:.2f} budget. Restricting to the "
        f"{controllable.n_movable} features an attacker can actually shape raises that to "
        f"{controllable.median_certified:.3f}, and forbidding the physically impossible "
        f"direction — you can pad a flow, delay it, add dummy packets, but you cannot un-send "
        f"bytes already on the wire — reaches {inflate.median_certified:.3f}, "
        f"**{lift:.1f}x** the unrestricted radius, with "
        f"{inflate.share_certified_at_budget:.1%} of caught attacks provably robust.\n\n"
        "None of those numbers is more correct than the others; they answer different "
        "questions. But only the last one answers the question an operator has, and the "
        "distance between the first and the last is a measure of how much apparent fragility "
        "is an artefact of certifying against an adversary who does not exist. Reporting the "
        "unrestricted number alone would understate the detector; reporting only the "
        "restricted one would be marketing. Both belong in the table."
    )


def _render(study: VerifyStudy, sandwich: Path, threat: Path) -> str:
    return f"""# NetSentry — Deterministic Verification: Proving the Verdict, Not Sampling It

_Synthetic stand-in. Honest temporal/binary split. The deployed ensemble ({study.n_trees} trees,
{study.n_nodes:,} nodes) is flattened and verified exactly; radii are L-infinity in the same
standardised feature units the [evasion](robustness.md) and [certification](certify.md) studies
use, at the validated {study.threshold_fpr:.1%}-FPR operating point. {study.n_flows} caught
attack flows verified per threat model._

## Why this report exists

Three reports here ask how far an attacker must push a flow before the detector lets it
through, and they answer with different kinds of statement:

{_COMPARISON_TABLE}

Gradient-boosted trees are one of the few model families where the third row is available. A
tree ensemble is piecewise-constant over axis-aligned boxes, so its output over an input box can
be bounded by interval arithmetic: at each split, if the box lies wholly on one side, follow that
child; if it straddles the threshold, follow both and keep the extremes. Sum the per-tree minima
and you have a sound lower bound on the ensemble margin over the entire box. If that still clears
the decision threshold, **no** perturbation inside the box can flip the verdict — not "none
found", none.

## First, is it the same model?

A proof about a re-implementation is worth nothing. The flattened arrays are checked against
LightGBM's own `raw_score` on {study.exactness_checked} test flows; the largest disagreement is
**{study.max_reconstruction_error:.2e}**, i.e. floating-point noise. The run aborts rather than
reports if that check fails, because a verification report about an approximation of the deployed
model is worse than no verification report.

## The sandwich

{_results_table(study)}

{_sandwich_read(study)}

![certified and attacked radii](../figures/{sandwich.name})

## Which adversary are we certifying against?

{_threat_read(study)}

![robust share vs radius, by threat model](../figures/{threat.name})

## Scope

The bound is **sound but incomplete**. Bounding each tree independently ignores that every tree
reads the same input, so the summed extremes may correspond to a combination of leaves no single
input can realise; the bound therefore never certifies an attackable point, but does refuse to
certify some safe ones. Closing that gap exactly means searching consistent leaf tuples — a
max-clique problem (Chen et al., NeurIPS 2019) — which is not implemented here, and the sandwich
above is the honest substitute: it prices the looseness rather than hiding it. The attack side of
the sandwich is a coordinate-wise best response plus random search, so it is an upper bound from
a specific adversary and a stronger attack would narrow the bracket from above. Verification runs
in the **standardised feature space** the model sees, so a radius of `r` means `r` standard
deviations on every movable feature at once, which is the same convention
[certify.md](certify.md) and [robustness.md](robustness.md) use and lets the three read against
each other. Only attack flows the detector currently catches are verified — the robustness of a
verdict it is already getting wrong is not an interesting quantity. Feature interactions imposed
by the pipeline (a rate and its numerator moving together) are not modelled: the box treats every
feature as independently movable, which makes the certificate conservative in the operator's
favour and the attack optimistic in the attacker's."""
