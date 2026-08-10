"""Not all misclassifications cost the same — evaluating against the ATT&CK taxonomy.

The multiclass report treats the label set as flat: `DoS Hulk`, `DoS GoldenEye`, `PortScan`
and `BENIGN` are four unrelated symbols, and getting any one of them wrong is one unit of
error. An analyst does not experience it that way. Calling `DoS Hulk` a `DoS GoldenEye` sends
them to the same playbook, the same containment step and the same after-action note — the
mistake is invisible in the response. Calling it `PortScan` sends them to a different
playbook. Calling it `BENIGN` sends them nowhere. Flat accuracy scores those three failures
identically, which means the metric is measuring something the operator does not care about.

The fix is to score against a **hierarchy**, and the hierarchy does not have to be invented:
this repository already maps every class onto MITRE ATT&CK, and ATT&CK is itself a tree.
Tactic (*why* the adversary is doing it) contains technique (*how*), which contains the
concrete class. That yields a four-level taxonomy — verdict, tactic, technique, class —
built entirely from `netsentry.intel.attack_mapping`, so it cannot drift away from what the
API already tells its callers and it is not a structure chosen to make the numbers look good.

Two things follow. First, **hierarchical precision/recall/F1** (Kiritchenko et al. 2006):
score the *ancestor sets* rather than the leaves, so a prediction that gets the tactic right
and the class wrong earns partial credit in exact proportion to how much of the path it got
right. Second, an **error decomposition** an operator can act on: every mistake is exactly one
of within-technique, within-tactic, cross-tactic, missed attack, or false alarm, and those
five categories have wildly different operational costs. Pricing them turns "88% accurate"
into "the wrong playbook runs on 3% of alerts".

The study then asks whether *training* hierarchically helps, not just scoring: a local
classifier per parent node (route benign/attack, then pick a tactic, then pick a class within
it) against the deployed flat multiclass model. The flat model optimises exact accuracy and
the hierarchical one optimises each decision in the context of the one above it, so if the
hierarchy carries real structure the second should make cheaper mistakes even where it makes
more of them. Whether it does is measured, not assumed.

Run on the **stratified** split, deliberately. The headline temporal split shares no attack
classes across the day boundary at all (see [uncertainty](uncertainty.md)), so a multiclass
question is not even well posed on it — every test class would be one the classifier has never
seen, and a taxonomy comparison would be measuring novelty rather than the taxonomy. The
stratified split is optimistic about detection and correct about class structure, which is
what this study is about; the report says so rather than borrowing the headline's authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import f1_score

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.intel.attack_mapping import technique_for
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import HierarchyConfig

logger = get_logger(__name__)

REPORT_NAME = "hierarchy.md"
COST_FIGURE = "hierarchy_cost.png"
ERROR_FIGURE = "hierarchy_errors.png"

ROOT = "root"
ATTACK_NODE = "attack"
BENIGN_NODE = "benign"

# The five ways a multiclass verdict can be wrong, in increasing operational severity.
ERROR_KINDS: tuple[str, ...] = (
    "exact",
    "within_technique",
    "within_tactic",
    "cross_tactic",
    "false_alarm",
    "missed_attack",
)


# --------------------------------------------------------------------------------------
# The taxonomy (pure; derived from the ATT&CK mapping, unit-tested)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """Label -> ancestor path, rooted at the verdict and refined through ATT&CK.

    Paths exclude the root itself, because a root every label shares would inflate
    hierarchical precision and recall by a constant that says nothing.
    """

    paths: dict[str, tuple[str, ...]]

    def ancestors(self, label: str) -> tuple[str, ...]:
        """Path from the verdict down to (and including) the label."""
        return self.paths.get(label, (label,))

    def depth(self) -> int:
        """Longest path in the tree, for reporting."""
        return max((len(p) for p in self.paths.values()), default=0)

    def parent_of(self, label: str, level: int) -> str:
        """Ancestor at a given depth (0 = verdict, 1 = tactic, 2 = technique)."""
        path = self.ancestors(label)
        return path[level] if level < len(path) else path[-1]

    def path_distance(self, a: str, b: str) -> int:
        """Edges between two labels through their lowest common ancestor."""
        pa, pb = self.ancestors(a), self.ancestors(b)
        shared = 0
        for x, y in zip(pa, pb, strict=False):
            if x != y:
                break
            shared += 1
        return (len(pa) - shared) + (len(pb) - shared)


def build_taxonomy(labels: list[str], benign_label: str) -> Taxonomy:
    """Four-level taxonomy (verdict / tactic / technique / class) from the ATT&CK mapping.

    A class the mapping does not know still gets a path — it hangs off the attack node
    directly rather than being dropped — so an unmapped label degrades the resolution of the
    hierarchy instead of silently leaving the evaluation.
    """
    paths: dict[str, tuple[str, ...]] = {}
    for label in labels:
        if label == benign_label:
            paths[label] = (BENIGN_NODE, label)
            continue
        technique = technique_for(label)
        if technique is None:
            paths[label] = (ATTACK_NODE, label)
            continue
        paths[label] = (
            ATTACK_NODE,
            technique.tactic,
            f"{technique.technique_id} {technique.technique_name}",
            label,
        )
    return Taxonomy(paths=paths)


def hierarchical_prf(
    y_true: np.ndarray, y_pred: np.ndarray, taxonomy: Taxonomy
) -> tuple[float, float, float]:
    """Hierarchical precision/recall/F1 over ancestor sets (Kiritchenko et al. 2006).

    A prediction is credited with every ancestor it shares with the truth, so getting the
    tactic right and the class wrong is worth strictly more than getting neither right, and
    an exact hit is worth the whole path. On a flat taxonomy (every path of length one) these
    collapse to micro-averaged precision/recall/F1, which is the property that makes them a
    generalisation rather than a different metric.
    """
    inter = correct = predicted = 0
    for t, p in zip(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), strict=True):
        at, ap = set(taxonomy.ancestors(t)), set(taxonomy.ancestors(p))
        inter += len(at & ap)
        correct += len(at)
        predicted += len(ap)
    precision = inter / predicted if predicted else 0.0
    recall = inter / correct if correct else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def error_kind(true: str, pred: str, taxonomy: Taxonomy, benign_label: str) -> str:
    """Which of the five operationally distinct outcomes this verdict is.

    Ordered by what it costs a responder: the same playbook, a sibling playbook, the wrong
    playbook, a wasted investigation, or no investigation at all.
    """
    if true == pred:
        return "exact"
    if true == benign_label:
        return "false_alarm"
    if pred == benign_label:
        return "missed_attack"
    pt, pp = taxonomy.ancestors(true), taxonomy.ancestors(pred)
    if len(pt) > 2 and len(pp) > 2 and pt[2] == pp[2]:
        return "within_technique"
    if len(pt) > 1 and len(pp) > 1 and pt[1] == pp[1]:
        return "within_tactic"
    return "cross_tactic"


def error_profile(
    y_true: np.ndarray, y_pred: np.ndarray, taxonomy: Taxonomy, benign_label: str
) -> dict[str, float]:
    """Share of verdicts falling in each error category (sums to one)."""
    counts = Counter(
        error_kind(t, p, taxonomy, benign_label)
        for t, p in zip(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), strict=True)
    )
    n = max(sum(counts.values()), 1)
    return {kind: counts.get(kind, 0) / n for kind in ERROR_KINDS}


def playbook_cost(profile: dict[str, float], costs: dict[str, float]) -> float:
    """Expected response cost per verdict under a per-error-kind playbook price."""
    return float(sum(profile.get(kind, 0.0) * costs.get(kind, 0.0) for kind in ERROR_KINDS))


# --------------------------------------------------------------------------------------
# The two classifiers
# --------------------------------------------------------------------------------------
def _fit_flat(
    settings: Settings, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """The deployed approach: one multiclass model over the leaves."""
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train)
    return np.asarray(model.predict(x_test)).astype(str)


def _fit_node(
    settings: Settings, x: np.ndarray, y: np.ndarray, x_test: np.ndarray
) -> np.ndarray | str:
    """Fit one router; a node with a single child needs no model and returns that child."""
    unique = np.unique(y)
    if len(unique) < 2:
        return str(unique[0])
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x, y)
    return np.asarray(model.predict(x_test)).astype(str)


def fit_local_per_parent(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    taxonomy: Taxonomy,
) -> np.ndarray:
    """Local-classifier-per-parent-node: route down the tree, one model per branching node.

    Each model answers a question in the context of the answer above it — "given this is an
    attack, which tactic?" — so it is trained on, and only ever asked about, rows where that
    question arises. The cost is exposure to error propagation: a row misrouted at the
    verdict node can never be recovered further down, which is the standard objection to
    hierarchical classification and one the error decomposition below makes visible.
    """
    labels = np.asarray(y_train).astype(str)
    level0 = np.array([taxonomy.parent_of(v, 0) for v in labels])
    routed = _fit_node(settings, x_train, level0, x_test)
    verdict = np.full(len(x_test), routed) if isinstance(routed, str) else routed

    out = np.array([BENIGN_NODE] * len(x_test), dtype=object)
    benign_leaf = next(
        (lab for lab, path in taxonomy.paths.items() if path[0] == BENIGN_NODE), BENIGN_NODE
    )
    out[verdict == BENIGN_NODE] = benign_leaf

    attack_rows = np.flatnonzero(labels != benign_leaf)
    attack_test = np.flatnonzero(verdict == ATTACK_NODE)
    if not attack_rows.size or not attack_test.size:
        return out.astype(str)

    tactics = np.array([taxonomy.parent_of(v, 1) for v in labels[attack_rows]])
    routed_tactic = _fit_node(settings, x_train[attack_rows], tactics, x_test[attack_test])
    chosen = (
        np.full(len(attack_test), routed_tactic)
        if isinstance(routed_tactic, str)
        else routed_tactic
    )

    for tactic in np.unique(chosen):
        train_rows = attack_rows[tactics == tactic]
        test_rows = attack_test[chosen == tactic]
        if not test_rows.size:
            continue
        if not train_rows.size:  # a tactic the router invented; fall back to the modal class
            out[test_rows] = str(Counter(labels[attack_rows]).most_common(1)[0][0])
            continue
        leaf = _fit_node(settings, x_train[train_rows], labels[train_rows], x_test[test_rows])
        out[test_rows] = np.full(len(test_rows), leaf) if isinstance(leaf, str) else leaf
    return out.astype(str)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class ModelHierarchy:
    """One classifier scored flat and hierarchically, plus what its errors cost."""

    name: str
    exact_accuracy: float
    macro_f1: float
    h_precision: float
    h_recall: float
    h_f1: float
    mean_distance: float
    profile: dict[str, float]
    cost: float


@dataclass
class ClassRow:
    """Where one class's errors land in the taxonomy."""

    attack_class: str
    n_flows: int
    exact: float
    within_tactic_or_better: float


@dataclass
class HierarchyStudy:
    """Everything the report renders."""

    depth: int
    max_distance: int
    n_leaves: int
    n_tactics: int
    models: list[ModelHierarchy]
    classes: list[ClassRow]
    costs: dict[str, float]
    taxonomy_lines: list[str] = field(default_factory=list)


def _score(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, taxonomy: Taxonomy, settings: Settings
) -> ModelHierarchy:
    benign = settings.labels.benign_label
    hp, hr, hf = hierarchical_prf(y_true, y_pred, taxonomy)
    profile = error_profile(y_true, y_pred, taxonomy, benign)
    distances = [
        taxonomy.path_distance(t, p)
        for t, p in zip(y_true.astype(str), y_pred.astype(str), strict=True)
    ]
    costs = settings.hierarchy.error_costs()
    return ModelHierarchy(
        name=name,
        exact_accuracy=float(np.mean(y_true.astype(str) == y_pred.astype(str))),
        macro_f1=float(f1_score(y_true.astype(str), y_pred.astype(str), average="macro")),
        h_precision=hp,
        h_recall=hr,
        h_f1=hf,
        mean_distance=float(np.mean(distances)) if distances else 0.0,
        profile=profile,
        cost=playbook_cost(profile, costs),
    )


def run_hierarchy(settings: Settings) -> HierarchyStudy:
    """Score the flat and hierarchical classifiers against the ATT&CK taxonomy."""
    cfg: HierarchyConfig = settings.hierarchy
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "multiclass"
    variant.mlflow.enabled = False

    train = load_split(variant, "stratified", "train")
    test = load_split(variant, "stratified", "test")
    y_train = train[MULTICLASS_TARGET].astype(str).to_numpy()
    y_test = test[MULTICLASS_TARGET].astype(str).to_numpy()

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_test = pipeline.transform(test)

    taxonomy = build_taxonomy(
        sorted(set(y_train.tolist()) | set(y_test.tolist())), variant.labels.benign_label
    )
    logger.info("Built taxonomy", extra={"leaves": len(taxonomy.paths), "depth": taxonomy.depth()})

    flat_pred = _fit_flat(variant, x_train, y_train, x_test)
    local_pred = fit_local_per_parent(variant, x_train, y_train, x_test, taxonomy)

    models = [
        _score("flat multiclass (deployed)", y_test, flat_pred, taxonomy, variant),
        _score("local classifier per parent", y_test, local_pred, taxonomy, variant),
    ]
    for m in models:
        logger.info(
            "Hierarchy model scored",
            extra={
                "model": m.name,
                "accuracy": round(m.exact_accuracy, 4),
                "h_f1": round(m.h_f1, 4),
            },
        )

    classes = _class_rows(y_test, flat_pred, taxonomy, variant.labels.benign_label, cfg)
    tactics = {p[1] for p in taxonomy.paths.values() if len(p) > 2}
    leaves = sorted(taxonomy.paths)
    return HierarchyStudy(
        depth=taxonomy.depth(),
        max_distance=max((taxonomy.path_distance(a, b) for a in leaves for b in leaves), default=0),
        n_leaves=len(taxonomy.paths),
        n_tactics=len(tactics),
        models=models,
        classes=classes,
        costs=cfg.error_costs(),
        taxonomy_lines=_taxonomy_lines(taxonomy, variant.labels.benign_label),
    )


def _class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taxonomy: Taxonomy,
    benign_label: str,
    cfg: HierarchyConfig,
) -> list[ClassRow]:
    """Per class: exact hits, and hits that at least reached the right tactic."""
    rows: list[ClassRow] = []
    truth = y_true.astype(str)
    pred = y_pred.astype(str)
    for cls in sorted(set(truth.tolist())):
        if cls == benign_label:
            continue
        idx = np.flatnonzero(truth == cls)
        if idx.size < cfg.min_class_rows:
            continue
        kinds = [error_kind(cls, p, taxonomy, benign_label) for p in pred[idx]]
        near = {"exact", "within_technique", "within_tactic"}
        rows.append(
            ClassRow(
                attack_class=cls,
                n_flows=int(idx.size),
                exact=float(np.mean([k == "exact" for k in kinds])),
                within_tactic_or_better=float(np.mean([k in near for k in kinds])),
            )
        )
    return sorted(rows, key=lambda r: r.within_tactic_or_better - r.exact, reverse=True)


def _taxonomy_lines(taxonomy: Taxonomy, benign_label: str) -> list[str]:
    """The tree, rendered as indented text for the report."""
    lines: list[str] = []
    by_tactic: dict[str, dict[str, list[str]]] = {}
    for label, path in sorted(taxonomy.paths.items()):
        if path[0] == BENIGN_NODE:
            continue
        tactic = path[1] if len(path) > 1 else "(unmapped)"
        technique = path[2] if len(path) > 2 else "(unmapped)"
        by_tactic.setdefault(tactic, {}).setdefault(technique, []).append(label)
    lines.append(f"{BENIGN_NODE}/")
    lines.append(f"  {benign_label}")
    lines.append(f"{ATTACK_NODE}/")
    for tactic, techniques in sorted(by_tactic.items()):
        lines.append(f"  {tactic}/")
        for technique, leaves in sorted(techniques.items()):
            lines.append(f"    {technique}/")
            lines.extend(f"      {leaf}" for leaf in sorted(leaves))
    return lines


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_hierarchy_report(settings: Settings) -> Path:
    """Run the taxonomy study and write the report + figures."""
    study = run_hierarchy(settings)

    cost_fig = plots.plot_barh(
        [m.name for m in study.models],
        [m.cost for m in study.models],
        xlabel="expected response cost per verdict (playbook units)",
        title="What the errors cost, not how many there are",
        out_path=settings.paths.figures_dir / COST_FIGURE,
        xmax=max((m.cost for m in study.models), default=1.0) * 1.2,
    )
    flat = study.models[0]
    error_fig = plots.plot_barh(
        [k.replace("_", " ") for k in ERROR_KINDS if k != "exact"],
        [flat.profile.get(k, 0.0) for k in ERROR_KINDS if k != "exact"],
        xlabel="share of all verdicts",
        title="How the deployed model's errors distribute over the taxonomy",
        out_path=settings.paths.figures_dir / ERROR_FIGURE,
        xmax=max((flat.profile.get(k, 0.0) for k in ERROR_KINDS if k != "exact"), default=0.1)
        * 1.3,
    )

    report = _render(study, cost_fig, error_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote hierarchy report", extra={"path": str(out_path)})

    with track_run(settings, "hierarchy") as run:
        run.log_params({"depth": study.depth, "leaves": study.n_leaves})
        metrics: dict[str, float] = {}
        for m in study.models:
            key = m.name.split()[0]
            metrics[f"accuracy_{key}"] = m.exact_accuracy
            metrics[f"h_f1_{key}"] = m.h_f1
            metrics[f"cost_{key}"] = m.cost
        run.log_metrics(metrics)
        run.log_artifact(cost_fig)
        run.log_artifact(error_fig)
        run.log_artifact(out_path)
    return out_path


def _model_table(study: HierarchyStudy) -> str:
    rows = [
        "| classifier | exact accuracy | macro-F1 | hier. P | hier. R | hier. F1 "
        "| mean tree distance | cost/verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in study.models:
        rows.append(
            f"| {m.name} | {m.exact_accuracy:.3f} | {m.macro_f1:.3f} | {m.h_precision:.3f} "
            f"| {m.h_recall:.3f} | **{m.h_f1:.3f}** | {m.mean_distance:.2f} | {m.cost:.3f} |"
        )
    return "\n".join(rows)


def _profile_table(study: HierarchyStudy) -> str:
    header = "| outcome | playbook cost | " + " | ".join(m.name for m in study.models) + " |"
    rows = [header, "|---|---|" + "---|" * len(study.models)]
    for kind in ERROR_KINDS:
        cells = " | ".join(f"{m.profile.get(kind, 0.0):.2%}" for m in study.models)
        rows.append(f"| {kind.replace('_', ' ')} | {study.costs.get(kind, 0.0):.1f} | {cells} |")
    return "\n".join(rows)


def _class_table(study: HierarchyStudy) -> str:
    rows = [
        "| attack class | flows | named exactly | reached the right tactic | partial credit |",
        "|---|---|---|---|---|",
    ]
    for c in study.classes:
        rows.append(
            f"| {c.attack_class} | {c.n_flows} | {c.exact:.1%} "
            f"| {c.within_tactic_or_better:.1%} | {c.within_tactic_or_better - c.exact:+.1%} |"
        )
    return "\n".join(rows)


def _headline_read(study: HierarchyStudy) -> str:
    flat = study.models[0]
    errors = 1.0 - flat.exact_accuracy
    cheap = flat.profile.get("within_technique", 0.0) + flat.profile.get("within_tactic", 0.0)
    worst = max(
        ((k, flat.profile.get(k, 0.0)) for k in ERROR_KINDS if k != "exact"), key=lambda kv: kv[1]
    )
    cheap_share = cheap / errors if errors > 0 else 0.0
    worst_share = worst[1] / errors if errors > 0 else 0.0
    direction = "below" if flat.h_f1 < flat.exact_accuracy else "above"
    return (
        f"The deployed flat model is {flat.exact_accuracy:.1%} accurate, so {errors:.1%} of its "
        f"verdicts are wrong, and the taxonomy says what kind of wrong. Only "
        f"**{cheap_share:.0%} of those errors are the forgivable kind** — a sibling name under "
        f"the right tactic, where the analyst runs the correct playbook anyway. "
        f"**{worst_share:.0%} are {worst[0].replace('_', ' ')}**, the most expensive row in the "
        f"schedule. So the honest reading of hierarchical F1 here is not that the flat metric "
        f"was too harsh: hF1 lands at {flat.h_f1:.3f}, *{direction}* the {flat.exact_accuracy:.3f} "
        f"flat accuracy reports.\n\nThat direction is worth pausing on, because partial credit "
        f"can only add. It goes the other way because hierarchical recall divides by path "
        f"length, and in this tree an attack is four levels deep while benign is two — so "
        f"calling an attack benign costs twice what calling a benign flow an attack does, "
        f"automatically, with nobody choosing a weight. For a detector that is the right "
        f"asymmetry, and it is a property of the taxonomy rather than of the cost schedule "
        f"below. The mean error travels {flat.mean_distance:.2f} edges of a possible "
        f"{study.max_distance}."
    )


def _comparison_read(study: HierarchyStudy) -> str:
    if len(study.models) < 2:
        return ""
    flat, local = study.models[0], study.models[1]
    d_acc = local.exact_accuracy - flat.exact_accuracy
    d_cost = local.cost - flat.cost
    if d_cost < -1e-9 and d_acc < 0:
        d_macro = local.macro_f1 - flat.macro_f1
        rare = (
            f" It also gains {d_macro:+.3f} macro-F1, which is the same effect seen from the "
            "other side: macro-F1 weights every class equally and the rare classes are exactly "
            "the ones a flat model starves, because splitting a fixed model capacity across "
            "thirteen leaves spends most of it on the two that dominate the rows."
            if d_macro > 0
            else ""
        )
        return (
            f"Training hierarchically is the trade it is supposed to be, and it is a trade "
            f"worth making here: the local-per-parent classifier gives up {-d_acc:.1%} of exact "
            f"accuracy and returns {-d_cost:.3f} of cost per verdict — a "
            f"{-d_cost / flat.cost:.0%} reduction — because the errors it makes are cheaper "
            f"ones. Specifically it converts missed attacks into false alarms: "
            f"{flat.profile['missed_attack']:.2%} down to {local.profile['missed_attack']:.2%}, "
            f"against false alarms {flat.profile['false_alarm']:.2%} up to "
            f"{local.profile['false_alarm']:.2%}. Routing benign-versus-attack as its own "
            "decision, before any question about which attack, is what buys that: the router "
            "sees every hostile flow as one class instead of thirteen sparse ones, so it is "
            f"better at the only question whose error costs five units.{rare} A flat metric "
            "scores this model as the worse of the two. An operator would deploy it."
        )
    if d_cost < -1e-9:
        return (
            f"The local-per-parent classifier wins on both axes here — {d_acc:+.1%} accuracy "
            f"and {d_cost:+.3f} cost per verdict — which is a stronger result than the method "
            "promises and therefore worth being suspicious of. The mechanism is visible in the "
            "profile above: routing benign-vs-attack first lets that model use every attack "
            "row as one class instead of splitting its capacity across rare leaves, and the "
            "rare leaves are where flat multiclass loses."
        )
    return (
        f"Training hierarchically does **not** pay here: the local-per-parent classifier is "
        f"{d_acc:+.1%} on exact accuracy and {d_cost:+.3f} on cost per verdict, so it is worse "
        "or no better on the metric it was supposed to improve. Error propagation is the "
        "likely mechanism — a row misrouted at the verdict node cannot be recovered by any "
        "model below it, and the flat classifier has no such chokepoint. The scoring half of "
        "this study stands on its own regardless: the taxonomy is the right yardstick for "
        "these errors whether or not it is the right training structure."
    )


def _class_read(study: HierarchyStudy) -> str:
    if not study.classes:
        return ""
    best = study.classes[0]
    gain = best.within_tactic_or_better - best.exact
    if gain < 0.01:
        return (
            "No class gains meaningfully from partial credit, which means the model's errors "
            "are not near-misses within a family — when it is wrong it is wrong about the "
            "tactic, and the taxonomy cannot rescue it."
        )
    return (
        f"**{best.attack_class}** is where the flat metric understates most: named exactly "
        f"{best.exact:.1%} of the time, but routed to the right tactic {gain:+.1%} more often "
        "than that. Those are the verdicts where an analyst opens the alert, sees a sibling "
        "class name, and runs the correct playbook anyway. Classes with no gap are the honest "
        "failures — when the model is wrong about them it is wrong about what the adversary "
        "was trying to do, and no amount of partial credit should hide that."
    )


_SCOPE = """The taxonomy is derived from `netsentry.intel.attack_mapping`, and those ATT&CK
mappings are **indicative**: CIC-IDS2017 is not labelled with ATT&CK IDs, so the tactic and
technique assigned to each class is a documented judgement about the capture scenario rather
than ground truth. Every number here inherits that judgement. Changing one class's tactic
would move partial credit between the within-tactic and cross-tactic rows, which is exactly
why the mapping lives in one module that serving, the coverage report and this study all read
rather than being restated here.

The playbook costs are a stated schedule (`hierarchy.cost_*` in config), not a measurement.
They encode an ordering nobody would dispute — a missed attack costs more than the wrong
playbook, which costs more than a sibling name — but their magnitudes are illustrative, and
the comparison between classifiers is only as meaningful as that ordering. The alternative,
leaving every error priced identically, is the flat metric this report exists to replace.

This runs on the **stratified** split, which is optimistic about detection: it is the
reference split, not the headline. That is deliberate rather than convenient — the temporal
split shares no attack classes across the day boundary, so on it every test class is unseen
and a multiclass taxonomy comparison would be measuring novelty rather than structure. Read
the detection numbers here alongside [evaluation](evaluation.md), which states the gap."""


def _render(study: HierarchyStudy, cost_fig: Path, error_fig: Path) -> str:
    tree = "\n".join(study.taxonomy_lines)
    return f"""# NetSentry — Errors That Cost Different Amounts

_Synthetic stand-in. Stratified/multiclass split (see Scope). Taxonomy: {study.n_leaves}
classes under {study.n_tactics} ATT&CK tactics, depth {study.depth}._

## Why this report exists

The multiclass report treats the label set as flat, so calling `DoS Hulk` a `DoS GoldenEye`
and calling it `BENIGN` are both worth exactly one unit of error. They are not the same
mistake. The first sends an analyst to the right playbook under a slightly wrong name; the
second sends them nowhere. A metric that cannot tell those apart is not measuring the thing
the response cares about.

The hierarchy needed to tell them apart already exists in this repository: every class is
mapped onto MITRE ATT&CK, and ATT&CK is a tree of tactics containing techniques containing
concrete behaviours. Using it costs nothing and cannot be accused of being chosen to flatter
the model.

```
{tree}
```

## Scoring against the tree

{_model_table(study)}

{_headline_read(study)}

![how the errors distribute](../figures/{error_fig.name})

## Where every verdict lands

{_profile_table(study)}

{_comparison_read(study)}

![what the errors cost](../figures/{cost_fig.name})

## Which classes the flat metric understates

{_class_table(study)}

{_class_read(study)}

## Scope

{_SCOPE}"""
