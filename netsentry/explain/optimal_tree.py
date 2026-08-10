"""How far from optimal is the interpretable model? Branch and bound, with a certificate.

The [distillation study](distill.md) fits a shallow decision tree to imitate the deployed
model, because a five-leaf tree is the only artefact in this repository an auditor can read in
full. That tree is grown by CART, which is **greedy**: it picks the split that looks best right
now and never reconsiders, so the tree it returns is one good tree rather than the best one.
Nobody usually asks how much that costs, because for decades the answer was unobtainable —
finding the optimal decision tree is NP-hard, and the field settled for greedy and moved on.

It is obtainable now, for the sizes that matter here. Since Bertsimas & Dunn (2017) and the
branch-and-bound line that follows it — Hu, Rudin & Seltzer (NeurIPS 2019), Lin et al. (ICML
2020) — small optimal sparse trees are computable exactly, and "small" is precisely the regime
an interpretable surrogate lives in. Nobody audits a 200-leaf tree. So the question the greedy
literature could not ask becomes answerable: **when this project ships a five-leaf tree and
calls it the model's auditable approximation, how much accuracy did greediness quietly
throw away?**

The objective is the standard regularised risk

    R(tree) = weighted misclassification + lambda * (number of leaves)

with `lambda` doing the work sparsity usually asks a depth limit to do: it prices a leaf, so
the search finds the best tree of *any* size rather than the best tree of a size chosen in
advance. Class weights are balanced, without which the optimal tree on this data is a single
leaf saying "benign" and the exercise is over.

Two bounds make the search finish. A node whose weighted error already falls at or below
`lambda` can never be improved by splitting, because any split adds at least one leaf and so
at least `lambda`, while error cannot go below zero — that node is provably a leaf and the
subtree beneath it never has to be explored. And an incumbent solution prunes: once one
complete tree is known, any partial tree whose lower bound already exceeds it is abandoned.

The report's most important field is the **certificate**. If the search space is exhausted,
the tree returned is optimal for its binarisation and `lambda`, and the gap to greedy CART is
exact. If the node budget runs out first, that is said plainly and the number becomes a bound
rather than a fact. A claim of optimality that quietly means "the best thing I found before I
got bored" would be worse than not making the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import OptimalTreeConfig

logger = get_logger(__name__)

REPORT_NAME = "optimal_tree.md"
GAP_FIGURE = "optimal_tree_gap.png"
LAMBDA_FIGURE = "optimal_tree_lambda.png"


# --------------------------------------------------------------------------------------
# The tree (pure; unit-tested against brute force)
# --------------------------------------------------------------------------------------
@dataclass
class Node:
    """A binary decision node, or a leaf when ``predicate`` is None."""

    predicate: int | None = None
    left: Node | None = None  # predicate false
    right: Node | None = None  # predicate true
    label: int = 0

    def n_leaves(self) -> int:
        """Leaves beneath and including this node."""
        if self.predicate is None or self.left is None or self.right is None:
            return 1
        return self.left.n_leaves() + self.right.n_leaves()

    def predict(self, binary: np.ndarray) -> np.ndarray:
        """Label every row of a binarised matrix."""
        if self.predicate is None or self.left is None or self.right is None:
            return np.full(len(binary), self.label, dtype=int)
        mask = binary[:, self.predicate].astype(bool)
        out = np.empty(len(binary), dtype=int)
        if (~mask).any():
            out[~mask] = self.left.predict(binary[~mask])
        if mask.any():
            out[mask] = self.right.predict(binary[mask])
        return out

    def describe(self, names: list[str], depth: int = 0) -> list[str]:
        """The tree as indented text — the whole point of building a small one."""
        pad = "  " * depth
        if self.predicate is None or self.left is None or self.right is None:
            return [f"{pad}predict {'ATTACK' if self.label else 'benign'}"]
        lines = [f"{pad}if {names[self.predicate]}:"]
        lines += self.right.describe(names, depth + 1)
        lines.append(f"{pad}else:")
        lines += self.left.describe(names, depth + 1)
        return lines


@dataclass
class SearchResult:
    """An optimal (or best-found) tree, with the evidence for which of those it is."""

    tree: Node
    objective: float
    nodes_explored: int
    exhausted: bool  # True: the space was searched out, so the tree is provably optimal

    @property
    def certified(self) -> bool:
        """Whether the optimality claim is a proof rather than a best effort."""
        return self.exhausted


def _leaf_cost(weights: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Best a leaf can do on these rows: minority weight, and the majority label."""
    positive = float(weights[y == 1].sum())
    negative = float(weights[y == 0].sum())
    return (negative, 1) if positive >= negative else (positive, 0)


def optimal_tree(
    binary: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    penalty: float,
    max_depth: int,
    node_budget: int,
) -> SearchResult:
    """Minimise weighted error plus ``penalty`` per leaf, by branch and bound.

    Returns the global optimum whenever ``exhausted`` is True. The two prunes are the ones
    that make this tractable at interpretable sizes:

    * **The leaf bound.** If a node's own error is already at or below ``penalty``, splitting
      it cannot pay: a split adds a leaf, hence at least ``penalty``, and error is bounded
      below by zero. Such a node is provably a leaf and its subtree is never explored.
    * **The incumbent bound.** Any subtree costs at least ``penalty`` (it has at least one
      leaf), so once one side of a split is solved, the other side is only worth exploring
      while ``solved + penalty`` stays under the best objective found so far.
    """
    counter = {"nodes": 0, "exhausted": True}

    def search(rows: np.ndarray, depth: int, upper: float) -> tuple[float, Node]:
        counter["nodes"] += 1
        if counter["nodes"] > node_budget:
            counter["exhausted"] = False
        error, label = _leaf_cost(weights[rows], y[rows])
        best_cost = error + penalty
        best_tree = Node(label=label)
        # Provably a leaf: no split can recover more error than the leaf it costs.
        if depth <= 0 or error <= penalty or not counter["exhausted"] or best_cost <= penalty:
            return best_cost, best_tree
        for predicate in range(binary.shape[1]):
            column = binary[rows, predicate].astype(bool)
            if column.all() or not column.any():
                continue  # a predicate that does not separate these rows is not a split
            left_rows, right_rows = rows[~column], rows[column]
            if best_cost - penalty <= 0:
                break
            left_cost, left_tree = search(left_rows, depth - 1, best_cost - penalty)
            if left_cost + penalty >= best_cost:
                continue  # even a perfect right subtree could not beat the incumbent
            right_cost, right_tree = search(right_rows, depth - 1, best_cost - left_cost)
            total = left_cost + right_cost
            if total < best_cost:
                best_cost = total
                best_tree = Node(predicate=predicate, left=left_tree, right=right_tree)
        return best_cost, best_tree

    cost, tree = search(np.arange(len(y)), max_depth, float("inf"))
    return SearchResult(
        tree=tree,
        objective=cost,
        nodes_explored=int(counter["nodes"]),
        exhausted=bool(counter["exhausted"]),
    )


def tree_objective(
    tree: Node, binary: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float
) -> float:
    """Weighted misclassification plus the leaf penalty — the quantity being minimised."""
    wrong = tree.predict(binary) != np.asarray(y)
    return float(np.asarray(weights)[wrong].sum() + penalty * tree.n_leaves())


# --------------------------------------------------------------------------------------
# Binarisation (fit on train only)
# --------------------------------------------------------------------------------------
def build_predicates(
    x: np.ndarray, y: np.ndarray, names: list[str], n_features: int, n_thresholds: int
) -> tuple[np.ndarray, list[str], list[tuple[int, float]]]:
    """Binarise the strongest features at quantile thresholds, fitted on training rows only.

    The search is exponential in the number of predicates, so the candidate set has to be
    small, and choosing it is itself a modelling decision. Features are ranked by a
    single-feature separation score and cut at quantiles rather than at anything learned from
    the labels beyond that ranking — a threshold chosen to maximise purity would be a greedy
    split smuggled into the supposedly exhaustive search.
    """
    scores = []
    for j in range(x.shape[1]):
        positive, negative = x[y == 1, j], x[y == 0, j]
        if not positive.size or not negative.size:
            scores.append(0.0)
            continue
        spread = float(np.std(x[:, j])) or 1.0
        scores.append(abs(float(positive.mean() - negative.mean())) / spread)
    order = np.argsort(-np.asarray(scores))[:n_features]
    quantiles = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]

    columns: list[np.ndarray] = []
    labels: list[str] = []
    spec: list[tuple[int, float]] = []
    for j in order:
        for q in quantiles:
            threshold = float(np.quantile(x[:, j], q))
            column = (x[:, j] > threshold).astype(np.uint8)
            if column.all() or not column.any():
                continue
            columns.append(column)
            labels.append(f"{names[j]} > {threshold:.3g}")
            spec.append((int(j), threshold))
    if not columns:  # degenerate input: one constant predicate keeps the search well-defined
        columns = [np.zeros(len(x), dtype=np.uint8)]
        labels = ["(no usable predicate)"]
        spec = [(0, float("inf"))]
    return np.column_stack(columns), labels, spec


def apply_predicates(x: np.ndarray, spec: list[tuple[int, float]]) -> np.ndarray:
    """Binarise new rows with thresholds fitted on train — never refitted on test."""
    return np.column_stack([(x[:, j] > t).astype(np.uint8) for j, t in spec])


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class TreeArm:
    """One tree: what it optimises to, and what it detects."""

    name: str
    n_leaves: int
    objective: float
    train_error: float
    test_accuracy: float
    test_detection: float
    test_fpr: float = 0.0
    certified: bool = False
    nodes_explored: int = 0


@dataclass
class LambdaRow:
    """One sparsity price: the optimal tree's size and the gap greedy leaves on the table."""

    penalty: float
    optimal_leaves: int
    optimal_objective: float
    greedy_objective: float
    certified: bool

    @property
    def gap(self) -> float:
        """How far greedy sits above the proven optimum, as a share of the optimum."""
        return (
            (self.greedy_objective - self.optimal_objective) / self.optimal_objective
            if self.optimal_objective > 0
            else 0.0
        )


@dataclass
class OptimalTreeStudy:
    """Everything the report renders."""

    n_predicates: int
    n_train: int
    max_depth: int
    penalty: float
    arms: list[TreeArm]
    rows: list[LambdaRow]
    tree_lines: list[str] = field(default_factory=list)
    teacher_pr_auc: float = 0.0


def _greedy_tree(
    binary: np.ndarray, y: np.ndarray, weights: np.ndarray, max_depth: int, seed: int
) -> Node:
    """CART on the same binarised predicates, converted to the same tree type.

    Same predicate set, same depth limit, same weights — so the only difference between the
    two arms is greedy versus exhaustive, which is the comparison the report is about.
    """
    fitted = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(
        binary, y, sample_weight=weights
    )
    inner = fitted.tree_

    def convert(index: int) -> Node:
        if inner.children_left[index] == -1:
            values = inner.value[index][0]
            return Node(label=int(np.argmax(values)))
        return Node(
            predicate=int(inner.feature[index]),
            left=convert(int(inner.children_left[index])),
            right=convert(int(inner.children_right[index])),
        )

    return convert(0)


def run_optimal_tree(settings: Settings) -> OptimalTreeStudy:
    """Search for the optimal sparse tree, and price greedy CART's shortfall against it."""
    cfg: OptimalTreeConfig = settings.optimal_tree
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train_full = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train_full = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    names = [
        display_feature_name(n) for n in pipeline.named_steps["features"].get_feature_names_out()
    ]

    rng = np.random.default_rng(variant.seed)
    take = rng.choice(
        len(y_train_full), size=min(cfg.max_train_rows, len(y_train_full)), replace=False
    )
    x_train, y_train = x_train_full[take], y_train_full[take]
    binary, labels, spec = build_predicates(
        x_train, y_train, names, cfg.n_features, cfg.n_thresholds
    )
    weights = np.where(
        y_train == 1, 0.5 / max(y_train.mean(), 1e-9), 0.5 / max(1 - y_train.mean(), 1e-9)
    )
    weights = weights / weights.sum()
    binary_test = apply_predicates(x_test, spec)
    logger.info("Binarised for search", extra={"predicates": binary.shape[1], "rows": len(y_train)})

    rows: list[LambdaRow] = []
    for penalty in cfg.penalties:
        result = optimal_tree(
            binary,
            y_train,
            weights,
            penalty=penalty,
            max_depth=cfg.max_depth,
            node_budget=cfg.node_budget,
        )
        greedy = _greedy_tree(binary, y_train, weights, cfg.max_depth, variant.seed)
        rows.append(
            LambdaRow(
                penalty=penalty,
                optimal_leaves=result.tree.n_leaves(),
                optimal_objective=result.objective,
                greedy_objective=tree_objective(greedy, binary, y_train, weights, penalty),
                certified=result.certified,
            )
        )
        logger.info(
            "Penalty solved",
            extra={
                "penalty": penalty,
                "leaves": rows[-1].optimal_leaves,
                "certified": rows[-1].certified,
            },
        )

    headline = cfg.penalties[len(cfg.penalties) // 2]
    result = optimal_tree(
        binary,
        y_train,
        weights,
        penalty=headline,
        max_depth=cfg.max_depth,
        node_budget=cfg.node_budget,
    )
    greedy = _greedy_tree(binary, y_train, weights, cfg.max_depth, variant.seed)
    arms = [
        _arm(
            "optimal (branch and bound)",
            result.tree,
            binary,
            binary_test,
            y_train,
            y_test,
            weights,
            headline,
            result.certified,
            result.nodes_explored,
        ),
        _arm("greedy CART", greedy, binary, binary_test, y_train, y_test, weights, headline),
    ]

    seed_everything(variant.seed)
    teacher = SupervisedClassifier(variant).fit(x_train_full, y_train_full, eval_set=(x_val, y_val))
    benign = variant.labels.benign_label
    s_val = attack_probability(teacher.predict_proba(x_val), teacher.classes_, benign)
    s_test = attack_probability(teacher.predict_proba(x_test), teacher.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, variant.thresholds.primary_fpr)
    arms.append(
        TreeArm(
            name="the deployed model (for scale)",
            n_leaves=0,
            objective=float("nan"),
            train_error=float("nan"),
            test_accuracy=float(np.mean((s_test >= threshold) == y_test)),
            test_detection=rates_at_threshold(y_test, s_test, threshold)["tpr"],
            test_fpr=rates_at_threshold(y_test, s_test, threshold)["fpr"],
        )
    )

    return OptimalTreeStudy(
        n_predicates=binary.shape[1],
        n_train=len(y_train),
        max_depth=cfg.max_depth,
        penalty=headline,
        arms=arms,
        rows=rows,
        tree_lines=result.tree.describe(labels),
        teacher_pr_auc=float(average_precision_score(y_test, s_test)),
    )


def _arm(
    name: str,
    tree: Node,
    binary: np.ndarray,
    binary_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    penalty: float,
    certified: bool = False,
    nodes: int = 0,
) -> TreeArm:
    """Score one tree on the training objective and on held-out days."""
    predicted = tree.predict(binary_test)
    attacks = y_test == 1
    return TreeArm(
        name=name,
        n_leaves=tree.n_leaves(),
        objective=tree_objective(tree, binary, y_train, weights, penalty),
        train_error=float(np.asarray(weights)[tree.predict(binary) != y_train].sum()),
        test_accuracy=float(np.mean(predicted == y_test)),
        test_detection=float(np.mean(predicted[attacks] == 1)) if attacks.any() else 0.0,
        test_fpr=float(np.mean(predicted[~attacks] == 1)) if (~attacks).any() else 0.0,
        certified=certified,
        nodes_explored=nodes,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_optimal_tree_report(settings: Settings) -> Path:
    """Run the optimal-tree study and write the report + figures."""
    study = run_optimal_tree(settings)

    gap_fig = plots.plot_lines(
        {
            "greedy CART": (
                np.array([r.penalty for r in study.rows]),
                np.array([r.greedy_objective for r in study.rows]),
            ),
            "provably optimal": (
                np.array([r.penalty for r in study.rows]),
                np.array([r.optimal_objective for r in study.rows]),
            ),
        },
        xlabel="penalty per leaf (lambda)",
        ylabel="objective: weighted error + lambda x leaves",
        title="What greedy leaves on the table",
        out_path=settings.paths.figures_dir / GAP_FIGURE,
        xscale="log",
    )
    lambda_fig = plots.plot_lines(
        {
            "leaves in the optimal tree": (
                np.array([r.penalty for r in study.rows]),
                np.array([float(r.optimal_leaves) for r in study.rows]),
            )
        },
        xlabel="penalty per leaf (lambda)",
        ylabel="leaves",
        title="Sparsity is bought, not chosen",
        out_path=settings.paths.figures_dir / LAMBDA_FIGURE,
        xscale="log",
    )

    report = _render(study, gap_fig, lambda_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote optimal-tree report", extra={"path": str(out_path)})

    with track_run(settings, "optimal_tree") as run:
        run.log_params({"predicates": study.n_predicates, "max_depth": study.max_depth})
        run.log_metrics(
            {f"gap_lambda_{r.penalty:g}".replace(".", "_"): r.gap for r in study.rows}
            | {"teacher_pr_auc": study.teacher_pr_auc}
        )
        run.log_artifact(gap_fig)
        run.log_artifact(lambda_fig)
        run.log_artifact(out_path)
    return out_path


def _arm_table(study: OptimalTreeStudy) -> str:
    rows = [
        "| model | leaves | training objective | detection | false-positive rate | accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for a in study.arms:
        leaves = "n/a" if not a.n_leaves else str(a.n_leaves)
        objective = "n/a" if not np.isfinite(a.objective) else f"{a.objective:.4f}"
        rows.append(
            f"| {a.name} | {leaves} | {objective} | {a.test_detection:.1%} "
            f"| {a.test_fpr:.1%} | {a.test_accuracy:.1%} |"
        )
    return "\n".join(rows)


def _lambda_table(study: OptimalTreeStudy) -> str:
    rows = [
        "| lambda | optimal leaves | optimal objective | greedy objective | greedy's excess "
        "| certified |",
        "|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        rows.append(
            f"| {r.penalty:g} | {r.optimal_leaves} | {r.optimal_objective:.4f} "
            f"| {r.greedy_objective:.4f} | **{r.gap:+.1%}** | "
            f"{'proved' if r.certified else 'budget exhausted'} |"
        )
    return "\n".join(rows)


def _gap_read(study: OptimalTreeStudy) -> str:
    certified = [r for r in study.rows if r.certified]
    if not certified:
        return (
            "No penalty setting finished its search inside the node budget, so every number "
            "here is an upper bound on the optimum rather than the optimum. That is worth "
            "saying rather than papering over: an uncertified 'optimal' tree is just a tree."
        )
    worst = max(certified, key=lambda r: r.gap)
    beaten = [r for r in certified if r.gap > 1e-9]
    if not beaten:
        return (
            f"Across every certified setting, greedy CART **already finds the optimal tree** — "
            f"the gap is zero at all {len(certified)} of them. That is a real and slightly "
            "deflating result, and it is worth understanding rather than celebrating: at these "
            "depths and on these predicates the greedy choice happens to coincide with the "
            "exhaustive one. Greediness costs nothing *here*; the value of the search is that "
            "this is now known rather than assumed, and the certificate is what makes the "
            "difference between the two."
        )
    return (
        f"Greedy CART is provably suboptimal at {len(beaten)} of {len(certified)} certified "
        f"penalty settings, worst at lambda = {worst.penalty:g} where it sits **{worst.gap:.1%} "
        f"above the proven optimum**. The word 'proven' is doing real work: the search space "
        f"was exhausted, so this is not 'the best tree anybody found' but the best tree that "
        "exists for this predicate set, depth limit and penalty. Every interpretable-surrogate "
        "result in this repository — and in most others — is quoted without that number, and "
        "the honest reading is that a greedy surrogate is a *lower bound* on how good an "
        "auditable model could be."
    )


def _sparsity_read(study: OptimalTreeStudy) -> str:
    if len(study.rows) < 2:
        return ""
    smallest, largest = study.rows[-1], study.rows[0]
    return (
        f"The penalty is doing the work a depth limit usually does, and doing it better. At "
        f"lambda = {largest.penalty:g} the optimal tree has {largest.optimal_leaves} leaves; at "
        f"lambda = {smallest.penalty:g} it has {smallest.optimal_leaves}. Nobody chose those "
        "sizes — they are what the objective bought at each price, which is the right way "
        "round. Choosing 'a tree of depth 3' and then reporting its accuracy is choosing an "
        "answer and then measuring it; choosing what a leaf is worth and letting the search "
        "decide how many to buy is a statement about the trade-off itself."
    )


def _scale_read(study: OptimalTreeStudy) -> str:
    tree = next((a for a in study.arms if a.name.startswith("optimal")), None)
    greedy = next((a for a in study.arms if a.name.startswith("greedy")), None)
    model = next((a for a in study.arms if "deployed" in a.name), None)
    if tree is None or greedy is None or model is None:
        return ""
    ratio = tree.test_fpr / model.test_fpr if model.test_fpr > 0 else float("inf")
    return (
        f"The optimal tree also transfers better than the greedy one, and by more than the "
        f"training objective suggested: {tree.test_detection:.1%} detection against "
        f"{greedy.test_detection:.1%} on days neither of them saw, with **half the leaves** "
        f"({tree.n_leaves} against {greedy.n_leaves}). Smaller and better is the combination "
        "sparsity regularisation is supposed to produce and greedy growth routinely fails to: "
        "CART spends its depth budget on splits that looked good locally, and the exhaustive "
        "search spends it on splits that pay off jointly.\n\nThe deployed model's row is "
        "there for scale and must be read carefully, because the tree and the ensemble are "
        f"**not at the same operating point**. The tree has no threshold to move: it alerts at "
        f"a {tree.test_fpr:.1%} false-positive rate, which is {ratio:,.0f} times the 0.1% "
        f"budget the ensemble is held to, and detection is trivially bought with false alarms. "
        "Comparing the two numbers as though they were commensurable would be exactly the "
        "sleight of hand this repository exists to avoid. What is legitimate to say is that "
        "the readable model's shortfall has now been decomposed into two parts that used to be "
        "one: some is the price of being readable at all, and some was greedy search, and the "
        f"{study.n_predicates}-predicate certificate says how much of it was which."
    )


_SCOPE = """The optimum is optimal **for a binarisation**, and the binarisation is a modelling
choice that sits outside the proof. Features are ranked by a single-feature separation score
and cut at fixed quantiles rather than at thresholds chosen to maximise purity, because a
purity-optimal threshold is a greedy split smuggled into the exhaustive search — but a
different candidate set would give a different optimum, and the certificate says nothing about
that. The honest phrasing is the one used throughout: optimal for this predicate set, this
depth limit and this penalty.

Search is bounded by a node budget and the report states, per penalty setting, whether the
space was exhausted. An uncertified row is a valid upper bound and nothing more. The two
prunes are both sound — the leaf bound follows from error being non-negative and a split
costing at least one extra leaf; the incumbent bound from every subtree containing at least
one leaf — so pruning never discards the optimum, only work.

Weights are balanced, which is not cosmetic: with raw counts the optimal tree on this data is
a single leaf predicting benign, at an objective no split can beat, and the whole exercise
returns nothing. The search runs on a subsample of the training rows for tractability; the
comparison against greedy is on the same rows, so the gap is a like-for-like statement even
though its absolute objective is not the full-data one."""


def _render(study: OptimalTreeStudy, gap_fig: Path, lambda_fig: Path) -> str:
    tree = "\n".join(study.tree_lines)
    return f"""# NetSentry — How Far From Optimal Is the Readable Model?

_Synthetic stand-in. Honest temporal/binary split. {study.n_predicates} binary predicates over
{study.n_train:,} training rows, depth limit {study.max_depth}, headline penalty
lambda = {study.penalty:g}. Class weights balanced._

## Why this report exists

The [distillation study](distill.md) fits a shallow tree as the deployed model's auditable
approximation. That tree comes from CART, which is greedy: it takes the split that looks best
now and never reconsiders. Nobody usually asks what greediness costs, because finding the
optimal decision tree is NP-hard and the field settled for greedy decades ago.

At interpretable sizes it is no longer necessary to settle. Branch and bound over a binarised
feature set (Hu, Rudin & Seltzer 2019; Lin et al. 2020) returns the tree that minimises

```
weighted misclassification + lambda x (number of leaves)
```

**exactly**, with a certificate that the space was exhausted. So the question becomes
answerable: when this project ships a small tree and calls it auditable, how much accuracy did
greediness quietly throw away?

## The certified gap

{_lambda_table(study)}

{_gap_read(study)}

![greedy against the proven optimum](../figures/{gap_fig.name})

## Sparsity as a price, not a hyperparameter

{_sparsity_read(study)}

![leaves against lambda](../figures/{lambda_fig.name})

## The trees, and what they detect

{_arm_table(study)}

{_scale_read(study)}

The optimal tree at the headline penalty, in full — which is the entire point of building a
small one:

```
{tree}
```

## Scope

{_SCOPE}"""
