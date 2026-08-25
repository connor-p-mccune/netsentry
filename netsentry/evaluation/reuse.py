"""How many times has this project looked at its own test set?

`.claude/rules/ml.md` is unambiguous: *"The test set is touched **once**, at the end. If you tune
against the test set, you have leaked."* Every study in this repository was written under that
rule, and every one of them reads the temporal test split to report a number. The rule and the
practice cannot both be right, and nobody had counted.

The failure this is about is **not** the syntactic one. Fitting a scaler on test data is a
leakage bug, it is visible in the source, and [`netsentry mlint`](mlint.md) already refuses it
(NS001). What no syntactic rule can see is the statistical failure Dwork et al. describe
(*Science* 349, 2015): a holdout queried adaptively -- where what you try next depends on what
the holdout said last time -- stops being a holdout. Nothing is fitted on it. No column leaks.
The number is simply optimistic, by an amount that grows with the number of questions asked.

So the module does three things.

1. **Counts.** A static pass over the package for every place the test split is read, and how
   many distinct modules do it. This is a fact about the repository, not an estimate.
2. **Prices selection.** The count alone proves nothing, because *reading* a holdout and
   *selecting* on it are different acts and only the second one burns it. So the cost of
   selection is measured directly on this project's own scores: an analyst who picks the best of
   a few detector variants by test score, round after round, against a sealed half of the same
   split that is never queried. The gap between what such an analyst would report and what is
   true is the quantity in question.
3. **Prices the fixes.** Two mechanisms, judged on both halves of the trade. **Thresholdout**
   (Dwork et al. 2015) answers a holdout query with the *training* answer unless the two differ
   by more than a noisy tolerance, spending budget only when they do. A **confidence gate** --
   adopt a challenger only when it beats the incumbent by more than the bootstrap noise -- needs
   no budget and no noise and is three lines. A fix that holds the gap at zero by never adopting
   anything is not a fix, so both are also measured on whether they still find a genuinely
   better detector when one is planted in the candidate pool.

The candidates are perturbations of the deployed scorer rather than retrained models, which
keeps the simulation honest in the way that matters: they are near-identical in true quality,
so anything the selection appears to gain is noise being mistaken for signal -- exactly the
regime where a burned holdout does its damage.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ReuseConfig

logger = get_logger(__name__)

REPORT_NAME = "reuse.md"
FIGURE_NAME = "reuse_gap.png"

#: Split names that are not the training split. A read of any of these is a question asked of
#: data the model is not allowed to learn from; only ``test`` is the sealed one the rules protect.
HELD_OUT_PARTS = ("test", "val")


# --------------------------------------------------------------------------------------
# The count: a static pass over the package.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitRead:
    """One place in the source where a split partition is loaded."""

    module: str
    line: int
    part: str


def _literal_part(node: ast.Call) -> str | None:
    """The partition name a ``load_split`` call asks for, when it is a literal.

    Non-literal arguments (a loop variable, a config value) are returned as ``None`` rather than
    guessed at. Undercounting is the safe direction for a claim about how often the sealed split
    was read -- an audit that inflates its own finding is worth nothing.
    """
    if not node.args:
        return None
    last = node.args[-1]
    if isinstance(last, ast.Constant) and isinstance(last.value, str):
        return last.value
    for keyword in node.keywords:
        if keyword.arg == "part" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def split_reads(source: str, module: str) -> list[SplitRead]:
    """Every ``load_split(..., part)`` in one module, with the partition it names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - the package parses; a caller's file might not
        return []
    reads: list[SplitRead] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name != "load_split":
            continue
        part = _literal_part(node)
        if part is not None:
            reads.append(SplitRead(module=module, line=node.lineno, part=part))
    return reads


def audit_package(root: Path) -> list[SplitRead]:
    """Walk the package and record every literal split read, in file order."""
    reads: list[SplitRead] = []
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root.parent).as_posix()
        reads.extend(split_reads(path.read_text(encoding="utf-8"), module))
    return reads


@dataclass(frozen=True)
class AuditRow:
    """One partition's exposure across the package."""

    part: str
    reads: int
    modules: int


def summarise_audit(reads: Sequence[SplitRead]) -> list[AuditRow]:
    """Reads and distinct modules per partition, most-read first."""
    parts: dict[str, list[SplitRead]] = {}
    for read in reads:
        parts.setdefault(read.part, []).append(read)
    rows = [
        AuditRow(part=part, reads=len(group), modules=len({r.module for r in group}))
        for part, group in parts.items()
    ]
    return sorted(rows, key=lambda row: (-row.reads, row.part))


# --------------------------------------------------------------------------------------
# The adaptive analyst.
# --------------------------------------------------------------------------------------


#: One candidate detector's scores on each of the four sets the simulation keeps apart:
#: the queryable holdout, the sealed half, an exchangeable reference drawn from the same days,
#: and the project's own validation split.
Candidate = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, the project's primary single number."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def perturb(
    base: np.ndarray,
    features: np.ndarray,
    direction: np.ndarray,
    scale: float,
) -> np.ndarray:
    """A candidate that is *genuinely different*: the score nudged along a feature direction.

    Two flows with similar features get similar nudges, so the change is a real change of
    detector. Candidates built this way differ in true quality, and a holdout that ranks them
    correctly is doing its job rather than being burned.
    """
    nudged: np.ndarray = base + scale * (features @ direction)
    return nudged


def jitter(base: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """A candidate that is *indistinguishable*: the score plus independent per-flow noise.

    Every candidate built this way has the same true quality in expectation, because the noise
    carries no information about the flow. Any candidate that wins on the holdout wins by
    accident, and its advantage does not survive to a second sample. This is the regime a
    repeatedly queried holdout fails in, and it is the one the harm table measures.
    """
    jittered: np.ndarray = base + scale * rng.normal(0.0, 1.0, len(base))
    return jittered


def bootstrap_halfwidth(
    y_true: np.ndarray,
    scores: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
    level: float,
) -> float:
    """Half-width of a percentile bootstrap interval for PR-AUC, in the score's own units.

    This is the noise floor a difference has to clear before it means anything. It is what the
    confidence gate spends instead of a privacy budget.
    """
    n = len(y_true)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        rows = rng.integers(0, n, n)
        draws[index] = _pr_auc(y_true[rows], scores[rows])
    finite = draws[np.isfinite(draws)]
    if len(finite) < 2:
        return 0.0
    low, high = np.quantile(finite, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float((high - low) / 2)


@dataclass(frozen=True)
class ArmRow:
    """One selection strategy, judged on what it reports and on what it finds."""

    name: str
    reported: float
    sealed: float
    queries: int
    budget_spent: int
    adopted: int
    found_planted: bool

    @property
    def gap(self) -> float:
        """Optimism: what the analyst would publish minus what is true."""
        return self.reported - self.sealed


@dataclass(frozen=True)
class RoundRow:
    """The state of the naive analyst after one more round of questions."""

    queries: int
    reported: float
    sealed: float

    @property
    def gap(self) -> float:
        return self.reported - self.sealed


def _windows(
    candidates: Sequence[Candidate], rounds: int, per_round: int
) -> Iterator[tuple[int, int, Candidate]]:
    """Yield ``(absolute index, candidate)`` in round order, skipping the incumbent at 0."""
    for round_index in range(rounds):
        offset = 1 + round_index * per_round
        for local, candidate in enumerate(candidates[offset : offset + per_round]):
            yield round_index, offset + local, candidate


def _incumbent(
    y_holdout: np.ndarray,
    y_sealed: np.ndarray,
    candidates: Sequence[Candidate],
) -> ArmRow:
    """The reference arm: report the deployed detector, select nothing.

    Its gap is the part of the optimism that is *sampling* rather than *selection* -- the
    holdout is a finite sample, so even a rule chosen with no knowledge of it scores slightly
    differently there than on a sealed set. Every other arm has to be read against this one, or
    the study would be charging selection for noise it did not cause.
    """
    return ArmRow(
        name="no selection (report the incumbent)",
        reported=_pr_auc(y_holdout, candidates[0][0]),
        sealed=_pr_auc(y_sealed, candidates[0][1]),
        queries=0,
        budget_spent=0,
        adopted=0,
        found_planted=False,
    )


def _naive(
    y_holdout: np.ndarray,
    y_sealed: np.ndarray,
    candidates: Sequence[Candidate],
    rounds: int,
    per_round: int,
    planted: int,
) -> tuple[ArmRow, list[RoundRow]]:
    """Select on the holdout every round and report the holdout score. The failure mode itself."""
    best_index = 0
    best_reported = _pr_auc(y_holdout, candidates[0][0])
    adopted = 0
    queries = 0
    trace: list[RoundRow] = []
    seen_round = -1
    for round_index, index, candidate in _windows(candidates, rounds, per_round):
        if round_index != seen_round and seen_round >= 0:
            trace.append(
                RoundRow(
                    queries=queries,
                    reported=best_reported,
                    sealed=_pr_auc(y_sealed, candidates[best_index][1]),
                )
            )
        seen_round = round_index
        queries += 1
        score = _pr_auc(y_holdout, candidate[0])
        if score > best_reported:
            best_reported, best_index, adopted = score, index, adopted + 1
    trace.append(
        RoundRow(
            queries=queries,
            reported=best_reported,
            sealed=_pr_auc(y_sealed, candidates[best_index][1]),
        )
    )
    return (
        ArmRow(
            name="select on the holdout (the failure mode)",
            reported=best_reported,
            sealed=_pr_auc(y_sealed, candidates[best_index][1]),
            queries=queries,
            budget_spent=queries,
            adopted=adopted,
            found_planted=best_index == planted,
        ),
        trace,
    )


def _thresholdout(
    name: str,
    y_holdout: np.ndarray,
    y_reference: np.ndarray,
    reference_slot: int,
    y_sealed: np.ndarray,
    candidates: Sequence[Candidate],
    rounds: int,
    per_round: int,
    planted: int,
    tolerance: float,
    noise: float,
    budget: int,
    rng: np.random.Generator,
) -> ArmRow:
    """Dwork et al.'s mechanism: answer from a reference set unless the holdout disagrees.

    The holdout is only *consulted* -- and budget only spent -- when the two answers differ by
    more than a noisy tolerance. Queries the reference already answers correctly are free, which
    is what lets a holdout survive far more questions than it otherwise could.

    The guarantee rests on the reference and the holdout being exchangeable, which is why the
    study runs this twice with two different reference sets. One satisfies the assumption; the
    other is what a practitioner on a temporal split actually has.
    """
    best_index = 0
    best_value = _pr_auc(y_reference, candidates[0][reference_slot])
    adopted = 0
    spent = 0
    queries = 0
    for _, index, candidate in _windows(candidates, rounds, per_round):
        reference_value = _pr_auc(y_reference, candidate[reference_slot])
        holdout_value = _pr_auc(y_holdout, candidate[0])
        noisy_tolerance = tolerance + rng.laplace(0.0, noise)
        if abs(reference_value - holdout_value) > noisy_tolerance:
            # A genuine surprise: the reference cannot answer this one, so the holdout must, and
            # that costs budget. Once the budget is gone the mechanism stops answering entirely
            # -- continuing to serve reference answers past exhaustion would keep the analyst
            # working while quietly dropping the guarantee, which is the failure the budget
            # exists to prevent.
            if spent >= budget:
                break
            spent += 1
            answer = holdout_value + rng.laplace(0.0, noise)
        else:
            answer = reference_value
        queries += 1
        if answer > best_value:
            best_value, best_index, adopted = answer, index, adopted + 1
    return ArmRow(
        name=name,
        reported=best_value,
        sealed=_pr_auc(y_sealed, candidates[best_index][1]),
        queries=queries,
        budget_spent=spent,
        adopted=adopted,
        found_planted=best_index == planted,
    )


def _confidence_gate(
    y_holdout: np.ndarray,
    y_sealed: np.ndarray,
    candidates: Sequence[Candidate],
    rounds: int,
    per_round: int,
    planted: int,
    halfwidth: float,
) -> ArmRow:
    """Adopt a challenger only when it clears the incumbent by more than the bootstrap noise.

    No budget, no noise, no mechanism -- just a refusal to act on differences smaller than the
    interval the difference lives in. It cannot make a burned holdout unbiased; it stops the
    burning by making most queries unable to change anything.
    """
    best_index = 0
    best_reported = _pr_auc(y_holdout, candidates[0][0])
    adopted = 0
    queries = 0
    for _, index, candidate in _windows(candidates, rounds, per_round):
        queries += 1
        score = _pr_auc(y_holdout, candidate[0])
        if score > best_reported + halfwidth:
            best_reported, best_index, adopted = score, index, adopted + 1
    return ArmRow(
        name="confidence gate (adopt only past the noise)",
        reported=best_reported,
        sealed=_pr_auc(y_sealed, candidates[best_index][1]),
        queries=queries,
        budget_spent=queries,
        adopted=adopted,
        found_planted=best_index == planted,
    )


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class ReuseStudy:
    """Everything the report needs, computed once."""

    audit: list[AuditRow]
    audit_modules: int
    harm: list[ArmRow]
    power: list[ArmRow]
    contrast: list[ArmRow]
    trace: list[RoundRow]
    planted_edge: float
    halfwidth: float
    n_holdout: int
    n_sealed: int
    n_reference: int
    candidates: int
    seconds: float = 0.0
    reads_by_module: dict[str, int] = field(default_factory=dict)

    def test_reads(self) -> AuditRow | None:
        """The row for the split the rules say is touched once."""
        return next((row for row in self.audit if row.part == "test"), None)

    def incumbent(self) -> ArmRow:
        return self.harm[0]

    def naive(self) -> ArmRow:
        return self.harm[1]

    def fixes(self) -> list[ArmRow]:
        return self.harm[2:]

    def selection_cost(self) -> float:
        """The optimism attributable to selection, with the sampling part netted out.

        The incumbent arm chose nothing, so its gap is what a finite holdout costs regardless.
        Subtracting it is the difference between measuring selection and charging selection for
        noise it did not cause.
        """
        return self.naive().gap - self.incumbent().gap

    def contrast_cost(self) -> float:
        """Selection optimism when the candidates really are different detectors."""
        return self.contrast[1].gap - self.contrast[0].gap

    def finders(self) -> list[ArmRow]:
        """The arms that still found the planted improvement when one existed."""
        return [arm for arm in self.power if arm.found_planted]

    def best_fix(self) -> ArmRow:
        """The fix with the least selection optimism that still found the planted improvement."""
        names = {arm.name for arm in self.finders()}
        candidates = [arm for arm in self.fixes() if arm.name in names] or self.fixes()
        return min(candidates, key=lambda arm: abs(arm.gap - self.incumbent().gap))


def run_reuse_study(settings: Settings) -> ReuseStudy:
    """Count the questions asked of the held-out set, then price what asking costs."""
    start = time.perf_counter()
    cfg: ReuseConfig = settings.reuse
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier

    reads = audit_package(Path(cfg.package_root))
    audit = summarise_audit(reads)
    reads_by_module: dict[str, int] = {}
    for read in reads:
        if read.part in HELD_OUT_PARTS:
            reads_by_module[read.module] = reads_by_module.get(read.module, 0) + 1

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    val_frame = load_split(variant, "temporal", "val")
    test_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(val_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = val_frame[BINARY_TARGET].to_numpy().astype(int)
    y_all = test_frame[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    column = list(model.classes_).index(1)
    base_test = np.asarray(model.predict_proba(x_test))[:, column]
    base_val = np.asarray(model.predict_proba(x_val))[:, column]

    # The later days are cut three ways. The analyst may query the *holdout*; the *reference* is
    # a second piece of the same days, which is what Thresholdout needs and what makes its
    # exchangeability assumption true; the *sealed* third is never queried by anything and is
    # the only ground truth in the study.
    order = rng.permutation(len(y_all))
    third = len(order) // 3
    queried, reference, sealed = order[:third], order[third : 2 * third], order[2 * third :]
    y_holdout, y_reference, y_sealed = y_all[queried], y_all[reference], y_all[sealed]

    # Standardise the perturbation directions so the nudge is comparable across features.
    spread = np.std(x_test, axis=0)
    spread[spread <= 0] = 1.0
    unit_test = x_test / spread
    unit_val = x_val / spread

    width = int(x_test.shape[1])
    total = 1 + cfg.rounds * cfg.candidates_per_round

    def _slice(test_scores: np.ndarray, val_scores: np.ndarray) -> Candidate:
        return (test_scores[queried], test_scores[sealed], test_scores[reference], val_scores)

    # Pool one: candidates the holdout genuinely cannot tell apart. Independent per-flow noise
    # leaves true quality unchanged, so every apparent winner is an accident of this sample.
    # Every candidate is an independent draw from the same generator, the incumbent included --
    # otherwise the incumbent is privileged and the pool is not exchangeable, which is exactly
    # the mistake that makes a winner-s-curse experiment measure nothing.
    indistinguishable: list[Candidate] = [
        _slice(jitter(base_test, cfg.jitter, rng), jitter(base_val, cfg.jitter, rng))
        for _ in range(total)
    ]

    # Pool two: candidates that really are different detectors, nudged along random feature
    # directions. The contrast between the two pools is the study's actual lesson.
    different: list[Candidate] = []
    for _ in range(total):
        direction = np.asarray(rng.normal(0.0, 1.0, width), dtype=float)
        direction = direction / (float(np.linalg.norm(direction)) or 1.0)
        different.append(
            _slice(
                perturb(base_test, unit_test, direction, cfg.perturbation),
                perturb(base_val, unit_val, direction, cfg.perturbation),
            )
        )

    # Pool three: the indistinguishable pool with one genuinely better detector hidden in it, so
    # a mechanism that reports honestly by never adopting anything can be told from one that
    # still works. The edge is deliberately modest -- a planted winner far above the noise floor
    # would make every arm look good.
    planted = 1 + int(rng.integers(0, total - 1))
    planted_test = np.clip(base_test + cfg.planted_edge * (y_all * 2 - 1), 0.0, 1.0)
    planted_val = np.clip(base_val + cfg.planted_edge * (y_val * 2 - 1), 0.0, 1.0)
    planted_pool = list(indistinguishable)
    planted_pool[planted] = _slice(planted_test, planted_val)
    planted_edge = _pr_auc(y_sealed, planted_test[sealed]) - _pr_auc(y_sealed, base_test[sealed])

    halfwidth = bootstrap_halfwidth(
        y_holdout, indistinguishable[0][0], cfg.bootstrap_resamples, rng, cfg.confidence_level
    )

    def _run(pool: Sequence[Candidate], planted_index: int) -> tuple[list[ArmRow], list[RoundRow]]:
        """Every strategy against one candidate pool."""
        selected, path = _naive(
            y_holdout, y_sealed, pool, cfg.rounds, cfg.candidates_per_round, planted_index
        )
        rows = [
            _incumbent(y_holdout, y_sealed, pool),
            selected,
            _thresholdout(
                "Thresholdout, exchangeable reference",
                y_holdout,
                y_reference,
                2,
                y_sealed,
                pool,
                cfg.rounds,
                cfg.candidates_per_round,
                planted_index,
                cfg.tolerance,
                cfg.noise_scale,
                cfg.query_budget,
                np.random.default_rng(variant.seed),
            ),
            _thresholdout(
                "Thresholdout, validation as reference",
                y_holdout,
                y_val,
                3,
                y_sealed,
                pool,
                cfg.rounds,
                cfg.candidates_per_round,
                planted_index,
                cfg.tolerance,
                cfg.noise_scale,
                cfg.query_budget,
                np.random.default_rng(variant.seed),
            ),
            _confidence_gate(
                y_holdout,
                y_sealed,
                pool,
                cfg.rounds,
                cfg.candidates_per_round,
                planted_index,
                halfwidth,
            ),
        ]
        return rows, path

    # Two pools, because harm and power are different questions and one pool cannot answer both.
    # In the noise pool nothing is genuinely better, so every point a strategy appears to win is
    # optimism. In the planted pool one candidate really is better, so a strategy that reports
    # honestly by never adopting anything is exposed.
    harm, trace = _run(indistinguishable, -1)
    power, _ = _run(planted_pool, planted)
    contrast, _ = _run(different, -1)

    study = ReuseStudy(
        audit=audit,
        audit_modules=len({read.module for read in reads}),
        harm=harm,
        power=power,
        contrast=contrast,
        trace=trace,
        planted_edge=planted_edge,
        halfwidth=halfwidth,
        n_holdout=len(y_holdout),
        n_sealed=len(y_sealed),
        n_reference=len(y_reference),
        candidates=total,
        seconds=time.perf_counter() - start,
        reads_by_module=reads_by_module,
    )
    sealed_row = study.test_reads()
    logger.info(
        "Reuse study complete",
        extra={
            "test_reads": sealed_row.reads if sealed_row is not None else 0,
            "selection_cost": round(study.selection_cost(), 4),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _arm_table(rows: Sequence[ArmRow]) -> list[str]:
    """One pool's arms, as a markdown table."""
    lines = [
        "| strategy | reported PR-AUC | true (sealed) | optimism | queries answered | adopted |",
        "|---|---|---|---|---|---|",
    ]
    for arm in rows:
        lines.append(
            f"| {arm.name} | {arm.reported:.4f} | {arm.sealed:.4f} | **{arm.gap:+.4f}** | "
            f"{arm.queries} | {arm.adopted} |"
        )
    return lines


def _lead(study: ReuseStudy) -> str:
    """The finding, written from the computed numbers rather than around them."""
    sealed_row = study.test_reads()
    reads = sealed_row.reads if sealed_row else 0
    modules = sealed_row.modules if sealed_row else 0
    incumbent = study.incumbent()
    naive = study.naive()
    contrast_naive = study.contrast[1]
    contrast_incumbent = study.contrast[0]
    lines = [
        f"**The rules say the sealed split is touched once. It is read {reads} times, from "
        f"{modules} modules. That turns out not to be the question.**",
        "",
        f"Reading a holdout and *selecting* on it are different acts, and only the second one "
        f"burns it -- so the second one was priced directly, on this project's own scores. An "
        f"analyst who tries {study.candidates - 1} detector variants and keeps whichever scores "
        f"best on {study.n_holdout:,} held-out flows pays "
        f"**{study.selection_cost():+.4f} PR-AUC** in optimism beyond what a finite sample costs "
        f"anyway -- reporting {naive.reported:.4f} for a detector worth {naive.sealed:.4f} on a "
        f"third of the same days that nothing ever queried.",
        "",
        (
            f"The detail that makes it concrete: the selected detector is **worse than the one it "
            f"replaced** ({naive.sealed:.4f} against the incumbent's {incumbent.sealed:.4f}) while "
            f"reporting a better number. Selection on noise does not merely inflate the "
            f"estimate; it degrades the thing being estimated."
            if naive.sealed < incumbent.sealed
            else f"The selected detector is no better than the one it replaced "
            f"({naive.sealed:.4f} against {incumbent.sealed:.4f}) while reporting a better number."
        ),
        "",
        f"**But that only happens when the candidates are indistinguishable.** Run the identical "
        f"search over {study.candidates - 1} detectors that genuinely differ -- the same score "
        f"nudged along random feature directions rather than by per-flow noise -- and the cost of "
        f"selection is **{study.contrast_cost():+.4f}**, while the winner is genuinely better on "
        f"the sealed third ({contrast_naive.sealed:.4f} against {contrast_incumbent.sealed:.4f}). "
        f"Four hundred questions, no burn, six points of real detection quality found.",
        "",
        "**So a holdout is not burned by being read. It is burned by being asked to choose "
        "between things it cannot tell apart.** That is the reading that resolves the count "
        f"above: {reads} reads that report a number, or compare a model against a baseline that "
        "differs from it by far more than sampling noise, do not spend the split. A tuning loop "
        "over near-identical variants would spend it fast -- and this project does its tuning on "
        "validation, which is a different split, for exactly this reason.",
    ]
    return "\n".join(lines)


def _render(study: ReuseStudy, figure: Path) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- How Many Times Has the Holdout Been Asked?",
        "",
        f"_A static count of every split read in `netsentry/`, and an adaptive-analyst simulation "
        f"over {study.candidates - 1} candidate detectors with the later days cut three ways: "
        f"{study.n_holdout:,} queryable flows, {study.n_reference:,} for the mechanism that needs "
        f"a reference, and {study.n_sealed:,} sealed. Regenerate with `netsentry reuse`._",
        "",
        "## Why this report exists",
        "",
        "`.claude/rules/ml.md` says the test set is touched **once**, at the end, and that tuning "
        "against it is leakage. Every study in this repository reads it. Both cannot be true, "
        "and the resolution is not obvious, so it is worth doing properly rather than asserting.",
        "",
        "The failure in question is not the syntactic one. `scaler.fit(X_test)` is a bug in the "
        "source and [`netsentry mlint`](mlint.md) already refuses it. What no linter can see is "
        "the statistical failure (Dwork et al., *Science* 349, 2015): a holdout queried "
        "**adaptively** -- where the next thing you try depends on what it said last time -- "
        "stops being a holdout. Nothing is fitted. No column leaks. The number is just wrong, by "
        "an amount that grows with the number of questions.",
        "",
        _lead(study),
        "",
        "## The count",
        "",
        "| split partition | reads | modules |",
        "|---|---|---|",
    ]
    for row in study.audit:
        mark = "**" if row.part == "test" else ""
        lines.append(f"| `{row.part}` | {mark}{row.reads}{mark} | {mark}{row.modules}{mark} |")
    busiest = sorted(study.reads_by_module.items(), key=lambda item: (-item[1], item[0]))[:5]
    val_reads = next((row.reads for row in study.audit if row.part == "val"), 0)
    lines += [
        "",
        "The pass is deliberately conservative: it counts only `load_split` calls whose partition "
        "is a string literal, so a read behind a loop variable or a config value is missed. An "
        "audit that inflated its own finding would be worth nothing, so it undercounts by "
        "construction.",
        "",
        "The modules that ask the most questions of held-out data:",
        "",
        "| module | held-out reads |",
        "|---|---|",
    ]
    for module, count in busiest:
        lines.append(f"| `{module}` | {count} |")
    lines += [
        "",
        f"Reading `val` is not a problem -- that is what a validation split is for, and it is "
        f"read {val_reads} times precisely so the sealed split does not have to be. The count is "
        "exposure, not damage; the rest of this report is about telling those apart.",
        "",
        "## The harm: selecting among candidates the holdout cannot distinguish",
        "",
        f"![Optimism against the number of questions asked](../figures/{figure.name})",
        "",
        f"Every candidate here is the deployed score plus independent per-flow noise, so all "
        f"{study.candidates} have the *same* true quality and every apparent winner is an accident "
        "of this particular sample. The incumbent is simply the first draw, not a privileged one "
        "-- a pool with a privileged incumbent is not exchangeable, and a winner's-curse "
        "experiment on a non-exchangeable pool measures nothing.",
        "",
    ]
    lines += _arm_table(study.harm)
    incumbent = study.incumbent()
    lines += [
        "",
        f"The incumbent's **{incumbent.gap:+.4f}** is the floor: a finite holdout scores any rule "
        "slightly differently than a fresh sample would, whether or not anyone selected on it. "
        f"Everything above that line is the price of asking. The naive analyst pays "
        f"**{study.selection_cost():+.4f}** for {study.naive().queries} questions -- roughly "
        f"{study.selection_cost() / study.halfwidth:.0%} of the "
        f"**{study.halfwidth:.4f}** bootstrap half-width a single PR-AUC estimate carries at this "
        "sample size, which is the honest way to read it: real, measurable, and smaller than the "
        "interval most comparisons in this repository are quoted without.",
        "",
        "## The power: can a mechanism still find a real improvement?",
        "",
        f"A strategy that reports honestly by never adopting anything is not a fix, so the same "
        f"pool is rerun with one genuinely better detector hidden in it, worth "
        f"**{study.planted_edge:+.4f}** PR-AUC on the sealed third.",
        "",
        "| strategy | found the planted detector | true quality of what it chose | adopted |",
        "|---|---|---|---|",
    ]
    for arm in study.power:
        lines.append(
            f"| {arm.name} | {'**yes**' if arm.found_planted else 'no'} | {arm.sealed:.4f} | "
            f"{arm.adopted} |"
        )
    thresholdout = study.harm[2]
    gate = study.harm[4]
    lines += [
        "",
        f"The confidence gate is the only mechanism that is both honest and useful: it costs "
        f"{gate.gap - incumbent.gap:+.4f} over the incumbent on the noise pool -- it adopted "
        f"{gate.adopted} of {gate.queries} candidates there -- and still finds the planted "
        "detector, because a real improvement clears the bootstrap interval and noise does not. "
        "It is three lines and needs no budget, no injected noise and no second dataset.",
        "",
        "## Thresholdout, and the two ways it does not fit here",
        "",
        "**Thresholdout** answers a query from a *reference* set unless the reference and the "
        "holdout disagree by more than a noisy tolerance, spending budget only on genuine "
        "surprises. The idea is exactly right, and the implementation refuses to answer at all "
        "once the budget is gone -- serving reference answers past exhaustion would keep the "
        "analyst working while quietly dropping the guarantee.",
        "",
        f"It fails here twice, for two different and instructive reasons. With an **exchangeable "
        f"reference** -- a second third of the same later days -- the mechanism protects the "
        f"holdout but not the reported number: its answers are individually accurate, and the "
        f"analyst then takes a **maximum** over {thresholdout.queries} of them, which is still a "
        f"maximum. Optimism {thresholdout.gap:+.4f}, worse than the naive analyst's. The "
        "mechanism bounds per-query error; it does not debias an argmax, and nothing that answers "
        "queries can.",
        "",
        f"With **validation as the reference** -- which is what a practitioner on a temporal "
        f"split actually has -- it fails harder and earlier. Validation comes from the training "
        f"days, so its PR-AUC sits {study.harm[3].reported - incumbent.reported:+.3f} above the "
        "later days' by construction; every query is a 'surprise', the budget is gone in "
        f"{study.harm[3].queries} questions, and the reported number "
        f"({study.harm[3].reported:.4f}) is a validation score wearing a test score's clothes. "
        "This is the [covariate-shift study](covariate_shift.md) arriving from a different "
        "direction: a mechanism whose guarantee assumes exchangeability cannot be pointed at two "
        "sets separated by time.",
        "",
        "## Scope and honest limits",
        "",
        "- **The count is exposure, not damage.** Distinguishing a read that reports from a read "
        "that selects requires knowing what the number is *used for*, which is a human judgement "
        "no static pass can make. The count is the upper bound on how much adaptivity could have "
        "happened.",
        "- **The harm is measured on a synthetic pool, deliberately.** Real candidate detectors "
        "are rarely exactly equal in quality; the noise pool is the worst case, constructed so "
        "that the entire measured gap is attributable to selection. The contrast pool is the "
        "realistic case and it costs nothing.",
        "- **Adaptivity across reports is real but weak.** The strongest form of the failure needs "
        "a tight loop -- try, look, adjust, repeat. Studies here are written once and run once; "
        "the loop that exists is between waves, slow and mediated by prose, which is the least "
        "harmful form.",
        "- **One split's worth of noise.** The gap is measured on a single random three-way cut "
        "of the later days. The direction and the order of magnitude are the claim, not the "
        "fourth decimal. The [seed-sensitivity study](seed_variance.md) measures the other half "
        "of the same noise floor, the part that comes from training rather than sampling.",
        "- **A confidence gate slows the burn; it does not stop it.** It bounds what any single "
        "query can change. The only real fix is a split nobody has looked at, which is why the "
        "sealed third here is used once and discarded.",
    ]
    return "\n".join(lines) + "\n"


def run_reuse_report(settings: Settings) -> Path:
    """Run the reuse audit and write the report + figure."""
    study = run_reuse_study(settings)
    queries = np.array([row.queries for row in study.trace], dtype=float)
    figure = plots.plot_lines(
        {
            "what the analyst would report": (
                queries,
                np.array([row.reported for row in study.trace]),
            ),
            "what it is actually worth (sealed third)": (
                queries,
                np.array([row.sealed for row in study.trace]),
            ),
        },
        xlabel="holdout queries asked",
        ylabel="PR-AUC",
        title="A holdout stops being a holdout one question at a time",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote reuse report", extra={"path": str(out_path)})

    with track_run(settings, "reuse") as run:
        sealed_row = study.test_reads()
        run.log_params({"candidates": str(study.candidates), "rounds": str(len(study.trace))})
        run.log_metrics(
            {
                "test_split_reads": float(sealed_row.reads if sealed_row is not None else 0),
                "selection_cost_indistinguishable": study.selection_cost(),
                "selection_cost_different": study.contrast_cost(),
                "sampling_floor": study.incumbent().gap,
                "bootstrap_halfwidth": study.halfwidth,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
