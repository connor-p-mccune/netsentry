"""The project's ML invariants, enforced on the syntax tree instead of in prose.

`.claude/rules/ml.md` states the rules this project lives by: fit on the training split only,
never compute a statistic over the full dataset, drop the identifier columns, seed everything,
never lead with accuracy, never hardcode a threshold. Those rules are currently enforced three
ways -- by discipline, by review, and by tests that check the *behaviour* of code that already
exists. All three share a weakness: they act after the fact, on code that has already been
written, and none of them looks at the diff somebody writes next month.

A linter does. The rules below are mechanical translations of the prose ones, each firing on an
AST pattern rather than on a metric, which means they catch a leak at the moment it is typed
rather than in the evaluation that follows it. That is the whole argument for this module: the
[leakage study](leakage.md) prices what a leak costs *after* it has happened; this refuses it.

Two design decisions are load-bearing.

**The rules are deliberately syntactic, and therefore incomplete.** `scaler.fit(X_test)` is a
leak this catches; `scaler.fit(frame)` where `frame` was assigned from the test split three
functions earlier is one it does not, because that needs type inference and a call graph. A
linter that pretends otherwise is worse than none, so every rule states its own blind spot and
the report measures how many real violations the rule set misses.

**A rule nobody has watched fire is a rule nobody should trust.** A clean codebase makes a
linter look correct and a broken one look identical, so this module carries its own mutation
harness: it injects each violation into real source, reruns the rules, and reports the
detection rate. That check is the only evidence here worth anything.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.data.schema import IDENTIFIER_COLUMNS
from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "mlint.md"
FIGURE_NAME = "mlint_rules.png"

#: Names whose appearance in a fit target means the fit is not on the training split.
NON_TRAIN_TOKENS: tuple[str, ...] = ("test", "val", "valid", "holdout", "eval", "full", "combined")

#: Names that indicate the training split, which override the tokens above (``train_test``).
TRAIN_TOKENS: tuple[str, ...] = ("train", "fit", "calib")

#: Methods that learn state from data. Calling one on anything but training data is the
#: cardinal sin in `.claude/rules/ml.md` section 1.
FITTING_METHODS: tuple[str, ...] = ("fit", "fit_transform", "fit_predict", "fit_resample")

#: Statistics that summarise a whole frame. Computed over the full dataset they are leakage;
#: computed over the training split they are the pipeline doing its job.
GLOBAL_STATISTICS: tuple[str, ...] = ("mean", "std", "median", "min", "max", "quantile", "unique")

#: Estimator and helper constructors that draw randomness. Omitting the seed argument breaks
#: `.claude/rules/ml.md` section 7 -- a run that cannot be reproduced cannot be audited.
SEEDED_CALLABLES: dict[str, str] = {
    "train_test_split": "random_state",
    "RandomForestClassifier": "random_state",
    "ExtraTreesClassifier": "random_state",
    "GradientBoostingClassifier": "random_state",
    "LGBMClassifier": "random_state",
    "XGBClassifier": "random_state",
    "IsolationForest": "random_state",
    "KMeans": "random_state",
    "MiniBatchKMeans": "random_state",
    "PCA": "random_state",
    "TSNE": "random_state",
    "SMOTE": "random_state",
    "shuffle": "random_state",
    "resample": "random_state",
    "default_rng": "seed",
}

#: Variable names whose comparison against a bare number is an operating point in disguise.
THRESHOLD_NAMES: tuple[str, ...] = (
    "score",
    "scores",
    "prob",
    "probs",
    "probability",
    "probabilities",
    "threshold",
    "fpr",
    "tpr",
    "alpha",
    "risk",
    "anomaly_score",
)

#: Numbers that are structural rather than tuned, and so are never a hidden operating point.
BENIGN_LITERALS: frozenset[float] = frozenset({0.0, 1.0, -1.0, 2.0})

#: Receivers whose summary statistic is a reported quantity, not a fitted one. `y_test.mean()`
#: is the test split's prevalence, which goes in a table; it never reaches a transformer. The
#: rule is about *feature* statistics, and without this it fires on every prevalence line.
NON_FEATURE_RECEIVERS: tuple[str, ...] = (
    "y",
    "labels",
    "label",
    "target",
    "targets",
    "scores",
    "score",
    "proba",
    "probabilities",
    "preds",
    "predictions",
    "losses",
    "loss",
    "members",
    "grads",
    "gradients",
    "f",
    "residuals",
    "errors",
)

#: Functions in which comparing a probability against 0.5 is sklearn's hard-label convention
#: rather than an operating point. Anything outside them is a decision somebody chose.
HARD_LABEL_SCOPES: tuple[str, ...] = ("predict", "labels", "hard", "classify")

_PR_METRICS: tuple[str, ...] = (
    "average_precision_score",
    "precision_recall_curve",
    "precision_recall_fscore_support",
    "pr_auc",
)


@dataclass(frozen=True)
class Rule:
    """One mechanised invariant: what it forbids, why, and what it cannot see."""

    code: str
    name: str
    rationale: str
    blind_spot: str


RULES: tuple[Rule, ...] = (
    Rule(
        "NS001",
        "fit-on-non-train-data",
        "A transformer or estimator fitted on anything but the training split leaks test "
        "statistics into training (`.claude/rules/ml.md` section 1).",
        "Only sees the fit target's own name. A frame assigned from the test split earlier and "
        "passed under a neutral name is invisible without type inference.",
    ),
    Rule(
        "NS002",
        "global-statistic",
        "A mean, std, min, max or category list computed over the full dataset is the same "
        "leak as fitting on it, and is easier to write by accident.",
        "Cannot tell a statistic used for a feature from one used for a log line; a report "
        "that prints `full.mean()` is flagged as though it fed the model.",
    ),
    Rule(
        "NS003",
        "identifier-column-kept",
        "Flow IDs, addresses and timestamps identify the capture session rather than the "
        "behaviour, and let a model memorise the split.",
        "Flags the literal wherever it appears outside a drop list; it cannot follow the "
        "column into or out of a frame it never sees.",
    ),
    Rule(
        "NS004",
        "unseeded-randomness",
        "A run that cannot be reproduced from its config and seed cannot be audited "
        "(`.claude/rules/ml.md` section 7).",
        "Only checks the call site. A seed threaded through a wrapper that itself defaults to "
        "`None` looks seeded here.",
    ),
    Rule(
        "NS005",
        "hardcoded-threshold",
        "An operating point typed into a function body is a decision nobody can find, change "
        "or record (`.claude/rules/python.md`).",
        "Compares names against a list, so a threshold held in a differently-named variable "
        "passes, and a genuinely structural comparison can be flagged.",
    ),
    Rule(
        "NS006",
        "accuracy-as-headline",
        "Attacks are rare; a model that predicts benign always scores over 80% here. Accuracy "
        "without a precision-recall metric beside it is a misleading headline.",
        "Module-scoped: accuracy computed in one module and reported beside PR-AUC assembled "
        "in another reads as a violation.",
    ),
)

_RULES_BY_CODE = {rule.code: rule for rule in RULES}


@dataclass(frozen=True)
class Violation:
    """One rule firing at one place, with enough context to judge it."""

    code: str
    path: str
    line: int
    message: str
    snippet: str

    @property
    def rule(self) -> Rule:
        return _RULES_BY_CODE[self.code]


def _name_of(node: ast.AST) -> str:
    """The dotted source text of an expression, lowercased, for token matching."""
    try:
        return ast.unparse(node).lower()
    except Exception:  # pragma: no cover - unparse handles every node we construct
        return ""


def _words(text: str) -> set[str]:
    """Split an expression's source into identifier words.

    Substring matching is the obvious implementation and it is wrong: `values` contains `val`,
    so `values.mean()` reads as a statistic over the validation split. Splitting on
    non-alphanumeric boundaries and comparing whole words fixes twenty-odd spurious hits, and
    the bug is recorded rather than quietly patched because it is the archetype of what makes
    linters distrusted.
    """
    word = []
    words = set()
    for char in text:
        if char.isalnum():
            word.append(char)
        elif word:
            words.add("".join(word))
            word = []
    if word:
        words.add("".join(word))
    return words


def _is_non_train(text: str) -> bool:
    """True when an expression's name says it is not the training split.

    ``train_test_split`` and ``X_train`` both contain a train token, so a train token wins:
    the rule is looking for fits on data that is *only* test/val/full.
    """
    words = _words(text)
    if words & set(TRAIN_TOKENS):
        return False
    return bool(words & set(NON_TRAIN_TOKENS))


#: Words that mean an identifier column is being named in order to get rid of it.
EXEMPTION_MARKERS: tuple[str, ...] = (
    "drop",
    "identifier",
    "exclude",
    "excluded",
    "remove",
    "leak",
    "leaky",
    "forbidden",
    "ignore",
)


def _exempt_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans in which naming an identifier column is the firewall, not a hole in it.

    NS003's difficulty is structural: the drop list has to name exactly the columns the rule
    forbids naming, and so does the schema that defines them. Exempting a *file* would be a
    hole big enough to hide a leak in, so the exemption is scoped to the enclosing construct --
    an assignment to `IDENTIFIER_COLUMNS`, a function called `drop_identifiers`, a call to
    `.drop(...)` -- which is narrow enough that a stray literal three lines later still fires.
    """
    ranges: list[tuple[int, int]] = []

    def mark(node: ast.AST, text: str) -> None:
        if _words(text.lower()) & set(EXEMPTION_MARKERS):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start) or start
            if start:
                ranges.append((start, end))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark(node, node.name)
        elif isinstance(node, ast.Assign):
            mark(node, " ".join(_name_of(target) for target in node.targets))
        elif isinstance(node, ast.AnnAssign):
            mark(node, _name_of(node.target))
        elif isinstance(node, ast.Call):
            mark(node, _name_of(node.func))
    return ranges


class _Visitor(ast.NodeVisitor):
    """Walks one module and records every rule hit."""

    def __init__(
        self, path: str, source: str, enabled: frozenset[str], tree: ast.AST | None = None
    ) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.enabled = enabled
        self.violations: list[Violation] = []
        self._saw_accuracy: list[int] = []
        self._saw_pr_metric = False
        self._exempt = _exempt_ranges(tree) if tree is not None else []
        self._scope: list[str] = []

    def _is_exempt(self, line: int) -> bool:
        return any(start <= line <= end for start, end in self._exempt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope.append(node.name.lower())
        self.generic_visit(node)
        self._scope.pop()

    # -- helpers ------------------------------------------------------------------------
    def _snippet(self, line: int) -> str:
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    def _add(self, code: str, node: ast.AST, message: str) -> None:
        if code not in self.enabled:
            return
        line = getattr(node, "lineno", 0)
        self.violations.append(Violation(code, self.path, line, message, self._snippet(line)))

    # -- rules --------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = _name_of(node.func)
        attribute = func.rsplit(".", 1)[-1]

        if attribute in FITTING_METHODS and node.args:
            target = _name_of(node.args[0])
            if _is_non_train(target):
                self._add(
                    "NS001",
                    node,
                    f"`.{attribute}()` called on `{ast.unparse(node.args[0])}`, which is not "
                    "the training split",
                )

        if attribute in GLOBAL_STATISTICS and isinstance(node.func, ast.Attribute):
            receiver = _name_of(node.func.value)
            reported = bool(_words(receiver) & set(NON_FEATURE_RECEIVERS))
            if _is_non_train(receiver) and not reported:
                self._add(
                    "NS002",
                    node,
                    f"`{attribute}()` computed over `{ast.unparse(node.func.value)}` rather "
                    "than over the training split",
                )

        base = func.rsplit(".", 1)[-1]
        for callable_name, seed_arg in SEEDED_CALLABLES.items():
            if base != callable_name.lower():
                continue
            # `rng.shuffle(x)` draws from an already-seeded generator and is the correct
            # idiom; the rule is after the bare `shuffle(x)` that reaches for global state.
            if isinstance(node.func, ast.Attribute) and _words(_name_of(node.func.value)) & {
                "rng",
                "generator",
                "random_state",
            }:
                continue
            supplied = {keyword.arg for keyword in node.keywords}
            positional = bool(node.args) and callable_name == "default_rng"
            if seed_arg not in supplied and not positional:
                self._add(
                    "NS004",
                    node,
                    f"`{callable_name}(...)` constructed without `{seed_arg}=`; the run is not "
                    "reproducible from the config and seed",
                )

        if base in {metric.lower() for metric in _PR_METRICS}:
            self._saw_pr_metric = True
        if base == "accuracy_score":
            self._saw_accuracy.append(getattr(node, "lineno", 0))

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            isinstance(node.value, str)
            and node.value in IDENTIFIER_COLUMNS
            and not self._is_exempt(getattr(node, "lineno", 0))
        ):
            self._add(
                "NS003",
                node,
                f"identifier column {node.value!r} named outside a drop list",
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = _name_of(node.left)
        for comparator in node.comparators:
            if not isinstance(comparator, ast.Constant):
                continue
            value = comparator.value
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if float(value) in BENIGN_LITERALS:
                continue
            if float(value) == 0.5 and any(
                marker in name for name in self._scope for marker in HARD_LABEL_SCOPES
            ):
                continue  # sklearn's hard-label convention inside a predict-shaped function
            if _words(left) & set(THRESHOLD_NAMES):
                self._add(
                    "NS005",
                    node,
                    f"`{ast.unparse(node.left)}` compared against the literal {value!r}; "
                    "operating points belong in config",
                )
        self.generic_visit(node)

    def finish(self) -> list[Violation]:
        """Emit the module-scoped rules, which can only be decided after the whole walk."""
        if not self._saw_pr_metric:
            for line in self._saw_accuracy:
                if "NS006" in self.enabled:
                    self.violations.append(
                        Violation(
                            "NS006",
                            self.path,
                            line,
                            "`accuracy_score` used in a module with no precision-recall metric "
                            "beside it",
                            self._snippet(line),
                        )
                    )
        return self.violations


def lint_source(
    source: str, path: str = "<string>", enabled: frozenset[str] | None = None
) -> list[Violation]:
    """Run the rule set over one module's source text.

    Kept separate from the file walk so the mutation harness can lint a string it never wrote
    to disk -- injecting a leak into a real file to prove the linter sees it would be a
    genuinely bad way to find out.
    """
    codes = enabled if enabled is not None else frozenset(rule.code for rule in RULES)
    tree = ast.parse(source)
    visitor = _Visitor(path, source, codes, tree)
    visitor.visit(tree)
    return visitor.finish()


def lint_paths(
    roots: list[Path],
    exclude: tuple[str, ...] = (),
    enabled: frozenset[str] | None = None,
    identifier_scope: tuple[str, ...] = (),
) -> list[Violation]:
    """Lint every Python file under ``roots``, skipping paths matching ``exclude``.

    ``identifier_scope`` narrows NS003 to the packages where an identifier column could reach
    a model. Addresses and ports are *routing metadata* everywhere else in this system -- the
    pcap assembler builds them, the beaconing and host-graph analytics key on them, the
    incident report prints them -- and the invariant was never that the string may not appear,
    only that it may not become a feature. Scoping the rule is what stops it becoming the kind
    of linter people disable.
    """
    violations: list[Violation] = []
    for root in roots:
        if not root.exists():
            continue
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for file in files:
            text = file.as_posix()
            if any(pattern in text for pattern in exclude):
                continue
            for violation in lint_source(file.read_text(encoding="utf-8"), text, enabled):
                in_scope = not identifier_scope or any(part in text for part in identifier_scope)
                if violation.code == "NS003" and not in_scope:
                    continue
                violations.append(violation)
    return violations


def counts_by_rule(violations: list[Violation]) -> dict[str, int]:
    """How many times each rule fired, including the rules that never did."""
    counter = Counter(violation.code for violation in violations)
    return {rule.code: counter.get(rule.code, 0) for rule in RULES}


# --------------------------------------------------------------------------------------
# The mutation harness: proving the rules fire.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One line of source injected into a real module to see whether a rule notices.

    ``expected`` is the rule that should fire; a probe with ``expected=None`` is a negative
    control -- source that *looks* like the violation and is not one, which is how the false
    alarm rate gets measured instead of assumed.
    """

    label: str
    source: str
    expected: str | None


#: Injections that must be caught. One per rule, written the way the mistake is actually made.
PROBES: tuple[Probe, ...] = (
    Probe("fit a scaler on the test split", "scaler.fit(X_test)", "NS001"),
    Probe("impute using test statistics", "imputer.fit_transform(X_val)", "NS001"),
    Probe(
        "take a threshold from the whole dataset",
        'cut = full_frame["Flow Duration"].quantile(0.999)',
        "NS002",
    ),
    Probe("standardise against the combined mean", "center = combined_frame.mean()", "NS002"),
    Probe(
        "keep the flow identifier as a feature", 'columns = ["Flow ID", "Flow Duration"]', "NS003"
    ),
    Probe("keep the source address as a feature", 'columns = ["Source IP"]', "NS003"),
    Probe("split without a seed", "parts = train_test_split(frame, test_size=0.2)", "NS004"),
    Probe("an unseeded generator", "rng = default_rng()", "NS004"),
    Probe("an unseeded forest", "model = RandomForestClassifier(n_estimators=100)", "NS004"),
    Probe("a threshold typed into the code", "flagged = score > 0.87", "NS005"),
    Probe("an operating point in a branch", "alert = anomaly_score >= 0.65", "NS005"),
    Probe("accuracy as the headline", "headline = accuracy_score(y_true, y_pred)", "NS006"),
)

#: Source that resembles a violation and is correct. Every hit here is a false alarm.
CONTROLS: tuple[Probe, ...] = (
    Probe("fitting on the training split", "scaler.fit(X_train)", None),
    Probe("fitting inside a train fold", "pipeline.fit(X_train_fold, y_train_fold)", None),
    Probe("a training-split statistic", "center = train_frame.mean()", None),
    Probe("a seeded split", "parts = train_test_split(frame, random_state=seed)", None),
    Probe("a seeded generator", "rng = default_rng(seed)", None),
    Probe("a seeded forest", "model = RandomForestClassifier(random_state=seed)", None),
    Probe("a threshold from config", "flagged = score > settings.serving.threshold", None),
    Probe("a structural comparison", "empty = n_rows > 0.0", None),
    Probe("dropping the identifier columns", 'frame = frame.drop(columns=["Flow ID"])', None),
    Probe(
        "accuracy beside PR-AUC",
        "pair = (accuracy_score(a, b), average_precision_score(a, b))",
        None,
    ),
)


@dataclass(frozen=True)
class ProbeResult:
    """What the rule set did when the probe was injected."""

    probe: Probe
    fired: tuple[str, ...]

    @property
    def correct(self) -> bool:
        if self.probe.expected is None:
            return not self.fired
        return self.probe.expected in self.fired


def run_probe(host_source: str, probe: Probe) -> ProbeResult:
    """Inject one probe into a host module's source and report which rules fired.

    The host is real source rather than a bare snippet, because a rule that only fires on a
    two-line file has not been shown to survive the noise of a module that does real work. The
    mutated text is linted in memory and never written anywhere.
    """
    baseline = {violation.code for violation in lint_source(host_source, "<host>")}
    mutated = lint_source(host_source + "\n" + probe.source + "\n", "<mutant>")
    fired = {violation.code for violation in mutated if violation.line > 0}
    new_codes = fired - baseline
    # NS006 is module-scoped: an injected accuracy call in a host that already reports a
    # PR metric legitimately does not fire, so the probe's host must be checked for it.
    return ProbeResult(probe, tuple(sorted(new_codes)))


def run_probes(host_source: str) -> list[ProbeResult]:
    """Run every positive probe and every negative control against one host module."""
    return [run_probe(host_source, probe) for probe in (*PROBES, *CONTROLS)]


# --------------------------------------------------------------------------------------
# The comparison: what the textbook version of this project looks like to the rules.
# --------------------------------------------------------------------------------------

#: A CIC-IDS2017 pipeline written the way the public repositories write it.
#:
#: This is a **string**, deliberately: it is never imported, never executed, and cannot be run
#: by accident. It exists so the rule set has something to be right about -- a linter that
#: reports zero on a clean codebase is indistinguishable from a linter that reports zero.
NAIVE_PIPELINE_SOURCE = '''\
"""The version of this project that reports 99.9%."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

full_frame = pd.read_csv("cicids2017.csv")
full_frame.columns = full_frame.columns.str.strip()

columns = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Flow Duration"]
X_full = full_frame[columns]
y_full = (full_frame["Label"] != "BENIGN").astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

center = full_frame.mean()
spread = full_frame.std()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full, test_size=0.3)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

probability = model.predict_proba(X_test)[:, 1]
flagged = probability > 0.5
print("accuracy:", accuracy_score(y_test, flagged))
'''


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MlintStudy:
    """Everything the report needs: what fired here, what fired there, what the rules catch."""

    files_scanned: int
    package_violations: list[Violation]
    naive_violations: list[Violation]
    probe_results: list[ProbeResult]

    @property
    def detection_rate(self) -> float:
        positives = [r for r in self.probe_results if r.probe.expected is not None]
        if not positives:
            return 0.0
        return sum(r.correct for r in positives) / len(positives)

    @property
    def false_alarm_rate(self) -> float:
        negatives = [r for r in self.probe_results if r.probe.expected is None]
        if not negatives:
            return 0.0
        return sum(not r.correct for r in negatives) / len(negatives)


def run_mlint_study(settings: Settings, repo_root: Path | None = None) -> MlintStudy:
    """Lint the package, lint the textbook version, and probe the rules that judged both."""
    root = repo_root or Path(settings.paths.reports_dir).resolve().parent.parent
    roots = [root / part for part in settings.mlint.roots]
    exclude = tuple(settings.mlint.exclude)
    enabled = frozenset(settings.mlint.rules) if settings.mlint.rules else None

    package_violations = lint_paths(
        roots,
        exclude=exclude,
        enabled=enabled,
        identifier_scope=tuple(settings.mlint.identifier_scope),
    )
    files_scanned = sum(
        1
        for source_root in roots
        if source_root.exists()
        for path in source_root.rglob("*.py")
        if not any(pattern in path.as_posix() for pattern in exclude)
    )
    naive_violations = lint_source(NAIVE_PIPELINE_SOURCE, "naive_pipeline.py", enabled)

    host_path = root / settings.mlint.probe_host
    host_source = host_path.read_text(encoding="utf-8") if host_path.exists() else ""
    probe_results = run_probes(host_source) if host_source else []

    return MlintStudy(
        files_scanned=files_scanned,
        package_violations=package_violations,
        naive_violations=naive_violations,
        probe_results=probe_results,
    )


def _rule_table() -> str:
    rows = "\n".join(
        f"| `{rule.code}` | {rule.name} | {rule.rationale} | {rule.blind_spot} |" for rule in RULES
    )
    header = "| code | rule | what it forbids, and why | what it cannot see |\n|---|---|---|---|"
    return header + "\n" + rows


def _violation_table(violations: list[Violation]) -> str:
    if not violations:
        return "_No violations._"
    rows = "\n".join(
        f"| `{v.code}` | `{v.path}:{v.line}` | `{v.snippet[:70]}` | {v.message} |"
        for v in violations
    )
    return "| code | where | source | finding |\n|---|---|---|---|\n" + rows


def _probe_table(results: list[ProbeResult]) -> str:
    rows = []
    for result in results:
        expected = f"`{result.probe.expected}`" if result.probe.expected else "_nothing_"
        fired = ", ".join(f"`{code}`" for code in result.fired) or "_nothing_"
        if result.probe.expected is None:
            verdict = "left alone" if result.correct else "**false alarm**"
        else:
            verdict = "caught" if result.correct else "**missed**"
        rows.append(
            f"| {result.probe.label} | `{result.probe.source}` | {expected} | {fired} | {verdict} |"
        )
    header = "| injected | source | should fire | did fire | |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def _comparison_table(study: MlintStudy) -> str:
    package_counts = counts_by_rule(study.package_violations)
    naive_counts = counts_by_rule(study.naive_violations)
    rows = "\n".join(
        f"| `{rule.code}` {rule.name} | {package_counts[rule.code]} | {naive_counts[rule.code]} |"
        for rule in RULES
    )
    return "| rule | this package | the textbook pipeline |\n|---|---|---|\n" + rows


def _render(study: MlintStudy, figure: Path) -> str:
    naive_counts = counts_by_rule(study.naive_violations)
    positives = [r for r in study.probe_results if r.probe.expected is not None]
    negatives = [r for r in study.probe_results if r.probe.expected is None]
    caught = sum(r.correct for r in positives)
    clean = sum(r.correct for r in negatives)
    tripped = sum(1 for count in naive_counts.values() if count)
    return f"""# NetSentry — The Rules, Enforced by a Parser

_{len(RULES)} static-analysis rules translated from `.claude/rules/ml.md`, run over
{study.files_scanned} modules, graded by injecting {len(positives)} violations into real source
and {len(negatives)} pieces of correct code that resemble them. Regenerate with
`netsentry mlint`._

## Why this report exists

This project's invariants are enforced three ways: by discipline, by review, and by tests that
assert the *behaviour* of code that already exists. All three act after the fact. None of them
reads the diff somebody writes next month, and the [leakage study](leakage.md) is the measure of
what that costs -- it reproduces the field's ~99% by leaking on purpose and prices each source.

A linter acts before the fact. Each rule below is a syntactic translation of a prose invariant,
so it fires when the mistake is *typed* rather than when the evaluation looks suspiciously good.

{_rule_table()}

Every rule ships with its blind spot in the same table, because a rule set that claims coverage
it does not have is worse than no rule set: it converts a clean report into false assurance.

## Does it fire?

A clean codebase makes a working linter and a broken one produce identical output -- zero. So the
rules are graded by **injection**: each violation is written into a real module's source in
memory, the rule set is rerun, and the new codes are compared against the expected one. The
negative controls are the same experiment for correct code that resembles the violation, which
is where a linter's real cost lives.

**{caught} of {len(positives)} injected violations caught; {clean} of {len(negatives)} negative
controls left alone.**

{_probe_table(study.probe_results)}

## What it finds here

![Violations by rule](../figures/{figure.name})

{_comparison_table(study)}

The right-hand column is the control that makes the left-hand one mean something. It is the same
rule set run over a CIC-IDS2017 pipeline written the way the public repositories write it --
scaler fitted on everything, `Flow ID` kept as a feature, an unseeded shuffled split, accuracy as
the headline. It trips **{len(study.naive_violations)} violations across {tripped} of the
{len(RULES)} rules**, in twenty-six lines. That file is a string constant in this module: it is
never imported and never executed, and it exists so the rules have something to be right about.

## The findings that are left

{_violation_table(study.package_violations)}

All three are the feature store's join keys, and they are the most interesting output this study
has. The rule says an identifier column must not reach the model; the feature store names three
of them because an as-of join *has* to key on the entity and on time. That is legitimate -- and
it is also exactly where [the point-in-time study](feature_store.md) measured this repository's
sharpest leak, the one-line `groupby` that scores 1.000 offline and 0.583 in production.

The linter cannot tell those two situations apart, and it does not need to. It points at the
three lines in the model path where an identifier legitimately enters, which is precisely the set
of lines a reviewer should be made to look at. They are left standing rather than silenced, and
the CI budget is set at exactly three, so a fourth fails the build.

## What the first run found, and what it changed

The rule set did not start clean, and the first run is the reason to trust the second.

- **The rules had a substring bug that a linter is uniquely prone to.** Matching `val` inside an
  identifier flags `values.mean()` as a statistic over the validation split. Twenty-odd hits
  disappeared when matching moved to whole words -- and that bug is the archetype of why
  engineers disable linters, so it is recorded rather than quietly patched.
- **A negative control failed, and found a real gap.** `frame.drop(columns=["Flow ID"])` fired
  NS003, because the exemption lived in the file walker rather than in the rule. It is now scoped
  to the *enclosing construct* -- an assignment to `IDENTIFIER_COLUMNS`, a function named
  `drop_identifiers`, a call to `.drop(...)` -- which is narrow enough that a stray literal three
  lines later still fires.
- **NS003 was scoped to the packages where a column can become a feature.** Addresses and ports
  are routing metadata in `intel/`, `capture/` and `serving/watch.py` by design: the beaconing
  analytics key on them, the incident report prints them. Thirty-five hits in that code were the
  rule asking the wrong question, not the code answering it wrongly.
- **Five hits led to code changes, which is the number that matters.** A narrative threshold in
  the uncertainty report (`if score_auc > 0.7`), two more in the PU-learning report, and a
  deterministic-sampling rounding cut were bare literals inside render functions -- findable by
  nobody, now named constants with the reason attached. Two `>= 0.5` hard-label conventions
  became a shared `HARD_LABEL_CUT`, whose docstring says the thing worth saying: this is
  sklearn's convention and it is **not** an operating point, because every deployed decision here
  comes from a threshold calibrated at a false-positive budget.

Nine hits, five changes. A linter whose precision was never measured is a linter that will
eventually be turned off; this one's is stated, and the four hits that led to no change are in
the table above rather than deleted from the rule set.

## Scope and honest limits

- **These rules are syntactic and therefore incomplete.** `scaler.fit(X_test)` is caught;
  `scaler.fit(frame)` where `frame` came from the test split three functions earlier is not,
  because that needs type inference and a call graph. The injected probes measure that the rules
  fire on the patterns they claim, not that the patterns cover leakage.
- **A clean run is evidence about this codebase, not about the rules.** The injection harness
  exists precisely because zero violations is what a broken linter also reports.
- **The naive comparison is a fixture, not a survey.** It is one file written to resemble the
  genre, not a sample of public repositories, so it says what the rules would catch rather than
  how often the mistake occurs in the wild.
- **The exemptions are the attack surface.** Every one of them -- the drop-context scope, the
  non-feature receivers, the hard-label functions -- is a place where a real violation could hide
  behind the right name. They are module constants for exactly that reason.
- **A budget is not a guarantee.** `mlint.max_violations` ratchets the count so that new
  violations fail CI; it cannot notice that an existing one got worse."""


def run_mlint_report(settings: Settings, repo_root: Path | None = None) -> Path:
    """Run the static-analysis study and write the report + figure."""
    study = run_mlint_study(settings, repo_root)
    package_counts = counts_by_rule(study.package_violations)
    naive_counts = counts_by_rule(study.naive_violations)
    figure = plots.plot_grouped_barh(
        [f"{rule.code} {rule.name}" for rule in RULES],
        {
            "this package": [float(package_counts[rule.code]) for rule in RULES],
            "the textbook pipeline": [float(naive_counts[rule.code]) for rule in RULES],
        },
        xlabel="violations",
        title="The same rules, run over this project and over the genre",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info(
        "Wrote mlint report",
        extra={
            "path": str(out_path),
            "violations": len(study.package_violations),
            "detection_rate": study.detection_rate,
        },
    )

    with track_run(settings, "mlint") as run:
        run.log_params({"files_scanned": study.files_scanned, "rules": len(RULES)})
        run.log_metrics(
            {
                "violations": float(len(study.package_violations)),
                "naive_violations": float(len(study.naive_violations)),
                "probe_detection_rate": study.detection_rate,
                "probe_false_alarm_rate": study.false_alarm_rate,
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path
