"""The static-analysis rules, and the evidence that they fire.

A linter's tests are unusual: the thing that can go wrong is not that it crashes, it is that it
silently reports nothing forever. So the load-bearing tests here are the ones that assert each
rule fires on the violation it claims to catch **and** stays quiet on the correct code that
resembles it -- a rule that only does the first half is a rule that will be disabled inside a
month.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netsentry.config import Settings
from netsentry.governance.mlint import (
    CONTROLS,
    NAIVE_PIPELINE_SOURCE,
    PROBES,
    RULES,
    counts_by_rule,
    lint_paths,
    lint_source,
    run_mlint_report,
    run_mlint_study,
    run_probe,
    run_probes,
)


def _codes(source: str) -> set[str]:
    return {violation.code for violation in lint_source(source)}


# --------------------------------------------------------------------------------------
# The rule set as a whole.
# --------------------------------------------------------------------------------------


def test_every_rule_states_its_rationale_and_its_blind_spot() -> None:
    # A rule that claims coverage it does not have turns a clean report into false assurance.
    for rule in RULES:
        assert rule.code.startswith("NS") and rule.name
        assert len(rule.rationale) > 60, rule.code
        assert len(rule.blind_spot) > 40, rule.code


def test_rule_codes_are_unique() -> None:
    codes = [rule.code for rule in RULES]
    assert len(codes) == len(set(codes))


# --------------------------------------------------------------------------------------
# NS001 -- fitting on data that is not the training split.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "scaler.fit(X_test)",
        "scaler.fit_transform(X_val)",
        "encoder.fit(full_frame)",
        "sampler.fit_resample(combined, y)",
    ],
)
def test_fitting_on_non_train_data_is_a_violation(source: str) -> None:
    assert "NS001" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        "scaler.fit(X_train)",
        "pipeline.fit(X_train_fold, y_train_fold)",
        "model.fit(X_train, y_train, eval_set=[(X_val, y_val)])",
        "calibrator.fit(X_calib, y_calib)",
    ],
)
def test_fitting_on_the_training_split_is_not(source: str) -> None:
    assert "NS001" not in _codes(source)


def test_a_train_token_wins_over_a_test_token() -> None:
    # `train_test_split` and `X_train_test` both contain "test"; the rule is looking for data
    # that is *only* test, so a train token has to dominate or the rule fires on everything.
    assert "NS001" not in _codes("model.fit(X_train_and_test)")


# --------------------------------------------------------------------------------------
# NS002 -- statistics over data the model should not have seen.
# --------------------------------------------------------------------------------------


def test_a_statistic_over_the_full_dataset_is_a_violation() -> None:
    assert "NS002" in _codes('cut = full_frame["Flow Duration"].quantile(0.999)')


def test_a_statistic_over_the_training_split_is_not() -> None:
    assert "NS002" not in _codes("center = train_frame.mean()")


def test_the_word_values_does_not_read_as_the_validation_split() -> None:
    """The substring bug this rule set shipped with, kept as a regression test.

    Matching `val` inside an identifier flags `values.mean()` as a validation-split statistic
    and produced twenty-odd spurious hits. It is the archetype of what makes linters distrusted.
    """
    assert "NS002" not in _codes("pr_auc = float(values[:, 0].mean())")


def test_a_statistic_over_the_test_split_is_a_violation() -> None:
    """The hit that found a real leak in this repository's own bandit study.

    Standardising a stream by its own mean gives an online learner a statistic of flows it has
    not seen yet, which is not a context at all.
    """
    assert "NS002" in _codes("centred = (s_test - s_test.mean()) / s_test.std()")


def test_a_statistic_over_the_validation_split_is_not() -> None:
    """The asymmetry between NS001 and NS002, and it comes straight from the project's rules.

    A transformer must be *fitted* on training data only, so NS001 counts validation as
    off-limits. Choosing a threshold on validation is the prescribed method, so NS002 does not.
    Getting this wrong in either direction breaks a rule people would otherwise keep.
    """
    assert "NS002" not in _codes("centre = s_val.mean()")
    assert "NS001" in _codes("scaler.fit(X_val)")


def test_a_reported_label_prevalence_is_not_a_leak() -> None:
    # `y_test.mean()` is the test split's prevalence, which goes in a table. It never reaches a
    # transformer, and a rule that cannot tell the difference fires on every report.
    assert "NS002" not in _codes("prevalence = float(y_test.mean())")


# --------------------------------------------------------------------------------------
# NS003 -- identifier columns, and the exemption that makes the rule usable.
# --------------------------------------------------------------------------------------


def test_keeping_an_identifier_column_is_a_violation() -> None:
    assert "NS003" in _codes('columns = ["Flow ID", "Flow Duration"]')


@pytest.mark.parametrize(
    "source",
    [
        'frame = frame.drop(columns=["Flow ID"])',
        'IDENTIFIER_COLUMNS = ("Flow ID", "Source IP")',
        'def drop_identifiers(frame):\n    return frame.drop(columns=["Timestamp"])',
        'EXCLUDED = ["Source IP"]',
    ],
)
def test_naming_an_identifier_in_order_to_remove_it_is_not(source: str) -> None:
    assert "NS003" not in _codes(source)


def test_the_exemption_is_scoped_to_its_construct_not_to_the_file() -> None:
    """Exempting a whole file would be a hole big enough to hide a leak in."""
    source = 'DROPPED = ["Flow ID"]\n\nkept = frame["Source IP"]\n'
    assert "NS003" in _codes(source)


# --------------------------------------------------------------------------------------
# NS004 -- randomness nobody can reproduce.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "parts = train_test_split(frame, test_size=0.2)",
        "rng = default_rng()",
        "model = RandomForestClassifier(n_estimators=100)",
        "model = IsolationForest(contamination=0.01)",
    ],
)
def test_unseeded_randomness_is_a_violation(source: str) -> None:
    assert "NS004" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        "parts = train_test_split(frame, random_state=seed)",
        "rng = default_rng(seed)",
        "rng = default_rng(settings.seed)",
        "model = RandomForestClassifier(random_state=seed)",
    ],
)
def test_a_seeded_call_is_not(source: str) -> None:
    assert "NS004" not in _codes(source)


def test_drawing_from_an_already_seeded_generator_is_not_a_violation() -> None:
    # `rng.shuffle(x)` is the correct idiom; the rule is after the bare `shuffle(x)` that
    # reaches for global state.
    assert "NS004" not in _codes("rng.shuffle(indices)")


# --------------------------------------------------------------------------------------
# NS005 -- operating points typed into a function body.
# --------------------------------------------------------------------------------------


def test_a_hardcoded_threshold_is_a_violation() -> None:
    assert "NS005" in _codes("flagged = score > 0.87")


def test_a_threshold_from_config_is_not() -> None:
    assert "NS005" not in _codes("flagged = score > settings.thresholds.primary_fpr")


def test_a_structural_comparison_is_not() -> None:
    assert "NS005" not in _codes("nonempty = score > 0.0")


def test_the_hard_label_convention_is_exempt_inside_a_predict_function() -> None:
    source = "def predict(self, X):\n    return (self.scores(X) >= 0.5).astype(int)\n"
    assert "NS005" not in _codes(source)


def test_the_same_literal_outside_a_predict_function_still_fires() -> None:
    source = "def choose_operating_point(scores):\n    return scores >= 0.5\n"
    assert "NS005" in _codes(source)


# --------------------------------------------------------------------------------------
# NS006 -- accuracy without a precision-recall metric beside it.
# --------------------------------------------------------------------------------------


def test_accuracy_alone_is_a_violation() -> None:
    assert "NS006" in _codes("headline = accuracy_score(y_true, y_pred)")


def test_accuracy_beside_pr_auc_is_not() -> None:
    source = "a = accuracy_score(y, p)\nb = average_precision_score(y, p)\n"
    assert "NS006" not in _codes(source)


# --------------------------------------------------------------------------------------
# The mutation harness: the only evidence here worth anything.
# --------------------------------------------------------------------------------------


def test_every_probe_is_caught_in_a_real_module(repo_root: Path) -> None:
    """Inject each violation into a module that does real work, and require the rule to fire.

    A rule that has only fired on a two-line fixture has not been shown to survive the noise of
    a module doing real work -- which is the only place it will ever run.
    """
    host = (repo_root / "netsentry" / "features" / "pipeline.py").read_text(encoding="utf-8")
    missed = [probe.label for probe in PROBES if not run_probe(host, probe).correct]
    assert not missed, f"rules failed to fire on: {missed}"


def test_no_negative_control_raises_a_false_alarm(repo_root: Path) -> None:
    host = (repo_root / "netsentry" / "features" / "pipeline.py").read_text(encoding="utf-8")
    false_alarms = [probe.label for probe in CONTROLS if not run_probe(host, probe).correct]
    assert not false_alarms, f"correct code flagged: {false_alarms}"


def test_the_probe_never_writes_to_the_host(repo_root: Path) -> None:
    path = repo_root / "netsentry" / "features" / "pipeline.py"
    before = path.read_bytes()
    run_probes(path.read_text(encoding="utf-8"))
    assert path.read_bytes() == before


# --------------------------------------------------------------------------------------
# The comparison, and the repository itself.
# --------------------------------------------------------------------------------------


def test_the_textbook_pipeline_trips_every_rule() -> None:
    """The control that makes a clean run mean something."""
    counts = counts_by_rule(lint_source(NAIVE_PIPELINE_SOURCE, "naive.py"))
    silent = [code for code, count in counts.items() if count == 0]
    assert not silent, f"rules that the textbook pipeline did not trip: {silent}"


def test_this_package_stays_inside_its_violation_budget(
    repo_root: Path, settings: Settings
) -> None:
    """The ratchet: three standing violations, audited in the report. A fourth fails here."""
    violations = lint_paths(
        [repo_root / part for part in settings.mlint.roots],
        exclude=tuple(settings.mlint.exclude),
        identifier_scope=tuple(settings.mlint.identifier_scope),
    )
    detail = "\n".join(f"{v.code} {v.path}:{v.line} {v.message}" for v in violations)
    assert len(violations) <= settings.mlint.max_violations, detail


def test_the_study_reports_a_perfect_probe_score(repo_root: Path, settings: Settings) -> None:
    study = run_mlint_study(settings, repo_root)
    assert study.files_scanned > 100
    assert study.detection_rate == pytest.approx(1.0)
    assert study.false_alarm_rate == pytest.approx(0.0)


def test_the_report_is_written(settings: Settings, repo_root: Path, tmp_path: Path) -> None:
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.figures_dir = tmp_path / "figures"
    out = run_mlint_report(settings, repo_root)
    text = out.read_text(encoding="utf-8")
    assert "NS001" in text and "blind spot" in text
    assert "textbook pipeline" in text
