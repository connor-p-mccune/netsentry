"""Rule-engine tests: clause semantics, NaN/missing-column safety, union decisions,
and the per-rule statistics the comparison report is built on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.config.settings import RuleClause, RuleDefinition, RulesConfig
from netsentry.data import schema
from netsentry.evaluation.rules import rule_statistics
from netsentry.models.rules import RuleEngine


def _rule(name: str, *clauses: tuple[str, str, float]) -> RuleDefinition:
    return RuleDefinition(
        name=name,
        description=name,
        clauses=[RuleClause(feature=f, op=op, value=v) for f, op, v in clauses],  # type: ignore[arg-type]
    )


def test_default_rules_reference_only_schema_features() -> None:
    # A typo'd feature name would silently disable a rule; catch it structurally.
    for rule in RulesConfig().definitions:
        for clause in rule.clauses:
            assert clause.feature in schema.FEATURE_COLUMNS, (rule.name, clause.feature)


def test_rule_fires_only_when_every_clause_holds() -> None:
    engine = RuleEngine(
        [_rule("flood", ("Flow Packets/s", "ge", 100.0), ("SYN Flag Count", "ge", 2.0))]
    )
    df = pd.DataFrame(
        {
            "Flow Packets/s": [500.0, 500.0, 50.0],
            "SYN Flag Count": [3.0, 1.0, 3.0],
        }
    )
    np.testing.assert_array_equal(engine.decisions(df), [True, False, False])


def test_nan_never_satisfies_a_clause() -> None:
    engine = RuleEngine([_rule("r", ("Flow Duration", "le", 100.0))])
    df = pd.DataFrame({"Flow Duration": [50.0, np.nan]})
    np.testing.assert_array_equal(engine.decisions(df), [True, False])


def test_missing_feature_disables_the_rule_without_raising() -> None:
    engine = RuleEngine(
        [
            _rule("needs-absent", ("Not A Column", "ge", 1.0)),
            _rule("works", ("Flow Duration", "ge", 10.0)),
        ]
    )
    df = pd.DataFrame({"Flow Duration": [100.0, 1.0]})
    matches = engine.matches(df)
    assert not matches["needs-absent"].any()
    np.testing.assert_array_equal(matches["works"].to_numpy(), [True, False])


def test_eq_clause_matches_ports() -> None:
    engine = RuleEngine([_rule("ssh", ("Destination Port", "eq", 22.0))])
    df = pd.DataFrame({"Destination Port": [22, 80, 22]})
    np.testing.assert_array_equal(engine.decisions(df), [True, False, True])


def test_score_is_fraction_of_rules_fired() -> None:
    engine = RuleEngine(
        [
            _rule("a", ("Flow Duration", "ge", 10.0)),
            _rule("b", ("Flow Duration", "ge", 1000.0)),
        ]
    )
    df = pd.DataFrame({"Flow Duration": [5000.0, 100.0, 1.0]})
    np.testing.assert_allclose(engine.score(df), [1.0, 0.5, 0.0])


def test_rule_statistics_precision_recall_and_dominant_hit() -> None:
    engine = RuleEngine([_rule("r", ("Flow Packets/s", "ge", 100.0))])
    df = pd.DataFrame({"Flow Packets/s": [500.0, 500.0, 500.0, 1.0]})
    y_bin = np.array([1, 1, 0, 1])  # rule fires on rows 0-2: two attacks, one benign
    labels = np.array(["DDoS", "DoS Hulk", "BENIGN", "PortScan"])
    (stats,) = rule_statistics(engine.matches(df), y_bin, labels, engine)
    assert stats.fired == 3
    assert stats.precision == 2 / 3
    assert stats.recall_all_attacks == 2 / 3
    assert stats.dominant_hit in {"DDoS", "DoS Hulk"}  # tie between the two hits


def test_empty_ruleset_never_flags() -> None:
    engine = RuleEngine([])
    df = pd.DataFrame({"Flow Duration": [1.0, 2.0]})
    assert not engine.decisions(df).any()
    np.testing.assert_allclose(engine.score(df), [0.0, 0.0])
