"""The parser, the numeric matcher, the gate, and the harness that watches them fire.

Two of these are regression tests for bugs the study found in itself: the matcher used to treat
a quote as a point rather than as the interval it asserts (so `2.31` was a fault against a report
saying `2.312`), and the harness used to ask whether a token still verified *after replacing it*,
which no token can and which turned a measurement into a tautology.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.governance.claims import (
    TRACEABLE,
    UNSOURCED,
    VERIFIED,
    Budget,
    Corpus,
    Quantity,
    audit,
    broken_links,
    orphan_reports,
    perturb_token,
    readings,
    replace_at,
    run_injections,
    sections,
)

README = """# Title

## Detection quality

```bash
python -m netsentry.cli eval   # -> docs/reports/evaluation.md
```

PR-AUC is 0.529 on the temporal split and 2.7% of flows alert.

## Something else

See [the drift study](docs/reports/drift.md). PSI reaches 0.184.
"""

REPORTS = {
    "evaluation": "PR-AUC 0.529 with an alert rate of 0.027 and a baseline of 0.250.\n",
    "drift": "The worst feature's PSI is 0.184.\n",
    "unlinked": "Nobody points here, and it mentions 0.777.\n",
}


@pytest.fixture()
def corpus() -> Corpus:
    return Corpus.of(REPORTS)


# --------------------------------------------------------------------------------------
# Parsing.
# --------------------------------------------------------------------------------------


def test_sections_split_on_headings_and_collect_their_links() -> None:
    found = {section.title: section.reports for section in sections(README)}
    assert found["Detection quality"] == ("evaluation",)
    assert found["Something else"] == ("drift",)


def test_a_section_without_a_report_link_contributes_no_claims(corpus: Corpus) -> None:
    readme = "## Orphan\n\nA number: 0.529\n"
    assert audit(readme, corpus) == []


def test_a_claim_records_the_readme_line_it_sits_on(corpus: Corpus) -> None:
    claim = next(c for c in audit(README, corpus) if c.token == "0.184")
    assert README.splitlines()[claim.line - 1].startswith("See [the drift study]")


# --------------------------------------------------------------------------------------
# What counts as a claim.
# --------------------------------------------------------------------------------------


def test_a_two_decimal_number_is_a_claim(corpus: Corpus) -> None:
    assert any(claim.token == "0.529" for claim in audit(README, corpus))


def test_a_one_decimal_percentage_is_a_claim(corpus: Corpus) -> None:
    assert any(claim.token == "2.7%" for claim in audit(README, corpus))


def test_a_round_figure_is_not_a_claim(corpus: Corpus) -> None:
    """`30%` carries no decimal, so it is prose rather than a number worth checking."""
    readme = "## S\n\nSee docs/reports/drift.md -- roughly 30% of flows, or 5 per day.\n"
    assert audit(readme, corpus) == []


def test_a_version_string_is_not_a_claim(corpus: Corpus) -> None:
    readme = "## S\n\nSee docs/reports/drift.md, released in v1.2.3.\n"
    assert [c.token for c in audit(readme, corpus)] == []


# --------------------------------------------------------------------------------------
# Numeric matching -- the part that had to be rewritten twice.
# --------------------------------------------------------------------------------------


def test_a_quote_is_the_interval_it_asserts() -> None:
    """`2.31` claims the report says something rounding to 2.31, not exactly 2.31."""
    quantity = readings("2.31")[0]
    assert quantity.covers(np.array([2.312])) == 1
    assert quantity.covers(np.array([2.32])) == 0


def test_a_percentage_matches_the_fraction_a_report_states(corpus: Corpus) -> None:
    """The report says `0.027`; the README says `2.7%`. Same fact, different units."""
    claim = next(c for c in audit(README, corpus) if c.token == "2.7%")
    assert claim.verdict == VERIFIED
    assert not claim.exact


def test_an_exact_token_is_marked_exact(corpus: Corpus) -> None:
    claim = next(c for c in audit(README, corpus) if c.token == "0.529")
    assert claim.verdict == VERIFIED and claim.exact


def test_a_number_only_in_another_report_is_traceable(corpus: Corpus) -> None:
    readme = "## S\n\nSee docs/reports/drift.md -- the baseline is 0.250.\n"
    assert [c.verdict for c in audit(readme, corpus)] == [TRACEABLE]


def test_a_number_in_no_report_is_unsourced(corpus: Corpus) -> None:
    readme = "## S\n\nSee docs/reports/drift.md -- the figure is 0.4242.\n"
    assert [c.verdict for c in audit(readme, corpus)] == [UNSOURCED]


def test_a_quantity_with_a_wide_tolerance_covers_more() -> None:
    stated = np.array([0.51, 0.52, 0.53])
    assert Quantity(0.52, 0.005).covers(stated) == 1
    assert Quantity(0.52, 0.05).covers(stated) == 3


# --------------------------------------------------------------------------------------
# The other direction.
# --------------------------------------------------------------------------------------


def test_a_link_to_a_missing_report_is_broken(corpus: Corpus) -> None:
    readme = "## S\n\nSee docs/reports/nowhere.md.\n"
    assert broken_links(readme, corpus) == [("S", "nowhere")]


def test_a_report_nobody_links_is_an_orphan(corpus: Corpus) -> None:
    assert orphan_reports(README, corpus) == ["unlinked"]


# --------------------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------------------


def test_the_gate_refuses_an_unsourced_claim(corpus: Corpus) -> None:
    readme = "## S\n\nSee docs/reports/drift.md -- the figure is 0.4242.\n"
    assert Budget(unsourced=0, traceable=0).fails(audit(readme, corpus))


def test_the_gate_refuses_the_traceable_count_growing(corpus: Corpus) -> None:
    """Drift into the milder class must fail too, or the gate is weaker than the checker."""
    readme = "## S\n\nSee docs/reports/drift.md -- the baseline is 0.250.\n"
    claims = audit(readme, corpus)
    assert Budget(unsourced=0, traceable=0).fails(claims)
    assert not Budget(unsourced=0, traceable=1).fails(claims)


def test_a_clean_readme_passes_its_own_budget(corpus: Corpus) -> None:
    assert not Budget(unsourced=0, traceable=0).fails(audit(README, corpus))


# --------------------------------------------------------------------------------------
# The harness.
# --------------------------------------------------------------------------------------


def test_a_perturbed_token_keeps_its_shape() -> None:
    rng = np.random.default_rng(0)
    for token in ("0.529", "2.7%", "10.10"):
        mutated = perturb_token(token, rng)
        assert mutated != token
        assert len(mutated) == len(token)
        assert mutated.endswith("%") == token.endswith("%")


def test_a_replacement_only_touches_its_own_line() -> None:
    """Replacing the first match anywhere would mutate a different claim than the one tested."""
    text = "0.529 here\nand 0.529 there\n"
    assert replace_at(text, 2, "0.529", "0.599") == "0.529 here\nand 0.599 there"


def test_a_replacement_on_the_wrong_line_is_refused() -> None:
    assert replace_at("a\nb\n", 1, "0.529", "0.599") is None


def test_the_harness_catches_a_drifted_number(corpus: Corpus) -> None:
    rows = {
        row.fault: row
        for row in run_injections(
            README, corpus, samples=5, budget=Budget(0, 0), rng=np.random.default_rng(1)
        )
    }
    drift = rows["a number drifts one digit"]
    assert drift.injected > 0
    assert drift.detection_rate == 1.0


def test_the_harness_catches_a_broken_link(corpus: Corpus) -> None:
    rows = {
        row.fault: row
        for row in run_injections(
            README, corpus, samples=5, budget=Budget(0, 0), rng=np.random.default_rng(2)
        )
    }
    assert rows["a report link points at nothing"].detection_rate == 1.0


def test_the_harness_reports_zero_rather_than_dividing_by_zero(corpus: Corpus) -> None:
    rows = run_injections(
        "## S\n\nNothing here.\n",
        corpus,
        samples=3,
        budget=Budget(0, 0),
        rng=np.random.default_rng(3),
    )
    drift = rows[0]
    assert drift.injected == 0 and drift.detection_rate == 0.0
