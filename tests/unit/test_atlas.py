"""ATLAS coverage: the mapping's integrity and the verification that keeps it honest.

The point of this mapping is that it cannot quietly rot. These tests pin the two mechanisms
that make that true — every coverage claim names evidence that must exist on disk, and a
claim whose evidence is missing is downgraded rather than believed — plus the arithmetic of
the coverage summary, where out-of-scope techniques must not be counted as wins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netsentry.intel.atlas import (
    ATLAS_MAPPING,
    ATTACKED,
    DEFENDED,
    MEASURED,
    MITIGATED,
    NOT_COVERED,
    OUT_OF_SCOPE,
    AtlasEntry,
    build_layer,
    coverage_counts,
    coverage_ratio,
    verify_mapping,
)


def test_every_technique_id_is_unique() -> None:
    ids = [e.technique_id for e in ATLAS_MAPPING]
    assert len(ids) == len(set(ids))


def test_every_technique_id_uses_the_atlas_namespace() -> None:
    for e in ATLAS_MAPPING:
        assert e.technique_id.startswith("AML.T"), e.technique_id
        assert e.url.endswith(e.technique_id)


def test_every_entry_has_a_tactic_and_a_summary() -> None:
    for e in ATLAS_MAPPING:
        assert e.tactic and len(e.summary) > 40, e.technique_id


def test_covered_entries_name_evidence_and_uncovered_ones_do_not() -> None:
    for e in ATLAS_MAPPING:
        if e.grade in {ATTACKED, MITIGATED, DEFENDED}:
            assert e.module and e.report and e.command, f"{e.technique_id} claims coverage"
        if e.grade in {NOT_COVERED, OUT_OF_SCOPE}:
            assert not e.module, f"{e.technique_id} is a gap but names a module"


def test_the_shipped_mapping_verifies_against_the_repository(repo_root: Path) -> None:
    # The mechanism, exercised on the real repo: every claim's module and report exist.
    unverified = [v.entry.technique_id for v in verify_mapping(repo_root) if not v.verified]
    assert unverified == []


def test_a_claim_whose_module_vanished_is_downgraded(tmp_path: Path) -> None:
    ghost = AtlasEntry(
        technique_id="AML.T9999",
        technique="Ghost Technique",
        tactic="Impact",
        grade=DEFENDED,
        summary="Claims a defense backed by a module that does not exist anywhere on disk.",
        module="netsentry/robustness/deleted.py",
        report="docs/reports/deleted.md",
        command="netsentry ghost",
    )
    verified = verify_mapping(tmp_path, (ghost,))
    assert not verified[0].verified
    assert verified[0].effective_grade == NOT_COVERED  # not the claimed DEFENDED


def test_a_gap_entry_needs_no_evidence_to_be_verified(tmp_path: Path) -> None:
    gap = AtlasEntry(
        technique_id="AML.T9998",
        technique="Uncovered Technique",
        tactic="Impact",
        grade=NOT_COVERED,
        summary="A technique with no implementation, which is exactly what it claims to be.",
    )
    assert verify_mapping(tmp_path, (gap,))[0].verified


def test_out_of_scope_techniques_are_excluded_from_the_coverage_denominator(
    repo_root: Path,
) -> None:
    # A system with no language model must not earn credit for prompt-injection immunity.
    verified = verify_mapping(repo_root)
    counts = coverage_counts(verified)
    covered = counts[ATTACKED] + counts[MITIGATED] + counts[DEFENDED]
    in_scope = len(verified) - counts[OUT_OF_SCOPE]
    assert coverage_ratio(verified) == pytest.approx(covered / in_scope)
    assert counts[OUT_OF_SCOPE] > 0  # the mapping does record out-of-scope entries


def test_coverage_counts_cover_every_entry(repo_root: Path) -> None:
    verified = verify_mapping(repo_root)
    assert sum(coverage_counts(verified).values()) == len(verified)


def test_the_mapping_records_real_gaps(repo_root: Path) -> None:
    # A threat model that lists only successes is marketing; assert this one does not.
    counts = coverage_counts(verify_mapping(repo_root))
    assert counts[NOT_COVERED] > 0 or counts[MEASURED] > 0


def test_layer_is_a_valid_navigator_document(repo_root: Path) -> None:
    layer = build_layer(verify_mapping(repo_root))
    assert layer["domain"] == "atlas"
    techniques = layer["techniques"]
    assert isinstance(techniques, list) and len(techniques) == len(ATLAS_MAPPING)
    for t in techniques:
        assert set(t) >= {"techniqueID", "tactic", "score", "comment", "enabled", "metadata"}
        assert 0 <= t["score"] <= 100


def test_layer_scores_rank_defended_above_uncovered(repo_root: Path) -> None:
    layer = build_layer(verify_mapping(repo_root))
    scores = {t["techniqueID"]: t["score"] for t in layer["techniques"]}
    defended = [e.technique_id for e in ATLAS_MAPPING if e.grade == DEFENDED]
    uncovered = [e.technique_id for e in ATLAS_MAPPING if e.grade == NOT_COVERED]
    assert min(scores[t] for t in defended) > max(scores[t] for t in uncovered)


def test_an_unattacked_control_scores_below_a_defended_technique(repo_root: Path) -> None:
    # The grading distinction that keeps the report honest: a control that was built but
    # never attacked is weaker evidence than one that survived a measured attack.
    layer = build_layer(verify_mapping(repo_root))
    scores = {t["techniqueID"]: t["score"] for t in layer["techniques"]}
    mitigated = [e.technique_id for e in ATLAS_MAPPING if e.grade == MITIGATED]
    defended = [e.technique_id for e in ATLAS_MAPPING if e.grade == DEFENDED]
    assert mitigated and defended
    assert max(scores[t] for t in mitigated) < min(scores[t] for t in defended)
