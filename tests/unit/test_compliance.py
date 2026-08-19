"""The conformance mapping, and the mechanism that stops it becoming a wall-chart.

A compliance document's usual failure is not dishonesty, it is that the evidence moved and the
document did not. So the load-bearing test here is the one that deletes the evidence and checks
the control downgrades itself — everything else is arithmetic around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from netsentry.config import Settings
from netsentry.governance.compliance import (
    CONTROLS,
    MET,
    NOT_APPLICABLE,
    PARTIAL,
    UNMET,
    Control,
    build_mapping,
    coverage_ratio,
    grade_counts,
    run_compliance_report,
    verify_controls,
)

# --------------------------------------------------------------------------------------
# The mapping itself.
# --------------------------------------------------------------------------------------


def test_every_control_carries_a_framework_reference_and_evidence_text() -> None:
    for control in CONTROLS:
        assert control.framework in {"NIST AI RMF", "EU AI Act"}
        assert control.reference and control.title
        assert len(control.evidence) > 40, control.reference
        assert control.grade in {MET, PARTIAL, UNMET, NOT_APPLICABLE}


def test_references_are_unique() -> None:
    keys = [(control.framework, control.reference) for control in CONTROLS]
    assert len(keys) == len(set(keys))


def test_a_claim_of_coverage_must_name_evidence() -> None:
    # The rule that keeps the document honest: "met" without a file is an assertion, and the
    # whole point of this module is that assertions do not count.
    for control in CONTROLS:
        if control.grade in {MET, PARTIAL}:
            assert control.module or control.report, control.reference


def test_gaps_deliberately_carry_no_evidence() -> None:
    for control in CONTROLS:
        if control.grade in {UNMET, NOT_APPLICABLE}:
            assert not control.module and not control.report, control.reference


def test_the_real_mapping_verifies_against_this_repository(repo_root: Path) -> None:
    verified = verify_controls(repo_root)
    unbacked = [
        item.control.reference
        for item in verified
        if not item.verified and item.control.grade in {MET, PARTIAL}
    ]
    assert not unbacked, f"controls claiming evidence that is missing: {unbacked}"


# --------------------------------------------------------------------------------------
# The mechanism.
# --------------------------------------------------------------------------------------


def test_a_control_whose_evidence_is_missing_downgrades_itself(tmp_path: Path) -> None:
    """The load-bearing test: delete the evidence, lose the claim.

    Nothing else in this module matters if a renamed study can keep satisfying a regulator.
    """
    control = Control(
        framework="EU AI Act",
        reference="Article 99",
        title="A control whose evidence has been deleted",
        grade=MET,
        evidence="A study that used to exist and no longer does, which must not still count.",
        module="netsentry/does_not_exist.py",
        report="docs/reports/gone.md",
    )
    verified = verify_controls(tmp_path, (control,))[0]
    assert not verified.verified
    assert verified.effective_grade == UNMET
    assert verified.control.grade == MET  # the *claim* is preserved; the effect is not


def test_a_control_with_present_evidence_keeps_its_grade(tmp_path: Path) -> None:
    (tmp_path / "netsentry").mkdir()
    (tmp_path / "netsentry" / "present.py").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "present.md").write_text("", encoding="utf-8")
    control = Control(
        framework="NIST AI RMF",
        reference="GOVERN 0.0",
        title="A control whose evidence is present",
        grade=PARTIAL,
        evidence="Evidence that exists on disk and should therefore keep its claimed grade.",
        module="netsentry/present.py",
        report="docs/present.md",
    )
    verified = verify_controls(tmp_path, (control,))[0]
    assert verified.verified
    assert verified.effective_grade == PARTIAL


def test_partial_evidence_is_not_partial_credit(tmp_path: Path) -> None:
    # Half the evidence is not half a claim: if either artifact is missing the control fails.
    (tmp_path / "netsentry").mkdir()
    (tmp_path / "netsentry" / "half.py").write_text("", encoding="utf-8")
    control = Control(
        framework="EU AI Act",
        reference="Article 98",
        title="Half-present evidence",
        grade=MET,
        evidence="A module that exists and a report that does not, which must not count.",
        module="netsentry/half.py",
        report="docs/reports/missing.md",
    )
    assert verify_controls(tmp_path, (control,))[0].effective_grade == UNMET


# --------------------------------------------------------------------------------------
# The arithmetic.
# --------------------------------------------------------------------------------------


def test_not_applicable_controls_leave_the_denominator(tmp_path: Path) -> None:
    # A system with no deployer does not earn credit for a deployer's obligation, and it must
    # not be penalised for it either.
    controls = (
        Control("EU AI Act", "A", "met", MET, "x" * 50, report="NOTES.md"),
        Control("EU AI Act", "B", "not applicable", NOT_APPLICABLE, "y" * 50),
    )
    (tmp_path / "NOTES.md").write_text("", encoding="utf-8")
    assert coverage_ratio(verify_controls(tmp_path, controls)) == pytest.approx(1.0)


def test_a_partial_counts_as_half(tmp_path: Path) -> None:
    (tmp_path / "NOTES.md").write_text("", encoding="utf-8")
    controls = (
        Control("EU AI Act", "A", "met", MET, "x" * 50, report="NOTES.md"),
        Control("EU AI Act", "B", "partial", PARTIAL, "y" * 50, report="NOTES.md"),
    )
    assert coverage_ratio(verify_controls(tmp_path, controls)) == pytest.approx(0.75)


def test_coverage_can_be_filtered_by_framework(repo_root: Path) -> None:
    verified = verify_controls(repo_root)
    both = coverage_ratio(verified)
    nist = coverage_ratio(verified, "NIST AI RMF")
    eu = coverage_ratio(verified, "EU AI Act")
    assert 0.0 <= eu <= 1.0 and 0.0 <= nist <= 1.0
    assert min(nist, eu) <= both <= max(nist, eu)


def test_grade_counts_sum_to_the_control_count(repo_root: Path) -> None:
    verified = verify_controls(repo_root)
    assert sum(grade_counts(verified).values()) == len(verified)


# --------------------------------------------------------------------------------------
# The artifacts.
# --------------------------------------------------------------------------------------


def test_the_json_mapping_records_both_the_claim_and_its_effect(repo_root: Path) -> None:
    mapping = build_mapping(verify_controls(repo_root))
    assert mapping["frameworks"]
    controls = mapping["controls"]
    assert isinstance(controls, list) and controls
    for entry in controls:
        assert {"claimed_grade", "effective_grade", "verified", "evidence"} <= set(entry)
    assert json.dumps(mapping)  # serialisable, since an auditor consumes the file not the object


def test_the_report_and_mapping_are_written(settings: Settings, tmp_path: Path) -> None:
    settings.paths.reports_dir = tmp_path / "docs" / "reports"
    out = run_compliance_report(settings)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "EU AI Act" in text and "NIST AI RMF" in text
    assert "legal advice" in text  # the disclaimer is not optional
    assert (tmp_path / "docs" / "reports" / "compliance_mapping.json").exists()
