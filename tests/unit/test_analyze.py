"""The analysis-suite index writer (orchestration is covered by the report tests)."""

from __future__ import annotations

from pathlib import Path

from netsentry.evaluation.analyze import AnalysisEntry, write_index


def test_write_index_links_ok_and_reports_failures(tmp_path: Path) -> None:
    entries = [
        AnalysisEntry("Eval", "metrics", "evaluation.md", ok=True),
        AnalysisEntry("Cost", "economics", "cost.md", ok=False, error="boom"),
    ]
    out = write_index(tmp_path, entries)
    text = out.read_text(encoding="utf-8")

    assert out.name == "INDEX.md"
    assert "[open](evaluation.md)" in text  # successful report is linked
    assert "failed — boom" in text  # failed report records the error, no link
    assert "(cost.md)" not in text
