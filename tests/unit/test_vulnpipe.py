"""vulnpipe triage: severity normalisation and the severity+traffic fusion ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from netsentry.config import Settings, load_settings
from netsentry.integrations.vulnpipe import (
    VulnFinding,
    load_findings,
    render_triage_markdown,
    triage_findings,
)


@pytest.fixture
def settings(default_config_path: Path) -> Settings:
    return load_settings(default_config_path)


class _StubBundle:
    """A minimal bundle that returns preset attack probabilities (no anomaly detector)."""

    classes = np.array([0, 1])
    anomaly_detector = None
    anomaly_threshold = None
    metadata: ClassVar[dict[str, object]] = {"input_columns": ["Flow Packets/s"]}

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, frame: object) -> np.ndarray:
        p = self._probs[: len(frame)]  # type: ignore[arg-type]
        return np.column_stack([1.0 - p, p])


def test_severity_score_bucket_and_cvss() -> None:
    assert VulnFinding(id="a", severity="low").severity_score() == pytest.approx(0.25)
    assert VulnFinding(id="b", severity="critical").severity_score() == pytest.approx(1.0)
    # CVSS overrides the bucket and is clamped to 0-10.
    assert VulnFinding(id="c", severity="low", cvss=9.0).severity_score() == pytest.approx(0.9)
    assert VulnFinding(id="d", severity="low", cvss=15.0).severity_score() == pytest.approx(1.0)


def test_traffic_context_reranks_above_higher_severity(settings: Settings) -> None:
    findings = [
        VulnFinding(id="A", severity="high", flow={}),  # severe but quiet
        VulnFinding(id="B", severity="medium", flow={}),  # less severe but attack-like
    ]
    triaged = triage_findings(findings, _StubBundle([0.10, 0.95]), settings)  # type: ignore[arg-type]
    assert [t.finding.id for t in triaged] == ["B", "A"]  # B's traffic floats it up
    assert triaged[0].priority == 1
    # B risk = (0.5*0.5 + 0.35*0.95 + 0.15*0) / 1.0
    assert triaged[0].risk == pytest.approx((0.5 * 0.5 + 0.35 * 0.95) / 1.0, abs=1e-6)


def test_equal_severity_broken_by_attack_probability(settings: Settings) -> None:
    findings = [VulnFinding(id="X", severity="high"), VulnFinding(id="Y", severity="high")]
    triaged = triage_findings(findings, _StubBundle([0.2, 0.8]), settings)  # type: ignore[arg-type]
    assert [t.finding.id for t in triaged] == ["Y", "X"]


def test_load_findings_and_render(settings: Settings, tmp_path: Path) -> None:
    payload = {"findings": [{"id": "CVE-1", "severity": "high", "asset": "h1", "flow": {}}]}
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    findings = load_findings(path)
    assert findings[0].id == "CVE-1" and findings[0].asset == "h1"
    triaged = triage_findings(findings, _StubBundle([0.5]), settings)  # type: ignore[arg-type]
    md = render_triage_markdown(triaged)
    assert "vulnpipe Triage" in md and "CVE-1" in md


def test_empty_findings_returns_empty(settings: Settings) -> None:
    assert triage_findings([], _StubBundle([]), settings) == []  # type: ignore[arg-type]
