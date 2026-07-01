"""Data-quality gates: structural failures vs quality warnings."""

from __future__ import annotations

import pandas as pd

from netsentry.config import Settings
from netsentry.data import schema
from netsentry.data.validation import render_markdown, validate_dataframe


def test_clean_data_passes(clean_synth: pd.DataFrame, settings: Settings) -> None:
    report = validate_dataframe(clean_synth, settings)
    assert report.ok
    assert report.n_fail == 0
    names = {c.name: c.status for c in report.checks}
    assert names["required_features"] == "pass"
    assert names["non_empty"] == "pass"


def test_empty_data_fails(settings: Settings) -> None:
    report = validate_dataframe(pd.DataFrame(), settings)
    assert not report.ok
    assert report.checks[0].name == "non_empty"


def test_missing_feature_column_fails(clean_synth: pd.DataFrame, settings: Settings) -> None:
    broken = clean_synth.drop(columns=["Flow Duration"])
    report = validate_dataframe(broken, settings)
    assert not report.ok
    assert any(c.name == "required_features" and c.status == "fail" for c in report.checks)


def test_unknown_label_fails(clean_synth: pd.DataFrame, settings: Settings) -> None:
    tainted = clean_synth.copy()
    tainted[schema.LABEL_COLUMN] = "Totally Novel Attack"
    report = validate_dataframe(tainted, settings)
    assert not report.ok
    assert any(c.name == "label_vocabulary" and c.status == "fail" for c in report.checks)


def test_dash_variant_label_is_recognised(clean_synth: pd.DataFrame, settings: Settings) -> None:
    # The cp1252 en-dash web-attack label should normalize to a known class.
    tagged = clean_synth.copy()
    tagged[schema.LABEL_COLUMN] = "Web Attack \x96 XSS"
    report = validate_dataframe(tagged, settings)
    assert any(c.name == "label_vocabulary" and c.status == "pass" for c in report.checks)


def test_duplicates_warn(clean_synth: pd.DataFrame, settings: Settings) -> None:
    dupey = pd.concat([clean_synth.head(10)] * 4, ignore_index=True)  # ~75% duplicates
    report = validate_dataframe(dupey, settings)
    assert any(c.name == "duplicates" and c.status == "warn" for c in report.checks)
    assert report.ok  # a warning is not a failure


def test_render_markdown_has_verdict(clean_synth: pd.DataFrame, settings: Settings) -> None:
    md = render_markdown(validate_dataframe(clean_synth, settings))
    assert "Data Quality Report" in md
    assert "Verdict" in md
