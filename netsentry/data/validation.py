"""Input data-quality gates — validate flow data against the schema contract.

NetSentry is itself about input hygiene, so it validates its own inputs at the
boundary rather than trusting them. Structural problems (missing feature columns,
unknown labels, no rows) are **failures**; quality problems (high missingness,
duplicates, degenerate class balance) are **warnings**. The result is a report a CI
job or an operator can gate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.data.clean import normalize_label
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "data_quality.md"
Status = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    """One data-quality check."""

    name: str
    status: Status
    detail: str


@dataclass
class DataQualityReport:
    """A collection of check results with an overall verdict."""

    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str) -> None:
        self.checks.append(CheckResult(name, status, detail))

    @property
    def ok(self) -> bool:
        """True iff no check failed (warnings are allowed)."""
        return all(c.status != "fail" for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(c.status == "fail" for c in self.checks)

    @property
    def n_warn(self) -> int:
        return sum(c.status == "warn" for c in self.checks)


def _known_labels(settings: Settings) -> set[str]:
    consolidation = settings.labels.consolidation
    return set(schema.RAW_LABELS) | set(consolidation.values()) | {settings.labels.benign_label}


def validate_dataframe(df: pd.DataFrame, settings: Settings) -> DataQualityReport:
    """Run the data-quality checks against a flow frame."""
    report = DataQualityReport()
    columns = {str(c).strip() for c in df.columns}
    stripped = df.rename(columns={c: str(c).strip() for c in df.columns})

    # 1. Non-empty.
    if len(df) == 0:
        report.add("non_empty", "fail", "dataset has 0 rows")
        return report
    report.add("non_empty", "pass", f"{len(df):,} rows")

    # 2. Required feature columns present.
    expected = set(schema.feature_columns(include_destination_port=True))
    missing = sorted(expected - columns)
    if missing:
        report.add("required_features", "fail", f"{len(missing)} missing, e.g. {missing[:5]}")
    else:
        report.add("required_features", "pass", f"all {len(expected)} feature columns present")

    # 3. Label vocabulary (only if a label column is present).
    if schema.LABEL_COLUMN in columns:
        seen = {normalize_label(v) for v in stripped[schema.LABEL_COLUMN].dropna().unique()}
        unknown = sorted(seen - _known_labels(settings))
        if unknown:
            report.add("label_vocabulary", "fail", f"unknown labels: {unknown[:5]}")
        else:
            report.add("label_vocabulary", "pass", f"{len(seen)} known labels")

    # 4. Feature dtypes are numeric.
    present_features = [c for c in expected if c in columns]
    non_numeric = [c for c in present_features if not pd.api.types.is_numeric_dtype(stripped[c])]
    if non_numeric:
        report.add("numeric_features", "warn", f"non-numeric: {non_numeric[:5]}")
    elif present_features:
        report.add("numeric_features", "pass", "all present features numeric")

    # 5. Missing / infinite values per feature column.
    if present_features:
        block = stripped[present_features].replace([np.inf, -np.inf], np.nan)
        frac = block.isna().mean()
        worst = frac.max()
        over = frac[frac > settings.validation.max_nan_fraction]
        col, val = frac.idxmax(), float(worst)
        if len(over):
            report.add(
                "missing_values", "warn", f"{len(over)} cols > threshold; worst {col}={val:.2f}"
            )
        else:
            report.add("missing_values", "pass", f"max missing/inf fraction {val:.3f} ({col})")

    # 6. Duplicate rows.
    dup_frac = float(stripped.duplicated().mean())
    if dup_frac > settings.validation.max_duplicate_fraction:
        report.add("duplicates", "warn", f"{dup_frac:.1%} exact duplicate rows")
    else:
        report.add("duplicates", "pass", f"{dup_frac:.1%} exact duplicate rows")

    # 7. Class balance (degenerate = a single class present).
    if schema.LABEL_COLUMN in columns:
        attack_frac = float(
            (
                stripped[schema.LABEL_COLUMN].map(normalize_label) != settings.labels.benign_label
            ).mean()
        )
        if attack_frac in (0.0, 1.0):
            report.add("class_balance", "warn", f"degenerate: attack fraction = {attack_frac:.0f}")
        else:
            report.add("class_balance", "pass", f"attack fraction {attack_frac:.3f}")

    return report


def render_markdown(report: DataQualityReport) -> str:
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    rows = ["| check | status | detail |", "|---|---|---|"]
    rows += [f"| {c.name} | {icon[c.status]} | {c.detail} |" for c in report.checks]
    verdict = "PASS (no failures)" if report.ok else f"FAIL ({report.n_fail} failing checks)"
    return (
        "# NetSentry — Data Quality Report\n\n"
        f"**Verdict: {verdict}** — {report.n_warn} warning(s).\n\n"
        "Structural problems (missing columns, unknown labels, empty data) fail; "
        "quality problems (missingness, duplicates, degenerate balance) warn.\n\n"
        f"{chr(10).join(rows)}\n"
    )


def run_validation(
    settings: Settings, data_path: Path | None = None
) -> tuple[Path, DataQualityReport]:
    """Validate a dataset (default: the cleaned parquet) and write the report."""
    from netsentry.data.clean import CLEAN_FILENAME

    path = data_path or (settings.paths.data_processed / CLEAN_FILENAME)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `netsentry prep` first or pass a file.")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    report = validate_dataframe(df, settings)

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info(
        "Validated dataset",
        extra={"path": str(path), "ok": report.ok, "warnings": report.n_warn},
    )
    return out_path, report
