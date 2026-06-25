"""CLI contract: all documented subcommands are present and help works."""

from __future__ import annotations

from typer.testing import CliRunner

from netsentry.cli import app

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("download", "prep", "train", "eval", "serve", "benchmark"):
        assert command in result.output


def test_train_has_supervised_and_anomaly() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "supervised" in result.output
    assert "anomaly" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "netsentry" in result.output
