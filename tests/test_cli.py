"""CLI smoke tests via Typer's runner.

We assert exit codes and filesystem side effects rather than captured stdout: Rich binds to the
real stdout at import time, so its output is not reliably captured by the test runner.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tfx.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
runner = CliRunner()


def test_version_exits_zero():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_info_exits_zero():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


def test_pull_then_report(tmp_path):
    cdir = tmp_path / "cache"
    pull_result = runner.invoke(
        app,
        [
            "data", "pull", "--symbols", "EUR/USD,S&P 500",
            "--start", "2023-01-02", "--end", "2023-03-01",
            "--cache-dir", str(cdir), "--sample-dir", str(SAMPLE_DIR),
        ],
    )
    assert pull_result.exit_code == 0, pull_result.output
    assert (cdir / "manifest.json").exists()
    assert (cdir / "EUR_USD.parquet").exists()

    report_result = runner.invoke(
        app,
        [
            "data", "report", "--symbols", "EUR/USD,S&P 500",
            "--start", "2023-01-02", "--end", "2023-03-01", "--cache-dir", str(cdir),
        ],
    )
    assert report_result.exit_code == 0, report_result.output


def test_unknown_source_fails(tmp_path):
    result = runner.invoke(
        app,
        ["data", "pull", "--source", "nope", "--symbols", "EUR/USD",
         "--cache-dir", str(tmp_path / "c")],
    )
    assert result.exit_code == 1


def test_report_without_cache_fails(tmp_path):
    result = runner.invoke(
        app, ["data", "report", "--cache-dir", str(tmp_path / "empty")]
    )
    assert result.exit_code == 1
