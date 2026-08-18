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


def test_backtest_run_after_pull(tmp_path):
    cdir = tmp_path / "cache"
    pull_result = runner.invoke(
        app,
        [
            "data", "pull", "--symbols", "EUR/USD,Gold",
            "--start", "2023-01-02", "--end", "2024-12-31",
            "--cache-dir", str(cdir), "--sample-dir", str(SAMPLE_DIR),
        ],
    )
    assert pull_result.exit_code == 0, pull_result.output

    result = runner.invoke(
        app,
        ["backtest", "run", "--symbols", "EUR/USD,Gold",
         "--start", "2023-01-02", "--end", "2024-12-31", "--cache-dir", str(cdir)],
    )
    assert result.exit_code == 0, result.output


def test_backtest_without_cache_fails(tmp_path):
    result = runner.invoke(
        app, ["backtest", "run", "--cache-dir", str(tmp_path / "missing")]
    )
    assert result.exit_code == 1


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


def test_validate_run_geometry_override_is_marked_non_canonical(tmp_path):
    """The bundled fixtures (~520 bars) are far too short for ValidateProtocol()'s default
    geometry (needs ~3800+ bars for 3 folds), so exercising `validate run` at all requires a
    geometry override -- which conveniently is exactly the path that must be marked
    NON-CANONICAL (finding #9: a silent override is a moving-goalpost risk)."""
    cdir = tmp_path / "cache"
    pulled = runner.invoke(
        app,
        [
            "data", "pull", "--symbols", "EUR/USD,Gold",
            "--start", "2023-01-02", "--end", "2024-12-31",
            "--cache-dir", str(cdir), "--sample-dir", str(SAMPLE_DIR),
        ],
    )
    assert pulled.exit_code == 0, pulled.output

    result = runner.invoke(
        app,
        ["validate", "run", "--symbols", "EUR/USD,Gold",
         "--start", "2023-01-02", "--end", "2024-12-31", "--cache-dir", str(cdir),
         "--train-bars", "270", "--test-bars", "50", "--step-bars", "50"],
    )
    # exit code is 0 or 1 depending on the (deterministic, but not the point of this test)
    # verdict -- either way the run must complete and mark itself non-canonical.
    assert result.exit_code in (0, 1), result.output
    assert "NON-CANONICAL" in result.output


def test_validate_run_bad_override_fails_cleanly_not_a_traceback(tmp_path):
    """A --train-bars too small for the (uncustomizable via CLI) default grid's warm-up must
    hit the clean CLI error path, not an unhandled pydantic traceback."""
    cdir = tmp_path / "cache"
    pulled = runner.invoke(
        app,
        [
            "data", "pull", "--symbols", "EUR/USD",
            "--start", "2023-01-02", "--end", "2024-12-31",
            "--cache-dir", str(cdir), "--sample-dir", str(SAMPLE_DIR),
        ],
    )
    assert pulled.exit_code == 0, pulled.output

    result = runner.invoke(
        app,
        ["validate", "run", "--symbols", "EUR/USD", "--cache-dir", str(cdir),
         "--start", "2023-01-02", "--end", "2024-12-31", "--train-bars", "5"],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
