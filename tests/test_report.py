"""Acceptance tests for `report` (BUILD.md Step 8, tests 1-6) + units.

Hand-metric checks run on a synthetic equity curve injected into a BacktestResult
(tests/fixtures/report_handcalc.md); structural metrics (contributions, correlation, turnover)
are pinned against a real engine run so they must satisfy the engine's accounting identity.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tfx.backtest import BacktestParams, run
from tfx.backtest.engine import (
    COST_COLUMNS,
    RISK_EVENT_COLUMNS,
    TRADE_COLUMNS,
    BacktestResult,
)
from tfx.core.risk import RiskParams
from tfx.core.signal import SignalParams
from tfx.core.sizing import SizingParams
from tfx.data.align import FIELD_LEVEL, INSTRUMENT_LEVEL
from tfx.data.schema import TIMESTAMP_INDEX_NAME
from tfx.report import generate, to_markdown

FX, COM = "EUR/USD", "Gold"

RETURNS = [0.10, -0.10, 0.05, -0.05, 0.10]
EQUITY = [1.0, 1.10, 0.99, 1.0395, 0.987525, 1.0862775]


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC",
                         name=TIMESTAMP_INDEX_NAME)


def _synthetic_result(equity_values: list[float]) -> BacktestResult:
    """A minimal BacktestResult carrying a hand-chosen equity curve (single instrument, its
    attribution IS the portfolio return, so the accounting identity holds by construction)."""
    index = _index(len(equity_values))
    equity = pd.Series(equity_values, index=index, name="equity")
    returns = (equity / equity.shift(1) - 1.0).fillna(0.0)
    attribution = returns.to_frame(FX)
    zeros = pd.Series(0.0, index=index)
    return BacktestResult(
        equity_curve=equity,
        positions=attribution.where(attribution != 0, 0.0),  # placeholder weights
        approved=attribution * 0.0,
        trades=pd.DataFrame(columns=list(TRADE_COLUMNS)),
        costs=pd.DataFrame(columns=list(COST_COLUMNS)),
        attribution=attribution,
        risk_events=pd.DataFrame(columns=list(RISK_EVENT_COLUMNS)),
    ), zeros


def _real_result() -> BacktestResult:
    closes = {
        FX: [1.0 * (1 + 0.004 * (1.5 if i % 3 else -1.0)) ** i for i in range(20)],
        COM: [1900.0 * (1 + 0.006 * (1.0 if i % 4 else -1.2)) ** i for i in range(20)],
    }
    n = 20
    index = _index(n)
    frames = {
        symbol: pd.DataFrame({"close": values, "quality": 0}, index=index)
        for symbol, values in closes.items()
    }
    panel = pd.concat(
        frames.values(), axis=1, keys=list(frames.keys()), names=[INSTRUMENT_LEVEL, FIELD_LEVEL]
    )
    params = BacktestParams(
        signal=SignalParams(lookbacks=(1,)),
        sizing=SizingParams(ewma_halflife=1.0, leverage_cap=100.0),
        risk=RiskParams(per_instrument_cap=100.0, asset_class_cap=100.0,
                        portfolio_heat_cap=100.0, drawdown_halt=0.99),
    )
    return run(panel, params)


# 1 — Each metric matches a hand-computed value on a fixture equity curve.
def test_01_handcomputed_metrics():
    result, _ = _synthetic_result(EQUITY)
    metrics = generate(result).metrics

    r = np.array(RETURNS)
    assert metrics["total_return"] == pytest.approx(0.0862775, rel=1e-12)
    assert metrics["cagr"] == pytest.approx(1.0862775 ** (252.0 / 6.0) - 1.0, rel=1e-12)
    assert metrics["sharpe"] == pytest.approx(
        r.mean() / r.std(ddof=1) * math.sqrt(252.0), rel=1e-12
    )
    # downside deviation = RMS of min(r, 0) over ALL bars (target 0), not the std of losing
    # bars alone around their own mean -- see _sortino's docstring for why that's a different,
    # inflated (or NaN-on-uniform-losses) statistic.
    downside_deviation = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    assert metrics["sortino"] == pytest.approx(
        r.mean() / downside_deviation * math.sqrt(252.0), rel=1e-12
    )
    assert metrics["hit_rate"] == pytest.approx(0.6, rel=1e-12)
    assert metrics["avg_win"] == pytest.approx(0.25 / 3.0, rel=1e-12)
    assert metrics["avg_loss"] == pytest.approx(-0.075, rel=1e-12)


# 2 — Drawdown correct: peak-to-trough, sign, duration.
def test_02_drawdown_sign_trough_duration():
    result, _ = _synthetic_result(EQUITY)
    report = generate(result)

    assert report.metrics["max_drawdown"] == pytest.approx(0.987525 / 1.10 - 1.0, rel=1e-12)
    assert report.metrics["max_drawdown"] < 0  # peak-to-trough is NEGATIVE
    assert report.metrics["drawdown_duration_bars"] == 4  # bars 2..5, still under water
    assert report.metrics["time_in_drawdown"] == pytest.approx(4.0 / 6.0, rel=1e-12)
    assert (report.drawdown <= 0).all()
    assert report.drawdown.iloc[1] == 0.0  # at the running peak


# 3 — Per-instrument contributions sum to the total (the engine's accounting identity).
def test_03_contributions_sum_to_total():
    result = _real_result()
    report = generate(result)
    assert report.contributions.sum() == pytest.approx(
        result.attribution.to_numpy().sum(), rel=1e-12
    )
    assert set(report.contributions.index) == {FX, COM}
    # correlation matrix is over the active instruments, unit diagonal
    assert list(report.correlation.columns) == list(report.correlation.index)
    assert np.allclose(np.diag(report.correlation.to_numpy()), 1.0)


# 4 — Numeric metrics deterministic.
def test_04_deterministic():
    result = _real_result()
    first = generate(result)
    second = generate(result)
    assert first.metrics == second.metrics
    pd.testing.assert_series_equal(first.contributions, second.contributions)
    pd.testing.assert_frame_equal(first.correlation, second.correlation)
    assert to_markdown(first) == to_markdown(second)


# 5 — Read-only: report has no path that mutates positions/signals — observational only.
def test_05_read_only_no_core_imports_no_mutation():
    import tfx.report.metrics as metrics_module
    import tfx.report.render as render_module

    for module in (metrics_module, render_module):
        imported = {
            m.__name__ for m in vars(module).values() if isinstance(m, types.ModuleType)
        }
        assert not any(name.startswith("tfx.core") for name in imported)
    # tfx.report must not import the decision core at all (observational only)
    assert not any(
        name.startswith("tfx.core") and "report" in repr(sys.modules[name])
        for name in sys.modules
    )

    result = _real_result()
    frozen = {
        "equity": result.equity_curve.copy(deep=True),
        "positions": result.positions.copy(deep=True),
        "attribution": result.attribution.copy(deep=True),
        "trades": result.trades.copy(deep=True),
    }
    report = generate(result)
    to_markdown(report)
    pd.testing.assert_series_equal(result.equity_curve, frozen["equity"])
    pd.testing.assert_frame_equal(result.positions, frozen["positions"])
    pd.testing.assert_frame_equal(result.attribution, frozen["attribution"])
    pd.testing.assert_frame_equal(result.trades, frozen["trades"])


# 6 — Edge cases: zero trades, all-losing, single instrument.
def test_06_edge_cases():
    # zero trades / flat equity: defined, not crashing; NaN where genuinely undefined
    flat, _ = _synthetic_result([1.0] * 6)
    metrics = generate(flat).metrics
    assert metrics["total_return"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["n_trades"] == 0
    assert math.isnan(metrics["hit_rate"])  # no non-zero bars
    assert metrics["time_in_drawdown"] == 0.0

    # all-losing: hit rate 0, drawdown equals total loss, duration spans the whole decline
    losing, _ = _synthetic_result([1.0, 0.95, 0.9025, 0.857375])
    metrics = generate(losing).metrics
    assert metrics["hit_rate"] == 0.0
    assert metrics["max_drawdown"] == pytest.approx(0.857375 - 1.0, rel=1e-12)
    assert metrics["drawdown_duration_bars"] == 3
    assert math.isnan(metrics["avg_win"])
    # uniform losses (-5% every bar): a real, finite, deeply negative Sortino -- NOT NaN. The
    # old (buggy) std-of-losing-bars formula divided by zero here (identical losses have zero
    # sample variance around their own mean) and silently reported NaN, hiding the worst case.
    assert not math.isnan(metrics["sortino"])
    assert metrics["sortino"] < 0

    # single instrument: correlation degenerates to an empty frame, everything else defined
    single, _ = _synthetic_result(EQUITY)
    report = generate(single)
    assert report.correlation.empty
    markdown = to_markdown(report)
    assert "## Metrics" in markdown and FX in markdown


# --- units -------------------------------------------------------------------------------------
def test_sortino_downside_deviation_not_inflated_by_consistent_large_losses():
    """Regression: consistent, severe losses must produce a small/very-negative Sortino, not an
    inflated one. The old (buggy) formula divided by the STD of losing bars around their OWN
    mean -- for near-identical large losses that denominator collapses toward zero, making
    Sortino explode in magnitude exactly when downside risk is worst."""
    from tfx.report.metrics import _sortino

    # small mixed wins/losses: healthy, moderate Sortino
    mild = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    # severe, nearly-uniform losses with tiny wins: the dangerous case
    severe = pd.Series([0.001, -0.05, 0.001, -0.0499, 0.001, -0.0501])

    mild_sortino = _sortino(mild)
    severe_sortino = _sortino(severe)

    # the severe-loss series must NOT report a larger (less negative / more positive) Sortino
    # than the mild one just because its losses happen to be nearly identical in size.
    assert severe_sortino < mild_sortino
    # and it must be a real, finite, clearly negative number -- not NaN, not exploded.
    assert not math.isnan(severe_sortino)
    assert -100 < severe_sortino < 0


def test_markdown_renders_real_run():
    report = generate(_real_result())
    markdown = to_markdown(report)
    assert "## Per-instrument contribution" in markdown
    assert "## Attribution correlation" in markdown
    assert "observational" in markdown


def test_cli_report_command(tmp_path):
    from pathlib import Path

    from typer.testing import CliRunner

    from tfx.cli import app

    runner = CliRunner()
    cache = tmp_path / "cache"
    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    pulled = runner.invoke(
        app,
        ["data", "pull", "--symbols", "EUR/USD,Gold", "--cache-dir", str(cache),
         "--sample-dir", str(sample)],
    )
    assert pulled.exit_code == 0, pulled.output

    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        ["report", "--symbols", "EUR/USD,Gold", "--cache-dir", str(cache),
         "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and "## Metrics" in out.read_text(encoding="utf-8")
