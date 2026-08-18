"""Metric computation over a BacktestResult. Pure, deterministic, read-only.

Conventions (documented so the numbers are auditable):
- returns are per-bar portfolio returns from the equity curve; annualization by 252.
- hit rate / avg win / avg loss are BAR-level over non-zero return bars (v1 has no per-trade
  round-trip pairing; the trade log is deltas, not round trips).
- per-instrument contributions are summed attribution (net of that instrument's costs); they
  sum to the sum of per-bar portfolio returns by the engine's accounting identity.
- turnover is annualized gross traded weight: sum(|delta_weight|) * 252 / n_bars.
- drawdown duration is the LONGEST run of consecutive bars below the running peak.

Reuses the deterministic primitives in tfx.validate.stats (single implementation of Sharpe et
al. across validate and report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..validate.stats import (
    TRADING_DAYS_PER_YEAR,
    daily_returns,
    max_drawdown,
    sharpe,
)


@dataclass(frozen=True)
class Report:
    """Everything `render` needs. Frames are copies -- the source result is never mutated."""

    metrics: dict
    equity_curve: pd.Series
    drawdown: pd.Series                 # equity / running peak - 1  (<= 0)
    contributions: pd.Series            # per instrument, sums to sum of per-bar returns
    correlation: pd.DataFrame           # of per-instrument attribution streams
    gross_exposure: pd.Series


def _sortino(returns: pd.Series) -> float:
    """Annualized Sortino: mean return over the DOWNSIDE DEVIATION -- the root-mean-square of
    returns below the target (0), computed over ALL bars (wins count as zero shortfall, they
    are not excluded). This is deliberately NOT the sample std of the losing bars alone around
    THEIR OWN mean: that statistic shrinks toward zero as losses become large but consistent
    (the mean is subtracted out), inflating Sortino exactly when downside risk is worst, and
    returns NaN for uniform losses instead of a small/negative Sortino."""
    values = returns.to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    downside_deviation = math.sqrt(float(np.mean(np.minimum(values, 0.0) ** 2)))
    if downside_deviation == 0.0:
        return float("nan")
    return float(values.mean() / downside_deviation * math.sqrt(TRADING_DAYS_PER_YEAR))


def _longest_run(below_peak: pd.Series) -> int:
    longest = current = 0
    for flag in below_peak.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def generate(result: BacktestResult) -> Report:
    """Compute the full metric set. Read-only: consumes the result, returns new objects."""
    equity = result.equity_curve.copy()
    returns = daily_returns(equity)
    n_bars = len(equity)
    years = n_bars / TRADING_DAYS_PER_YEAR

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    below_peak = drawdown < 0

    nonzero = returns[returns != 0.0]
    wins = nonzero[nonzero > 0]
    losses = nonzero[nonzero < 0]

    # equity[0] IS the initial equity: the engine's bar 0 can carry no pnl and no fill (there is
    # no prior decision), an invariant its own tests pin.
    final_over_initial = (
        float(equity.iloc[-1]) / float(equity.iloc[0]) if n_bars else float("nan")
    )
    cagr = final_over_initial ** (1.0 / years) - 1.0 if years > 0 else float("nan")

    turnover = (
        float(result.trades["delta_weight"].abs().sum()) * TRADING_DAYS_PER_YEAR / n_bars
        if n_bars
        else float("nan")
    )

    contributions = result.attribution.sum()
    active = result.attribution.loc[:, (result.attribution != 0).any()]
    correlation = active.corr() if active.shape[1] >= 2 else pd.DataFrame()
    gross = result.positions.abs().sum(axis=1)

    metrics = {
        "n_bars": n_bars,
        "n_trades": int(len(result.trades)),
        "total_return": final_over_initial - 1.0,
        "cagr": cagr,
        "sharpe": sharpe(returns),
        "sortino": _sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "drawdown_duration_bars": _longest_run(below_peak),
        "time_in_drawdown": float(below_peak.mean()) if n_bars else float("nan"),
        "hit_rate": float(len(wins) / len(nonzero)) if len(nonzero) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "annual_turnover": turnover,
        "total_costs": float(result.costs["amount"].sum()) if len(result.costs) else 0.0,
        "mean_gross_exposure": float(gross.mean()) if n_bars else float("nan"),
        "max_gross_exposure": float(gross.max()) if n_bars else float("nan"),
    }

    return Report(
        metrics=metrics,
        equity_curve=equity,
        drawdown=drawdown,
        contributions=contributions,
        correlation=correlation,
        gross_exposure=gross,
    )
