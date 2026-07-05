"""Deterministic statistics shared by validate (and later, report). No RNG, no bootstrap.

Conventions: daily simple returns; sample std with ddof=1; annualization by sqrt(252). The
Deflated Sharpe Ratio is the Bailey & Lopez de Prado closed form -- it discounts the observed
Sharpe for the number of PRE-REGISTERED trials in the sweep, non-normality included; stdlib
NormalDist keeps it fully deterministic.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
_EULER_GAMMA = 0.5772156649015329


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple per-bar returns of an equity curve (first bar dropped)."""
    return (equity / equity.shift(1) - 1.0).dropna()


def sharpe(returns: pd.Series, periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe (rf=0). NaN when undefined (fewer than 2 obs or zero variance)."""
    values = returns.to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0.0:
        return float("nan")
    return float(values.mean() / std * math.sqrt(periods_per_year))


def t_stat(returns: pd.Series) -> float:
    """t-statistic of the mean per-bar return (mean/std * sqrt(n))."""
    values = returns.to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0.0:
        return float("nan")
    return float(values.mean() / std * math.sqrt(len(values)))


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough fraction (<= 0)."""
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float:
    """Probability the true Sharpe exceeds the expected max Sharpe of ``n_trials`` unskilled
    candidates (Bailey & Lopez de Prado). Uses the PER-PERIOD Sharpe and the sample skew /
    (non-excess) kurtosis. NaN when undefined."""
    values = returns.to_numpy(dtype=float)
    n = len(values)
    if n < 4 or n_trials < 1:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0.0:
        return float("nan")
    sr = float(values.mean() / std)

    centered = values - values.mean()
    m2 = float((centered**2).mean())
    skew = float((centered**3).mean() / m2**1.5)
    kurt = float((centered**4).mean() / m2**2)  # non-excess (normal = 3)

    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        z_hi = NormalDist().inv_cdf(1.0 - 1.0 / n_trials)
        z_lo = NormalDist().inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        sr_benchmark = math.sqrt(1.0 / (n - 1)) * (
            (1.0 - _EULER_GAMMA) * z_hi + _EULER_GAMMA * z_lo
        )

    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if variance_term <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return float(NormalDist().cdf(z))


def stitch(pieces: list[pd.Series]) -> pd.Series:
    """Concatenate sequential non-overlapping return series (walk-forward test windows)."""
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces)
    if not out.index.is_monotonic_increasing or not out.index.is_unique:
        raise ValueError("test windows overlap or are out of order -- fold geometry is broken")
    return out
