"""Position sizing: volatility targeting + correlation-aware portfolio construction
(BUILD.md Step 4 -- "the highest-impact module").

Two-stage construction, both point-in-time (data <= t only):
  1. Inverse-vol weighting: each instrument's raw weight is direction / its own EWMA vol, so
     equal-conviction instruments contribute ~equal risk regardless of how volatile they are.
  2. A single portfolio-level scalar, derived from the full EWMA covariance matrix (so genuine
     correlation is priced in -- two instruments that move together can't jointly carry more risk
     than the correlation actually implies), rescales the whole raw vector so ex-ante portfolio
     vol hits `target_vol`. A hard gross-leverage cap is applied last.

Pure function of (panel, directions, params) -- no I/O, no wall-clock, no RNG, no hidden state.
Vol/covariance at bar t use only returns through t (bar t's own close is known at t, the same
convention `signal` uses), so nothing here can look ahead: perturbing a future bar cannot change
an earlier weight, because the EWMA recursion only ever runs forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..data.align import FIELD_LEVEL, periods_per_year, tradable_mask

# PROVISIONAL: placeholders. BUILD.md Step 4 leaves the real values validate-driven (Step 7).
DEFAULT_EWMA_HALFLIFE = 20.0
DEFAULT_TARGET_VOL = 0.10
DEFAULT_LEVERAGE_CAP = 3.0


class SizingParams(BaseModel):
    """Vol-targeting parameters. Immutable."""

    model_config = ConfigDict(frozen=True)

    ewma_halflife: float = Field(default=DEFAULT_EWMA_HALFLIFE, gt=0)  # bars
    target_vol: float = Field(default=DEFAULT_TARGET_VOL, gt=0)  # annualized
    leverage_cap: float = Field(default=DEFAULT_LEVERAGE_CAP, gt=0)  # gross, sum(|weight|)


def size(panel: pd.DataFrame, directions: pd.DataFrame, params: SizingParams) -> pd.DataFrame:
    """Position weight per instrument per bar.

    Each bar's weight is built from an EWMA covariance estimate updated only with returns through
    that bar. NaN wherever direction is unknown, the bar isn't tradable, or that INSTRUMENT's own
    covariance diagonal hasn't accumulated enough history yet (``ceil(ewma_halflife)`` valid
    return observations); exactly ``0.0`` wherever direction is flat, regardless of volatility.

    The covariance update is PER-PAIR masked, not all-or-nothing: on a bar where instrument B is
    unavailable (alignment pad) but A is not, the (A, A) entry -- and every other pair not
    touching B -- still updates from A's real return; only entries touching B (its own variance,
    and every A-B cross term) hold over from the last bar B was live. A single instrument's
    holiday must never freeze the WHOLE portfolio's covariance estimate, and must never leave the
    other instruments' burn-in hostage to that one instrument's history -- each instrument's own
    diagonal reaching ``min_periods`` valid observations gates its OWN usability independently.

    On any given bar, the vol-target/portfolio-scalar math runs only over the *usable* subset of
    instruments -- burned in, a real direction, tradable, and a strictly positive estimated vol.
    Any one instrument failing that (e.g. a single exchange holiday, or a degenerate zero-vol
    series) must never silently affect another instrument's position that day: a naive
    whole-basket matrix computation would let a single NaN (or a zero on the diagonal, dividing
    another instrument's weight by it) poison every dot product it touches, not just its own
    entry.
    """
    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    instruments = list(close.columns)
    n_instr = len(instruments)
    annualization = periods_per_year(close.index)
    tradable = tradable_mask(panel)[instruments]
    log_returns = np.log(close / close.shift(1))

    lam = 0.5 ** (1.0 / params.ewma_halflife)
    min_periods = int(np.ceil(params.ewma_halflife))

    cov = np.zeros((n_instr, n_instr))
    # Per-instrument count of valid return observations behind its OWN diagonal entry -- not a
    # single whole-basket scalar. B's pad must not hold A's burn-in hostage to B's history.
    valid_count = np.zeros(n_instr, dtype=int)
    rows: list[np.ndarray] = []

    for t in close.index:
        r = log_returns.loc[t].to_numpy()
        live = ~np.isnan(r)
        if live.any():
            idx_live = np.where(live)[0]
            # Update ONLY the (live x live) block. A pair's covariance entry decays/updates on a
            # bar only when BOTH its instruments have a real return that bar; every other pair
            # updates normally, unaffected by which instruments happen to be padded today.
            block = np.outer(r[idx_live], r[idx_live])
            cov[np.ix_(idx_live, idx_live)] = (
                lam * cov[np.ix_(idx_live, idx_live)] + (1.0 - lam) * block
            )
            valid_count[idx_live] += 1

        direction_t = directions.loc[t, instruments].to_numpy(dtype=float)
        tradable_t = tradable.loc[t].to_numpy()
        burned_in = valid_count >= min_periods
        vol_t = np.sqrt(np.diag(cov)) * np.sqrt(annualization)

        weight = np.full(n_instr, np.nan)
        usable = burned_in & ~np.isnan(direction_t) & tradable_t & (vol_t > 0)
        if usable.any():
            idx = np.where(usable)[0]
            sub_cov = cov[np.ix_(idx, idx)] * annualization
            sub_direction = direction_t[idx]
            sub_vol = vol_t[idx]

            raw_weight = sub_direction / sub_vol
            portfolio_var = float(raw_weight @ sub_cov @ raw_weight)
            if portfolio_var > 0:
                scalar = params.target_vol / np.sqrt(portfolio_var)
                sub_weight = raw_weight * scalar
                gross = np.abs(sub_weight).sum()
                if gross > params.leverage_cap:
                    sub_weight = sub_weight * (params.leverage_cap / gross)
            else:
                sub_weight = np.zeros(len(idx))

            sub_weight = np.where(sub_direction == 0, 0.0, sub_weight)
            weight[idx] = sub_weight

        rows.append(weight)

    return pd.DataFrame(rows, index=close.index, columns=instruments)
