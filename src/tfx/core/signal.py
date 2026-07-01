"""The trend signal: time-series momentum (BUILD.md Step 3).

Pure function of the point-in-time panel -- no I/O, no wall-clock, no RNG, no hidden state -- so
it is deterministic and free of look-ahead by construction: bar t's direction is a function of
close[t] and close[t - lookback] alone, never a later bar.

Direction is discrete {-1, 0, +1} (sign of the trailing `lookback`-bar return). Signal decides
direction only; conviction/position-sizing sophistication belongs to `sizing` (Step 4), not here
-- BUILD.md's "boring, proven signal ... real engineering in risk and position sizing."

Timing convention: `compute(panel, params)` returns the direction *decided* at bar t, using data
<= t. It is *acted on* at the next tradeable bar, t+1. `shift_for_execution` is the one shared
implementation of that t -> t+1 rule -- `backtest` and (later) `live` both import it instead of
re-deriving the shift, so the convention can never drift between them (CLAUDE.md non-negotiable
#1, backtest-live parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..data.align import FIELD_LEVEL, tradable_mask

# PROVISIONAL: a single placeholder lookback. BUILD.md Step 3 leaves the real parameter range to
# be decided at the `validate` stage (Step 7), swept and evidenced -- never hand-chosen here.
DEFAULT_LOOKBACK = 20


class SignalParams(BaseModel):
    """Time-series-momentum parameters. Immutable; `lookback` is the only free parameter."""

    model_config = ConfigDict(frozen=True)

    lookback: int = Field(default=DEFAULT_LOOKBACK, gt=0)  # trailing window, in bars


def compute(panel: pd.DataFrame, params: SignalParams) -> pd.DataFrame:
    """Direction per instrument per bar: sign of the trailing `lookback`-bar return.

    Returns -1.0 / 0.0 / +1.0, indexed exactly like ``panel`` with instrument columns. NaN
    wherever the bar (or its lookback-bars-earlier anchor) isn't tradable -- an alignment pad, or
    simply not enough history yet -- so a padded/missing bar never produces a signal.
    """
    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    trailing_return = close / close.shift(params.lookback) - 1.0
    direction = np.sign(trailing_return)
    valid = tradable_mask(panel) & trailing_return.notna()
    return direction.where(valid)


def shift_for_execution(directions: pd.DataFrame) -> pd.DataFrame:
    """The one shared t -> t+1 timing rule: the direction decided at bar t is the position held
    starting at bar t+1 (the next tradeable bar). Both `backtest` and `live` call this rather
    than re-deriving the shift, so the convention is identical in both (parity)."""
    return directions.shift(1)
