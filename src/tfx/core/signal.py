"""The trend signal: multi-lookback time-series momentum (BUILD.md Step 3).

Pure function of the point-in-time panel -- no I/O, no wall-clock, no RNG, no hidden state -- so
it is deterministic and free of look-ahead by construction: bar t's direction is a function of
close[t] and each close[t - lookback] alone, never a later bar.

Direction is the equal-weight mean of sign(trailing return) across the lookback ensemble, graded
in [-1, +1] in steps of 1/len(lookbacks); a single-entry tuple degenerates to the discrete
{-1, 0, +1} rule. Signal decides direction only; conviction/position-sizing sophistication
belongs to `sizing` (Step 4), not here -- BUILD.md's "boring, proven signal ... real engineering
in risk and position sizing."

Timing convention: `compute(panel, params)` returns the direction *decided* at bar t, using data
<= t. It is *acted on* at the next tradeable bar, t+1. `shift_for_execution` is the one shared
implementation of that t -> t+1 rule -- `backtest` and (later) `live` both import it instead of
re-deriving the shift, so the convention can never drift between them (CLAUDE.md non-negotiable
#1, backtest-live parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..data.align import FIELD_LEVEL, tradable_mask

# PROVISIONAL: a placeholder ensemble spanning ~1-12 months of daily bars. BUILD.md Step 3 leaves
# the real choice -- which lookbacks, and ensemble vs a single one -- to be swept and evidenced at
# the `validate` stage (Step 7), never hand-chosen here.
DEFAULT_LOOKBACKS = (21, 63, 126, 252)


class SignalParams(BaseModel):
    """Time-series-momentum parameters. Immutable; `lookbacks` is the only free parameter.

    A single-entry tuple is the degenerate single-lookback configuration; more entries form an
    equal-weight ensemble of sign(trailing return) across horizons. Strictly increasing is
    required so every distinct ensemble has exactly one canonical spelling (deterministic
    param-grid identity at the validate stage).
    """

    model_config = ConfigDict(frozen=True)

    lookbacks: tuple[int, ...] = Field(default=DEFAULT_LOOKBACKS)  # trailing windows, in bars

    @field_validator("lookbacks")
    @classmethod
    def _non_empty_positive_strictly_increasing(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("lookbacks must be non-empty")
        if any(lookback <= 0 for lookback in value):
            raise ValueError("every lookback must be > 0")
        if any(b <= a for a, b in zip(value, value[1:], strict=False)):
            raise ValueError("lookbacks must be strictly increasing")
        return value


def compute(panel: pd.DataFrame, params: SignalParams) -> pd.DataFrame:
    """Direction per instrument per bar: mean of sign(trailing return) across the ensemble.

    Returns values in [-1.0, +1.0] (steps of 1/len(lookbacks)), indexed exactly like ``panel``
    with instrument columns. NaN unless the bar is tradable AND every lookback's anchor exists:
    requiring all horizons keeps early-history bars from being scored by a different effective
    model than later ones -- no regime discontinuity partway through a validate fold.
    """
    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    valid = tradable_mask(panel)
    sign_sum = None
    for lookback in params.lookbacks:
        trailing_return = close / close.shift(lookback) - 1.0
        sign = np.sign(trailing_return)
        sign_sum = sign if sign_sum is None else sign_sum + sign
        valid &= trailing_return.notna()
    direction = sign_sum / float(len(params.lookbacks))
    return direction.where(valid)


def shift_for_execution(directions: pd.DataFrame) -> pd.DataFrame:
    """The one shared t -> t+1 timing rule: the direction decided at bar t is the position held
    starting at bar t+1 (the next tradeable bar). Both `backtest` and `live` call this rather
    than re-deriving the shift, so the convention is identical in both (parity)."""
    return directions.shift(1)
