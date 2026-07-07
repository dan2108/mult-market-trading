"""The pre-registered protocol: grid, fold geometry, haircut, and pass thresholds.

Everything that could move a goalpost lives HERE, in one frozen object, constructed before the
run and echoed verbatim into the result. The grid is deliberately tiny: `lookbacks` (where
ensemble-vs-single gets *evidenced*) x `ewma_halflife`. `target_vol` and the leverage cap are
scaling policy, not edge, and ALL risk limits are guardrails -- none of them are ever swept or
fit (a guardrail tuned to the data is not a guardrail).
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    FRAGILE = "fragile"                          # thresholds met, but the edge is a knife-edge
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GridPoint(BaseModel):
    """One pre-registered candidate: the only two swept dimensions in v1."""

    model_config = ConfigDict(frozen=True)

    lookbacks: tuple[int, ...]
    ewma_halflife: float = Field(gt=0)

    def label(self) -> str:
        inner = ",".join(str(lb) for lb in self.lookbacks)
        return f"lb=({inner}) hl={self.ewma_halflife:g}"

    def warmup_bars(self) -> int:
        """Bars before this candidate can produce its first real decision: signal needs
        `max(lookbacks)` bars for its longest anchor, sizing needs `ceil(ewma_halflife)` valid
        return observations for its covariance to burn in -- both run over the SAME timeline in
        parallel (not sequentially), so the bar their overlap first has a real weight is the
        MAX of the two, not their sum."""
        return max(max(self.lookbacks), math.ceil(self.ewma_halflife))


# The default pre-registered grid: 6 lookback sets (four singles + two ensembles, so the sweep
# itself decides ensemble-vs-single) x 3 vol halflives = 18 candidates. PROVISIONAL like every
# other default -- but changing it after seeing results is exactly what the echo prevents.
DEFAULT_GRID: tuple[GridPoint, ...] = tuple(
    GridPoint(lookbacks=lookbacks, ewma_halflife=halflife)
    for lookbacks in (
        (21,), (63,), (126,), (252,), (63, 126, 252), (21, 63, 126, 252),
    )
    for halflife in (10.0, 20.0, 60.0)
)


# Minimum active (post-warm-up) train bars every grid candidate must have room to score a
# Sharpe over. Without this, a candidate whose warm-up consumes the whole train window gets an
# undefined (NaN) train score, which the runner's max-score selection has no honest way to
# handle -- it would silently fall back to grid order (the FIRST candidate wins by position, not
# evidence), and a gate built on that can return a verdict with zero real selection behind it.
MIN_ACTIVE_TRAIN_BARS = 10


class ValidateProtocol(BaseModel):
    """Frozen before the run; echoed verbatim into the result. No moving goalposts."""

    model_config = ConfigDict(frozen=True)

    # rolling walk-forward geometry, in bars (~4y train / 1y test / 1y step on daily data)
    train_bars: int = Field(default=1008, gt=0)
    test_bars: int = Field(default=252, gt=0)
    step_bars: int = Field(default=252, gt=0)
    min_folds: int = Field(default=3, gt=0)

    grid: tuple[GridPoint, ...] = DEFAULT_GRID

    # the honesty devices
    haircut: float = Field(default=0.5, gt=0, le=1)      # thresholds apply to haircut numbers
    min_sharpe_after_haircut: float = Field(default=0.5)
    min_tstat: float = Field(default=2.0)
    min_trades: int = Field(default=100, gt=0)

    # robustness: the modal winning candidate must be selected in at least this share of folds,
    # and the mean OOS Sharpe across the WHOLE grid must be positive (the edge can't live in
    # one lucky cell)
    min_modal_share: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def _warmup_fits_train_window(self) -> ValidateProtocol:
        worst_case_warmup = max(point.warmup_bars() for point in self.grid)
        if worst_case_warmup + MIN_ACTIVE_TRAIN_BARS > self.train_bars:
            raise ValueError(
                f"train_bars={self.train_bars} cannot fit the grid's worst-case warm-up "
                f"({worst_case_warmup} bars: max(max(lookbacks), ceil(ewma_halflife)) over the "
                f"grid) plus a minimum {MIN_ACTIVE_TRAIN_BARS} active bars to score a train "
                f"Sharpe on. Without this, selection silently falls back to grid order with an "
                f"undefined score for the offending candidate(s) -- increase train_bars, or "
                f"shrink the grid's lookbacks/ewma_halflife."
            )
        return self
