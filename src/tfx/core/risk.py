"""The deterministic risk veto (BUILD.md Step 5): sized weights in, approved weights out.

Sits between `sizing` and execution in BOTH backtest and live (parity). It can only ever REDUCE
risk -- every stage multiplies weights by a scalar in [0, 1], so monotonic safety holds by
construction: |approved| <= |proposed| elementwise, and a sign can never flip.

Fixed application order (documented, load-bearing):

    kill switch -> drawdown halt -> per-instrument cap -> asset-class cap -> portfolio-heat cap

Design decisions:
- Pure per-bar function. Drawdown depends on running equity, which only exists bar-by-bar in the
  backtest loop (and cycle-by-cycle live), so `check` takes ONE bar's proposal plus the account
  truth and stores nothing. All state arrives as arguments -- no hidden state, no wall-clock.
- Caps live in weight space and never touch the panel. Sizing already prices statistical
  correlation through its EWMA covariance; risk's value is its INDEPENDENCE -- a dumb backstop
  that cannot drift with an estimator. The correlation/exposure cap is therefore gross exposure
  per `tfx.instruments.AssetClass` (deterministic clustering), not a correlation estimate.
- NaN proposal means "no decision" upstream and passes through as NaN -- risk cannot approve a
  trade nobody proposed. The exceptions are the kill switch and the drawdown halt, which force
  0.0 everywhere: flat means flat.
- Unknown symbols raise (same no-silent-fallthrough rule as `tfx.costs.params.get_params`).
- The drawdown halt latches emergently: halted -> flat -> equity freezes -> drawdown stays at or
  past the threshold -> still halted. Resuming is a HUMAN decision (equity injection / param
  change), deliberately not automated in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..instruments import get_instrument

# PROVISIONAL guardrail levels, like every other DEFAULT_* in the core -- but unlike signal /
# sizing parameters these are POLICY, never swept or fit at the validate stage (a guardrail that
# is tuned to the data is not a guardrail).
DEFAULT_PER_INSTRUMENT_CAP = 0.5   # max |weight| for any single instrument
DEFAULT_ASSET_CLASS_CAP = 1.0      # max gross Σ|weight| within one asset class
DEFAULT_PORTFOLIO_HEAT_CAP = 2.0   # max gross Σ|weight| overall (< sizing's leverage cap 3.0)
DEFAULT_DRAWDOWN_HALT = 0.20       # halt all risk at >= 20% drawdown from peak equity


class RiskParams(BaseModel):
    """Risk limits. Immutable; all levels provisional but policy (not validate-swept)."""

    model_config = ConfigDict(frozen=True)

    per_instrument_cap: float = Field(default=DEFAULT_PER_INSTRUMENT_CAP, gt=0)
    asset_class_cap: float = Field(default=DEFAULT_ASSET_CLASS_CAP, gt=0)
    portfolio_heat_cap: float = Field(default=DEFAULT_PORTFOLIO_HEAT_CAP, gt=0)
    drawdown_halt: float = Field(default=DEFAULT_DRAWDOWN_HALT, gt=0, lt=1)


class AccountState(BaseModel):
    """The account truth `check` needs, passed in by the caller (backtest loop / live cycle).
    The caller maintains it (`peak_equity = max(peak_equity, equity)`); risk only reads it."""

    model_config = ConfigDict(frozen=True)

    equity: float = Field(gt=0)
    peak_equity: float = Field(gt=0)
    kill_switch: bool = False

    @property
    def drawdown(self) -> float:
        """Fractional drawdown from peak, clamped at 0 (equity above peak is not a drawdown)."""
        return max(0.0, 1.0 - self.equity / self.peak_equity)


class RiskLimit(StrEnum):
    KILL_SWITCH = "kill_switch"
    DRAWDOWN_HALT = "drawdown_halt"
    PER_INSTRUMENT = "per_instrument"
    ASSET_CLASS = "asset_class"
    PORTFOLIO_HEAT = "portfolio_heat"


@dataclass(frozen=True)
class RiskReason:
    """One reduction, for the audit trail: which limit fired, on what, by how much."""

    limit: RiskLimit
    instrument: str | None   # None for portfolio-level limits (halt, heat, class)
    scale: float             # multiplier applied, in [0, 1]
    detail: str


@dataclass(frozen=True)
class RiskDecision:
    """Approved weights plus the reasons for every reduction. `halted` means flat-everything
    (kill switch or drawdown halt) -- the caller must trade to flat, not merely skip orders."""

    approved: pd.Series
    reasons: tuple[RiskReason, ...]
    halted: bool


def check(proposed: pd.Series, state: AccountState, params: RiskParams) -> RiskDecision:
    """Veto/scale one bar's proposed weights. Only ever reduces: every stage multiplies by a
    scalar in [0, 1]. Raises KeyError on any symbol not in the instrument basket."""
    asset_class_of = {symbol: get_instrument(symbol).asset_class for symbol in proposed.index}

    if state.kill_switch:
        return RiskDecision(
            approved=pd.Series(0.0, index=proposed.index),
            reasons=(
                RiskReason(RiskLimit.KILL_SWITCH, None, 0.0, "kill switch set: all flat"),
            ),
            halted=True,
        )

    if state.drawdown >= params.drawdown_halt:
        detail = (
            f"drawdown {state.drawdown:.6g} >= halt {params.drawdown_halt:.6g}: all flat"
        )
        return RiskDecision(
            approved=pd.Series(0.0, index=proposed.index),
            reasons=(RiskReason(RiskLimit.DRAWDOWN_HALT, None, 0.0, detail),),
            halted=True,
        )

    approved = proposed.astype(float).copy()
    reasons: list[RiskReason] = []

    # Per-instrument cap: |w_i| <= cap, sign kept.
    for symbol in approved.index:
        weight = approved[symbol]
        if pd.notna(weight) and abs(weight) > params.per_instrument_cap:
            scale = params.per_instrument_cap / abs(weight)
            approved[symbol] = weight * scale
            reasons.append(
                RiskReason(
                    RiskLimit.PER_INSTRUMENT,
                    symbol,
                    scale,
                    f"|{weight:.6g}| > per-instrument cap {params.per_instrument_cap:.6g}",
                )
            )

    # Asset-class cap: gross Σ|w| within each class <= cap, members scaled pro-rata.
    for asset_class in sorted(set(asset_class_of.values())):
        members = [s for s in approved.index if asset_class_of[s] == asset_class]
        gross = float(approved[members].abs().sum())  # NaN contributes nothing
        if gross > params.asset_class_cap:
            scale = params.asset_class_cap / gross
            approved[members] = approved[members] * scale
            reasons.append(
                RiskReason(
                    RiskLimit.ASSET_CLASS,
                    None,
                    scale,
                    f"{asset_class} gross {gross:.6g} > class cap "
                    f"{params.asset_class_cap:.6g}",
                )
            )

    # Portfolio-heat cap: total gross Σ|w| <= cap, everything scaled pro-rata.
    gross = float(approved.abs().sum())
    if gross > params.portfolio_heat_cap:
        scale = params.portfolio_heat_cap / gross
        approved = approved * scale
        reasons.append(
            RiskReason(
                RiskLimit.PORTFOLIO_HEAT,
                None,
                scale,
                f"portfolio gross {gross:.6g} > heat cap {params.portfolio_heat_cap:.6g}",
            )
        )

    return RiskDecision(approved=approved, reasons=tuple(reasons), halted=False)
