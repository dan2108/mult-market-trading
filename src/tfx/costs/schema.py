"""Cost-model schema: side, per-instrument params, and an itemised cost breakdown.

Units contract (see also tfx.costs.__init__): every cost is a **return fraction of notional** —
dimensionless, currency-agnostic — so costs compose across instruments quoted in different
currencies and plug straight into a return-space backtest. A cost is always a **non-negative**
number the caller SUBTRACTS from PnL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class CostParams(BaseModel):
    """Per-instrument cost parameters. Immutable. All values are PROVISIONAL until reconciled
    against a real broker (see tfx.costs.params)."""

    model_config = ConfigDict(frozen=True)

    spread: float = Field(ge=0.0)         # full quoted spread, in the instrument's PRICE units
    spread_floor: float = Field(ge=0.0)   # pessimism floor on the full spread, same price units
    commission_bps: float = Field(ge=0.0)  # per-fill commission, bps of notional
    slippage_frac: float = Field(ge=0.0)  # adverse slippage, fraction of notional, per fill
    swap_long_bps: float                  # nightly financing for a LONG, bps of notional
    swap_short_bps: float                 # nightly financing for a SHORT, bps of notional
    provisional: bool = True              # True until broker-reconciled


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised cost for one event, all fields in return-fraction units (non-negative, except
    ``swap_raw`` which keeps the SIGNED financing for later reconciliation)."""

    instrument: str
    spread: float
    commission: float
    slippage: float
    swap: float          # financing actually charged (carry credits clamped to 0)
    swap_raw: float      # signed financing before the pessimistic clamp (for reconciliation)
    provisional: bool

    @property
    def total(self) -> float:
        """All-in cost = spread + commission + slippage + (clamped) swap."""
        return self.spread + self.commission + self.slippage + self.swap
