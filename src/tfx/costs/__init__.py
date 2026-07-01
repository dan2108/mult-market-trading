"""Deterministic, pessimistic all-in cost model (BUILD.md Step 2).

**Units contract (load-bearing — the backtest inherits it):** every cost is a NON-NEGATIVE
**return fraction of equity**. ``size`` is a position weight (notional as a fraction of account
equity — exactly what `sizing` outputs), so cost = per-unit-notional rate x |size| and scales
linearly with size, while staying currency-agnostic (no FX conversion needed).

**Backtest integration contract (documented; backtest not built yet):**
  * On every FILL: charge the per-fill components (spread + commission + slippage) on the traded
    weight with ``nights_held=0`` -> ``CostModel.apply(instrument, side, size, price)``.
  * On every OVERNIGHT HOLD with no trade: charge ``breakdown(...).swap`` on the held weight.
  * Costs are always SUBTRACTED from PnL. A do-nothing run has flat equity; churning bleeds.

**Pessimism:** modeled spread is floored per instrument (no zero-cost trades); positive carry is
clamped to zero in the all-in cost (the signed value survives in ``CostBreakdown.swap_raw`` for
later broker reconciliation). All params are PROVISIONAL until reconciled (see params.py).
"""

from .model import CostModel
from .params import (
    ASSET_CLASS_DEFAULTS,
    INSTRUMENT_OVERRIDES,
    default_cost_params,
    get_params,
)
from .schema import CostBreakdown, CostParams, Side

__all__ = [
    "ASSET_CLASS_DEFAULTS",
    "INSTRUMENT_OVERRIDES",
    "CostBreakdown",
    "CostModel",
    "CostParams",
    "Side",
    "default_cost_params",
    "get_params",
]
