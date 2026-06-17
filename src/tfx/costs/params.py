"""PROVISIONAL cost parameters and the per-instrument resolver.

Every number here is a deliberately PESSIMISTIC, retail-broker-magnitude placeholder, marked
``provisional=True``. They MUST be reconciled against real broker costs before the validate gate
is trusted (BUILD.md Step 2 "done" criterion). Spread/floor are in each instrument's price units;
commission/slippage/swap are in bps or fractions (see CostParams).

Resolution mirrors tfx.instruments.get_instrument: per-symbol override first, then an asset-class
default, else raise — never a silent zero-cost fallthrough.
"""

from __future__ import annotations

from ..instruments import AssetClass, Instrument, default_symbols, get_instrument
from .schema import CostParams

# Generic pessimistic fallbacks for any symbol not explicitly overridden below.
ASSET_CLASS_DEFAULTS: dict[AssetClass, CostParams] = {
    AssetClass.FX: CostParams(
        spread=0.00030, spread_floor=0.00025, commission_bps=0.0,
        slippage_frac=0.00005, swap_long_bps=2.0, swap_short_bps=2.0,
    ),
    AssetClass.EQUITY_INDEX: CostParams(
        spread=2.0, spread_floor=1.5, commission_bps=0.0,
        slippage_frac=0.00005, swap_long_bps=1.5, swap_short_bps=1.2,
    ),
    AssetClass.COMMODITY: CostParams(
        spread=0.50, spread_floor=0.40, commission_bps=0.0,
        slippage_frac=0.00006, swap_long_bps=2.0, swap_short_bps=1.8,
    ),
    AssetClass.EQUITY: CostParams(
        spread=0.05, spread_floor=0.02, commission_bps=1.0,
        slippage_frac=0.00006, swap_long_bps=1.8, swap_short_bps=1.8,
    ),
    AssetClass.CRYPTO: CostParams(
        spread=0.0010, spread_floor=0.0005, commission_bps=10.0,
        slippage_frac=0.00020, swap_long_bps=5.0, swap_short_bps=5.0,
    ),
}

# Per-instrument overrides for the v1 basket (provisional, pessimistic).
INSTRUMENT_OVERRIDES: dict[str, CostParams] = {
    "EUR/USD": CostParams(spread=0.00008, spread_floor=0.00006, commission_bps=0.0,
                          slippage_frac=0.00002, swap_long_bps=1.0, swap_short_bps=0.6),
    "AUD/JPY": CostParams(spread=0.015, spread_floor=0.012, commission_bps=0.0,
                          slippage_frac=0.00003, swap_long_bps=0.5, swap_short_bps=1.5),
    "GBP/CHF": CostParams(spread=0.00030, spread_floor=0.00025, commission_bps=0.0,
                          slippage_frac=0.00004, swap_long_bps=1.8, swap_short_bps=1.2),
    "S&P 500": CostParams(spread=0.6, spread_floor=0.5, commission_bps=0.0,
                          slippage_frac=0.00003, swap_long_bps=1.2, swap_short_bps=0.9),
    "DAX": CostParams(spread=1.5, spread_floor=1.2, commission_bps=0.0,
                      slippage_frac=0.00004, swap_long_bps=1.3, swap_short_bps=1.0),
    "Gold": CostParams(spread=0.35, spread_floor=0.30, commission_bps=0.0,
                       slippage_frac=0.00004, swap_long_bps=1.5, swap_short_bps=1.2),
    "Oil": CostParams(spread=0.05, spread_floor=0.04, commission_bps=0.0,
                      slippage_frac=0.00006, swap_long_bps=2.0, swap_short_bps=1.7),
}


def get_params(instrument: str | Instrument) -> CostParams:
    """Resolve cost params: per-symbol override, then asset-class default, else raise."""
    instr = instrument if isinstance(instrument, Instrument) else get_instrument(instrument)
    if instr.symbol in INSTRUMENT_OVERRIDES:
        return INSTRUMENT_OVERRIDES[instr.symbol]
    if instr.asset_class in ASSET_CLASS_DEFAULTS:
        return ASSET_CLASS_DEFAULTS[instr.asset_class]
    raise KeyError(f"No cost params for {instr.symbol!r} (asset_class {instr.asset_class}).")


def default_cost_params() -> dict[str, CostParams]:
    """Cost params for the default basket, keyed by canonical symbol."""
    return {symbol: get_params(symbol) for symbol in default_symbols()}
