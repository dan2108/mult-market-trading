"""The deterministic all-in cost model.

Pure arithmetic on its arguments — no I/O, no wall-clock, no RNG, no hidden state — so it is
trivially deterministic and free of look-ahead (it never receives a time index or future data).
See :mod:`tfx.costs.schema` for the units contract and :mod:`tfx.costs.params` for parameters.
"""

from __future__ import annotations

from ..instruments import Instrument
from .params import default_cost_params, get_params
from .schema import CostBreakdown, CostParams, Side


class CostModel:
    """Applies spread + commission + slippage to a fill, and swap to an overnight hold.

    ``size`` is a position weight (notional as a fraction of account equity), so the returned
    cost is a fraction of equity and scales linearly with size. ``side`` only selects the swap
    rate; per-fill costs are direction-symmetric.
    """

    def __init__(self, params: dict[str, CostParams] | None = None) -> None:
        self._params: dict[str, CostParams] = (
            dict(params) if params is not None else default_cost_params()
        )

    def _resolve(self, instrument: str | Instrument) -> CostParams:
        symbol = instrument.symbol if isinstance(instrument, Instrument) else instrument
        if symbol in self._params:
            return self._params[symbol]
        return get_params(instrument)

    def breakdown(
        self,
        instrument: str | Instrument,
        side: Side | str,
        size: float,
        price: float,
        nights_held: int = 0,
    ) -> CostBreakdown:
        """Itemised cost for one event. ``apply()`` returns ``.total``."""
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        if size == 0:
            raise ValueError("size must be non-zero")
        if nights_held < 0:
            raise ValueError(f"nights_held must be >= 0, got {nights_held}")

        side = Side(side)
        params = self._resolve(instrument)
        symbol = instrument.symbol if isinstance(instrument, Instrument) else instrument
        qty = abs(float(size))

        effective_spread = max(params.spread, params.spread_floor)  # pessimism floor
        spread = qty * (effective_spread / 2.0) / price            # half-spread per fill
        commission = qty * (params.commission_bps * 1e-4)
        slippage = qty * params.slippage_frac                      # always adverse (>= 0)

        swap_bps = params.swap_long_bps if side is Side.LONG else params.swap_short_bps
        swap_raw = qty * (swap_bps * 1e-4) * nights_held           # signed (carry can be a credit)
        swap = qty * (max(0.0, swap_bps) * 1e-4) * nights_held     # clamped: never a credit

        return CostBreakdown(
            instrument=symbol,
            spread=spread,
            commission=commission,
            slippage=slippage,
            swap=swap,
            swap_raw=swap_raw,
            provisional=params.provisional,
        )

    def apply(
        self,
        instrument: str | Instrument,
        side: Side | str,
        size: float,
        price: float,
        nights_held: int = 0,
    ) -> float:
        """All-in cost as a non-negative fraction of equity; the caller subtracts it from PnL."""
        return self.breakdown(instrument, side, size, price, nights_held).total
