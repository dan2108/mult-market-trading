"""Bar-by-bar backtest engine (BUILD.md Step 6).

Timing (the one convention, end to end):
- close[t]: the signal DECIDES a direction from data <= t; sizing turns it into a raw weight;
  risk vetoes/scales it against the account state AS OF t. That approved target becomes the
  pending order.
- close[t+1]: the pending order FILLS at the next bar's close -- never the same close the signal
  saw (BUILD.md 6.2). The position first earns the close[t+1] -> close[t+2] return. This is
  `shift_for_execution` made operational, and one bar MORE lag than the theoretical minimum --
  deliberately pessimistic, per the repo's bias.

Accounting (multiplicative, in return space):
- weights are fractions of current equity; costs are return fractions (the CostModel contract),
  so equity[t] = equity[t-1] * (1 + sum_i pnl_i[t] - sum_i costs_i[t]).
- pnl marks held positions close-to-close against the last KNOWN close, so a position carried
  across a padded bar (exchange holiday) earns its multi-day return when the market reopens.
- every fill pays spread + commission + slippage at the fill price; every overnight hold pays
  swap for the CALENDAR nights since the previous bar (a Fri->Mon weekday gap charges 3 nights;
  a 24/7 calendar charges 1 per bar -- same total, no free weekends).

Order handling:
- Where sizing has no decision (signal warm-up, sizing burn-in, padded bar), the engine proposes
  the CURRENT book weight for that leg instead of skipping it: risk must see and cap the whole
  book every bar -- if carried legs were invisible to the veto, stale legs from staggered
  warm-ups/holidays could stack past the heat cap unseen. An unchanged approved leg produces no
  order (delta 0), so this adds no churn. Halt/kill zeros are REAL 0.0 targets and trade to flat.
- An untradable bar cannot fill; the pending order for that instrument lapses (the next bar's
  fresh decision supersedes it -- no stale orders). A cap breach on a leg whose market is closed
  is therefore enforced at its next tradable bar, not silently ignored.

Determinism: pure function of (panel, params, cost_model); no wall-clock, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..core import risk as risk_module
from ..core import signal as signal_module
from ..core import sizing as sizing_module
from ..core.risk import AccountState, RiskParams
from ..core.signal import SignalParams
from ..core.sizing import SizingParams
from ..costs.model import CostModel
from ..costs.schema import Side
from ..data.align import FIELD_LEVEL, tradable_mask

# Parity bindings: the engine calls the shared core through these module attributes, bound once
# at import. Tests assert function-object identity (`engine.signal_compute is
# tfx.core.signal.compute`) so a quiet re-implementation can never satisfy the suite.
signal_compute = signal_module.compute
sizing_size = sizing_module.size
risk_check = risk_module.check
shift_for_execution = signal_module.shift_for_execution

TRADE_COLUMNS = (
    "timestamp", "instrument", "side", "delta_weight", "fill_price",
    "spread", "commission", "slippage", "total",
)
COST_COLUMNS = ("timestamp", "instrument", "component", "amount")
RISK_EVENT_COLUMNS = ("timestamp", "limit", "instrument", "scale", "detail", "halted")


class BacktestParams(BaseModel):
    """One frozen bundle: initial equity + the exact core params the run is evaluated under."""

    model_config = ConfigDict(frozen=True)

    initial_equity: float = Field(default=1.0, gt=0)
    signal: SignalParams = Field(default_factory=SignalParams)
    sizing: SizingParams = Field(default_factory=SizingParams)
    risk: RiskParams = Field(default_factory=RiskParams)


@dataclass(frozen=True)
class BacktestResult:
    """Everything downstream consumers (validate, report) need; nothing hidden in the engine.

    ``positions`` are the weights ON THE BOOKS at each bar's close (post-fill). ``attribution``
    rows sum to the bar's total portfolio return: attribution[t, i] = pnl_i - costs_i, and
    equity[t] = equity[t-1] * (1 + attribution.loc[t].sum()) -- the accounting identity the
    acceptance tests recompute independently.
    """

    equity_curve: pd.Series
    positions: pd.DataFrame
    approved: pd.DataFrame          # risk-approved targets decided at each bar (pre-shift)
    trades: pd.DataFrame
    costs: pd.DataFrame             # long form: (timestamp, instrument, component, amount)
    attribution: pd.DataFrame
    risk_events: pd.DataFrame


def run(
    panel: pd.DataFrame,
    params: BacktestParams | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Simulate the shared core over ``panel``. Deterministic; costs on every fill and hold."""
    params = params or BacktestParams()
    cost_model = cost_model or CostModel()

    directions = signal_compute(panel, params.signal)
    raw_targets = sizing_size(panel, directions, params.sizing)

    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    tradable = tradable_mask(panel)
    instruments: list[str] = list(close.columns)
    index = close.index
    n_bars, n_instruments = close.shape

    close_np = close.to_numpy(dtype=float)
    tradable_np = tradable.to_numpy(dtype=bool)

    equity = params.initial_equity
    peak = equity
    held = np.zeros(n_instruments)
    last_close = np.full(n_instruments, np.nan)
    pending: np.ndarray | None = None  # NaN entries = no order for that instrument

    equity_values = np.empty(n_bars)
    positions_np = np.empty((n_bars, n_instruments))
    approved_np = np.full((n_bars, n_instruments), np.nan)
    attribution_np = np.zeros((n_bars, n_instruments))
    trade_rows: list[tuple] = []
    cost_rows: list[tuple] = []
    risk_rows: list[tuple] = []

    prev_ts: pd.Timestamp | None = None
    for t in range(n_bars):
        ts = index[t]
        bar_close = close_np[t]
        pnl = np.zeros(n_instruments)
        costs = np.zeros(n_instruments)

        # 1) mark held positions on today's close vs the last KNOWN close (padded bars defer
        #    their return to the reopen -- the position was held throughout).
        for i in range(n_instruments):
            if held[i] != 0.0 and not np.isnan(bar_close[i]) and not np.isnan(last_close[i]):
                pnl[i] = held[i] * (bar_close[i] / last_close[i] - 1.0)

        # 2) swap on every position held overnight: calendar nights since the previous bar.
        if prev_ts is not None:
            nights = int((ts.normalize() - prev_ts.normalize()).days)
            if nights > 0:
                for i in range(n_instruments):
                    if held[i] != 0.0 and not np.isnan(last_close[i]):
                        side = Side.LONG if held[i] > 0 else Side.SHORT
                        swap = cost_model.breakdown(
                            instruments[i], side, float(held[i]), float(last_close[i]),
                            nights_held=nights,
                        ).swap
                        if swap != 0.0:
                            costs[i] += swap
                            cost_rows.append((ts, instruments[i], "swap", swap))

        # 3) fill the pending order (decided at the previous close) at TODAY's close. A fill at
        #    the mark price has no same-bar pnl; it starts earning from the NEXT return.
        if pending is not None:
            for i in range(n_instruments):
                target = pending[i]
                if np.isnan(target) or not tradable_np[t, i]:
                    continue  # no decision, or nothing to fill against -- position carries
                delta = target - held[i]
                if delta == 0.0:
                    continue
                fill_price = float(bar_close[i])
                side = Side.LONG if delta > 0 else Side.SHORT
                breakdown = cost_model.breakdown(
                    instruments[i], side, float(delta), fill_price, nights_held=0
                )
                fill_cost = breakdown.spread + breakdown.commission + breakdown.slippage
                costs[i] += fill_cost
                held[i] = target
                trade_rows.append(
                    (ts, instruments[i], side.value, float(delta), fill_price,
                     breakdown.spread, breakdown.commission, breakdown.slippage, fill_cost)
                )
                cost_rows.append((ts, instruments[i], "fill", fill_cost))

        for i in range(n_instruments):
            if not np.isnan(bar_close[i]):
                last_close[i] = bar_close[i]

        # 4) multiplicative equity update; the account truth risk sees next.
        attribution_np[t] = pnl - costs
        equity *= 1.0 + float(attribution_np[t].sum())
        if equity <= 0.0:
            raise ValueError(
                f"equity wiped out at {ts.date()} -- leverage/vol params are unsafe for this "
                f"panel; refusing to continue with a negative account"
            )
        peak = max(peak, equity)
        equity_values[t] = equity
        positions_np[t] = held

        # 5) decide the NEXT bar's order from today's sized targets and today's account state.
        #    Legs sizing has no decision for are proposed AT their current book weight, so the
        #    veto always sees (and can cap) the whole book, carried legs included.
        raw = raw_targets.iloc[t].to_numpy(dtype=float)
        proposal = np.where(np.isnan(raw), held, raw)
        decision = risk_check(
            pd.Series(proposal, index=instruments),
            AccountState(equity=equity, peak_equity=peak),
            params.risk,
        )
        approved_np[t] = decision.approved.to_numpy(dtype=float)
        for reason in decision.reasons:
            risk_rows.append(
                (ts, reason.limit.value, reason.instrument, reason.scale, reason.detail,
                 decision.halted)
            )
        pending = approved_np[t]
        prev_ts = ts

    positions = pd.DataFrame(positions_np, index=index, columns=instruments)
    approved = pd.DataFrame(approved_np, index=index, columns=instruments)

    # Internal parity reconciliation: the loop's positions must be exactly the shared
    # `shift_for_execution` timing rule plus the carry rule (NaN / untradable -> hold). If the
    # loop ever drifts from the shared helper, this trips before any result leaves the engine.
    shifted = shift_for_execution(approved)
    fillable = shifted.notna() & tradable
    expected = shifted.where(fillable).ffill().fillna(0.0)
    if not np.allclose(positions.to_numpy(), expected.to_numpy(), rtol=0.0, atol=0.0):
        raise AssertionError(
            "engine positions diverged from shift_for_execution(approved) + carry -- "
            "timing-parity invariant broken"
        )

    return BacktestResult(
        equity_curve=pd.Series(equity_values, index=index, name="equity"),
        positions=positions,
        approved=approved,
        trades=pd.DataFrame(trade_rows, columns=list(TRADE_COLUMNS)),
        costs=pd.DataFrame(cost_rows, columns=list(COST_COLUMNS)),
        attribution=pd.DataFrame(attribution_np, index=index, columns=instruments),
        risk_events=pd.DataFrame(risk_rows, columns=list(RISK_EVENT_COLUMNS)),
    )
