"""Acceptance tests for the `backtest` engine (BUILD.md Step 6, tests 1-9) + unit tests.

Real basket symbols are used throughout (risk clusters by asset class and costs resolve per
instrument, and both raise on unknown symbols). The no-look-ahead test is the cardinal one; the
oracle test re-implements the engine mechanics independently in test code, the same
dual-implementation pattern the sizing tests use (see tests/fixtures/backtest_handcalc.md).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import tfx.backtest.engine as engine
import tfx.core.risk
import tfx.core.signal
import tfx.core.sizing
from tfx.backtest import BacktestParams, run
from tfx.core.risk import AccountState, RiskParams
from tfx.core.signal import SignalParams
from tfx.core.sizing import SizingParams
from tfx.costs.model import CostModel
from tfx.costs.schema import Side
from tfx.data.align import FIELD_LEVEL, INSTRUMENT_LEVEL, tradable_mask
from tfx.data.schema import TIMESTAMP_INDEX_NAME, QualityFlag

FX, COM = "EUR/USD", "Gold"

# Risk levels that never bind: engine tests isolate engine mechanics; risk binding is test 8.
PASSTHROUGH_RISK = RiskParams(
    per_instrument_cap=100.0, asset_class_cap=100.0, portfolio_heat_cap=100.0,
    drawdown_halt=0.99,
)


def _panel(
    closes: dict[str, list[float]], pad: dict[str, set[int]] | None = None, freq: str = "B"
) -> pd.DataFrame:
    """(close, quality) panel shaped like align()'s output. freq='B' (business days) so a
    Friday->Monday hold spans 3 calendar nights, exactly like the real weekday calendars."""
    pad = pad or {}
    n = len(next(iter(closes.values())))
    index = pd.date_range(
        "2024-01-01", periods=n, freq=freq, tz="UTC", name=TIMESTAMP_INDEX_NAME
    )
    frames = {}
    for symbol, values in closes.items():
        padded = pad.get(symbol, set())
        close = [float("nan") if i in padded else float(v) for i, v in enumerate(values)]
        quality = [int(QualityFlag.ALIGN_PAD) if i in padded else 0 for i in range(n)]
        frames[symbol] = pd.DataFrame({"close": close, "quality": quality}, index=index)
    return pd.concat(
        frames.values(), axis=1, keys=list(frames.keys()), names=[INSTRUMENT_LEVEL, FIELD_LEVEL]
    )


def _params(risk: RiskParams | None = None) -> BacktestParams:
    """Fast-warmup params: 1-bar lookback, 1-bar vol halflife -> first fill at bar 2."""
    return BacktestParams(
        signal=SignalParams(lookbacks=(1,)),
        sizing=SizingParams(ewma_halflife=1.0, target_vol=0.10, leverage_cap=100.0),
        risk=risk or PASSTHROUGH_RISK,
    )


def _wiggly(start: float, step: float, n: int) -> list[float]:
    """Deterministic non-trivial path: alternating up/down with drift, no RNG."""
    values = [start]
    for i in range(n - 1):
        factor = 1 + step * (1.6 if i % 3 else -1.0) * (1 + 0.1 * (i % 5))
        values.append(values[-1] * factor)
    return values


# 1 — No look-ahead (cardinal): future bars must not change any historical trade or fill.
def test_01_no_lookahead_perturb_future():
    fx = _wiggly(1.00, 0.004, 14)
    com = _wiggly(1900.0, 0.006, 14)
    base = run(_panel({FX: fx, COM: com}), _params())

    # future values differ but stay survivable (a synthetic 90% one-bar crash on a leveraged
    # book would legitimately wipe the account -- not what this test is about)
    tail_factors = (1.05, 0.92, 1.08, 0.95, 1.03, 0.97)
    scrambled_tail = [v * f for v, f in zip(fx[8:], tail_factors, strict=True)]
    scrambled = run(
        _panel({FX: fx[:8] + scrambled_tail, COM: com[:8] + list(com[8:])}), _params()
    )
    longer = run(
        _panel({
            FX: fx + [fx[-1] * 1.04, fx[-1] * 0.98],
            COM: com + [com[-1] * 0.95, com[-1] * 1.02],
        }),
        _params(),
    )

    cut = base.equity_curve.index[8]
    for other in (scrambled, longer):
        pd.testing.assert_series_equal(
            base.equity_curve.iloc[:8], other.equity_curve.iloc[:8]
        )
        base_trades = base.trades[base.trades["timestamp"] < cut].reset_index(drop=True)
        other_trades = other.trades[other.trades["timestamp"] < cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(base_trades, other_trades)


# 2 — Execution timing: fills at the NEXT bar's close, never the decision bar's close.
def test_02_fill_at_next_bar_close():
    closes = [100.0] * 4 + [101.0, 102.01, 103.03, 104.06]
    panel = _panel({COM: closes})
    result = run(panel, _params())

    # constant prices: zero vol -> NaN weight -> no order. First real decision happens on the
    # first moving bar (t=4); its fill lands at t=5, at close[5].
    index = result.equity_curve.index
    assert not result.trades.empty
    first = result.trades.iloc[0]
    assert first["timestamp"] == index[5]
    assert first["fill_price"] == pytest.approx(closes[5])
    assert (result.trades["timestamp"] >= index[5]).all()
    assert (result.positions.loc[: index[4]] == 0.0).to_numpy().all()


# 3 — Costs: a do-nothing run has flat equity; churning bleeds; weekend holds pay 3 swap nights.
def test_03_costs_do_nothing_churn_and_weekend_swap():
    # (a) do-nothing: constant prices -> no decisions ever -> exactly flat equity, zero rows
    flat = run(_panel({FX: [1.1] * 10, COM: [1900.0] * 10}), _params())
    assert (flat.equity_curve == 1.0).all()
    assert flat.trades.empty and flat.costs.empty

    # (b) churn: a period-4 path (up, up, down, down) is maximally adverse to 1-bar momentum
    # UNDER THIS ENGINE'S TIMING -- the position decided at t fills at t+1 and first earns
    # ret[t+2] = -ret[t] -- so the strategy is always positioned the wrong way AND pays fills
    # on every flip: equity must bleed below start, every fill costing > 0
    factors = [1.01, 1.01, 1 / 1.01, 1 / 1.01] * 4
    path = [1.0]
    for factor in factors:
        path.append(path[-1] * factor)
    churn = run(_panel({FX: path}), _params())
    assert churn.equity_curve.iloc[-1] < 1.0
    assert len(churn.trades) > 5
    assert (churn.trades["total"] > 0).all()

    # (c) weekend swap: a steady long held across Fri->Mon pays exactly 3 nights at the long
    # rate (EUR/USD swap_long_bps = 1.0); a plain weekday night pays exactly 1
    trend = run(_panel({FX: [1.0 * 1.01**i for i in range(15)]}), _params())
    swaps = trend.costs[trend.costs["component"] == "swap"].set_index("timestamp")
    positions = trend.positions[FX]
    mondays = [ts for ts in swaps.index if ts.weekday() == 0]
    tuesdays = [ts for ts in swaps.index if ts.weekday() == 1]
    assert mondays and tuesdays
    for ts, nights in ((mondays[-1], 3), (tuesdays[-1], 1)):
        held_over = positions[positions.index < ts].iloc[-1]
        expected = abs(held_over) * (1.0 * 1e-4) * nights
        assert swaps.loc[ts, "amount"] == pytest.approx(expected, rel=1e-12)


# 4 — Accounting integrity: equity reconciles bar-by-bar from the published frames alone.
def test_04_accounting_identity_recomputed_independently():
    fx = _wiggly(1.00, 0.005, 20)
    com = _wiggly(1900.0, 0.007, 20)
    panel = _panel({FX: fx, COM: com})
    result = run(panel, _params())

    # equity[t] = initial * prod(1 + sum_i attribution[t, i])
    recomputed = 1.0 * (1.0 + result.attribution.sum(axis=1)).cumprod()
    np.testing.assert_allclose(
        result.equity_curve.to_numpy(), recomputed.to_numpy(), rtol=1e-12
    )

    # attribution + costs = raw close-to-close pnl of yesterday's position (no pads here)
    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    returns = close / close.shift(1) - 1.0
    costs_by_bar = (
        result.costs.pivot_table(
            index="timestamp", columns="instrument", values="amount", aggfunc="sum"
        )
        .reindex(index=result.equity_curve.index, columns=[FX, COM])
        .fillna(0.0)
    )
    pnl = result.attribution + costs_by_bar
    expected_pnl = (result.positions.shift(1) * returns).fillna(0.0)
    np.testing.assert_allclose(
        pnl.to_numpy(), expected_pnl.to_numpy(), rtol=0, atol=1e-15
    )

    # the trade log and the cost ledger agree on every fill
    fills = result.costs[result.costs["component"] == "fill"]
    assert fills["amount"].sum() == pytest.approx(result.trades["total"].sum(), rel=1e-12)
    assert len(fills) == len(result.trades)


# 5 — Parity: the engine binds the EXACT shared-core functions, and its positions are exactly
# shift_for_execution(approved) + the carry rule.
def test_05_parity_function_identity_and_timing_reconstruction():
    assert engine.signal_compute is tfx.core.signal.compute
    assert engine.sizing_size is tfx.core.sizing.size
    assert engine.risk_check is tfx.core.risk.check
    assert engine.shift_for_execution is tfx.core.signal.shift_for_execution

    fx = _wiggly(1.00, 0.004, 16)
    panel = _panel({FX: fx, COM: _wiggly(1900.0, 0.006, 16)}, pad={COM: {6}})
    result = run(panel, _params())

    shifted = tfx.core.signal.shift_for_execution(result.approved)
    fillable = shifted.notna() & tradable_mask(panel)
    expected = shifted.where(fillable).ffill().fillna(0.0)
    pd.testing.assert_frame_equal(result.positions, expected)


# 6 — Deterministic: identical inputs -> identical trade log and equity curve.
def test_06_deterministic():
    panel = _panel({FX: _wiggly(1.00, 0.004, 14), COM: _wiggly(1900.0, 0.006, 14)})
    first = run(panel, _params())
    second = run(panel, _params())
    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)
    pd.testing.assert_frame_equal(first.trades, second.trades)
    pd.testing.assert_frame_equal(first.positions, second.positions)
    pd.testing.assert_frame_equal(first.costs, second.costs)


# 7 — Padded bars: no trading a non-existent bar; positions carry across the gap.
def test_07_padded_bars_no_trade_position_carries():
    n = 14
    fx = _wiggly(1.00, 0.004, n)
    com = [1900.0 * 1.01**i for i in range(n)]  # steady uptrend -> steadily long
    panel = _panel({FX: fx, COM: com}, pad={COM: {8}})
    result = run(panel, _params())

    index = result.equity_curve.index
    gold_trades = result.trades[result.trades["instrument"] == COM]
    assert not (gold_trades["timestamp"] == index[8]).any()
    assert result.positions.loc[index[8], COM] == result.positions.loc[index[7], COM]
    # the pnl part of Gold's attribution is zero on the padded bar (only swap may accrue)
    gold_costs = result.costs[
        (result.costs["instrument"] == COM) & (result.costs["timestamp"] == index[8])
    ]["amount"].sum()
    assert result.attribution.loc[index[8], COM] == pytest.approx(-gold_costs, abs=1e-15)
    # FX keeps trading normally through Gold's holiday
    assert ((result.trades["instrument"] == FX) & (result.trades["timestamp"] == index[8])).any()


# 8 — Risk veto respected: caps bind on the books; a drawdown breach flattens and stays flat.
def test_08_risk_veto_and_drawdown_halt_latch():
    fx = _wiggly(1.00, 0.004, 16)
    tight = RiskParams(
        per_instrument_cap=0.05, asset_class_cap=100.0, portfolio_heat_cap=100.0,
        drawdown_halt=0.99,
    )
    capped = run(_panel({FX: fx}), _params(risk=tight))
    assert (capped.positions.abs() <= 0.05 + 1e-12).to_numpy().all()
    assert (capped.risk_events["limit"] == "per_instrument").any()

    # ramp up (get long), then crash hard enough to breach a 5% drawdown halt
    path = [1.0 * 1.01**i for i in range(8)] + [1.0 * 1.01**7 * 0.85**i for i in range(1, 7)]
    halt = RiskParams(
        per_instrument_cap=100.0, asset_class_cap=100.0, portfolio_heat_cap=100.0,
        drawdown_halt=0.05,
    )
    crashed = run(_panel({FX: path}), _params(risk=halt))
    events = crashed.risk_events
    assert (events["limit"] == "drawdown_halt").any()
    first_halt = events.loc[events["limit"] == "drawdown_halt", "timestamp"].iloc[0]
    index = crashed.equity_curve.index
    after = index[index.get_loc(first_halt) + 1 :]
    assert (crashed.positions.loc[after] == 0.0).to_numpy().all()  # flattened at the next bar
    # once flat, equity freezes and the halt latches (no re-entry for the rest of the run)
    flat_equity = crashed.equity_curve.loc[after]
    if len(after) > 1:
        assert flat_equity.nunique() == 1
        halted_bars = events.loc[events["limit"] == "drawdown_halt", "timestamp"]
        assert set(after) <= set(halted_bars)


# 9 — Hand-calc + oracle: exact expected trades/equity (tests/fixtures/backtest_handcalc.md).
def test_09_handcalc_and_oracle():
    closes = [round(1.0 * 1.01**i, 10) for i in range(9)]
    panel = _panel({FX: closes})
    params = _params()
    result = run(panel, params)
    index = result.equity_curve.index

    # --- hand calc (backtest_handcalc.md): first decision, first fill, equity after the fill
    w1 = 0.10 / (math.log(closes[1] / closes[0]) * math.sqrt(126.0))
    assert result.approved[FX].iloc[1] == pytest.approx(w1, rel=1e-12)
    first = result.trades.iloc[0]
    assert first["timestamp"] == index[2]
    assert first["fill_price"] == pytest.approx(closes[2], rel=1e-12)
    assert first["delta_weight"] == pytest.approx(w1, rel=1e-12)
    fill_cost = w1 * ((0.00008 / 2.0) / closes[2] + 0.00002)
    assert first["total"] == pytest.approx(fill_cost, rel=1e-12)
    assert result.equity_curve.iloc[2] == pytest.approx(1.0 - fill_cost, rel=1e-12)

    # --- oracle: independent re-implementation of the engine mechanics in test code
    directions = tfx.core.signal.compute(panel, params.signal)
    weights = tfx.core.sizing.size(panel, directions, params.sizing)
    cost_model = CostModel()
    equity, peak, held = 1.0, 1.0, 0.0
    pending = float("nan")
    expected_equity = []
    prev_ts = None
    for t, ts in enumerate(index):
        price, bar_return = closes[t], 0.0
        if held and t > 0:
            bar_return += held * (closes[t] / closes[t - 1] - 1.0)
            nights = (ts.normalize() - prev_ts.normalize()).days
            side = Side.LONG if held > 0 else Side.SHORT
            bar_return -= cost_model.breakdown(FX, side, held, closes[t - 1], nights).swap
        if not math.isnan(pending) and pending != held:
            delta = pending - held
            side = Side.LONG if delta > 0 else Side.SHORT
            breakdown = cost_model.breakdown(FX, side, delta, price, 0)
            bar_return -= breakdown.spread + breakdown.commission + breakdown.slippage
            held = pending
        equity *= 1.0 + bar_return
        peak = max(peak, equity)
        expected_equity.append(equity)
        raw_weight = weights[FX].iloc[t]
        proposal = held if math.isnan(raw_weight) else float(raw_weight)  # book-aware, as engine
        decision = tfx.core.risk.check(
            pd.Series({FX: proposal}),
            AccountState(equity=equity, peak_equity=peak),
            params.risk,
        )
        pending = float(decision.approved[FX])
        prev_ts = ts

    np.testing.assert_allclose(
        result.equity_curve.to_numpy(), np.array(expected_equity), rtol=1e-14
    )


# --- additional unit tests -------------------------------------------------------------------
def test_initial_equity_scales_linearly():
    panel = _panel({FX: _wiggly(1.00, 0.004, 12)})
    one = run(panel, _params())
    two = run(
        panel,
        BacktestParams(
            initial_equity=2.0, signal=SignalParams(lookbacks=(1,)),
            sizing=SizingParams(ewma_halflife=1.0, target_vol=0.10, leverage_cap=100.0),
            risk=PASSTHROUGH_RISK,
        ),
    )
    np.testing.assert_allclose(
        two.equity_curve.to_numpy(), 2.0 * one.equity_curve.to_numpy(), rtol=1e-14
    )


def test_one_bar_panel_no_crash():
    result = run(_panel({FX: [1.1]}), _params())
    assert result.equity_curve.iloc[0] == 1.0
    assert result.trades.empty


def test_params_frozen_and_unknown_symbol_raises():
    params = _params()
    with pytest.raises(ValidationError):
        params.initial_equity = 5.0  # type: ignore[misc]
    with pytest.raises(KeyError):
        run(_panel({"NOPE/NONE": [1.0, 1.1, 1.2]}), params)


def test_run_on_real_aligned_panel(cache_dir, manifest, symbols, window):
    """Integration: the real pulled panel, default-ish params, must produce a sane result."""
    from tfx.data.loader import load

    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir, manifest=manifest)
    params = BacktestParams(
        signal=SignalParams(lookbacks=(21, 63)),  # fits the ~2y fixture window
        sizing=SizingParams(),
        risk=RiskParams(),
    )
    result = run(panel, params)
    assert len(result.equity_curve) == len(panel.index)
    assert (result.equity_curve > 0).all()
    assert not result.trades.empty
    # every individual leg is a previously-approved weight, so the per-instrument cap holds on
    # EVERY bar; the portfolio heat cap holds on every bar where all legs could fill (on a
    # holiday, a reduce order for the closed leg lapses and is enforced at the reopen -- the
    # documented can't-trade-a-closed-market exception)
    assert (result.positions.abs() <= params.risk.per_instrument_cap + 1e-9).to_numpy().all()
    gross = result.positions.abs().sum(axis=1)
    all_tradable = tradable_mask(panel).all(axis=1)
    assert (gross[all_tradable] <= params.risk.portfolio_heat_cap + 1e-9).all()
    breach = gross[gross > params.risk.portfolio_heat_cap + 1e-9]
    assert (~all_tradable.loc[breach.index]).all()  # every breach bar has a closed leg
