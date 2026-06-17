"""Acceptance tests for the `costs` module (BUILD.md Step 2, tests 1-9) + unit tests.

Costs are return fractions of equity; `size` is a position weight. Floats compared with a tight
tolerance (green != strict elsewhere, but cost arithmetic is exact within float here).
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from tfx.costs import CostModel, CostParams, Side, default_cost_params, get_params
from tfx.costs.model import CostModel as CostModelClass
from tfx.instruments import default_symbols


def approx(value: float):
    return pytest.approx(value, rel=1e-12, abs=1e-15)


def _params(**overrides) -> CostParams:
    base = dict(
        spread=1.0, spread_floor=0.5, commission_bps=2.0, slippage_frac=0.0,
        swap_long_bps=1.0, swap_short_bps=3.0,
    )
    base.update(overrides)
    return CostParams(**base)


# 1 — Round-trip at a flat price loses exactly the modeled round-trip cost (spread + commission).
def test_01_roundtrip_flat_price_costs_spread_plus_commission():
    params = _params(slippage_frac=0.0)
    model = CostModel({"X": params})
    buy = model.apply("X", Side.LONG, 10.0, 100.0)
    sell = model.apply("X", Side.SHORT, 10.0, 100.0)
    effective_spread = max(params.spread, params.spread_floor)
    expected = 10.0 * (effective_spread / 100.0 + 2 * params.commission_bps * 1e-4)
    assert buy + sell == approx(expected)
    assert buy + sell == approx(0.104)


# 2 — Swap accrues per night; N nights -> N x nightly swap; sign-correct (long vs short differ).
def test_02_swap_accrues_per_night_sign_correct():
    model = CostModel({"X": _params()})
    one = model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=1).swap
    five = model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=5).swap
    assert five == approx(5 * one)

    long_swap = model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=1).swap
    short_swap = model.breakdown("X", Side.SHORT, 10.0, 100.0, nights_held=1).swap
    assert long_swap != short_swap
    assert short_swap == approx(3 * long_swap)  # 3.0 vs 1.0 bps


# 3 — Cost scales linearly with size.
def test_03_cost_linear_in_size():
    model = CostModel({"X": _params(slippage_frac=0.001)})
    small = model.apply("X", Side.LONG, 10.0, 100.0, nights_held=2)
    big = model.apply("X", Side.LONG, 20.0, 100.0, nights_held=2)
    assert big == approx(2 * small)


# 4 — Slippage is always adverse; never improves a fill.
def test_04_slippage_always_adverse():
    no_slip = CostModel({"X": _params(slippage_frac=0.0)})
    with_slip = CostModel({"X": _params(slippage_frac=0.001)})
    for side in (Side.LONG, Side.SHORT):
        assert with_slip.breakdown("X", side, 10.0, 100.0).slippage >= 0.0
        assert with_slip.apply("X", side, 10.0, 100.0) > no_slip.apply("X", side, 10.0, 100.0)


# 5 — Pessimism floor: modeled spread >= configured floor; no zero-cost trades.
def test_05_pessimism_floor_no_zero_cost():
    model = CostModel({"X": _params(spread=0.0, spread_floor=0.5, commission_bps=0.0)})
    breakdown = model.breakdown("X", Side.LONG, 10.0, 100.0)
    assert breakdown.spread == approx(10.0 * (0.5 / 2) / 100.0)  # floor used, not 0
    assert model.apply("X", Side.LONG, 10.0, 100.0) > 0.0
    for symbol in default_symbols():
        assert get_params(symbol).spread_floor > 0.0


# 6 — Per-asset-class params resolve correctly (FX swap != index != commodity); overrides win.
def test_06_asset_class_resolution_and_override():
    fx = get_params("EUR/USD")
    index = get_params("S&P 500")
    commodity = get_params("Gold")
    assert fx.swap_long_bps != index.swap_long_bps
    assert index.swap_long_bps != commodity.swap_long_bps
    assert fx.swap_long_bps != commodity.swap_long_bps

    override = _params(swap_long_bps=99.0)
    model = CostModel({"EUR/USD": override})
    swap = model.breakdown("EUR/USD", Side.LONG, 10.0, 1.08, nights_held=1).swap
    assert swap == approx(10.0 * 99.0 * 1e-4)

    with pytest.raises(KeyError):
        get_params("NOPE/NONE")


# 7 — No look-ahead: cost uses only params known at t (no time/series argument; static, frozen).
def test_07_no_lookahead_structural():
    signature = set(inspect.signature(CostModelClass.apply).parameters)
    assert not (signature & {"date", "t", "time", "panel", "series", "history", "future"})

    params = get_params("EUR/USD")
    with pytest.raises((ValidationError, TypeError)):  # frozen CostParams is immutable
        params.spread = 0.0  # type: ignore[misc]

    model = CostModel()
    assert model.apply("EUR/USD", Side.LONG, 10.0, 1.08, 2) == model.apply(
        "EUR/USD", Side.LONG, 10.0, 1.08, 2
    )


# 8 — Deterministic; apply() == breakdown().total (units consistent).
def test_08_deterministic_and_total_consistent():
    model = CostModel()
    first = model.apply("Gold", Side.SHORT, 7.5, 1900.0, 4)
    for _ in range(3):
        assert model.apply("Gold", Side.SHORT, 7.5, 1900.0, 4) == first
    breakdown = model.breakdown("Gold", Side.SHORT, 7.5, 1900.0, 4)
    assert first == approx(breakdown.total)


# 9 — Fixture trade matches a hand-computed cost exactly (see tests/fixtures/costs_handcalc.md).
def test_09_handcomputed_fixture_trade():
    hand = _params(slippage_frac=0.0010)  # spread=1.0, floor=0.5, comm_bps=2.0, swap 1.0/3.0
    model = CostModel({"X": hand})

    breakdown = model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=0)
    assert breakdown.spread == approx(0.05)
    assert breakdown.commission == approx(0.002)
    assert breakdown.slippage == approx(0.010)
    assert breakdown.swap == approx(0.0)
    assert breakdown.total == approx(0.062)
    assert model.apply("X", Side.LONG, 10.0, 100.0) == approx(0.062)

    assert model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=3).swap == approx(0.003)
    assert model.breakdown("X", Side.SHORT, 10.0, 100.0, nights_held=3).swap == approx(0.009)


# --- additional unit tests -------------------------------------------------------------------
def test_carry_credit_clamped_but_recorded():
    credit = _params(swap_long_bps=-2.0)  # a financing credit
    model = CostModel({"X": credit})
    breakdown = model.breakdown("X", Side.LONG, 10.0, 100.0, nights_held=1)
    assert breakdown.swap == 0.0          # clamped: a credit never reduces cost
    assert breakdown.swap_raw < 0.0       # signed value retained for reconciliation
    assert model.apply("X", Side.LONG, 10.0, 100.0, nights_held=1) >= 0.0


def test_invalid_inputs_raise():
    model = CostModel()
    with pytest.raises(ValueError):
        model.apply("EUR/USD", Side.LONG, 0.0, 1.08)
    with pytest.raises(ValueError):
        model.apply("EUR/USD", Side.LONG, 10.0, 0.0)
    with pytest.raises(ValueError):
        model.apply("EUR/USD", Side.LONG, 10.0, 1.08, nights_held=-1)


def test_defaults_cover_basket_and_are_provisional():
    params = default_cost_params()
    for symbol in default_symbols():
        assert symbol in params
        assert params[symbol].provisional is True
