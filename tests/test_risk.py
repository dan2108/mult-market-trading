"""Acceptance tests for the `risk` module (BUILD.md Step 5, tests 1-8) + unit tests.

The veto can only ever REDUCE risk: every stage multiplies by a scalar in [0, 1], so
|approved| <= |proposed| elementwise and a sign can never flip. Real basket symbols are used
throughout so the asset-class clustering is the real `tfx.instruments` lookup, not a stub.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tfx.core.risk import AccountState, RiskLimit, RiskParams, check

# Real basket symbols spanning three asset classes (fx, fx, index, commodity).
FX_A, FX_B, IDX, COM = "EUR/USD", "AUD/JPY", "S&P 500", "Gold"

HEALTHY = AccountState(equity=100_000.0, peak_equity=100_000.0)


def _limits(reasons) -> set[RiskLimit]:
    return {reason.limit for reason in reasons}


# 1 — Per-trade cap: no approved position risks more than the cap; oversized proposals scaled.
def test_01_per_instrument_cap_scales_to_exactly_cap():
    params = RiskParams(per_instrument_cap=0.5, asset_class_cap=100.0, portfolio_heat_cap=100.0)
    proposed = pd.Series({FX_A: 0.8, FX_B: -0.7, COM: 0.2})

    decision = check(proposed, HEALTHY, params)

    assert decision.approved[FX_A] == pytest.approx(0.5)
    assert decision.approved[FX_B] == pytest.approx(-0.5)  # sign kept
    assert decision.approved[COM] == pytest.approx(0.2)    # compliant -> untouched
    assert _limits(decision.reasons) == {RiskLimit.PER_INSTRUMENT}
    assert len(decision.reasons) == 2
    assert not decision.halted

    # a fully-compliant proposal passes through identically, with zero reasons
    compliant = pd.Series({FX_A: 0.3, FX_B: -0.2})
    untouched = check(compliant, HEALTHY, params)
    assert untouched.approved.equals(compliant.astype(float))
    assert untouched.reasons == ()


# 2 — Portfolio-heat cap: total gross <= cap; positions scaled pro-rata if exceeded.
def test_02_portfolio_heat_cap_pro_rata():
    params = RiskParams(per_instrument_cap=100.0, asset_class_cap=100.0, portfolio_heat_cap=1.0)
    proposed = pd.Series({FX_A: 1.2, IDX: -0.6, COM: 0.2})  # gross 2.0 -> scale 0.5

    decision = check(proposed, HEALTHY, params)

    assert decision.approved[FX_A] == pytest.approx(0.6)
    assert decision.approved[IDX] == pytest.approx(-0.3)
    assert decision.approved[COM] == pytest.approx(0.1)
    assert float(decision.approved.abs().sum()) == pytest.approx(1.0)  # lands exactly on cap
    assert _limits(decision.reasons) == {RiskLimit.PORTFOLIO_HEAT}
    assert decision.reasons[0].scale == pytest.approx(0.5)


# 3 — Correlation/exposure cap: clustered (same asset class) exposure <= cap, others untouched.
def test_03_asset_class_cap_scales_cluster_only():
    params = RiskParams(per_instrument_cap=100.0, asset_class_cap=0.6, portfolio_heat_cap=100.0)
    proposed = pd.Series({FX_A: 0.5, FX_B: -0.3, COM: 0.4})  # fx gross 0.8 > 0.6

    decision = check(proposed, HEALTHY, params)

    assert decision.approved[FX_A] == pytest.approx(0.5 * 0.75)   # scale 0.6/0.8
    assert decision.approved[FX_B] == pytest.approx(-0.3 * 0.75)
    assert decision.approved[COM] == pytest.approx(0.4)           # different class: untouched
    fx_gross = abs(decision.approved[FX_A]) + abs(decision.approved[FX_B])
    assert fx_gross == pytest.approx(0.6)
    assert _limits(decision.reasons) == {RiskLimit.ASSET_CLASS}


# 4 — Drawdown halt: breaching the DD limit flattens everything; a tick under does nothing.
# halt = 0.25 and equity/peak quarters are binary-exact, so the >= boundary is tested without
# float fuzz (0.20 vs 1 - 0.8 would differ by one ulp and test rounding, not semantics).
def test_04_drawdown_halt_boundary_both_sides():
    params = RiskParams(drawdown_halt=0.25)
    proposed = pd.Series({FX_A: 0.3, COM: np.nan})

    # one tick under the threshold: untouched (NaN stays NaN)
    under = AccountState(equity=75_001.0, peak_equity=100_000.0)
    ok = check(proposed, under, params)
    assert ok.approved[FX_A] == pytest.approx(0.3)
    assert pd.isna(ok.approved[COM])
    assert not ok.halted

    # at exactly the threshold (dd == 0.25 exactly) and past it: all flat, INCLUDING the NaN
    # (flat means flat)
    for equity in (75_000.0, 60_000.0):
        halted = check(proposed, AccountState(equity=equity, peak_equity=100_000.0), params)
        assert (halted.approved == 0.0).all()
        assert halted.halted
        assert _limits(halted.reasons) == {RiskLimit.DRAWDOWN_HALT}


# 5 — Kill switch: when set, all -> flat, no new trades, regardless of everything else.
def test_05_kill_switch_flattens_everything():
    state = AccountState(equity=100_000.0, peak_equity=100_000.0, kill_switch=True)
    proposed = pd.Series({FX_A: 0.4, FX_B: -0.2, COM: np.nan})

    decision = check(proposed, state, RiskParams())

    assert (decision.approved == 0.0).all()
    assert decision.halted
    assert _limits(decision.reasons) == {RiskLimit.KILL_SWITCH}
    assert len(decision.reasons) == 1


# 6 — Monotonic safety (property, deterministic grid): never increases a size, never flips a
# sign -- across proposals x account states x params, after ALL stages combined.
def test_06_monotonic_safety_property():
    values = [-2.5, -0.6, -0.5, -0.1, 0.0, 0.2, 0.5, 3.0, float("nan")]
    states = [
        HEALTHY,
        AccountState(equity=70_000.0, peak_equity=100_000.0),                    # dd-halted
        AccountState(equity=100_000.0, peak_equity=100_000.0, kill_switch=True),  # killed
    ]
    params_grid = [
        RiskParams(),
        RiskParams(
            per_instrument_cap=0.25, asset_class_cap=0.4, portfolio_heat_cap=0.5,
            drawdown_halt=0.1,
        ),
    ]

    for combo in itertools.product(values, repeat=3):
        proposed = pd.Series({FX_A: combo[0], FX_B: combo[1], COM: combo[2]})
        for state, params in itertools.product(states, params_grid):
            approved = check(proposed, state, params).approved
            for symbol in proposed.index:
                p, a = proposed[symbol], approved[symbol]
                if math.isnan(p):
                    assert math.isnan(a) or a == 0.0  # NaN passthrough, or forced flat on halt
                    continue
                assert not math.isnan(a)
                assert abs(a) <= abs(p) + 1e-12          # never increases
                assert a == 0.0 or math.copysign(1, a) == math.copysign(1, p)  # never flips


# 7 — Limits actually fire: one stress proposal breaches per-instrument + class + heat at once
# (halt states short-circuit, so they are proven in tests 4/5); every cap holds simultaneously.
def test_07_all_limits_fire_and_compose():
    params = RiskParams(per_instrument_cap=0.5, asset_class_cap=0.75, portfolio_heat_cap=1.0)
    proposed = pd.Series({FX_A: 0.8, FX_B: -0.4, COM: 0.3})

    decision = check(proposed, HEALTHY, params)

    assert _limits(decision.reasons) == {
        RiskLimit.PER_INSTRUMENT, RiskLimit.ASSET_CLASS, RiskLimit.PORTFOLIO_HEAT,
    }
    approved = decision.approved
    assert (approved.abs() <= params.per_instrument_cap + 1e-12).all()
    fx_gross = abs(approved[FX_A]) + abs(approved[FX_B])
    assert fx_gross <= params.asset_class_cap + 1e-12
    assert float(approved.abs().sum()) <= params.portfolio_heat_cap + 1e-12


# 8 — Pure & deterministic; reasons for every reduction; fixture matches the hand calc
# (tests/fixtures/risk_handcalc.md).
def test_08_handcomputed_fixture_slice_and_purity():
    params = RiskParams(per_instrument_cap=0.5, asset_class_cap=0.75, portfolio_heat_cap=1.0)
    proposed = pd.Series({FX_A: 0.8, FX_B: -0.4, COM: 0.3})
    before = proposed.copy(deep=True)

    first = check(proposed, HEALTHY, params)
    second = check(proposed, HEALTHY, params)

    assert first.approved.equals(second.approved)
    assert first.reasons == second.reasons
    pd.testing.assert_series_equal(proposed, before)  # input not mutated

    # exact hand-computed values: 25/63, -20/63, 2/7; final gross exactly 1.0
    assert first.approved[FX_A] == pytest.approx(25.0 / 63.0, rel=1e-12)
    assert first.approved[FX_B] == pytest.approx(-20.0 / 63.0, rel=1e-12)
    assert first.approved[COM] == pytest.approx(2.0 / 7.0, rel=1e-12)
    assert float(first.approved.abs().sum()) == pytest.approx(1.0, rel=1e-12)

    by_limit = {reason.limit: reason for reason in first.reasons}
    assert by_limit[RiskLimit.PER_INSTRUMENT].instrument == FX_A
    assert by_limit[RiskLimit.PER_INSTRUMENT].scale == pytest.approx(0.625)
    assert by_limit[RiskLimit.ASSET_CLASS].scale == pytest.approx(5.0 / 6.0)
    assert by_limit[RiskLimit.PORTFOLIO_HEAT].scale == pytest.approx(20.0 / 21.0)

    with pytest.raises(ValidationError):
        params.per_instrument_cap = 9.9  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HEALTHY.equity = 1.0  # type: ignore[misc]


# --- additional unit tests -------------------------------------------------------------------
def test_nan_passthrough_and_exact_zeros():
    proposed = pd.Series({FX_A: float("nan"), FX_B: 0.0})
    decision = check(proposed, HEALTHY, RiskParams())
    assert pd.isna(decision.approved[FX_A])
    assert decision.approved[FX_B] == 0.0
    assert decision.reasons == ()


def test_unknown_symbol_raises_before_any_decision():
    proposed = pd.Series({"NOPE/NONE": 0.1})
    with pytest.raises(KeyError):
        check(proposed, HEALTHY, RiskParams())
    # even when halted -- a malformed request must never be silently "approved flat"
    killed = AccountState(equity=1.0, peak_equity=1.0, kill_switch=True)
    with pytest.raises(KeyError):
        check(proposed, killed, RiskParams())


def test_drawdown_property_clamps_at_zero():
    above_peak = AccountState(equity=120_000.0, peak_equity=100_000.0)
    assert above_peak.drawdown == 0.0
    at_peak = AccountState(equity=1.0, peak_equity=1.0)
    assert at_peak.drawdown == 0.0


def test_params_and_state_bounds_validated():
    with pytest.raises(ValidationError):
        RiskParams(per_instrument_cap=0)
    with pytest.raises(ValidationError):
        RiskParams(drawdown_halt=1.0)  # must be < 1
    with pytest.raises(ValidationError):
        AccountState(equity=0.0, peak_equity=1.0)
