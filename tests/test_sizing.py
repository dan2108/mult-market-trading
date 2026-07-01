"""Acceptance tests for the `sizing` module (BUILD.md Step 4, tests 1-8) + unit tests.

Two-stage vol-targeting: inverse-EWMA-vol per instrument, then a single portfolio-level scalar
(from the full EWMA covariance matrix) so ex-ante portfolio vol hits `target_vol`, then a hard
gross-leverage clamp. Several tests below re-derive the EWMA covariance recursion independently
in plain numpy (no import from `tfx.core.sizing`) as a cross-check oracle -- see
tests/fixtures/sizing_handcalc.md for why, given the matrix math isn't practical to fully
transcribe by hand the way `costs`/`signal`'s scalar arithmetic was.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tfx.core.signal import SignalParams
from tfx.core.signal import compute as signal_compute
from tfx.core.sizing import SizingParams, size
from tfx.data.align import FIELD_LEVEL, INSTRUMENT_LEVEL
from tfx.data.loader import load
from tfx.data.schema import TIMESTAMP_INDEX_NAME, QualityFlag


def _panel(
    closes: dict[str, list[float]], pad: dict[str, set[int]] | None = None
) -> pd.DataFrame:
    """A minimal (close, quality) panel, shaped exactly like `align()`'s output."""
    pad = pad or {}
    n = len(next(iter(closes.values())))
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC", name=TIMESTAMP_INDEX_NAME)
    frames = {}
    for symbol, values in closes.items():
        padded = pad.get(symbol, set())
        close = [float("nan") if i in padded else float(v) for i, v in enumerate(values)]
        quality = [int(QualityFlag.ALIGN_PAD) if i in padded else 0 for i in range(n)]
        frames[symbol] = pd.DataFrame({"close": close, "quality": quality}, index=index)
    return pd.concat(
        frames.values(), axis=1, keys=list(frames.keys()), names=[INSTRUMENT_LEVEL, FIELD_LEVEL]
    )


def _reference_annualized_cov(prices: dict[str, list[float]], halflife: float) -> np.ndarray:
    """Independent (non-tfx.core.sizing) EWMA covariance recursion, annualized -- an oracle, not
    a copy of the implementation. Returns an (n_bars, k, k) array; NaN before the burn-in."""
    close = pd.DataFrame(prices)
    log_returns = np.log(close / close.shift(1)).to_numpy()
    lam = 0.5 ** (1.0 / halflife)
    min_periods = math.ceil(halflife)
    n_bars, k = close.shape
    cov = np.zeros((k, k))
    valid_count = 0
    out = np.full((n_bars, k, k), np.nan)
    for t in range(n_bars):
        r = log_returns[t]
        if not np.isnan(r).any():
            cov = lam * cov + (1.0 - lam) * np.outer(r, r)
            valid_count += 1
        if valid_count >= min_periods:
            out[t] = cov * 252.0
    return out


def _oscillating(start: float, up: float, period: int, n: int) -> list[float]:
    """Deterministic, no-RNG oscillation: +up% every `period`-th step, -up% otherwise."""
    values = [start]
    for i in range(n - 1):
        values.append(values[-1] * (1.0 + up) if i % period == 0 else values[-1] / (1.0 + up))
    return values


# 1 — Equal-risk: for equal directions, a more volatile instrument gets a smaller position.
def test_01_equal_risk_more_volatile_gets_smaller_position():
    low_vol = _oscillating(100.0, 0.001, 2, 40)
    high_vol = _oscillating(100.0, 0.01, 2, 40)
    panel = _panel({"LOW": low_vol, "HIGH": high_vol})
    directions = pd.DataFrame(1.0, index=panel.index, columns=["LOW", "HIGH"])
    weight = size(panel, directions, SizingParams(ewma_halflife=5.0, leverage_cap=100.0))

    last = weight.iloc[-1]
    assert last["LOW"] > 0 and last["HIGH"] > 0
    assert last["HIGH"] < last["LOW"]


# 2 — Portfolio vol target: ex-ante portfolio vol from the estimates ~= target, within tolerance.
def test_02_portfolio_vol_hits_target_when_cap_not_binding():
    a = _oscillating(100.0, 0.015, 2, 40)
    b = _oscillating(100.0, 0.010, 3, 40)
    panel = _panel({"A": a, "B": b})
    directions = pd.DataFrame(1.0, index=panel.index, columns=["A", "B"])
    params = SizingParams(ewma_halflife=5.0, target_vol=0.10, leverage_cap=100.0)

    weight = size(panel, directions, params)
    cov = _reference_annualized_cov({"A": a, "B": b}, halflife=5.0)

    w = weight.iloc[-1].to_numpy()
    sigma = cov[-1]
    portfolio_vol = math.sqrt(float(w @ sigma @ w))
    assert portfolio_vol == pytest.approx(params.target_vol, rel=1e-6)


# 3 — No look-ahead: vol/correlation at t use data <= t -- future bars don't change size[t].
def test_03_no_lookahead_perturb_future():
    a = [100.0, 101, 99, 103, 97, 105, 102, 108, 95, 110, 90, 115]
    b = [100.0, 99, 102, 98, 104, 96, 108, 101, 106, 99, 112, 97]
    panel_a = _panel({"A": a, "B": b})
    perturbed_a = a[:8] + [500.0, 1.0, 999.0, 2.0]
    panel_b = _panel({"A": perturbed_a, "B": b})
    directions = pd.DataFrame(1.0, index=panel_a.index, columns=["A", "B"])
    params = SizingParams(ewma_halflife=3.0, leverage_cap=100.0)

    weight_a = size(panel_a, directions, params)
    weight_b = size(panel_b, directions, params)

    prefix = weight_a.index[:8]
    assert np.array_equal(
        weight_a.loc[prefix].to_numpy(), weight_b.loc[prefix].to_numpy(), equal_nan=True
    )


# 4 — Flat signal -> zero size.
def test_04_flat_signal_zero_size():
    a = _oscillating(100.0, 0.01, 2, 30)
    b = _oscillating(100.0, 0.02, 3, 30)
    panel = _panel({"A": a, "B": b})
    directions = pd.DataFrame(0.0, index=panel.index, columns=["A", "B"])
    weight = size(panel, directions, SizingParams(ewma_halflife=5.0))

    tail = weight.iloc[10:]
    assert (tail == 0.0).to_numpy().all()


# 5 — Leverage cap respected: gross exposure <= cap; the cap must actually bind, not be slack.
def test_05_leverage_cap_respected():
    a = _oscillating(100.0, 0.0001, 2, 30)  # extremely calm -> huge uncapped weight
    b = _oscillating(100.0, 0.0001, 2, 30)
    panel = _panel({"A": a, "B": b})
    directions = pd.DataFrame(1.0, index=panel.index, columns=["A", "B"])
    params = SizingParams(ewma_halflife=5.0, target_vol=0.10, leverage_cap=1.5)

    weight = size(panel, directions, params)
    gross = weight.iloc[10:].abs().sum(axis=1)
    assert (gross <= 1.5 + 1e-9).all()
    assert gross.max() == pytest.approx(1.5, rel=1e-6)  # actually binds


# 6 — Correlation handled: a perfectly-correlated pair can't jointly carry more risk than the
# correlation implies (no double-counting) relative to a less-correlated pair with the same vols.
def test_06_correlation_no_double_counting():
    base = _oscillating(100.0, 0.02, 2, 40)
    different = _oscillating(100.0, 0.02, 3, 40)
    params = SizingParams(ewma_halflife=5.0, target_vol=0.10, leverage_cap=100.0)

    panel_correlated = _panel({"A": base, "B": base})  # B identical to A -> correlation 1
    directions = pd.DataFrame(1.0, index=panel_correlated.index, columns=["A", "B"])
    correlated_weight = size(panel_correlated, directions, params)

    panel_diverse = _panel({"A": base, "B": different})
    diverse_weight = size(panel_diverse, directions, params)

    correlated_gross = correlated_weight.iloc[-1].abs().sum()
    diverse_gross = diverse_weight.iloc[-1].abs().sum()
    assert correlated_gross < diverse_gross


# 7 — Pure & deterministic; padded bars -> no size; params frozen; no mutation.
def test_07_pure_deterministic_no_mutation_and_padded_bars_no_size():
    a = [100.0, 101, 99, 103, 97, 105, 102, 108]
    b = [100.0, 99, 101, 98, 104, 96, 107, 101]
    panel = _panel({"A": a, "B": b}, pad={"B": {4}})
    before = panel.copy(deep=True)
    directions = pd.DataFrame(
        {"A": [np.nan, 1, 1, -1, 1, -1, 1, 1], "B": [np.nan, 1, -1, 1, np.nan, -1, 1, -1]},
        index=panel.index,
    )
    params = SizingParams(ewma_halflife=2.0, leverage_cap=100.0)

    first = size(panel, directions, params)
    second = size(panel, directions, params)
    assert first.equals(second)
    pd.testing.assert_frame_equal(panel, before)
    assert pd.isna(first["B"].iloc[4])

    with pytest.raises(ValidationError):
        params.target_vol = 0.5  # type: ignore[misc]


# 8 — Fixture slice matches a hand-computed signal (see tests/fixtures/sizing_handcalc.md).
def test_08_handcomputed_fixture_slice():
    prices = {"A": [100.0, 110.0, 90.0], "B": [100.0, 105.0, 100.0]}
    panel = _panel(prices)
    directions = pd.DataFrame(
        {"A": [np.nan, 1.0, 1.0], "B": [np.nan, 1.0, -1.0]}, index=panel.index
    )
    params = SizingParams(ewma_halflife=1.0, target_vol=0.10, leverage_cap=100.0)
    weight = size(panel, directions, params)

    cov = _reference_annualized_cov(prices, halflife=1.0)
    expected = np.full((3, 2), np.nan)
    for t in (1, 2):
        sigma = cov[t]
        vol = np.sqrt(np.diag(sigma))
        d = directions.iloc[t].to_numpy()
        raw = d / vol
        portfolio_var = float(raw @ sigma @ raw)
        scalar = params.target_vol / math.sqrt(portfolio_var)
        expected[t] = raw * scalar

    np.testing.assert_allclose(weight.to_numpy(), expected, rtol=1e-9, equal_nan=True)


# --- additional unit tests -------------------------------------------------------------------
def test_partial_active_basket_does_not_poison_other_instruments():
    """One instrument lacking a direction on a bar must not silently zero the others: a single
    exchange holiday for instrument B must not flatten instrument A's real position that day.
    The vol-target math must fall back to just the active subset, not the whole basket."""
    a = [100.0, 101, 99, 103, 97, 105, 102, 108]
    b = [100.0, 99, 101, 98, 104, 96, 107, 101]
    panel = _panel({"A": a, "B": b})
    directions = pd.DataFrame(
        {"A": [np.nan, 1, 1, -1, 1, -1, 1, 1], "B": [np.nan, 1, -1, 1, np.nan, -1, 1, -1]},
        index=panel.index,
    )
    params = SizingParams(ewma_halflife=2.0, leverage_cap=100.0)
    weight = size(panel, directions, params)

    # at bar 4, B has no direction -- A must be sized as if it were the ONLY active instrument
    # that day: weight = target_vol / vol_A, no correlation term (B isn't in the active set).
    cov = _reference_annualized_cov({"A": a, "B": b}, halflife=2.0)
    vol_a = math.sqrt(cov[4][0, 0])
    expected_a = params.target_vol / vol_a  # direction_A == +1 at bar 4

    assert pd.isna(weight["B"].iloc[4])
    assert weight["A"].iloc[4] == pytest.approx(expected_a, rel=1e-9)


def test_zero_vol_instrument_does_not_poison_others():
    """A degenerate zero-variance instrument (e.g. a data glitch, or a genuinely flat price)
    must not poison other instruments' weights, and must itself size as NaN (an inverse-vol
    weight is undefined when vol is exactly zero)."""
    a = [100.0, 101, 99, 103, 97, 105, 102, 108, 95, 110]
    b = [100.0, 99, 101, 98, 104, 96, 107, 101, 106, 99]
    c = [100.0] * 10  # perfectly flat: zero variance
    panel = _panel({"A": a, "B": b, "C": c})
    directions = pd.DataFrame(1.0, index=panel.index, columns=["A", "B", "C"])
    params = SizingParams(ewma_halflife=3.0, leverage_cap=100.0)

    weight = size(panel, directions, params)
    tail = weight.iloc[4:]  # past the min_periods=3 burn-in, with margin

    assert tail["C"].isna().all()
    assert tail["A"].notna().all() and (tail["A"] != 0.0).all()
    assert tail["B"].notna().all() and (tail["B"] != 0.0).all()


def test_sizing_on_real_aligned_panel_with_real_signal(cache_dir, manifest, symbols, window):
    """Integration sanity check: real data-pull panel, real signal.compute output, both feeding
    the real sizing.size -- not just hand-rolled fixtures."""
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir, manifest=manifest)
    directions = signal_compute(panel, SignalParams())
    params = SizingParams()

    weight = size(panel, directions, params)
    assert list(weight.columns) == symbols
    assert weight.index.equals(panel.index)

    gross = weight.abs().sum(axis=1)
    assert (gross.dropna() <= params.leverage_cap + 1e-9).all()
    assert weight.notna().to_numpy().sum() > 0


def test_params_bounds_validated():
    with pytest.raises(ValidationError):
        SizingParams(ewma_halflife=0)
    with pytest.raises(ValidationError):
        SizingParams(target_vol=0)
    with pytest.raises(ValidationError):
        SizingParams(leverage_cap=0)
