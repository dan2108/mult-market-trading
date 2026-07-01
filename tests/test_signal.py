"""Acceptance tests for the `signal` module (BUILD.md Step 3, tests 1-8) + unit tests.

Direction is discrete {-1, 0, +1}: the sign of the trailing `lookback`-bar return, NaN wherever
the bar (or its lookback-bars-earlier anchor) isn't tradable. Panels here are built with the same
(instrument, field) MultiIndex `tfx.data.align.align` produces, so these tests exercise the real
shape `compute` will see from `backtest`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tfx.core.signal import SignalParams, compute, shift_for_execution
from tfx.data.align import FIELD_LEVEL, INSTRUMENT_LEVEL
from tfx.data.loader import load
from tfx.data.schema import TIMESTAMP_INDEX_NAME, QualityFlag


def _panel(
    closes: dict[str, list[float]], pad: dict[str, set[int]] | None = None
) -> pd.DataFrame:
    """A minimal (close, quality) panel, shaped exactly like `align()`'s output. `pad` marks bar
    positions per instrument as an alignment pad (NaN close, ALIGN_PAD quality)."""
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


# 1 — No look-ahead: perturbing future bars must not change signal[t].
def test_01_no_lookahead_perturb_future():
    closes = [100, 100, 100, 90, 121, 100, 105, 95, 130, 80]
    panel_a = _panel({"A": closes})
    perturbed = closes[:6] + [999.0, 1.0, 500.0, 2.0]  # scramble everything after position 6
    panel_b = _panel({"A": perturbed})
    params = SignalParams(lookback=2)

    direction_a = compute(panel_a, params)
    direction_b = compute(panel_b, params)

    prefix = direction_a.index[:6]
    assert np.array_equal(
        direction_a.loc[prefix, "A"].to_numpy(),
        direction_b.loc[prefix, "A"].to_numpy(),
        equal_nan=True,
    )


# 2 — Timing: the position from signal[t] is applied at t+1, via the shared shift helper.
def test_02_timing_shift_for_execution():
    panel = _panel({"A": [100, 100, 100, 90, 121, 100]})
    direction = compute(panel, SignalParams(lookback=2))
    executed = shift_for_execution(direction)

    assert pd.isna(executed["A"].iloc[0])  # no prior direction to hold on the first bar
    assert np.array_equal(
        executed["A"].to_numpy()[1:], direction["A"].to_numpy()[:-1], equal_nan=True
    )


# 3 — Pure & deterministic: same panel+params -> same signals; no global state; no mutation.
def test_03_pure_deterministic_no_mutation_and_params_frozen():
    panel = _panel({"A": [100, 100, 90, 121, 100, 105]})
    params = SignalParams(lookback=2)
    before = panel.copy(deep=True)

    first = compute(panel, params)
    second = compute(panel, params)
    assert first.equals(second)
    pd.testing.assert_frame_equal(panel, before)  # compute must not mutate its input

    with pytest.raises(ValidationError):
        params.lookback = 5  # type: ignore[misc]


# 4 — Direction sanity: long on uptrend, short on downtrend, ~flat (no net bias) on noise.
def test_04_direction_sanity_uptrend_downtrend_noise():
    params = SignalParams(lookback=5)
    up = [100.0 + i for i in range(20)]
    down = [100.0 - i for i in range(20)]
    panel = _panel({"UP": up, "DOWN": down})
    direction = compute(panel, params)
    assert (direction["UP"].iloc[5:] == 1.0).all()
    assert (direction["DOWN"].iloc[5:] == -1.0).all()

    # whipsaw / no persistent trend: alternating +1 tick, single-bar lookback -> zero net bias.
    # 21 bars (not 20): closes the up/down cycle evenly so the 20 valid directions split 10/10 --
    # an odd bar count would leave one direction unbalanced by construction, not by trend.
    noise = [100.0, 101.0] * 10 + [100.0]
    noise_panel = _panel({"N": noise})
    noise_direction = compute(noise_panel, SignalParams(lookback=1))
    realized = noise_direction["N"].dropna()
    assert realized.mean() == pytest.approx(0.0)
    assert set(realized.unique()) == {1.0, -1.0}


# 5 — Padded/NaN bars produce no signal (no trading a non-existent bar).
def test_05_padded_bars_produce_no_signal():
    closes = {"A": [100, 100, 90, 121, 100, 105], "B": [100, 100, 90, 121, 100, 105]}
    panel = _panel(closes, pad={"B": {3}})
    direction = compute(panel, SignalParams(lookback=2))
    assert pd.isna(direction["B"].iloc[3])
    assert not pd.isna(direction["A"].iloc[3])  # A unaffected by B's pad


# 6 — Parameter bounds respected; degenerate params handled.
def test_06_parameter_bounds_and_degenerate_handling():
    with pytest.raises(ValidationError):
        SignalParams(lookback=0)
    with pytest.raises(ValidationError):
        SignalParams(lookback=-3)

    # lookback longer than available history -> all-NaN, not a crash
    panel = _panel({"A": [100.0, 101.0, 99.0, 103.0]})
    direction = compute(panel, SignalParams(lookback=10))
    assert direction["A"].isna().all()


# 7 — Per-instrument independence across the basket.
def test_07_per_instrument_independence():
    base = {"A": [100, 100, 90, 121, 100, 105], "B": [100, 100, 90, 121, 100, 105]}
    panel_1 = _panel(base)
    perturbed = dict(base)
    perturbed["A"] = [50, 60, 40, 200, 10, 300]
    panel_2 = _panel(perturbed)
    params = SignalParams(lookback=2)

    direction_1 = compute(panel_1, params)
    direction_2 = compute(panel_2, params)
    assert direction_1["B"].equals(direction_2["B"])  # B unaffected by A's perturbation


# 8 — Fixture slice matches a hand-computed signal (see tests/fixtures/signal_handcalc.md).
def test_08_handcomputed_fixture_slice():
    panel = _panel({"X": [100, 100, 100, 90, 121, 100]})
    direction = compute(panel, SignalParams(lookback=2))["X"]
    assert pd.isna(direction.iloc[0])
    assert pd.isna(direction.iloc[1])
    assert direction.iloc[2] == 0.0
    assert direction.iloc[3] == -1.0
    assert direction.iloc[4] == 1.0
    assert direction.iloc[5] == 1.0


# --- additional unit tests -------------------------------------------------------------------
def test_compute_on_real_aligned_panel(cache_dir, manifest, symbols, window):
    """Sanity check against the real data-pull pipeline output, not just hand-rolled panels."""
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir, manifest=manifest)
    direction = compute(panel, SignalParams())
    assert list(direction.columns) == symbols
    assert direction.index.equals(panel.index)
    # well past warm-up, most instrument-bars should carry a real signal (not silently all-NaN)
    assert direction.iloc[30:].notna().to_numpy().mean() > 0.5
    assert set(direction.stack().unique()) <= {-1.0, 0.0, 1.0}


def test_default_lookback_is_positive():
    assert SignalParams().lookback > 0
