"""Acceptance tests for the `signal` module (BUILD.md Step 3, tests 1-8) + unit tests.

Direction is the equal-weight mean of sign(trailing return) across the lookback ensemble, graded
in [-1, +1] in steps of 1/len(lookbacks); a single-entry tuple degenerates to the discrete
{-1, 0, +1} rule (test 8 pins that case to the original hand-calc verbatim). NaN wherever the bar
isn't tradable or any lookback's anchor is missing. Panels here are built with the same
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
    params = SignalParams(lookbacks=(1, 2))

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
    direction = compute(panel, SignalParams(lookbacks=(2,)))
    executed = shift_for_execution(direction)

    assert pd.isna(executed["A"].iloc[0])  # no prior direction to hold on the first bar
    assert np.array_equal(
        executed["A"].to_numpy()[1:], direction["A"].to_numpy()[:-1], equal_nan=True
    )


# 3 — Pure & deterministic: same panel+params -> same signals; no global state; no mutation.
def test_03_pure_deterministic_no_mutation_and_params_frozen():
    panel = _panel({"A": [100, 100, 90, 121, 100, 105]})
    params = SignalParams(lookbacks=(1, 2))
    before = panel.copy(deep=True)

    first = compute(panel, params)
    second = compute(panel, params)
    assert first.equals(second)
    pd.testing.assert_frame_equal(panel, before)  # compute must not mutate its input

    with pytest.raises(ValidationError):
        params.lookbacks = (5,)  # type: ignore[misc]


# 4 — Direction sanity: long on uptrend, short on downtrend, ~flat (no net bias) on noise.
def test_04_direction_sanity_uptrend_downtrend_noise():
    up = [100.0 + i for i in range(20)]
    down = [100.0 - i for i in range(20)]
    panel = _panel({"UP": up, "DOWN": down})

    # single lookback and a fast/slow ensemble must both saturate on a monotone trend: every
    # horizon's sign agrees, so the mean is exactly +/-1 past the longest warm-up.
    for params in (SignalParams(lookbacks=(5,)), SignalParams(lookbacks=(1, 5))):
        direction = compute(panel, params)
        warm = max(params.lookbacks)
        assert (direction["UP"].iloc[warm:] == 1.0).all()
        assert (direction["DOWN"].iloc[warm:] == -1.0).all()

    # whipsaw / no persistent trend: alternating +1 tick, single-bar lookback -> zero net bias.
    # 21 bars (not 20): closes the up/down cycle evenly so the 20 valid directions split 10/10 --
    # an odd bar count would leave one direction unbalanced by construction, not by trend.
    noise = [100.0, 101.0] * 10 + [100.0]
    noise_panel = _panel({"N": noise})
    noise_direction = compute(noise_panel, SignalParams(lookbacks=(1,)))
    realized = noise_direction["N"].dropna()
    assert realized.mean() == pytest.approx(0.0)
    assert set(realized.unique()) == {1.0, -1.0}


# 5 — Padded/NaN bars produce no signal (no trading a non-existent bar).
def test_05_padded_bars_produce_no_signal():
    closes = {"A": [100, 100, 90, 121, 100, 105], "B": [100, 100, 90, 121, 100, 105]}
    panel = _panel(closes, pad={"B": {3}})
    direction = compute(panel, SignalParams(lookbacks=(2,)))
    assert pd.isna(direction["B"].iloc[3])
    assert not pd.isna(direction["A"].iloc[3])  # A unaffected by B's pad


# Regression: an anchor landing on THIS instrument's own holiday pad must not blank an
# otherwise-computable signal (row-offset vs tradable-bar-offset lookback counting).
def test_anchor_skips_own_pad_counts_tradable_bars_not_calendar_rows():
    closes = [100, 102, 104, 106, 108, 110, 112, 114]
    panel = _panel({"X": closes}, pad={"X": {3}})
    direction = compute(panel, SignalParams(lookbacks=(2,)))["X"]

    # position 3 is the pad itself: still untradable, still no signal, regardless of anchoring.
    assert pd.isna(direction.iloc[3])

    # THE regression check: position 5's naive row-offset anchor (close.shift(2)) would land
    # exactly on position 3 (the pad) and blank an otherwise fully-available signal. Counting
    # back 2 of X's own TRADABLE bars instead anchors on position 2 (close=104): a real,
    # non-NaN direction. Confirm the two rules actually disagree here, or this test proves
    # nothing.
    close = panel[("X", "close")]
    old_row_offset_anchor = close.shift(2).iloc[5]
    assert pd.isna(old_row_offset_anchor)
    assert not pd.isna(direction.iloc[5])
    assert direction.iloc[5] == 1.0  # close[5]=110 vs the tradable-bar anchor close[2]=104

    # bars unaffected by the pad's 2-bar shadow (position 2, before the pad) are untouched.
    assert direction.iloc[2] == 1.0  # close[2]=104 vs close[0]=100


# 6 — Parameter bounds respected; degenerate params handled.
def test_06_parameter_bounds_and_degenerate_handling():
    for bad in ((), (0,), (-3,), (2, 2), (5, 3), (1, -2, 3)):
        with pytest.raises(ValidationError):
            SignalParams(lookbacks=bad)

    # any lookback longer than available history -> all-NaN, not a crash: the ensemble requires
    # EVERY horizon's anchor, so one over-length member blanks the whole signal.
    panel = _panel({"A": [100.0, 101.0, 99.0, 103.0]})
    assert compute(panel, SignalParams(lookbacks=(10,)))["A"].isna().all()
    assert compute(panel, SignalParams(lookbacks=(2, 10)))["A"].isna().all()


# 7 — Per-instrument independence across the basket.
def test_07_per_instrument_independence():
    base = {"A": [100, 100, 90, 121, 100, 105], "B": [100, 100, 90, 121, 100, 105]}
    panel_1 = _panel(base)
    perturbed = dict(base)
    perturbed["A"] = [50, 60, 40, 200, 10, 300]
    panel_2 = _panel(perturbed)
    params = SignalParams(lookbacks=(1, 2))

    direction_1 = compute(panel_1, params)
    direction_2 = compute(panel_2, params)
    assert direction_1["B"].equals(direction_2["B"])  # B unaffected by A's perturbation


# 8 — Fixture slice matches a hand-computed signal (see tests/fixtures/signal_handcalc.md).
# The single-entry tuple is the degenerate config: these expected values are the ORIGINAL
# single-lookback hand calc, retained verbatim as the regression anchor for the ensemble rewrite.
def test_08_handcomputed_fixture_slice():
    panel = _panel({"X": [100, 100, 100, 90, 121, 100]})
    direction = compute(panel, SignalParams(lookbacks=(2,)))["X"]
    assert pd.isna(direction.iloc[0])
    assert pd.isna(direction.iloc[1])
    assert direction.iloc[2] == 0.0
    assert direction.iloc[3] == -1.0
    assert direction.iloc[4] == 1.0
    assert direction.iloc[5] == 1.0


# --- ensemble unit tests ----------------------------------------------------------------------
def test_ensemble_handcomputed_fixture_slice():
    """Hand calc for lookbacks=(1, 3) hitting every grid value {-1, -0.5, 0, +0.5, +1} -- see
    tests/fixtures/signal_handcalc.md."""
    panel = _panel({"X": [100, 102, 101, 101, 99, 104, 103, 90, 90]})
    direction = compute(panel, SignalParams(lookbacks=(1, 3)))["X"]

    assert direction.iloc[:3].isna().all()  # warm-up: the 3-bar anchor is missing until t=3
    expected = [0.5, -1.0, 1.0, 0.0, -1.0, -0.5]
    assert direction.iloc[3:].tolist() == expected


def test_ensemble_is_mean_of_single_lookback_signals():
    """Compositional property: the ensemble equals the elementwise mean of its single-lookback
    members on the jointly-valid region (deterministic, no RNG)."""
    closes = {"A": [100, 102, 101, 101, 99, 104, 103, 90, 90, 95, 91, 108]}
    panel = _panel(closes)

    ensemble = compute(panel, SignalParams(lookbacks=(1, 3)))
    single_1 = compute(panel, SignalParams(lookbacks=(1,)))
    single_3 = compute(panel, SignalParams(lookbacks=(3,)))
    mean = (single_1 + single_3) / 2.0  # NaN wherever either member is NaN

    pd.testing.assert_frame_equal(ensemble, mean)


def test_ensemble_burn_in_is_max_lookback():
    """NaN strictly before max(lookbacks) valid bars, real values from there on."""
    n = 12
    panel = _panel({"A": [100.0 + i for i in range(n)]})
    direction = compute(panel, SignalParams(lookbacks=(2, 5)))["A"]
    assert direction.iloc[:5].isna().all()
    assert direction.iloc[5:].notna().all()


def test_compute_on_real_aligned_panel(cache_dir, manifest, symbols, window):
    """Sanity check against the real data-pull pipeline output, not just hand-rolled panels."""
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir, manifest=manifest)
    params = SignalParams()
    direction = compute(panel, params)
    assert list(direction.columns) == symbols
    assert direction.index.equals(panel.index)

    # well past the longest warm-up, most instrument-bars should carry a real signal
    warm = max(params.lookbacks)
    assert direction.iloc[warm + 10 :].notna().to_numpy().mean() > 0.5

    # every value on the 1/len(lookbacks) grid within [-1, +1]
    realized = direction.stack().to_numpy()
    assert np.abs(realized).max() <= 1.0
    scaled = realized * len(params.lookbacks)
    np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-12)


def test_default_lookbacks_are_canonical():
    lookbacks = SignalParams().lookbacks
    assert lookbacks, "default ensemble must be non-empty"
    assert all(lb > 0 for lb in lookbacks)
    assert list(lookbacks) == sorted(set(lookbacks))  # strictly increasing, no duplicates
