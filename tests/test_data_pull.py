"""The 10 acceptance tests from BUILD.md, Step 1 (`data-pull`).

All green or it's not done. Each test maps to one numbered acceptance criterion.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from tfx.data.align import tradable_mask
from tfx.data.cache import read_manifest
from tfx.data.clean import clean
from tfx.data.loader import build_panel_uncached, load, load_instrument
from tfx.data.pull import pull
from tfx.data.quality import build_report
from tfx.data.schema import POINT_IN_TIME_COLUMNS, TIMESTAMP_INDEX_NAME, QualityFlag
from tfx.instruments import get_instrument

OHLC = ["open", "high", "low", "close"]


# 1 — Coverage: every basket instrument has data across the range; gaps are reported.
def test_01_coverage(cache_dir, symbols, window):
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir)
    assert set(panel.columns.get_level_values("instrument")) == set(symbols)

    report = build_report(cache_dir, symbols, start, end)
    by_symbol = {q.symbol: q for q in report.instruments}
    for symbol in symbols:
        assert by_symbol[symbol].rows > 0
        assert 0.9 <= by_symbol[symbol].coverage <= 1.0
    # gaps are *reported* (equities carry classified holiday gaps), not silently dropped
    assert any(q.n_holiday_gaps > 0 for q in report.instruments)


# 2 — Timestamps: UTC-normalized, sorted ascending, no duplicates.
def test_02_timestamps(cache_dir, symbols, window):
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir)
    assert str(panel.index.tz) == "UTC"
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique
    for symbol in symbols:
        frame = load_instrument(symbol, start, end, cache_dir=cache_dir)
        assert str(frame.index.tz) == "UTC"
        assert frame.index.is_monotonic_increasing
        assert frame.index.is_unique


# 3 — Calendar gaps: weekend/holiday gaps expected & marked; no unexpected mid-week gaps.
def test_03_calendar_gaps(cache_dir, symbols, window):
    start, end = window
    report = build_report(cache_dir, symbols, start, end, max_weekday_gap_days=4)
    for quality in report.instruments:
        assert quality.n_unexpected_gaps == 0, quality.unexpected_runs
    # weekend bars only where the asset class actually trades weekends (crypto is 24/7; every
    # other class must never store a Saturday/Sunday bar)
    for symbol in symbols:
        frame = load_instrument(symbol, start, end, cache_dir=cache_dir)
        weekdays = {ts.weekday() for ts in frame.index}
        if get_instrument(symbol).weekly_closed_days:
            assert 5 not in weekdays and 6 not in weekdays
        else:
            assert {5, 6} <= weekdays  # a 24/7 instrument must actually cover weekends
    # equity indices have classified holiday gaps
    equities = [q for q in report.instruments if q.asset_class == "equity_index"]
    assert equities and all(q.n_holiday_gaps > 0 for q in equities)


# 4 — Alignment: one common index; a missing date is marked, never silently forward-filled.
def test_04_alignment_no_forward_fill(cache_dir, symbols, window):
    start, end = window
    panel = load(symbols, start, end, cache_dir=cache_dir)
    assert panel.index.is_monotonic_increasing and panel.index.is_unique

    quality = panel.xs("quality", level="field", axis=1).astype("int64")
    close = panel.xs("close", level="field", axis=1)
    pad = (quality & int(QualityFlag.ALIGN_PAD)) != 0
    assert pad.to_numpy().any()  # equity holidays vs FX trading -> some pads exist

    # every padded cell is an explicit NaN gap (NOT a carried-forward stale price)
    padded_closes = close.to_numpy()[pad.to_numpy()]
    assert np.isnan(padded_closes).all()
    # and padded cells are not tradable
    mask = tradable_mask(panel)
    assert not mask.to_numpy()[pad.to_numpy()].any()


# 5 — OHLC sanity on every stored bar.
def test_05_ohlc_sanity(cache_dir, symbols, window):
    start, end = window
    for symbol in symbols:
        frame = load_instrument(symbol, start, end, cache_dir=cache_dir, adjust=False)
        o, h, low, c = frame["open"], frame["high"], frame["low"], frame["close"]
        assert (h >= o).all() and (h >= c).all() and (h >= low).all()
        assert (low <= o).all() and (low <= c).all()
        assert (frame[OHLC] > 0).to_numpy().all()


# 6 — No NaNs in OHLC after cleaning; any imputed field is flagged.
def test_06_no_nan_ohlc_and_imputation_flagged(cache_dir, symbols, window):
    start, end = window
    for symbol in symbols:
        frame = load_instrument(symbol, start, end, cache_dir=cache_dir, adjust=False)
        assert not frame[OHLC].isna().to_numpy().any()

    # a bar missing a field is repaired from same-bar data AND flagged IMPUTED (no NaN remains)
    raw = pd.DataFrame(
        {"open": [1.0, 1.1], "high": [1.2, np.nan], "low": [0.9, 1.0], "close": [1.1, 1.2]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    result = clean(raw, get_instrument("EUR/USD"))
    assert not result.frame[OHLC].isna().to_numpy().any()
    flagged = (result.frame["quality"].astype("int64") & int(QualityFlag.IMPUTED)) != 0
    assert bool(flagged.iloc[1])


# 7 — Determinism: pulling twice over the same range yields byte-identical cached output.
def test_07_determinism_byte_identical(tmp_path, symbols, window, sample_dir):
    start, end = window
    d1, d2 = tmp_path / "c1", tmp_path / "c2"
    pull(symbols, start, end, source="fixture", cache_dir=d1, sample_dir=sample_dir)
    pull(symbols, start, end, source="fixture", cache_dir=d2, sample_dir=sample_dir)

    assert (d1 / "manifest.json").read_bytes() == (d2 / "manifest.json").read_bytes()
    for symbol in symbols:
        slug = get_instrument(symbol).slug
        assert (d1 / f"{slug}.parquet").read_bytes() == (d2 / f"{slug}.parquet").read_bytes()


# 8 — Cache round-trip: load-from-cache equals fresh-fetch for the same range.
def test_08_cache_roundtrip_equals_fresh(cache_dir, symbols, window, sample_dir):
    start, end = window
    cached = load(symbols, start, end, cache_dir=cache_dir, adjust=True)
    fresh = build_panel_uncached(
        symbols, start, end, source="fixture", sample_dir=sample_dir, adjust=True
    )
    pd.testing.assert_frame_equal(cached, fresh)


# 9 — No look-ahead in storage: cached bars hold only point-in-time fields.
def test_09_no_lookahead_in_storage(cache_dir, symbols):
    allowed = set(POINT_IN_TIME_COLUMNS) | {TIMESTAMP_INDEX_NAME}
    forbidden = ("future", "fwd", "forward", "zscore", "norm", "rolling", "adj", "return")
    for symbol in symbols:
        slug = get_instrument(symbol).slug
        disk = pd.read_parquet(cache_dir / f"{slug}.parquet")
        assert set(disk.columns) <= allowed
        for column in disk.columns:
            assert not any(token in column.lower() for token in forbidden), column
    # corporate actions live in the manifest, not as per-bar columns
    assert "instruments" in read_manifest(cache_dir)


# 10 — Fixture check: a hand-verified one-week slice matches expected OHLC exactly.
def test_10_handverified_fixture_slice(tmp_path, fixture_dir):
    start, end = date(2024, 3, 4), date(2024, 3, 8)
    cdir = tmp_path / "hv"
    pull(["EUR/USD"], start, end, source="fixture", cache_dir=cdir, sample_dir=fixture_dir)
    frame = load_instrument("EUR/USD", start, end, cache_dir=cdir, adjust=False)

    assert len(frame) == 5

    def at(day: str) -> pd.Series:
        return frame.loc[pd.Timestamp(day, tz="UTC")]

    assert at("2024-03-04")["open"] == 1.08000
    assert at("2024-03-04")["bid"] == 1.08390
    assert at("2024-03-04")["ask"] == 1.08410
    assert at("2024-03-06")["high"] == 1.09100
    assert at("2024-03-06")["close"] == 1.09000
    assert at("2024-03-07")["close"] == 1.09450
    assert at("2024-03-08")["low"] == 1.08800
