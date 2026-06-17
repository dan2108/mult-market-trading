"""The read API that `backtest` consumes.

:func:`load` returns an aligned wide panel (MultiIndex columns: instrument x field) over a date
range, optionally split/dividend-adjusted. :func:`load_instrument` returns a single instrument's
series on its native calendar. :func:`build_panel_uncached` reproduces the same panel directly
from a source without touching the cache — used to prove cache round-trip == fresh fetch.
"""

from __future__ import annotations

import pandas as pd

from ..instruments import get_instrument
from .adjust import adjustment_factors, apply_adjustment
from .align import align
from .cache import read_corporate_actions, read_instrument, read_manifest
from .clean import clean
from .sources import get_source


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.normalize()


def _restrict(frame: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    return frame.loc[(frame.index >= _utc(start)) & (frame.index <= _utc(end))]


def _maybe_adjust(
    frame: pd.DataFrame, corporate_actions: pd.DataFrame, adjust: bool
) -> pd.DataFrame:
    if not adjust:
        return frame
    factors = adjustment_factors(frame["close"], corporate_actions)
    return apply_adjustment(frame, factors)


def load_instrument(
    symbol: str,
    start: object,
    end: object,
    *,
    cache_dir: object,
    adjust: bool = True,
    manifest: dict | None = None,
) -> pd.DataFrame:
    """Load one instrument's cached series over ``[start, end]`` on its native calendar."""
    manifest = manifest or read_manifest(cache_dir)
    instrument = get_instrument(symbol)
    frame = _restrict(read_instrument(cache_dir, instrument, manifest=manifest), start, end)
    corporate_actions = read_corporate_actions(cache_dir, instrument, manifest=manifest)
    return _maybe_adjust(frame, corporate_actions, adjust)


def load(
    symbols: list[str],
    start: object,
    end: object,
    *,
    cache_dir: object,
    adjust: bool = True,
    manifest: dict | None = None,
) -> pd.DataFrame:
    """Load an aligned panel for ``symbols`` over ``[start, end]`` (what backtest consumes)."""
    manifest = manifest or read_manifest(cache_dir)
    frames = {
        get_instrument(s).symbol: load_instrument(
            s, start, end, cache_dir=cache_dir, adjust=adjust, manifest=manifest
        )
        for s in symbols
    }
    return align(frames, start=start, end=end)


def build_panel_uncached(
    symbols: list[str],
    start: object,
    end: object,
    *,
    source: str = "fixture",
    sample_dir: object | None = None,
    adjust: bool = True,
) -> pd.DataFrame:
    """Build the same aligned panel as :func:`load`, but straight from a source (no cache)."""
    src = get_source(source, sample_dir=sample_dir)
    frames: dict[str, pd.DataFrame] = {}
    for s in symbols:
        instrument = get_instrument(s)
        cleaned = clean(src.fetch_ohlc(instrument, start, end), instrument).frame
        frame = _restrict(cleaned, start, end)
        corporate_actions = src.fetch_corporate_actions(instrument)
        frames[instrument.symbol] = _maybe_adjust(frame, corporate_actions, adjust)
    return align(frames, start=start, end=end)
