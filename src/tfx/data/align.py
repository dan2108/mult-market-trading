"""Cross-instrument alignment onto one common calendar.

The master calendar is the **union** of every instrument's native trading days within the range.
Where an instrument has no native bar on a master date, that row is inserted with NaN OHLC and
flagged ``ALIGN_PAD`` — it is information ("market closed for this instrument"), never a
forward-filled price. This is the explicit, documented rule for test #4: no silent fill that
could leak a stale price into a day the instrument did not trade.

Output is a wide panel with a MultiIndex on the columns: level ``instrument`` x level ``field``.
"""

from __future__ import annotations

import pandas as pd

from .schema import POINT_IN_TIME_COLUMNS, QUALITY_COLUMN, TIMESTAMP_INDEX_NAME, QualityFlag

INSTRUMENT_LEVEL = "instrument"
FIELD_LEVEL = "field"


def _utc_ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.normalize()


def master_calendar(
    frames: dict[str, pd.DataFrame],
    *,
    start: object | None = None,
    end: object | None = None,
) -> pd.DatetimeIndex:
    """Union of all instrument calendars, sorted & unique, clipped to [start, end]."""
    master: pd.DatetimeIndex | None = None
    for frame in frames.values():
        master = frame.index if master is None else master.union(frame.index)
    if master is None:
        master = pd.DatetimeIndex([], tz="UTC")
    master = pd.DatetimeIndex(master).sort_values()
    if start is not None:
        master = master[master >= _utc_ts(start)]
    if end is not None:
        master = master[master <= _utc_ts(end)]
    master.name = TIMESTAMP_INDEX_NAME
    return master


def align(
    frames: dict[str, pd.DataFrame],
    *,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    """Align per-instrument frames onto the union master calendar. Never forward-fills."""
    if not frames:
        raise ValueError("align() requires at least one instrument frame")

    master = master_calendar(frames, start=start, end=end)
    pad = int(QualityFlag.ALIGN_PAD)

    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        reindexed = frame.reindex(master)
        is_pad = ~master.isin(frame.index)
        quality = reindexed[QUALITY_COLUMN].where(~is_pad, other=pad)
        reindexed[QUALITY_COLUMN] = quality.fillna(pad).astype("int64")
        aligned[symbol] = reindexed[list(POINT_IN_TIME_COLUMNS)]

    panel = pd.concat(
        aligned.values(),
        axis=1,
        keys=list(aligned.keys()),
        names=[INSTRUMENT_LEVEL, FIELD_LEVEL],
    )
    panel.index.name = TIMESTAMP_INDEX_NAME
    return panel


def tradable_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """Boolean (timestamp x instrument): True where a real, tradable bar exists (not an
    alignment pad and a non-NaN close). IMPUTED/REPAIRED real bars remain tradable."""
    quality = panel.xs(QUALITY_COLUMN, level=FIELD_LEVEL, axis=1).astype("int64")
    close = panel.xs("close", level=FIELD_LEVEL, axis=1)
    not_pad = (quality & int(QualityFlag.ALIGN_PAD)) == 0
    return not_pad & close.notna()


WEEKDAY_PERIODS_PER_YEAR = 252.0  # standard convention: every instrument closes on weekends
CALENDAR_PERIODS_PER_YEAR = 365.0  # the basket includes a 24/7 instrument (e.g. crypto)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Annualization factor implied by a panel's OWN calendar, not a silently-hardcoded
    constant: 365 if any row falls on a weekend, else the standard 252 weekday convention.

    Since `align()`'s master calendar is the UNION of every instrument's own trading days, a
    weekend row can exist only because some instrument in the basket actually trades that day
    (e.g. crypto). Deriving the factor from the index -- rather than from a module-level
    constant -- means adding a 24/7 instrument to a weekday basket correctly changes every
    consumer's annualization (vol targeting, Sharpe, CAGR, turnover) instead of leaving them
    silently measuring against the wrong number of bars per year.
    """
    if (index.dayofweek >= 5).any():
        return CALENDAR_PERIODS_PER_YEAR
    return WEEKDAY_PERIODS_PER_YEAR
