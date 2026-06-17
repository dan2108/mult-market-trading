"""Clean RAW source bars into the canonical, point-in-time per-instrument series.

Output guarantees (per instrument, on its own native trading calendar):
  * index: tz-aware UTC, normalized to the session date, sorted ascending, unique;
  * columns exactly ``POINT_IN_TIME_COLUMNS`` (open/high/low/close/bid/ask/quality);
  * no NaNs in OHLC — any field repaired from same-bar data is flagged ``IMPUTED``;
  * OHLC inequalities corrected (high/low widened to bound open/close) and flagged
    ``OHLC_REPAIRED``;
  * duplicate timestamps collapsed (keep last) and flagged ``DUP_RESOLVED``.

Every repair uses ONLY data from the same bar — never another bar and never the future — so no
look-ahead can enter here. Bars with no usable price at all are dropped and counted (surfaced as
gaps by the quality report), never silently forward-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..instruments import Instrument
from .errors import DataQualityError
from .schema import (
    OHLC_COLUMNS,
    POINT_IN_TIME_COLUMNS,
    QUALITY_COLUMN,
    QUOTE_COLUMNS,
    TIMESTAMP_INDEX_NAME,
    QualityFlag,
)

_PRICE_AND_QUOTE = tuple(c for c in POINT_IN_TIME_COLUMNS if c != QUALITY_COLUMN)


@dataclass(frozen=True)
class CleanResult:
    frame: pd.DataFrame
    n_input: int
    n_output: int
    n_dups_removed: int
    n_dropped_unusable: int


def _to_utc_daily_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.normalize()


def clean(raw: pd.DataFrame, instrument: Instrument) -> CleanResult:
    """Normalize, validate, repair (same-bar only) and flag raw bars for one instrument."""
    n_input = len(raw)
    df = raw.copy()

    try:
        df.index = _to_utc_daily_index(df.index)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed data error with context
        raise DataQualityError(f"{instrument.symbol}: cannot interpret timestamps: {exc}") from exc
    df.index.name = TIMESTAMP_INDEX_NAME

    missing = [c for c in OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"{instrument.symbol}: raw data missing OHLC columns {missing}")
    for col in QUOTE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[list(_PRICE_AND_QUOTE)].astype("float64")
    df[QUALITY_COLUMN] = 0

    # --- collapse duplicate timestamps (keep last; flag survivors) ---
    dup_ts = df.index[df.index.duplicated(keep=False)].unique()
    df = df.sort_index(kind="stable")
    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    n_dups_removed = n_before - len(df)
    if len(dup_ts):
        df.loc[df.index.isin(dup_ts), QUALITY_COLUMN] |= int(QualityFlag.DUP_RESOLVED)

    # --- drop bars with no usable price at all (cannot be repaired from same-bar data) ---
    unusable = df["open"].isna() & df["close"].isna()
    n_dropped = int(unusable.sum())
    if n_dropped:
        df = df[~unusable]

    # --- impute missing OHLC fields from the SAME bar only, flagging IMPUTED ---
    def _impute(mask: pd.Series, col: str, values: pd.Series) -> None:
        if mask.any():
            df.loc[mask, col] = values[mask]
            df.loc[mask, QUALITY_COLUMN] |= int(QualityFlag.IMPUTED)

    _impute(df["open"].isna() & df["close"].notna(), "open", df["close"])
    _impute(df["close"].isna() & df["open"].notna(), "close", df["open"])
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    _impute(df["high"].isna(), "high", oc_max)
    _impute(df["low"].isna(), "low", oc_min)

    # --- hard invariant: prices must be strictly positive ---
    ohlc = df[list(OHLC_COLUMNS)]
    if bool((ohlc <= 0).to_numpy().any()):
        raise DataQualityError(f"{instrument.symbol}: non-positive OHLC value(s) present")

    # --- repair OHLC inequalities by widening high/low to bound the bar; flag OHLC_REPAIRED ---
    row_high = ohlc.max(axis=1)
    row_low = ohlc.min(axis=1)
    need_repair = (df["high"] < row_high) | (df["low"] > row_low)
    if need_repair.any():
        df.loc[need_repair, "high"] = row_high[need_repair]
        df.loc[need_repair, "low"] = row_low[need_repair]
        df.loc[need_repair, QUALITY_COLUMN] |= int(QualityFlag.OHLC_REPAIRED)

    df[QUALITY_COLUMN] = df[QUALITY_COLUMN].astype("int64")
    df = df[list(POINT_IN_TIME_COLUMNS)]
    return CleanResult(
        frame=df,
        n_input=n_input,
        n_output=len(df),
        n_dups_removed=n_dups_removed,
        n_dropped_unusable=n_dropped,
    )
