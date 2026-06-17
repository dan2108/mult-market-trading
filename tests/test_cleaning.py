"""Unit tests for the same-bar-only cleaning rules (dedup, impute, repair, drop)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfx.data.clean import clean
from tfx.data.schema import QualityFlag
from tfx.instruments import get_instrument

INSTR = get_instrument("EUR/USD")


def _raw(dates, **columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.to_datetime(dates))


def _ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_dedup_keeps_last_and_flags():
    raw = _raw(
        ["2024-01-02", "2024-01-02", "2024-01-03"],
        open=[1.00, 1.05, 1.10], high=[1.10, 1.15, 1.20],
        low=[0.95, 1.00, 1.05], close=[1.05, 1.07, 1.15],
    )
    result = clean(raw, INSTR)
    assert result.n_dups_removed == 1
    assert result.frame.index.is_unique
    assert result.frame.loc[_ts("2024-01-02"), "close"] == 1.07  # kept the last duplicate
    assert int(result.frame.loc[_ts("2024-01-02"), "quality"]) & int(QualityFlag.DUP_RESOLVED)


def test_ohlc_inequality_repaired_and_flagged():
    raw = _raw(["2024-01-02"], open=[1.0], high=[1.0], low=[0.9], close=[1.2])
    result = clean(raw, INSTR)
    row = result.frame.iloc[0]
    assert row["high"] >= max(row["open"], row["close"])
    assert row["low"] <= min(row["open"], row["close"])
    assert int(row["quality"]) & int(QualityFlag.OHLC_REPAIRED)


def test_impute_missing_high_flagged_no_nan():
    raw = _raw(["2024-01-02"], open=[1.0], high=[np.nan], low=[0.9], close=[1.1])
    result = clean(raw, INSTR)
    row = result.frame.iloc[0]
    assert not np.isnan(row["high"])
    assert row["high"] == max(1.0, 1.1)
    assert int(row["quality"]) & int(QualityFlag.IMPUTED)


def test_drop_unusable_bar_counted():
    raw = _raw(
        ["2024-01-02", "2024-01-03"],
        open=[np.nan, 1.1], high=[np.nan, 1.2], low=[np.nan, 1.0], close=[np.nan, 1.15],
    )
    result = clean(raw, INSTR)
    assert result.n_dropped_unusable == 1
    assert len(result.frame) == 1
    assert result.frame.index[0] == _ts("2024-01-03")


def test_index_localized_sorted_and_quote_columns_present():
    raw = _raw(
        ["2024-01-03", "2024-01-02"],  # unsorted on purpose
        open=[1.1, 1.0], high=[1.2, 1.1], low=[1.0, 0.9], close=[1.15, 1.05],
    )
    result = clean(raw, INSTR)
    assert result.frame.index.is_monotonic_increasing
    assert str(result.frame.index.tz) == "UTC"
    # quote columns are always present (NaN when the source lacks them)
    assert "bid" in result.frame.columns and "ask" in result.frame.columns
    assert result.frame["bid"].isna().all()
