"""Unit tests for on-read split/dividend back-adjustment."""

from __future__ import annotations

import pandas as pd

from tfx.data.adjust import adjustment_factors, apply_adjustment, empty_corporate_actions


def _close(dates, values) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates, utc=True), name="close")


def _ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_no_actions_factor_is_one():
    close = _close(["2024-01-02", "2024-01-03"], [100.0, 101.0])
    factors = adjustment_factors(close, empty_corporate_actions())
    assert (factors == 1.0).all()


def test_split_back_adjustment():
    close = _close(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 50.0])
    actions = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2024-01-04")], "type": ["split"], "ratio": [2.0]}
    )
    factors = adjustment_factors(close, actions)
    assert factors.loc[_ts("2024-01-02")] == 0.5
    assert factors.loc[_ts("2024-01-03")] == 0.5
    assert factors.loc[_ts("2024-01-04")] == 1.0

    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 50.0], "high": [100.0, 100.0, 50.0],
            "low": [100.0, 100.0, 50.0], "close": [100.0, 100.0, 50.0],
            "bid": [float("nan")] * 3, "ask": [float("nan")] * 3,
        },
        index=close.index,
    )
    adjusted = apply_adjustment(frame, factors)
    # the pre-split level is now continuous with the post-split level
    assert adjusted["close"].iloc[0] == 50.0
    assert adjusted["close"].iloc[2] == 50.0


def test_dividend_back_adjustment():
    close = _close(["2024-01-02", "2024-01-03"], [100.0, 100.0])
    actions = pd.DataFrame(
        {"ex_date": [pd.Timestamp("2024-01-03")], "type": ["dividend"], "ratio": [1.0]}
    )
    factors = adjustment_factors(close, actions)
    assert abs(factors.loc[_ts("2024-01-02")] - 0.99) < 1e-12  # (100 - 1) / 100
    assert factors.loc[_ts("2024-01-03")] == 1.0
