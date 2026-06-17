"""Split/dividend adjustment, applied ON READ — never stored in the per-bar cache.

The per-bar cache holds immutable RAW OHLC (truly point-in-time). Corporate actions are stored
separately; here we derive a back-adjustment factor series and apply it to produce a continuous
("adjusted") series for trend signals. Keeping this out of the cache is what lets the cache stay
byte-deterministic and look-ahead-free (acceptance test #9), while still giving research a
dividend/split-continuous price.

ratio convention: split N:1 -> ratio=N (price scales by 1/N before the ex-date); cash dividend
-> ratio is the dividend amount in the quote currency.
"""

from __future__ import annotations

import pandas as pd

from .schema import CORP_ACTION_COLUMNS, OHLC_COLUMNS, QUOTE_COLUMNS

PRICE_COLUMNS: tuple[str, ...] = (*OHLC_COLUMNS, *QUOTE_COLUMNS)


def empty_corporate_actions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CORP_ACTION_COLUMNS))


def _ex_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.normalize()


def adjustment_factors(close: pd.Series, corporate_actions: pd.DataFrame | None) -> pd.Series:
    """Back-adjustment factor per bar (1.0 = unadjusted). Applied to bars strictly BEFORE each
    ex-date, cumulatively, most-recent action affecting the most history."""
    factors = pd.Series(1.0, index=close.index, name="adj_factor")
    if corporate_actions is None or corporate_actions.empty:
        return factors

    ca = corporate_actions.copy()
    ca["ex_date"] = ca["ex_date"].map(_ex_timestamp)
    ca = ca.sort_values("ex_date")

    for action in ca.itertuples(index=False):
        ex = action.ex_date
        before = close.index < ex
        if not before.any():
            continue
        kind = str(action.type).lower()
        ratio = float(action.ratio)
        if kind == "split":
            if ratio <= 0:
                raise ValueError(f"split ratio must be > 0, got {ratio}")
            factor = 1.0 / ratio
        elif kind == "dividend":
            prev_close = float(close.loc[close.index[before][-1]])
            if prev_close <= 0:
                continue
            factor = (prev_close - ratio) / prev_close
        else:
            raise ValueError(f"unknown corporate action type {action.type!r}")
        factors.loc[before] *= factor

    return factors


def apply_adjustment(frame: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """Return a copy of ``frame`` with price/quote columns scaled by ``factors`` (quality kept)."""
    out = frame.copy()
    f = factors.reindex(out.index).fillna(1.0)
    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col] * f
    return out
