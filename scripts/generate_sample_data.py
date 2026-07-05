#!/usr/bin/env python
"""Generate deterministic sample OHLC fixtures for the v1 basket.

Run:  python scripts/generate_sample_data.py

Writes ``data/sample/<slug>.csv`` (committed). Pure stdlib and fully deterministic — fixed
per-instrument seeds, no wall-clock — so re-running reproduces byte-identical files. These are
SYNTHETIC fixtures that back the CLI demo and most acceptance tests; they are NOT market data.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
START = date(2023, 1, 2)
END = date(2024, 12, 31)

# A handful of weekday exchange holidays so equity instruments have realistic holiday gaps that
# differ from FX/commodities — this exercises gap classification and cross-instrument alignment.
US_EQ_HOLIDAYS = {
    date(2023, 1, 16), date(2023, 2, 20), date(2023, 5, 29), date(2023, 7, 4),
    date(2023, 9, 4), date(2023, 11, 23), date(2023, 12, 25),
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 5, 27),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28), date(2024, 12, 25),
}
DE_EQ_HOLIDAYS = {
    date(2023, 4, 7), date(2023, 4, 10), date(2023, 5, 1), date(2023, 10, 3),
    date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 3, 29), date(2024, 4, 1), date(2024, 5, 1), date(2024, 10, 3),
    date(2024, 12, 25), date(2024, 12, 26),
}
JP_EQ_HOLIDAYS = {
    date(2023, 1, 2), date(2023, 1, 9), date(2023, 2, 23), date(2023, 5, 3),
    date(2023, 5, 4), date(2023, 5, 5), date(2023, 11, 23),
    date(2024, 1, 1), date(2024, 2, 12), date(2024, 4, 29), date(2024, 5, 3),
    date(2024, 8, 12), date(2024, 11, 4),
}
UK_EQ_HOLIDAYS = {
    date(2023, 4, 7), date(2023, 4, 10), date(2023, 5, 1), date(2023, 5, 8),
    date(2023, 8, 28), date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 3, 29), date(2024, 4, 1), date(2024, 5, 6), date(2024, 8, 26),
    date(2024, 12, 25), date(2024, 12, 26),
}

# Keys are instrument slugs (Instrument.slug). Existing entries keep their original seeds so the
# original seven files regenerate byte-identically; additions only ever append new seeds.
SPECS: dict[str, dict] = {
    "EUR_USD": dict(seed=1001, p0=1.08000, vol=0.005, dp=5, fx=True, holidays=frozenset()),
    "AUD_JPY": dict(seed=1002, p0=95.000, vol=0.006, dp=3, fx=True, holidays=frozenset()),
    "GBP_CHF": dict(seed=1003, p0=1.12000, vol=0.004, dp=5, fx=True, holidays=frozenset()),
    "SP_500": dict(seed=1004, p0=4500.00, vol=0.009, dp=2, fx=False, holidays=US_EQ_HOLIDAYS),
    "DAX": dict(seed=1005, p0=15000.00, vol=0.010, dp=2, fx=False, holidays=DE_EQ_HOLIDAYS),
    "Gold": dict(seed=1006, p0=1900.00, vol=0.008, dp=2, fx=False, holidays=frozenset()),
    "Oil": dict(seed=1007, p0=78.00, vol=0.015, dp=2, fx=False, holidays=frozenset()),
    "USD_JPY": dict(seed=1008, p0=145.000, vol=0.006, dp=3, fx=True, holidays=frozenset()),
    "GBP_USD": dict(seed=1009, p0=1.27000, vol=0.005, dp=5, fx=True, holidays=frozenset()),
    "USD_CAD": dict(seed=1010, p0=1.35000, vol=0.004, dp=5, fx=True, holidays=frozenset()),
    "NZD_USD": dict(seed=1011, p0=0.61000, vol=0.006, dp=5, fx=True, holidays=frozenset()),
    "Nikkei_225": dict(seed=1012, p0=33000.00, vol=0.011, dp=2, fx=False,
                       holidays=JP_EQ_HOLIDAYS),
    "FTSE_100": dict(seed=1013, p0=7600.00, vol=0.008, dp=2, fx=False, holidays=UK_EQ_HOLIDAYS),
    "Silver": dict(seed=1014, p0=24.000, vol=0.014, dp=3, fx=False, holidays=frozenset()),
    "Nat_Gas": dict(seed=1015, p0=2.800, vol=0.025, dp=3, fx=False, holidays=frozenset()),
    "BTC_USD": dict(seed=1016, p0=42000.00, vol=0.030, dp=2, fx=False, holidays=frozenset(),
                    weekends=True),
    "ETH_USD": dict(seed=1017, p0=2500.00, vol=0.035, dp=2, fx=False, holidays=frozenset(),
                    weekends=True),
}
FX_HALF_SPREAD = 3e-5  # fractional half-spread for synthetic bid/ask


def trading_days(start: date, end: date, holidays: frozenset[date], weekends: bool = False):
    d, one = start, timedelta(days=1)
    while d <= end:
        if (weekends or d.weekday() < 5) and d not in holidays:
            yield d
        d += one


def generate(spec: dict) -> list[dict]:
    rng = random.Random(spec["seed"])
    dp, fx, vol = spec["dp"], spec["fx"], spec["vol"]
    price = spec["p0"]
    rows: list[dict] = []
    days = trading_days(START, END, spec["holidays"], spec.get("weekends", False))
    for t, d in enumerate(days):
        trend = 0.0004 * math.sin(t / 40.0) + 0.0003 * math.sin(t / 13.0 + 1.0)
        ret = trend + rng.gauss(0, vol)
        open_ = price * (1 + rng.gauss(0, vol * 0.3))
        close = price * (1 + ret)
        hi = max(open_, close) * (1 + abs(rng.gauss(0, vol * 0.5)))
        lo = min(open_, close) * (1 - abs(rng.gauss(0, vol * 0.5)))
        o, c = round(open_, dp), round(close, dp)
        h, low = round(max(hi, o, c), dp), round(min(lo, o, c), dp)
        row = {
            "date": d.isoformat(),
            "open": f"{o:.{dp}f}", "high": f"{h:.{dp}f}",
            "low": f"{low:.{dp}f}", "close": f"{c:.{dp}f}",
        }
        if fx:
            row["bid"] = f"{c * (1 - FX_HALF_SPREAD):.{dp}f}"
            row["ask"] = f"{c * (1 + FX_HALF_SPREAD):.{dp}f}"
        rows.append(row)
        price = close
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, spec in SPECS.items():
        rows = generate(spec)
        fields = ["date", "open", "high", "low", "close"] + (["bid", "ask"] if spec["fx"] else [])
        path = OUT_DIR / f"{slug}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} bars)")


if __name__ == "__main__":
    main()
