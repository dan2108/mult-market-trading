"""Data-quality report: per-instrument coverage, date ranges, gap classification, repairs.

Gaps are computed against each instrument's expected trading days (asset-class aware). A run of
consecutive missing expected days no longer than the tolerance is treated as a holiday/expected
gap; a longer run is flagged as *unexpected* and listed explicitly so it can't hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..config import get_settings
from ..instruments import get_instrument
from .cache import read_instrument, read_manifest
from .schema import QualityFlag

TABLE_COLUMNS = ("Symbol", "Cls", "Bars", "Cov%", "First", "Last", "Imp", "Rep", "Gaps")

# Compact asset-class codes so the report fits an 80-column terminal.
_CLASS_CODE = {
    "fx": "FX", "equity_index": "IDX", "commodity": "COM", "crypto": "CRY", "equity": "EQ",
}


@dataclass(frozen=True)
class InstrumentQuality:
    symbol: str
    asset_class: str
    rows: int
    first: str | None
    last: str | None
    coverage: float
    n_expected: int
    n_missing_expected: int
    n_imputed: int
    n_repaired: int
    n_dups_removed: int
    n_dropped: int
    n_holiday_gaps: int
    n_unexpected_gaps: int
    unexpected_runs: list[tuple[str, str, int]]
    # (date, one-bar close return) beyond the extreme-move threshold — the roll-artifact screen
    # for futures continuations (an unadjusted roll shows up as a huge fake one-day return that
    # would corrupt a trend signal), and a general fat-finger/bad-tick screen for everything else.
    extreme_moves: list[tuple[str, float]]

    @property
    def years_covered(self) -> float:
        if not self.first or not self.last:
            return 0.0
        span = pd.Timestamp(self.last) - pd.Timestamp(self.first)
        return span.days / 365.25

    def table_row(self) -> list[str]:
        cls = _CLASS_CODE.get(self.asset_class, self.asset_class[:3].upper())
        return [
            self.symbol, cls, str(self.rows), f"{self.coverage * 100:.1f}",
            self.first or "-", self.last or "-", str(self.n_imputed), str(self.n_repaired),
            f"{self.n_holiday_gaps}/{self.n_unexpected_gaps}",
        ]


@dataclass(frozen=True)
class QualityReport:
    start: str
    end: str
    instruments: list[InstrumentQuality]

    @property
    def has_unexpected_gaps(self) -> bool:
        return any(q.n_unexpected_gaps for q in self.instruments)

    @property
    def cleaning_anomalies(self) -> list[tuple[str, int, int]]:
        """(symbol, dups_removed, dropped) for instruments that needed dedup/drop."""
        return [
            (q.symbol, q.n_dups_removed, q.n_dropped)
            for q in self.instruments
            if q.n_dups_removed or q.n_dropped
        ]

    def strict_failures(self, *, min_years: float = 10.0) -> list[str]:
        """Provider-onboarding gate: human-readable failures, empty when the data is fit for the
        validate gate. Crypto is exempt from the history minimum (the asset class is young)."""
        failures: list[str] = []
        for q in self.instruments:
            if q.rows == 0:
                failures.append(f"{q.symbol}: no bars in range")
                continue
            if q.asset_class != "crypto" and q.years_covered < min_years:
                failures.append(
                    f"{q.symbol}: only {q.years_covered:.1f}y of history "
                    f"(< {min_years:g}y minimum)"
                )
            if q.n_unexpected_gaps:
                runs = ", ".join(f"{a}..{b} ({n}d)" for a, b, n in q.unexpected_runs[:3])
                failures.append(f"{q.symbol}: {q.n_unexpected_gaps} unexpected gap(s): {runs}")
            for day, move in q.extreme_moves[:3]:
                failures.append(
                    f"{q.symbol}: extreme one-bar move {move:+.1%} on {day} "
                    f"(roll artifact / bad tick? verify or drop the instrument)"
                )
        return failures


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts.normalize()


def _as_date(value: date | str) -> date:
    if type(value) is date:
        return value
    return pd.Timestamp(value).normalize().date()


def _expected_days(symbol: str, start: date | str, end: date | str) -> list[date]:
    closed = get_instrument(symbol).weekly_closed_days
    days = pd.date_range(start=_as_date(start), end=_as_date(end), freq="D")
    return [d.date() for d in days if d.weekday() not in closed]


def _missing_runs(expected: list[date], present: set[date]) -> list[list[date]]:
    runs: list[list[date]] = []
    current: list[date] = []
    for day in expected:
        if day not in present:
            current.append(day)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


# One-bar |close return| beyond this is flagged by the strict gate. Wide enough for real crisis
# moves in liquid dailies; an unadjusted futures roll or a bad tick is typically far beyond it.
DEFAULT_EXTREME_MOVE = 0.25


def build_instrument_quality(
    cache_dir: object,
    symbol: str,
    start: date | str,
    end: date | str,
    *,
    manifest: dict,
    max_weekday_gap_days: int,
    extreme_move_threshold: float = DEFAULT_EXTREME_MOVE,
) -> InstrumentQuality:
    instrument = get_instrument(symbol)
    entry = manifest.get("instruments", {}).get(instrument.slug, {})
    frame = read_instrument(cache_dir, instrument, manifest=manifest)
    frame = frame.loc[(frame.index >= _utc(start)) & (frame.index <= _utc(end))]

    quality = frame["quality"].astype("int64")
    n_imputed = int(((quality & int(QualityFlag.IMPUTED)) != 0).sum())
    n_repaired = int(((quality & int(QualityFlag.OHLC_REPAIRED)) != 0).sum())

    expected = _expected_days(symbol, start, end)
    present = {ts.date() for ts in frame.index}
    runs = _missing_runs(expected, present)
    holiday_gaps = sum(1 for r in runs if len(r) <= max_weekday_gap_days)
    unexpected = [
        (r[0].isoformat(), r[-1].isoformat(), len(r))
        for r in runs
        if len(r) > max_weekday_gap_days
    ]
    n_missing = sum(len(r) for r in runs)
    n_expected = len(expected)
    coverage = (len(frame) / n_expected) if n_expected else 1.0

    # native-calendar one-bar close returns (the cache holds no pads, so consecutive rows are
    # consecutive trading sessions for this instrument)
    returns = frame["close"].pct_change()
    extreme = returns[returns.abs() > extreme_move_threshold]
    extreme_moves = [
        (ts.date().isoformat(), float(value)) for ts, value in extreme.items()
    ]

    return InstrumentQuality(
        symbol=instrument.symbol,
        asset_class=instrument.asset_class.value,
        rows=len(frame),
        first=frame.index.min().date().isoformat() if len(frame) else None,
        last=frame.index.max().date().isoformat() if len(frame) else None,
        coverage=coverage,
        n_expected=n_expected,
        n_missing_expected=n_missing,
        n_imputed=n_imputed,
        n_repaired=n_repaired,
        n_dups_removed=int(entry.get("n_dups_removed", 0)),
        n_dropped=int(entry.get("n_dropped_unusable", 0)),
        n_holiday_gaps=holiday_gaps,
        n_unexpected_gaps=len(unexpected),
        unexpected_runs=unexpected,
        extreme_moves=extreme_moves,
    )


def build_report(
    cache_dir: object,
    symbols: list[str],
    start: date | str,
    end: date | str,
    *,
    manifest: dict | None = None,
    max_weekday_gap_days: int | None = None,
    extreme_move_threshold: float = DEFAULT_EXTREME_MOVE,
) -> QualityReport:
    manifest = manifest or read_manifest(cache_dir)
    if max_weekday_gap_days is None:
        max_weekday_gap_days = get_settings().max_weekday_gap_days
    items = [
        build_instrument_quality(
            cache_dir, s, start, end, manifest=manifest,
            max_weekday_gap_days=max_weekday_gap_days,
            extreme_move_threshold=extreme_move_threshold,
        )
        for s in symbols
    ]
    return QualityReport(start=str(_as_date(start)), end=str(_as_date(end)), instruments=items)
