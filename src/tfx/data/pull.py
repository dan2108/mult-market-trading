"""Orchestration: fetch -> clean -> cache, for a list of instruments over a date range.

This is the write side of the data layer. The read side (:mod:`tfx.data.loader`) is what
`backtest` consumes. Both go through the same cleaned, point-in-time representation, so what you
backtest is exactly what was stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..instruments import get_instrument
from .cache import write_instrument, write_manifest
from .clean import clean
from .sources import get_source


@dataclass(frozen=True)
class PullResult:
    cache_dir: Path
    manifest: dict

    @property
    def n_instruments(self) -> int:
        return len(self.manifest.get("instruments", {}))


def _as_date(value: date | str) -> date:
    if type(value) is date:
        return value
    return pd.Timestamp(value).normalize().date()


def pull(
    symbols: list[str],
    start: date | str,
    end: date | str,
    *,
    source: str = "fixture",
    cache_dir: Path | str,
    sample_dir: Path | str | None = None,
) -> PullResult:
    """Fetch, clean and cache ``symbols`` over ``[start, end]``; write the manifest. Returns the
    manifest and cache directory. Deterministic for a deterministic source."""
    start, end = _as_date(start), _as_date(end)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    src = get_source(source, sample_dir=sample_dir)
    instruments = [get_instrument(s) for s in symbols]

    entries: dict[str, dict] = {}
    for instrument in instruments:
        raw = src.fetch_ohlc(instrument, start, end)
        cleaned = clean(raw, instrument)
        corporate_actions = src.fetch_corporate_actions(instrument)
        entries[instrument.slug] = write_instrument(
            cache_dir, instrument, cleaned, corporate_actions
        )

    manifest = write_manifest(
        cache_dir,
        source=source,
        symbols=[i.symbol for i in instruments],
        start=start,
        end=end,
        entries=entries,
    )
    return PullResult(cache_dir=Path(cache_dir), manifest=manifest)
