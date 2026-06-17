"""Placeholder for a real (live/paid) data source.

v1 intentionally ships no live feed. This stub fails loudly with instructions instead of silently
returning empty data, so a misconfigured `--source` can never masquerade as "no data".
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ...instruments import Instrument
from .base import DataSource

_WIRING_HELP = (
    "Live data sources are not wired in v1 (use `--source fixture`). To add one: subclass "
    "tfx.data.sources.base.DataSource, implement fetch_ohlc (and fetch_corporate_actions if "
    "applicable), and register it in tfx.data.sources.registry. Credentials load from .env via "
    "tfx.config (TFX_DATA__PROVIDER_API_KEY, ...)."
)


class LiveStubSource(DataSource):
    def __init__(self, name: str = "stub"):
        self.name = name

    def fetch_ohlc(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError(f"Source {self.name!r}: {_WIRING_HELP}")
