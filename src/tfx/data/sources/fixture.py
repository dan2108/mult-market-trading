"""Deterministic CSV fixture source.

Reads ``<sample_dir>/<slug>.csv`` (and optional ``<sample_dir>/corporate_actions/<slug>.csv``).
Because the CSVs are static and parsing is deterministic, this source makes the whole pipeline
hermetic and byte-reproducible — ideal for the acceptance-test gate and the CLI demo.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ...instruments import Instrument
from ..schema import (
    CORP_ACTION_COLUMNS,
    OHLC_COLUMNS,
    QUOTE_COLUMNS,
    TIMESTAMP_INDEX_NAME,
)
from .base import DataSource

_KNOWN_INPUT_COLUMNS: tuple[str, ...] = (*OHLC_COLUMNS, *QUOTE_COLUMNS)


class FixtureSource(DataSource):
    name = "fixture"

    def __init__(self, sample_dir: Path | str):
        self.sample_dir = Path(sample_dir)

    def _csv_path(self, instrument: Instrument) -> Path:
        return self.sample_dir / f"{instrument.slug}.csv"

    def fetch_ohlc(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        path = self._csv_path(instrument)
        if not path.exists():
            raise FileNotFoundError(
                f"No fixture for {instrument.symbol!r} at {path}. "
                f"Generate fixtures with `python scripts/generate_sample_data.py` or point "
                f"--sample-dir / TFX_SAMPLE_DIR at a directory containing {instrument.slug}.csv."
            )
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"Fixture {path} must have a 'date' column; got {list(df.columns)}.")
        index = pd.DatetimeIndex(pd.to_datetime(df["date"]), name=TIMESTAMP_INDEX_NAME)
        df = df.drop(columns=["date"]).set_axis(index, axis=0)
        # Keep only recognized OHLC/quote columns (e.g. drop a 'volume' column), preserving order.
        df = df[[c for c in _KNOWN_INPUT_COLUMNS if c in df.columns]]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]

    def fetch_corporate_actions(self, instrument: Instrument) -> pd.DataFrame:
        path = self.sample_dir / "corporate_actions" / f"{instrument.slug}.csv"
        if not path.exists():
            return super().fetch_corporate_actions(instrument)
        ca = pd.read_csv(path)
        ca["ex_date"] = pd.to_datetime(ca["ex_date"])
        return ca[list(CORP_ACTION_COLUMNS)]
