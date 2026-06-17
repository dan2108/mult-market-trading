"""The data-source interface every adapter implements.

A ``DataSource`` returns RAW bars exactly as the upstream provides them — possibly with warts
(duplicate timestamps, small gaps, the occasional missing field). Normalization, validation and
quality flagging are deliberately NOT done here; they belong to :mod:`tfx.data.clean`, so every
source benefits from the same hardening and the cache layout stays source-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from ...instruments import Instrument
from ..schema import CORP_ACTION_COLUMNS


class DataSource(ABC):
    """Abstract source of raw daily bars and (optional) corporate actions."""

    #: short, stable identifier; recorded in the cache manifest for provenance
    name: str = "base"

    @abstractmethod
    def fetch_ohlc(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        """Return RAW daily bars for ``[start, end]`` inclusive.

        Contract:
          * Indexed by date (tz-naive session date, or tz-aware — cleaning normalizes to UTC).
          * Columns are a subset of ``{open, high, low, close, bid, ask}``; no quality column.
          * May be returned unsorted / with duplicates / with minor NaNs — cleaning fixes those.
          * May be empty if the source has no data for the range (cleaning reports the gap).
        """
        raise NotImplementedError

    def fetch_corporate_actions(self, instrument: Instrument) -> pd.DataFrame:
        """Return splits/dividends as columns ``(ex_date, type, ratio)``. Default: none.

        Correct for FX and commodities, and an acceptable v1 default for index proxies. Override
        for sources that supply corporate actions.
        """
        return pd.DataFrame(columns=list(CORP_ACTION_COLUMNS))
