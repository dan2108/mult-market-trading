"""Pluggable data sources.

A source returns RAW bars; normalization/validation/flagging happen in :mod:`tfx.data.clean`,
so the cache layout and quality guarantees are identical no matter where the data came from.
"""

from .base import DataSource
from .fixture import FixtureSource
from .live_stub import LiveStubSource
from .registry import available_sources, get_source

__all__ = [
    "DataSource",
    "FixtureSource",
    "LiveStubSource",
    "available_sources",
    "get_source",
]
