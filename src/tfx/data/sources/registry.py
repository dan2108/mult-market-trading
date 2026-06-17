"""Source registry — maps a source name to a constructed :class:`DataSource`."""

from __future__ import annotations

from pathlib import Path

from ...config import get_settings
from .base import DataSource
from .fixture import FixtureSource
from .live_stub import LiveStubSource

FIXTURE = "fixture"
# Names accepted as not-yet-implemented live placeholders (fail loudly when actually used).
_LIVE_PLACEHOLDERS = frozenset({"stub", "live"})


def available_sources() -> list[str]:
    return [FIXTURE, *sorted(_LIVE_PLACEHOLDERS)]


def get_source(name: str | None = None, *, sample_dir: Path | str | None = None) -> DataSource:
    """Construct a data source by name. Defaults to the fixture source."""
    name = (name or FIXTURE).lower()
    if name == FIXTURE:
        sd = sample_dir if sample_dir is not None else get_settings().sample_dir
        return FixtureSource(sample_dir=sd)
    if name in _LIVE_PLACEHOLDERS:
        return LiveStubSource(name)
    raise ValueError(f"Unknown source {name!r}. Available: {', '.join(available_sources())}")
