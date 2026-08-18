"""Source registry — maps a source name to a constructed :class:`DataSource`."""

from __future__ import annotations

from pathlib import Path

from ...config import get_settings
from .base import DataSource
from .fixture import FixtureSource
from .live_stub import LiveStubSource
from .stooq import StooqSource

FIXTURE = "fixture"
STOOQ = "stooq"
# Names accepted as not-yet-implemented live placeholders (fail loudly when actually used).
_LIVE_PLACEHOLDERS = frozenset({"stub", "live"})


def available_sources() -> list[str]:
    return [FIXTURE, STOOQ, *sorted(_LIVE_PLACEHOLDERS)]


def get_source(
    name: str | None = None,
    *,
    sample_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> DataSource:
    """Construct a data source by name. Defaults to the fixture source.

    ``raw_dir``/``offline``/``refresh`` apply to snapshot-backed remote sources (stooq);
    ``sample_dir`` applies to the fixture source. Irrelevant options are ignored.
    """
    name = (name or FIXTURE).lower()
    if name == FIXTURE:
        sd = sample_dir if sample_dir is not None else get_settings().sample_dir
        return FixtureSource(sample_dir=sd)
    if name == STOOQ:
        rd = raw_dir if raw_dir is not None else get_settings().raw_dir
        return StooqSource(snapshot_dir=Path(rd) / STOOQ, offline=offline, refresh=refresh)
    if name in _LIVE_PLACEHOLDERS:
        return LiveStubSource(name)
    raise ValueError(f"Unknown source {name!r}. Available: {', '.join(available_sources())}")
