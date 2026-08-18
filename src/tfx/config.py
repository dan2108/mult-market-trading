"""Typed runtime configuration.

Settings load from environment (prefix ``TFX_``, nested delimiter ``__``) and an optional
``.env`` file. Secrets belong in ``.env`` (gitignored), never in code. Core data functions accept
explicit paths so tests stay hermetic; the CLI fills them in from these settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .instruments import default_symbols


class DataSettings(BaseModel):
    """Data-source selection and (later) provider credentials."""

    provider: str = "fixture"            # fixture | <live providers wired later>
    provider_api_key: str | None = None
    provider_base_url: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TFX_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Path("data/cache")
    sample_dir: Path = Path("data/sample")
    # Raw provider snapshots (verbatim HTTP bytes + sha256 sidecars), one subdir per source.
    # Gitignored build input, like data/cache -- committed mini-snapshots live in tests/fixtures.
    raw_dir: Path = Path("data/raw")

    # Max consecutive missing *weekday* bars tolerated before a gap is flagged "unexpected"
    # (covers long holiday weekends). Weekend gaps are always expected.
    max_weekday_gap_days: int = 4

    symbols: list[str] = Field(default_factory=default_symbols)

    data: DataSettings = Field(default_factory=DataSettings)


def get_settings() -> Settings:
    """Construct settings from environment / .env. Cheap; call where needed."""
    return Settings()
