"""Shared pytest fixtures.

A session-scoped cache is built once from the committed deterministic fixtures, so the acceptance
tests run against the real pull→clean→align→cache pipeline without re-pulling per test.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from tfx.data.cache import read_manifest
from tfx.data.pull import pull
from tfx.instruments import default_symbols

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return REPO_ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def symbols() -> list[str]:
    return default_symbols()


@pytest.fixture(scope="session")
def window() -> tuple[date, date]:
    return date(2023, 1, 2), date(2024, 12, 31)


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory, symbols, window, sample_dir) -> Path:
    start, end = window
    path = tmp_path_factory.mktemp("cache")
    pull(symbols, start, end, source="fixture", cache_dir=path, sample_dir=sample_dir)
    return path


@pytest.fixture(scope="session")
def manifest(cache_dir) -> dict:
    return read_manifest(cache_dir)
