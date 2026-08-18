"""Tests for the stooq snapshot-backed source.

Everything here is hermetic: it runs against committed mini-snapshots in tests/fixtures/stooq/
(offline=True, no network). The one live-endpoint smoke test is opt-in via `pytest -m network`.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tfx.data.pull import pull
from tfx.data.schema import TIMESTAMP_INDEX_NAME
from tfx.data.sources import available_sources, get_source
from tfx.data.sources.stooq import (
    STOOQ_SYMBOLS,
    SnapshotIntegrityError,
    StooqSource,
    stooq_symbol,
)
from tfx.instruments import BASKET, AssetClass, Instrument, get_instrument

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "stooq"
RAW_DIR = SNAPSHOT_DIR.parent  # get_source appends /stooq
START, END = date(2024, 3, 4), date(2024, 3, 15)


def _source() -> StooqSource:
    return StooqSource(SNAPSHOT_DIR, offline=True)


def test_parse_contract_fx_shape():
    bars = _source().fetch_ohlc(get_instrument("EUR/USD"), START, END)
    assert list(bars.columns) == ["open", "high", "low", "close"]  # FX file has no volume
    assert bars.index.name == TIMESTAMP_INDEX_NAME
    assert len(bars) == 10
    assert bars["close"].iloc[0] == pytest.approx(1.08510)


def test_parse_contract_raw_warts_pass_through():
    """The source returns RAW bars: the duplicate and out-of-order rows in the SP_500 snapshot
    must survive (cleaning, not the source, dedups and sorts) and volume must be dropped."""
    bars = _source().fetch_ohlc(get_instrument("S&P 500"), START, END)
    assert "volume" not in bars.columns
    assert len(bars) == 11                          # 10 days + 1 duplicate
    assert bars.index.duplicated().any()            # duplicate preserved
    assert not bars.index.is_monotonic_increasing   # out-of-order row preserved


def test_range_filter_inclusive():
    bars = _source().fetch_ohlc(get_instrument("EUR/USD"), date(2024, 3, 5), date(2024, 3, 8))
    assert len(bars) == 4
    assert bars.index.min() == pd.Timestamp("2024-03-05")
    assert bars.index.max() == pd.Timestamp("2024-03-08")


def test_every_basket_symbol_is_mapped():
    for instrument in BASKET:
        assert stooq_symbol(instrument)  # raises KeyError if unmapped
    assert set(STOOQ_SYMBOLS) >= {i.symbol for i in BASKET}


def test_unmapped_symbol_raises_loudly():
    fake = Instrument(
        symbol="FAKE/PAIR", name="Fake", asset_class=AssetClass.FX,
        session_close_tz="America/New_York",
    )
    with pytest.raises(KeyError, match="No stooq symbol mapped"):
        _source().fetch_ohlc(fake, START, END)


def test_offline_missing_snapshot_raises_with_instructions(tmp_path):
    source = StooqSource(tmp_path, offline=True)
    with pytest.raises(FileNotFoundError, match="offline"):
        source.fetch_ohlc(get_instrument("EUR/USD"), START, END)


def test_offline_and_refresh_contradictory(tmp_path):
    with pytest.raises(ValueError):
        StooqSource(tmp_path, offline=True, refresh=True)


def test_snapshot_tamper_and_missing_sidecar_detected(tmp_path):
    snap = tmp_path / "stooq"
    shutil.copytree(SNAPSHOT_DIR, snap)
    source = StooqSource(snap, offline=True)
    eur = get_instrument("EUR/USD")

    csv = snap / "EUR_USD.csv"
    csv.write_bytes(csv.read_bytes() + b"2024-03-18,1.0,1.1,0.9,1.05\n")
    with pytest.raises(SnapshotIntegrityError, match="sha256"):
        source.fetch_ohlc(eur, START, END)

    (snap / "SP_500.snapshot.json").unlink()
    with pytest.raises(SnapshotIntegrityError, match="sidecar"):
        source.fetch_ohlc(get_instrument("S&P 500"), START, END)


def test_write_snapshot_roundtrip_and_stable_sidecar(tmp_path):
    source = StooqSource(tmp_path)
    eur = get_instrument("EUR/USD")
    raw = (SNAPSHOT_DIR / "EUR_USD.csv").read_bytes()

    source.write_snapshot(eur, raw)
    first_sidecar = (tmp_path / "EUR_USD.snapshot.json").read_bytes()
    source.write_snapshot(eur, raw)  # identical bytes -> identical sidecar (no wall-clock)
    second_sidecar = (tmp_path / "EUR_USD.snapshot.json").read_bytes()

    assert first_sidecar == second_sidecar
    offline = StooqSource(tmp_path, offline=True)
    assert offline.fetch_ohlc(eur, START, END).equals(_source().fetch_ohlc(eur, START, END))


def test_registry_wires_stooq():
    assert "stooq" in available_sources()
    source = get_source("stooq", raw_dir=RAW_DIR, offline=True)
    assert isinstance(source, StooqSource)
    assert source.snapshot_dir == RAW_DIR / "stooq"


def test_determinism_snapshot_to_cache_bytes(tmp_path):
    """The re-serialization determinism guarantee: from the same snapshot, two full
    pull -> clean -> cache runs produce byte-identical parquet + manifest (the raw warts in the
    SP_500 snapshot go through dedup/sort on the way)."""
    symbols = ["EUR/USD", "S&P 500"]
    results = []
    for sub in ("one", "two"):
        cache_dir = tmp_path / sub
        pull(
            symbols, START, END,
            source="stooq", cache_dir=cache_dir, raw_dir=RAW_DIR, offline=True,
        )
        results.append(cache_dir)

    for name in ("manifest.json", "EUR_USD.parquet", "SP_500.parquet"):
        first = (results[0] / name).read_bytes()
        second = (results[1] / name).read_bytes()
        assert first == second, f"{name} not byte-identical across pulls"


def test_html_challenge_detected_on_download(tmp_path, monkeypatch):
    """A JS anti-bot challenge (HTML instead of CSV) must fail loudly with remediation, never be
    written as a snapshot."""
    import requests as requests_module

    class _FakeResponse:
        content = b"<!DOCTYPE html><html><body>verify your browser</body></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests_module, "get", lambda *a, **k: _FakeResponse())
    source = StooqSource(tmp_path / "stooq")
    from tfx.data.errors import DataError

    with pytest.raises(DataError, match="anti-bot"):
        source.fetch_ohlc(get_instrument("EUR/USD"), START, END)
    assert not (tmp_path / "stooq" / "EUR_USD.csv").exists()


def test_cli_snapshot_registers_browser_download(tmp_path):
    """`tfx data snapshot` turns a manually-downloaded CSV into a verified snapshot usable by an
    offline pull (the anti-bot escape hatch)."""
    from typer.testing import CliRunner

    from tfx.cli import app

    downloaded = tmp_path / "eurusd.csv"
    downloaded.write_bytes((SNAPSHOT_DIR / "EUR_USD.csv").read_bytes())
    raw_dir = tmp_path / "raw"

    result = CliRunner().invoke(
        app,
        ["data", "snapshot", "--symbol", "EUR/USD", "--file", str(downloaded),
         "--raw-dir", str(raw_dir)],
    )
    assert result.exit_code == 0, result.output

    source = StooqSource(raw_dir / "stooq", offline=True)
    bars = source.fetch_ohlc(get_instrument("EUR/USD"), START, END)
    assert len(bars) == 10


@pytest.mark.network
def test_live_endpoint_smoke(tmp_path):
    """Opt-in (pytest -m network): one real download, proving the URL scheme and parser against
    the live endpoint. Not part of the hermetic gate."""
    source = StooqSource(tmp_path, offline=False)
    bars = source.fetch_ohlc(get_instrument("EUR/USD"), date(2024, 1, 2), date(2024, 1, 31))
    assert not bars.empty
    assert {"open", "high", "low", "close"} <= set(bars.columns)
    assert (tmp_path / "EUR_USD.csv").exists()
    assert (tmp_path / "EUR_USD.snapshot.json").exists()
