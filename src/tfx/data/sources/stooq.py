"""Stooq daily-EOD source with snapshot-on-pull determinism.

Stooq (https://stooq.com) is a keyless daily-EOD endpoint covering FX pairs, cash equity
indices, spot metals, futures continuations and crypto in one place — research-grade breadth for
the `validate` gate. It is unofficial and has no SLA; Phase-2 live prices come from the broker
feed, and parity is preserved through `tfx.data.loader`, not the provider.

Determinism model — snapshot-on-pull:
- A download writes the raw HTTP bytes VERBATIM to ``<snapshot_dir>/<slug>.csv`` plus a sidecar
  ``<slug>.snapshot.json`` (url, sha256) BEFORE any parsing. Parsing always reads the snapshot,
  so what was parsed is exactly what is on disk.
- Byte-determinism applies to re-serialization: from a given snapshot, fetch -> clean -> cache
  reproduces identical parquet bytes + manifest. A re-download may legitimately differ (upstream
  revisions); the sidecar hash and the cache manifest's content hashes make any drift evident,
  never silent.
- The sidecar carries no wall-clock timestamp: identical bytes -> identical sidecar. When a
  snapshot was taken is version-control metadata, not data.

Modes: ``offline=True`` never touches the network (missing snapshot raises with instructions —
this is what CI uses, against committed mini-snapshots). Default: reuse the snapshot when
present, download only when missing. ``refresh=True`` forces a re-download.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ...instruments import Instrument
from ..errors import DataError, DataQualityError
from ..schema import OHLC_COLUMNS, TIMESTAMP_INDEX_NAME
from .base import DataSource

#: canonical symbol -> stooq query symbol. Explicit map, extended per basket addition; an
#: unmapped instrument RAISES (same no-silent-fallthrough rule as tfx.costs.params.get_params).
STOOQ_SYMBOLS: dict[str, str] = {
    "EUR/USD": "eurusd",
    "USD/JPY": "usdjpy",
    "GBP/USD": "gbpusd",
    "AUD/JPY": "audjpy",
    "USD/CAD": "usdcad",
    "GBP/CHF": "gbpchf",
    "NZD/USD": "nzdusd",
    "S&P 500": "^spx",
    "DAX": "^dax",
    "Nikkei 225": "^nkx",
    "FTSE 100": "^ukx",
    "Gold": "xauusd",    # spot XAU/USD
    "Silver": "xagusd",  # spot XAG/USD
    "Oil": "cl.f",       # WTI front-month continuation (roll artifacts screened at data report)
    "Nat Gas": "ng.f",   # Henry Hub front-month continuation (same screening applies)
    "BTC/USD": "btcusd",
    "ETH/USD": "ethusd",
}

_BASE_URL = "https://stooq.com/q/d/l/"
_TIMEOUT_SECONDS = 30.0
_SIDECAR_SUFFIX = ".snapshot.json"
# A plain browser UA: stooq 404s the default python-requests agent. Some networks (datacenter
# IPs) still get a JavaScript anti-bot challenge -- detected below, with remediation.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


class SnapshotIntegrityError(DataError):
    """A snapshot's bytes do not match its recorded sha256 (or the sidecar is missing)."""


def stooq_symbol(instrument: Instrument) -> str:
    """The stooq query symbol for an instrument. Raises KeyError if unmapped."""
    try:
        return STOOQ_SYMBOLS[instrument.symbol]
    except KeyError:
        raise KeyError(
            f"No stooq symbol mapped for {instrument.symbol!r}. Add it to "
            f"tfx.data.sources.stooq.STOOQ_SYMBOLS (known: {', '.join(sorted(STOOQ_SYMBOLS))})."
        ) from None


class StooqSource(DataSource):
    name = "stooq"

    def __init__(self, snapshot_dir: Path | str, *, offline: bool = False,
                 refresh: bool = False):
        self.snapshot_dir = Path(snapshot_dir)
        self.offline = offline
        self.refresh = refresh
        if offline and refresh:
            raise ValueError("offline=True and refresh=True are contradictory")

    # --- snapshot layer -----------------------------------------------------------------------
    def _csv_path(self, instrument: Instrument) -> Path:
        return self.snapshot_dir / f"{instrument.slug}.csv"

    def _sidecar_path(self, instrument: Instrument) -> Path:
        return self.snapshot_dir / f"{instrument.slug}{_SIDECAR_SUFFIX}"

    def _url(self, instrument: Instrument) -> str:
        # full history, daily interval; [start, end] is filtered at parse time so one snapshot
        # serves any window (and re-pulls of a narrower window reuse identical bytes).
        return f"{_BASE_URL}?s={stooq_symbol(instrument)}&i=d"

    def _download(self, instrument: Instrument) -> bytes:
        url = self._url(instrument)
        try:
            response = requests.get(
                url, timeout=_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT}
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DataError(
                f"stooq download failed for {instrument.symbol!r} ({url}): {exc}"
            ) from exc
        content = response.content
        if content.lstrip()[:1] == b"<":  # HTML, not CSV: anti-bot challenge or error page
            raise DataError(
                f"stooq answered {instrument.symbol!r} with an HTML page instead of CSV -- "
                f"usually a JavaScript anti-bot challenge on this network (or a daily quota). "
                f"Remedies: retry from another network, or download {url} in a browser and "
                f"register the file with `tfx data snapshot --symbol {instrument.symbol!r} "
                f"--file <downloaded.csv>`."
            )
        return content

    def write_snapshot(self, instrument: Instrument, raw: bytes) -> Path:
        """Write raw bytes verbatim + the sha256 sidecar. Also used to build test fixtures."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._csv_path(instrument)
        csv_path.write_bytes(raw)
        sidecar = {
            "symbol": instrument.symbol,
            "stooq_symbol": stooq_symbol(instrument),
            "url": self._url(instrument),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        self._sidecar_path(instrument).write_text(
            json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return csv_path

    def _read_snapshot(self, instrument: Instrument) -> bytes:
        csv_path = self._csv_path(instrument)
        raw = csv_path.read_bytes()
        sidecar_path = self._sidecar_path(instrument)
        if not sidecar_path.exists():
            raise SnapshotIntegrityError(
                f"Snapshot {csv_path} has no sidecar {sidecar_path.name}; refusing to trust it."
            )
        recorded = json.loads(sidecar_path.read_text(encoding="utf-8"))["sha256"]
        actual = hashlib.sha256(raw).hexdigest()
        if actual != recorded:
            raise SnapshotIntegrityError(
                f"Snapshot {csv_path} sha256 {actual} != recorded {recorded}; the snapshot was "
                f"modified outside write_snapshot. Delete and re-pull, or restore the original."
            )
        return raw

    def _ensure_snapshot(self, instrument: Instrument) -> bytes:
        exists = self._csv_path(instrument).exists()
        if self.refresh or not exists:
            if self.offline:
                raise FileNotFoundError(
                    f"No stooq snapshot for {instrument.symbol!r} at "
                    f"{self._csv_path(instrument)} and offline=True. Pull once online "
                    f"(e.g. `tfx data pull --source stooq`) to create it."
                )
            self.write_snapshot(instrument, self._download(instrument))
        return self._read_snapshot(instrument)

    # --- DataSource interface -----------------------------------------------------------------
    def fetch_ohlc(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        stooq_symbol(instrument)  # unmapped symbol raises BEFORE any snapshot/network lookup
        raw = self._ensure_snapshot(instrument)
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise DataQualityError(
                f"stooq snapshot for {instrument.symbol!r} is not parseable CSV: {exc}"
            ) from exc
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if "date" not in frame.columns or "close" not in frame.columns:
            head = raw[:120].decode("utf-8", errors="replace")
            raise DataQualityError(
                f"stooq returned no usable data for {instrument.symbol!r} "
                f"(columns {list(frame.columns)}; starts {head!r}). The symbol map may be wrong."
            )
        index = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name=TIMESTAMP_INDEX_NAME)
        frame = frame.drop(columns=["date"]).set_axis(index, axis=0)
        # OHLC only (drops volume etc.); stooq provides no bid/ask -- cleaning adds NaN quotes.
        frame = frame[[c for c in OHLC_COLUMNS if c in frame.columns]]
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        return frame.loc[mask]
