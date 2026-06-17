"""Deterministic on-disk cache: one Parquet per instrument + a single hashed manifest.

Design choices that make the cache trustworthy:
  * **Point-in-time only** — each ``<slug>.parquet`` holds RAW OHLC/quote/quality and nothing
    derived from the future. Corporate actions live in the manifest (applied on read), so the
    per-bar files never carry look-ahead (acceptance test #9).
  * **Byte-deterministic** — fixed column order, fixed compression, no wall-clock anywhere, and
    a manifest serialized with sorted keys. Re-pulling the same range reproduces identical bytes
    (test #7).
  * **Tamper/stale evident** — the manifest stores a stable content hash per instrument and a
    ``schema_version``. Reads verify both and refuse mismatched/stale caches rather than trust
    them silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..instruments import Instrument
from .clean import CleanResult
from .errors import CacheError, SchemaMismatchError
from .schema import (
    CORP_ACTION_COLUMNS,
    POINT_IN_TIME_COLUMNS,
    QUALITY_COLUMN,
    SCHEMA_VERSION,
    TIMESTAMP_INDEX_NAME,
)

MANIFEST_NAME = "manifest.json"
_PARQUET_COMPRESSION = "snappy"  # deterministic codec


# --------------------------------------------------------------------------- paths
def instrument_path(cache_dir: Path | str, instrument: Instrument) -> Path:
    return Path(cache_dir) / f"{instrument.slug}.parquet"


def manifest_path(cache_dir: Path | str) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


# ------------------------------------------------------------------- hashing / shape
def content_hash(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over the canonical frame (values + index). Independent of file encoding and
    reproducible across machines/runs (pandas object hashing is deterministic)."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[list(POINT_IN_TIME_COLUMNS)].copy()
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = TIMESTAMP_INDEX_NAME
    return out


def _disk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _canonical(frame).reset_index()
    return out[[TIMESTAMP_INDEX_NAME, *POINT_IN_TIME_COLUMNS]]


# --------------------------------------------------------------- corporate actions <-> json
def _ca_to_records(corporate_actions: pd.DataFrame | None) -> list[dict]:
    if corporate_actions is None or corporate_actions.empty:
        return []
    records: list[dict] = []
    for action in corporate_actions.itertuples(index=False):
        ts = pd.Timestamp(action.ex_date)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        records.append(
            {
                "ex_date": ts.date().isoformat(),
                "type": str(action.type).lower(),
                "ratio": float(action.ratio),
            }
        )
    records.sort(key=lambda r: (r["ex_date"], r["type"]))
    return records


def _ca_from_records(records: list[dict] | None) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            {
                "ex_date": pd.Series([], dtype="datetime64[ns, UTC]"),
                "type": pd.Series([], dtype="object"),
                "ratio": pd.Series([], dtype="float64"),
            }
        )
    df = pd.DataFrame(records)
    df["ex_date"] = pd.to_datetime(df["ex_date"], utc=True)
    df["ratio"] = df["ratio"].astype("float64")
    return df[list(CORP_ACTION_COLUMNS)]


# --------------------------------------------------------------------------- write
def write_instrument(
    cache_dir: Path | str,
    instrument: Instrument,
    clean_result: CleanResult,
    corporate_actions: pd.DataFrame | None = None,
) -> dict:
    """Write one instrument's per-bar Parquet and return its manifest entry."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = _canonical(clean_result.frame)
    _disk_frame(frame).to_parquet(
        instrument_path(cache_dir, instrument),
        engine="pyarrow",
        compression=_PARQUET_COMPRESSION,
        index=False,
    )
    has_rows = len(frame) > 0
    return {
        "symbol": instrument.symbol,
        "slug": instrument.slug,
        "asset_class": instrument.asset_class.value,
        "rows": int(len(frame)),
        "first": frame.index.min().isoformat() if has_rows else None,
        "last": frame.index.max().isoformat() if has_rows else None,
        "content_hash": content_hash(frame),
        "n_input": int(clean_result.n_input),
        "n_dups_removed": int(clean_result.n_dups_removed),
        "n_dropped_unusable": int(clean_result.n_dropped_unusable),
        "corporate_actions": _ca_to_records(corporate_actions),
    }


def write_manifest(
    cache_dir: Path | str,
    *,
    source: str,
    symbols: list[str],
    start: object,
    end: object,
    entries: dict[str, dict],
) -> dict:
    """Write the deterministic manifest (sorted keys, no wall-clock)."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "symbols": list(symbols),
        "start": str(start),
        "end": str(end),
        "instruments": entries,
    }
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    manifest_path(cache_dir).write_text(text, encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------- read
def read_manifest(cache_dir: Path | str) -> dict:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise CacheError(f"No cache manifest at {path}. Run `tfx data pull` first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"Cache schema v{version} != expected v{SCHEMA_VERSION}. Re-pull: `tfx data pull`."
        )
    return manifest


def read_instrument(
    cache_dir: Path | str,
    instrument: Instrument,
    *,
    manifest: dict | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """Read one instrument's cached series back into canonical form, verifying integrity."""
    path = instrument_path(cache_dir, instrument)
    if not path.exists():
        raise CacheError(f"{instrument.symbol}: not cached at {path}. Run `tfx data pull`.")
    disk = pd.read_parquet(path, engine="pyarrow")
    if TIMESTAMP_INDEX_NAME not in disk.columns:
        raise CacheError(f"{path}: missing '{TIMESTAMP_INDEX_NAME}' column.")
    frame = disk.set_index(TIMESTAMP_INDEX_NAME)[list(POINT_IN_TIME_COLUMNS)]
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = TIMESTAMP_INDEX_NAME
    frame[QUALITY_COLUMN] = frame[QUALITY_COLUMN].astype("int64")

    if verify:
        manifest = manifest or read_manifest(cache_dir)
        entry = manifest.get("instruments", {}).get(instrument.slug)
        if entry is None:
            raise CacheError(f"{instrument.symbol}: no manifest entry; cache is inconsistent.")
        if content_hash(frame) != entry.get("content_hash"):
            raise SchemaMismatchError(
                f"{instrument.symbol}: cache content-hash mismatch (stale or corrupt). "
                f"Re-pull: `tfx data pull`."
            )
    return frame


def read_corporate_actions(
    cache_dir: Path | str,
    instrument: Instrument,
    *,
    manifest: dict | None = None,
) -> pd.DataFrame:
    manifest = manifest or read_manifest(cache_dir)
    entry = manifest.get("instruments", {}).get(instrument.slug, {})
    return _ca_from_records(entry.get("corporate_actions", []))
