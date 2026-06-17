"""Canonical data schema, quality flags, and the point-in-time column contract.

The per-bar cache may contain ONLY fields knowable at that bar's own timestamp. This is the
mechanical guard against look-ahead leaking into storage (acceptance test #9). Anything derived
from the whole series or from the future (normalizations, forward returns, split/dividend
adjustment factors) is computed downstream or stored separately — never baked into a cached bar.
"""

from __future__ import annotations

from enum import IntFlag

# Bump when the on-disk per-bar layout changes; the cache refuses to load a mismatched version.
SCHEMA_VERSION = 1

TIMESTAMP_INDEX_NAME = "timestamp"

OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
QUOTE_COLUMNS: tuple[str, ...] = ("bid", "ask")
QUALITY_COLUMN = "quality"

# Exactly the columns allowed in a per-instrument cached bar. Enforced by the no-look-ahead test.
# NOTE: split/dividend adjustment factors are deliberately NOT here — they depend on future
# corporate actions and live in a separate corporate-actions table, applied on read.
POINT_IN_TIME_COLUMNS: tuple[str, ...] = (*OHLC_COLUMNS, *QUOTE_COLUMNS, QUALITY_COLUMN)

# Corporate actions (splits/dividends) are stored OUTSIDE the per-bar cache because the
# back-adjustment factor for a past bar depends on the FUTURE. They are applied on read to
# derive an adjusted series. ratio convention: split N:1 -> ratio=N; cash dividend -> ratio is
# the dividend amount in the instrument's quote currency.
CORP_ACTION_COLUMNS: tuple[str, ...] = ("ex_date", "type", "ratio")
CORP_ACTION_TYPES: frozenset[str] = frozenset({"split", "dividend"})


class QualityFlag(IntFlag):
    """Bitmask describing how (if at all) a bar was modified during cleaning/alignment.

    ``OK`` (0) means a verified, untouched real bar. Flags are additive: a bar may be both
    ``DUP_RESOLVED`` and ``IMPUTED``.
    """

    OK = 0
    IMPUTED = 1          # a missing OHLC/quote field was repaired from same-bar data
    OHLC_REPAIRED = 2    # an OHLC inequality was corrected (high/low widened to bound open/close)
    DUP_RESOLVED = 4     # a duplicate timestamp was collapsed to a single bar
    ALIGN_PAD = 8        # row inserted by cross-instrument alignment; OHLC is NaN (panel only)


def describe_quality(value: int) -> str:
    """Human-readable, stable description of a quality bitmask (e.g. 'IMPUTED|DUP_RESOLVED')."""
    flag = QualityFlag(value)
    if flag is QualityFlag.OK:
        return "OK"
    parts = [f.name for f in QualityFlag if f is not QualityFlag.OK and f in flag]
    return "|".join(parts)
