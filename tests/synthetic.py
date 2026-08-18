"""Seeded synthetic panels for the validate gate's verdict tests.

TEST-ONLY. The committed data/sample fixtures contain deliberate sin-wave trend terms (good for
exercising the pipeline, useless as a null hypothesis), so the gate's own two-sided sanity tests
need purpose-built processes: a driftless random walk that MUST fail, and a persistent trend
that MUST pass. Seeded numpy Generators keep every panel bit-reproducible; production code
(tfx/core, tfx/data) remains RNG-free -- randomness lives only in this test fixture module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfx.data.align import FIELD_LEVEL, INSTRUMENT_LEVEL
from tfx.data.schema import TIMESTAMP_INDEX_NAME


def panel_from_closes(closes: dict[str, np.ndarray]) -> pd.DataFrame:
    """(close, quality) panel shaped like align()'s output, business-day calendar."""
    n = len(next(iter(closes.values())))
    index = pd.date_range(
        "2015-01-05", periods=n, freq="B", tz="UTC", name=TIMESTAMP_INDEX_NAME
    )
    frames = {
        symbol: pd.DataFrame(
            {"close": np.asarray(values, dtype=float), "quality": 0}, index=index
        )
        for symbol, values in closes.items()
    }
    return pd.concat(
        frames.values(), axis=1, keys=list(frames.keys()), names=[INSTRUMENT_LEVEL, FIELD_LEVEL]
    )


def _walk(rng: np.random.Generator, n_bars: int, drift: float, vol: float) -> np.ndarray:
    log_returns = rng.normal(loc=drift, scale=vol, size=n_bars - 1)
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))


def gbm_panel(
    symbols: list[str], n_bars: int, seed: int, vol: float = 0.01
) -> pd.DataFrame:
    """DRIFTLESS geometric random walks: no edge exists by construction. The gate must FAIL."""
    rng = np.random.default_rng(seed)
    return panel_from_closes(
        {symbol: _walk(rng, n_bars, drift=0.0, vol=vol) for symbol in symbols}
    )


def trending_panel(
    symbols: list[str], n_bars: int, seed: int, drift: float = 0.003, vol: float = 0.006
) -> pd.DataFrame:
    """Persistent-drift processes (alternating sign per instrument): a real, strong trend a
    momentum rule must catch. The gate must PASS."""
    rng = np.random.default_rng(seed)
    closes = {}
    for k, symbol in enumerate(symbols):
        sign = 1.0 if k % 2 == 0 else -1.0
        closes[symbol] = _walk(rng, n_bars, drift=sign * drift, vol=vol)
    return panel_from_closes(closes)
