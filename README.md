# mult-market-trading

A diversified, trend-following **multi-market (FX / stock / crypto)** trading platform.

A boring, proven signal (time-series momentum) run across many uncorrelated markets, with the
real engineering poured into **risk and position sizing** — not the entry. Edge is *discovered
and validated*, never chosen because it's popular. Nothing goes live until validation shows
positive expectancy **out-of-sample, after realistic costs, after a ~50% haircut**.

> Status: **v1 — building the core** (`data-pull` ✓ · `costs` ✓ · `signal` ✓ · now: `sizing`).
> See [BUILD.md](BUILD.md) for the full plan and [CLAUDE.md](CLAUDE.md) for the standing
> engineering rules.

## Why this design

- **Backtest–live parity.** `signal`, `sizing`, `risk` are one shared deterministic core; the
  backtester and the live engine import the *same code*. What you validate is what trades.
- **No LLM in the decision/execution path.** The trade rule is transparent code.
- **Pessimistic costs** (spread + swap + slippage) on every fill.
- **Point-in-time data**, zero look-ahead, byte-reproducible cache.

## Architecture

```
            ┌─────────────── shared deterministic core ───────────────┐
 data ───►  │  signal  ──►  sizing (vol-target)  ──►  risk (veto)      │  ──► positions
            └──────────────────────────────────────────────────────────┘
   research/backtest:  data-pull → costs → backtest → validate → report
   live (Phase 2):     data feed → core → broker → monitor
```

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"

tfx --help                       # see all commands
tfx data pull --source fixture   # fetch → clean → align → cache the basket
tfx data report                  # data-quality report (coverage, gaps, imputations)

pytest -q                        # the acceptance-test gate
ruff check .                     # lint
```

## What `data-pull` does (v1)

Fetches, cleans, timezone-normalizes, cross-instrument aligns, and caches historical **daily
OHLC** for the instrument basket, behind a clean interface that `backtest` consumes. Built behind
a **pluggable data source** so the feed can be swapped (v1 ships a deterministic CSV-fixture
source + a stubbed live adapter).

Key guarantees (enforced by `tests/test_data_pull.py`):
coverage · UTC-sorted unique timestamps · explicit & classified calendar gaps · single aligned
calendar with marked gaps (never a silent forward-fill) · OHLC sanity · no NaNs (imputations
flagged) · **byte-deterministic** cache · cache round-trip == fresh fetch · **no look-ahead in
storage** · a hand-verified fixture slice.

### Price adjustment (splits/dividends)
The per-bar cache stores **immutable raw OHLC** (true point-in-time). Corporate actions are stored
separately; adjusted series are derived **on read**. This keeps the cache point-in-time clean and
byte-deterministic while still giving trend signals a continuous (adjusted) series.

## Instrument basket (v1)

17 markets across four asset classes — breadth across lowly-correlated markets is the dominant
driver of trend-following performance:

- **FX (7):** `EUR/USD`, `USD/JPY`, `GBP/USD`, `AUD/JPY`, `USD/CAD`, `GBP/CHF`, `NZD/USD`
- **Equity indices (4):** `S&P 500`, `DAX`, `Nikkei 225`, `FTSE 100`
- **Commodities (4):** `Gold`, `Silver`, `Oil` (WTI), `Nat Gas`
- **Crypto (2):** `BTC/USD`, `ETH/USD` (24/7 calendar)

Bonds/rates are deferred: the current provider carries US treasuries only as yields, not prices.
Final list confirmed against the live broker/data source later.

## Roadmap

`data-pull` ✓ → `costs` ✓ → core (`signal` ✓ / `sizing` / `risk`) → `backtest` → `validate` →
`report` → Phase 2 (`broker` → `live` → `monitor`).
