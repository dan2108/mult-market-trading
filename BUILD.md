# BUILD.md — Diversified Trend FX Engine

Living build doc and source of truth. Extend it as we add pieces. Built in interactive (live) sessions.

---

## North star

A boring, proven signal (trend / time-series momentum), run across **many uncorrelated markets**, with all the real engineering poured into **risk and position sizing** — not the entry. Edge is *discovered and validated*, never chosen because it's popular.

**The gate:** nothing goes live until `validate` shows positive expectancy **out-of-sample**, **after realistic costs**, and after a **~50% haircut** on backtested performance. Build quality ≠ edge. The gate decides.

---

## Non-negotiable principles (the rules that protect the money)

1. **Backtest–live parity.** `signal`, `sizing`, `risk` are written **once** as a shared deterministic core. Both the backtester and the live engine import the *same code*. What you validate is literally what trades.
2. **No LLM in the live decision/execution path.** LLMs are for research, building, and ops *observation* only. Never deciding or executing trades.
3. **Deterministic core; LLM agents orbit it.** No ML making trade calls in v1 — the decision rule is transparent code.
4. **Costs modeled pessimistically** — spread + swap + slippage — applied to every fill. Never backtest on mid-price.
5. **Staged rollout:** backtest → paper (demo) → tiny live → scale. Guardrails always on (per-trade risk cap, portfolio-heat cap, correlation cap, drawdown auto-halt, kill switch).
6. **Build interactively and review the core hard.** When building `signal`/`sizing`/`risk` and `backtest`, go slow and run `/ultrareview` (or `/code-review ultra`) on the diff — pointed at parity, look-ahead, cost realism, and whether the risk limits actually fire. These are where money is silently lost.

---

## Architecture

```
            ┌─────────────── shared deterministic core ───────────────┐
 data ───►  │  signal  ──►  sizing (vol-target)  ──►  risk (veto)      │  ──► positions
            └──────────────────────────────────────────────────────────┘
                 ▲ identical code path in BOTH ▼
   research/backtest:  data-pull → costs → backtest → validate → report
   live (Phase 2):     data feed → core → broker → monitor
```

**Shared core (deterministic libs, identical in backtest + live):** `signal`, `sizing`, `risk`
**Research / backtest tooling:** `data-pull`, `costs`, `backtest`, `validate`, `report`
**Live (Phase 2, later):** `broker`, `live` runtime, `monitor`

Skills/commands are thin wrappers around these modules. LLM lives in the research + ops layers, never in the core.

---

## Proposed repo layout

```
trend-fx/
  CLAUDE.md              # standing rules — loaded into every session
  core/                  # deterministic, shared by backtest + live — no LLM, ever
    signal.py
    sizing.py
    risk.py
  data/
    pull.py              # ← step 1
    cache/               # cached datasets (gitignored)
  costs/      model.py
  backtest/   engine.py
  validate/   walkforward.py
  report/     metrics.py
  tests/
    test_data_pull.py    # ← step 1 acceptance tests
  build/BUILD.md
  .claude/skills/        # skills wrap the modules above
```

---

## Build order

`data-pull` → `costs` → core (`signal` / `sizing` / `risk`) → `backtest` → `validate` → `report` → *(Phase 2: `broker` → `live` → `monitor`)*

---

## STEP 1 — `data-pull`

**Start here:** mechanical, bounded, and correctness is *checkable by tests*. The strategy core is **not** in scope at this step.

### Goal
Fetch, clean, time-align, and cache historical **daily OHLC** for the instrument basket into a local cache, behind a clean documented interface that `backtest` will consume. Zero look-ahead in how data is stored or served.

> Frequency = **daily bars** by default (suits a days-to-weeks holding period; easiest to source cleanly). Adjustable later. Bid/ask captured *where the source provides it*, but for daily bars spread realism is the job of `costs`, not `data-pull`.

### Scope
- **IN:** acquisition, cleaning, timezone normalization, cross-instrument alignment, caching, a data-quality report, and the integrity tests below.
- **OUT (not at this step):** `signal`, `sizing`, `risk`, backtest logic, any live/broker code.

### Interface (what `backtest` expects)
- A loader (function + CLI) that returns aligned data for a list of instruments over a date range.
- Explicit schema: UTC timestamp index; per instrument `open/high/low/close` (+ optional `bid/ask`); a quality flag column for any cleaned/imputed bar.

### Acceptance tests (all green or it's not done)
1. **Coverage** — every basket instrument has data across the requested range; gaps are *reported*, not silently dropped.
2. **Timestamps** — UTC-normalized, sorted ascending, no duplicates.
3. **Calendar gaps** — weekend/holiday gaps are expected and explicitly marked; no unexpected mid-week gaps beyond a documented tolerance.
4. **Alignment** — all instruments share one common date index; a date missing for one instrument is handled by an explicit, documented rule (skip/mark) — **never a silent forward-fill that could leak**.
5. **OHLC sanity** — every bar: `high ≥ open`, `high ≥ close`, `low ≤ open`, `low ≤ close`, `high ≥ low`, all values `> 0`.
6. **No NaNs** — none in OHLC after cleaning; any imputed value is flagged in the quality column.
7. **Determinism** — running twice over the same range yields byte-identical cached output.
8. **Cache round-trip** — load-from-cache equals fresh-fetch for the same range.
9. **No look-ahead in storage** — each row holds only fields knowable at that timestamp; no future-derived or whole-series-normalized columns baked into the per-bar cache.
10. **Fixture check** — a small hand-verified slice (one instrument, one week) matches expected OHLC exactly.

### Definition of done
All 10 tests green · interface documented · a short data-quality report emitted (per-instrument coverage, date ranges, gap summary).

### Build it
Write the 10 tests first, implement until all green, then run `/ultrareview` on the diff. Green = done; not green = not done, regardless of what any agent claims.

### Data source
The one input you supply: your broker API or a data provider you have access to. Build it behind a **pluggable interface** so it can be swapped later. If the source isn't ready, build the interface + integrity logic + tests against a sample CSV fixture and stub the live adapter — the mechanical part is done and tested, and you wire the real feed in next.

---

## The instrument basket  *(confirm vs broker)*
Starter candidates chosen for **low cross-correlation** (not six USD pairs): `EUR/USD`, `AUD/JPY`, `GBP/CHF`, `S&P 500`, `DAX`, `Gold`, `Oil`. Final list depends on what your broker/data source actually offers.

---

## Open decisions (parked, visible)
- [ ] Instrument basket — confirm against broker availability
- [ ] Data source/provider — the one thing `data-pull` needs pointed at it
- [ ] Bar frequency — daily assumed; revisit if needed
- [ ] Trend signal + parameter ranges — decided at the core stage, validate-driven
- [ ] Broker for Phase 2 (live)
