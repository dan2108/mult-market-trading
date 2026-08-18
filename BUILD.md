# BUILD.md — Diversified Trend FX Engine

Living build doc and source of truth — the full spec, built **incrementally, one module at a time** in build order. `CLAUDE.md` (repo root) is the enforcement layer; this is the map.

---

## North star

A boring, proven signal (trend / time-series momentum) run across **many uncorrelated markets**, with the real engineering in **risk and position sizing** — not the entry. Edge is *discovered and validated*, never chosen because it's popular.

**The gate:** nothing goes live until `validate` shows positive expectancy **out-of-sample**, **after realistic costs**, after a **~50% haircut**. Build quality ≠ edge. The gate decides.

---

## Non-negotiable principles (full text in CLAUDE.md)

1. **Backtest–live parity** — `signal`/`sizing`/`risk` are one shared deterministic core, imported by both backtest and live. What you validate is what trades.
2. **No LLM in the live decision/execution path. Ever.** LLMs are research, build, and ops-observation only.
3. **Deterministic core** — pure functions, no hidden state, no network calls inside it. No ML trade calls in v1.
4. **Costs modeled pessimistically** — spread + swap + slippage, applied to every fill. Never backtest on mid-price.
5. **Tests are the definition of done** — never weaken, skip, or delete a test to pass it. If it can't pass, stop and explain.
6. **No look-ahead, anywhere** — no decision at time t may use data unavailable at t.
7. **Staged rollout, guardrails always on** — backtest → paper → tiny live → scale.

---

## How this gets built

Each module: planned with `/ultraplan` (review the plan before it executes) → built (`/agents` for parallelisable peripheral work) → `/code-review ultra` on the **core and backtest** diffs, pointed at **parity, look-ahead, cost realism, and whether risk limits fire**. One module at a time, in build order. A module is **done only when its acceptance tests are green** — and green ≠ strict, so the core/backtest/validate test *implementations* get eyeballed, not just run.

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

**Shared core (identical in backtest + live):** `signal`, `sizing`, `risk`
**Research/backtest:** `data-pull`, `costs`, `backtest`, `validate`, `report`
**Live (Phase 2, gated on validate):** `broker`, `live`, `monitor`

---

## Repo layout

```
trend-fx/
  CLAUDE.md
  core/        signal.py  sizing.py  risk.py     # deterministic, shared — no LLM
  data/        sources/  clean  align  adjust  cache  loader  quality  pull
  costs/       model.py
  backtest/    engine.py
  validate/    walkforward.py
  report/      metrics.py
  tests/
  build/BUILD.md
  .claude/skills/
```

> Layout is schematic — each module is implemented under `src/tfx/<name>/` (e.g. `src/tfx/costs/`).

---

## Build order

`data-pull` ✓ → `costs` ✓ → `signal` ✓ → `sizing` ✓ → `risk` → `backtest` → `validate` (the gate) → `report` → *(Phase 2: `broker` → `live` → `monitor`)*

---

# Module specs

## Step 1 — `data-pull`  ·  DONE (fixture layer)

Delivered: fetch → clean → timezone-align → cache for the basket, behind `tfx.data.loader.load(...)` — the point-in-time aligned panel both backtest and live consume (parity). Raw-OHLC cache, corporate-actions-on-read, asset-class calendars with flagged `ALIGN_PAD` (NaN, never carried), hashed schema-versioned manifest. All 10 acceptance tests + unit tests green; CI on 3.11/3.12.

**Open before building on it:** eyeball integrity tests #4 (no silent fill) and #9 (no look-ahead in storage) — they're load-bearing. Real-data hardening parked behind the live adapter (stub ready).

---

## Step 2 — `costs`  ·  DONE

Delivered: deterministic all-in cost model behind `CostModel.apply()`/`.breakdown()` — spread
(pessimism-floored) + commission + slippage (always adverse) + sign-correct swap accrual (long vs
short), resolved per-instrument then per-asset-class, never a silent zero-cost fallthrough. Costs
are a non-negative return-fraction-of-equity that scales linearly with size — currency-agnostic.
All 9 acceptance tests + unit tests green (35 total); ruff clean.

**Open before building on it:** every param is flagged **provisional** — placeholder,
pessimistic retail-broker magnitudes, not yet reconciled against a real broker (tracked for
Phase 2 `broker`). Carry credits are clamped to zero in the all-in cost, but the signed value
survives in `CostBreakdown.swap_raw` for that future reconciliation.

---

## Step 3 — `signal`  (core)  ·  DONE

Delivered: deterministic time-series momentum behind `signal.compute(panel, params)` — per
instrument per bar, direction = sign of the trailing `lookback`-bar close return, in {-1, 0, +1};
NaN through warm-up, on padded bars, and wherever no real close exists (`tradable_mask`). Timing
is one shared rule: `shift_for_execution(directions)` applies the decision at t to the next
tradeable bar — imported by both backtest and live so the convention can never drift. Pure
function of the point-in-time panel (no I/O, wall-clock, RNG, hidden state). All 8 acceptance
tests + unit tests green, anchored by a hand-computed fixture slice (`signal_handcalc.md`).

**Open before building on it:** `lookback` (default 20) is a PROVISIONAL placeholder — the real
parameter range is evidenced at the `validate` stage, never hand-chosen. A multi-lookback
ensemble upgrade (mean of signs across ~1–12-month lookbacks, graded [-1, +1]) is planned before
`backtest` lands; ensemble-vs-single is decided by `validate`, not by taste.

---

## Step 4 — `sizing`  (core)  ·  DONE

Delivered: two-stage vol-targeted, correlation-aware sizing behind `sizing.size(panel, directions,
params)`. Stage 1: inverse-vol raw weight per instrument (`direction / EWMA vol`, halflife 20 bars,
log returns, annualized ×√252). Stage 2: one portfolio scalar from the full EWMA covariance
quadratic form rescales the vector to `target_vol` (0.10 annualized) — off-diagonal covariance is
where correlation is priced in (a correlated pair can't jointly carry more risk than the
correlation implies). Hard gross-leverage cap (3.0) applied last. Covariance updates are
all-or-nothing per bar (never partial/asymmetric) and only run forward; the math runs over the
usable subset only (`np.ix_`) so one NaN/holiday/zero-vol instrument can't poison the rest. Flat
direction → exactly 0.0; unknown/untradable/insufficient history → NaN. All 8 acceptance tests +
units green, cross-checked against an independent plain-numpy oracle (`sizing_handcalc.md`).

**Open before building on it:** `ewma_halflife` / `target_vol` / `leverage_cap` are PROVISIONAL
placeholders — halflife is swept at `validate`; target vol and the leverage cap are policy, held
fixed there. Signal warm-up (`lookback` bars) and sizing burn-in (`ceil(halflife)` NaN-free bars)
gate independently — the backtest treats NaN as "no decision, carry position."

---

## Step 5 — `risk`  (core)

**Goal:** the deterministic veto between sized positions and execution — per-trade cap, portfolio-heat cap, correlation/exposure cap, drawdown auto-halt, kill switch. Can only ever **reduce** risk. Identical in backtest and live.

**Scope —** in: the limits above + account state (equity, drawdown, open exposure). out: signal, sizing, execution, data.

**Interface:** `risk.check(proposed_positions, account_state, params) -> approved_positions + reasons`.

**Acceptance tests:**
1. Per-trade cap: no approved position risks more than the cap; oversized proposals scaled down.
2. Portfolio-heat cap: total risk ≤ cap; positions scaled pro-rata if exceeded.
3. Correlation/exposure cap: clustered exposure ≤ cap.
4. Drawdown halt: breaching the DD limit blocks new risk / flattens.
5. Kill switch: when set, all → flat, no new trades.
6. **Monotonic safety:** `risk.check` never increases a size or flips a sign — only reduces/zeros.
7. **Limits actually fire:** a stress fixture breaches each limit; assert the veto.
8. Pure & deterministic; reasons returned for every reduction (auditability).

**Done:** tests green; **every limit has a test proving it fires**; documented.

---

## Step 6 — `backtest`

**Goal:** the simulator. Walk the panel bar by bar, call the **exact shared core** (signal→sizing→risk) with point-in-time data, apply costs on every fill and swap on every hold, track positions/equity/trades. Deterministic. Output: trade log + equity curve.

**Scope —** in: the event loop, execution at the correct next-bar price, cost/swap application, accounting, trade log, equity curve. out: the core (imported), walk-forward (validate), metrics (report), live.

**Interface:** `backtest.run(panel, core, costs, params) -> {trade_log, equity_curve, positions}`.

**Acceptance tests:**
1. **No look-ahead (cardinal):** decisions at t use data ≤ t and execute at t+1 — injecting future bars must not change any historical trade or fill.
2. **Execution timing:** fills at the next-bar price, never the same-bar close the signal was computed on.
3. **Costs applied:** every entry/exit pays spread/commission; every overnight hold pays swap. A do-nothing run has flat equity; a churning run bleeds costs.
4. **Accounting integrity:** equity reconciles bar-by-bar (start + realized + unrealized − costs); no money created or destroyed.
5. **Parity:** the core invoked is the importable shared module live uses — not a reimplementation (assert it).
6. Deterministic — identical trade log + equity curve for identical inputs.
7. Padded/NaN bars: no trading; positions carry correctly across calendar gaps.
8. Risk veto respected — logged positions never exceed risk-approved sizes.
9. Fixture mini-scenario (few bars, known signal) yields the exact expected trades/equity.

**Done:** tests green; the no-look-ahead test is explicit and strict; parity asserted; accounting reconciles.

---

## Step 7 — `validate`  ·  THE GATE

**Goal:** the honest verdict on whether an edge exists. Runs `backtest` under walk-forward / out-of-sample protocols, applies the haircut, checks robustness and significance, returns **pass/fail against pre-set thresholds**. This is what stands between a pretty backtest and real money.

**Scope —** in: walk-forward (train→test, rolling), out-of-sample holdout, parameter-robustness (does it survive nearby params or is it a knife-edge?), the ~50% haircut, trade-count/significance gate, and — if multiple signals — the standalone-vs-combined + correlation experiments. out: the backtest engine (orchestrates it), visuals (report), live.

**Interface:** `validate.run(panel, core, costs, protocol) -> {oos_metrics, walkforward, robustness, verdict}`.

**Acceptance tests:**
1. **Strict fold separation (the whole point):** test-window metrics use no data or parameter fit from the test window — no leakage between folds.
2. **Walk-forward correctness:** windows roll forward only; no fold sees its own future.
3. **Catches overfitting:** on a fixture tuned to in-sample noise, the OOS verdict is fail/poor.
4. **Catches no-edge:** on a random-walk fixture, the verdict is **fail**. (Two-sided sanity with #5.)
5. **Passes real edge:** on a genuinely-trending synthetic fixture, the verdict is pass.
6. **Robustness:** a knife-edge parameter optimum flags as fragile.
7. **Haircut applied:** reported expectation ≤ in-sample × haircut factor.
8. **Significance gate:** below a minimum trade count, verdict = "insufficient evidence," not pass.
9. **Pre-set thresholds:** pass/fail uses thresholds fixed in config *before* the run — no moving goalposts. Deterministic.

**Done:** tests green; the no-leakage, random-data-fails, and overfit-fails tests all exist and are strict; thresholds documented and pre-set.

---

## Step 8 — `report`

**Goal:** compute and present the full metric set from a backtest/validate run, and have Claude (research layer — allowed) interpret and flag red flags. Read-only; never feeds back into decisions.

**Scope —** in: metrics (CAGR, Sharpe, Sortino, max DD, time-in-DD, hit rate, avg win/loss, turnover, per-instrument contribution, correlation matrix, exposure-over-time), equity/drawdown plots, an LLM interpretation flagging overfitting/fragility. out: the simulation (consumes output), live, any trade decision.

**Interface:** `report.generate(results) -> metrics + plots + written summary` (LLM prose clearly observational).

**Acceptance tests:**
1. Each metric matches a hand-computed value on a fixture equity curve (Sharpe, max DD, CAGR, hit rate).
2. Drawdown correct — peak-to-trough, sign, duration.
3. Per-instrument contributions sum to the total.
4. **Numeric metrics deterministic** (LLM prose may vary; numbers must not).
5. **Read-only:** report has no path that mutates positions/signals — observational only (honours "no LLM in the live path").
6. Handles edge cases — zero trades, all-losing, single instrument.

**Done:** tests green; metrics validated against a hand-computed fixture; LLM output labelled observational.

---

# Phase 2 — live (do NOT start until `validate` returns a clear pass)

Spec'd lighter — broker choice and final signal/params are still parked; detail firms up at the gate.

## `broker`
**Goal:** execution abstraction — connect, place/cancel orders, fills, position/balance reconciliation, live prices. Idempotent. (Live-stub already in place from v1.)
**Scope —** in: orders, fills, reconciliation, live price adapter, idempotency keys, retry/error handling. out: the core.
**Interface:** a `Broker` protocol (`place_order`, `cancel`, `get_positions`, `get_fills`, `get_price`).
**Key tests:** idempotency (retry never double-orders); reconciliation (broker truth vs internal state, mismatches flagged); partial-fill and rejection handling; demo/paper mode behind the same interface.
**Done:** tested against the broker sandbox; reconciliation + idempotency proven.

## `live`
**Goal:** the deployed deterministic runtime — on schedule: latest data → exact shared core → broker. No LLM. Runs on a VPS.
**Scope —** in: the scheduled loop, data→core→broker wiring, state persistence, restart safety. out: the core (imported), monitoring narrative.
**Key tests:** parity (same core as backtest — assert); idempotent restart (crash/restart never double-trades); respects risk veto + kill switch; deterministic given the same inputs; paper mode matches backtest on the same data before any real money.
**Done:** runs in paper mode matching backtest behaviour; parity asserted; restart-safe; no LLM in path.

## `monitor`
**Goal:** ops/observability beside `live` — heartbeat, alerts, kill-switch control, health. Claude may write plain-English ops summaries (observational only). Never in the decision path.
**Scope —** in: heartbeat/health, alerts (drawdown breach, connectivity loss, no-trade-when-expected, reconciliation mismatch), kill-switch trigger, daily summary. out: any trade decision.
**Key tests:** alerts fire on simulated breaches; kill switch actually halts `live`; monitor **cannot** place/alter trades (observational-only assertion); deterministic alert logic.
**Done:** alerts proven on simulated conditions; kill-switch wiring proven; observational-only asserted.

---

## Parked decisions
- [ ] Live data provider (the one input `data-pull` needs pointed at it for real-data hardening)
- [ ] Broker for Phase 2
- [ ] Trend signal family + parameter ranges — decided at the `signal`/`validate` stage, validate-driven
- [ ] Bar frequency — daily assumed; revisit if needed
