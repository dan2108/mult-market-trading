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

`data-pull` ✓ → `costs` ✓ → `signal` ✓ → `sizing` ✓ → `risk` ✓ → `backtest` ✓ → `validate` (the gate) ✓ → `report` ✓ → *(Phase 2: `broker` → `live` → `monitor`)*

> `validate` the *module* is built and green; the **gate run itself** — full basket, real
> provider data, ~2010→present, pre-set thresholds — has NOT happened yet. Phase 2 stays parked
> until that run returns PASS.

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

Delivered: deterministic multi-lookback time-series momentum behind `signal.compute(panel,
params)` — per instrument per bar, direction = equal-weight mean of sign(trailing return) across
the lookback ensemble, graded in [-1, +1] (a single-entry tuple degenerates to the discrete
{-1, 0, +1} rule, pinned verbatim to the original hand calc as a regression anchor). NaN through
warm-up (ALL horizons' anchors required — no regime discontinuity inside validate folds), on
padded bars, and wherever no real close exists (`tradable_mask`). Timing is one shared rule:
`shift_for_execution(directions)` applies the decision at t to the next tradeable bar — imported
by both backtest and live so the convention can never drift. Pure function of the point-in-time
panel (no I/O, wall-clock, RNG, hidden state). All 8 acceptance tests + ensemble unit tests
green, anchored by hand-computed fixture slices (`signal_handcalc.md`) and a compositional
property (ensemble ≡ mean of its single-lookback members).

**Open before building on it:** `lookbacks` (default `(21, 63, 126, 252)`, ~1–12 months) is a
PROVISIONAL placeholder — the real choice, including ensemble vs single lookback, is swept and
evidenced at the `validate` stage, never hand-chosen.

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

## Step 5 — `risk`  (core)  ·  DONE

Delivered: the deterministic veto behind `risk.check(proposed, account_state, params) ->
RiskDecision` (approved weights + a machine-readable reason per reduction + a `halted` flag).
Fixed application order: kill switch → drawdown auto-halt → per-instrument cap → asset-class
gross cap → portfolio-heat (gross) cap — every stage multiplies by a scalar in [0, 1], so
**monotonic safety holds by construction** (|approved| ≤ |proposed|, sign never flips; pinned by
a deterministic grid property test). Pure per-bar function: all state (equity, peak equity, kill
switch) arrives via a frozen `AccountState` the caller maintains — nothing stored, identical in
backtest and live. Caps live in weight space and never touch the panel: sizing already prices
statistical correlation, so risk's exposure clustering is deterministic per `AssetClass` — a
backstop that cannot drift with an estimator. NaN proposals pass through as "no decision" except
under halt/kill, which force real 0.0 everywhere (flat means flat). Unknown symbols raise. All 8
acceptance tests + units green; every limit has a test proving it fires; exact hand-calc fixture
(`risk_handcalc.md`) where all three caps bind at once and final gross lands on the heat cap
exactly.

**Open before building on it:** cap levels are PROVISIONAL **policy** — never swept or fit at
validate (a guardrail tuned to the data is not a guardrail). The drawdown halt latches
emergently (halted → flat → equity freezes → still halted); resuming is a deliberate human
decision in v1.

---

## Step 6 — `backtest`  ·  DONE

Delivered: bar-by-bar simulator behind `backtest.run(panel, params, cost_model) ->
BacktestResult` (equity curve, on-the-books positions, approved targets, trade log, cost ledger,
per-instrument attribution, risk-event audit trail). **Timing:** decision at close[t] → fill at
close[t+1] (never the close the signal saw) → first earned return close[t+1]→close[t+2] — one
bar more lag than the theoretical minimum, deliberately pessimistic. **Parity is structural:**
the engine binds `signal.compute` / `sizing.size` / `risk.check` / `shift_for_execution` as
module attributes (function-object identity asserted in tests) and internally reconciles its
positions against `shift_for_execution(approved)` + the carry rule every run. **Costs:** every
fill pays floored spread + commission + adverse slippage at the fill price; every overnight hold
pays swap for the CALENDAR nights since the previous bar (Fri→Mon = 3 nights; 24/7 calendars
accrue nightly). **Whole-book risk:** legs sizing has no decision for are re-proposed at their
current book weight so the veto sees and caps the entire book — carried legs can never stack
past the heat cap unseen (a reduce order for a closed market lapses and is enforced at the
reopen, documented). Accounting is multiplicative in return space; equity reconciles bar-by-bar
from the published frames alone. All 9 acceptance tests + units green, anchored by a hand-calc
slice (`backtest_handcalc.md`) and a full independent oracle re-implementation of the engine
mechanics in test code. CLI: `tfx backtest run`.

**Open before building on it:** research output only — the verdict comes from `validate`
(Step 7), never from a single backtest table. Fill-at-next-close is the v1 convention; a
next-bar-open variant is a possible later refinement, strictly behind the gate.

---

## Step 7 — `validate`  ·  THE GATE  ·  DONE (module — the gate RUN is still pending)

Delivered: rolling walk-forward behind `validate.run(panel, protocol, cost_model) ->
ValidateResult` with verdict ∈ {PASS, FAIL, FRAGILE, INSUFFICIENT_EVIDENCE}; CLI `tfx validate
run` exits 0 only on PASS. **Geometry:** train 1008 bars (~4y) / test 252 / step 252, rolling.
**Two backtests per (fold, candidate), not one combined run:** selection is a standalone
train-only run scored on its OWN active bars only (`GridPoint.warmup_bars()` trims each
candidate's warm-up prefix before scoring Sharpe — untrimmed, a longer lookback/halflife dilutes
Sharpe toward zero and structurally favors short horizons); OOS scoring is a FRESH run starting
only `warmup_bars() + 5` bars before test_start, never train_start — so a genuinely severe
in-training drawdown can't inflate the peak an OOS-window halt is measured against, and every
candidate (not just the fold's winner) gets this fresh run so the robustness check's
cross-candidate comparison stays apples-to-apples. `ValidateProtocol` validates at construction
that the grid's worst-case warm-up actually fits inside `train_bars` — without it, selection
would silently fall back to grid order with an undefined score. **Pre-registered grid (18):**
`lookbacks` ∈ {four singles, two ensembles — the sweep itself decides ensemble-vs-single} ×
`ewma_halflife` ∈ {10, 20, 60}. `target_vol`, leverage cap and ALL risk limits are held fixed
(policy, never fit). **Honesty devices:** thresholds live in one frozen `ValidateProtocol`
echoed verbatim into the result (no moving goalposts — the CLI's fold-geometry override flags
exist for research only and mark any run that uses them NON-CANONICAL, loudly, in the output);
the ~50% haircut applies to the numbers the thresholds see; significance = t-stat ≥ 2 + minimum
trade count (else INSUFFICIENT_EVIDENCE); Deflated Sharpe (Bailey–López de Prado closed form,
n_trials = grid size, stdlib NormalDist — deterministic, no bootstrap) reported; robustness =
modal winner in ≥ half the folds AND grid-mean OOS Sharpe > 0, else FRAGILE. All 9 acceptance
tests + units green — including the two-sided sanity pair on seeded synthetic processes:
driftless GBM **fails**, persistent trend **passes**, and a lucky-in-sample noise panel (train
Sharpe up to ~1.0) still fails OOS.

**Open before trusting it:** the gate RUN on real provider data (full basket, ~2010→present,
`tfx data report --strict` first) has not happened; costs are still provisional. The verdict —
not the build — unparks Phase 2.

---

## Step 8 — `report`  ·  DONE

Delivered: `report.generate(BacktestResult) -> Report` (CAGR, Sharpe, Sortino, max DD +
duration + time-in-DD, bar-level hit rate / avg win / avg loss, annualized turnover, total
costs, per-instrument contribution — sums to the total by the engine's accounting identity —
attribution correlation matrix, gross-exposure stats) plus deterministic markdown rendering
(`to_markdown`). Stats primitives shared with `validate` (one implementation of Sharpe et al.).
CLI: `tfx report [--out report.md]`. **Strictly observational:** the package imports nothing
from `tfx.core` (structurally asserted in tests), mutates nothing, and contains NO LLM call —
prose interpretation happens outside the module in the research workflow, labelled
observational, never feeding back into a decision. All 6 acceptance tests + units green,
anchored by a hand-computed fixture curve (`report_handcalc.md`).

**Open:** plots deferred (numbers are the artifact in v1); per-trade round-trip stats deferred
(the trade log is deltas, not round trips — revisit with the broker layer).

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
