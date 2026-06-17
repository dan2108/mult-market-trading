# CLAUDE.md — standing rules for this repo

Loaded into every session. This is the operating contract for working on this codebase.
**BUILD.md is the living source of truth** for what we are building and why — read it.

## What this is
A diversified, trend-following multi-market (FX / stock / crypto) trading platform. A boring,
proven signal (time-series momentum) run across many uncorrelated markets, with the real
engineering poured into **risk and position sizing** — not the entry. Edge is *discovered and
validated*, never chosen because it's popular.

## Non-negotiable principles (these protect the money)
1. **Backtest–live parity.** `tfx/core` (`signal`, `sizing`, `risk`) is written ONCE as a shared,
   deterministic library. The backtester and the live engine import the SAME code. What we
   validate is literally what trades.
2. **No LLM in the live decision/execution path.** LLMs are for research, building, and ops
   observation only — never deciding or executing trades.
3. **Deterministic core.** No wall-clock / RNG non-determinism in `tfx/core` or `tfx/data`.
   No ML making trade calls in v1 — the decision rule is transparent code.
4. **Costs modeled pessimistically** — spread + swap + slippage on every fill. Never backtest
   on mid-price.
5. **Point-in-time discipline / no look-ahead.** Every stored row holds only fields knowable at
   that timestamp. No whole-series normalization or future-derived columns in the per-bar cache.
6. **Staged rollout:** backtest → paper → tiny live → scale. Guardrails always on (per-trade risk
   cap, portfolio-heat cap, correlation cap, drawdown auto-halt, kill switch).

## The gate
Nothing goes live until `validate` shows positive expectancy out-of-sample, after realistic costs,
and after a ~50% haircut on backtested performance. **Build quality ≠ edge. The gate decides.**
Green tests = done; not green = not done, regardless of what any agent claims.

## Conventions
- Python ≥ 3.11, `src/` layout, single CLI entrypoint `tfx` (Typer). Import package: `tfx`.
- Test-first for `tfx/data` and `tfx/core`. When touching `core` (signal/sizing/risk) or
  `backtest`, go slow and run `/code-review ultra` on the diff — pointed at parity, look-ahead,
  cost realism, and whether risk limits actually fire.
- Secrets live in `.env` (gitignored); never commit credentials. `data/cache/` is gitignored;
  `data/sample/` (deterministic fixtures) is committed.
- Determinism: cache writes must be byte-reproducible (stable sort + fixed serialization).

## Commands
- Install:   `pip install -e ".[dev]"`
- Test:      `pytest -q`
- Lint:      `ruff check .`
- CLI help:  `tfx --help`
- Pull data: `tfx data pull --source fixture`
- Quality:   `tfx data report`

## Build order
`data-pull` → `costs` → core (`signal`/`sizing`/`risk`) → `backtest` → `validate` → `report`
→ (Phase 2: `broker` → `live` → `monitor`). **Currently at: costs (data-pull done).**
