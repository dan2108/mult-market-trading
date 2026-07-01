"""The shared deterministic core: `signal`, `sizing`, `risk`.

Identical code path in backtest and live (Phase 2) -- what `validate` proves is exactly what
executes. Pure functions only: no I/O, no wall-clock, no RNG, no hidden state, and never an LLM
in this path (CLAUDE.md non-negotiables #1-3).
"""
