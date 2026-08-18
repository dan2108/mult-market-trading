"""The simulator (BUILD.md Step 6): walk the panel bar by bar through the EXACT shared core.

Not part of the deterministic core itself -- the engine *imports* `signal`/`sizing`/`risk` (the
same modules `live` will import) and never re-implements any of their logic. Parity is asserted
structurally: the engine binds the core functions as module attributes and the tests check
function-object identity.
"""

from .engine import BacktestParams, BacktestResult, run

__all__ = ["BacktestParams", "BacktestResult", "run"]
