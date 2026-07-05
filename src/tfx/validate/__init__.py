"""THE GATE (BUILD.md Step 7): the honest verdict on whether an edge exists.

Runs the backtest under a rolling walk-forward protocol with a PRE-REGISTERED parameter grid and
PRE-SET thresholds, applies the ~50% haircut, checks significance and robustness, and returns
pass/fail. Deterministic end to end -- no bootstrap, no RNG anywhere. Nothing goes live on any
other evidence.
"""

from .protocol import GridPoint, ValidateProtocol, Verdict
from .runner import ValidateResult, run
from .walkforward import Fold, folds

__all__ = [
    "Fold",
    "GridPoint",
    "ValidateProtocol",
    "ValidateResult",
    "Verdict",
    "folds",
    "run",
]
