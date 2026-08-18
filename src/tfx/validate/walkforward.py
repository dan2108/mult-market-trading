"""Rolling walk-forward fold geometry: pure index arithmetic, windows roll FORWARD only.

Rolling (not anchored): every fold trains on the ``train_bars`` immediately before its test
window, so no fold overweights ancient regimes and no fold ever sees its own future. Parameters
are fit on train windows only; indicator warm-up inside a window is point-in-time data
discipline, not parameter leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import ValidateProtocol


@dataclass(frozen=True)
class Fold:
    """Positional half-open index ranges into the panel: [start, end)."""

    fold: int
    train_start: int
    train_end: int      # == test_start (train ends where test begins)
    test_start: int
    test_end: int


def folds(n_bars: int, protocol: ValidateProtocol) -> list[Fold]:
    """All folds that fit ``n_bars``. Raises if history supports fewer than ``min_folds`` --
    an underpowered validation must fail loudly, never quietly shrink."""
    out: list[Fold] = []
    test_start = protocol.train_bars
    while test_start + protocol.test_bars <= n_bars:
        out.append(
            Fold(
                fold=len(out),
                train_start=test_start - protocol.train_bars,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + protocol.test_bars,
            )
        )
        test_start += protocol.step_bars
    if len(out) < protocol.min_folds:
        raise ValueError(
            f"history supports only {len(out)} fold(s); protocol requires >= "
            f"{protocol.min_folds} (train {protocol.train_bars} + test {protocol.test_bars} "
            f"bars, step {protocol.step_bars}, {n_bars} bars available)"
        )
    return out
