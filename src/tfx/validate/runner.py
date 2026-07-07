"""The walk-forward runner: per fold, evaluate the pre-registered grid, select on TRAIN only,
score on TEST only; stitch the out-of-sample record; apply haircut, significance and robustness;
return the verdict.

TWO backtests per (fold, candidate), not one combined run:

1. Selection: a standalone run over [train_start, test_start). Equivalent to scoring the
   train-prefix of a combined [train_start, test_end) run (the engine is prefix-invariant -- its
   cardinal no-look-ahead property, proven in its own acceptance tests -- so a standalone
   train-only run produces bit-identical bars over that range), but needs no test-window data at
   all and no post-hoc slicing. Selection trims each candidate's OWN warm-up prefix before
   scoring Sharpe (`GridPoint.warmup_bars()`): a longer lookback/halflife means more exact-zero
   warm-up bars, and including them dilutes Sharpe toward zero by roughly sqrt(active_fraction)
   -- a structural bias toward short-horizon candidates that has nothing to do with which one
   actually trades better.

2. OOS scoring: a FRESH run starting `train_bars` before test_start (the protocol guarantees
   every candidate's warm-up fits inside that buffer) through test_end, using only the
   [test_start, test_end) portion of ITS OWN output. This is deliberately NOT the same backtest
   selection used -- a single run spanning train into test lets drawdown accumulated during
   TRAINING carry into the test window's equity/peak, so the halt (or its absence) reflects
   training-era history a genuinely fresh deployment at test_start would never have seen. Every
   candidate gets this fresh run (not just the fold's eventual winner), because the same
   contamination would otherwise sit inside `per_candidate_oos` -- the robustness check's own
   evidence that the edge isn't a single lucky cell.

Selection ties break by pre-registered grid order -- deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest import BacktestParams
from ..backtest import run as run_backtest
from ..core.risk import RiskParams
from ..core.signal import SignalParams
from ..core.sizing import SizingParams
from ..costs.model import CostModel
from .protocol import GridPoint, ValidateProtocol, Verdict
from .stats import (
    daily_returns,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe,
    stitch,
    t_stat,
)
from .walkforward import Fold, folds


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected: GridPoint
    train_sharpe: float
    test_sharpe: float
    test_trades: int


@dataclass(frozen=True)
class ValidateResult:
    folds: tuple[FoldResult, ...]
    oos_returns: pd.Series          # stitched test-window daily returns of the selected params
    oos_metrics: dict               # raw AND haircut numbers, side by side
    robustness: dict
    protocol: ValidateProtocol      # echoed verbatim: the thresholds the verdict used
    verdict: Verdict


def _backtest_params(point: GridPoint) -> BacktestParams:
    """Grid point -> full param bundle. Only lookbacks and halflife vary; target vol, leverage
    cap and every risk limit are held fixed (policy, not edge)."""
    return BacktestParams(
        signal=SignalParams(lookbacks=point.lookbacks),
        sizing=SizingParams(ewma_halflife=point.ewma_halflife),
        risk=RiskParams(),
    )


def _score(value: float) -> float:
    """Selection score: NaN-safe (a candidate with no defined Sharpe can never win)."""
    return -math.inf if math.isnan(value) else value


def _active_returns(equity: pd.Series, point: GridPoint) -> pd.Series:
    """Daily returns of an equity curve, excluding this candidate's own warm-up prefix (exact
    zero-return bars, since no position exists yet). See module docstring for why including
    warm-up in a cross-candidate Sharpe comparison is a structural bias, not a neutral choice."""
    returns = daily_returns(equity)
    warmup = point.warmup_bars()
    if warmup >= len(equity.index):
        return returns.iloc[0:0]
    cutoff = equity.index[warmup]
    return returns[returns.index >= cutoff]


def _robustness(
    selected: list[GridPoint], per_candidate_oos: dict[str, float], protocol: ValidateProtocol
) -> dict:
    labels = [point.label() for point in selected]
    modal_label = max(sorted(set(labels)), key=labels.count)  # sorted() = deterministic ties
    modal_share = labels.count(modal_label) / len(labels)

    grid_sharpes = np.array(list(per_candidate_oos.values()), dtype=float)
    defined = grid_sharpes[~np.isnan(grid_sharpes)]
    grid_mean = float(defined.mean()) if len(defined) else float("nan")

    fragile = modal_share < protocol.min_modal_share or not grid_mean > 0.0
    return {
        "modal_candidate": modal_label,
        "modal_share": modal_share,
        "grid_mean_oos_sharpe": grid_mean,
        "fragile": fragile,
    }


def _verdict(oos_metrics: dict, robustness: dict, protocol: ValidateProtocol) -> Verdict:
    """Fixed decision order: enough evidence? -> thresholds (on HAIRCUT numbers)? -> robust?"""
    if (
        oos_metrics["n_trades"] < protocol.min_trades
        or math.isnan(oos_metrics["sharpe_haircut"])
        or math.isnan(oos_metrics["t_stat"])
    ):
        return Verdict.INSUFFICIENT_EVIDENCE
    if (
        oos_metrics["sharpe_haircut"] < protocol.min_sharpe_after_haircut
        or oos_metrics["t_stat"] < protocol.min_tstat
    ):
        return Verdict.FAIL
    if robustness["fragile"]:
        return Verdict.FRAGILE
    return Verdict.PASS


def run(
    panel: pd.DataFrame,
    protocol: ValidateProtocol | None = None,
    cost_model: CostModel | None = None,
) -> ValidateResult:
    """Walk the folds. Deterministic: same panel + protocol -> same verdict, bit for bit."""
    protocol = protocol or ValidateProtocol()
    cost_model = cost_model or CostModel()
    index = panel.index
    fold_list: list[Fold] = folds(len(index), protocol)

    fold_results: list[FoldResult] = []
    selected_points: list[GridPoint] = []
    oos_pieces: list[pd.Series] = []
    oos_trades = 0
    per_candidate_pieces: dict[str, list[pd.Series]] = {p.label(): [] for p in protocol.grid}

    # Small safety margin beyond a candidate's own warm-up: keeps the first scored OOS bar's
    # return comfortably past the exact bar the candidate's first real decision appears on.
    oos_buffer_margin_bars = 5

    for fold in fold_list:
        train_window = panel.iloc[fold.train_start : fold.test_start]
        test_start_ts = index[fold.test_start]

        best: tuple[float, GridPoint, pd.Series, int] | None = None
        for point in protocol.grid:
            train_result = run_backtest(train_window, _backtest_params(point), cost_model)
            train_returns = _active_returns(train_result.equity_curve, point)
            score = _score(sharpe(train_returns))

            # Fresh OOS run: starts only `warmup_bars` before test_start -- NOT at train_start
            # -- so equity/peak/drawdown-halt state at test_start reflects a genuine fresh
            # deployment. During the buffer itself the candidate has no real decision yet (that
            # is what "warm-up" means), so the position stays flat and equity stays at
            # initial_equity right up to test_start: the (potentially much longer) training
            # period's drawdown history never enters this run's account state at all.
            buffer_bars = point.warmup_bars() + oos_buffer_margin_bars
            oos_window = panel.iloc[max(0, fold.test_start - buffer_bars) : fold.test_end]
            oos_result = run_backtest(oos_window, _backtest_params(point), cost_model)
            oos_returns_full = daily_returns(oos_result.equity_curve)
            test_returns = oos_returns_full[oos_returns_full.index >= test_start_ts]
            per_candidate_pieces[point.label()].append(test_returns)

            if best is None or score > best[0]:  # strict >: ties keep earlier grid order
                n_test_trades = int((oos_result.trades["timestamp"] >= test_start_ts).sum())
                best = (score, point, test_returns, n_test_trades)

        train_score, point, test_returns, n_test_trades = best
        selected_points.append(point)
        oos_pieces.append(test_returns)
        oos_trades += n_test_trades
        fold_results.append(
            FoldResult(
                fold=fold.fold,
                train_start=index[fold.train_start],
                test_start=index[fold.test_start],
                test_end=index[fold.test_end - 1],
                selected=point,
                train_sharpe=train_score if train_score != -math.inf else float("nan"),
                test_sharpe=sharpe(test_returns),
                test_trades=n_test_trades,
            )
        )

    oos_returns = stitch(oos_pieces)
    per_candidate_oos = {
        label: sharpe(stitch(pieces)) for label, pieces in per_candidate_pieces.items()
    }

    raw_sharpe = sharpe(oos_returns)
    mean_daily = float(oos_returns.mean()) if len(oos_returns) else float("nan")
    oos_equity = (1.0 + oos_returns).cumprod()
    oos_metrics = {
        "n_bars": int(len(oos_returns)),
        "n_trades": int(oos_trades),
        "sharpe_raw": raw_sharpe,
        "sharpe_haircut": raw_sharpe * protocol.haircut,
        "mean_daily_raw": mean_daily,
        "mean_daily_haircut": mean_daily * protocol.haircut,
        "t_stat": t_stat(oos_returns),
        "max_drawdown": max_drawdown(oos_equity),
        "deflated_sharpe": deflated_sharpe_ratio(oos_returns, n_trials=len(protocol.grid)),
    }
    robustness = _robustness(selected_points, per_candidate_oos, protocol)

    return ValidateResult(
        folds=tuple(fold_results),
        oos_returns=oos_returns,
        oos_metrics=oos_metrics,
        robustness=robustness,
        protocol=protocol,
        verdict=_verdict(oos_metrics, robustness, protocol),
    )
