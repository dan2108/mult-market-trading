"""Acceptance tests for `validate` — THE GATE (BUILD.md Step 7, tests 1-9) + units.

The two-sided sanity pair is the heart: a driftless random walk MUST fail and a persistent
trend MUST pass, on the same protocol. Everything is seeded/deterministic — no bootstrap, no
RNG in the module under test (randomness lives only in tests/synthetic.py fixtures).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd
import pytest
from pydantic import ValidationError

from synthetic import gbm_panel, panel_from_closes, trending_panel
from tfx.validate import GridPoint, ValidateProtocol, Verdict, folds, run
from tfx.validate.runner import _robustness, _verdict
from tfx.validate.stats import deflated_sharpe_ratio, sharpe, stitch, t_stat

SYMBOLS = ["EUR/USD", "Gold"]

# Small, fast protocol for the verdict tests: 3+ folds on 360 bars, 2-candidate grid (so the
# modal-share robustness check cannot be tripped by noise alone on a clean fixture).
PROTOCOL = ValidateProtocol(
    train_bars=120, test_bars=60, step_bars=60, min_folds=3,
    grid=(
        GridPoint(lookbacks=(5,), ewma_halflife=10.0),
        GridPoint(lookbacks=(10,), ewma_halflife=10.0),
    ),
    min_trades=10,
)
N_BARS = 360  # -> 4 folds


# 1 — Strict fold separation: a fold's selection is a function of its train window only.
def test_01_fold_separation_no_leakage():
    panel = trending_panel(SYMBOLS, N_BARS, seed=7)
    base = run(panel, PROTOCOL)

    fold_list = folds(len(panel.index), PROTOCOL)
    for fold in fold_list:
        assert fold.train_end == fold.test_start          # adjacent, never overlapping
        assert fold.train_start < fold.train_end <= fold.test_start < fold.test_end

    # perturb ONLY fold 1's test window (bars 180..240): fold 1's selection and train score
    # must not move (they are train-only), and fold 0 (entirely earlier) must be untouched
    target = fold_list[1]
    perturbed_close = {
        symbol: panel[(symbol, "close")].to_numpy().copy() for symbol in SYMBOLS
    }
    for symbol in SYMBOLS:
        segment = perturbed_close[symbol][target.test_start : target.test_end]
        perturbed_close[symbol][target.test_start : target.test_end] = segment * 1.03
    perturbed = run(panel_from_closes(perturbed_close), PROTOCOL)

    assert perturbed.folds[1].selected == base.folds[1].selected
    assert perturbed.folds[1].train_sharpe == pytest.approx(base.folds[1].train_sharpe)
    assert perturbed.folds[0] == base.folds[0]


# 2 — Walk-forward correctness: windows roll forward only; no fold sees its own future.
def test_02_windows_roll_forward_only():
    fold_list = folds(500, ValidateProtocol(train_bars=120, test_bars=60, step_bars=60))
    assert [f.fold for f in fold_list] == list(range(len(fold_list)))
    for previous, current in zip(fold_list, fold_list[1:], strict=False):
        assert current.test_start == previous.test_start + 60
        assert current.train_start == previous.train_start + 60
    # exact geometry of the first fold
    first = fold_list[0]
    assert (first.train_start, first.train_end, first.test_start, first.test_end) == (
        0, 120, 120, 180,
    )


# 3 — Catches overfitting: in-sample selection finds lucky params on noise; OOS says no.
def test_03_overfit_on_noise_fails_oos():
    # seed=17 is a genuine trap: several folds find in-sample Sharpe up to ~1.0 on pure noise
    result = run(gbm_panel(SYMBOLS, N_BARS, seed=17), PROTOCOL)
    # selection DID find something in-sample (that's the trap the gate must not fall for)
    assert any(f.train_sharpe > 0 for f in result.folds)
    assert result.verdict is not Verdict.PASS


# 4 — Catches no-edge: on a driftless random walk the verdict is FAIL.
def test_04_random_walk_fails():
    result = run(gbm_panel(SYMBOLS, N_BARS, seed=29), PROTOCOL)
    assert result.verdict in (Verdict.FAIL, Verdict.INSUFFICIENT_EVIDENCE)
    # and specifically not a squeaker: haircut Sharpe nowhere near the bar
    assert not result.oos_metrics["sharpe_haircut"] >= PROTOCOL.min_sharpe_after_haircut or (
        result.oos_metrics["t_stat"] < PROTOCOL.min_tstat
    )


# 5 — Passes real edge: a genuinely-trending fixture gets a PASS.
def test_05_trending_passes():
    result = run(trending_panel(SYMBOLS, N_BARS, seed=7), PROTOCOL)
    assert result.verdict is Verdict.PASS
    assert result.oos_metrics["sharpe_haircut"] >= PROTOCOL.min_sharpe_after_haircut
    assert result.oos_metrics["t_stat"] >= PROTOCOL.min_tstat
    assert result.oos_metrics["n_trades"] >= PROTOCOL.min_trades


# 6 — Robustness: a knife-edge optimum flags FRAGILE (unit-tested on the decision helpers,
# which the integration path above already exercises end to end).
def test_06_knife_edge_flags_fragile():
    a = GridPoint(lookbacks=(5,), ewma_halflife=10.0)
    b = GridPoint(lookbacks=(10,), ewma_halflife=10.0)
    protocol = PROTOCOL

    unstable = _robustness([a, b, a, b], {a.label(): 1.0, b.label(): 1.0}, protocol)
    assert not unstable["fragile"]  # 2/4 modal share == min_modal_share: not fragile
    scattered = _robustness([a, b, b, a][:3], {a.label(): 1.0, b.label(): 1.0}, protocol)
    assert scattered["modal_share"] < 1.0

    # the edge lives in one cell: grid mean OOS <= 0 -> fragile even with a stable winner
    lucky_cell = _robustness([a, a, a, a], {a.label(): 2.0, b.label(): -2.5}, protocol)
    assert lucky_cell["fragile"]

    # unstable selection -> fragile
    churny = _robustness([a, b, a, b, b, a][:4], {a.label(): 1.0, b.label(): 0.9}, protocol)
    assert churny["modal_share"] == 0.5
    strict_protocol = ValidateProtocol(
        train_bars=120, test_bars=60, grid=(a, b), min_modal_share=0.75,
    )
    assert _robustness([a, b, a, b], {a.label(): 1.0, b.label(): 0.9}, strict_protocol)[
        "fragile"
    ]

    # and a fragile flag downgrades an otherwise-passing result to FRAGILE
    passing_metrics = {
        "n_trades": 1000, "sharpe_haircut": 1.0, "t_stat": 5.0,
    }
    assert (
        _verdict(passing_metrics, {"fragile": True}, protocol) is Verdict.FRAGILE
    )
    assert _verdict(passing_metrics, {"fragile": False}, protocol) is Verdict.PASS


# 7 — Haircut applied: thresholds see haircut numbers; both raw and haircut are reported.
def test_07_haircut_arithmetic_and_gating():
    result = run(trending_panel(SYMBOLS, N_BARS, seed=7), PROTOCOL)
    metrics = result.oos_metrics
    assert metrics["sharpe_haircut"] == pytest.approx(metrics["sharpe_raw"] * 0.5, rel=1e-12)
    assert metrics["mean_daily_haircut"] == pytest.approx(
        metrics["mean_daily_raw"] * 0.5, rel=1e-12
    )

    # a raw Sharpe that would pass un-haircut but not after the haircut must FAIL
    borderline = {"n_trades": 1000, "sharpe_haircut": 0.45, "t_stat": 5.0}
    assert _verdict(borderline, {"fragile": False}, PROTOCOL) is Verdict.FAIL


# 8 — Significance gate: below the minimum trade count the verdict is INSUFFICIENT_EVIDENCE.
def test_08_insufficient_evidence_below_min_trades():
    demanding = ValidateProtocol(
        train_bars=120, test_bars=60, step_bars=60, min_folds=3,
        grid=PROTOCOL.grid, min_trades=100_000,
    )
    result = run(trending_panel(SYMBOLS, N_BARS, seed=7), demanding)
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE

    starved = {"n_trades": 3, "sharpe_haircut": 2.0, "t_stat": 9.0}
    assert _verdict(starved, {"fragile": False}, PROTOCOL) is Verdict.INSUFFICIENT_EVIDENCE
    undefined = {"n_trades": 1000, "sharpe_haircut": float("nan"), "t_stat": float("nan")}
    assert _verdict(undefined, {"fragile": False}, PROTOCOL) is Verdict.INSUFFICIENT_EVIDENCE


# 9 — Pre-set thresholds + full determinism: frozen protocol, verbatim echo, identical reruns.
def test_09_thresholds_frozen_echoed_and_run_deterministic():
    with pytest.raises(ValidationError):
        PROTOCOL.haircut = 0.9  # type: ignore[misc]

    panel = trending_panel(SYMBOLS, N_BARS, seed=7)
    first = run(panel, PROTOCOL)
    second = run(panel, PROTOCOL)

    assert first.protocol == PROTOCOL and second.protocol == PROTOCOL  # echoed verbatim
    assert first.verdict == second.verdict
    assert first.folds == second.folds
    pd.testing.assert_series_equal(first.oos_returns, second.oos_returns)
    assert first.oos_metrics == second.oos_metrics
    assert first.robustness == second.robustness


# --- additional unit tests -------------------------------------------------------------------
def test_folds_too_short_history_raises():
    with pytest.raises(ValueError, match="fold"):
        folds(200, ValidateProtocol(train_bars=120, test_bars=60, step_bars=60, min_folds=3))


def test_stitch_rejects_overlap():
    index = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    piece = pd.Series(0.01, index=index)
    with pytest.raises(ValueError, match="overlap"):
        stitch([piece, piece])


def test_sharpe_and_tstat_edge_cases():
    empty = pd.Series(dtype=float)
    assert math.isnan(sharpe(empty)) and math.isnan(t_stat(empty))
    flat = pd.Series([0.01, 0.01, 0.01])
    assert math.isnan(sharpe(flat))  # zero variance is undefined, not infinite


def test_deflated_sharpe_matches_independent_formula():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.007, -0.01, 0.003, 0.015])
    n_trials = 5
    actual = deflated_sharpe_ratio(returns, n_trials)

    # independent recomputation (Bailey & Lopez de Prado closed form), test-side
    values = returns.to_numpy()
    n = len(values)
    sr = values.mean() / values.std(ddof=1)
    centered = values - values.mean()
    m2 = (centered**2).mean()
    skew = (centered**3).mean() / m2**1.5
    kurt = (centered**4).mean() / m2**2
    euler_gamma = 0.5772156649015329
    z_hi = NormalDist().inv_cdf(1 - 1 / n_trials)
    z_lo = NormalDist().inv_cdf(1 - 1 / (n_trials * math.e))
    sr0 = math.sqrt(1 / (n - 1)) * ((1 - euler_gamma) * z_hi + euler_gamma * z_lo)
    expected = NormalDist().cdf(
        (sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    )
    assert actual == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= actual <= 1.0
    # more trials can only deflate (weakly) -- the multiple-testing discount
    assert deflated_sharpe_ratio(returns, 50) <= actual + 1e-12


def test_grid_default_is_preregistered_and_canonical():
    protocol = ValidateProtocol()
    assert len(protocol.grid) == 18
    labels = [p.label() for p in protocol.grid]
    assert len(set(labels)) == 18  # no duplicate candidates
    singles = [p for p in protocol.grid if len(p.lookbacks) == 1]
    ensembles = [p for p in protocol.grid if len(p.lookbacks) > 1]
    assert singles and ensembles  # ensemble-vs-single is decided BY the sweep


def test_verdict_ordering_thresholds_before_robustness():
    failing = {"n_trades": 1000, "sharpe_haircut": 0.1, "t_stat": 0.5}
    # a result that fails thresholds is FAIL even if it is also fragile
    assert _verdict(failing, {"fragile": True}, PROTOCOL) is Verdict.FAIL
