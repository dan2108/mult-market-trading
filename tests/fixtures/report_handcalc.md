# Hand-computed report metrics (backs `test_01`/`test_02` in tests/test_report.py)

Fixture equity curve (bar 0 = initial equity 1.0), per-bar returns chosen round:

```
returns r:          +0.10   -0.10   +0.05   -0.05   +0.10
equity (cumprod):    1.10    0.99    1.0395  0.987525  1.0862775
full curve:  [1.0, 1.10, 0.99, 1.0395, 0.987525, 1.0862775]   (6 bars)
```

- **total_return** = 1.0862775 - 1 = +0.0862775 exactly
- **CAGR** = 1.0862775^(252/6) - 1  (years = n_bars/252 = 6/252; asserted via the formula)
- **Sharpe** = mean(r)/std(r, ddof=1) * sqrt(252), mean = 0.02, asserted via the formula
- **Sortino**: downside DEVIATION (not the std of losing bars alone) = RMS of min(r, 0) over
  ALL 5 bars, target 0: min(r,0) = {0, -0.10, 0, -0.05, 0}; squared = {0, 0.01, 0, 0.0025, 0};
  mean = 0.0025; sqrt = 0.05 exactly. Sortino = 0.02 / 0.05 * sqrt(252) = 6.34980...
- **max_drawdown**: running peak hits 1.10 at bar 1; trough 0.987525 at bar 4:
  0.987525/1.10 - 1 = **-0.10225** exactly
- **drawdown duration**: bars 2..5 are all below the 1.10 peak (bar 5 ends at 1.0862775 < 1.10,
  still under water) -> longest run = **4 bars**; **time_in_drawdown** = 4/6
- **hit rate** = 3 wins / 5 non-zero bars = **0.6**;
  **avg win** = (0.10 + 0.05 + 0.10)/3 = 0.0833...; **avg loss** = (-0.10 - 0.05)/2 = -0.075

The equity curve is injected into a synthetic BacktestResult so these are pure-arithmetic
checks; contribution/correlation/turnover metrics are pinned against a REAL engine run in the
other tests (they must satisfy the engine's accounting identity, not a hand-typed number).
