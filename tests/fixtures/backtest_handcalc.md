# Hand-computed backtest slice (backs `test_09_handcalc_and_oracle` in tests/test_backtest.py)

Single instrument `EUR/USD` (real basket symbol, so real cost params: spread 0.00008,
spread_floor 0.00006, slippage 0.00002, swap_long 1.0 bps), business-day bars, closes rising
exactly +1% per bar:

```
t:      0        1        2          3
close:  1.00000  1.01000  1.02010    1.03030
```

Params: `lookbacks = (1,)`, `ewma_halflife = 1.0` (lambda = 0.5, min_periods = 1),
`target_vol = 0.10`, leverage/risk caps set far out of the way. Initial equity 1.0.

## Bar 1 — first decision

- direction[1] = sign(1.01/1.00 - 1) = +1
- log return r1 = ln(1.01) = 0.00995033...
- EWMA covariance after bar 1 = (1 - 0.5) * r1^2 ; annualized sigma
  = sqrt(0.5 * r1^2 * 252) = r1 * sqrt(126)
  = 0.00995033 * 11.2249722 = 0.111692...
- Single instrument, so the portfolio scalar reduces exactly to weight = target_vol / sigma_ann:

      w1 = 0.10 / (ln(1.01) * sqrt(126)) = 0.895322...

- Risk passes it through (caps not binding). w1 becomes the pending order.

## Bar 2 — first fill (at the NEXT bar's close, never the decision bar's)

- Fill at close[2] = 1.02010, delta = w1 (from flat).
- Fill cost = w1 * ( (max(spread, floor)/2) / fill_price + slippage_frac )
            = w1 * ( (0.00008/2) / 1.0201 + 0.00002 )
            = w1 * 5.92118e-5  =  5.30144e-5 (approximately; the test asserts the formula
              to rel=1e-12, not a rounded decimal)
- No pnl at the fill bar (position was flat overnight; the fill marks at the fill price).
- No swap (held was zero overnight).

      equity[2] = 1.0 * (1 - w1 * ((0.00008/2)/1.0201 + 0.00002))  ~  0.99994699

## Bar 3 — first earning bar + first swap

- pnl = w1 * (close[3]/close[2] - 1) = w1 * 0.01 (the +1% bar)
- swap for 1 calendar night at 1.0 bps: w1 * 1e-4
- plus the rebalance fill cost on |w2 - w1| at close[3] (w2 from the updated EWMA vol).

Beyond bar 3 the arithmetic compounds; the test cross-checks the ENTIRE run against an
independent oracle loop written in test code (same shared-core functions, independent
re-implementation of the engine mechanics: fills, costs, swap nights, equity compounding), the
same dual-implementation pattern `sizing_handcalc.md` uses for matrix math.
