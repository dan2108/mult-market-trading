# Hand-computed cost example (backs `test_09_handcomputed_fixture_trade`)

Clean round params (NOT the provisional table — chosen so the arithmetic is transparent):

```
spread        = 1.00   (price units)
spread_floor  = 0.50
commission_bps= 2.0
slippage_frac = 0.0010
swap_long_bps = 1.0
swap_short_bps= 3.0
```

Units: cost is a **return fraction of equity**; `size` is a position weight. `notional weight = |size|`.

## Trade: LONG, size = 10.0, price = 100.0, nights_held = 0

```
effective_spread = max(1.00, 0.50)            = 1.00
spread (half)    = 10 * (1.00 / 2) / 100.0    = 10 * 0.005   = 0.050
commission       = 10 * (2.0 * 1e-4)          = 10 * 0.0002  = 0.002
slippage         = 10 * 0.0010                              = 0.010
swap             = 0 (nights_held = 0)                       = 0.000
-------------------------------------------------------------------
total = apply()  = 0.050 + 0.002 + 0.010 + 0  = 0.062
```

## Holding the same position overnight

```
swap, 3 nights LONG  = 10 * (1.0 * 1e-4) * 3  = 10 * 0.0001 * 3 = 0.003
swap, 3 nights SHORT = 10 * (3.0 * 1e-4) * 3  = 10 * 0.0003 * 3 = 0.009
```

Long vs short swaps differ (1.0 vs 3.0 bps/night) and scale linearly with nights — the
sign/asymmetry checks in acceptance test #2.
