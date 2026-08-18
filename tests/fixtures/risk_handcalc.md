# Hand-computed risk-veto example (backs `test_07_all_limits_fire_and_compose` and
# `test_08_handcomputed_fixture_slice` in tests/test_risk.py)

Real basket symbols so the asset-class lookup is exercised: `EUR/USD` (fx), `AUD/JPY` (fx),
`Gold` (commodity).

Params: `per_instrument_cap = 0.5`, `asset_class_cap = 0.75`, `portfolio_heat_cap = 1.0`,
account healthy (no drawdown, no kill switch).

Proposal: `EUR/USD = +0.8`, `AUD/JPY = -0.4`, `Gold = +0.3`.

Stages apply in the fixed order per-instrument -> asset-class -> heat:

```
1. per-instrument cap (0.5):
   EUR/USD |+0.8| > 0.5 -> scale 0.5/0.8 = 0.625 -> +0.5
   AUD/JPY, Gold unchanged                  -> {+0.5, -0.4, +0.3}

2. asset-class cap (0.75) on fx gross:
   fx gross = 0.5 + 0.4 = 0.9 > 0.75 -> scale 0.75/0.9 = 5/6
   EUR/USD = 0.5 * 5/6  = 5/12  (= 0.41666...)
   AUD/JPY = -0.4 * 5/6 = -1/3  (= -0.33333...)
   commodity gross = 0.3 <= 0.75 -> Gold unchanged

3. portfolio-heat cap (1.0):
   gross = 5/12 + 1/3 + 3/10 = (25 + 20 + 18)/60 = 63/60 = 1.05 > 1.0
   scale = 1/1.05 = 20/21
   EUR/USD = 5/12 * 20/21 = 25/63  (= +0.396825...)
   AUD/JPY = -1/3 * 20/21 = -20/63 (= -0.317460...)
   Gold    = 3/10 * 20/21 = 2/7    (= +0.285714...)

   final gross = 25/63 + 20/63 + 18/63 = 63/63 = 1.0 exactly (the heat cap binds tight)
```

Three reasons emitted, one per stage: PER_INSTRUMENT (EUR/USD, scale 0.625),
ASSET_CLASS (fx, scale 5/6), PORTFOLIO_HEAT (scale 20/21). Every |final| <= |proposed|,
no sign flipped, and all three caps hold simultaneously.
