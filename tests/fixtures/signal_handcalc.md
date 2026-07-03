# Hand-computed signal examples

## Single lookback (backs `test_08_handcomputed_fixture_slice`)

The degenerate single-entry ensemble `lookbacks = (2,)` — these are the ORIGINAL pre-ensemble
hand-calc values, retained verbatim as the regression anchor. Single instrument `X`, closes over
6 bars:

```
t:      0     1     2     3     4     5
close: 100   100   100    90   121   100
```

Direction = sign(close[t] / close[t - lookback] - 1), NaN where `t - lookback < 0`:

```
t=0: no anchor (t - 2 < 0)              -> NaN
t=1: no anchor (t - 2 < 0)              -> NaN
t=2: 100 / 100 - 1 =  0.0000            -> direction =  0   (flat: exact tie)
t=3:  90 / 100 - 1 = -0.1000            -> direction = -1
t=4: 121 / 100 - 1 = +0.2100            -> direction = +1
t=5: 100 /  90 - 1 = +0.1111            -> direction = +1
```

Covers all three discrete outcomes (0 / -1 / +1) plus the NaN warm-up period, in one auditable
slice.

## Lookback ensemble (backs `test_ensemble_handcomputed_fixture_slice`)

`lookbacks = (1, 3)`, single instrument `X`, closes over 9 bars:

```
t:      0     1     2     3     4     5     6     7     8
close: 100   102   101   101    99   104   103    90    90
```

Direction = mean of sign(close[t] / close[t - L] - 1) over L in {1, 3}; NaN until every anchor
exists (t - 3 >= 0):

```
t=0..2: 3-bar anchor missing                                               -> NaN
t=3: L=1: 101/101 - 1 =  0      -> 0    L=3: 101/100 - 1 = +0.0100 -> +1   mean = +0.5
t=4: L=1:  99/101 - 1 = -0.0198 -> -1   L=3:  99/102 - 1 = -0.0294 -> -1   mean = -1.0
t=5: L=1: 104/99  - 1 = +0.0505 -> +1   L=3: 104/101 - 1 = +0.0297 -> +1   mean = +1.0
t=6: L=1: 103/104 - 1 = -0.0096 -> -1   L=3: 103/101 - 1 = +0.0198 -> +1   mean =  0.0
t=7: L=1:  90/103 - 1 = -0.1262 -> -1   L=3:  90/99  - 1 = -0.0909 -> -1   mean = -1.0
t=8: L=1:  90/90  - 1 =  0      -> 0    L=3:  90/104 - 1 = -0.1346 -> -1   mean = -0.5
```

Hits every value on the len-2 grid {-1, -0.5, 0, +0.5, +1} plus the all-horizons warm-up rule,
in one auditable slice.
