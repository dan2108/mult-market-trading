# Hand-computed signal example (backs `test_08_handcomputed_fixture_slice`)

`lookback = 2`, single instrument `X`, closes over 6 bars:

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
