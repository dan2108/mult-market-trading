# Hand-computed sizing example (backs `test_08_handcomputed_fixture_slice`)

Model (see `tfx.core.sizing`): at each bar t,

```
lambda          = 0.5 ** (1 / halflife)
cov_t           = lambda * cov_{t-1} + (1 - lambda) * outer(r_t, r_t)      # r_t = log returns
annualized_t    = cov_t * 252
vol_t           = sqrt(diag(annualized_t))
raw_weight_t    = direction_t / vol_t                                      # inverse-vol, per name
portfolio_var_t = raw_weight_t . annualized_t . raw_weight_t                # quadratic form
scalar_t        = target_vol / sqrt(portfolio_var_t)
weight_t        = raw_weight_t * scalar_t                                  # then leverage-capped
```

`cov_0 = 0` (no prior state); a bar only counts toward the burn-in once its return is real
(`min_periods = ceil(halflife)` valid bars needed before a weight is produced).

## Setup

2 instruments, 3 bars, `halflife = 1` (so `lambda = 0.5`, `min_periods = 1` -- a weight is
produced from the first bar with a real return onward), `target_vol = 0.10`, leverage cap slack
(never binds in this fixture):

```
t:        0      1      2
A close: 100    110     90
B close: 100    105    100
```

Directions: `A = [NaN, +1, +1]`, `B = [NaN, +1, -1]` (bar 0 has no return yet, matching
`signal`'s own warm-up convention).

## Why this isn't typed out further by hand

Two return vectors feed a running 2x2 covariance recursion, then a matrix quadratic form per
bar -- exact enough to do with a calculator, but transcribing every multiplication here is more
error-prone than verifying it in code. `test_08` (and `test_02`, on a longer fixture) instead
re-derive this exact recursion **independently** in the test file, using plain numpy with no
import from `tfx.core.sizing` -- a from-scratch reference implementation of the formula above,
compared against the real `size()` output. That is the audit trail for this module: two
independent code paths for the same spec, not a hand-typed number.
