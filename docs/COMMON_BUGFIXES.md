# Common bug fixes

Apply to every new candidate, including baseline. Historical trial outputs remain archived; they are not comparisons of this updated baseline.

| Bug | Effect | Shared fix |
| --- | --- | --- |
| Dry-month transition counts erased when one state absent | Undefined rainfall probabilities; arid fits fail | Keep valid counts; unsupported state uses observed monthly wet fraction |
| Correlation cells selected by finite values | Constant/dry series break matrix dimensions or cell alignment | Fixed active-cell mask; undefined off-diagonal correlation = 0 |
| Legacy temperature lag covariance transposed | Wrong direction of lagged cross-cell/variable dependence | Current-state rows, previous-state columns, matching simulation |
| Additive temperature conditioning produces nonpositive SD | Invalid hourly temperature scales | Both seasonal fits use proportional fallback, preserving daily mean scale |
| Old coefficients reused after fixes | New run silently retains old bugs | Versioned fit cache; reject stale fits in existing experiments |

Fallback conventions are shared safeguards, not evidence of optimal climatology reproduction. Unsupported positive-scale inputs fail; no arbitrary clipping.

Optional: calendar-hourly seasonal fit, matched residual conditioning, forward least-squares VAR, adjustment switches, pooled rainfall mapping. These remain scientific candidates.

## Existing fits

Use a new simulation name with `Reuse Training From` pointing to the old experiment and `Regenerate Data: false`. Prepared observations reused; stale fitted coefficients rebuilt. Alternative: full regeneration. Existing weather/results are not rewritten by these fixes.

[N=300 baseline trial](BUGFIX_BASELINE.md): common fixes only, paired against archived members.
