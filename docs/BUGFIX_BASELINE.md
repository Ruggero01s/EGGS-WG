# Shared-bugfix baseline · N=300

[Plots/tables](../outputs/bugfix_baseline_n300/Assessment/index.html) · [Scores CSV](../outputs/bugfix_baseline_n300/Assessment/scores.csv)

| Setting | Value |
| --- | --- |
| Cases | Capitanata, Egypt, Algeria, Tunisia; Vietnam excluded |
| New members | 300 per case; same seeds as archived members 1–300 |
| Simulation | January–December 2026; stationary 2020 reference |
| Training | All available 1990–2025; same prepared inputs/masks as archived runs |
| Common fixes | [Shared safeguards](COMMON_BUGFIXES.md); coefficients rebuilt |
| Original adjustments | On, matching archived baseline |
| Experimental corrections | Off; legacy seasonal/residual/VAR estimator retained with corrected lag orientation |
| Assessment | Full year + April–October; historical P5/P50/P95, bias, SD ratio, coverage |
| Storage | Compressed float32 weather; float64 fitting |

Coverage target 90%; SD ratio target 1; bias target 0. No single pass/fail score. N=300 and 35–36 historical years limit tail precision. Tunisia: two cells. This is in-sample ERA5-Land reproduction, not forecast validation.

## Main results

Annual fixed baseline:

| Case | Rain SD ratio | Maximum-T bias, °C | ETo SD ratio |
| --- | ---: | ---: | ---: |
| Capitanata | 0.65 | +3.23 | 0.38 |
| Egypt | 0.55 | +0.65 | 0.28 |
| Algeria | 0.59 | +1.50 | 0.27 |
| Tunisia | 1.04 | −0.69 | 0.30 |

- Bug fixes produce small, mixed distribution changes; no general envelope improvement.
- ETo spread remains too narrow in every case; rainfall spread remains narrow outside Tunisia.
- Temperature bias/spread still case-dependent. Egypt annual minimum-temperature coverage remains 0%.
- All 1,200 generated members checked: finite in-domain weather, nonnegative rain, positive fitted scales, stable VAR, paired seeds. Archived files preserved.

## Reproduce / resume

```bash
uv run --locked python scripts/run_bugfix_baselines.py --n 300 --jobs 2
```

Configs: `configs/bugfix_CASE_n300.yaml`. Logs/status and paired case reports: `outputs/bugfix_baseline_n300/CASE/`. Original experiments preserved.
