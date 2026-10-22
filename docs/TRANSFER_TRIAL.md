# Transfer trial · N=1,000 per variant

[Combined report](../outputs/transfer_n1000/Assessment/index.html) · [Scores](../outputs/transfer_n1000/Assessment/scores.csv)

## Results

**The temperature change does not generalize as-is.** Annual maximum-temperature bias (generated minus historical, °C):

| Case | Baseline | Temperature + rain changes |
| --- | ---: | ---: |
| Capitanata | +3.14 | +0.34 |
| Egypt | +0.79 | −1.86 |
| Algeria | +1.43 | −1.72 |
| Tunisia | −0.63 | −2.11 |

- Annual rainfall and ETo P5–P95 coverage moves closer to 90% in all four cases; ETo spread remains too narrow.
- April–October rainfall coverage: unchanged in three cases, worse in Tunisia.
- Egypt rainfall performs better **without** pooled correction: coverage 77% versus 69%; mean bias +2 versus −22 mm/year.
- Minimum-temperature and ETo results improve in several cases; this does not resolve maximum-temperature or seasonal-spread failures.
- Checks: 82 tests passed; offline/mobile report passed; original outputs unchanged.

## Design and limits

| Case | Available cells | Training |
| --- | ---: | --- |
| Capitanata | 46 | Existing 1990–2025 experiment |
| Egypt | 32 | 1990–2025, study polygon |
| Algeria | 25 | 1990–2025, study polygon |
| Tunisia | 2 | 1990–2025, study polygon; repaired May 2002 retained |

- Vietnam excluded by request. Four cases total, including Capitanata.
- Three variants: baseline; temperature-template changes; temperature changes + shared rainfall mapping.
- Paired 1,000 evaluation seeds; 50 separate calibration members per case.
- Full-year 2026 simulation; annual and April–October assessment; stationary 2020 reference.
- Fit all available historical data; compare against each case's ERA5-Land climatology.
- Read coverage **and** bias/spread. Higher coverage alone does not establish a better fit.
- Two-cell Tunisia informs local marginal behaviour, with limited spatial evidence.
- Transfer safeguards: Egypt needed active-cell indexing and dry-month transition handling in all variants. Positive hourly-scale fallback was needed for 404 day/cell combinations in Algeria and 419 in Egypt; it preserves fitted daily mean scales. The original failed fit is retained in `fit_failures/`.
- Egypt's numerical baseline also contains 1,269 nonpositive fitted wet-hour scales; the standard workflow permits these. [Scale audit](../outputs/transfer_n1000/fit_failures/baseline_scale_audit.csv).
- Rain below 1 mm/day is omitted by the generator, but is 19% of historical Egyptian rainfall and 4.5–6.1% in the other cases. Wet-day amount correction cannot restore this component. [Trace-rain audit](../outputs/transfer_n1000/trace_rain.csv).

## Run / resume

```bash
uv run --locked python scripts/run_transfer_trials.py --cases tunisia algeria egypt --jobs 2
uv run --locked python -c "from ai4water_eggs.transfer import build_transfer_report; build_transfer_report('outputs/transfer_n1000')"
```

Study configurations: `configs/transfer_CASE_baseline.yaml`; logs/status: `outputs/transfer_n1000/CASE/`.

New scenario files use float32 + compression; fitting/generation calculations remain float64. Two-member storage check: <0.0001°C temperature and <0.00002 mm precipitation difference. Original Capitanata values remain unchanged, including after lossless recompression. The baseline's cached observational cycle gives bitwise-identical outputs in the two-member check.

[Plan](../outputs/transfer_n1000/plan.json) · [Storage checks](../outputs/transfer_n1000/storage_checks.json)
