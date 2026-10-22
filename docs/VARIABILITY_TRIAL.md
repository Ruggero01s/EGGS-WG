# Variability repair trial

Updated: 2026-10-22. [Report](../outputs/capitanata_fix_trial/Assessment/index.html) · [Scores](../outputs/capitanata_fix_trial/Assessment/metric_comparison.csv)

Latest: [N=1,000 temperature/rainfall trial](TEMPERATURE_TRIAL.md).

## Design

- Fit: all 1990–2025; stationary, 2020 temperature reference; `tp`, `t2m`.
- Calibration: 50 raw members, seeds 20261021–20261070.
- Evaluation: 200 different members, seeds 20261071–20261270, identical across variants.
- Windows: annual and April–October; same historical references as baseline.

| Variant | Change |
| --- | --- |
| Current EGGS | Original per-member rainfall/temperature adjustments |
| Raw | Both adjustments off |
| Pooled rain + T cycle | Fixed month/cell rainfall mapping and month/hour/cell temperature offsets |
| Refit T · raw | Matched wet/dry residual normalization + forward VAR; no adjustment |
| Refit T + pooled | Both fitting changes + fixed pooled corrections |
| Pooled rain only | Fixed rainfall mapping; temperature adjustment off |

## Annual result

SD ratio = generated/historical variability; target 1.

| Metric | Current | Pooled rain only |
| --- | ---: | ---: |
| Rainfall SD ratio | 0.67 | 0.88 |
| Mean-temperature SD ratio | 0.68 | 0.68 |
| Daily-temperature-range SD ratio | 0.12 | 0.38 |
| ETo SD ratio | 0.35 | 0.52 |
| ETo mean bias, mm | +35.4 | −5.8 |
| Peak-temperature mean bias, °C | +3.1 | +2.4 |

**Partial improvement.** April–October rainfall spread remains 0.75× history; ETo bias becomes −23.3 mm. Temperature variability and extremes remain deficient.

The synthetic VAR test confirms the old lag-covariance orientation error. The combined temperature refit does not resolve the observed envelope errors; with pooled cycle correction, annual ETo bias rises to +67.8 mm. Experimental fitting modes remain opt-in; baseline behavior is unchanged.

## Commands

Run from the checkout. Existing baseline and calibration files are prerequisites. Use fresh names to retain trials.

```bash
uv run --locked python ensemble.py generate configs/capitanata_fix_trial.yaml
uv run --locked python ensemble.py fit-correction \
  outputs/capitanata_annual_2026_n1000/Diagnostics/without_adjustment_n50 --output /tmp/pooled.json
uv run --locked python ensemble.py correct outputs/capitanata_fix_trial/raw_n200 \
  --correction /tmp/pooled.json --output outputs/capitanata_fix_trial/my_rain_trial --no-temperature
```

Temperature-fit experiment: `configs/capitanata_consistent_trial.yaml`. Keys:
`Temperature Residual Conditioning: matched`; `Temperature VAR Fit: forward`.
Fresh frozen-fit reuse detaches incompatible coefficient files before refitting.

`ensemble.py subset SOURCE --members FIRST LAST --output DESTINATION` extracts inclusive member ranges and rebuilds metrics. The consistent-fit run uses members 1–50 for calibration and 51–250 for evaluation.

Validate each variant with `ensemble.py validate RUN`; add `--window 2026-04-01 2026-11-01 --output RUN/Validation_AprOct` for the seasonal window.
Build the comparison with `ensemble.py compare --reports REPORT_JSON... --output outputs/capitanata_fix_trial/Assessment`.

## Evidence / next step

- Three figures: annual spread; percentile intervals; rainfall variance over 1/7/30/90 days and whole years.
- Regional rainfall spread remains below individual-cell spread: spatial dependence still matters.
- Fixed corrections preserve member differences; they do not guarantee correct tails or persistence.
- Heat-bias follow-up: [controlled replay](../outputs/capitanata_fix_trial/TemperatureDiagnosis/README.md) isolates conditioned Fourier intercept/scale issues; no production changes.
- Reproduce: `uv run python scripts/diagnose_temperature.py --raw outputs/capitanata_fix_trial/raw_n200 --adjusted outputs/capitanata_fix_trial/current_n200 --output outputs/capitanata_fix_trial/TemperatureDiagnosis`.
- Next: temperature-template refit and rainfall spatial/persistence calibration.
- Dewpoint, radiation and wind cover all 432 source months. They were audited but not added to this matched trial.
