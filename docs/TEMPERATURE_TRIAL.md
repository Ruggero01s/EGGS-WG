# Temperature and rainfall trial · N=1,000

[Plots/tables](../outputs/capitanata_temperature_n1000/Assessment/index.html) · [Coverage CSV](../outputs/capitanata_temperature_n1000/Assessment/coverage.csv) · [All metrics, including SPI/SPEI](../outputs/capitanata_temperature_n1000/Assessment/all_metric_scores.csv)

## Design

- Full year 2026; 46 cells; stationary 2020 reference; fit all available 1990–2025.
- Same 1,000 evaluation seeds as baseline; 50 different seeds calibrate rainfall.
- Temperature: calendar-weighted mean intercept; seasonal scales from hourly SD; original adjustments off.
- Rainfall: one fixed wet-day amount mapping per month/cell, shared across members. Wet/dry days unchanged.
- Historical inputs, rainfall model and temperature residual dynamics retained.

## Annual coverage

Historical values inside generated P5–P95; nominal 90%. In-sample climatology reproduction.

| Metric | Baseline | Temperature fit | + pooled rain |
| --- | ---: | ---: | ---: |
| Maximum temperature | 43% | 94% | 94% |
| Minimum temperature | 60% | 54% | 54% |
| Total rainfall | 74% | 71% | 86% |
| ETo | 31% | 49% | 49% |

- Mean peak bias: **+3.14 → +0.34°C**.
- Combined rainfall mean bias: **−14 mm**; SD ratio **0.87**.
- ETo mean bias: **−16 mm**; SD ratio **0.52**. Variability remains too narrow.
- Minimum-temperature coverage worsens despite a smaller mean bias; mean-temperature variability remains narrow.
- April–October: rainfall coverage **78 → 78%**; SPEI coverage **67 → 61%**. ETo bias switches **+33 → −36 mm**. Annual gains do not resolve the growing-season envelope.
- Coverage alone is insufficient: inspect bias, spread and seasonal results in the report.

## Reproduce

Generation config: [capitanata_temperature_n1050.yaml](../configs/capitanata_temperature_n1050.yaml). Generates 1,000 evaluation + 50 calibration members.

`Temperature Seasonal Fit: calendar-hourly` is opt-in. Keep residual conditioning and VAR fit `legacy` to isolate the template changes. `Simulation Workers: 4` controls generation concurrency; default remains 1.

Use `ensemble.py subset` for members 1–1000 and 1001–1050, then `fit-correction --no-temperature` and `correct --no-temperature`. Validate annual and April–October windows; compare against the baseline. [CLI examples](VARIABILITY_TRIAL.md).

[Run checks](../outputs/capitanata_temperature_n1000/run_checks.json) · [Fit/replay checks](../outputs/capitanata_temperature_n1000/smoke_checks.json)
