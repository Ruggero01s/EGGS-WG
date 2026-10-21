# Problems and limitations

Updated: 2026-10-21. Scope: weather inputs for irrigation demand and reservoir hydrology.

## Current evidence

- **N=1,000**, January–December 2026; 46 Capitanata cells; `tp`, `t2m`.
- Fit: **all 1990–2025**; stationary temperature at 2020 reference; no new holdouts.
- Reference: 35 complete calendar years (1990–2024); 36 April–October seasons.
- [Plots and tables](../outputs/capitanata_annual_2026_n1000/Assessment/index.html) · [Percentiles](../outputs/capitanata_annual_2026_n1000/Assessment/percentiles.csv) · [Run config](../outputs/capitanata_annual_2026_n1000/run_config.json)

## Measured problems

Annual values. SD ratio = generated/historical variability; containment = historical values inside generated P5–P95 (nominal 90%).

| ID | Metric | History mean | EGGS mean | SD ratio | Containment |
| --- | --- | ---: | ---: | ---: | ---: |
| D01 | Rainfall, mm | 575.61 | 547.79 | 0.63 | 74.3% |
| D02 | Mean temperature, °C | 16.59 | 16.64 | 0.73 | 74.3% |
| D03 | Peak hourly cell temperature, °C | 41.78 | 44.92 | 1.22 | 42.9% |
| D04 | Reference ETo, mm | 1110.08 | 1146.15 | 0.37 | 31.4% |
| D05 | Rainfall − ETo, mm | -534.47 | -598.36 | 0.55 | 60.0% |
| D06 | Longest regional dry spell, days | 28.60 | 23.16 | 0.75 | 85.7% |
| D07 | Regional hot days ≥30 °C | 70.94 | 76.37 | 0.65 | 62.9% |
| D10 | Mean daily temperature range, °C | 9.72 | 10.02 | 0.11 | 11.4% |
| D11 | Lowest hourly cell temperature, °C | -6.50 | -4.10 | 0.42 | 60.0% |

- **D01:** rainfall P95: **652 vs 724 mm**. Mean-bias 95% interval includes zero; spread mismatch is clearer than mean bias.
- **D08:** monthly rainfall cell-pair correlation: **0.61–0.82 generated vs 0.84–0.94 historical**. Regional aggregation is affected.
- **D09:** annual SPI mean/SD: **-0.27/0.67 vs -0.00/1.01**; SPEI: **-0.52/0.61 vs +0.01/0.99**. Generated drought-index distributions are shifted/compressed.
- **April–October:** rainfall SD ratio **0.70**; peak-temperature bias **+3.1°C**; ETo SD ratio **0.38**. Same main failures.

## Adjustment diagnosis · paired N=50

Same model, dates and seeds; only `Variable Adjust` changes.

| Annual SD | History | Adjustment on | Off |
| --- | ---: | ---: | ---: |
| Rainfall, mm | 96.37 | 61.69 | 90.73 |
| Mean daily temperature range, °C | 0.293 | 0.033 | 0.105 |
| Reference ETo, mm | 30.83 | 12.57 | 17.25 |

Adjustment contributes to compressed variability. Disabling it raises mean rainfall to **622 vs 576 mm historical**; temperature-range variability remains deficient. No fix adopted.
[Diagnostic CSV](../outputs/capitanata_annual_2026_n1000/Diagnostics/adjustment_comparison.csv).

## Interpretation / application gaps

| ID | Limit / next check |
| --- | --- |
| V01 | In-sample climatology reproduction; no forecast-skill claim. Previous holdouts remain [archived](../outputs/Assessment/index.html). |
| V02 | Only 35–36 historical seasons; no fitted-parameter uncertainty. N=1,000 reduces simulation sampling noise, not model bias. |
| V03–04 | Independent station/inflow evidence, district aggregates, compound events and persistence still need validation. |
| A01 | Irrigation boundary may omit reservoir catchments; confirm coherent forcing over contributing areas. |
| A02 | Weather/P−ETo are not inflow, reservoir storage or crop demand; downstream models need separate validation. |
| A03 | January start adds winter forcing; initial soil/storage states and any additional warmup remain necessary. |
| A04 | One-metric selection can miss compound stress; select complete members for connected downstream models. |
| A05 | 0.1° cell-centre mask: 46/50 centres available; no polygon-intersection weighting or sub-grid representation. |
| C01–02 | Temperature is detrended to 2020, not a 2026 forecast; rainfall is not detrended. Non-stationary trends are clamped to training dates. |
| C03 | Hargreaves-Samani ETo only. Downloading wind/radiation/dewpoint does not add coherent generation or Penman–Monteith ETo. |
| C04 | Scenario-selection percentiles rank EGGS members; SPI/SPEI use historical fits. They do not make biased ensembles historically calibrated. |
| C05 | SPI/SPEI need ≥30 complete historical values per month/scale. Without prior weather, SPI-12 first appears in December. Missing indices remain null. |
| C06 | Pointwise bands combine members; they are not coherent trajectories or simultaneous coverage probabilities. |

## Operational limits

| ID | Limit / action |
| --- | --- |
| I01 | Rain accumulation needs next-day 00 UTC; missing endpoint excludes 2025-12-31. Leap days remain in historical totals/spells. |
| I02 | One cell/date needed wet-climatology interpolation; observations unchanged. Unsupported conditions fail. |
| I03 | ecCodes/pyproj conflict: retain subprocess reprojection. |
| I04 | This run: **34 min**, **13.1 MiB/member**, **12.8 GiB weather + 1.1 GiB fit**. Serial generation; `Workers` affects fitting. |
| I05 | Prefer live viewer/selected exports at N=1,000. Offline ensemble HTML grows with N and requires `DecompressionStream`; assessment HTML is self-contained. |
| I06 | Regeneration replaces weather; revalidate/re-export. Reused fits ignore new downloads. |
| I07 | Other domains lack equivalent validation; Tunisia excludes 2002 because May is missing. |

Fixed and regression-checked: hourly residual indexing; negative-rain clipping; lag gaps; missing rainfall endpoints; local Plotly; idle viewer sockets; annual leap-day handling.

Next: separate rainfall/temperature adjustment controls; diagnose raw persistence/spatial dependence and temperature tails; agree downstream tolerances.

[Reproduction commands](VALIDATION.md) · [Setup](SETUP.md)
