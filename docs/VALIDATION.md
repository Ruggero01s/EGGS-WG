# Historical-envelope comparison

Run from `code_models/EGGS-WG`. [Setup](SETUP.md) · [Limitations](LIMITATIONS.md)

## Current experiment

| Setting | Value |
| --- | --- |
| Config | `configs/capitanata_annual_2026_n1000.yaml` |
| Fit | All available years: 1990–2025; stationary, 2020 temperature reference |
| Generation | N=1,000; January–December 2026; `tp`, `t2m`; 46 cells |
| Comparison | Full year and April–October subset; no holdouts |
| Percentiles | P5, P10, P25, P50, P75, P90, P95 |
| History | 35 complete calendar years; 36 April–October seasons |

2025 lacks its final daily rainfall endpoint. Leap days remain in annual totals,
spells and monthly statistics; daily plots align matching month/day slots.

## Reproduce

```bash
uv run --locked python ensemble.py generate configs/capitanata_annual_2026_n1000.yaml
uv run --locked --group viewer python ensemble.py validate outputs/capitanata_annual_2026_n1000
uv run --locked --group viewer python ensemble.py validate outputs/capitanata_annual_2026_n1000 \
  --window 2026-04-01 2026-11-01 --output outputs/capitanata_annual_2026_n1000/Validation_AprOct
uv run --locked python scripts/assess_envelopes.py --reports \
  outputs/capitanata_annual_2026_n1000/Validation/validation.json \
  outputs/capitanata_annual_2026_n1000/Validation_AprOct/validation.json \
  --output outputs/capitanata_annual_2026_n1000/Assessment
```

Generating again replaces that experiment's weather. Use a new name to retain it.
Validation/assessment commands only read existing weather.

## Outputs

| Under `outputs/capitanata_annual_2026_n1000/` | Contents |
| --- | --- |
| `Assessment/index.html` | Concise findings, percentile tables and embedded plots |
| `Assessment/assessment.pdf`, `Assessment/figures/` | Exportable figures |
| `Assessment/percentiles.csv` | Historical / EGGS percentiles and differences |
| `Assessment/metric_scores.csv` | Means, spread ratios, containment and diagnostic flags |
| `Assessment/monthly_comparison.csv` | Monthly rainfall, temperature, ETo and temperature range |
| `Validation/`, `Validation_AprOct/` | Detailed interactive reports and reference data |

Live viewer: `http://127.0.0.1:8765/assessment` opens the latest completed report.
The previous 100-member/holdout analysis remains in `outputs/Assessment/`.

Matched adjustment diagnostic (optional; reuses the full fit):

```bash
uv run --locked python ensemble.py generate configs/capitanata_annual_diagnostic_raw_n50.yaml
uv run --locked python scripts/diagnose_adjustment.py \
  --adjusted outputs/capitanata_annual_2026_n1000 \
  --unadjusted outputs/capitanata_annual_2026_n1000/Diagnostics/without_adjustment_n50 \
  --output outputs/capitanata_annual_2026_n1000/Diagnostics
```

Append `--diagnostic outputs/capitanata_annual_2026_n1000/Diagnostics/adjustment_summary.json`
to the assessment command to include its N=50 table.

## Interpretation / checks

- Primary check: reproduction of the fitted historical climatology.
- Match calendar window, spatial mask, weights and temperature reference.
- Historical tails use only 35–36 seasons; 1,000 members reduce simulation sampling noise.
- Flags are descriptive; sampling intervals omit parameter uncertainty and interannual dependence.
- Pointwise envelopes are not complete weather trajectories.

```bash
uv run --locked --group viewer python -m unittest discover -s tests -v
uv run --locked --with playwright python tests/check_assessment_browser.py \
  outputs/capitanata_annual_2026_n1000/Assessment --live-url http://127.0.0.1:8765
```
