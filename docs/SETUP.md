# Setup and workflow

Working directory: `code_models/EGGS-WG`. Python 3.12 + **uv**; no Anaconda.

## Install / check

```bash
uv sync --locked --group viewer
uv run --locked python scripts/check_setup.py
uv run --locked --group viewer python -m unittest discover -s tests -v
```

Independent `.venv` and lockfile. Optional notebook tools: `--group notebooks`.

## Configurations

| YAML in `configs/` | Domain | Output root from checkout |
| --- | --- | --- |
| `capitanata_annual_2026_n1000.yaml` | Full-year 2026; N=1,000; all historical years | `outputs/` |
| `capitanata_test.yaml` | Boundary-masked; April–October 2026; currently N=100 | `outputs/` |
| `italy_pilot.yaml` | Single cell, annual | `../../analysis_results/eggs_wg/` |
| `italy.yaml`, `egypt.yaml`, `algeria.yaml`, `tunisia.yaml` | Legacy bounding-box examples | `../../analysis_results/eggs_wg/` |
| `transfer_CASE_baseline.yaml` | Egypt/Algeria/Tunisia study polygons; N=1,000 | `outputs/transfer_n1000/CASE/` |

- YAML paths resolve relative to the **YAML file**.
- Capitanata: training 1990–2025; reference 2020; stationary; 46 cells. Annual: 365 days; April–October: 214 days.
- Config name: `capitanata_ensemble_apr_oct_2026`; existing run `capitanata_test_n100` used a name override.
- [Transfer trial](TRANSFER_TRIAL.md): study-area masks; Tunisia's repaired May 2002 retained.

## Train / generate

```bash
# Prepare inputs only; honors Regenerate Data.
uv run --locked python ensemble.py generate configs/capitanata_test.yaml --prepare-only

# Fit and generate; new name preserves the existing experiment.
uv run --locked python ensemble.py generate configs/capitanata_test.yaml \
  --n 30 --name capitanata_apr_oct_2026_n30 --seed 20261019
```

| Mode | YAML settings |
| --- | --- |
| Full retrain | New `Simulation name`; `Regenerate Data: true` |
| Reuse compatible cached fit | Same training/name; `Regenerate Data: false` |
| New ensemble from frozen fit | New name; `Regenerate Data: false`; `Reuse Training From: ../outputs/capitanata_test_n100` |

- Same-name generation replaces weather/metrics. `--prepare-only` can replace prepared data while leaving old simulations; use a new name when changing training.
- Frozen reuse requires matching training settings; cannot combine with regeneration; ignores new downloads.
- [Common bug fixes](COMMON_BUGFIXES.md) apply to every candidate. Older unversioned fits require a new experiment with `Reuse Training From` (prepared observations reused; coefficients rebuilt), or full regeneration.
- Training years: inclusive. Simulation end: exclusive (`2026-11-01` means through October 31).
- Keep `Non-stationary Mode: false` for this reference-climate workflow.
- Member i seed = base + i − 1. `Workers` controls fitting; `Simulation Workers` controls generation (default 1).

### NetCDF storage

New runs default to compressed float32 weather files (DEFLATE level 1 + shuffle). Fitting/generation calculations remain float64; storage settings propagate through postprocessing.

```yaml
Weather Output Precision: float32
NetCDF Compression: true
```

One annual Capitanata member: **13.10 → 1.51 MiB** (88.5% smaller); roughly 1.5 GiB per 1,000 members, weather only. Measured storage rounding: <0.00002°C and <0.000004 mm.

For exact float64 values, set `Weather Output Precision: float64` and keep compression enabled: **3.58 MiB/member** in the same check (72.7% smaller). Actual sizes depend on domain/weather. Changing defaults alone does not rewrite existing files; the latest transfer runs already use compact storage.

Compress existing weather files, preserving their precision and checking every raw value before replacement:

```bash
uv run --locked python scripts/compress_weather.py outputs --workers 4 --apply
```

Already compressed files are skipped; shared hardlinks are preserved. Logs: `outputs/compression-*.jsonl`; totals: matching `.json`. Omitting `--apply` only inventories files. Run while generation is idle.

## View / select

```bash
# On VM/server:
uv run --locked --group viewer python viewer.py --root outputs --port 8765

# On local computer, when using SSH:
ssh -N -L 8765:127.0.0.1:8765 USER@VM
```

Open `http://127.0.0.1:8765`. VS Code port forwarding also works.
For older outputs, use `--root ../../analysis_results/eggs_wg`.

```bash
uv run --locked python ensemble.py analyze outputs/capitanata_test_n100 --scales 1 3 6 7
uv run --locked python ensemble.py select outputs/capitanata_test_n100 \
  --metric total_precipitation_mm --percentile 10 --output /tmp/rainfall_p10.json
```

- Other metrics: `spei_period`, `spi_period`, temperature, ETo, P−ETo, spells and extremes; definitions in catalog.
- Selection: actual member nearest the ensemble quantile; full domain/period. Plot controls do not change ranking.
- **Ensemble P10 ≠ historical P10.** See [limitations](LIMITATIONS.md).

## Offline export

```bash
uv run --locked --group viewer python viewer.py --root outputs \
  --export /tmp/ensemble.html --experiment capitanata_test_n100
uv run --locked --group viewer python viewer.py --root outputs \
  --export /tmp/scenario_1.html --run capitanata_test_n100/Simulation/Simulation_1
```

Open HTML directly. Local Plotly; ensemble decoding needs `DecompressionStream`.
Re-export and [revalidate](VALIDATION.md) after weather/data changes.

## Data contract

| Item | Requirement / output |
| --- | --- |
| Variables | `tp`, `t2m`; optional aligned `d2m` every month; no wind/radiation generation |
| Input | Hourly `time, latitude, longitude`; temperature K/degC; precipitation m/mm |
| Rain convention | Daily cumulative ERA5-Land; day D needs D+1 at 00 UTC, including next month/year |
| Missing data | Drop/report invalid paired days; no zero-filling. Snapshot: 13,148 days through 2025-12-30 |
| Output units | `simulation_t2m` / `simulation_d2m`: K; `simulation_Precipitation`: mm/day; RH: % |
| Aggregation | Fixed available-cell mask; cos(latitude) weights; viewer temperature: °C |
| Drought / PET | ≥30 historical values/month/scale; Hargreaves-Samani ETo; methods in catalog |

## Experiment files

```text
run_config.json                 Resolved settings
Data/input_report.json          Inventory, exclusions, signature
Data/*.nc                       History + fitted coefficients
Simulation/Simulation_N/*.nc    Generated member
scenario_catalog.json           Definitions, metrics, seeds, reference fits
scenario_*metrics.csv           Seasonal/monthly tables
Validation/                     Historical report and scores
explorer.html                   Optional exported snapshot
```

`ensemble.py validate EXPERIMENT`: ensemble/history comparison; see [VALIDATION.md](VALIDATION.md).

Layout: [README](../README.md). `Main.py CONFIG [--prepare-only]` remains a compatibility wrapper.
The legacy downloader, dependency checker and Conda configuration have been removed.
