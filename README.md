# EGGS-WG for AI4Water

ERA5-Land stochastic weather generator: uv setup, configurable training, ensembles and viewer.

| Guide | Contents |
| --- | --- |
| [Setup](docs/SETUP.md) | Install, train/retrain, generate N, select, view through SSH |
| [Validation](docs/VALIDATION.md) | Historical envelopes, percentiles and reproduction commands |
| [Limitations](docs/LIMITATIONS.md) | Measured problems, interpretation and application gaps |
| [Common bug fixes](docs/COMMON_BUGFIXES.md) | Shared fixes for every new candidate; cached fits require rebuilding |
| [Manual](docs/references/EGGS_WG_Operations.pdf) / [paper](docs/references/eggsws_paper.pdf) | Upstream references |

```bash
uv sync --locked --group viewer
uv run --locked python scripts/check_setup.py
uv run --locked --group viewer python viewer.py --root outputs --port 8765
```

Current local run: `outputs/capitanata_annual_2026_n1000`; full year, 1,000 members,
46 cells; fitted on all 1990–2025. **Mismatches:** narrow rainfall/ETo spread and temperature extremes.
[Baseline assessment](outputs/capitanata_annual_2026_n1000/Assessment/index.html): annual and April–October comparisons.
[Capitanata temperature trial](outputs/capitanata_temperature_n1000/Assessment/index.html): 1,000 matched members; temperature-template changes with/without pooled rainfall correction. [Results/method](docs/TEMPERATURE_TRIAL.md).
[Earlier repair trial](outputs/capitanata_fix_trial/Assessment/index.html): 200 matched members. [Commands](docs/VARIABILITY_TRIAL.md).
[Temperature diagnosis](outputs/capitanata_fix_trial/TemperatureDiagnosis/README.md): controlled replay isolates seasonal mean/scale fitting issues.
[Transfer trial](outputs/transfer_n1000/Assessment/index.html): Capitanata, Egypt, Algeria and Tunisia; 1,000 paired members per variant. Maximum-temperature improvements do not generalize; rainfall/ETo results are mixed. [Results/run](docs/TRANSFER_TRIAL.md).
[Latest: common-bugfix baseline](outputs/bugfix_baseline_n300/Assessment/index.html): 300 members per case, matched archived seeds; original adjustments on, experimental corrections off. Main spread limitations remain. [Results/run](docs/BUGFIX_BASELINE.md).

## Layout

```text
ensemble.py / viewer.py           Main entry points
Main.py                          Compatibility wrapper for generate
SWG_*.py                          Fitting and generation
ai4water_eggs/                    Local workflow, metrics, validation, viewer
configs/                          YAML; paths relative to each YAML
scripts/                          Setup check, assessments and diagnostics
viewer/                           Web assets and local Plotly
tests/                            Unit, integration and browser checks
docs/                             Guides and reference PDFs
outputs/                          Local data, fits and runs; Git-ignored
pyproject.toml / uv.lock           Environment
```

Use uv and `configs/`. Inputs come from the existing AI4Water NetCDF archive.
`Main.py CONFIG --prepare-only` remains supported; new commands use `ensemble.py`.

## License and acknowledgment

The source code for this EGGS-WG is licensed under the GNU GENERAL PUBLIC LICENSE. By contributing source code to this project, you are agreeing to license your contributions under and on the terms of the GNU GENERAL PUBLIC LICENSE. Please note that your licence is irrevocable and royalty-free. You or your licensors retain any copyright in your contributions while allowing others to re-use the source code in any way they like as long as they meet the requirements of the licence. All usages of the EGGS-WG code must appropriately acknowledge and cite the EGGS-WG developers. 
