"""Run upstream EGGS-WG fitting on configurable local ERA5-Land training years."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
import xarray as xr

from ai4water_eggs.inputs import input_signature, prepare_inputs, training_configuration
import SWG_Build_Functions as build
import SWG_Model as model


def reuse_training(source, root, signature):
    """Reuse matching prepared inputs/fits while keeping scenario outputs separate."""
    source, root = Path(source), Path(root)
    report = json.loads((source / 'Data/input_report.json').read_text())
    if report['signature'] != signature:
        raise ValueError('Reuse Training From does not match the requested inputs and training configuration.')

    def link_or_copy(original, target):
        try:
            os.link(original, target)
        except OSError:
            shutil.copy2(original, target)

    shutil.copytree(source / 'Data', root / 'Data', copy_function=link_or_copy)
    if (source / 'boundary.geojson').is_file():
        shutil.copy2(source / 'boundary.geojson', root / 'boundary.geojson')
    print(f'Reusing prepared inputs and fitted coefficients from {source}', flush=True)


def simulation_interval(params):
    if 'Simulation Start Date' in params or 'Simulation End Date' in params:
        if not all(key in params for key in ('Simulation Start Date', 'Simulation End Date')):
            raise ValueError('Supply both Simulation Start Date and Simulation End Date (exclusive).')
        first = np.datetime64(str(params['Simulation Start Date']), 'D')
        stop = np.datetime64(str(params['Simulation End Date']), 'D')
    else:
        first = np.datetime64(f"{params['Simulation Start Year']}-01-01", 'D')
        stop = np.datetime64(f"{params['Simulation End Year']}-01-01", 'D')
    if np.isnat(first) or np.isnat(stop) or stop <= first:
        raise ValueError('Simulation end must follow simulation start; the end date is exclusive.')
    return first, stop


def run(params, config_path, prepare_only=False):
    params = dict(params)
    params.setdefault('Variables', ['tp', 't2m'])
    if set(params['Variables']) not in ({'tp', 't2m'}, {'tp', 't2m', 'd2m'}):
        raise ValueError('Variables must contain tp and t2m, optionally d2m.')
    params['Variables'] = [name for name in ('tp', 't2m', 'd2m') if name in params['Variables']]
    start, end = params['Training Start Year'], params['Training End Year']
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError('Training years must be integers in increasing order (both inclusive).')
    excluded = params.get('Excluded Training Years', [])
    if any(year < start or year > end for year in excluded):
        raise ValueError('Excluded Training Years must lie within the training period.')
    first_date, stop_date = simulation_interval(params)
    params['Simulation Start Date'] = str(first_date)
    params['Simulation End Date'] = str(stop_date)
    first = int(str(first_date)[:4])
    stop = max(first + 1, int(str(stop_date)[:4]))
    simulations, workers = params['Number of Simulations'], params.get('Workers', 4)
    if (stop <= first or type(simulations) is not int or simulations < 1
            or type(workers) is not int or workers < 1):
        raise ValueError('Set increasing simulation years, at least one simulation, and positive Workers.')
    config_path = Path(config_path).resolve()
    for key in ('ERA5 Directory', 'Data Output Directory'):
        params[key] = str((config_path.parent / params[key]).resolve())
    if params.get('Boundary File'):
        params['Boundary File'] = str((config_path.parent / params['Boundary File']).resolve())
    if params.get('Reuse Training From'):
        params['Reuse Training From'] = str((config_path.parent / params['Reuse Training From']).resolve())
    name = params['Simulation name']
    if not name or Path(name).name != name or name in ('.', '..') or '\\' in name:
        raise ValueError('Simulation name must be a single directory name.')
    if params.get('Reuse Training From'):
        source = Path(params['Reuse Training From'])
        source_config = json.loads((source / 'run_config.json').read_text())
        if training_configuration(source_config) != training_configuration(params):
            raise ValueError('Reuse Training From has a different training configuration.')
        source_report = json.loads((source / 'Data/input_report.json').read_text())
        # Reuse a frozen prepared snapshot. New raw-file variables/downloads
        # must not silently change this ensemble's historical reference.
        inventory = {'signature': source_report['signature']}
        params['Training Snapshot'] = str(source)
    else:
        inventory = input_signature(params)
    root = Path(params['Data Output Directory']) / name
    data = root / 'Data'
    report_path = data / 'input_report.json'
    regenerate = params.get('Regenerate Data', False)
    if params.get('Reuse Training From') and regenerate:
        raise ValueError('Choose either Reuse Training From or Regenerate Data.')
    if params.get('Reuse Training From') and not data.exists():
        root.mkdir(parents=True, exist_ok=True)
        reuse_training(params['Reuse Training From'], root, inventory['signature'])
    if report_path.is_file():
        previous = json.loads(report_path.read_text())
        if previous['signature'] != inventory['signature'] and not regenerate:
            raise ValueError('Training inputs/configuration changed. Use a new Simulation name or set Regenerate Data: true.')
    elif data.exists() and any(data.iterdir()) and not regenerate:
        raise ValueError('This Data directory lacks an input signature. Use a new Simulation name or regenerate it.')
    if regenerate and data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True, exist_ok=True)
    period = f'{start}_{end}'
    temperature_names = [name for name in params['Variables'] if name != 'tp']
    prepared = [data / f'{name}_{period}.nc' for name in temperature_names]
    prepared.append(data / f'daily_tp_{period}.nc')
    if not report_path.is_file() or not all(path.is_file() for path in prepared):
        if params.get('Reuse Training From'):
            raise ValueError('The frozen training snapshot is incomplete; prepare its inputs before reuse.')
        report = prepare_inputs(params, data, inventory)
    else:
        report = json.loads(report_path.read_text())
        print(f"Reusing {report['retained_days']} prepared training days.", flush=True)
    (root / 'run_config.json').write_text(json.dumps(params, indent=2) + '\n')
    if prepare_only:
        print(f'Input preparation complete: {report_path}', flush=True)
        return root
    (root / '.in_progress').write_text('Generation/postprocessing in progress.\n')
    directory = str(data) + '/'
    period_args = {'start_year': start, 'end_year': end}

    def stage(filename, function, *args, **kwargs):
        if not (data / filename).is_file():
            print(f'Building {filename}', flush=True)
            function(*args, **period_args, **kwargs)

    for name in temperature_names:
        stage(f'Detrended_{name}_{period}.nc', build.Variable_Detrender, name, directory,
              reference_year=params.get('Detrend Reference Year'))
    for prefix, function in (
        ('Markov', build.Markov_Calculation), ('Corr', build.Cross_Correlator),
        ('O_Omega', build.O_Omega), ('tp_Params', build.Amounts_Coeff),
        ('Amounts_Corr', build.Amounts_Corr), ('A_Omega', build.A_Omega),
    ):
        kwargs = {'workers': workers, 'seed': params.get('Random Seed', 20261019)} if prefix in ('O_Omega', 'A_Omega') else {}
        stage(f'{prefix}_{period}.nc', function, directory, **kwargs)
    for name in temperature_names:
        filename = f'{name}_PreSim_Coeff_{period}.nc'
        stage(filename, build.Fourier_Generation, directory, name, workers=workers)
        with xr.open_dataset(data / filename) as coefficients:
            for key in ('Fourier_Mean_Wet', 'Fourier_Mean_Dry', 'Fourier_Std_Wet',
                        'Fourier_Std_Dry', 'Residual_Grid'):
                if not np.isfinite(coefficients[key].values).all():
                    raise ValueError(f'{filename}: non-finite {key}; insufficient training support for this fit.')
    stage('_'.join(temperature_names) + '_XY_Matrix.nc', build.XY_Hourly,
          directory, temperature_names, workers=workers)
    simulation_dir = root / 'Simulation'
    for filename in ('scenario_catalog.json', 'scenario_metrics.csv', 'scenario_monthly_metrics.csv'):
        (root / filename).unlink(missing_ok=True)
    if simulation_dir.exists():
        shutil.rmtree(simulation_dir)
    simulation_dir.mkdir()
    for number in range(simulations):
        model.Model_run(str(root) + '/', number, first, stop, temperature_names,
                        **period_args, spin_up_days=params.get('Spin Up Days', 1000),
                        start_date=str(first_date), end_date=str(stop_date),
                        seed=params.get('Random Seed', 20261019) + number)
    if params.get('Variable Adjust', True):
        model.Precipitation_Adjust(str(root) + '/', directory, simulations, **period_args)
        model.Sim_replace(str(root) + '/', simulations)
        model.Temperature_Cycle_Adjust(str(root) + '/', directory, simulations, **period_args)
        model.Sim_replace(str(root) + '/', simulations)
    if params.get('Non-stationary Mode', False):
        model.Nonstationairy_mode(str(root) + '/', directory, simulations, **period_args)
        model.Sim_replace(str(root) + '/', simulations)
    model.metadata_handling(str(root) + '/', simulations)
    for path in sorted(simulation_dir.glob('Simulation_*/*.nc')):
        number = int(path.parent.name.removeprefix('Simulation_'))
        with xr.open_dataset(path) as source:
            output = source.load()
        output.attrs.update(
            training_start=report['training_start'], training_end=report['training_end'],
            training_excluded_years=json.dumps(excluded), training_input_signature=inventory['signature'],
            daily_interval='[00:00 UTC, next-day 00:00 UTC)',
            simulation_start=str(first_date), simulation_end_exclusive=str(stop_date),
            random_seed=params.get('Random Seed', 20261019) + number - 1,
            scenario_number=number,
            generated_by='EGGS-WG with AI4Water local-input adapter')
        output['simulation_Precipitation'].attrs['description'] = 'Total for the labelled UTC day'
        output.to_netcdf(path)
        output.close()
    print(f'Completed: {simulation_dir}', flush=True)
    if params.get('Scenario Metrics', False):
        from ai4water_eggs.metrics import analyze_ensemble
        analyze_ensemble(root, scales=params.get('Drought Scales', [1, 3, 6]))
    (root / '.in_progress').unlink(missing_ok=True)
    return root
