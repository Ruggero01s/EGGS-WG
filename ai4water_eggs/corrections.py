"""Frozen, pooled postprocessing for packaged stationary tp/t2m ensembles.

Fit once on calibration seeds; apply to disjoint seeds without recomputing any
member-specific statistics. Scenario metrics must be rebuilt after applying.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import xarray as xr

from ai4water_eggs.metrics import scenario_files

SCHEMA = 'eggs-pooled-corrections-v1'


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _context(root):
    root = Path(root).resolve()
    config = _read_json(root / 'run_config.json')
    if set(config.get('Variables', [])) != {'tp', 't2m'}:
        raise ValueError('Pooled corrections currently require tp and t2m only.')
    if config.get('Non-stationary Mode', False):
        raise ValueError('Pooled corrections currently require stationary simulations.')
    period = f"{config['Training Start Year']}_{config['Training End Year']}"
    report = _read_json(root / 'Data/input_report.json')
    with xr.open_dataset(root / 'Data' / f'A_Omega_{period}.nc') as data:
        latitude, longitude = data.latitude.values.copy(), data.longitude.values.copy()
        mask = np.isfinite(data.Station_Mask.values)
    if not mask.any() or not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError('The fitted domain has no active cells.')
    names = [f'A_Omega_{period}.nc', f'O_Omega_{period}.nc',
             f't2m_PreSim_Coeff_{period}.nc', 't2m_XY_Matrix.nc']
    identity = {'training_input_signature': report['signature'],
                'training_start': report['training_start'], 'training_end': report['training_end'],
                'training_years': [config['Training Start Year'], config['Training End Year']],
                'excluded_years': config.get('Excluded Training Years', []),
                'detrend_reference_year': config.get('Detrend Reference Year'),
                'stationary': True,
                'temperature_residual_conditioning': config.get('Temperature Residual Conditioning', 'legacy'),
                'temperature_var_fit': config.get('Temperature VAR Fit', 'legacy'),
                'fitted_sha256': {name: _digest(root / 'Data' / name) for name in names},
                'latitude': latitude.tolist(), 'longitude': longitude.tolist(), 'mask': mask.tolist()}
    if config.get('Temperature Seasonal Fit', 'legacy') != 'legacy':
        identity['temperature_seasonal_fit'] = config['Temperature Seasonal Fit']
    return root, config, period, identity, mask


def _raw_config(config):
    if config.get('Variable Adjust', True) or config.get('Pooled Corrections'):
        raise ValueError('Calibration and evaluation must use unadjusted raw weather.')


def _check_dataset(data, identity, mask):
    if set(data.data_vars) != {'simulation_Precipitation', 'simulation_t2m'}:
        raise ValueError('Expected packaged precipitation and temperature only.')
    if (tuple(data.simulation_Precipitation.dims) != ('simulation_day', 'latitude', 'longitude')
            or tuple(data.simulation_t2m.dims) != ('simulation_time', 'latitude', 'longitude')):
        raise ValueError('Packaged weather dimensions do not match the correction schema.')
    if (not np.array_equal(data.latitude.values, identity['latitude'])
            or not np.array_equal(data.longitude.values, identity['longitude'])):
        raise ValueError('Scenario and fitted spatial grids differ.')
    if data.attrs.get('training_input_signature') != identity['training_input_signature']:
        raise ValueError('Scenario training signature differs from its fitted reference.')
    if ('pooled_correction_sha256' in data.attrs or 'precipitation_adjustment' in data.attrs
            or 'temperature_cycle_adjustment' in data.attrs):
        raise ValueError('Scenario has already been adjusted.')
    days = data.simulation_day.values.astype('datetime64[D]')
    if not len(days) or np.any(np.diff(days) != np.timedelta64(1, 'D')):
        raise ValueError('Scenario dates must be consecutive UTC days.')
    expected = (days[:, None].astype('datetime64[h]') + np.arange(24) * np.timedelta64(1, 'h')).ravel()
    if not np.array_equal(data.simulation_time.values, expected):
        raise ValueError('Scenario hours must contain each labelled UTC day exactly once.')
    if 'random_seed' not in data.attrs:
        raise ValueError('Scenario seed metadata is required.')
    return days, int(data.attrs['random_seed'])


def _inventory(root, identity, mask):
    if (root / '.in_progress').exists():
        raise ValueError('Wait for raw generation to finish before calibrating or applying corrections.')
    scenarios = scenario_files(root)
    if not scenarios:
        raise ValueError('No packaged scenarios found.')
    expected = _read_json(root / 'run_config.json').get('Number of Simulations')
    if expected is not None and len(scenarios) != expected:
        raise ValueError('Packaged scenario count differs from the configured ensemble size.')
    seeds, calendar, sources = [], None, []
    for number, directory, paths in scenarios:
        member_days, seed = [], None
        for path in paths:
            with xr.open_dataset(path) as data:
                days, part_seed = _check_dataset(data, identity, mask)
            if seed is not None and seed != part_seed:
                raise ValueError('A scenario contains inconsistent random seeds.')
            seed = part_seed
            member_days.append(days)
        dates = np.concatenate(member_days)
        if np.any(np.diff(dates) != np.timedelta64(1, 'D')):
            raise ValueError('Scenario files must form consecutive, non-overlapping days.')
        if calendar is not None and not np.array_equal(calendar, dates):
            raise ValueError('Every scenario must cover the same dates.')
        calendar = dates
        seeds.append(seed)
        sources.append({'number': number, 'seed': seed,
                        'files': [path.relative_to(root).as_posix() for path in paths]})
    if len(seeds) != len(set(seeds)):
        raise ValueError('Scenario seeds must be unique.')
    return scenarios, seeds, calendar, sources


def fit_rain_mapping(calibration, historical, quantiles=101):
    """Monotone wet-day quantile map with a 1-mm anchor and unbounded tail."""
    calibration, historical = np.asarray(calibration), np.asarray(historical)
    calibration = calibration[np.isfinite(calibration) & (calibration >= 1)]
    historical = historical[np.isfinite(historical) & (historical >= 1)]
    if not len(calibration) or not len(historical):
        raise ValueError('Wet-day samples are required to fit a rainfall mapping.')
    if type(quantiles) is not int or quantiles < 5:
        raise ValueError('Use at least five quantile knots.')
    probabilities = np.linspace(.01, .99, quantiles - 1)
    x, y = np.quantile(calibration, probabilities), np.quantile(historical, probabilities)
    # Collapse ties without adding arbitrary jitter; keep the physical anchor.
    knots = np.unique(x[x > 1])
    mapped = np.array([y[x == value].mean() for value in knots])
    x, y = np.r_[1., knots], np.r_[1., mapped]
    if len(x) < 2:
        if np.ptp(historical) > 1e-12 or not np.allclose(historical, 1):
            raise ValueError('Constant 1-mm calibration rain cannot support a varying wet-day mapping.')
        x, y = np.array([1., 2.]), np.array([1., 2.])
    span = max(1, int(np.ceil(.1 * (len(x) - 1))))
    slope = (y[-1] - y[-1-span]) / (x[-1] - x[-1-span])
    if slope <= 0:
        slope = max(1e-8, (y[-1] - 1) / (x[-1] - 1))
    return {'x_mm': x.tolist(), 'y_mm': y.tolist(), 'upper_slope': float(slope),
            'calibration_wet_days': len(calibration), 'historical_wet_days': len(historical)}


def map_rain(values, mapping):
    """Apply one frozen monotone mapping; retain zeros and wet-day occurrence."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 1) & (values != 0)):
        raise ValueError('Raw rainfall must be zero or at least 1 mm, with no missing active cells.')
    x, y = np.asarray(mapping['x_mm']), np.asarray(mapping['y_mm'])
    slope = float(mapping['upper_slope'])
    if (len(x) < 2 or len(x) != len(y) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or x[0] != 1 or y[0] != 1
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0)
            or not np.isfinite(slope) or slope <= 0):
        raise ValueError('Invalid monotone rainfall mapping.')
    result = values.copy()
    wet = values >= 1
    result[wet] = np.interp(values[wet], x, y)
    tail = values > x[-1]
    result[tail] = y[-1] + slope * (values[tail] - x[-1])
    return result


def _pool_months(month, available):
    season = next(group for group in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)) if month in group)
    return [[month], [value for value in season if value in available], available]


def fit_corrections(calibration, output, historical=None, *, rain=True, temperature=True,
                    minimum_wet=100, quantiles=101):
    """Write a frozen JSON model. Historical defaults to calibration's Data."""
    if not rain and not temperature:
        raise ValueError('Select at least one correction component.')
    if type(minimum_wet) is not int or minimum_wet < 2:
        raise ValueError('minimum_wet must be an integer of at least two.')
    root, config, period, identity, mask = _context(calibration)
    _raw_config(config)
    reference, _, ref_period, ref_identity, _ = _context(historical or root)
    if identity != ref_identity:
        raise ValueError('Calibration and historical grids, inputs or fitted models differ.')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Correction model already exists: {output}')
    scenarios, seeds, dates, sources = _inventory(root, identity, mask)
    months = sorted(set((dates.astype('datetime64[M]').astype(int) % 12 + 1).tolist()))
    cells = int(mask.sum())
    rain_samples = {month: [] for month in months}
    sums = {month: np.zeros((24, cells)) for month in months}
    counts = {month: 0 for month in months}
    for _, _, paths in scenarios:
        for path in paths:
            with xr.open_dataset(path) as data:
                days = data.simulation_day.values.astype('datetime64[D]')
                m = days.astype('datetime64[M]').astype(int) % 12 + 1
                if rain:
                    full_rain = data.simulation_Precipitation.values
                    if np.isfinite(full_rain[:, ~mask]).any():
                        raise ValueError('Calibration rainfall extends outside its fitted domain.')
                    raw_rain = full_rain[:, mask]
                    if (not np.isfinite(raw_rain).all()
                            or np.any((raw_rain < 1) & (raw_rain != 0))):
                        raise ValueError('Calibration rainfall must be zero or >=1 mm inside its domain.')
                    for month in months:
                        rain_samples[month].append(raw_rain[m == month])
                if temperature:
                    hourly = data.simulation_t2m.values.reshape(len(days), 24, *mask.shape)[:, :, mask]
                    if not np.isfinite(hourly).all():
                        raise ValueError('Missing calibration temperatures inside the domain.')
                    for month in months:
                        sums[month] += hourly[m == month].sum(axis=0)
                        counts[month] += int(np.sum(m == month))
    model = {'schema': SCHEMA, 'implementation_sha256': _digest(__file__), 'identity': identity, 'calibration': {'path': str(root),
             'members': len(scenarios), 'seeds': seeds, 'start': str(dates[0]),
             'last_day': str(dates[-1]), 'sources': sources,
             'weather_sha256': {path.relative_to(root).as_posix(): _digest(path)
                                for _, _, paths in scenarios for path in paths}},
             'historical': {'path': str(reference), 'period': ref_period},
             'months': months, 'rain': None, 'temperature': None,
             'methods': {'rain': 'Fixed per-cell wet-day quantile mapping pooled across calibration members. Month, then season, then represented-month fallback. 1-mm anchor; piecewise linear monotone interior; positive linear extrapolation above P99; zeros preserved.',
                         'temperature': 'Fixed per-month/hour/cell historical-minus-calibration diurnal anomaly; each 24-hour correction vector has zero mean. Daily means and member-to-member hourly differences are preserved; nonlinear extrema variability is not guaranteed.',
                         'application': 'No member-specific adaptation. Evaluation seeds must be disjoint from calibration. Recompute all derived metrics after application.'}}
    historical_hashes = {}
    if rain:
        path = reference / 'Data' / f'daily_tp_{ref_period}.nc'
        historical_hashes[path.name] = _digest(path)
        with xr.open_dataset(path) as data:
            if (not np.array_equal(data.latitude.values, identity['latitude'])
                    or not np.array_equal(data.longitude.values, identity['longitude'])):
                raise ValueError('Historical rainfall grid differs from the fitted domain.')
            historical_rain = data.tp.values[:, mask]
            historical_month = data.time.values.astype('datetime64[M]').astype(int) % 12 + 1
        if not np.isfinite(historical_rain).all() or np.any(historical_rain < 0):
            raise ValueError('Historical rainfall contains missing or negative active-cell values.')
        rain_samples = {month: np.concatenate(rain_samples[month]) for month in months}
        mappings = {}
        for month in months:
            mappings[str(month)] = []
            for cell in range(cells):
                for pool in _pool_months(month, months):
                    simulated = np.concatenate([rain_samples[value][:, cell] for value in pool])
                    observed = historical_rain[np.isin(historical_month, pool), cell]
                    n, nh = np.sum(simulated >= 1), np.sum(observed >= 1)
                    if n >= minimum_wet and nh >= minimum_wet:
                        break
                if n < minimum_wet or nh < minimum_wet:
                    raise ValueError(f'Insufficient wet-day support for month {month}, cell {cell}: {n}/{nh}.')
                mapping = fit_rain_mapping(simulated, observed, quantiles)
                mapping['pooled_months'] = pool
                mappings[str(month)].append(mapping)
        model['rain'] = {'minimum_wet': minimum_wet, 'quantile_knots': quantiles,
                         'mappings': mappings, 'wet_threshold_mm': 1.}
    if temperature:
        path = reference / 'Data' / f'Detrended_t2m_{ref_period}.nc'
        historical_hashes[path.name] = _digest(path)
        with xr.open_dataset(path) as data:
            if (not np.array_equal(data.latitude.values, identity['latitude'])
                    or not np.array_equal(data.longitude.values, identity['longitude'])
                    or data.sizes.get('step') != 24):
                raise ValueError('Historical temperature grid/hour dimension differs from the fitted domain.')
            historical_month = data.time.values.astype('datetime64[M]').astype(int) % 12 + 1
            corrections, support = {}, {}
            for month in months:
                selection = np.flatnonzero(historical_month == month)
                if not len(selection) or not counts[month]:
                    raise ValueError(f'Missing diurnal-cycle support in month {month}.')
                historical_hourly = data.t2m.isel(time=selection).values[:, :, mask]
                if not np.isfinite(historical_hourly).all():
                    raise ValueError('Missing historical temperatures inside the domain.')
                curve = historical_hourly.mean(axis=0) - sums[month] / counts[month]
                curve -= curve.mean(axis=0, keepdims=True)
                # Set the last hour from the others to minimize floating-point drift.
                curve[-1] = -curve[:-1].sum(axis=0)
                corrections[str(month)] = curve.tolist()
                support[str(month)] = {'historical_days': len(selection), 'calibration_member_days': counts[month]}
            reference_date = data.attrs.get('detrend_reference_date')
        model['temperature'] = {'offsets_K': corrections, 'support': support,
                                'detrend_reference_date': reference_date}
    model['historical']['sha256'] = historical_hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, model)
    return output


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def apply_corrections(source, output, correction_path, *, rain=True, temperature=True):
    """Clone raw packaged weather, apply frozen corrections; leave metrics absent."""
    root, config, _, identity, mask = _context(source)
    _raw_config(config)
    correction_path = Path(correction_path).resolve()
    model = _read_json(correction_path)
    if model.get('schema') != SCHEMA or model.get('identity') != identity:
        raise ValueError('Correction schema, training inputs, fitted model or spatial domain differ.')
    if not rain and not temperature:
        raise ValueError('Select at least one correction component.')
    if (rain and model.get('rain') is None) or (temperature and model.get('temperature') is None):
        raise ValueError('Requested correction component was not fitted.')
    scenarios, seeds, dates, sources = _inventory(root, identity, mask)
    if set(seeds) & set(model['calibration']['seeds']):
        raise ValueError('Evaluation seeds overlap correction calibration seeds.')
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    if not set(months).issubset(model['months']):
        raise ValueError('Evaluation months lack correction calibration support.')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Correction output already exists: {output}')
    output.mkdir(parents=True)
    marker = output / '.in_progress'
    marker.write_text('Frozen pooled correction in progress. Recompute metrics before publishing.\n')
    shutil.copytree(root / 'Data', output / 'Data', copy_function=_link_or_copy)
    if (root / 'boundary.geojson').is_file():
        shutil.copy2(root / 'boundary.geojson', output / 'boundary.geojson')
    checksum = _digest(correction_path)
    shutil.copy2(correction_path, output / 'pooled_correction.json')
    config.update({'Simulation name': output.name, 'Trial Label': output.name, 'Data Output Directory': str(output.parent),
                   'Variable Adjust': False, 'Scenario Metrics': False,
                   'Pooled Corrections': {'schema': SCHEMA, 'source': str(root),
                                          'sha256': checksum, 'rain': rain, 'temperature': temperature}})
    _write_json(output / 'run_config.json', config)
    for _, directory, paths in scenarios:
        destination = output / 'Simulation' / directory.name
        destination.mkdir(parents=True)
        for path in paths:
            with xr.open_dataset(path) as data:
                result = data.load()
            days = result.simulation_day.values.astype('datetime64[D]')
            month_numbers = days.astype('datetime64[M]').astype(int) % 12 + 1
            if rain:
                rainfall = result.simulation_Precipitation.values
                if np.isfinite(rainfall[:, ~mask]).any():
                    raise ValueError('Evaluation rainfall extends outside its fitted domain.')
                active = rainfall[:, mask].copy()
                for month in np.unique(month_numbers):
                    selection = month_numbers == month
                    for cell, mapping in enumerate(model['rain']['mappings'][str(month)]):
                        active[selection, cell] = map_rain(active[selection, cell], mapping)
                rainfall[:, mask] = active
                result.attrs['precipitation_adjustment'] = 'Frozen pooled monotone wet-day mapping'
            if temperature:
                temperatures = result.simulation_t2m.values.reshape(len(days), 24, *mask.shape)
                active = temperatures[:, :, mask].copy()
                if not np.isfinite(active).all():
                    raise ValueError('Missing evaluation temperatures inside the domain.')
                for month in np.unique(month_numbers):
                    offset = np.asarray(model['temperature']['offsets_K'][str(month)])
                    if (offset.shape != (24, int(mask.sum())) or not np.isfinite(offset).all()
                            or not np.allclose(offset.sum(axis=0), 0, rtol=0, atol=1e-10)):
                        raise ValueError('Invalid zero-mean diurnal-cycle correction.')
                    active[month_numbers == month] += offset
                if np.any(active <= 0):
                    raise ValueError('Temperature correction produced nonpositive Kelvin temperatures.')
                temperatures[:, :, mask] = active
                result.attrs['temperature_cycle_adjustment'] = 'Frozen pooled month/hour zero-mean additive correction'
            result.attrs.update(pooled_correction_sha256=checksum,
                                pooled_correction_rain=int(rain), pooled_correction_temperature=int(temperature))
            target = destination / path.name
            from ai4water_eggs.storage import write_weather
            write_weather(result, target)
            result.close()
    _write_json(output / 'correction_application.json',
                {'schema': SCHEMA, 'source': str(root), 'correction_sha256': checksum,
                 'rain': rain, 'temperature': temperature, 'evaluation_seeds': seeds,
                 'calibration_seeds': model['calibration']['seeds'], 'sources': sources,
                 'metrics_status': 'Not computed; run metrics.analyze_ensemble before publishing.'})
    return output
