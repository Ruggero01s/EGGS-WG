"""Regional ensemble metrics and historical-reference SPI/SPEI.

SPI uses a zero-adjusted gamma fit (MLE); SPEI uses the generalized logistic
parameterization of the three-parameter log-logistic with unbiased L-moments,
as used by the R SPEI package. All fits are separate by ending calendar month.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import gamma, norm
import xarray as xr

PROBABILITY_EPSILON = 1e-10
MIN_REFERENCE_YEARS = 30
SOURCES = {
    'pet': 'https://www.fao.org/4/X0490E/x0490e07.htm',
    'spi': 'https://www.droughtmanagement.info/literature/WMO_standardized_precipitation_index_user_guide_en_2012.pdf',
    'spei': 'https://github.com/cran/SPEI/blob/master/R/spei.R',
}


def hargreaves_pet(days, latitude, minimum, maximum):
    """Cell ETo in mm/day, FAO-56 eqs. 21, 23–25, 52; inputs in Celsius.

    Radiation is initially MJ/m²/day and is converted to equivalent mm/day.
    The temperature mean in this equation is (Tmin + Tmax)/2.
    """
    days = np.asarray(days, dtype='datetime64[D]')
    julian = (days - days.astype('datetime64[Y]')).astype(int) + 1
    angle = 2 * np.pi * julian[:, None] / 365
    latitude = np.deg2rad(np.asarray(latitude))[None, :]
    distance = 1 + 0.033 * np.cos(angle)
    declination = 0.409 * np.sin(angle - 1.39)
    sunset = np.arccos(np.clip(-np.tan(latitude) * np.tan(declination), -1, 1))
    radiation = (24 * 60 / np.pi * 0.0820 * distance
                 * (sunset * np.sin(latitude) * np.sin(declination)
                    + np.cos(latitude) * np.cos(declination) * np.sin(sunset)))
    return np.maximum(0, 0.0023 * 0.408 * radiation
                      * ((minimum + maximum) / 2 + 17.8)
                      * np.sqrt(np.maximum(0, maximum - minimum)))


def regional_daily(days, precipitation, hourly_celsius, latitude, mask):
    """Reduce the same fixed cells for history and every scenario."""
    days = np.asarray(days, dtype='datetime64[D]')
    if len(np.unique(days)) != len(days) or np.any(np.diff(days) <= np.timedelta64(0, 'D')):
        raise ValueError('Daily dates must be unique and increasing.')
    rain = np.asarray(precipitation)[:, mask]
    temperature = np.asarray(hourly_celsius)[:, :, mask]
    if temperature.shape != (len(days), 24, int(mask.sum())) or rain.shape[0] != len(days):
        raise ValueError('Expected one precipitation field and 24 temperatures per daily date.')
    if not np.isfinite(rain).all() or not np.isfinite(temperature).all():
        raise ValueError('Missing values inside the fixed regional domain; metrics were not calculated.')
    if np.any(rain < 0):
        raise ValueError('Negative precipitation in a scenario; correct the weather output before ranking.')
    cell_latitudes = np.broadcast_to(np.asarray(latitude)[:, None], mask.shape)[mask]
    weights = np.cos(np.deg2rad(cell_latitudes))
    weights /= weights.sum()
    minimum, maximum = temperature.min(axis=1), temperature.max(axis=1)
    pet = hargreaves_pet(days, cell_latitudes, minimum, maximum)
    return {'days': days, 'rain': rain @ weights, 'pet': pet @ weights,
            'tmean': temperature.mean(axis=1) @ weights,
            'tmin': minimum @ weights, 'tmax': maximum @ weights,
            'dtr': (maximum-minimum) @ weights,
            'peak_temperature': float(temperature.max()), 'land_cells': int(mask.sum())}


def monthly_series(daily):
    """Keep missing/partial calendar months as NaN, including excluded years."""
    dates = daily['days']
    first, last = dates[0].astype('datetime64[M]'), dates[-1].astype('datetime64[M]')
    months = np.arange(first, last + np.timedelta64(1, 'M'), dtype='datetime64[M]')
    result = {'months': months, 'complete': [], 'rain': [], 'pet': [], 'temperature': []}
    for month in months:
        selected = dates.astype('datetime64[M]') == month
        expected = np.arange(month.astype('datetime64[D]'),
                             (month + np.timedelta64(1, 'M')).astype('datetime64[D]'))
        complete = np.array_equal(dates[selected], expected)
        result['complete'].append(complete)
        for target, key, reduction in (('rain', 'rain', np.sum), ('pet', 'pet', np.sum),
                                       ('temperature', 'tmean', np.mean)):
            result[target].append(float(reduction(daily[key][selected])) if complete else np.nan)
    for key in ('rain', 'pet', 'temperature'):
        result[key] = np.asarray(result[key])
    return result


def rolling_sum(values, scale):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for index in range(scale - 1, len(values)):
        block = values[index - scale + 1:index + 1]
        if np.isfinite(block).all():
            result[index] = block.sum()
    return result


def fit_distribution(values, index, minimum_years=MIN_REFERENCE_YEARS):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    fit = {'n': len(values), 'valid': False}
    if len(values) < minimum_years:
        return {**fit, 'reason': f'Fewer than {minimum_years} complete reference years.'}
    if np.ptp(values) <= 1e-12:
        return {**fit, 'reason': 'Constant historical values.'}
    if index == 'spi':
        positive = values[values > 0]
        if len(positive) < 4 or np.ptp(positive) <= 1e-12:
            return {**fit, 'reason': 'Insufficient varying positive precipitation.'}
        shape, _, scale = gamma.fit(positive, floc=0)
        return {**fit, 'valid': True, 'shape': float(shape), 'scale': float(scale),
                'zero_probability': float(np.mean(values == 0))}
    # Unbiased probability-weighted moments -> L1, L2, L3 -> generalized logistic.
    values = np.sort(values)
    n = len(values)
    order = np.arange(n)
    b0 = values.mean()
    b1 = np.mean(values * order / (n - 1))
    b2 = np.mean(values * order * (order - 1) / ((n - 1) * (n - 2)))
    l2, l3 = 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    shape = -l3 / l2
    if l2 <= 0 or not np.isfinite(shape) or abs(shape) >= 1:
        return {**fit, 'reason': 'Invalid historical L-moments for a log-logistic fit.'}
    log_factor = gammaln(1 + shape) + gammaln(1 - shape)
    scale = l2 * np.exp(-log_factor)
    location = b0 + scale * np.expm1(log_factor) / shape if abs(shape) > 1e-7 else b0
    return {**fit, 'valid': True, 'location': float(location), 'scale': float(scale),
            'shape': float(shape)}


def index_value(value, fit, index):
    """Return (normal score, probability clipped at finite tail bound)."""
    if not fit['valid'] or not np.isfinite(value):
        return None, False
    if index == 'spi':
        q = fit['zero_probability']
        probability = q + (1 - q) * gamma.cdf(value, fit['shape'], scale=fit['scale'])
    else:
        standardized = (value - fit['location']) / fit['scale']
        shape = fit['shape']
        support = 1 - shape * standardized
        if abs(shape) < 1e-7:
            probability = expit(standardized)
        elif support <= 0:
            probability = 1.0 if shape > 0 else 0.0
        else:
            probability = expit(-np.log1p(-shape * standardized) / shape)
    bounded = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return float(norm.ppf(bounded)), bool(probability != bounded)


def fit_reference(monthly, scales, minimum_years=MIN_REFERENCE_YEARS):
    month_numbers = monthly['months'].astype(int) % 12 + 1
    result = {}
    for index in ('spi', 'spei'):
        values = monthly['rain'] if index == 'spi' else monthly['rain'] - monthly['pet']
        for scale in scales:
            accumulated = rolling_sum(values, scale)
            result[f'{index}_{scale}'] = {
                str(month): fit_distribution(accumulated[month_numbers == month], index, minimum_years)
                for month in range(1, 13)}
    return result


def monthly_indices(monthly, fits):
    rows = []
    for i, month in enumerate(monthly['months']):
        rows.append({'month': str(month), 'complete': bool(monthly['complete'][i]),
                     'precipitation_mm': finite_or_none(monthly['rain'][i]),
                     'pet_mm': finite_or_none(monthly['pet'][i]),
                     'mean_temperature_C': finite_or_none(monthly['temperature'][i]),
                     'clipped_indices': []})
    for key, month_fits in fits.items():
        index, scale = key.split('_')
        data = monthly['rain'] if index == 'spi' else monthly['rain'] - monthly['pet']
        accumulated = rolling_sum(data, int(scale))
        for i, row in enumerate(rows):
            month = str(int(monthly['months'][i].astype(int) % 12 + 1))
            row[key], clipped = index_value(accumulated[i], month_fits[month], index)
            if clipped:
                row['clipped_indices'].append(key)
    return rows


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def longest_spell(condition):
    best = current = 0
    for value in condition:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def metric_definitions(scales, period_scale):
    definitions = [
        ('total_precipitation_mm', 'Total precipitation', 'mm', 'Area-weighted accumulated precipitation.'),
        ('mean_temperature_C', 'Mean temperature', '°C', 'Mean hourly temperature over all days and domain cells.'),
        ('peak_temperature_C', 'Peak cell temperature', '°C', 'Highest hourly temperature at any domain cell.'),
        ('pet_mm', 'Estimated evaporative demand', 'mm', 'Area-weighted Hargreaves reference ETo, accumulated over the period.'),
        ('water_balance_mm', 'Precipitation minus PET', 'mm', 'Accumulated regional precipitation minus estimated ETo.'),
        ('wet_days', 'Regional wet days', 'days', 'Days with regional mean rainfall at least 1 mm.'),
        ('longest_dry_spell_days', 'Longest regional dry spell', 'days', 'Consecutive days with regional mean rainfall below 1 mm.'),
        ('hot_days_30', 'Regional hot days ≥30 °C', 'days', 'Days with area-weighted daily maximum temperature at least 30 °C.'),
        ('max_daily_precipitation_mm', 'Maximum daily precipitation', 'mm', 'Largest single-day regional mean rainfall.'),
    ]
    for index in ('spi', 'spei'):
        for scale in scales:
            definitions.extend([
                (f'{index}_{scale}_end', f'{index.upper()}-{scale} at period end', '',
                 f'{scale}-month {index.upper()} ending in the final simulated calendar month.'),
                (f'{index}_{scale}_min', f'Minimum {index.upper()}-{scale}', '',
                 f'Lowest available {scale}-month {index.upper()} within the simulation; initial incomplete windows are omitted.'),
            ])
        if period_scale:
            definitions.append((f'{index}_period', f'{index.upper()} for the whole period ({period_scale} months)', '',
                                f'{period_scale}-month {index.upper()} covering the complete simulation window.'))
    return [{'id': key, 'label': label, 'unit': unit, 'description': description}
            for key, label, unit, description in definitions]


def summarize(daily, monthly, scales, period_scale):
    rain, pet = daily['rain'], daily['pet']
    values = {'total_precipitation_mm': float(rain.sum()), 'mean_temperature_C': float(daily['tmean'].mean()),
              'peak_temperature_C': daily['peak_temperature'], 'pet_mm': float(pet.sum()),
              'water_balance_mm': float(rain.sum() - pet.sum()), 'wet_days': int(np.sum(rain >= 1)),
              'longest_dry_spell_days': longest_spell(rain < 1),
              'hot_days_30': int(np.sum(daily['tmax'] >= 30)),
              'max_daily_precipitation_mm': float(rain.max())}
    for index in ('spi', 'spei'):
        for scale in scales:
            key = f'{index}_{scale}'
            valid = [row[key] for row in monthly if row[key] is not None]
            values[key + '_end'] = monthly[-1][key]
            values[key + '_min'] = min(valid) if valid else None
        if period_scale:
            values[index + '_period'] = monthly[-1][f'{index}_{period_scale}']
    return values


def scenario_files(experiment):
    directories = (Path(experiment) / 'Simulation').glob('Simulation_*')
    result = []
    for directory in directories:
        if directory.name.removeprefix('Simulation_').isdigit():
            paths = sorted(path for path in directory.glob('*.nc') if not path.name.startswith('tmp_'))
            if paths:
                result.append((int(directory.name.removeprefix('Simulation_')), directory, paths))
    return sorted(result)


def read_scenario(paths, mask, latitude, longitude, window=None):
    chunks = []
    for path in paths:
        with xr.open_dataset(path) as source:
            chunks.append(source.load())
    source = chunks[0] if len(chunks) == 1 else xr.combine_by_coords(chunks, combine_attrs='override')
    if not np.array_equal(source.latitude.values, latitude) or not np.array_equal(source.longitude.values, longitude):
        raise ValueError('Scenario and historical spatial grids differ.')
    days = source.simulation_day.values.astype('datetime64[D]')
    if not np.all(np.diff(days) == np.timedelta64(1, 'D')):
        raise ValueError('Simulation days must be consecutive.')
    expected_hours = (days[:, None].astype('datetime64[h]') + np.arange(24) * np.timedelta64(1, 'h')).ravel()
    if not np.array_equal(source.simulation_time.values, expected_hours):
        raise ValueError('Scenario hours are not aligned to the daily UTC dates.')
    temperature = source.simulation_t2m.values.reshape(len(days), 24, *mask.shape) - 273.15
    selected = np.ones(len(days), dtype=bool) if window is None else ((days >= np.datetime64(window[0], 'D')) & (days < np.datetime64(window[1], 'D')))
    if not selected.any():
        raise ValueError('Evaluation window contains no scenario days.')
    daily = regional_daily(days[selected], source.simulation_Precipitation.values[selected], temperature[selected], latitude, mask)
    if np.isfinite(source.simulation_Precipitation.values[:, ~mask]).any():
        raise ValueError('Scenario contains rainfall outside the historical domain mask.')
    daily['seed'] = int(source.attrs['random_seed']) if 'random_seed' in source.attrs else None
    daily['negative_precipitation_clipped'] = int(source.attrs.get('negative_precipitation_clipped', 0))
    source.close()
    return daily


def analyze_ensemble(experiment, scales=(1, 3, 6), minimum_years=MIN_REFERENCE_YEARS):
    experiment = Path(experiment).resolve()
    scenarios = scenario_files(experiment)
    if not scenarios:
        raise ValueError('No completed scenarios found.')
    if not scales or any(type(scale) is not int or scale < 1 for scale in scales):
        raise ValueError('Drought scales must be positive integer month counts.')
    config = json.loads((experiment / 'run_config.json').read_text())
    years = f"{config['Training Start Year']}_{config['Training End Year']}"
    with xr.open_dataset(experiment / 'Data' / f'A_Omega_{years}.nc') as source:
        mask = np.isfinite(source.Station_Mask.values)
        latitude, longitude = source.latitude.values.copy(), source.longitude.values.copy()
    with xr.open_dataset(experiment / 'Data' / f'daily_tp_{years}.nc') as source:
        rain, dates = source.tp.values, source.time.values.astype('datetime64[D]')
    with xr.open_dataset(experiment / 'Data' / f'Detrended_t2m_{years}.nc') as source:
        if not np.array_equal(source.time.values.astype('datetime64[D]'), dates):
            raise ValueError('Historical temperature and precipitation dates differ.')
        historical = regional_daily(dates, rain, source.t2m.values - 273.15, latitude, mask)
        reference_date = source.attrs.get('detrend_reference_date')
    baseline = monthly_series(historical)
    first_daily = read_scenario(scenarios[0][2], mask, latitude, longitude)
    first_monthly = monthly_series(first_daily)
    period_scale = len(first_monthly['months']) if all(first_monthly['complete']) else None
    scales = sorted(set(scales) | ({period_scale} if period_scale else set()))
    fits = fit_reference(baseline, scales, minimum_years)
    rows = []
    for number, directory, paths in scenarios:
        daily = first_daily if number == scenarios[0][0] else read_scenario(paths, mask, latitude, longitude)
        if not np.array_equal(daily['days'], first_daily['days']):
            raise ValueError('Every scenario must cover the same dates to compare percentiles.')
        monthly = monthly_indices(monthly_series(daily), fits)
        rows.append({'id': directory.relative_to(experiment.parent).as_posix(), 'number': number,
                     'seed': daily['seed'], 'start': str(daily['days'][0]), 'last_day': str(daily['days'][-1]),
                     'days': len(daily['days']), 'land_cells': daily['land_cells'],
                     'quality': {'negative_precipitation_clipped': daily['negative_precipitation_clipped']},
                     'files': [path.relative_to(experiment).as_posix() for path in paths],
                     'values': summarize(daily, monthly, scales, period_scale), 'monthly': monthly})
        print(f'Metrics: scenario {number}/{len(scenarios)}', flush=True)
    warnings = [f'{key}, ending month {month}: {fit["reason"]}'
                for key, months in fits.items() for month, fit in months.items() if not fit['valid']]
    catalog = {'schema_version': 1, 'experiment': experiment.name, 'scenario_count': len(rows),
               'start': rows[0]['start'], 'last_day': rows[0]['last_day'],
               'baseline': {'start': str(dates[0]), 'last_day': str(dates[-1]),
                            'temperature_reference_date': reference_date, 'land_cells': int(mask.sum()),
                            'complete_months': int(sum(baseline['complete']))},
               'methods': {'spatial': 'cos(latitude) weights over the fixed modelled cells; nonlinear PET is computed per cell before averaging',
                           'pet': 'Hargreaves-Samani FAO-56 reference ETo from temperature range and latitude; no wind/humidity/radiation fields',
                           'spi': 'Gamma MLE with a zero-precipitation probability mass; separate fit per ending month and scale',
                           'spei': 'Generalized logistic (three-parameter log-logistic), unbiased probability-weighted L-moments; separate fit per ending month and scale',
                           'minimum_reference_years': minimum_years,
                           'missing_months': 'No infilling or antecedent weather: partial months and incomplete rolling windows produce null indices',
                           'probability_clip': PROBABILITY_EPSILON,
                           'selection': 'Linear ensemble quantile, then closest actual metric value; distances tied within 1e-12 times max(1, abs(metric values), abs(target)) resolve to the lowest scenario number',
                           'percentile_rank': '100 * (count below + (count equal - 1)/2) / (N - 1); a single valid member ranks P50',
                           'sources': SOURCES},
               'scales_months': scales, 'metrics': metric_definitions(scales, period_scale),
               'reference_fits': fits, 'warnings': warnings, 'scenarios': rows}
    output = experiment / 'scenario_catalog.json'
    output.write_text(json.dumps(catalog, indent=2, allow_nan=False) + '\n')
    fields = ['scenario_id', 'scenario_number', 'seed'] + [metric['id'] for metric in catalog['metrics']]
    with (experiment / 'scenario_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({'scenario_id': row['id'], 'scenario_number': row['number'],
                             'seed': row['seed'], **row['values']})
    with (experiment / 'scenario_monthly_metrics.csv').open('w', newline='') as stream:
        fields = ['scenario_id', 'month', 'complete', 'precipitation_mm', 'pet_mm', 'mean_temperature_C'] + list(fits)
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerows({'scenario_id': row['id'], **month} for month in row['monthly'])
    print(f'Scenario catalog written: {output}', flush=True)
    return catalog


def select_scenario(catalog, metric, percentile):
    if metric not in {item['id'] for item in catalog['metrics']}:
        raise ValueError(f'Unknown scenario metric: {metric}')
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('Percentile must lie between 0 and 100.')
    candidates = [row for row in catalog['scenarios'] if row['values'].get(metric) is not None]
    if not candidates:
        raise ValueError(f'No scenarios have a valid {metric}; inspect historical fit support and simulation length.')
    values = np.array([row['values'][metric] for row in candidates])
    target = float(np.percentile(values, percentile, method='linear'))
    closest = float(np.min(np.abs(values - target)))
    tolerance = 1e-12 * max(1., float(np.max(np.abs(values))), abs(target))
    tied = [row for row in candidates if abs(row['values'][metric] - target) <= closest + tolerance]
    chosen = min(tied, key=lambda row: row['number'])
    actual = chosen['values'][metric]
    rank = (100 * (np.sum(values < actual) + (np.sum(values == actual) - 1) / 2) / (len(values) - 1)
            if len(values) > 1 else 50.0)
    return {'metric': metric, 'requested_percentile': float(percentile), 'target_value': target,
            'actual_value': actual, 'actual_ensemble_percentile': float(rank),
            'valid_scenarios': len(values), 'scenario': chosen}
