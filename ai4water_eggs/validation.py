"""Compare generated seasons with matching historical seasons, without refitting.

Historical temperatures are normalized with the generator's trained trend and
reference date. A separate reference experiment/year range can supply held-out
seasons; the trend and SPI/SPEI fits always come from the evaluated model.
"""

import calendar
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
import xarray as xr

from ai4water_eggs.metrics import (finite_or_none, longest_spell, monthly_indices, monthly_series,
                             read_scenario, regional_daily, scenario_files, summarize, metric_definitions)
from ai4water_eggs.viewer import ASSETS, plotly_javascript

QUANTILES = [5, 10, 25, 50, 75, 90, 95]
SOURCES = ['https://nhess.copernicus.org/articles/25/3075/2025/nhess-25-3075-2025.html',
           'https://hess.copernicus.org/preprints/hess-2022-21/hess-2022-21.pdf']


def seasonal_dates(template, year):
    """Retain actual historical leap days; reject target dates absent in a year."""
    template = np.asarray(template, dtype='datetime64[D]')
    base = int(str(template[0])[:4])
    result = []
    for day in template.astype(str):
        try:
            result.append(np.datetime64(f'{year + int(day[:4]) - base}{day[4:]}', 'D'))
        except ValueError:
            return None
    result = np.asarray(result)
    complete = np.arange(result[0], result[-1] + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    missing = np.setdiff1d(complete, result)
    if any(str(day)[5:] != '02-29' for day in missing):
        return None
    return complete


def normalize_temperature(kelvin, dates, slope, reference_date):
    offset = (np.datetime64(reference_date, 'D') - np.asarray(dates, dtype='datetime64[D]')).astype(int)
    return np.asarray(kelvin) - 273.15 + slope * offset[:, None, None, None]


def rolling_average(values, days=7):
    values = np.asarray(values)
    result = np.full_like(values, np.nan, dtype=float)
    if values.shape[-1] >= days:
        windows = np.lib.stride_tricks.sliding_window_view(values, days, axis=-1)
        result[..., days - 1:] = windows.mean(axis=-1)
    return result


def quantile_bands(values):
    values = np.asarray(values, dtype=float)
    result = np.full((len(QUANTILES), values.shape[1]), np.nan)
    valid = np.isfinite(values).all(axis=0)
    if valid.any():
        result[:, valid] = np.percentile(values[:, valid], QUANTILES, axis=0)
    return {str(q): [finite_or_none(value) for value in row] for q, row in zip(QUANTILES, result)}


def distribution_comparison(historical, generated, bootstrap=1000, seed=42):
    historical = np.array([v for v in historical if v is not None and np.isfinite(v)], dtype=float)
    generated = np.array([v for v in generated if v is not None and np.isfinite(v)], dtype=float)
    if not len(historical) or not len(generated):
        return {'historical_n': len(historical), 'ensemble_n': len(generated), 'available': False}
    hq, gq = np.percentile(historical, QUANTILES), np.percentile(generated, QUANTILES)
    hs = np.std(historical, ddof=1) if len(historical) > 1 else np.nan
    gs = np.std(generated, ddof=1) if len(generated) > 1 else np.nan
    rng = np.random.default_rng(seed)
    bias_interval = None
    if bootstrap > 0:
        # Resample entire seasonal metric values, never individual days.
        hm = rng.choice(historical, (bootstrap, len(historical)), replace=True).mean(axis=1)
        gm = rng.choice(generated, (bootstrap, len(generated)), replace=True).mean(axis=1)
        bias_interval = np.percentile(gm - hm, [2.5, 97.5]).tolist()
    return {'available': True, 'historical_n': len(historical), 'ensemble_n': len(generated),
            'historical_mean': float(historical.mean()), 'ensemble_mean': float(generated.mean()),
            'mean_bias': float(generated.mean() - historical.mean()),
            'mean_bias_bootstrap_95': bias_interval,
            'historical_std': finite_or_none(hs), 'ensemble_std': finite_or_none(gs),
            'spread_ratio': float(gs / hs) if np.isfinite(hs) and hs > 1e-12 else None,
            'historical_quantiles': dict(zip(map(str, QUANTILES), hq.tolist())),
            'ensemble_quantiles': dict(zip(map(str, QUANTILES), gq.tolist())),
            'historical_inside_ensemble_p5_p95_percent': float(100 * np.mean((historical >= gq[0]) & (historical <= gq[-1]))),
            'wasserstein_distance': float(wasserstein_distance(historical, generated)),
            'ks_distance': float(ks_2samp(historical, generated, method='asymp').statistic)}


def _cell_summary(rain, hourly, mask):
    rain = rain[:, mask]
    return {'precipitation_mm': rain.sum(axis=0),
            'minimum_temperature_C': float(hourly[:, :, mask].min()),
            'temperature_C': hourly[:, :, mask].mean(axis=(0, 1)),
            'wet_day_percent': 100 * (rain >= 1).mean(axis=0),
            'dry_spell_days': np.array([longest_spell(rain[:, i] < 1) for i in range(rain.shape[1])]),
            'rain': rain}


def _member_cells(paths, mask, window=None):
    rain, hourly = [], []
    for path in paths:
        with xr.open_dataset(path) as source:
            days = source.simulation_day.values.astype('datetime64[D]')
            selected = np.ones(len(days), bool) if window is None else ((days >= np.datetime64(window[0], 'D')) & (days < np.datetime64(window[1], 'D')))
            rain.append(source.simulation_Precipitation.values[selected])
            values = source.simulation_t2m.values.reshape(-1, 24, *mask.shape) - 273.15
            hourly.append(values[selected])
    return _cell_summary(np.concatenate(rain), np.concatenate(hourly), mask)


def _pack_sample(label, daily, cell, fits, scales, period_scale):
    monthly = monthly_indices(monthly_series(daily), fits)
    return {'label': str(label), 'daily': daily, 'cells': cell,
            'metrics': {**summarize(daily, monthly, scales, period_scale),
                        'minimum_temperature_C': cell['minimum_temperature_C'],
                        'mean_daily_temperature_range_C': float(np.mean(daily['dtr']))}}


def _historical_samples(experiment, reference, template, catalog, mask, latitude, longitude, years):
    model_config = json.loads((experiment / 'run_config.json').read_text())
    reference_config = json.loads((reference / 'run_config.json').read_text())
    training_period = f"{model_config['Training Start Year']}_{model_config['Training End Year']}"
    source_period = f"{reference_config['Training Start Year']}_{reference_config['Training End Year']}"
    with xr.open_dataset(experiment / 'Data' / f'Detrended_t2m_{training_period}.nc') as fitted:
        coefficient = fitted['Detrend Coefficients'].values
        slope = float(coefficient[1])
        pivot = int(fitted['Detrend Pivot'].values)
        reference_date = str(fitted.time.values[pivot].astype('datetime64[D]'))
    rain_path = reference / 'Data' / f'daily_tp_{source_period}.nc'
    temperature_path = reference / 'Data' / f't2m_{source_period}.nc'
    if not temperature_path.is_file():
        raise ValueError('Historical comparison requires prepared raw t2m inputs so only the evaluated model trend is used.')
    result, skipped = [], []
    with xr.open_dataset(rain_path) as rain, xr.open_dataset(temperature_path) as temperature:
        for source in (rain, temperature):
            if not np.array_equal(source.latitude.values, latitude) or not np.array_equal(source.longitude.values, longitude):
                raise ValueError('Historical and model grids differ. Use identically prepared spatial domains.')
        dates = rain.time.values.astype('datetime64[D]')
        if not np.array_equal(dates, temperature.time.values.astype('datetime64[D]')):
            raise ValueError('Historical precipitation and temperature dates differ.')
        if len(np.unique(dates)) != len(dates):
            raise ValueError('Duplicate historical dates.')
        lookup = {day: i for i, day in enumerate(dates)}
        candidates = range(int(str(dates[0])[:4]), int(str(dates[-1])[:4]) + 1) if years is None else range(years[0], years[1] + 1)
        month_count = len(np.unique(template.astype('datetime64[M]')))
        full_months = all(monthly_series({'days': template, 'rain': np.zeros(len(template)),
                                         'pet': np.zeros(len(template)), 'tmean': np.zeros(len(template))})['complete'])
        period_scale = month_count if full_months else None
        for year in candidates:
            expected = seasonal_dates(template, year)
            if expected is None:
                skipped.append({'year': year, 'reason': 'Calendar/leap-day window is incompatible.'})
                continue
            missing = [str(day) for day in expected if day not in lookup]
            if missing:
                skipped.append({'year': year, 'reason': 'Incomplete season.', 'first_missing_date': missing[0], 'missing_days': len(missing)})
                continue
            indices = [lookup[day] for day in expected]
            precipitation = rain.tp.isel(time=indices).values
            celsius = normalize_temperature(temperature.t2m.isel(time=indices).values,
                                            expected, slope, reference_date)
            daily = regional_daily(expected, precipitation, celsius, latitude, mask)
            cell = _cell_summary(precipitation, celsius, mask)
            sample = _pack_sample(year, daily, cell, catalog['reference_fits'], catalog['scales_months'], period_scale)
            sample['year'] = year
            result.append(sample)
    return result, skipped, {'temperature_reference_date': reference_date,
                             'trained_temperature_trend_C_per_year': slope * 365.25,
                             'temperature_normalization': 'Prepared raw temperatures shifted using only the evaluated model slope and reference date; the trend is never refitted to evaluation data.'}


def _derived_series(samples, template=None):
    template = samples[0]['daily']['days'] if template is None else template
    template_year = int(str(template[0])[:4])
    keys = [(int(str(day)[:4]) - template_year, str(day)[5:]) for day in template]
    result = {key: [] for key in ('cumulative_rain','cumulative_pet','cumulative_balance','temperature_7day')}
    for sample in samples:
        daily = sample['daily']
        base = int(str(daily['days'][0])[:4])
        lookup = {(int(str(day)[:4])-base, str(day)[5:]): i for i,day in enumerate(daily['days'])}
        indices = [lookup[key] for key in keys]
        actual = {'cumulative_rain': np.cumsum(daily['rain']),
                  'cumulative_pet': np.cumsum(daily['pet']),
                  'cumulative_balance': np.cumsum(daily['rain']-daily['pet']),
                  'temperature_7day': rolling_average(daily['tmean'])}
        for key,values in actual.items():
            result[key].append(values[indices])
    return {key: np.asarray(values) for key,values in result.items()}


def _month_selection(sample, template, month):
    days = sample['daily']['days']
    origin = days[0].astype('datetime64[Y]').astype('datetime64[M]')
    target_origin = template[0].astype('datetime64[Y]').astype('datetime64[M]')
    offset = int((month-target_origin).astype(int))
    return (days.astype('datetime64[M]')-origin).astype(int) == offset


def _correlation(values):
    centered = values - values.mean(axis=0)
    scale = np.sqrt(np.sum(centered ** 2, axis=0))
    return np.divide(centered.T @ centered, scale[:, None] * scale[None, :],
                     out=np.full((values.shape[1], values.shape[1]), np.nan),
                     where=(scale[:, None] * scale[None, :]) > 0)


def _spatial_comparison(historical, generated, template):
    months = template.astype('datetime64[M]')
    rows = []
    for month in np.unique(months):
        selected = months == month
        h = np.concatenate([sample['cells']['rain'][_month_selection(sample, template, month)] for sample in historical])
        g = np.concatenate([sample['cells']['rain'][_month_selection(sample, template, month)] for sample in generated])
        row = {'month': str(month)[5:]}
        for key, hs, gs in [('rainfall', h, g), ('wet_occurrence', (h >= 1).astype(float), (g >= 1).astype(float))]:
            hc, gc = _correlation(hs), _correlation(gs)
            pairs = np.triu(np.ones(hc.shape, dtype=bool), 1) & np.isfinite(hc) & np.isfinite(gc)
            row[key + '_pairs'] = int(pairs.sum())
            row[key + '_correlation_MAE'] = float(np.abs(hc[pairs] - gc[pairs]).mean()) if pairs.any() else None
            row[key + '_historical_mean_correlation'] = float(hc[pairs].mean()) if pairs.any() else None
            row[key + '_ensemble_mean_correlation'] = float(gc[pairs].mean()) if pairs.any() else None
        rows.append(row)
    return rows


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return finite_or_none(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def validate_ensemble(experiment, reference=None, years=None, output=None, bootstrap=1000, window=None):
    experiment = Path(experiment).resolve()
    reference = Path(reference).resolve() if reference else experiment
    output = Path(output).resolve() if output else experiment / 'Validation'
    config = json.loads((experiment / 'run_config.json').read_text())
    if config.get('Non-stationary Mode', False):
        raise ValueError('This comparison targets a stationary reference climate; use Non-stationary Mode: false.')
    if bootstrap < 0 or (years is not None and years[1] < years[0]):
        raise ValueError('Use nonnegative bootstrap samples and increasing historical years.')
    catalog = json.loads((experiment / 'scenario_catalog.json').read_text())
    scenarios = scenario_files(experiment)
    if len(scenarios) != catalog['scenario_count']:
        raise ValueError('Scenario catalog is stale. Run ensemble.py analyze before validation.')
    period = f"{config['Training Start Year']}_{config['Training End Year']}"
    with xr.open_dataset(experiment / 'Data' / f'A_Omega_{period}.nc') as fitted:
        mask = np.isfinite(fitted.Station_Mask.values)
        latitude, longitude = fitted.latitude.values.copy(), fitted.longitude.values.copy()
    if window is not None:
        window = tuple(str(np.datetime64(value, 'D')) for value in window)
        if np.datetime64(window[1]) <= np.datetime64(window[0]):
            raise ValueError('Evaluation window end must follow its start.')
    first = read_scenario(scenarios[0][2], mask, latitude, longitude, window)
    template = first['days']
    if window is not None and not np.array_equal(template, np.arange(window[0], window[1], dtype='datetime64[D]')):
        raise ValueError('Evaluation window must be fully covered by every scenario.')
    if len(template) > 366:
        raise ValueError('Choose a simulation window no longer than one year for seasonal envelope validation.')
    period_scale = len(np.unique(template.astype('datetime64[M]'))) if all(monthly_series(first)['complete']) else None
    historical, skipped, normalization = _historical_samples(experiment, reference, template, catalog,
                                                            mask, latitude, longitude, years)
    if len(historical) < 2:
        raise ValueError('At least two complete historical seasons are needed. Inspect the year range and calendar coverage.')
    generated = []
    for number, directory, paths in scenarios:
        daily = first if number == scenarios[0][0] else read_scenario(paths, mask, latitude, longitude, window)
        if not np.array_equal(daily['days'], template):
            raise ValueError('All scenarios must cover identical dates.')
        cell = _member_cells(paths, mask, window)
        generated.append(_pack_sample(number, daily, cell, catalog['reference_fits'], catalog['scales_months'], period_scale))
        if number == scenarios[0][0] or number % 20 == 0 or len(generated) == len(scenarios):
            print(f'Validation: scenario {len(generated)}/{len(scenarios)}', flush=True)
    trained = set(range(config['Training Start Year'], config['Training End Year'] + 1)) - set(config.get('Excluded Training Years', []))
    overlaps = [s['year'] for s in historical if any(int(str(day)[:4]) in trained for day in s['daily']['days'])]
    mode = 'in_sample' if len(overlaps) == len(historical) else 'out_of_sample' if not overlaps else 'mixed'
    definitions = metric_definitions(catalog['scales_months'], period_scale) + [
        {'id':'minimum_temperature_C','label':'Lowest cell temperature','unit':'°C','description':'Lowest hourly temperature at any domain cell in the evaluated window.'},
        {'id':'mean_daily_temperature_range_C','label':'Mean daily temperature range','unit':'°C','description':'Area-weighted daily cell Tmax minus Tmin, averaged over the evaluated window.'}]
    comparison = {definition['id']: {**definition, **distribution_comparison(
        [s['metrics'][definition['id']] for s in historical],
        [s['metrics'][definition['id']] for s in generated], bootstrap)} for definition in definitions}
    hs, gs = _derived_series(historical, template), _derived_series(generated, template)
    envelopes = {}
    for key in hs:
        hbands, gbands = quantile_bands(hs[key]), quantile_bands(gs[key])
        lower, upper = np.asarray(gbands['5'], dtype=float), np.asarray(gbands['95'], dtype=float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        envelopes[key] = {'historical': hbands, 'ensemble': gbands,
                          'historical_pointwise_containment_percent': float(100 * np.mean((hs[key][:, valid] >= lower[valid]) & (hs[key][:, valid] <= upper[valid])))}
    monthly = {}
    monthly_comparisons = []
    month_indices = template.astype('datetime64[M]')
    for key, source, total in [('rain', 'rain', True), ('temperature', 'tmean', False), ('pet', 'pet', True), ('temperature_range', 'dtr', False)]:
        monthly[key] = {}
        monthly_values = {}
        for group, samples in [('historical', historical), ('ensemble', generated)]:
            values = np.array([[getattr(np, 'sum' if total else 'mean')(sample['daily'][source][_month_selection(sample, template, month)])
                                for month in np.unique(month_indices)] for sample in samples])
            monthly[key][group] = quantile_bands(values)
            monthly_values[group] = values
        for index, month in enumerate(np.unique(month_indices)):
            monthly_comparisons.append({'month': str(month), 'metric': key,
                                        **distribution_comparison(monthly_values['historical'][:,index], monthly_values['ensemble'][:,index], bootstrap=0)})
    maps = {}
    for key in ['precipitation_mm', 'temperature_C', 'wet_day_percent', 'dry_spell_days']:
        hmean = np.mean([sample['cells'][key] for sample in historical], axis=0)
        gmean = np.mean([sample['cells'][key] for sample in generated], axis=0)
        grid = np.full(mask.shape, np.nan)
        grid[mask] = gmean - hmean
        maps[key] = grid
    spatial = _spatial_comparison(historical, generated, template)
    fit_quality = []
    for name in ('O_Omega', 'A_Omega'):
        path = experiment / 'Data' / f'{name}_{period}.nc'
        if path.is_file():
            with xr.open_dataset(path) as source:
                fit_quality.extend(json.loads(source.attrs.get('calibration_report', '[]')))
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'experiment': experiment.name, 'historical_experiment': reference.name,
              'comparison_id': f'{experiment.name}:{template[0]}:{template[-1]}',
              'window_label': ('Full year' if str(template[0])[5:] == '01-01' and str(template[-1])[5:] == '12-31' else calendar.month_abbr[int(str(template[0])[5:7])] + '–' + calendar.month_abbr[int(str(template[-1])[5:7])]),
              'comparison_mode': mode, 'training_years': sorted(trained),
              'historical_years': [s['year'] for s in historical], 'overlapping_training_years': overlaps,
              'ensemble_members': len(generated), 'historical_seasons': len(historical),
              'days_per_season': len(template), 'land_cells': int(mask.sum()),
              'start': str(template[0]), 'last_day': str(template[-1]),
              'skipped_historical_seasons': skipped, **normalization,
              'methods': {'aggregation': catalog['methods']['spatial'], 'pet': catalog['methods']['pet'],
                          'calendar': 'Actual historical 365/366-day totals and spells are retained. February 29 is omitted only from non-leap daily plots; its cumulative contribution is retained.',
                          'window': 'Metrics and drought indices use the displayed evaluation window; the full-year catalog retains full-year context.',
                          'bootstrap': f'{bootstrap} independent resamples of whole historical seasons and ensemble members; descriptive sampling variability, not parameter uncertainty or independent skill',
                          'envelopes': 'Pointwise P5/P25/P50/P75/P95 across complete matching seasons; an envelope is not a single physically coherent scenario or simultaneous trajectory coverage',
                          'scores': 'Unpaired seasonal distributions: mean bias, spread ratio, Wasserstein distance, KS distance (no p-values), empirical historical containment in the ensemble P5–P95 interval',
                          'spatial': 'Pearson daily all-rain and wet-occurrence correlations, pooled across matching seasonal months; finite off-diagonal station pairs',
                          'reference': 'Prepared historical ERA5-Land reanalysis on the exact model mask; historical temperature normalization uses only the evaluated model trend',
                          'interpretation': 'Historical climatology fidelity diagnostic; no date-matched forecast verification and no overall scientific pass/fail threshold',
                          'sources': SOURCES},
              'comparisons': comparison, 'envelopes': envelopes, 'monthly': monthly,
              'monthly_comparisons': monthly_comparisons,
              'temperature_range': {'historical_mean_C': float(np.mean([np.mean(sample['daily']['dtr']) for sample in historical])),
                                    'ensemble_mean_C': float(np.mean([np.mean(sample['daily']['dtr']) for sample in generated]))},
              'spatial_correlation': spatial, 'spatial_calibration': fit_quality,
              'latitude': latitude, 'longitude': longitude, 'mask': mask, 'bias_maps': maps,
              'boundary': json.loads((experiment / 'boundary.geojson').read_text()) if (experiment / 'boundary.geojson').is_file() else None,
              'dates': template.astype(str).tolist(), 'months': [str(month) for month in np.unique(month_indices)],
              'historical': [{'year': s['year'], 'calendar_days': len(s['daily']['days']), 'values': s['metrics'], 'series': {key: hs[key][i] for key in hs}} for i, s in enumerate(historical)],
              'ensemble': [{'number': int(s['label']), 'values': s['metrics']} for s in generated]}
    report = _json_safe(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'seasonal_metrics.csv').open('w', newline='') as stream:
        fields = ['sample', 'year_or_scenario'] + [item['id'] for item in definitions]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, samples in [('historical', historical), ('ensemble', generated)]:
            writer.writerows({'sample': group, 'year_or_scenario': sample['label'], **sample['metrics']} for sample in samples)
    with (output / 'monthly_comparison.csv').open('w', newline='') as stream:
        fields = ['month','metric','historical_mean','ensemble_mean','mean_bias','historical_std','ensemble_std','spread_ratio','historical_inside_ensemble_p5_p95_percent']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(monthly_comparisons)
    with (output / 'distribution_comparison.csv').open('w', newline='') as stream:
        fields = ['metric', 'unit', 'historical_n', 'ensemble_n', 'historical_mean', 'ensemble_mean',
                  'mean_bias', 'historical_std', 'ensemble_std', 'spread_ratio',
                  'historical_inside_ensemble_p5_p95_percent', 'wasserstein_distance', 'ks_distance']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({'metric': key, **value} for key, value in comparison.items())
    embedded = json.dumps(report, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    html = (ASSETS / 'validation.html').read_text()
    html = html.replace('<!--PLOTLY-->', '<script>' + plotly_javascript().replace('</script', '<\\/script') + '</script>')
    html = html.replace('<!--DATA-->', '<script>window.VALIDATION_DATA=' + embedded + ';</script>')
    html = html.replace('<!--APPLICATION-->', '<script>' + (ASSETS / 'validation.js').read_text().replace('</script', '<\\/script') + '</script>')
    (output / 'report.html').write_text(html)
    print(f'Historical comparison: {output / "report.html"} ({mode}; {len(historical)} historical seasons, {len(generated)} members)', flush=True)
    return report
