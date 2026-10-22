"""Separate rainfall variance loss by calendar timescale and spatial aggregation."""

from pathlib import Path

import numpy as np
import xarray as xr

from ai4water_eggs.metrics import scenario_files

SCALES = (1, 7, 30, 90, 365)
TEMPLATE = np.arange('2026-01-01', '2027-01-01', dtype='datetime64[D]')
SLOTS = {str(day)[5:]: i for i, day in enumerate(TEMPLATE)}


def _annual_sample(dates, rain):
    dates = np.asarray(dates, dtype='datetime64[D]')
    year = str(dates[0])[:4] if len(dates) else ''
    expected = np.arange(f'{year}-01-01', f'{int(year) + 1}-01-01', dtype='datetime64[D]') if year else []
    if not np.array_equal(dates, expected) or not np.isfinite(rain).all() or np.any(rain < 0):
        raise ValueError('Rainfall diagnostics require finite, nonnegative, complete calendar years.')
    return dates, np.asarray(rain, dtype=float)


def _aligned_totals(sample, scale):
    dates, rain = sample
    if scale == 365:
        return rain.sum(axis=0, keepdims=True)  # Full actual year, including February 29.
    cumulative = np.concatenate((np.zeros((1, rain.shape[1])), np.cumsum(rain, axis=0)))
    totals = cumulative[scale:] - cumulative[:-scale]
    result = np.full((365, rain.shape[1]), np.nan)
    for date, total in zip(dates[scale - 1:], totals):
        slot = SLOTS.get(str(date)[5:])
        if slot is not None:
            result[slot] = total
    return result


def _scale_statistics(samples, weights, scale):
    if len(samples) < 2:
        raise ValueError('At least two historical years/scenarios are required.')
    values = np.stack([_aligned_totals(sample, scale) for sample in samples])
    # A common endpoint must have a complete window in every realization.
    valid = np.isfinite(values).all(axis=(0, 2))
    values = values[:, valid]
    return {'regional': np.var(values @ weights, axis=0, ddof=1),
            'cell': np.var(values, axis=0, ddof=1).mean(axis=1)}


def _single(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one {pattern} in {directory}; found {len(paths)}.')
    return paths[0]


def compare_rainfall_scales(experiments, reference, years, labels=None, members=None):
    """Return RMS endpoint SDs/ratios; years lists complete individual years.

    Optional labels and member-number selections are keyed by experiment name.
    Only rainfall variables are read; hourly temperatures are never loaded.
    """
    labels, members, reference = labels or {}, members or {}, Path(reference)
    with xr.open_dataset(_single(reference / 'Data', 'A_Omega_*.nc')) as source:
        latitude, longitude = source.latitude.values.copy(), source.longitude.values.copy()
        mask = np.isfinite(source.Station_Mask.values)
    weights = np.cos(np.deg2rad(np.broadcast_to(latitude[:, None], mask.shape)[mask]))
    weights /= weights.sum()
    history_path = _single(reference / 'Data', 'daily_tp_*.nc')
    with xr.open_dataset(history_path) as source:
        if not (np.array_equal(source.latitude, latitude) and np.array_equal(source.longitude, longitude)):
            raise ValueError('Historical rainfall grid differs from the fitted domain.')
        dates, rain = source.time.values.astype('datetime64[D]'), source.tp.values[:, mask]
    historical = []
    for year in years:
        selected = dates.astype('datetime64[Y]') == np.datetime64(str(year), 'Y')
        historical.append(_annual_sample(dates[selected], rain[selected]))
    hv = {scale: _scale_statistics(historical, weights, scale) for scale in SCALES}
    rows = []
    for experiment in map(Path, experiments):
        generated, selected_numbers = [], []
        selected_members = members.get(experiment.name)
        for number, _, paths in scenario_files(experiment):
            if selected_members is not None and number not in selected_members:
                continue
            date_chunks, rain_chunks = [], []
            for path in paths:
                with xr.open_dataset(path) as source:
                    if not (np.array_equal(source.latitude, latitude) and np.array_equal(source.longitude, longitude)):
                        raise ValueError(f'{path}: scenario grid differs from the fitted domain.')
                    date_chunks.append(source.simulation_day.values.astype('datetime64[D]'))
                    rain_chunks.append(source.simulation_Precipitation.values[:, mask])
            generated.append(_annual_sample(np.concatenate(date_chunks), np.concatenate(rain_chunks)))
            selected_numbers.append(number)
        if selected_members is not None and set(selected_numbers) != set(selected_members):
            raise ValueError(f'{experiment}: requested scenario members are missing.')
        for scale in SCALES:
            gv = _scale_statistics(generated, weights, scale)
            for scope in ('regional', 'cell'):
                if hv[scale][scope].shape != gv[scope].shape:
                    raise ValueError('Historical/generated calendar endpoints differ.')
                h, g = float(np.sqrt(hv[scale][scope].mean())), float(np.sqrt(gv[scope].mean()))
                rows.append({'variant': labels.get(experiment.name, experiment.name), 'experiment': experiment.name,
                             'scale_days': scale, 'scope': scope, 'historical_rms_sd_mm': h,
                             'generated_rms_sd_mm': g, 'sd_ratio': g / h if h > 1e-12 else None,
                             'calendar_endpoints': len(gv[scope]), 'historical_years': len(historical),
                             'ensemble_members': len(generated), 'land_cells': int(mask.sum())})
    return {'rows': rows, 'historical_years': list(map(int, years)), 'reference_rainfall': str(history_path.resolve()),
            'methods': {
                'scales': '1, 7, 30 and 90-day rolling totals; 365 denotes complete actual annual totals, retaining leap days.',
                'alignment': 'Compute rolling totals on the actual calendar, then match 2026 month/day endpoints; February 29 is omitted only as an endpoint. No cross-year antecedents are supplied.',
                'variance': 'Sample variance (ddof=1) across realizations separately at each calendar endpoint; never pool seasonal means.',
                'summary': 'RMS SD = sqrt(mean(endpoint variances)); ratio = generated RMS SD / historical RMS SD.',
                'spatial': 'Regional rainfall uses fixed cos(latitude) cell weights before variance; individual-cell RMS averages cell variances equally across cells and endpoints.',
                'missing': 'Use endpoints complete in every realization; require complete years and fixed finite model cells. Read rainfall only.',
            }}
