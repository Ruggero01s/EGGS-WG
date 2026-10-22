"""Vectorized upstream Fourier projections with explicit sparse-day handling."""

import numpy as np


def fill_calendar_gaps(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError('No observations support this seasonal temperature condition.')
    if valid.all():
        return values.copy(), 0
    index = np.arange(len(values))
    return np.interp(index, index[valid], values[valid], period=len(values)), int((~valid).sum())


def project(climatology, harmonics, baseline):
    """Use the same sine/cosine coefficients and one-based times as upstream."""
    size = len(climatology)
    angle = 2 * np.pi * np.asarray(harmonics)[:, None] * np.arange(1, size + 1)[None, :] / size
    cosine, sine = np.cos(angle), np.sin(angle)
    alpha = 2 / size * (cosine @ climatology)
    beta = 2 / size * (sine @ climatology)
    return baseline + alpha @ cosine + beta @ sine


def fit_hourly(temperature, standard_deviations, day_numbers, station, station_count):
    print(f'Calculating Fourier Hours:{station}/{station_count}', flush=True)
    day_numbers = np.asarray(day_numbers)
    mean = np.empty((365, 24))
    std = np.empty_like(mean)
    for index in range(365):
        selected = temperature[day_numbers == index + 1]
        if not len(selected):
            raise ValueError(f'No unconditioned data for calendar day {index + 1}.')
        mean[index] = np.nanmean(selected, axis=0)
        std[index] = np.nanstd(selected, axis=0)
    harmonics = [1, 365, 730, 1095, 1460, 2190]
    return station, project(mean.ravel(), harmonics, np.nanmean(temperature)), \
        project(std.ravel(), harmonics, np.mean(standard_deviations))


def fit_conditioned(temperature, precipitation, dry_std, wet_std, day_numbers, station, station_count,
                    fit_mode='legacy'):
    if fit_mode not in ('legacy', 'calendar-hourly'):
        raise ValueError('Temperature Seasonal Fit must be legacy or calendar-hourly.')
    print(f'Calculating Fourier Days:{station}/{station_count}', flush=True)
    day_numbers = np.asarray(day_numbers)
    results, report = {}, {'station': station}
    for label, selected, base_std in (('wet', precipitation.astype(bool), wet_std),
                                      ('dry', ~precipitation.astype(bool), dry_std)):
        values = temperature[selected]
        days = day_numbers[selected]
        mean = np.full(365, np.nan)
        std = np.full(365, np.nan)
        for index in range(365):
            group = values[days == index + 1]
            if len(group):
                mean[index] = np.nanmean(group)
                std[index] = (np.nanstd(group) if fit_mode == 'legacy'
                              else np.nanmean(np.nanstd(group, axis=0)))
        mean, count = fill_calendar_gaps(mean)
        std, _ = fill_calendar_gaps(std)
        report[label + '_calendar_days_interpolated'] = count
        baseline = np.nanmean(values) if fit_mode == 'legacy' else mean.mean()
        results[label] = (project(mean, [1, 2, 3, 4, 6], baseline),
                          project(std, [1, 2, 3, 4, 6], np.nanmean(base_std)))
    return station, *results['wet'], *results['dry'], report
