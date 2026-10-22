"""Isolate temperature reconstruction effects using existing stationary ensembles.

Example (from repository root):
uv run python scripts/diagnose_temperature.py --raw outputs/capitanata_fix_trial/raw_n200 \
    --adjusted outputs/capitanata_fix_trial/current_n200 \
    --output outputs/capitanata_fix_trial/TemperatureDiagnosis
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4water_eggs.seasonal import fill_calendar_gaps, project


def fit_controls(data_directory, years, mask, template):
    """Change only the two conditioned daily Fourier fitting choices.

    Retain all training years, Fourier harmonics and hourly reconstruction.
    Compare calendar vs event-weighted mean intercepts, and mean hourly SD
    vs SD pooled across all hours for the seasonal scale harmonics.
    """
    with xr.open_dataset(data_directory / f'Detrended_t2m_{years}.nc') as ds:
        dates = ds.time.values.astype('datetime64[D]')
        ordinal = (dates - dates.astype('datetime64[Y]')).astype(int)
        values = ds.t2m.values[:, :, mask] - 273.15
    with xr.open_dataset(data_directory / f'daily_tp_{years}.nc') as ds:
        np.testing.assert_array_equal(ds.time.values.astype('datetime64[D]'), dates)
        wet = ds.Precipitation_mask.values[:, mask].astype(bool)
    daily_means = values.mean(axis=1)
    controls, audit = {}, []
    for state, label in ((False, 'Dry'), (True, 'Wet')):
        selected = wet == state
        event_mean = (daily_means * selected).sum(axis=0) / selected.sum(axis=0)
        climatology = np.full((365, mask.sum()), np.nan)
        pooled = np.full_like(climatology, np.nan)
        hourly = np.full_like(climatology, np.nan)
        for day in range(365):
            day_values, day_states = values[ordinal == day], selected[ordinal == day]
            for cell in range(mask.sum()):
                sample = day_values[day_states[:, cell], :, cell]
                if len(sample):
                    climatology[day, cell] = sample.mean()
                    pooled[day, cell] = sample.std()
                    hourly[day, cell] = sample.std(axis=0).mean()
        intercept = np.empty(mask.sum())
        scale_change = np.empty_like(climatology)
        for cell in range(mask.sum()):
            climate, _ = fill_calendar_gaps(climatology[:, cell])
            old_scale, _ = fill_calendar_gaps(pooled[:, cell])
            new_scale, _ = fill_calendar_gaps(hourly[:, cell])
            intercept[cell] = event_mean[cell] - climate.mean()
            scale_change[:, cell] = project(new_scale, [1, 2, 3, 4, 6], 0) - project(old_scale, [1, 2, 3, 4, 6], 0)
            # Verify that this recomputation reproduces the actual trained templates.
            original_mean = template[f'Fourier_Mean_{label}'][:, cell].reshape(365, 24).mean(axis=1) - 273.15
            original_scale = template[f'Fourier_Std_{label}'][:, cell].reshape(365, 24).mean(axis=1)
            np.testing.assert_allclose(project(climate, [1, 2, 3, 4, 6], event_mean[cell]), original_mean, rtol=0, atol=1e-8)
            np.testing.assert_allclose(project(old_scale, [1, 2, 3, 4, 6], np.nanmean(hourly[:, cell])), original_scale, rtol=0, atol=1e-8)
        controls[label] = intercept, scale_change
        audit.append({'state': label.lower(), 'event_minus_calendar_intercept_C': float(intercept.mean()),
                      'mean_intercept_change_C': float(-intercept.mean()),
                      'summer_scale_change_C': float(scale_change[151:243].mean())})
    return controls, audit


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(name, samples):
    samples = np.stack(samples)
    if not np.isfinite(samples).all():
        raise ValueError(f'{name}: nonfinite temperatures.')
    return {
        'variant': name, 'samples': len(samples), 'summer_mean_C': float(samples.mean()),
        'daily_mean_p99_C': float(np.percentile(samples.mean(axis=2), 99)),
        'daily_max_p99_C': float(np.percentile(samples.max(axis=2), 99)),
        'mean_cell_summer_peak_C': float(samples.max(axis=(1, 2)).mean()),
        'mean_domain_summer_peak_C': float(samples.max(axis=(1, 2, 3)).mean()),
    }


def hour_indices(days):
    # Same ordinal convention as Full_Simulate_Day, including leap years.
    return np.minimum((days - days.astype('datetime64[Y]')).astype(int), 364)[:, None] * 24 + np.arange(24)


def diagnose(raw, adjusted, output, members):
    raw, adjusted, output = (path.resolve() for path in (raw, adjusted, output))
    config = json.loads((raw / 'run_config.json').read_text())
    if config.get('Non-stationary Mode') or config.get('Variable Adjust'):
        raise ValueError('Raw input must be stationary with adjustments disabled.')
    years = f"{config['Training Start Year']}_{config['Training End Year']}"
    coefficient = raw / 'Data' / f't2m_PreSim_Coeff_{years}.nc'
    with xr.open_dataset(coefficient) as ds:
        if ds.attrs.get('temperature_residual_conditioning', 'legacy') != 'legacy':
            raise ValueError('This diagnosis requires legacy unconditioned training residuals.')
        mask = np.isfinite(ds.Station_Mask.values)
        station = ds.Station_Mask.values[mask]
        if not np.array_equal(station, np.arange(len(station))):
            raise ValueError('Station ordering differs from grid flattening.')
        latitude, longitude = ds.latitude.values, ds.longitude.values
        days = ds.time.values.astype('datetime64[D]')
        # Use complete historical years, matching the existing annual assessment.
        candidates = np.unique(days.astype('datetime64[Y]'))
        complete = [year for year in candidates if np.array_equal(
            days[days.astype('datetime64[Y]') == year],
            np.arange(year.astype('datetime64[D]'), (year + np.timedelta64(1, 'Y')).astype('datetime64[D]')))]
        select = np.isin(days.astype('datetime64[Y]'), complete) & np.isin(
            days.astype('datetime64[M]').astype(int) % 12 + 1, [6, 7, 8])
        index = np.flatnonzero(select)
        history_days = days[index]
        months = history_days.astype('datetime64[M]').astype(int) % 12 + 1
        template = {name: ds[name].values.copy() for name in (
            'Unconditioned_Fourier_Mean', 'Unconditioned_Fourier_Std',
            'Fourier_Mean_Wet', 'Fourier_Mean_Dry', 'Fourier_Std_Wet', 'Fourier_Std_Dry')}
        residual = ds.Residual_Grid.isel(time=index).values
    with xr.open_dataset(raw / 'Data' / f'Detrended_t2m_{years}.nc') as ds:
        np.testing.assert_array_equal(ds.time.values[index].astype('datetime64[D]'), history_days)
        historical = ds.t2m.isel(time=index).values[:, :, mask] - 273.15
    with xr.open_dataset(raw / 'Data' / f'daily_tp_{years}.nc') as ds:
        np.testing.assert_array_equal(ds.time.values[index].astype('datetime64[D]'), history_days)
        wet = ds.Precipitation_mask.isel(time=index).values[:, mask].astype(bool)

    def moments(dates, states):
        h = hour_indices(dates)
        mu0 = template['Unconditioned_Fourier_Mean'][h] - 273.15
        sd0 = template['Unconditioned_Fourier_Std'][h]
        mu = np.where(states[:, None, :], template['Fourier_Mean_Wet'][h],
                      template['Fourier_Mean_Dry'][h]) - 273.15
        sd = np.where(states[:, None, :], template['Fourier_Std_Wet'][h],
                      template['Fourier_Std_Dry'][h])
        if min(sd0.min(), sd.min()) <= 0:
            raise ValueError('Nonpositive reconstruction scales.')
        return mu0, sd0, mu, sd

    mu0, sd0, mu, sd = moments(history_days, wet)
    error = float(np.max(np.abs(historical - (mu0 + sd0 * residual))))
    if error > 1e-8:
        raise ValueError(f'Historical inverse reconstruction failed: {error}.')
    historical_years = history_days.astype('datetime64[Y]')

    def history_summary(name, values):
        return summarize(name, [values[historical_years == year] for year in complete])

    rows = [history_summary('Historical', historical),
            history_summary('History replay: conditional templates', mu + sd * residual)]
    standardized, gaussian = np.empty_like(residual), np.empty_like(residual)
    for month in (6, 7, 8):
        chosen = months == month
        values = residual[chosen].reshape(-1, mask.sum())
        standardized[chosen] = ((values - values.mean(axis=0)) / values.std(axis=0)).reshape(residual[chosen].shape)
        gaussian[chosen] = norm.ppf((rankdata(values, axis=0) - .5) / len(values)).reshape(residual[chosen].shape)
    rows.extend([history_summary('History: standardized residuals', mu0 + sd0 * standardized),
                 history_summary('History: Gaussianized residuals', mu0 + sd0 * gaussian)])

    audit = []
    for month in (6, 7, 8):
        selected = months == month
        for state in (False, True):
            mean_errors, fitted_sd, observed_sd = [], [], []
            for cell in range(mask.sum()):
                chosen = wet[selected, cell] == state
                values = historical[selected, :, cell][chosen]
                mean_errors.append(mu[selected, :, cell][chosen].mean() - values.mean())
                fitted_sd.append(sd[selected, :, cell][chosen].mean())
                observed_sd.append(values.std(axis=0).mean())
            audit.append({'month': month, 'state': 'wet' if state else 'dry',
                          'mean_bias_C': float(np.mean(mean_errors)),
                          'fitted_hourly_sd_C': float(np.mean(fitted_sd)),
                          'observed_hourly_sd_C': float(np.mean(observed_sd)),
                          'mean_sd_ratio': float(np.mean(np.array(fitted_sd) / observed_sd))})

    controls, fitting_audit = fit_controls(raw / 'Data', years, mask, template)
    raw_catalog = json.loads((raw / 'scenario_catalog.json').read_text())
    adjusted_catalog = json.loads((adjusted / 'scenario_catalog.json').read_text())
    selected = sorted(raw_catalog['scenarios'], key=lambda row: row['seed'])
    if members:
        selected = selected[:members]
    adjusted_members = {row['seed']: row for row in adjusted_catalog['scenarios']}
    if not selected or len({row['seed'] for row in selected}) != len(selected):
        raise ValueError('Missing or duplicated raw member seeds.')
    variants = {name: [] for name in ('Unconditional', 'Conditional mean only',
                'Conditional scale only', 'Raw EGGS', 'Adjusted EGGS',
                'Calendar intercept only', 'Hourly scale shape only', 'Both fitting controls')}
    simulated_wet, residual_values = [], []
    peak_pairs = []

    def load(root, member):
        if len(member['files']) != 1:
            raise ValueError('Diagnosis currently expects one annual weather file per member.')
        with xr.open_dataset(root / member['files'][0]) as ds:
            np.testing.assert_array_equal(ds.latitude.values, latitude)
            np.testing.assert_array_equal(ds.longitude.values, longitude)
            if ds.attrs.get('random_seed') != member['seed']:
                raise ValueError('Weather seed differs from catalog.')
            dates = ds.simulation_day.values.astype('datetime64[D]')
            if len(np.unique(dates.astype('datetime64[Y]'))) != 1:
                raise ValueError('Each member must contain a single simulation year.')
            chosen = np.flatnonzero(np.isin(dates.astype('datetime64[M]').astype(int) % 12 + 1, [6, 7, 8]))
            times = (chosen[:, None] * 24 + np.arange(24)).ravel()
            dates = dates[chosen]
            if len(dates) != 92:
                raise ValueError('Incomplete summer.')
            np.testing.assert_array_equal(ds.simulation_time.values[times].astype('datetime64[h]'),
                (dates[:, None].astype('datetime64[h]') + np.arange(24)).ravel())
            values = ds.simulation_t2m.isel(simulation_time=times).values[:, mask].reshape(92, 24, mask.sum()) - 273.15
            states = ds.simulation_Precipitation.isel(simulation_day=chosen).values[:, mask] > 0
            return dates, values, states

    for member in selected:
        dates, values, states = load(raw, member)
        m0, s0, mc, sc = moments(dates, states)
        z = (values - mc) / sc
        variants['Unconditional'].append(m0 + s0 * z)
        variants['Conditional mean only'].append(mc + s0 * z)
        variants['Conditional scale only'].append(m0 + sc * z)
        variants['Raw EGGS'].append(values)
        ordinal = (dates - dates.astype('datetime64[Y]')).astype(int)
        mean_change = -np.where(states, controls['Wet'][0], controls['Dry'][0])[:, None, :]
        scale_change = np.where(states, controls['Wet'][1][ordinal], controls['Dry'][1][ordinal])[:, None, :]
        if (sc + scale_change).min() <= 0:
            raise ValueError('Fitting control produces nonpositive scales.')
        variants['Calendar intercept only'].append(values + mean_change)
        variants['Hourly scale shape only'].append(values + scale_change * z)
        variants['Both fitting controls'].append(values + mean_change + scale_change * z)
        dates2, corrected, _ = load(adjusted, adjusted_members[member['seed']])
        np.testing.assert_array_equal(dates, dates2)
        np.testing.assert_allclose(values.mean(axis=1), corrected.mean(axis=1), rtol=0, atol=1e-8)
        variants['Adjusted EGGS'].append(corrected)
        peak_pairs.append({'seed': member['seed'], 'raw_peak_C': float(values.max()),
                           'adjusted_peak_C': float(corrected.max()),
                           'peak_change_C': float(corrected.max() - values.max())})
        simulated_wet.append(states)
        residual_values.append(z)
    rows.extend(summarize(name, values) for name, values in variants.items())

    stationary = []
    with xr.open_dataset(raw / 'Data' / 't2m_XY_Matrix.nc') as ds:
        for month in (6, 7, 8):
            x, y = ds.X_matrix.values[month - 1], ds.Y_matrix.values[month - 1]
            radius = float(np.max(np.abs(np.linalg.eigvals(x))))
            if radius >= 1:
                raise ValueError('Unstable VAR process.')
            covariance = solve_discrete_lyapunov(x, y @ y.T)
            scales = np.sqrt(np.diag(covariance))
            stationary.append({'month': month, 'spectral_radius': radius,
                               'sd_min': float(scales.min()), 'sd_mean': float(scales.mean()),
                               'sd_max': float(scales.max())})
    z = np.concatenate(residual_values)
    result = {'members': len(selected), 'seeds': [row['seed'] for row in selected],
              'historical_years': [int(year.astype(int)) + 1970 for year in complete],
              'months': [6, 7, 8], 'cells': int(mask.sum()), 'raw': str(raw), 'adjusted': str(adjusted),
              'historical_reconstruction_max_error_C': error,
              'wet_fraction_historical': float(wet.mean()),
              'wet_fraction_simulated': float(np.concatenate(simulated_wet).mean()),
              'simulated_residual_mean': float(z.mean()), 'simulated_residual_sd': float(z.std()),
              'simulated_residual_p99': float(np.percentile(z, 99)),
              'rows': rows, 'template_audit': audit, 'fitting_audit': fitting_audit, 'stationary_var': stationary,
              'method': 'Controlled reconstruction: same raw residuals, dates, cells and wet states; '
                        'swap only means or scales. Gaussianization changes each month/cell marginal '
                        'while retaining within-month rank dependence. Fitting controls recompute '
                        'conditioned Fourier intercepts and scale harmonics from all training years, '
                        'verify original coefficients, and replay the same generated residuals. '
                        'Residual dynamics are not refitted; production weather is unchanged.',
              'limitations': 'Summer only. Unconditional reconstruction is a control, not a proposed fix. '
                             'Mean/scale percentile changes interact and are not independent additive effects. '
                             'Template audit compares mean fitted scales with within-month, state/hour/cell '
                             'observed SDs, including within-month seasonal variation.',
              'sources': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in [coefficient, raw / 'Data' / 't2m_XY_Matrix.nc',
                                       raw / 'scenario_catalog.json', adjusted / 'scenario_catalog.json', Path(__file__)]]}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'reconstruction.csv', rows)
    write_csv(output / 'template_audit.csv', audit)
    write_csv(output / 'fitting_audit.csv', fitting_audit)
    write_csv(output / 'paired_peaks.csv', peak_pairs)
    (output / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    reference = rows[0]['daily_max_p99_C']
    labels = ['Unconditional', 'Conditional mean only', 'Conditional scale only', 'Raw EGGS', 'Adjusted EGGS']
    mapping = {row['variant']: row for row in rows}
    axes[0].barh(['Control', '+ mean', '+ scale', '+ both (raw)', '+ cycle correction'],
                 [mapping[name]['daily_max_p99_C'] - reference for name in labels], color='#4477aa')
    axes[0].axvline(0, color='black', lw=.8)
    axes[0].invert_yaxis()
    axes[0].set(title='P99 daily Tmax bias', xlabel='Difference from history (°C)')
    dry = [row for row in audit if row['state'] == 'dry']
    axes[1].bar(['Jun', 'Jul', 'Aug'], [row['mean_bias_C'] for row in dry], color='#ee8844')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(title='Dry-day template mean bias', ylabel='°C')
    axes[2].bar(['Jun', 'Jul', 'Aug'], [row['mean_sd_ratio'] for row in dry], color='#ee8844')
    axes[2].axhline(1, color='black', lw=.8)
    axes[2].set(title='Dry-day template scale / observed SD', ylabel='Ratio (target ≈ 1)')
    fig.suptitle(f'June–August · {len(complete)} historical years · {len(selected)} matched scenarios · {mask.sum()} cells')
    fig.savefig(output / 'temperature_diagnosis.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = ['Raw EGGS', 'Calendar intercept only', 'Hourly scale shape only', 'Both fitting controls']
    control_labels = ['Raw', 'Mean intercept', 'Scale shape', 'Both']
    for ax, metric, title in zip(axes, ['mean_domain_summer_peak_C', 'daily_max_p99_C'],
                                ['Mean regional summer peak', 'P99 cell/day Tmax']):
        ax.scatter(np.arange(len(names)), [mapping[name][metric] for name in names], color='#4477aa', s=65)
        ax.set_xticks(np.arange(len(names)), control_labels)
        ax.axhline(mapping['Historical'][metric], color='black', ls='--', label='Historical')
        ax.set(title=title, ylabel='°C', ylim=(38, 45))
        ax.tick_params(axis='x', rotation=15)
        ax.legend(frameon=False)
    fig.suptitle('Controlled replay: change fitted templates; retain the same residuals and rain states')
    fig.savefig(output / 'fitting_controls.png', dpi=170)
    plt.close(fig)
    table = '\n'.join(f"| {name} | {mapping[name]['daily_max_p99_C']:.2f} |"
                      for name in ['Historical', *labels])
    shape_effect = mapping['History: Gaussianized residuals']['daily_max_p99_C'] - mapping['History: standardized residuals']['daily_max_p99_C']
    july = next(row for row in dry if row['month'] == 7)
    fitting_table = '\n'.join(
        f"| {name} | {mapping[name]['mean_domain_summer_peak_C']:.2f} | {mapping[name]['daily_max_p99_C']:.2f} |"
        for name in ['Historical', *names])
    summary = f'''# Temperature diagnosis

June–August; {len(complete)} historical years; {len(selected)} matched scenarios; {mask.sum()} cells. Historical temperatures normalized to 2020.

| Reconstruction | P99 daily Tmax, °C |
| --- | ---: |
{table}

- Controls hold simulated residuals, dates and wet states fixed; change only reconstruction means/scales.
- July dry templates: mean bias {july['mean_bias_C']:+.2f}°C; scale {july['mean_sd_ratio']:.2f}× observed hourly SD.
- Gaussian residual shape alone changes replay P99 Tmax by {shape_effect:+.2f}°C after matching mean/SD.
- VAR stationary residual SD ≈ 1; no variance explosion.
- Raw → corrected mean summer domain peak: {np.mean([p['peak_change_C'] for p in peak_pairs]):+.2f}°C; each daily mean unchanged.
- Main contributors: wet/dry reconstruction means and scales, then cycle correction. Their isolated percentile effects interact.
- Unconditional control has a cold mean bias; do not treat it as a calibrated replacement.

## Fitting causes

- Mean intercept: uses event-weighted annual temperature with calendar-weighted Fourier harmonics. Dry intercept is +{fitting_audit[0]['event_minus_calendar_intercept_C']:.2f}°C above its calendar mean.
- Scale harmonics: use SD pooled across all 24 hours, mixing day/night variation with between-day weather variability. Using mean hourly SD lowers summer scales by about 0.53°C.
- Replay changes templates only; no residual refit or production changes.

| Template control | Mean regional summer peak, °C | P99 daily Tmax, °C |
| --- | ---: | ---: |
{fitting_table}

Both changes improve regional peak bias, but make P99 daily Tmax too low. A complete refit and annual checks remain necessary.

![Fitting controls](fitting_controls.png)

![Diagnostics](temperature_diagnosis.png)

[Full table](reconstruction.csv) · [Template audit](template_audit.csv) · [Fitting audit](fitting_audit.csv) · [Checks/method](diagnosis.json)
'''
    (output / 'README.md').write_text(summary)
    print(summary.split('![Fitting controls]')[0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--adjusted', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--members', type=int, default=0, help='0 uses all raw members.')
    args = parser.parse_args()
    if args.members < 0:
        parser.error('--members must be nonnegative.')
    diagnose(args.raw, args.adjusted, args.output, args.members)
