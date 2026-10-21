"""Adapt the local ERA5-Land archive to EGGS-WG's daily/24-hour training layout."""

import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

ADAPTER_VERSION = 1
NEGATIVE_TOLERANCE_MM = 0.001
MISMATCH_TOLERANCE_MM = 0.01


def source_path(params, year, month):
    template = params.get('Input Filename Template', 'era5_land_hourly_{year}_{month:02d}-{month:02d}.nc')
    return Path(params['ERA5 Directory']) / template.format(year=year, month=month)


def training_months(params):
    start, end = params['Training Start Year'], params['Training End Year']
    excluded = set(params.get('Excluded Training Years', []))
    years = [year for year in range(start, end + 1) if year not in excluded]
    if len(years) < 3:
        raise ValueError('At least three training years are required; use a multi-decade archive for fitting.')
    return [(year, month) for year in years for month in range(1, 13)]


def training_configuration(params):
    configuration = {key: params.get(key) for key in (
        'ERA5 Directory', 'Input Filename Template', 'Variables', 'Training Start Year',
        'Training End Year', 'Excluded Training Years', 'Lat Bounds', 'Lon Bounds',
        'Temperature Units', 'Dewpoint Units', 'Precipitation Units', 'Detrend Reference Year')}
    if params.get('Boundary File'):
        configuration['Boundary File'] = str(params['Boundary File'])
    return configuration


def input_signature(params):
    """Validate the month inventory before fitting and identify incompatible caches."""
    months = training_months(params)
    missing = [str(source_path(params, *month)) for month in months if not source_path(params, *month).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} training month(s) missing, beginning with {missing[0]}. '
            'Provide the missing data or explicitly exclude the affected training year; no months are filled.')
    paths = {source_path(params, year, month) for year, month in months}
    for year, month in months:
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        paths.add(source_path(params, next_year, next_month))
    configuration = training_configuration(params)
    if params.get('Boundary File'):
        boundary = Path(params['Boundary File'])
        configuration['Boundary File'] = str(boundary)
        for suffix in ('.shp', '.shx', '.dbf', '.prj'):
            component = boundary.with_suffix(suffix)
            if not component.is_file():
                raise FileNotFoundError(f'Boundary component missing: {component}')
            paths.add(component)
    payload = {'adapter_version': ADAPTER_VERSION, 'configuration': configuration,
               'source_files': [{'path': str(path), 'size': path.stat().st_size,
                                 'mtime_ns': path.stat().st_mtime_ns}
                                if path.is_file() else {'path': str(path), 'missing': True}
                                for path in sorted(paths)]}
    payload['signature'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def select_region(dataset, params):
    latitude, longitude = dataset['latitude'].values, dataset['longitude'].values
    north, south = max(params['Lat Bounds']), min(params['Lat Bounds'])
    east, west = max(params['Lon Bounds']), min(params['Lon Bounds'])
    y = np.flatnonzero((latitude <= north + 1e-6) & (latitude >= south - 1e-6))
    x = np.flatnonzero((longitude <= east + 1e-6) & (longitude >= west - 1e-6))
    if not len(y) or not len(x):
        raise ValueError('The requested bounds do not contain any source grid-cell centres.')
    return dataset.isel(latitude=y, longitude=x).sortby('latitude', ascending=False).sortby('longitude')


def units_for(dataset, name, params):
    key, default = {'t2m': ('Temperature Units', 'degC'), 'd2m': ('Dewpoint Units', 'K'),
                    'tp': ('Precipitation Units', 'mm')}[name]
    units = dataset[name].attrs.get('units') or params.get(key)
    if units is None and dataset.attrs.get('converted_by') == 'gen_mk2_netcdf.py' and name != 'd2m':
        units = default
    if units is None:
        raise ValueError(f'{name} has no unit metadata. Set {key} explicitly.')
    if name == 'tp':
        if units not in ('m', 'mm'):
            raise ValueError(f'Unsupported precipitation units: {units}')
    elif units not in ('K', 'k', 'degC', 'C', 'degrees_Celsius'):
        raise ValueError(f'Unsupported {name} units: {units}')
    return units


def read_month(params, year, month):
    path = source_path(params, year, month)
    names = params['Variables']
    with xr.open_dataset(path) as source:
        missing = set(names) - set(source.data_vars)
        if missing:
            raise ValueError(f'{path} lacks {sorted(missing)}. Request only available variables or supply those fields.')
        data = select_region(source[names], params)
        first = np.datetime64(f'{year}-{month:02d}-01T00', 'h')
        count = calendar.monthrange(year, month)[1] * 24
        expected = first + np.arange(count) * np.timedelta64(1, 'h')
        if not np.array_equal(data['time'].values, expected):
            raise ValueError(f'{path} must contain every hourly timestamp of its calendar month exactly once.')
        values = {}
        for name in names:
            if data[name].dims != ('time', 'latitude', 'longitude'):
                raise ValueError(f'{path}: {name} requires (time, latitude, longitude) dimensions.')
            unit = units_for(source, name, params)
            array = data[name].values.astype(np.float64)
            if name == 'tp' and unit == 'm':
                array *= 1000
            elif name != 'tp' and unit not in ('K', 'k'):
                array += 273.15
            values[name] = array.reshape(count // 24, 24, *array.shape[1:])
        return values, data['latitude'].values.copy(), data['longitude'].values.copy()


def next_midnight(params, year, month, latitude, longitude):
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    path = source_path(params, next_year, next_month)
    shape = (len(latitude), len(longitude))
    if not path.is_file():
        return np.full(shape, np.nan)
    with xr.open_dataset(path) as source:
        data = select_region(source[['tp']], params)
        if not (np.array_equal(data.latitude.values, latitude) and np.array_equal(data.longitude.values, longitude)):
            raise ValueError(f'{path}: boundary grid differs from training grid.')
        timestamp = np.datetime64(f'{next_year}-{next_month:02d}-01T00', 'ns')
        try:
            values = data['tp'].sel(time=timestamp).values.astype(float)
        except KeyError:
            return np.full(shape, np.nan)
        return values * (1000 if units_for(source, 'tp', params) == 'm' else 1)


def daily_precipitation(hourly, endpoint):
    """D's rainfall uses cumulative values at D 01 UTC through D+1 00 UTC."""
    cumulative = np.concatenate((hourly[:, 1:], np.concatenate((hourly[1:, :1], endpoint[None, None]), axis=0)), axis=1)
    increments = np.diff(cumulative, axis=1, prepend=np.zeros_like(cumulative[:, :1]))
    finite = np.isfinite(cumulative).all(axis=1)
    negative = np.any(increments < -NEGATIVE_TOLERANCE_MM, axis=1)
    clipped = np.maximum(increments, 0).sum(axis=1)
    mismatch = np.abs(clipped - cumulative[:, -1]) > MISMATCH_TOLERANCE_MM
    valid = finite & ~negative & ~mismatch
    return np.where(valid, np.maximum(cumulative[:, -1], 0), np.nan), {
        'missing_precipitation': ~finite, 'negative_precipitation_increment': negative,
        'precipitation_accumulation_mismatch': mismatch,
    }


def prepare_inputs(params, directory, inventory):
    """Write paired complete training days; never interpolate missing observations."""
    months = training_months(params)
    temperature_names = [name for name in params['Variables'] if name != 'tp']
    temperatures = {name: [] for name in temperature_names}
    rain, dates, dropped, quality = [], [], [], []
    latitude = longitude = land = None
    for year, month in months:
        values, lat, lon = read_month(params, year, month)
        current_land = np.any(np.isfinite(values['tp']), axis=(0, 1))
        if params.get('Boundary File'):
            from ai4water_eggs.geography import boundary_mask
            current_land &= boundary_mask(params['Boundary File'], lat, lon)
        for name in temperature_names:
            current_land &= np.any(np.isfinite(values[name]), axis=(0, 1))
        if land is None:
            latitude, longitude, land = lat, lon, current_land
            if not land.any():
                raise ValueError('No land cells have both temperature and precipitation in the selected region.')
        elif not (np.array_equal(lat, latitude) and np.array_equal(lon, longitude) and np.array_equal(current_land, land)):
            raise ValueError(f'{year}-{month:02d}: coordinates or available land-cell mask changed.')
        endpoint = next_midnight(params, year, month, latitude, longitude)
        daily, flags = daily_precipitation(values['tp'], endpoint)
        good = np.isfinite(daily[:, land]).all(axis=1)
        for name in temperature_names:
            finite = np.isfinite(values[name][:, :, land]).all(axis=(1, 2))
            flags['missing_' + name] = ~finite
            good &= finite
        first = np.datetime64(f'{year}-{month:02d}-01', 'D')
        month_dates = first + np.arange(len(daily)) * np.timedelta64(1, 'D')
        for index in np.flatnonzero(~good):
            reasons = [key for key, flag in flags.items() if np.any(flag[index]) if
                       flag.ndim == 1 or np.any(flag[index][land])]
            dropped.append({'date': str(month_dates[index]), 'reasons': reasons})
        dates.append(month_dates[good])
        rain.append(np.where(land, daily[good], np.nan))
        for name in temperature_names:
            temperatures[name].append(np.where(land, values[name][good], np.nan))
        quality.append({'month': f'{year}-{month:02d}', 'retained_days': int(good.sum()),
                        'dropped_days': int((~good).sum())})
    days = np.concatenate(dates).astype('datetime64[ns]')
    if len(days) < 365 * 2:
        raise ValueError('Too few complete paired training days remain after quality checks.')
    coords = {'time': days, 'latitude': latitude, 'longitude': longitude}
    period = f"{params['Training Start Year']}_{params['Training End Year']}"
    attributes = {'daily_interval': '[00:00 UTC, next-day 00:00 UTC)',
                  'training_signature': inventory['signature'], 'input_adapter': 'ai4water_eggs.inputs'}
    for name in temperature_names:
        dataset = xr.Dataset({name: (('time', 'step', 'latitude', 'longitude'), np.concatenate(temperatures[name]))},
                             coords={**coords, 'step': np.arange(24)}, attrs=attributes)
        dataset[name].attrs['units'] = 'K'
        dataset.to_netcdf(directory / f'{name}_{period}.nc')
        dataset.close()
    precipitation = np.concatenate(rain)
    dataset = xr.Dataset({'tp': (('time', 'latitude', 'longitude'), precipitation),
                          'Precipitation_mask': (('time', 'latitude', 'longitude'), precipitation >= 1.0)},
                         coords=coords, attrs=attributes)
    dataset['tp'].attrs.update(units='mm', description='UTC-day total from next-day 00 UTC ERA5-Land accumulation')
    dataset.to_netcdf(directory / f'daily_tp_{period}.nc')
    dataset.close()
    report = {'training_start': str(days[0].astype('datetime64[D]')),
              'training_end': str(days[-1].astype('datetime64[D]')), 'retained_days': len(days),
              'land_cells': int(land.sum()), 'excluded_years': params.get('Excluded Training Years', []),
              'dropped_days': dropped, 'months': quality, **inventory}
    (directory / 'input_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if params.get('Boundary File'):
        from ai4water_eggs.geography import write_boundary
        write_boundary(params['Boundary File'], directory.parent / 'boundary.geojson')
    print(f"Prepared {len(days)} training days, {int(land.sum())} land cells; dropped {len(dropped)} invalid days.", flush=True)
    return report
