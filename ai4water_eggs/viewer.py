"""Serve or export an offline interactive inspector for EGGS-WG simulations."""

import argparse
import base64
import calendar
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import gzip
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, urlparse

import numpy as np
import xarray as xr

BASE = Path(__file__).resolve().parents[1]
ASSETS = BASE / 'viewer'
DEFAULT_ROOT = BASE / 'outputs'

@lru_cache(maxsize=1)
def plotly_javascript():
    # The full bundle initializes unused map/worker code that can fail when a
    # file viewer blocks blob URLs. Cartesian includes our heatmaps and charts.
    return (ASSETS / 'vendor' / 'plotly-cartesian-3.7.0.min.js').read_text(encoding='utf-8')


def assessment_path(root):
    """Latest completed assessment, while retaining earlier report files."""
    root = Path(root)
    candidates = [root / 'Assessment/index.html', *root.glob('*/Assessment/index.html')]
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None


def numbers(values, digits=3):
    values = np.asarray(values)
    return np.where(np.isfinite(values), np.round(values, digits), None).tolist()


def discover_runs(root):
    result = []
    for directory in sorted(Path(root).glob('*/Simulation/Simulation_*')):
        if (directory.parents[1] / '.in_progress').is_file():
            continue
        paths = sorted(directory.glob('*.nc'))
        paths = [path for path in paths if not path.name.startswith('tmp_')]
        if paths:
            identifier = directory.relative_to(root).as_posix()
            number = int(directory.name.removeprefix('Simulation_'))
            experiment = directory.parents[1].name
            result.append({'id': identifier, 'label': f'{experiment} · Scenario {number}',
                           'experiment': experiment, 'number': number, 'files': len(paths)})
    return sorted(result, key=lambda row: (row['experiment'], row['number']))


def discover_catalogs(root, identifiers=None):
    allowed = set(identifiers) if identifiers is not None else {row['id'] for row in discover_runs(root)}
    result = []
    for path in sorted(Path(root).glob('*/scenario_catalog.json')):
        catalog = json.loads(path.read_text())
        catalog['scenarios'] = [row for row in catalog['scenarios'] if row['id'] in allowed]
        if catalog['scenarios']:
            catalog['source_scenario_count'] = catalog['scenario_count']
            catalog['scenario_count'] = len(catalog['scenarios'])
            catalog['validation_available'] = (path.parent / 'Validation/report.html').is_file()
            result.append(catalog)
    return result


def climatology(directory, years):
    rain_path = directory / f'daily_tp_{years}.nc'
    temperature_path = directory / f'Detrended_t2m_{years}.nc'
    if not (rain_path.is_file() and temperature_path.is_file()):
        return None
    return cached_climatology(directory, years, rain_path.stat().st_mtime_ns, temperature_path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def cached_climatology(directory, years, rain_stamp, temperature_stamp):
    rain_path = directory / f'daily_tp_{years}.nc'
    temperature_path = directory / f'Detrended_t2m_{years}.nc'
    with xr.open_dataset(rain_path) as source:
        rain = source.tp.values
        dates = source.time.values.astype('datetime64[D]')
    with xr.open_dataset(temperature_path) as source:
        temperature = source.t2m.mean('step').values - 273.15
        reference = source.attrs.get('detrend_reference_date', '')
    month_index = dates.astype('datetime64[M]')
    monthly_p, monthly_t, count = [], [], []
    for month in range(1, 13):
        totals, means = [], []
        for period in np.unique(month_index):
            year, number = map(int, str(period).split('-'))
            if number != month:
                continue
            selected = month_index == period
            if selected.sum() == calendar.monthrange(year, month)[1]:
                totals.append(rain[selected].sum(axis=0))
                means.append(temperature[selected].mean(axis=0))
        count.append(len(totals))
        if totals:
            monthly_p.append(np.mean(totals, axis=0))
            monthly_t.append(np.mean(means, axis=0))
        else:
            monthly_p.append(np.full(rain.shape[1:], np.nan))
            monthly_t.append(np.full(rain.shape[1:], np.nan))
    return {'rain': numbers(monthly_p), 'temperature': numbers(monthly_t),
            'complete_months': count, 'reference_date': reference}


def load_run(root, identifier):
    runs = {item['id'] for item in discover_runs(root)}
    if identifier not in runs:
        raise ValueError('Unknown simulation.')
    directory = Path(root) / identifier
    paths = [path for path in sorted(directory.glob('*.nc')) if not path.name.startswith('tmp_')]
    chunks = []
    for path in paths:
        with xr.open_dataset(path) as source:
            chunks.append(source.load())
    dataset = chunks[0] if len(chunks) == 1 else xr.combine_by_coords(chunks, combine_attrs='override')
    days = dataset.simulation_day.values.astype('datetime64[D]')
    hours = dataset.simulation_time.values.astype('datetime64[h]')
    expected = (days[:, None].astype('datetime64[h]') + np.arange(24)[None, :] * np.timedelta64(1, 'h')).ravel()
    if not np.array_equal(expected, hours):
        raise ValueError('Viewer expects 24 UTC hours aligned to each daily rainfall date.')
    temperature = dataset.simulation_t2m.values - 273.15
    daily = temperature.reshape(len(days), 24, dataset.sizes['latitude'], dataset.sizes['longitude'])
    rain = dataset.simulation_Precipitation.values
    mask = np.isfinite(rain).all(axis=0) & np.isfinite(temperature).all(axis=0)
    experiment = directory.parents[1]
    config_path = experiment / 'run_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    report_path = experiment / 'Data/input_report.json'
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    boundary_path = experiment / 'boundary.geojson'
    boundary = json.loads(boundary_path.read_text()) if boundary_path.is_file() else None
    years = f"{config.get('Training Start Year', 1990)}_{config.get('Training End Year', 2025)}"
    calibration = []
    seasonal_support = []
    temperature_fit = experiment / 'Data' / f't2m_PreSim_Coeff_{years}.nc'
    if temperature_fit.is_file():
        with xr.open_dataset(temperature_fit) as coefficients:
            seasonal_support = json.loads(coefficients.attrs.get('seasonal_support_report', '[]'))
    for prefix in ('O_Omega', 'A_Omega'):
        path = experiment / 'Data' / f'{prefix}_{years}.nc'
        if path.is_file():
            with xr.open_dataset(path) as coefficients:
                calibration.extend(json.loads(coefficients.attrs.get('calibration_report', '[]')))
    # Suppress empty coastal cells without changing the fixed geographic grid.
    mean = np.divide(np.nansum(daily, axis=1), np.isfinite(daily).sum(axis=1),
                     out=np.full_like(rain, np.nan), where=np.isfinite(daily).sum(axis=1) > 0)
    minimum = np.where(mask, np.min(np.where(np.isfinite(daily), daily, np.inf), axis=1), np.nan)
    maximum = np.where(mask, np.max(np.where(np.isfinite(daily), daily, -np.inf), axis=1), np.nan)
    payload = {'id': identifier, 'title': experiment.name.replace('_', ' '),
               'days': days.astype(str).tolist(), 'hours': hours.astype(str).tolist(),
               'latitude': numbers(dataset.latitude.values, 6), 'longitude': numbers(dataset.longitude.values, 6),
               'mask': mask.tolist(), 'boundary': boundary,
               'metrics': {'rain': numbers(rain), 'tmean': numbers(mean),
                           'tmin': numbers(minimum), 'tmax': numbers(maximum)},
               'hourly_temperature': numbers(temperature),
               'climatology': climatology(experiment / 'Data', years),
               'metadata': {key: str(value) for key, value in dataset.attrs.items()},
               'quality': {'retained_days': report.get('retained_days'),
                           'dropped_days': report.get('dropped_days', []),
                           'excluded_years': report.get('excluded_years', []),
                           'land_cells': int(mask.sum()), 'calibration': calibration,
                           'seasonal_support': seasonal_support}}
    dataset.close()
    return payload


def export_viewer(root, output, identifiers=None):
    runs = discover_runs(root)
    if identifiers:
        runs = [item for item in runs if item['id'] in identifiers]
    if not runs:
        raise ValueError('No completed simulations found to export.')
    data = {'runs': runs, 'catalogs': discover_catalogs(root, [item['id'] for item in runs])}
    if len(runs) == 1:
        data['data'] = {runs[0]['id']: load_run(root, runs[0]['id'])}
    else:
        # Compress each scenario separately so the browser only inflates the
        # selected one, rather than parsing 30 full weather cubes on startup.
        data['compressed_data'] = {}
        for item in runs:
            payload = json.dumps(load_run(root, item['id']), separators=(',', ':'), allow_nan=False).encode()
            data['compressed_data'][item['id']] = base64.b64encode(gzip.compress(payload, mtime=0)).decode('ascii')
            print(f'Embedding {item["label"]}', flush=True)
    content = (ASSETS / 'index.html').read_text()
    content = content.replace('<link rel="stylesheet" href="/styles.css">', '<style>' + (ASSETS / 'styles.css').read_text() + '</style>')
    content = content.replace('<script src="/plotly.min.js"></script>', '<script>' + plotly_javascript().replace('</script', '<\\/script') + '</script>')
    embedded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    application = (ASSETS / 'app.js').read_text().replace('</script', '<\\/script')
    content = content.replace('<script src="/app.js"></script>', '<script>window.EGGS_DATA=' + embedded + ';</script><script>' + application + '</script>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    print(f'Offline viewer written: {output.resolve()} ({output.stat().st_size / 1024**2:.1f} MiB)')


def serve(root, host='127.0.0.1', port=8765):
    root = Path(root).resolve()

    @lru_cache(maxsize=4)
    def encoded_run(identifier, stamp):
        return json.dumps(load_run(root, identifier), separators=(',', ':'), allow_nan=False).encode()

    # Browsers pre-open idle sockets. Accept them independently, while keeping
    # NetCDF access/cache work serialized across actual requests.
    request_lock = RLock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with request_lock:
                self.respond()

        def respond(self):
            request = urlparse(self.path)
            try:
                if request.path == '/api/runs':
                    body = json.dumps(discover_runs(root)).encode()
                    content_type = 'application/json'
                elif request.path == '/api/ensembles':
                    body = json.dumps(discover_catalogs(root), separators=(',', ':'), allow_nan=False).encode()
                    content_type = 'application/json'
                elif request.path == '/validation':
                    experiment = parse_qs(request.query).get('experiment', [''])[0]
                    allowed = {row['experiment'] for row in discover_runs(root)}
                    if experiment not in allowed:
                        raise ValueError('Unknown experiment.')
                    report = root / experiment / 'Validation/report.html'
                    if not report.is_file():
                        self.send_error(404, 'Run ensemble.py validate for this experiment first.')
                        return
                    body, content_type = report.read_bytes(), 'text/html'
                elif request.path == '/assessment':
                    report = assessment_path(root)
                    if report is None:
                        self.send_error(404, 'No historical-envelope assessment is available.')
                        return
                    body, content_type = report.read_bytes(), 'text/html'
                elif request.path == '/api/run':
                    identifier = parse_qs(request.query).get('id', [''])[0]
                    allowed = {item['id'] for item in discover_runs(root)}
                    if identifier not in allowed:
                        raise ValueError('Unknown simulation.')
                    stamp = max(path.stat().st_mtime_ns for path in (root / identifier).glob('*.nc'))
                    body, content_type = encoded_run(identifier, stamp), 'application/json'
                elif request.path == '/plotly.min.js':
                    body, content_type = plotly_javascript().encode(), 'text/javascript'
                elif request.path in ('/', '/index.html', '/styles.css', '/app.js'):
                    name = 'index.html' if request.path == '/' else request.path[1:]
                    body = (ASSETS / name).read_bytes()
                    if name == 'index.html' and assessment_path(root) is not None:
                        link = b'<p><a id="assessment-link" class="validation-link" href="/assessment" target="_blank" rel="noopener">Historical envelope assessment &#8599;</a></p>'
                        body = body.replace(b'<!--ASSESSMENT-->', link)
                    content_type = {'html': 'text/html', 'css': 'text/css', 'js': 'text/javascript'}[name.rsplit('.', 1)[1]]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type + '; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError) as error:
                self.send_error(400, str(error))

    server = ThreadingHTTPServer((host, port), Handler)
    print(f'EGGS-WG viewer: http://{host}:{port} · {root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--export', type=Path, help='Write one standalone HTML file with embedded data and charts.')
    parser.add_argument('--run', action='append', help='Simulation ID to include in the offline export; repeatable.')
    parser.add_argument('--experiment', help='Export every scenario in this experiment.')
    args = parser.parse_args()
    if args.export:
        identifiers = args.run
        if args.experiment:
            identifiers = [row['id'] for row in discover_runs(args.root.resolve()) if row['experiment'] == args.experiment]
            if not identifiers:
                parser.error('No completed scenarios in this experiment.')
        export_viewer(args.root.resolve(), args.export, identifiers)
    else:
        serve(args.root, args.host, args.port)


if __name__ == '__main__':
    main()
