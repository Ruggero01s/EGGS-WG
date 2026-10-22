"""Run/resume paired baseline, temperature and pooled-rain trials by study area.

uv run python scripts/run_transfer_trials.py --cases tunisia algeria egypt --jobs 2
Case baseline configurations live in configs/transfer_CASE_baseline.yaml.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'outputs/transfer_n1000'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, logfile):
    with logfile.open('a') as stream:
        subprocess.run([sys.executable, '-u', str(REPO/'ensemble.py'), *map(str, args)],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)


def complete(root, count, expected=None):
    catalog = root/'scenario_catalog.json'
    if (root/'.in_progress').exists() or not catalog.is_file():
        return False
    data = json.loads(catalog.read_text())
    if expected is not None:
        saved = json.loads((root/'run_config.json').read_text())
        keys = ['Variables', 'Training Start Year', 'Training End Year', 'Excluded Training Years',
                'Detrend Reference Year', 'ERA5 Directory', 'Lat Bounds', 'Lon Bounds', 'Boundary File',
                'Simulation Start Date', 'Simulation End Date', 'Random Seed', 'Variable Adjust',
                'Non-stationary Mode', 'Temperature Seasonal Fit', 'Temperature Residual Conditioning',
                'Temperature VAR Fit', 'Weather Output Precision', 'NetCDF Compression']
        if any(saved.get(key) != expected.get(key) for key in keys):
            raise ValueError(f'{root}: configuration changed; use a fresh experiment instead of resuming these files.')
    return data['scenario_count'] == count and all((root/path).is_file()
        for row in data['scenarios'] for path in row['files'])


def run_case(case):
    root = ROOT/case
    root.mkdir(parents=True, exist_ok=True)
    # Advisory locks survive crashes safely and prevent simultaneous writers.
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = root/'run.log'

        def stage(name):
            value = {'case': case, 'stage': name, 'updated_utc': datetime.now(timezone.utc).isoformat()}
            (root/'status.json').write_text(json.dumps(value, indent=2)+'\n')
            print(f'{case}: {name}', flush=True)

        try:
            config = REPO/'configs'/f'transfer_{case}_baseline.yaml'
            parameters = yaml.safe_load(config.read_text())
            baseline, source = root/'baseline_n1000', root/'temperature_n1050'
            raw, calibration, corrected = root/'temperature_n1000', root/'calibration_n50', root/'temperature_rain_n1000'
            stage('Fit and generate baseline, N=1000')
            if not complete(baseline, 1000, parameters):
                command(['generate', config], log)
            stage('Refit temperature templates and generate N=1050')
            temperature_config = REPO/'configs'/f'transfer_{case}_temperature.yaml'
            updated = {**parameters, 'Simulation name': source.name, 'Number of Simulations': 1050,
                       'Variable Adjust': False, 'Temperature Seasonal Fit': 'calendar-hourly',
                       'Reuse Training From': str(baseline), 'Trial Label': 'Temperature fit'}
            temperature_config.write_text(yaml.safe_dump(updated, sort_keys=False))
            if not complete(source, 1050, updated):
                command(['generate', temperature_config], log)
            stage('Separate 1000 evaluation / 50 calibration members')
            for destination, first, last, label in [(raw, 1, 1000, 'Temperature fit'),
                    (calibration, 1001, 1050, 'Rainfall calibration only')]:
                if not destination.exists():
                    command(['subset', source, '--members', first, last, '--output', destination, '--label', label], log)
                elif not complete(destination, last-first+1):
                    command(['analyze', destination, '--scales', 1, 3, 6, 7, 12], log)
                    (destination/'.in_progress').unlink(missing_ok=True)
            stage('Fit shared rainfall mapping on 50 separate seeds')
            mapping = root/'rainfall_mapping.json'
            if not mapping.exists():
                command(['fit-correction', calibration, '--output', mapping, '--no-temperature'], log)
            stage('Apply shared rainfall correction, N=1000')
            if not corrected.exists():
                command(['correct', raw, '--correction', mapping, '--output', corrected, '--no-temperature'], log)
            elif not complete(corrected, 1000):
                # Never publish or analyze a partially written correction.
                raise RuntimeError(f'Incomplete correction output: {corrected}; inspect before resuming.')
            reports = []
            for run in [baseline, raw, corrected]:
                for window, dates in [('Validation', None), ('Validation_AprOct', ('2026-04-01', '2026-11-01'))]:
                    stage(f'Validate {run.name}: {window}')
                    report = run/window/'validation.json'
                    if not report.exists() or report.stat().st_mtime_ns < (run/'scenario_catalog.json').stat().st_mtime_ns:
                        args = ['validate', run, '--output', run/window]
                        if dates:
                            args += ['--window', *dates]
                        command(args, log)
                    reports.append(report)
            stage('Build plots and paired comparison')
            # Use a subprocess so separate case reports never share matplotlib state.
            command(['compare', '--reports', *reports, '--output', root/'Assessment'], log)
            stage('Verify paired seeds and unchanged temperature metrics')
            catalogs = [json.loads((p/'scenario_catalog.json').read_text()) for p in [baseline, raw, corrected]]
            members = [sorted(c['scenarios'], key=lambda r:r['seed']) for c in catalogs]
            seeds = [[m['seed'] for m in group] for group in members]
            assert seeds[0] == seeds[1] == seeds[2] == list(range(20261021, 20262021))
            model = json.loads(mapping.read_text())
            assert len(model['calibration']['seeds']) == 50
            assert not set(model['calibration']['seeds']) & set(seeds[0])
            for a, b in zip(members[1], members[2]):
                for metric in ['mean_temperature_C', 'peak_temperature_C', 'pet_mm']:
                    assert a['values'][metric] == b['values'][metric]
            report = json.loads(reports[0].read_text())
            checks = {'case': case, 'members_each': 1000, 'seed_match': True,
                      'calibration_members': 50, 'calibration_seeds_disjoint': True,
                      'temperature_metrics_unchanged_by_rain_correction': True,
                      'cells': report['land_cells'], 'historical_annual_years': report['historical_years'],
                      'storage': 'float32 weather + DEFLATE; float64 fitted coefficients',
                      'source_config_sha256': digest(config),
                      'reports': [{'path': str(p), 'sha256': digest(p)} for p in reports]}
            (root/'run_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
            stage('Complete')
            return checks
        except Exception as error:
            stage('Failed: '+str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=['egypt', 'algeria', 'tunisia', 'vietnam'], required=True)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs not in (1, 2):
        parser.error('--jobs must be 1 or 2 to bound VM resources.')
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run_case, case): case for case in args.cases}
        for future in as_completed(pending):
            case = pending[future]
            try:
                future.result()
            except Exception as error:
                failures.append((case, str(error)))
                print(f'{case}: FAILED: {error}', flush=True)
    if failures:
        raise SystemExit(str(failures))
