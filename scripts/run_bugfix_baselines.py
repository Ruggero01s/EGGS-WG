"""Run shared-bugfix baselines and compare paired archived members with history."""
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


def run_case(case, count, root):
    source = REPO/'outputs'/('capitanata_annual_2026_n1000' if case == 'capitanata'
                             else f'transfer_n1000/{case}/baseline_n1000')
    destination = root/case
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def stage(message):
            (destination/'status.json').write_text(json.dumps({'case': case, 'stage': message,
                'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2)+'\n')
            print(f'{case}: {message}', flush=True)

        def command(*args):
            with (destination/'run.log').open('a') as log:
                subprocess.run([sys.executable, '-u', str(REPO/'ensemble.py'), *map(str, args)],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

        def complete(run):
            path = run/'scenario_catalog.json'
            return (not (run/'.in_progress').exists() and path.is_file()
                    and json.loads(path.read_text())['scenario_count'] == count)

        try:
            fixed, archived = destination/f'fixed_n{count}', destination/f'archived_n{count}'
            params = json.loads((source/'run_config.json').read_text())
            params.update({'Simulation name': fixed.name, 'Data Output Directory': str(destination),
                'Number of Simulations': count, 'Reuse Training From': str(source), 'Regenerate Data': False,
                'Workers': 2, 'Simulation Workers': 2, 'Variable Adjust': True,
                'Temperature Seasonal Fit': 'legacy', 'Temperature Residual Conditioning': 'legacy',
                'Temperature VAR Fit': 'legacy', 'Weather Output Precision': 'float32',
                'NetCDF Compression': True, 'Trial Label': 'Common bug fixes'})
            config = REPO/'configs'/f'bugfix_{case}_n{count}.yaml'
            config.write_text(yaml.safe_dump(params, sort_keys=False))
            if not complete(fixed):
                stage(f'Refit and generate {count} fixed members')
                command('generate', config)
            if not archived.exists():
                stage(f'Subset {count} paired archived members')
                command('subset', source, '--members', 1, count, '--output', archived, '--label', 'Archived baseline')
            if not complete(archived):
                command('analyze', archived, '--scales', 1, 3, 6, 7, 12)
                (archived/'.in_progress').unlink(missing_ok=True)
            reports = []
            for run in (archived, fixed):
                for window, dates in [('Validation', []), ('Validation_AprOct', ['2026-04-01', '2026-11-01'])]:
                    stage(f'Validate {run.name}: {window}')
                    report = run/window/'validation.json'
                    if not report.exists():
                        command('validate', run, '--output', run/window, *(['--window', *dates] if dates else []))
                    reports.append(report)
            stage('Build paired plots and tables')
            command('compare', '--reports', *reports, '--output', destination/'Assessment')
            seeds = []
            for run in (archived, fixed):
                catalog = json.loads((run/'scenario_catalog.json').read_text())
                seeds.append(sorted(row['seed'] for row in catalog['scenarios']))
            assert seeds[0] == seeds[1] == list(range(params['Random Seed'], params['Random Seed']+count))
            marker = json.loads((fixed/'Data/fit_safeguards.json').read_text())
            checks = {'case': case, 'members_each': count, 'paired_seeds': True, 'fit_safeguards': marker,
                'reports': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in reports]}
            (destination/'run_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
            stage('Complete')
        except Exception as error:
            stage('Failed: '+str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cases', nargs='+', choices=('capitanata', 'egypt', 'algeria', 'tunisia'),
                        default=['tunisia', 'algeria', 'egypt', 'capitanata'])
    args = parser.parse_args()
    if not 1 <= args.n <= 1000:
        parser.error('--n must be 1..1000 for paired archived members.')
    root = REPO/'outputs'/f'bugfix_baseline_n{args.n}'
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run_case, case, args.n, root): case for case in args.cases}
        failures = []
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                failures.append((pending[future], str(error)))
        if failures:
            raise SystemExit(str(failures))
    if all((root/case/'run_checks.json').is_file() for case in ('capitanata', 'egypt', 'algeria', 'tunisia')):
        subprocess.run([sys.executable, '-c',
            'import sys; from ai4water_eggs.baseline import build_baseline_report; '
            'build_baseline_report(sys.argv[1], int(sys.argv[2]))', str(root), str(args.n)],
            cwd=REPO, check=True)
