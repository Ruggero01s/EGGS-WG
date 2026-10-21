"""Compare matched seeds with/without Variable Adjust; no fitting or weather changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4water_eggs.metrics import read_scenario


def compare(adjusted, unadjusted, output):
    adjusted, unadjusted, output = map(Path, (adjusted, unadjusted, output))
    catalogs = [json.loads((p/'scenario_catalog.json').read_text()) for p in (adjusted, unadjusted)]
    reports = [json.loads((adjusted/p/'validation.json').read_text()) for p in ('Validation', 'Validation_AprOct')]
    mask = np.array(reports[0]['mask'], dtype=bool)
    latitude, longitude = (np.array(reports[0][key]) for key in ('latitude', 'longitude'))
    first = {row['seed']: row for row in catalogs[0]['scenarios']}
    paired = catalogs[1]['scenarios']
    if len({row['seed'] for row in paired}) != len(paired):
        raise ValueError('Duplicate diagnostic seeds.')
    values = {report['window_label']: {'adjusted': [], 'unadjusted': []} for report in reports}
    for raw in paired:
        corrected = first[raw['seed']]
        for mode, root, member in [('adjusted', adjusted, corrected), ('unadjusted', unadjusted, raw)]:
            daily = read_scenario([root/p for p in member['files']], mask, latitude, longitude)
            assert daily['seed'] == raw['seed']
            for report in reports:
                selected = ((daily['days'] >= np.datetime64(report['start']))
                            & (daily['days'] <= np.datetime64(report['last_day'])))
                assert int(selected.sum()) == report['days_per_season']
                values[report['window_label']][mode].append({
                    'mean_daily_temperature_range_C': float(np.mean(daily['dtr'][selected])),
                    'pet_mm': float(np.sum(daily['pet'][selected])),
                    'mean_temperature_C': float(np.mean(daily['tmean'][selected])),
                    'total_precipitation_mm': float(np.sum(daily['rain'][selected])),
                })
    for window in values.values():
        assert np.allclose([r['mean_temperature_C'] for r in window['adjusted']],
                           [r['mean_temperature_C'] for r in window['unadjusted']], rtol=0, atol=1e-10)
    rows = []
    for report in reports:
        for metric in next(iter(values[report['window_label']].values()))[0]:
            history = np.array([r['values'][metric] for r in report['historical']])
            row = {'window': report['window_label'], 'metric': metric, 'paired_members': len(paired),
                   'historical_mean': float(history.mean()), 'historical_sd': float(history.std(ddof=1))}
            for mode in ('adjusted', 'unadjusted'):
                v = np.array([r[metric] for r in values[report['window_label']][mode]])
                row.update({mode+'_mean': float(v.mean()), mode+'_sd': float(v.std(ddof=1)),
                            mode+'_sd_ratio': float(v.std(ddof=1)/history.std(ddof=1))})
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'adjustment_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {'paired_members': len(paired), 'seeds': [r['seed'] for r in paired],
              'adjusted': str(adjusted.resolve()), 'unadjusted': str(unadjusted.resolve()),
              'method': 'Same fitted model, domain, dates and seeds; Variable Adjust true vs false. '
                        'Adjustment changes both rainfall and the temperature cycle. '
                        'This N=50 check diagnoses adjustment; it is not a validated replacement model.',
              'source_catalogs': [{'path': str((p/'scenario_catalog.json').resolve()),
                                   'sha256': hashlib.sha256((p/'scenario_catalog.json').read_bytes()).hexdigest()}
                                  for p in (adjusted, unadjusted)],
              'rows': rows}
    (output/'adjustment_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjusted', required=True, type=Path)
    parser.add_argument('--unadjusted', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compare(args.adjusted, args.unadjusted, args.output)
