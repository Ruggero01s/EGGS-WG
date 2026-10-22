"""Concise cross-case comparison of the same paired generator modifications."""

import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np

METRICS = {
    'total_precipitation_mm': 'Rainfall', 'mean_temperature_C': 'Mean T',
    'peak_temperature_C': 'Maximum T', 'minimum_temperature_C': 'Minimum T',
    'mean_daily_temperature_range_C': 'Daily T range', 'pet_mm': 'ETo',
    'spi_period': 'SPI', 'spei_period': 'SPEI',
}
VARIANTS = ['Baseline', 'Temperature', 'Temperature + rain']


def improvement(before, after, target):
    """Positive means closer to the stated target, not simply a larger value."""
    return abs(before-target) - abs(after-target)


def build_transfer_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(root).resolve()
    outputs = root.parent
    definitions = [('Capitanata', [outputs/'capitanata_annual_2026_n1000',
                    outputs/'capitanata_temperature_n1000/temperature_n1000',
                    outputs/'capitanata_temperature_n1000/temperature_rain_n1000'])]
    definitions += [(name.title(), [root/name/'baseline_n1000', root/name/'temperature_n1000',
                                    root/name/'temperature_rain_n1000'])
                    for name in ('egypt', 'algeria', 'tunisia')]
    rows, sources, cases, pending = [], [], [], []
    for name, runs in definitions:
        paths = [run/window/'validation.json' for run in runs for window in ('Validation', 'Validation_AprOct')]
        if not all(path.is_file() for path in paths):
            pending.append(name)
            continue
        catalogs = [json.loads((run/'scenario_catalog.json').read_text()) for run in runs]
        seeds = [sorted(row['seed'] for row in catalog['scenarios']) for catalog in catalogs]
        if any(len(value) != 1000 or value != seeds[0] for value in seeds):
            raise ValueError(f'{name}: unpaired or incomplete ensembles.')
        cases.append(name)
        for variant, run in zip(VARIANTS, runs):
            for window in ('Validation', 'Validation_AprOct'):
                path = run/window/'validation.json'
                report = json.loads(path.read_text())
                if report['ensemble_members'] != 1000:
                    raise ValueError(f'{path}: validation is not for 1,000 members.')
                sources.append({'case': name, 'variant': variant, 'path': str(path),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                for metric in METRICS:
                    score = report['comparisons'][metric]
                    rows.append({'case': name, 'variant': variant, 'window': report['window_label'],
                                 'metric': metric, 'historical_years': len(report['historical_years']),
                                 'ensemble_members': report['ensemble_members'], 'cells': report['land_cells'],
                                 'coverage_percent': score.get('historical_inside_ensemble_p5_p95_percent'),
                                 'mean_bias': score.get('mean_bias'), 'sd_ratio': score.get('spread_ratio'),
                                 'historical_sd': score.get('historical_std')})
    if not cases:
        raise ValueError('No completed cases available.')
    output = root/'Assessment'
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    content = stream.getvalue()
    (output/'scores.csv').write_text(content)
    lookup = {(r['case'], r['variant'], r['window'], r['metric']): r for r in rows}
    windows = ['Full year', 'Apr–Oct']
    changes = []
    for case in cases:
        for window in windows:
            for metric in METRICS:
                a = lookup[(case, VARIANTS[0], window, metric)]
                for variant in VARIANTS[1:]:
                    b = lookup[(case, variant, window, metric)]
                    changes.append({'case': case, 'window': window, 'metric': metric, 'variant': variant,
                                    'coverage_closer_to_90_pp': None if a['coverage_percent'] is None or b['coverage_percent'] is None else improvement(a['coverage_percent'], b['coverage_percent'], 90),
                                    'sd_ratio_closer_to_1': None if a['sd_ratio'] is None or b['sd_ratio'] is None else improvement(a['sd_ratio'], b['sd_ratio'], 1),
                                    'absolute_mean_bias_reduction': None if a['mean_bias'] is None or b['mean_bias'] is None else improvement(a['mean_bias'], b['mean_bias'], 0)})
    with (output/'changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0])); writer.writeheader(); writer.writerows(changes)

    saved = []
    specifications = [('coverage_percent', 90, 'Coverage (%) · target 90', '01_coverage.png'),
                      ('sd_ratio', 1, 'Generated / historical SD · target 1', '02_spread.png'),
                      ('mean_bias', 0, 'Mean bias · °C, mm, or index units', '03_bias.png')]
    with plt.rc_context({'font.size': 10}), PdfPages(output/'comparison.pdf') as pdf:
        for field, target, title, filename in specifications:
            matrices, annotations = [], []
            for window in windows:
                values = np.full((len(cases), len(METRICS)), np.nan)
                labels = np.full(values.shape, '—', dtype=object)
                for i, case in enumerate(cases):
                    for j, metric in enumerate(METRICS):
                        a = lookup[(case, VARIANTS[0], window, metric)]
                        b = lookup[(case, VARIANTS[2], window, metric)]
                        av, bv = a[field], b[field]
                        if av is None or bv is None:
                            continue
                        scale = a['historical_sd'] if field == 'mean_bias' else 1
                        if scale is None or scale <= 1e-12:
                            continue
                        values[i, j] = improvement(av, bv, target) / scale
                        labels[i, j] = f'{av:.0f}→{bv:.0f}' if field == 'coverage_percent' else f'{av:.2f}→{bv:.2f}'
                matrices.append(values); annotations.append(labels)
            finite = np.concatenate([a[np.isfinite(a)] for a in matrices])
            limit = max(float(np.max(np.abs(finite))) if finite.size else 1, 1e-8)
            fig, axes = plt.subplots(2, 1, figsize=(14, max(6.5, len(cases)*1.35)), layout='constrained')
            cmap = plt.get_cmap('RdYlGn').copy(); cmap.set_bad('#dddddd')
            for ax, window, values, labels in zip(axes, windows, matrices, annotations):
                picture = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect='auto')
                ax.set(xticks=np.arange(len(METRICS)), xticklabels=list(METRICS.values()),
                       yticks=np.arange(len(cases)), yticklabels=cases, title=window)
                for index, label in np.ndenumerate(labels):
                    ax.text(index[1], index[0], label, ha='center', va='center', fontsize=9,
                            bbox={'facecolor':'white', 'alpha':.65, 'edgecolor':'none', 'pad':1})
            fig.colorbar(picture, ax=axes, shrink=.7, label='Closer to target → green; farther → red' +
                         (' (bias change / historical SD)' if field == 'mean_bias' else ''))
            fig.suptitle(title+' · annotations: Baseline → Temperature + rain')
            fig.savefig(figures/filename, dpi=160); pdf.savefig(fig); plt.close(fig)
            saved.append(filename)

    summaries = []
    for window in windows:
        for metric in ('total_precipitation_mm', 'peak_temperature_C', 'pet_mm'):
            subset = [r for r in changes if r['window'] == window and r['metric'] == metric
                      and r['variant'] == VARIANTS[2] and r['coverage_closer_to_90_pp'] is not None]
            better = sum(r['coverage_closer_to_90_pp'] > 1e-8 for r in subset)
            worse = sum(r['coverage_closer_to_90_pp'] < -1e-8 for r in subset)
            summaries.append({'window': window, 'metric': metric, 'closer_to_nominal': better,
                              'farther_from_nominal': worse, 'unchanged': len(subset)-better-worse,
                              'cases': len(subset)})
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'cases': cases, 'pending': pending,
                'members_each': 1000, 'rows': rows, 'changes': changes, 'coverage_summary': summaries,
                'sources': sources, 'figures': saved, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'method': 'Within-case paired ensembles; full available historical fit; stationary reference 2020. '
                          'Each case retains its own historical years, cells and units. No pooled cross-case statistical test. '
                          'Coverage alone is not sufficient; inspect mean bias and SD ratio. Vietnam excluded by request.'}
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    pics = ''.join('<img alt="'+name+'" src="data:image/png;base64,'+
                   base64.b64encode((figures/name).read_bytes()).decode()+'">' for name in saved)
    download = 'data:text/csv;base64,'+base64.b64encode(content.encode()).decode()
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EGGS cross-case trial</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:24px auto;padding:0 16px;color:#152838}img{width:100%;height:auto}.table{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}a{color:#066c91}</style>'
    body += '<h1>EGGS cross-case trial · N=1,000</h1><p>Baseline, temperature fix, and temperature fix + pooled rain. Annual and April–October; full-record climatology reproduction.</p>'
    if pending:
        body += '<p>Pending: '+html.escape(', '.join(pending))+'.</p>'
    body += '<table><tr><th>Case</th><th>Cells</th><th>Annual historical years</th></tr>'
    for case in cases:
        r = lookup[(case, VARIANTS[0], windows[0], next(iter(METRICS)))]
        body += f'<tr><td>{html.escape(case)}</td><td>{r["cells"]}</td><td>{r["historical_years"]}</td></tr>'
    body += '</table><h2>Annual results</h2><div class="table"><table><tr><th>Case</th><th>Peak T bias (°C)<br>Baseline → modified</th><th>ETo bias (mm)<br>Baseline → modified</th><th>Rain bias (mm)<br>Baseline / raw / pooled</th><th>Rain SD ratio<br>Baseline / raw / pooled</th></tr>'
    for case in cases:
        def value(variant, metric, field):
            v = lookup[(case, variant, 'Full year', metric)][field]
            return '—' if v is None else f'{v:+.2f}' if field == 'mean_bias' else f'{v:.2f}'
        fields = [case]
        fields += [' → '.join(value(v, m, 'mean_bias') for v in (VARIANTS[0], VARIANTS[2]))
                   for m in ('peak_temperature_C', 'pet_mm')]
        fields += [' / '.join(value(v, 'total_precipitation_mm', field) for v in VARIANTS)
                   for field in ('mean_bias', 'sd_ratio')]
        body += '<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in fields)+'</tr>'
    body += '</table></div><p>Raw = no rainfall adjustment; pooled = shared rainfall correction. Temperature is identical in the two modified variants. Rain SD target = 1.</p>'
    body += '<p>Green = closer to target; red = farther. Read actual values in each cell. Tunisia has two cells: regional results do not establish performance over large spatial domains. Vietnam excluded.</p>'+pics
    body += '<p>Transfer adaptations: Egypt needed fixes for active-cell indexing and unsupported transitions in dry months, applied to all variants. Positive hourly-scale fallback was needed in Algeria and Egypt; it retains fitted daily mean scales. This is not an unchanged transfer of the first Capitanata implementation.</p>'
    body += '<p>Remaining rainfall limit: the generator omits daily amounts below 1 mm. These account for 19% of historical Egyptian rain and 4.5–6.1% in the other cases; correcting wet-day amounts does not restore them.</p>'
    body += '<p>These cases test transfer of the modifications; they do not establish universal improvement. Reference: historical ERA5-Land, fitted on all available years; no independent station validation. New scenario files use compressed float32 storage; fitted coefficients remain float64.</p>'
    body += f'<p><a download="scores.csv" href="{download}">All scores CSV</a></p></html>'
    (output/'index.html').write_text(body)
    return metadata
