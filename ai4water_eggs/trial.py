"""Compact, offline comparison of matched-seed variability-repair experiments."""

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
    'total_precipitation_mm': 'Rainfall (mm)', 'mean_temperature_C': 'Mean T (°C)',
    'mean_daily_temperature_range_C': 'Daily T range (°C)', 'peak_temperature_C': 'Maximum T (°C)',
    'minimum_temperature_C': 'Minimum T (°C)', 'pet_mm': 'ETo (mm)',
    'water_balance_mm': 'Rainfall − ETo (mm)', 'longest_dry_spell_days': 'Longest dry spell (days)',
}
SPREAD = ['total_precipitation_mm', 'mean_temperature_C', 'mean_daily_temperature_range_C', 'pet_mm']
PLOT_METRICS = ['total_precipitation_mm', 'peak_temperature_C', 'pet_mm']
COVERAGE_METRICS = ['total_precipitation_mm', 'mean_temperature_C', 'peak_temperature_C',
                    'minimum_temperature_C', 'mean_daily_temperature_range_C', 'pet_mm']
ENVELOPES = {'cumulative_rain': 'Cumulative rainfall', 'cumulative_pet': 'Cumulative ETo',
             'cumulative_balance': 'Cumulative rain − ETo', 'temperature_7day': '7-day mean T'}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _seeds(root, report):
    path = root / 'scenario_catalog.json'
    if not path.is_file():
        return None
    catalog = json.loads(path.read_text())
    mapping = {row['number']: row.get('seed') for row in catalog['scenarios']}
    values = [mapping.get(row['number']) for row in report['ensemble']]
    if any(seed is None for seed in values):
        raise ValueError(f'{root}: validation member lacks a catalog seed.')
    if len(set(values)) != len(values):
        raise ValueError(f'{root}: duplicate scenario seeds.')
    return sorted(values)


def _number(value, signed=False):
    return '—' if value is None else format(value, '+.2f' if signed else '.2f')


def build_trial_report(paths, output, labels=None):
    """Compare standard validation reports; labels maps experiment names to titles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cases, groups, sources, labels = [], {}, [], labels or {}
    for filename in paths:
        path = Path(filename).resolve()
        report = json.loads(path.read_text())
        root = path.parent.parent
        config_path = root / 'run_config.json'
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        label = labels.get(report['experiment'], config.get('Trial Label', report['experiment']))
        window = report.get('window_label', f"{report['start']}–{report['last_day']}")
        case = {'report': report, 'root': root, 'label': label, 'seeds': _seeds(root, report)}
        group = groups.setdefault(window, [])
        if any(c['label'] == label for c in group):
            raise ValueError(f'Duplicate trial label in {window}: {label}')
        if group:
            reference = group[0]['report']
            for key in ('training_years', 'historical_years', 'temperature_reference_date',
                        'latitude', 'longitude', 'mask', 'start', 'last_day', 'ensemble_members'):
                if report.get(key) != reference.get(key):
                    raise ValueError(f'{window}: incompatible {key}.')
            for metric in METRICS:
                a, b = report['comparisons'].get(metric, {}), reference['comparisons'].get(metric, {})
                if a.get('available') and b.get('available'):
                    av = [a['historical_mean'], a['historical_std'], *a['historical_quantiles'].values()]
                    bv = [b['historical_mean'], b['historical_std'], *b['historical_quantiles'].values()]
                    if not np.allclose(av, bv, rtol=1e-9, atol=1e-9):
                        raise ValueError(f'{window}: historical {metric} reference differs.')
        group.append(case)
        cases.append(case)
        sources.append({'path': str(path), 'sha256': _hash(path), 'label': label, 'window': window})
    if not cases:
        raise ValueError('At least one validation report is required.')
    seeds = [c['seeds'] for c in cases if c['seeds'] is not None]
    if seeds and any(value != seeds[0] for value in seeds):
        raise ValueError('Trial reports do not use matching scenario seeds.')
    for case in cases[1:]:
        for key in ('training_years', 'temperature_reference_date', 'latitude', 'longitude', 'mask'):
            if case['report'].get(key) != cases[0]['report'].get(key):
                raise ValueError(f'Trial reports differ in {key}.')
    counts = {c['report']['ensemble_members'] for c in cases}
    if len(counts) != 1:
        raise ValueError('Trial reports must have the same ensemble size.')
    calibration_sets=[]
    for case in cases:
        model=case['root']/'pooled_correction.json'
        if model.is_file():
            correction=json.loads(model.read_text())
            calibration=correction['calibration']['seeds']
            if case['seeds'] and set(calibration)&set(case['seeds']):
                raise ValueError('Correction calibration and evaluation seeds overlap.')
            calibration_sets.append({'experiment':case['root'].name,'members':len(calibration),'sha256':_hash(model)})
    output = Path(output).resolve()
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    rows, quantiles, coverage = [], [], []
    for case in cases:
        r = case['report']
        for metric in COVERAGE_METRICS:
            score = r['comparisons'][metric]
            coverage.append({'variant': case['label'], 'window': r.get('window_label', ''),
                             'kind': 'season_metric', 'metric': metric,
                             'historical_years': len(r['historical_years']), 'nominal_percent': 90,
                             'coverage_percent': score['historical_inside_ensemble_p5_p95_percent']})
        for metric in ENVELOPES:
            coverage.append({'variant': case['label'], 'window': r.get('window_label', ''),
                             'kind': 'daily_pointwise', 'metric': metric,
                             'historical_years': len(r['historical_years']), 'nominal_percent': 90,
                             'coverage_percent': r['envelopes'][metric]['historical_pointwise_containment_percent']})
        for metric, score in r['comparisons'].items():
            if not score.get('available'):
                continue
            row = {'variant': case['label'], 'window': r.get('window_label', ''), 'metric': metric,
                   'unit': score.get('unit', ''), 'historical_n': score['historical_n'], 'ensemble_n': score['ensemble_n']}
            for key in ('historical_mean', 'ensemble_mean', 'mean_bias', 'historical_std', 'ensemble_std',
                        'spread_ratio', 'historical_inside_ensemble_p5_p95_percent', 'wasserstein_distance'):
                row[key] = score.get(key)
            for percentile in (5, 10, 25, 50, 75, 90, 95):
                h, g = score['historical_quantiles'].get(str(percentile)), score['ensemble_quantiles'].get(str(percentile))
                quantiles.append({**{k: row[k] for k in ('variant', 'window', 'metric', 'unit')},
                                  'percentile': percentile, 'historical': h, 'generated': g,
                                  'difference': None if h is None or g is None else g - h})
                if percentile in (5, 95):
                    row[f'historical_p{percentile}'], row[f'generated_p{percentile}'] = h, g
            rows.append(row)
    downloads = {}
    for name, values in (('metric_comparison.csv', rows), ('percentiles.csv', quantiles), ('coverage.csv', coverage)):
        content = _csv(values)
        (output / name).write_text(content)
        downloads[name] = 'data:text/csv;base64,' + base64.b64encode(content.encode()).decode()
    annual = groups.get('Full year', next(iter(groups.values())))
    from ai4water_eggs.variability import compare_rainfall_scales
    aggregation = compare_rainfall_scales(
        [c['root'] for c in annual], annual[0]['root'], annual[0]['report']['historical_years'],
        labels={c['root'].name:c['label'] for c in annual},
        members={c['root'].name:[v['number'] for v in c['report']['ensemble']] for c in annual})
    content = _csv(aggregation['rows'])
    (output / 'aggregation_scores.csv').write_text(content)
    downloads['aggregation_scores.csv'] = 'data:text/csv;base64,' + base64.b64encode(content.encode()).decode()
    saved = []
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}), PdfPages(output / 'assessment.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(11, 5.7), layout='constrained')
        x = np.arange(len(SPREAD)); width = .8 / len(annual)
        for i, case in enumerate(annual):
            values = [case['report']['comparisons'].get(m, {}).get('spread_ratio', np.nan) for m in SPREAD]
            ax.bar(x - .4 + width * (i + .5), values, width, label=case['label'])
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set(xticks=x, xticklabels=[METRICS[m] for m in SPREAD], ylabel='Generated SD / historical SD', title='Annual variability · reference = 1')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        fig.savefig(figures / '01_spread.png', dpi=160); pdf.savefig(fig); plt.close(fig)
        saved.append('01_spread.png')
        fig, axes = plt.subplots(len(groups), 3, figsize=(14, max(4, len(groups) * 3.7)), squeeze=False, layout='constrained')
        for ri, (window, group) in enumerate(groups.items()):
            names = ['History'] + [c['label'] for c in group]
            for ci, metric in enumerate(PLOT_METRICS):
                ax = axes[ri, ci]
                qs = [group[0]['report']['comparisons'][metric]['historical_quantiles']]
                qs += [c['report']['comparisons'][metric]['ensemble_quantiles'] for c in group]
                for y, q in enumerate(qs):
                    color = '#333333' if y == 0 else plt.get_cmap('tab10')((y - 1) % 10)
                    ax.plot([q['5'], q['95']], [y, y], color=color, lw=3)
                    ax.plot(q['50'], y, 'o', color=color, ms=5)
                ax.set(yticks=range(len(names)), yticklabels=names if ci == 0 else [], xlabel=METRICS[metric], title=window)
                ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        fig.suptitle('P5–P95 intervals · dot = P50')
        fig.savefig(figures / '02_percentiles.png', dpi=160); pdf.savefig(fig); plt.close(fig)
        saved.append('02_percentiles.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
        for ax, scope, title in zip(axes, ['cell','regional'], ['Individual-cell rainfall','Regional rainfall']):
            for case in annual:
                entries = [r for r in aggregation['rows'] if r['experiment']==case['root'].name and r['scope']==scope]
                ax.plot([r['scale_days'] for r in entries], [r['sd_ratio'] for r in entries], 'o-', label=case['label'], alpha=.8)
            ax.axhline(1, color='black', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set(xticks=[1,7,30,90,365],xticklabels=['1','7','30','90','Year'],xlabel='Accumulation period · days',ylabel='Generated RMS SD / historical RMS SD',title=title)
            ax.grid(alpha=.2)
        handles, legends=axes[0].get_legend_handles_labels()
        fig.legend(handles,legends,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('Rainfall variance by timescale · identical curves overlap')
        fig.savefig(figures / '03_aggregation.png', dpi=160); pdf.savefig(fig); plt.close(fig)
        saved.append('03_aggregation.png')
        fig, axes = plt.subplots(len(groups), 2, figsize=(13, 4.4 * len(groups)), squeeze=False, layout='constrained')
        for ri, (window, group) in enumerate(groups.items()):
            for ci, (kind, metrics, titles) in enumerate([
                    ('season_metric', COVERAGE_METRICS, METRICS),
                    ('daily_pointwise', list(ENVELOPES), ENVELOPES)]):
                ax = axes[ri, ci]
                y = np.arange(len(metrics))
                for i, case in enumerate(group):
                    by_metric = {row['metric']: row['coverage_percent'] for row in coverage
                                 if row['variant'] == case['label'] and row['window'] == window and row['kind'] == kind}
                    ax.scatter([by_metric[m] for m in metrics], y + (i - (len(group)-1)/2)*.18,
                               label=case['label'], color=plt.get_cmap('tab10')(i), s=35)
                ax.axvline(90, color='black', ls='--', lw=1)
                ax.set(yticks=y, yticklabels=[titles[m] for m in metrics], xlim=(-2, 102),
                       xlabel='Historical values inside generated P5–P95 (%)',
                       title=window + (' · totals / extremes' if ci == 0 else ' · daily pointwise bands'))
                ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
        handles, legends = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legends, loc='outside lower center', ncol=3, fontsize=9)
        fig.suptitle('Envelope coverage · dashed line = nominal 90% · descriptive, in-sample')
        fig.savefig(figures / '04_coverage.png', dpi=160); pdf.savefig(fig); plt.close(fig)
        saved.append('04_coverage.png')
    findings = []
    if len(annual) > 1:
        a, b = annual[0], annual[-1]
        summary = '; '.join(f"{METRICS[m].split(' (')[0]} SD {_number(a['report']['comparisons'][m]['spread_ratio'])} → {_number(b['report']['comparisons'][m]['spread_ratio'])}" for m in SPREAD)
        findings.append(f"{a['label']} → {b['label']}: {summary}.")
    last = annual[-1]
    findings.append(f"{last['label']}: " + '; '.join(f"{METRICS[m]} mean bias {_number(last['report']['comparisons'][m]['mean_bias'], True)}" for m in ('peak_temperature_C', 'minimum_temperature_C', 'pet_mm')) + '.')
    sections, coverage_sections = [], []
    for window, group in groups.items():
        body = []
        for case in group:
            scores = case['report']['comparisons']
            cells = [html.escape(case['label'])]
            for metric in SPREAD:
                score = scores[metric]
                cells.append(_number(score['mean_bias'], True) + ' / ' + _number(score['spread_ratio']))
            cells.append(_number(scores['peak_temperature_C']['mean_bias'], True) + ' / ' + _number(scores['minimum_temperature_C']['mean_bias'], True))
            cells.append(_number(scores['longest_dry_spell_days']['mean_bias'], True) + ' / ' + _number(scores['longest_dry_spell_days']['spread_ratio']))
            body.append('<tr>' + ''.join(f'<td>{value}</td>' for value in cells) + '</tr>')
        heads = ['Variant', 'Rain (mm)', 'Mean T (°C)', 'T range (°C)', 'ETo (mm)', 'Max / min T bias (°C)', 'Dry spell (days)']
        sections.append(f'<h2>{html.escape(window)}</h2><div class="table"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')
        body = []
        for metric in COVERAGE_METRICS:
            cells = [html.escape(METRICS[metric])] + [_number(case['report']['comparisons'][metric]['historical_inside_ensemble_p5_p95_percent']) for case in group]
            body.append('<tr>' + ''.join(f'<td>{value}</td>' for value in cells) + '</tr>')
        heads = ['Coverage (%)'] + [html.escape(case['label']) for case in group]
        coverage_sections.append(f'<h2>{html.escape(window)} · P5–P95 coverage</h2><div class="table"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in heads) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'ensemble_members': counts.pop(),
                'seed_match_verified': len(seeds) == len(cases), 'source_reports': sources,
                'code_sha256': _hash(__file__), 'figure_count': len(saved), 'windows': list(groups), 'findings': findings}
    metadata['aggregation'] = aggregation
    metadata['correction_calibrations']=calibration_sets
    (output / 'assessment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    pictures = ''.join('<img alt="' + name + '" src="data:image/png;base64,' + base64.b64encode((figures / name).read_bytes()).decode() + '">' for name in saved)
    links = ' · '.join(f'<a download="{name}" href="{uri}">{name}</a>' for name, uri in downloads.items())
    seed_note = 'Scenario seeds match across all reports.' if metadata['seed_match_verified'] else 'Seed matching could not be verified for every report.'
    history = '; '.join(f"{window}: {len(group[0]['report']['historical_years'])} historical years" for window, group in groups.items())
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Variability repair trial</title><style>body{font:16px/1.5 system-ui;margin:24px auto;padding:0 16px;max-width:1400px;color:#152838}h1{font-size:26px}h2{font-size:20px}img{display:block;width:100%;height:auto;margin:24px 0}.table{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccd5dc;text-align:right}td:first-child,th:first-child{text-align:left}p{max-width:1100px}a{color:#066c91}</style>'
    page += f'<h1>Variability repair trial · N={metadata["ensemble_members"]}</h1><p>{html.escape(history)}. Stationary climate; full-record fit.</p><ul>' + ''.join(f'<li>{html.escape(s)}</li>' for s in findings) + '</ul>'
    if calibration_sets:
        calibration_counts=sorted({c['members'] for c in calibration_sets})
        page+='<p>Frozen corrections calibrated on '+ '/'.join(map(str,calibration_counts))+' separate members; calibration and evaluation seeds do not overlap.</p>'
    page += ''.join(coverage_sections)
    page += '<p>Coverage is descriptive: nominal P5–P95 containment is 90%; higher containment alone does not establish a better fit. Check interval width and bias too.</p>'
    page += '<p>Provisional comparison. SD ratio = generated SD / historical SD; reference = 1. Tables below show <b>mean bias / SD ratio</b>, except extrema. No application acceptance threshold is imposed.</p>' + ''.join(sections) + pictures
    page += '<p>' + html.escape(seed_note) + ' Climatology reproduction, not independent validation or forecast skill. Annual and April–October results use the same members. ETo uses Hargreaves.</p><p>' + links + '</p></html>'
    (output / 'index.html').write_text(page)
    return metadata


def subset_experiment(source, output, first, last, label=None):
    """Create a read-only member subset; copy metadata and hardlink weather/fits."""
    import os
    import shutil
    from ai4water_eggs.metrics import scenario_files
    source, output=Path(source).resolve(),Path(output).resolve()
    if (source/'.in_progress').exists():
        raise ValueError('Source experiment is incomplete.')
    selected=[row for row in scenario_files(source) if first<=row[0]<=last]
    if last<first or [row[0] for row in selected]!=list(range(first,last+1)):
        raise ValueError('Requested member range is not completely available.')
    if output.exists():
        raise FileExistsError(output)
    def link(a,b):
        try:os.link(a,b)
        except OSError:shutil.copy2(a,b)
    output.mkdir(parents=True)
    (output/'.in_progress').write_text('Subset metrics pending.\n')
    shutil.copytree(source/'Data',output/'Data',copy_function=link)
    for _,directory,_ in selected:
        shutil.copytree(directory,output/'Simulation'/directory.name,copy_function=link)
    for filename in ('boundary.geojson','pooled_correction.json','correction_application.json'):
        if (source/filename).exists():shutil.copy2(source/filename,output/filename)
    config=json.loads((source/'run_config.json').read_text())
    config.update({'Simulation name':output.name,'Data Output Directory':str(output.parent),
                   'Number of Simulations':len(selected),'Subset Source':str(source),
                   'Subset Members':[first,last]})
    config['Trial Label']=label or output.name
    (output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    return output
