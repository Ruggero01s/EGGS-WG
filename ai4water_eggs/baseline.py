"""Four-case baseline assessment: paired archived/fixed members and history."""
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np

from ai4water_eggs.transfer import METRICS, improvement


def build_baseline_report(root, members=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(root).resolve()
    cases = ['capitanata', 'egypt', 'algeria', 'tunisia']
    variants = ['Archived', 'Fixed']
    windows = ['Full year', 'Apr–Oct']
    reports, rows, sources = {}, [], []
    for case in cases:
        seeds = []
        for variant, folder in zip(variants, ['archived', 'fixed']):
            run = root/case/f'{folder}_n{members}'
            catalog = json.loads((run/'scenario_catalog.json').read_text())
            assert catalog['scenario_count'] == members and not (run/'.in_progress').exists()
            seeds.append(sorted(row['seed'] for row in catalog['scenarios']))
            for window, directory in zip(windows, ['Validation', 'Validation_AprOct']):
                path = run/directory/'validation.json'
                report = json.loads(path.read_text())
                assert report['ensemble_members'] == members
                reports[case, variant, window] = report
                sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                for metric, label in METRICS.items():
                    score = report['comparisons'][metric]
                    row = {'case': case, 'variant': variant, 'window': window, 'metric': metric,
                        'unit': score['unit'], 'cells': report['land_cells'],
                        'historical_years': len(report['historical_years']), 'members': members,
                        'historical_mean': score['historical_mean'], 'ensemble_mean': score['ensemble_mean'],
                        'mean_bias': score['mean_bias'], 'historical_sd': score['historical_std'],
                        'sd_ratio': score['spread_ratio'],
                        'coverage_percent': score['historical_inside_ensemble_p5_p95_percent']}
                    for q in ('5', '50', '95'):
                        row[f'historical_p{q}'] = score['historical_quantiles'][q]
                        row[f'ensemble_p{q}'] = score['ensemble_quantiles'][q]
                    rows.append(row)
        assert len(seeds[0]) == members and seeds[0] == seeds[1]
        for window in windows:
            old, new = [reports[case, variant, window] for variant in variants]
            for key in ('historical_years', 'latitude', 'longitude', 'mask', 'training_years', 'temperature_reference_date'):
                assert old[key] == new[key], (case, key)
            for metric in METRICS:
                for key in ('historical_mean', 'historical_std'):
                    assert np.isclose(old['comparisons'][metric][key], new['comparisons'][metric][key], rtol=1e-9, atol=1e-9)

    output = root/'Assessment'
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (output/'scores.csv').write_text(buffer.getvalue())
    lookup = {(r['case'], r['variant'], r['window'], r['metric']): r for r in rows}
    saved = []
    selected = ['total_precipitation_mm', 'peak_temperature_C', 'minimum_temperature_C', 'pet_mm']
    with plt.rc_context({'font.size': 9}), PdfPages(output/'comparison.pdf') as pdf:
        for index, window in enumerate(windows):
            fig, axes = plt.subplots(4, 4, figsize=(13, 11), layout='constrained')
            for i, case in enumerate(cases):
                for j, metric in enumerate(selected):
                    ax = axes[i, j]
                    scores = [reports[case, v, window]['comparisons'][metric] for v in variants]
                    quantiles = [scores[0]['historical_quantiles'], *[s['ensemble_quantiles'] for s in scores]]
                    for x, quantile, color in zip(range(3), quantiles, ['#182b3a', '#b77716', '#007b91']):
                        ax.plot([x, x], [quantile['5'], quantile['95']], color=color, lw=3)
                        ax.scatter([x], [quantile['50']], color=color, s=35, zorder=3)
                        ax.scatter([x, x], [quantile['5'], quantile['95']], color=color, marker='_', s=80)
                    ax.set(xticks=range(3), xticklabels=['History', 'Archived', 'Fixed'], xlim=(-.6, 2.6),
                           ylabel=f'{case.title()} · {scores[0]["unit"]}')
                    if i == 0:
                        ax.set_title(METRICS[metric])
                    ax.grid(axis='y', alpha=.2)
            fig.suptitle(f'{window} · P5–P95 and median · {members} paired members per variant\n'
                         'Each case has its own historical reference; no pooled regional distribution.')
            name = f'0{index+1}_quantiles.png'
            fig.savefig(figures/name, dpi=160); pdf.savefig(fig); plt.close(fig); saved.append(name)

        for index, (field, target, title) in enumerate([
                ('coverage_percent', 90, 'Coverage (%) · target 90'),
                ('sd_ratio', 1, 'Generated / historical SD · target 1'),
                ('mean_bias', 0, 'Mean bias · mm, °C or index units')], 3):
            matrices, annotations = [], []
            for window in windows:
                values = np.full((4, len(METRICS)), np.nan)
                labels = np.full(values.shape, '—', dtype=object)
                for i, case in enumerate(cases):
                    for j, metric in enumerate(METRICS):
                        a, b = [lookup[case, variant, window, metric] for variant in variants]
                        if a[field] is None or b[field] is None:
                            continue
                        scale = a['historical_sd'] if field == 'mean_bias' else 1
                        if scale is None or scale <= 1e-12:
                            continue
                        values[i, j] = improvement(a[field], b[field], target)/scale
                        decimals = 0 if field == 'coverage_percent' else 2
                        labels[i, j] = f'{a[field]:.{decimals}f}→{b[field]:.{decimals}f}'
                matrices.append(values); annotations.append(labels)
            finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
            limit = max(float(np.max(np.abs(finite))) if finite.size else 1, 1e-8)
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout='constrained')
            for ax, window, values, labels in zip(axes, windows, matrices, annotations):
                pic = ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
                ax.set(xticks=range(len(METRICS)), xticklabels=list(METRICS.values()),
                       yticks=range(4), yticklabels=[c.title() for c in cases], title=window)
                for (i, j), label in np.ndenumerate(labels):
                    ax.text(j, i, label, ha='center', va='center', fontsize=9,
                            bbox={'facecolor':'white', 'alpha':.7, 'edgecolor':'none', 'pad':1})
            fig.colorbar(pic, ax=axes, shrink=.7, label='Green = closer to target; red = farther' +
                         ('; bias change / historical SD' if field == 'mean_bias' else ''))
            fig.suptitle(title+' · archived → fixed')
            name = f'0{index}_{field}.png'
            fig.savefig(figures/name, dpi=160); pdf.savefig(fig); plt.close(fig); saved.append(name)

    metadata = {'members_each': members, 'cases': cases, 'rows': rows, 'sources': sources,
                'paired_seeds': True, 'common_history_verified': True, 'figures': saved,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EGGS common-bugfix baseline</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:24px auto;padding:0 16px;color:#182b3a}img{width:100%;height:auto}.table{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
    page += f'<h1>Common-bugfix baseline · N={members}</h1><p>Four cases; paired archived/fixed seeds. Original EGGS adjustments on; experimental corrections off. Whole-year 2026; stationary 2020 reference; fit all available 1990–2025 data. Vietnam excluded.</p>'
    page += '<p>In-sample ERA5-Land climatology reproduction. Coverage alone is insufficient: inspect bias, quantiles and spread. No automatic candidate selection in this step.</p>'
    page += '<div class="table"><table><tr><th>Case</th><th>Cells</th><th>Annual historical years</th><th>Detailed plots</th></tr>'
    for case in cases:
        row = lookup[case, 'Fixed', 'Full year', selected[0]]
        page += f'<tr><td>{case.title()}</td><td>{row["cells"]}</td><td>{row["historical_years"]}</td><td><a href="../{case}/Assessment/index.html">Seasonal/daily assessment</a></td></tr>'
    page += '</table></div>'
    for window in windows:
        page += f'<h2>{window}</h2><div class="table"><table><tr><th>Case</th><th>Metric</th><th>Historical mean</th><th>Mean bias<br>archived → fixed</th><th>SD ratio<br>archived → fixed</th><th>P5–P95 coverage (%)<br>archived → fixed</th></tr>'
        for case in cases:
            for metric in selected:
                a, b = [lookup[case, variant, window, metric] for variant in variants]
                cells = [case.title(), f'{METRICS[metric]} ({a["unit"]})', f'{a["historical_mean"]:.2f}']
                cells += [f'{a[field]:.2f} → {b[field]:.2f}' for field in ('mean_bias', 'sd_ratio', 'coverage_percent')]
                page += '<tr>'+''.join('<td>'+html.escape(cell)+'</td>' for cell in cells)+'</tr>'
        page += '</table></div>'
    for name in saved:
        encoded = base64.b64encode((figures/name).read_bytes()).decode()
        page += f'<img alt="{html.escape(name)}" src="data:image/png;base64,{encoded}">'
    encoded = base64.b64encode(buffer.getvalue().encode()).decode()
    page += f'<p><a download="scores.csv" href="data:text/csv;base64,{encoded}">Scores CSV</a></p><p>Weather storage: compressed float32; coefficient calculations remain float64. Archived Capitanata uses float64. Small tail differences can reflect sampling from only {members} members and 35–36 historical years. Tunisia has two cells.</p></html>'
    (output/'index.html').write_text(page)
    return metadata
