"""Create a compact, exportable assessment from historical-validation JSON files."""

import base64
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import binomtest

PRIMARY = ['total_precipitation_mm', 'mean_temperature_C', 'peak_temperature_C',
           'pet_mm', 'water_balance_mm', 'longest_dry_spell_days', 'hot_days_30',
           'spi_period', 'spei_period']
LABELS = ['Rainfall', 'Mean temperature', 'Peak cell temperature', 'Reference ETo',
          'Rainfall − ETo', 'Longest regional dry spell', 'Hot days', 'Whole-period SPI', 'Whole-period SPEI']
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
HIST, EGGS = '#b96a34', '#087f8c'
METHODS = [
    'Flags are exploratory diagnostics, not application-specific acceptance thresholds or equivalence tests.',
    'Mean/SD flags: 95% percentile-bootstrap interval excludes zero/unit ratio. Whole seasons are resampled independently; interannual dependence and fitted-parameter uncertainty are omitted.',
    'Low containment: exact binomial 95% upper bound below 90%, conditional on the fixed ensemble interval. This omits finite-ensemble quantile uncertainty and assumes independent historical seasons.',
    'Multiple metrics are inspected without multiplicity adjustment. These are historical-climatology reproduction diagnostics, not forecasting tests.',
    'Pointwise bands are not whole-trajectory probabilities. Each comparison uses its own training-only detrending and drought-index fits.',
    'Annual and seasonal windows from the same ensemble are not independent samples. Calendar years retain leap days in totals and spells; daily plots use matching month/day slots.',
]
SOURCES = [
    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html',
    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
]


def numeric(values):
    return np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def interval(values):
    values = numeric(values)
    return np.percentile(values, [2.5, 97.5]).tolist() if len(values) else [None, None]


def score_metric(historical, generated, resamples=5000, seed=20261020):
    """Season-level diagnostics; conditional sampling uncertainty only."""
    if resamples < 100:
        raise ValueError('Use at least 100 resamples.')
    h, g = numeric(historical), numeric(generated)
    if min(len(h), len(g)) < 2:
        return {'available': False, 'historical_n': len(h), 'ensemble_n': len(g),
                'flags': [], 'assessment': 'Unavailable'}
    hs, gs = float(np.std(h, ddof=1)), float(np.std(g, ddof=1))
    rng = np.random.default_rng(seed)
    hb = rng.choice(h, (resamples, len(h)), replace=True)
    gb = rng.choice(g, (resamples, len(g)), replace=True)
    mean_ci = interval(gb.mean(axis=1) - hb.mean(axis=1))
    hb_std, gb_std = hb.std(axis=1, ddof=1), gb.std(axis=1, ddof=1)
    valid = hb_std > 1e-12
    sd_ci = interval(gb_std[valid] / hb_std[valid]) if hs > 1e-12 else [None, None]
    hq, gq = np.percentile(h, [5, 50, 95]), np.percentile(g, [5, 50, 95])
    inside = int(np.sum((h >= gq[0]) & (h <= gq[2])))
    ci = binomtest(inside, len(h)).proportion_ci(confidence_level=.95, method='exact')
    flags = []
    if mean_ci[0] > 0:
        flags.append('Mean high')
    elif mean_ci[1] < 0:
        flags.append('Mean low')
    if sd_ci[1] is not None and sd_ci[1] < 1:
        flags.append('Narrow spread')
    elif sd_ci[0] is not None and sd_ci[0] > 1:
        flags.append('Wide spread')
    if ci.high < .9:
        flags.append('Low containment')
    return {'available': True, 'historical_n': len(h), 'ensemble_n': len(g),
            'historical_mean': float(h.mean()), 'ensemble_mean': float(g.mean()),
            'mean_bias': float(g.mean() - h.mean()), 'mean_bias_ci_low': mean_ci[0],
            'mean_bias_ci_high': mean_ci[1], 'historical_sd': hs, 'ensemble_sd': gs,
            'sd_ratio': gs / hs if hs > 1e-12 else None,
            'sd_ratio_ci_low': sd_ci[0], 'sd_ratio_ci_high': sd_ci[1],
            'historical_p5': float(hq[0]), 'historical_median': float(hq[1]), 'historical_p95': float(hq[2]),
            'ensemble_p5': float(gq[0]), 'ensemble_median': float(gq[1]), 'ensemble_p95': float(gq[2]),
            'band_width_ratio': float((gq[2]-gq[0])/(hq[2]-hq[0])) if hq[2] > hq[0] else None,
            'inside_count': inside, 'containment_percent': 100 * inside / len(h),
            'containment_ci_low_percent': 100 * ci.low, 'containment_ci_high_percent': 100 * ci.high,
            'flags': flags, 'assessment': '; '.join(flags) if flags else
            ('Undefined historical spread' if hs <= 1e-12 else
             'Limited sample' if len(h) < 20 else 'No clear mismatch')}


def envelope_scores(report):
    """Keep entire historical seasons intact; do not treat daily counts as independent."""
    daily, summary = [], []
    for key, entry in report['envelopes'].items():
        hist = np.asarray([row['series'][key] for row in report['historical']], dtype=float)
        low, high = np.asarray(entry['ensemble']['5'], float), np.asarray(entry['ensemble']['95'], float)
        hlo, hhi = np.asarray(entry['historical']['5'], float), np.asarray(entry['historical']['95'], float)
        valid = np.isfinite(hist).all(axis=0) & np.isfinite(low) & np.isfinite(high)
        widths = []
        for i in np.flatnonzero(valid):
            width = (high[i]-low[i])/(hhi[i]-hlo[i]) if hhi[i] > hlo[i] else None
            inside = int(np.sum((hist[:, i] >= low[i]) & (hist[:, i] <= high[i])))
            daily.append({'experiment': report['experiment'], 'comparison_id': case_key(report), 'trajectory': key, 'date': report['dates'][i],
                          'historical_n': len(hist), 'inside_count': inside,
                          'containment_percent': 100*inside/len(hist), 'band_width_ratio': width})
            if width is not None:
                widths.append(width)
        summary.append({'experiment': report['experiment'], 'comparison_id': case_key(report), 'trajectory': key,
                        'valid_days': int(valid.sum()),
                        'pointwise_containment_percent': entry['historical_pointwise_containment_percent'],
                        'median_band_width_ratio': float(np.median(widths)) if widths else None})
    return daily, summary


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows({k: '; '.join(v) if isinstance(v, list) else v for k, v in row.items()}
                         for row in rows)


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def case_key(report):
    return report.get('comparison_id', report['experiment'])


def case_title(report):
    years = report['historical_years']
    name = report.get('window_label') or ('In sample' if report['comparison_mode'] == 'in_sample' else 'Held out')
    return f'{name}: {years[0]}–{years[-1]}'


def make_figures(reports, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#334b5c',
                         'text.color': '#173347', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    directory = output / 'figures'
    directory.mkdir(exist_ok=True)
    figures = []
    def save(fig, name, caption):
        fig.savefig(directory / f'{name}.png', dpi=155, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        figures.append({'name': name, 'caption': caption, 'path': directory/f'{name}.png'})
        plt.close(fig)
    def band(ax, x, values, group):
        color = HIST if group == 'Historical' else EGGS
        ax.fill_between(x, np.asarray(values['5'], float), np.asarray(values['95'], float),
                        color=color, alpha=.18, label=group+' P5–P95')
        ax.plot(x, np.asarray(values['50'], float), color=color, lw=1.8,
                ls='--' if group == 'Historical' else '-', label=group+' median')
    with PdfPages(output/'assessment.pdf') as pdf:
        for i, report in enumerate(reports):
            dates = np.asarray(report['dates'], dtype='datetime64[D]')
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            for ax, (key, title, unit) in zip(axes.flat, [
                    ('cumulative_rain','Cumulative precipitation','mm'),
                    ('cumulative_pet','Cumulative reference ETo','mm'),
                    ('cumulative_balance','Cumulative precipitation − ETo','mm'),
                    ('temperature_7day','Trailing 7-day mean temperature','°C')]):
                entry = report['envelopes'][key]
                band(ax, dates, entry['historical'], 'Historical')
                band(ax, dates, entry['ensemble'], 'EGGS')
                if report['historical_seasons'] <= 6:
                    for season in report['historical']:
                        ax.plot(dates, np.asarray(season['series'][key], float), color=HIST, alpha=.24, lw=.65)
                ax.set(title=title, ylabel=unit)
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.grid(alpha=.16)
                ax.text(.02, .96, f"Historical season-days inside EGGS band: {entry['historical_pointwise_containment_percent']:.1f}%",
                        transform=ax.transAxes, va='top', fontsize=8,
                        bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
            axes[0,0].legend(fontsize=8, loc='lower right')
            fig.suptitle(f"{case_title(report)} · {report['historical_seasons']} historical / {report['ensemble_members']} generated seasons\n"
                         f"Temperature reference: {report['temperature_reference_date']} · Pointwise bands, not trajectory probabilities", fontsize=12)
            save(fig, f'{i+1:02d}_envelopes', 'Compare band width and location; orange = history, teal = EGGS.')
        fig, axes = plt.subplots(len(reports), 3, figsize=(12, 2.7*len(reports)), squeeze=False, constrained_layout=True)
        for axes_row, report in zip(axes, reports):
            x = np.arange(len(report['months']))
            for ax, key, title in zip(axes_row, ['rain','temperature','pet'], ['Rainfall · mm','Mean temperature · °C','Reference ETo · mm']):
                band(ax, x, report['monthly'][key]['historical'], 'Historical')
                band(ax, x, report['monthly'][key]['ensemble'], 'EGGS')
                ax.set(title=title, xticks=x, xticklabels=[m[5:] for m in report['months']])
                ax.grid(alpha=.16)
            axes_row[0].set_ylabel(case_title(report), fontsize=9)
        axes[0,0].legend(fontsize=7, loc='best')
        fig.suptitle('Monthly P5–P95 envelopes · Compare within each row; reference climate is fitted separately', fontsize=12)
        save(fig, '05_monthly', 'Monthly envelopes identify seasonal timing and spread errors hidden by whole-period averages.')
        fig, axes = plt.subplots(len(reports), 3, figsize=(12, 2.7*len(reports)), squeeze=False, constrained_layout=True)
        for axes_row, report in zip(axes, reports):
            for ax, metric in zip(axes_row, ['total_precipitation_mm','longest_dry_spell_days','peak_temperature_C']):
                definition = report['comparisons'][metric]
                for group, label, color in [('historical','Historical',HIST),('ensemble','EGGS',EGGS)]:
                    values = np.sort(numeric([row['values'].get(metric) for row in report[group]]))
                    ax.step(np.r_[values[0],values],np.r_[0,100*np.arange(1,len(values)+1)/len(values)],
                            where='post',color=color,label=label,lw=1.7)
                ax.set(xlabel=f"{definition['label']} · {definition['unit']}", ylim=(0,101))
                ax.grid(alpha=.16)
            axes_row[0].set_ylabel(case_title(report)+'\nCumulative seasons · %', fontsize=9)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Seasonal distributions · Narrow horizontal range = insufficient variability', fontsize=12)
        save(fig, '06_distributions', 'Compare the full distribution and tails of rainfall, dry spells and peak hourly cell temperature.')
        fig, axes = plt.subplots(1, 3, figsize=(13, 7), constrained_layout=True)
        colors = ['#087f8c','#7456a0','#af6a21','#317eb1']
        for j, report in enumerate(reports):
            selected = {row['metric']: row for row in rows if row['comparison_id']==case_key(report)}
            y = np.arange(len(PRIMARY)) + (j-(len(reports)-1)/2)*.17
            color = colors[j%len(colors)]
            for ax, center, lower, upper in [(axes[0],'mean_bias','mean_bias_ci_low','mean_bias_ci_high'),
                                             (axes[1],'sd_ratio','sd_ratio_ci_low','sd_ratio_ci_high'),
                                             (axes[2],'containment_percent','containment_ci_low_percent','containment_ci_high_percent')]:
                for k, metric in enumerate(PRIMARY):
                    row=selected.get(metric,{})
                    value=row.get(center)
                    lo,hi=row.get(lower),row.get(upper)
                    if value is None or lo is None or hi is None:
                        continue
                    if center=='mean_bias':
                        scale=row['historical_sd']
                        if scale <= 1e-12:
                            continue
                        value,lo,hi=value/scale,lo/scale,hi/scale
                    ax.plot([lo,hi],[y[k],y[k]],color=color,alpha=.8,lw=1.2)
                    ax.plot(value,y[k],'o',ms=4,color=color,label=case_title(report) if k==0 else None)
        for ax in axes:
            ax.invert_yaxis()
            ax.set_yticks(np.arange(len(PRIMARY)), LABELS if ax is axes[0] else [])
            ax.grid(axis='x',alpha=.16)
        axes[0].axvline(0,color='#8899a3',ls='--',lw=1)
        axes[1].axvline(1,color='#8899a3',ls='--',lw=1)
        axes[2].axvline(90,color='#8899a3',ls='--',lw=1)
        axes[0].set_title('Mean bias / historical SD\nIdeal: 0',fontsize=11)
        axes[1].set_xscale('log', base=2)
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        axes[1].set_title('Generated SD / historical SD\nIdeal: 1 · log scale',fontsize=11)
        axes[2].set(title='Historical values inside P5–P95\nNominal: 90%', xlim=(0,102), xlabel='% of seasons')
        handles, legends = axes[0].get_legend_handles_labels()
        fig.legend(handles, legends, fontsize=8, loc='outside lower center', ncol=2)
        fig.suptitle('Window diagnostics · 95% conditional sampling intervals',fontsize=12)
        save(fig, '07_diagnostics', 'Left: mean shift; centre: variability; right: containment. These are exploratory checks, not application acceptance thresholds.')
        fig, axes = plt.subplots(len(reports),2,figsize=(11,2.5*len(reports)),squeeze=False,constrained_layout=True)
        for axes_row, report in zip(axes,reports):
            spatial=report['spatial_correlation']
            x=np.arange(len(spatial))
            for ax,key,title in zip(axes_row,['rainfall','wet_occurrence'],['Daily rainfall correlation','Wet-occurrence correlation']):
                ax.plot(x,[r[key+'_historical_mean_correlation'] for r in spatial],'o--',color=HIST,label='Historical')
                ax.plot(x,[r[key+'_ensemble_mean_correlation'] for r in spatial],'o-',color=EGGS,label='EGGS')
                ax.set(title=title,xticks=x,xticklabels=[r['month'] for r in spatial],ylim=(0,1))
                ax.grid(alpha=.16)
            axes_row[0].set_ylabel(case_title(report),fontsize=9)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Spatial coherence · Final weather correlations, including zero rainfall',fontsize=12)
        save(fig, '08_spatial', 'Averaged cell-pair Pearson correlations. Cell pairs are dependent; their count is not an independent sample size.')
        fig, axes = plt.subplots(len(reports), 3, figsize=(12, 3.2*len(reports)), squeeze=False, constrained_layout=True)
        for axes_row, report in zip(axes, reports):
            for ax, metric in zip(axes_row, ['total_precipitation_mm','peak_temperature_C','pet_mm']):
                definition = report['comparisons'][metric]
                h = numeric([row['values'].get(metric) for row in report['historical']])
                g = numeric([row['values'].get(metric) for row in report['ensemble']])
                if min(len(h),len(g)) < 2:
                    continue
                hq,gq = np.percentile(h,PERCENTILES),np.percentile(g,PERCENTILES)
                lo,hi = min(hq.min(),gq.min()),max(hq.max(),gq.max())
                margin = max(.01,(hi-lo)*.08)
                ax.plot([lo-margin,hi+margin],[lo-margin,hi+margin],color='#9aa8b0',ls='--',lw=1,label='Equal percentiles')
                ax.plot(hq,gq,'o-',color=EGGS,lw=1,ms=5)
                for j in [0,3,6]:
                    ax.annotate(f'P{PERCENTILES[j]}',(hq[j],gq[j]),xytext=(4,5),textcoords='offset points',fontsize=8)
                ax.set(title=definition['label'],xlabel=f"Historical · {definition['unit']}",
                       ylabel=f"EGGS · {definition['unit']}",xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin))
                ax.grid(alpha=.16)
            axes_row[0].set_ylabel(case_title(report)+'\nEGGS · mm',fontsize=9)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('P5 / P10 / P25 / P50 / P75 / P90 / P95 · Agreement follows the dashed line',fontsize=12)
        save(fig,'09_percentiles','Direct percentile comparison. Curvature or departures from the diagonal show spread, tail and location errors.')
        if all('temperature_range' in report['monthly'] for report in reports):
            fig, axes = plt.subplots(len(reports), 1, figsize=(11, 3*len(reports)), squeeze=False, constrained_layout=True)
            for axes_row, report in zip(axes,reports):
                ax=axes_row[0]
                x=np.arange(len(report['months']))
                band(ax,x,report['monthly']['temperature_range']['historical'],'Historical')
                band(ax,x,report['monthly']['temperature_range']['ensemble'],'EGGS')
                ax.set(title=case_title(report),xticks=x,xticklabels=[month[5:] for month in report['months']],ylabel='Daily Tmax − Tmin · °C')
                ax.grid(alpha=.16)
            axes[0,0].legend(fontsize=8)
            fig.suptitle('Daily temperature range · Historical and generated monthly distributions',fontsize=12)
            save(fig,'10_temperature_range','Temperature range is an input to Hargreaves ETo. This diagnostic separates diurnal-range errors from mean-temperature errors.')
    return figures


def build_assessment(paths, output, resamples=5000, seed=20261020, diagnostic=None):
    paths=[Path(path).resolve() for path in paths]
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    reports=[json.loads(path.read_text()) for path in paths]
    diagnostic_path=Path(diagnostic).resolve() if diagnostic else None
    adjustment=json.loads(diagnostic_path.read_text()) if diagnostic_path else None
    if adjustment and not all(path.parent.parent==Path(adjustment['adjusted']) for path in paths):
        raise ValueError('Diagnostic must describe the assessed ensemble.')
    for r in reports:
        if r['comparison_mode']=='out_of_sample' and r['overlapping_training_years']:
            raise ValueError('Held-out report contains training overlap.')
    for r in reports[1:]:
        if (r['mask']!=reports[0]['mask'] or
                r['latitude']!=reports[0]['latitude'] or r['longitude']!=reports[0]['longitude']):
            raise ValueError('Assess identical spatial domains.')
    if len({case_key(r) for r in reports}) != len(reports):
        raise ValueError('Duplicate comparison identifiers.')
    rows, daily, envelope, spatial=[],[],[],[]
    percentile_rows=[]
    for r in reports:
        for metric, definition in r['comparisons'].items():
            score=score_metric([v['values'].get(metric) for v in r['historical']],
                               [v['values'].get(metric) for v in r['ensemble']],resamples,seed)
            rows.append({'experiment':r['experiment'],'comparison_id':case_key(r),'comparison':case_title(r),'metric':metric,
                         'label':definition['label'],'unit':definition['unit'],**score})
        for metric,definition in r['comparisons'].items():
            h=numeric([row['values'].get(metric) for row in r['historical']])
            g=numeric([row['values'].get(metric) for row in r['ensemble']])
            if min(len(h),len(g)) < 2:
                continue
            for q,hv,gv in zip(PERCENTILES,np.percentile(h,PERCENTILES),np.percentile(g,PERCENTILES)):
                percentile_rows.append({'experiment':r['experiment'],'comparison_id':case_key(r),'comparison':case_title(r),
                                        'metric':metric,'label':definition['label'],'unit':definition['unit'],
                                        'percentile':q,'historical':float(hv),'eggs':float(gv),'difference':float(gv-hv)})
        d,s=envelope_scores(r)
        daily.extend(d);envelope.extend(s)
        spatial.extend({'experiment':r['experiment'],'comparison_id':case_key(r),**row} for row in r['spatial_correlation'])
    write_csv(output/'monthly_comparison.csv',[{'experiment':r['experiment'],'comparison_id':case_key(r),**row} for r in reports for row in r.get('monthly_comparisons',[])])
    write_csv(output/'percentiles.csv',percentile_rows)
    write_csv(output/'metric_scores.csv',rows)
    write_csv(output/'envelope_scores.csv',envelope)
    write_csv(output/'daily_containment.csv',daily)
    write_csv(output/'spatial_scores.csv',spatial)
    figures=make_figures(reports,rows,output)
    evidence=[]
    for path,r in zip(paths,reports):
        evidence.append({'source_report':str(path),'comparison_id':case_key(r),'window_start':r['start'],'window_end':r['last_day'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         **{k:r[k] for k in ['experiment','comparison_mode','training_years','historical_years',
                              'overlapping_training_years','ensemble_members','historical_seasons','temperature_reference_date']}})
    findings=[]
    for report in reports:
        selected={row['metric']:row for row in rows if row['comparison_id']==case_key(report)}
        rain=selected['total_precipitation_mm']
        findings.append(f"{report.get('window_label','Period')} rainfall: SD {fmt(rain['sd_ratio'])}× history; {rain['containment_percent']:.1f}% of historical totals inside EGGS P5–P95.")
    for metric,label,unit in [('peak_temperature_C','Peak hourly cell temperature','°C'),('minimum_temperature_C','Lowest hourly cell temperature','°C'),('pet_mm','Reference ETo','mm')]:
        values=[row for row in rows if row['metric']==metric and row.get('available')]
        if not values:
            continue
        findings.append(label+' mean bias: '+ '; '.join(f"{row['comparison'].split(':')[0]} {row['mean_bias']:+.1f} {unit}" for row in values)+'.')
    for metric,label in [('pet_mm','Reference ETo'),('mean_daily_temperature_range_C','Mean daily temperature range')]:
        values=[row for row in rows if row['metric']==metric and row.get('available')]
        if values:
            findings.append(label+' SD / historical SD: '+ '; '.join(f"{row['comparison'].split(':')[0]} {row['sd_ratio']:.2f}×" for row in values)+'.')
    if adjustment:
        findings.append(f"Matched N={adjustment['paired_members']} diagnostic: postprocessing further narrows rainfall and temperature-range variability; see the adjustment table below.")
    result={'created':datetime.now(ZoneInfo('Europe/Rome')).isoformat(),'resamples':resamples,'seed':seed,'findings':findings,
            'sources':evidence,'methods':METHODS,'method_references':SOURCES,'metrics':rows,'envelopes':envelope,
            'percentiles':percentile_rows,'figure_count':len(figures),'score_table_count':2+len(reports)+bool(adjustment),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if adjustment:
        result['adjustment_diagnostic']={'source':str(diagnostic_path),'sha256':hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),**adjustment}
    (output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    sections=[]
    markdown=['# EGGS historical-envelope assessment','',
              '**Verdict: distribution mismatches detected.** No application-specific acceptance thresholds agreed.','',
              '| Comparison | Training | Evaluation | Generated / historical |',
              '| --- | --- | --- | --- |']
    for r in reports:
        years=r['training_years'];held=r['historical_years']
        markdown.append(f"| {case_title(r)} | {years[0]}–{years[-1]} ({len(years)} years) | {held[0]}–{held[-1]} | {r['ensemble_members']} / {r['historical_seasons']} |")
    markdown+=['', *['- '+finding for finding in findings], '', '| Metric | Comparison | Historical mean | EGGS mean | SD ratio | Inside P5–P95 | Diagnostic |',
               '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in reports:
        cells=[]
        for row in rows:
            if row['comparison_id']!=case_key(r) or row['metric'] not in PRIMARY:
                continue
            if not row.get('available'):
                continue
            values=[row['label']+(f" ({row['unit']})" if row['unit'] else ''),
                    fmt(row['historical_mean']),fmt(row['ensemble_mean']),fmt(row['sd_ratio'])+'×',
                    f"{row['inside_count']}/{row['historical_n']} ({row['containment_percent']:.0f}%)",row['assessment']]
            klass='flag' if row['flags'] else 'limited'
            cells.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in values[:-1])+f'<td class="{klass}">{html.escape(values[-1])}</td></tr>')
            markdown.append('| '+' | '.join([values[0],case_title(r),*values[1:]])+' |')
        case_rows=[row for row in rows if row['comparison_id']==case_key(r) and row['metric'] in PRIMARY and row.get('available')]
        count=sum(bool(row.get('flags')) for row in case_rows)
        tag='Training fit' if r['comparison_mode']=='in_sample' else ('Forward holdout' if min(r['historical_years'])>max(r['training_years']) else 'Blocked climatology holdout')
        sections.append(f'<section><h2>{html.escape(case_title(r))}</h2><p>{tag} · {r["ensemble_members"]} generated / {r["historical_seasons"]} historical seasons · {count}/{len(case_rows)} primary metrics flagged.</p>'
                        '<div class="scroll"><table><thead><tr><th>Metric</th><th>History mean</th><th>EGGS mean</th><th>SD ratio<br>target 1</th><th>Inside P5–P95<br>nominal 90%</th><th>Diagnostic</th></tr></thead><tbody>'
                        +''.join(cells)+'</tbody></table></div></section>')
    markdown+=['','- SD ratio = generated/historical variability; <1 means narrower spread.',
               '- Flags use conditional sampling intervals; no clear mismatch does not establish equivalence.',
               '- Fit uses all available historical years; comparisons use complete calendar windows. No holdout tests.',
               '- Main files: `index.html`, `assessment.pdf`, `metric_scores.csv`; details/provenance: `assessment.json`.','']
    if adjustment:
        markdown += ['## Adjustment check · matched N='+str(adjustment['paired_members']), '',
                     '| Annual metric | Historical mean / SD | Adjustment on mean / SD | Off mean / SD |',
                     '| --- | ---: | ---: | ---: |']
        for row in adjustment['rows']:
            if row['window']!='Full year' or row['metric']=='mean_temperature_C':
                continue
            label={'total_precipitation_mm':'Rainfall, mm','pet_mm':'Reference ETo, mm','mean_daily_temperature_range_C':'Mean daily temperature range, °C'}[row['metric']]
            markdown.append('| '+label+' | '+ ' | '.join(f"{row[a+'_mean']:.2f} / {row[a+'_sd']:.2f}" for a in ['historical','adjusted','unadjusted'])+' |')
        markdown += ['', 'Same fit and seeds. Disabling adjustment also changes means; this is a diagnosis, not a validated fix.', '']
    (output/'SUMMARY.md').write_text('\n'.join(markdown))
    image_html=[]
    for figure in figures:
        data=base64.b64encode(figure['path'].read_bytes()).decode()
        image_html.append('<figure><img alt="'+html.escape(figure['name'])+'" src="data:image/png;base64,'+data+'"><figcaption>'+html.escape(figure['caption'])+'</figcaption></figure>')
    csv_data=base64.b64encode((output/'metric_scores.csv').read_bytes()).decode()
    percentile_data=base64.b64encode((output/'percentiles.csv').read_bytes()).decode()
    any_flags=any(row.get('flags') for row in rows if row['metric'] in PRIMARY)
    verdict='Historical envelope: mismatches detected' if any_flags else 'Historical envelope: no clear mismatch detected'
    # Do not claim mismatch when the input reports provide no such evidence.
    if not any_flags:
        text=(output/'SUMMARY.md').read_text().replace('**Verdict: distribution mismatches detected.**','**Verdict: no clear mismatch detected; equivalence not established.**')
        (output/'SUMMARY.md').write_text(text)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EGGS historical-envelope assessment</title><style>
*{box-sizing:border-box}body{font:14px/1.45 system-ui,sans-serif;background:#f3f6f8;color:#173347;margin:0}main{max-width:1250px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:18px}p,figcaption{color:#526776}section,figure,details{background:white;border:1px solid #dce5e9;border-radius:8px;padding:18px;margin:18px 0}.verdict{border-left:5px solid #b66a28;background:#fff8ed;padding:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;text-align:right;border-bottom:1px solid #e1e8eb;white-space:nowrap}th:first-child,td:first-child{text-align:left}details td:last-child{text-align:left}.flag{color:#8d391b;font-weight:600}.limited{color:#657480}img{width:100%;height:auto}figcaption{font-size:12px}a{color:#087f8c}summary{cursor:pointer;font-weight:600}@media(max-width:600px){main{padding:12px}section,figure{padding:10px}h1{font-size:23px}}
</style><main><h1>Does EGGS reproduce the historical envelope?</h1>'''
    page+=f'<p class="verdict"><strong>{html.escape(verdict)}</strong><br>Compare mean, spread and tails. These checks do not define fitness for reservoir or irrigation risk decisions.</p>'
    page+=f'<p>N = {reports[0]["ensemble_members"]} · {reports[0]["land_cells"]} cells · orange: history · teal: EGGS. Compare each window with matching historical dates.</p>'
    page+=f'<p><a download="metric_scores.csv" href="data:text/csv;base64,{csv_data}">Download all metric scores (CSV)</a> · Print/export figures: assessment.pdf beside this file.</p>'
    page+=f'<p><a download="percentiles.csv" href="data:text/csv;base64,{percentile_data}">Download P5–P95 comparisons (CSV)</a></p>'
    page+='<ul>'+''.join('<li>'+html.escape(finding)+'</li>' for finding in findings)+'</ul>'
    page+='<section><h2>Rainfall percentiles · mm</h2><p>Each value: historical / EGGS.</p><div class="scroll"><table><thead><tr><th>Window</th>'
    page+=''.join(f'<th>P{q}</th>' for q in PERCENTILES)+'</tr></thead><tbody>'
    for report in reports:
        page+='<tr><td>'+html.escape(case_title(report))+'</td>'
        for q in PERCENTILES:
            value=next(row for row in percentile_rows if row['comparison_id']==case_key(report) and row['metric']=='total_precipitation_mm' and row['percentile']==q)
            page+=f"<td>{value['historical']:.1f} / {value['eggs']:.1f}</td>"
        page+='</tr>'
    page+='</tbody></table></div></section>'
    page+='<section id="spread-scores"><h2>Spread and containment</h2><p>Each cell: generated/historical SD ratio · historical containment. Targets: 1× · 90%. Detailed means and flags below.</p><div class="scroll"><table><thead><tr><th>Metric</th>'
    page+=''.join('<th>'+html.escape(case_title(r))+'</th>' for r in reports)+'</tr></thead><tbody>'
    for metric,label in zip(PRIMARY,LABELS):
        page+='<tr><td>'+html.escape(label)+'</td>'
        for r in reports:
            row=next(row for row in rows if row['comparison_id']==case_key(r) and row['metric']==metric)
            value=f"{fmt(row.get('sd_ratio'))}× · {fmt(row.get('containment_percent'),0)}%"
            klass='flag' if row.get('flags') else 'limited'
            page+=f'<td class="{klass}">{value}</td>'
        page+='</tr>'
    page+='</tbody></table></div></section>'
    ordered=[i for i,f in enumerate(figures) if f['name']=='09_percentiles']+[i for i,f in enumerate(figures) if f['name']!='09_percentiles']
    titles={'05_monthly':'Monthly envelopes','06_distributions':'Seasonal distributions','08_spatial':'Spatial rainfall coherence','07_diagnostics':'Mean, spread and containment intervals','10_temperature_range':'Daily temperature range'}
    for i in ordered:
        figure=figures[i]
        if figure['name'] in ('09_percentiles','01_envelopes'):
            page+=image_html[i]
        else:
            title=titles.get(figure['name'])
            if title is None:
                case_number=int(figure['name'][:2])-1
                title=case_title(reports[case_number])+' · weather envelopes'
            page+='<details><summary>'+html.escape(title)+'</summary>'+image_html[i]+'</details>'
    for r,section in zip(reports,sections):
        page+='<details><summary>'+html.escape(case_title(r))+' · detailed scores</summary>'+section+'</details>'
    if adjustment:
        page+='<section id="adjustment-check"><h2>Adjustment check · matched N='+str(adjustment['paired_members'])+'</h2><p>Same fitted model and seeds. Values: annual mean / SD. This N=50 diagnostic is separate from the N=1,000 comparison.</p><div class="scroll"><table><thead><tr><th>Metric</th><th>History</th><th>Adjustment on</th><th>Adjustment off</th></tr></thead><tbody>'
        for row in adjustment['rows']:
            if row['window']!='Full year' or row['metric']=='mean_temperature_C':
                continue
            label={'total_precipitation_mm':'Rainfall, mm','pet_mm':'Reference ETo, mm','mean_daily_temperature_range_C':'Mean daily temperature range, °C'}[row['metric']]
            page+='<tr><td>'+label+'</td>'+''.join(f"<td>{row[a+'_mean']:.2f} / {row[a+'_sd']:.2f}</td>" for a in ['historical','adjusted','unadjusted'])+'</tr>'
        page+='</tbody></table></div><p>Adjustment further narrows both rainfall and temperature-range variability. Disabling it also changes mean rainfall and ETo; this is not a validated fix.</p></section>'
    page+='<details><summary>Methods and limits</summary><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in METHODS)+'</ul><p>Fit: all available years. Complete annual/seasonal windows are compared separately. February 29 remains in totals and spells; only its daily plot slot is omitted for non-leap alignment.</p><p>References: '+', '.join(f'<a href="{url}">SciPy method {i+1}</a>' for i,url in enumerate(SOURCES))+'</p></details></main></html>'
    (output/'index.html').write_text(page)
    print(f'Assessment: {output / "index.html"}',flush=True)
    return result
