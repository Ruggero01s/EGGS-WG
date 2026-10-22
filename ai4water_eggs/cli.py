"""Generate, score, and select regional weather scenarios."""

import argparse
import json
from pathlib import Path

import yaml

from ai4water_eggs.pipeline import run
from ai4water_eggs.metrics import analyze_ensemble, select_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Generate N scenarios and calculate their metrics.')
    generate.add_argument('config', type=Path)
    generate.add_argument('--prepare-only', action='store_true', help='Prepare inputs without fitting or generating weather.')
    generate.add_argument('--n', type=int, help='Override Number of Simulations.')
    generate.add_argument('--name', help='Override Simulation name to keep a separate ensemble.')
    generate.add_argument('--seed', type=int, help='Base seed; scenario i uses base + i - 1.')
    analyze = commands.add_parser('analyze', help='Score existing scenarios without regenerating weather.')
    analyze.add_argument('experiment', type=Path)
    analyze.add_argument('--scales', nargs='+', type=int, default=[1, 3, 6])
    validate = commands.add_parser('validate', help='Compare ensemble envelopes and distributions with matching historical seasons.')
    validate.add_argument('experiment', type=Path)
    validate.add_argument('--reference', type=Path, help='Experiment containing prepared raw historical inputs; defaults to the evaluated experiment.')
    validate.add_argument('--years', nargs=2, type=int, metavar=('FIRST', 'LAST'), help='Inclusive historical season-start years to evaluate.')
    validate.add_argument('--output', type=Path, help='Report directory; defaults to EXPERIMENT/Validation.')
    validate.add_argument('--window', nargs=2, metavar=('START', 'END_EXCLUSIVE'), help='Evaluate a date subset of the generated weather, without regenerating.')
    validate.add_argument('--bootstrap', type=int, default=1000, help='Whole-season resamples for descriptive mean-bias intervals.')
    select = commands.add_parser('select', help='Find the actual scenario closest to a metric percentile.')
    select.add_argument('experiment', type=Path)
    select.add_argument('--metric', default='total_precipitation_mm')
    select.add_argument('--percentile', type=float, required=True)
    select.add_argument('--output', type=Path, help='Write the selected scenario details as JSON.')
    fit = commands.add_parser('fit-correction', help='Fit fixed pooled rainfall/temperature corrections on calibration seeds.')
    fit.add_argument('calibration', type=Path)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--historical', type=Path)
    correct = commands.add_parser('correct', help='Apply a frozen correction to disjoint scenario seeds, then rebuild metrics.')
    correct.add_argument('experiment', type=Path)
    correct.add_argument('--correction', type=Path, required=True)
    correct.add_argument('--output', type=Path, required=True)
    for command in (fit, correct):
        command.add_argument('--no-rain', action='store_true')
        command.add_argument('--no-temperature', action='store_true')
    subset = commands.add_parser('subset', help='Extract a member range into a separate experiment and rebuild metrics.')
    subset.add_argument('experiment', type=Path)
    subset.add_argument('--members', nargs=2, type=int, required=True, metavar=('FIRST','LAST'))
    subset.add_argument('--output', type=Path, required=True)
    subset.add_argument('--label')
    compare = commands.add_parser('compare', help='Build a concise paired variability-trial report.')
    compare.add_argument('--reports', type=Path, nargs='+', required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        params = yaml.safe_load(args.config.read_text())
        for key, value in (('Number of Simulations', args.n), ('Simulation name', args.name), ('Random Seed', args.seed)):
            if value is not None:
                params[key] = value
        params['Scenario Metrics'] = True
        run(params, args.config, prepare_only=args.prepare_only)
    elif args.command == 'fit-correction':
        from ai4water_eggs.corrections import fit_corrections
        fit_corrections(args.calibration, args.output, args.historical,
                        rain=not args.no_rain, temperature=not args.no_temperature)
    elif args.command == 'correct':
        from ai4water_eggs.corrections import apply_corrections
        output = apply_corrections(args.experiment, args.output, args.correction,
                                   rain=not args.no_rain, temperature=not args.no_temperature)
        config = json.loads((output / 'run_config.json').read_text())
        analyze_ensemble(output, scales=config.get('Drought Scales', [1, 3, 6]))
        application = output / 'correction_application.json'
        provenance = json.loads(application.read_text())
        provenance['metrics_status'] = 'Complete; scenario_catalog.json'
        application.write_text(json.dumps(provenance, indent=2) + '\n')
        (output / '.in_progress').unlink(missing_ok=True)
    elif args.command == 'subset':
        from ai4water_eggs.trial import subset_experiment
        output = subset_experiment(args.experiment, args.output, *args.members, args.label)
        config = json.loads((output / 'run_config.json').read_text())
        analyze_ensemble(output, scales=config.get('Drought Scales', [1, 3, 6]))
        (output / '.in_progress').unlink(missing_ok=True)
    elif args.command == 'compare':
        from ai4water_eggs.trial import build_trial_report
        build_trial_report(args.reports, args.output)
    elif args.command == 'analyze':
        analyze_ensemble(args.experiment, args.scales)
    elif args.command == 'validate':
        from ai4water_eggs.validation import validate_ensemble
        validate_ensemble(args.experiment, args.reference, args.years, args.output, args.bootstrap, args.window)
    else:
        catalog = json.loads((args.experiment / 'scenario_catalog.json').read_text())
        result = select_scenario(catalog, args.metric, args.percentile)
        result['weather_files'] = [str((args.experiment / path).resolve()) for path in result['scenario']['files']]
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end='')


if __name__ == '__main__':
    main()
