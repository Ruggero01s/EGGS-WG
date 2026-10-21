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
    args = parser.parse_args()
    if args.command == 'generate':
        params = yaml.safe_load(args.config.read_text())
        for key, value in (('Number of Simulations', args.n), ('Simulation name', args.name), ('Random Seed', args.seed)):
            if value is not None:
                params[key] = value
        params['Scenario Metrics'] = True
        run(params, args.config, prepare_only=args.prepare_only)
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
