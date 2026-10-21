"""Build figures, tables and a concise assessment from existing validation reports."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai4water_eggs.assessment import build_assessment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--diagnostic', type=Path, help='Optional matched-adjustment diagnostic JSON')
    args = parser.parse_args()
    build_assessment(args.reports, args.output, args.resamples, args.seed, args.diagnostic)
