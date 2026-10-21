"""Compatibility entry point: Main.py CONFIG [--prepare-only]."""

import os
import sys


if __name__ == '__main__':
    for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '1'
    from ai4water_eggs.cli import main

    sys.argv.insert(1, 'generate')
    main()
