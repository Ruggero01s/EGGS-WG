"""Losslessly compress existing scenario NetCDFs; verify before atomic replacement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from netCDF4 import Dataset
import numpy as np


def compress_group(paths):
    paths = [Path(p) for p in paths]
    source = paths[0]
    stat = source.stat()
    result = {'file': str(source), 'aliases': len(paths), 'before': stat.st_size}
    if stat.st_nlink != len(paths):
        raise ValueError(f'Links outside scanned root: {source}')
    temporary = None
    try:
        with Dataset(source) as src:
            fields = [v for name, v in src.variables.items()
                      if name.startswith(('simulation_', 'Sim_')) and v.ndim > 1]
            if not fields or all((v.filters() or {}).get('zlib') for v in fields):
                return dict(result, status='already_compressed', after=stat.st_size)
            if src.groups:
                raise ValueError(f'Nested groups unsupported: {source}')
            src.set_auto_maskandscale(False)
            src.set_auto_chartostring(False)
            handle, temporary = tempfile.mkstemp(prefix='.compress-', suffix='.nc', dir=source.parent)
            os.close(handle)
            with Dataset(temporary, 'w', format='NETCDF4') as dst:
                dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
                for name, dimension in src.dimensions.items():
                    dst.createDimension(name, None if dimension.isunlimited() else len(dimension))
                for name, variable in src.variables.items():
                    attrs = {k: variable.getncattr(k) for k in variable.ncattrs()}
                    fill = {'fill_value': attrs.pop('_FillValue')} if '_FillValue' in attrs else {'fill_value': False}
                    compress = variable.ndim > 0 and np.dtype(variable.dtype).kind in 'biufc'
                    output = dst.createVariable(name, variable.datatype, variable.dimensions,
                                                zlib=compress, complevel=1, shuffle=compress, **fill)
                    output.set_auto_maskandscale(False)
                    output.set_auto_chartostring(False)
                    output.setncatts(attrs)
                    output[...] = variable[...]
            with Dataset(temporary) as checked:
                checked.set_auto_maskandscale(False)
                checked.set_auto_chartostring(False)
                assert checked.ncattrs() == src.ncattrs()
                assert list(checked.variables) == list(src.variables)
                for name, variable in src.variables.items():
                    other = checked[name]
                    assert other.dtype == variable.dtype and other.dimensions == variable.dimensions
                    assert other.ncattrs() == variable.ncattrs()
                    for key in variable.ncattrs():
                        np.testing.assert_equal(other.getncattr(key), variable.getncattr(key))
                    assert other[...].tobytes() == variable[...].tobytes(), f'Changed values: {source}:{name}'
                for key in src.ncattrs():
                    np.testing.assert_equal(checked.getncattr(key), src.getncattr(key))
        after = os.stat(temporary).st_size
        if after >= stat.st_size:
            return dict(result, status='no_saving', after=stat.st_size)
        for path in paths:
            current = path.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (stat.st_ino, stat.st_size, stat.st_mtime_ns):
                raise ValueError(f'File changed during compression: {path}')
        os.chmod(temporary, stat.st_mode & 0o777)
        os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, source)
        temporary = None
        for alias in paths[1:]:
            handle, link = tempfile.mkstemp(prefix='.compress-link-', dir=alias.parent)
            os.close(handle)
            os.unlink(link)
            os.link(source, link)
            os.replace(link, alias)
        return dict(result, status='compressed_verified', after=after)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--apply', action='store_true', help='Replace files after exact verification.')
    args = parser.parse_args()
    groups = {}
    for path in args.root.rglob('*.nc'):
        if 'Simulation' not in path.parts or path.name.startswith('.'):
            continue
        stat = path.stat()
        groups.setdefault((stat.st_dev, stat.st_ino), []).append(str(path))
    before = sum(Path(paths[0]).stat().st_size for paths in groups.values())
    print(json.dumps({'unique_weather_files': len(groups), 'before_GiB': before / 2**30}), flush=True)
    if not args.apply:
        return
    with (args.root / '.weather-compression.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.root / f'compression-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.jsonl'
        counts = {}
        after = 0
        with manifest.open('w') as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(compress_group, groups.values(), chunksize=4), 1):
                log.write(json.dumps(result) + '\n')
                log.flush()
                counts[result['status']] = counts.get(result['status'], 0) + 1
                after += result['after']
                if i % 250 == 0:
                    print(json.dumps({'done': i, 'total': len(groups), 'counts': counts}), flush=True)
        summary = {'before_bytes': before, 'after_bytes': after, 'saved_GiB': (before-after)/2**30,
                   'counts': counts, 'manifest': str(manifest), 'values_verified_exact': True}
        manifest.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
