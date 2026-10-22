"""Optional compact scenario storage; fitted coefficients retain their precision."""

import numpy as np


def write_weather(dataset, path, storage=None):
    """Write weather, retaining explicit storage settings through postprocessing.

    Defaults preserve the original float64 files. Compact mode uses float32
    weather fields and lossless DEFLATE compression of those stored values.
    Coordinates and integer masks are not quantized. At typical temperatures,
    float32 introduces at most a few 1e-5 K of storage rounding per write.
    """
    if storage is None and 'weather_storage_precision' in dataset.attrs:
        storage = {'precision': dataset.attrs['weather_storage_precision'],
                   'compression': bool(dataset.attrs['weather_storage_compression'])}
    if storage is None:
        dataset.to_netcdf(path)
        return
    precision = storage.get('precision', 'float64')
    compression = storage.get('compression', False)
    if precision not in ('float32', 'float64') or type(compression) is not bool:
        raise ValueError('Weather storage requires float32|float64 and a boolean compression setting.')
    dataset.attrs.update(weather_storage_precision=precision, weather_storage_compression=int(compression))
    fields = {'Sim_Precipitation', 'simulation_Precipitation', 'simulation_t2m', 'simulation_d2m', 'simulation_RH'}
    encoding = {}
    for name, variable in dataset.data_vars.items():
        if name in fields and np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {'dtype': precision, 'zlib': compression}
            if compression:
                encoding[name].update(complevel=1, shuffle=True)
    dataset.to_netcdf(path, encoding=encoding)
