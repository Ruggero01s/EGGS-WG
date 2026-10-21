"""Check the local runtime with tiny synthetic files; no CDS calls or model fitting."""

import importlib
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Scripts run directly from scripts/ need the upstream modules at the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eccodes
import numpy as np
import xarray as xr

import SWG_Build_Functions as build
import SWG_Model as model


def check_grib(directory):
    """Exercise the native ecCodes library and xarray's cfgrib backend together."""
    path = directory / "sample.grib"
    message = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        for key, value in {
            "Ni": 2,
            "Nj": 2,
            "latitudeOfFirstGridPointInDegrees": 42.0,
            "latitudeOfLastGridPointInDegrees": 41.0,
            "longitudeOfFirstGridPointInDegrees": 15.0,
            "longitudeOfLastGridPointInDegrees": 16.0,
            "iDirectionIncrementInDegrees": 1.0,
            "jDirectionIncrementInDegrees": 1.0,
            "dataDate": 19900101,
            "dataTime": 0,
            "shortName": "2t",
        }.items():
            eccodes.codes_set(message, key, value)
        expected = np.array([[280.0, 281.0], [282.0, 283.0]])
        eccodes.codes_set_values(message, expected.ravel())
        with path.open("wb") as stream:
            eccodes.codes_write(message, stream)
    finally:
        eccodes.codes_release(message)

    with xr.open_dataset(
        path, engine="cfgrib", backend_kwargs={"indexpath": ""},
        decode_timedelta=False,
    ) as dataset:
        np.testing.assert_allclose(dataset["t2m"].values, expected)
        loaded = dataset.load()
        for engine in ("netcdf4", "h5netcdf"):
            output = directory / f"sample_{engine}.nc"
            loaded.to_netcdf(output, engine=engine)
            with xr.open_dataset(output, engine=engine, decode_timedelta=False) as restored:
                xr.testing.assert_allclose(loaded, restored)
    print("PASS: GRIB decoding and NetCDF round trips (netcdf4, h5netcdf)")


def check_model_functions(directory):
    """Check ERA5-Land accumulation conversion, the wet-day mask, and humidity."""
    cumulative = np.zeros((2, 24, 2, 2))
    cumulative[0] = np.arange(1, 25)[:, None, None] * 0.001
    cumulative[:, :, 1, 1] = np.nan
    source = xr.Dataset(
        {"tp": (("time", "step", "latitude", "longitude"), cumulative)},
        coords={
            "time": np.array(["1990-01-01", "1990-01-02"], dtype="datetime64[ns]"),
            "step": np.arange(1, 25),
            "latitude": [42.0, 41.9],
            "longitude": [15.0, 15.1],
        },
    )
    source.to_netcdf(directory / "tp_1950_2022.nc")
    build.Daily_Precipitation_Extract(str(directory) + "/")
    with xr.open_dataset(directory / "daily_tp_1950_2022.nc") as daily:
        np.testing.assert_allclose(daily["tp"].values[:, 0, 0], [24.0, 0.0])
        np.testing.assert_array_equal(
            daily["Precipitation_mask"].values[:, 0, 0], [True, False]
        )
        assert np.isnan(daily["tp"].values[:, 1, 1]).all()

    np.testing.assert_allclose(model.RH_Calculate(np.array([280.0]), np.array([280.0])), 1.0)
    humidity = model.RH_Calculate(np.array([275.0]), np.array([280.0]))
    assert np.all((humidity > 0.0) & (humidity < 1.0))
    rain = model.Precipitation_Calculate(
        np.array([0.5]), np.array([1.0]), np.array([2.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
    )
    np.testing.assert_allclose(rain, 1.0 + 2.0 * np.log(2.0))
    print("PASS: upstream daily precipitation, wet-day mask, humidity, and rainfall calculations")


def main():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in (
        "numpy", "scipy", "xarray", "cfgrib", "eccodes", "matplotlib",
        "cftime", "yaml", "netCDF4", "h5netcdf", "h5py", "cdsapi",
    ):
        importlib.import_module(name)
    for distribution in ("numpy", "scipy", "xarray", "cfgrib", "eccodes", "netCDF4"):
        print(f"  {distribution}: {version(distribution)}")
    # Compile entry points and helpers without running their commands.
    for path in [*ROOT.glob("*.py"), *(ROOT / "ai4water_eggs").glob("*.py"),
                 *(ROOT / "scripts").glob("*.py")]:
        compile(path.read_bytes(), str(path), "exec")
    print("PASS: runtime imports and Python source compilation")
    with TemporaryDirectory(prefix="eggs-wg-check-") as temporary:
        directory = Path(temporary)
        check_grib(directory)
        check_model_functions(directory)
    print("Setup checks passed. A full model run still requires compatible ERA5-Land inputs.")


if __name__ == "__main__":
    main()
