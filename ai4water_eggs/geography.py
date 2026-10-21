"""Read the project study boundary and map its cell centres to the model grid."""

import json
from functools import lru_cache
from pathlib import Path
import subprocess
import sys

import numpy as np
import shapefile
from shapely import covers, points
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union


def project_boundary(path):
    from pyproj import CRS, Transformer
    path = Path(path)
    with shapefile.Reader(str(path)) as source:
        polygon = unary_union([shape(item.__geo_interface__) for item in source.shapes()])
    crs = CRS.from_wkt(path.with_suffix('.prj').read_text())
    projector = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    return transform(projector.transform, polygon)


@lru_cache(maxsize=16)
def read_boundary(path):
    # The installed ecCodes and pyproj wheels have conflicting native symbols.
    # Keep projection in a small independent process; masking stays in-process.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(Path(path).resolve())],
        check=True, capture_output=True, text=True)
    return shape(json.loads(completed.stdout))


def boundary_mask(path, latitude, longitude):
    polygon = read_boundary(path)
    x, y = np.meshgrid(longitude, latitude)
    return covers(polygon, points(x, y))


def write_boundary(path, output):
    Path(output).write_text(json.dumps(mapping(read_boundary(path))) + '\n')


if __name__ == '__main__':
    print(json.dumps(mapping(project_boundary(sys.argv[1]))))
