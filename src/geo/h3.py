import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from geo.dependencies import h3


def to_polygon(h3_hex):
    """Convert an H3 hexagon index to a Shapely polygon."""
    boundary = h3.h3_to_geo_boundary(h3_hex, geo_json=True)
    return Polygon(boundary)


def grid(area, level=9):
    hexs = h3.polyfill(area.__geo_interface__, level, geo_json_conformant=True)
    grid = gpd.GeoDataFrame(
        geometry=list(map(to_polygon, hexs)),
        index=pd.Index(hexs, name="cell_id"),
        crs="EPSG:4326",
    )
    return grid
