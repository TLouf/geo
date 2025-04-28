import geopandas as gpd
import pandas as pd
import shapely.geometry

from geo.dependencies import h3


def to_polygon(h3_hex):
    """Convert an H3 hexagon index to a Shapely polygon."""
    h3_shape = h3.cells_to_h3shape(list(h3_hex))
    return shapely.geometry.shape(h3_shape)


def grid(area, level=9):
    hexs = h3.geo_to_cells(area, level)
    geo = [h3.cells_to_h3shape([h]) for h in hexs]
    grid = gpd.GeoDataFrame(
        geometry=geo, index=pd.Index(hexs, name="cell_id"), crs="EPSG:4326"
    )
    return grid
