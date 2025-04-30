import geopandas as gpd
import shapely.geometry

from geo.dependencies import pygeohash as pgh


def to_polygon(geohash: str):
    """Convert a geohash to a Shapely polygon."""
    b = pgh.get_bounding_box(geohash)
    return shapely.geometry.box(b.min_lon, b.min_lat, b.max_lon, b.max_lat)


def bbox_from_shapely(shape):
    min_lon, min_lat, max_lon, max_lat = shape.bounds
    return pgh.BoundingBox(min_lat, min_lon, max_lat, max_lon)


def grid(area, level=9, clip=False):
    bbox = bbox_from_shapely(area)
    geohashes = pgh.geohashes_in_box(bbox, precision=level)
    gh_geos = [to_polygon(h) for h in geohashes]
    grid = gpd.GeoDataFrame(index=geohashes, geometry=gh_geos, crs="epsg:4326")
    if clip:
        grid = grid.clip(area)
    else:
        grid = grid.loc[grid.geometry.intersects(area)]
    return grid
