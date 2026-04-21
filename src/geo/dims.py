from __future__ import annotations

import numpy as np


def deg_to_rad(theta):
    return theta * np.pi / 180


def haversine(lon1, lat1, lon2, lat2, R=6367e3):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = (
        np.sin(deg_to_rad(dlat / 2)) ** 2
        + np.cos(deg_to_rad(lat1))
        * np.cos(deg_to_rad(lat2))
        * np.sin(deg_to_rad(dlon / 2)) ** 2
    )
    theta = 2 * np.arcsin(np.sqrt(hav))
    d = R * theta
    return d


def haversine_expr(lon1_col, lat1_col, lon2_col, lat2_col, R=6367e3):
    dlon = lon2_col - lon1_col
    dlat = lat2_col - lat1_col
    hav = (
        deg_to_rad(dlat / 2).sin() ** 2
        + deg_to_rad(lat1_col).cos()
        * deg_to_rad(lat2_col).cos()
        * deg_to_rad(dlon / 2).sin() ** 2
    )
    theta = 2 * hav.sqrt().arcsin()
    d = R * theta
    return d


def calc_bbox_dims(min_lon, min_lat, max_lon, max_lat):
    # For a given longitude extent, the width is maximum the closer to the
    # equator, so the closer the latitude is to 0.
    eq_crossed = min_lat * max_lat < 0
    lat_max_width = min(abs(min_lat), abs(max_lat)) * (1 - int(eq_crossed))
    width = haversine(min_lon, lat_max_width, max_lon, lat_max_width)
    height = haversine(min_lon, min_lat, min_lon, max_lat)
    return width, height


def calc_shape_dims(shape_df, latlon_proj="epsg:4326"):
    """
    Calculate the max width and height in meters of the bounding box of the shapes
    contained within `shape_df`.
    """
    min_lon, min_lat, max_lon, max_lat = shape_df.geometry.to_crs(
        latlon_proj
    ).total_bounds
    return calc_bbox_dims(min_lon, min_lat, max_lon, max_lat)
