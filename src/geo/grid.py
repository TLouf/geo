from __future__ import annotations

import geopandas as geopd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box


def load_ext_cells(cells_file_path, index_col, xy_proj='epsg:3857'):
    cells_geodf = geopd.read_file(cells_file_path)
    cells_geodf.columns = cells_geodf.columns.str.lower()
    cells_geodf = (
        cells_geodf.set_index(index_col.lower())
         .sort_index()
         .to_crs(xy_proj)
    )
    return cells_geodf

def gen_quadrants(
    left_x,
    right_x,
    bot_y,
    top_y,
    places_geodf,
    sub_cells_list,
    min_size=1000,
    max_nr_places=10,
):
    """
    Recursively splits each quadrant of the cell defined by the coordinates
    (`left_x`, `right_x`, `bot_y`, `top_y`), as long as there are more than
    `max_nr_places` contained within a subcell, stopping at a minimum cell size
    of `min_size`
    """
    mid_x = (left_x + right_x) / 2
    mid_y = (bot_y + top_y) / 2
    for quadrant in [
        (left_x, mid_x, bot_y, mid_y),
        (left_x, mid_x, mid_y, top_y),
        (mid_x, right_x, bot_y, mid_y),
        (mid_x, right_x, mid_y, top_y),
    ]:
        nr_contained_places = places_geodf.loc[
            (places_geodf["x_c"] < quadrant[1])
            & (places_geodf["x_c"] > quadrant[0])
            & (places_geodf["y_c"] < quadrant[3])
            & (places_geodf["y_c"] > quadrant[2])
        ].shape[0]
        if (
            nr_contained_places > max_nr_places
            and quadrant[1] - quadrant[0] > 2 * min_size
        ):
            gen_quadrants(*quadrant, places_geodf, sub_cells_list)
        else:
            sub_cells_list.append(quadrant)


def create_grid(
    shape_df,
    cell_size,
    region_id,
    xy_proj="epsg:3857",
    intersect=False,
    places_geodf=None,
    max_nr_places=None,
):
    """
    Creates a square grid over a given shape.
    `shape_df` (GeoDataFrame): single line GeoDataFrame containing the shape on
    which the grid is to be created,.
    `cell_size` (int): size of the sides of the square cells which constitute
    the grid, in meters.
    `intersect` (bool): determines whether the function computes
    `cells_in_shape_df`, the intersection of the created grid with the shape of
    the area of interest, so that the grid only covers the shape. Index is
    sorted.
    If `places_geodf` is given, cells are split in 4 when they countain more
    than `max_nr_places` places from this data frame.
    """
    shape_df = shape_df.to_crs(xy_proj)
    if places_geodf is not None:
        places_geodf["x_c"], places_geodf["y_c"] = zip(
            *places_geodf.geometry.apply(lambda geo: geo.centroid.coords[0])
        )
        if max_nr_places is None:
            max_nr_places = places_geodf.shape[0] // 100

    # We have a one line dataframe, we'll take only the geoseries with the
    # geometry, create a new dataframe out of it with the bounds of this series.
    # Then the values attribute is a one line matrix, so we take the first
    # line and get the bounds' in the format (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = shape_df["geometry"].total_bounds
    # We want to cover at least the whole shape of the area, because if we want
    # to restrict just to the shape we can then intersect the grid with its
    # shape. Hence the x,ymax+cell_size in the arange.
    x_grid = np.arange(x_min, x_max + cell_size, cell_size)
    y_grid = np.arange(y_min, y_max + cell_size, cell_size)
    Nx = len(x_grid)
    Ny = len(y_grid)
    # We want (x1 x1 ...(Ny times) x1 x2 x2 ...  xNx) and (y1 y2 ... yNy ...(Nx
    # times)), so we use repeat and tile:
    x_grid_re = np.repeat(x_grid, Ny)
    y_grid_re = np.tile(y_grid, Nx)
    # So then (x_grid_re[i], y_grid_re[i]) for all i are all the edges in the
    # grid
    cells_list = []
    for i in range(Nx - 1):
        left_x = x_grid_re[i * Ny]
        right_x = x_grid_re[(i + 1) * Ny]
        for j in range(Ny - 1):
            bot_y = y_grid_re[j]
            top_y = y_grid_re[j + 1]
            sub_cells_list = [(left_x, right_x, bot_y, top_y)]
            if places_geodf is not None:
                # Can be changed to be maximum population in each grid cell, ...
                nr_contained_places = places_geodf.loc[
                    (places_geodf["x_c"] < right_x)
                    & (places_geodf["x_c"] > left_x)
                    & (places_geodf["y_c"] < top_y)
                    & (places_geodf["y_c"] > bot_y)
                ].shape[0]
                if nr_contained_places > max_nr_places:
                    sub_cells_list = []
                    gen_quadrants(
                        left_x,
                        right_x,
                        bot_y,
                        top_y,
                        places_geodf,
                        sub_cells_list,
                        max_nr_places=max_nr_places,
                        min_size=cell_size / 10,
                    )

            for sub_left_x, sub_right_x, sub_bot_y, sub_top_y in sub_cells_list:
                # The Polygon closes itself, so no need to repeat the first
                # point at the end.
                cells_list.append(
                    Polygon(
                        [
                            (sub_left_x, sub_top_y),
                            (sub_right_x, sub_top_y),
                            (sub_right_x, sub_bot_y),
                            (sub_left_x, sub_bot_y),
                        ]
                    )
                )

    cells_df = geopd.GeoDataFrame(cells_list, crs=xy_proj, columns=["geometry"])
    # Prefix `region_id` to the index to keep a unique index when mixing data from
    # several regions.
    cells_df.index = region_id + "." + cells_df.index.astype(str)
    cells_df["cell_id"] = cells_df.index
    cells_df = cells_df.sort_index()
    if intersect:
        cells_in_shape_df = geopd.clip(cells_df, shape_df)
        cells_in_shape_df = cells_in_shape_df.set_index('cell_id', drop=False)
        cells_in_shape_df.cell_size = cell_size
    else:
        cells_in_shape_df = None

    cells_df.cell_size = cell_size
    return cells_df, cells_in_shape_df, Nx - 1, Ny - 1


def extract_shape(
    raw_shape_df: geopd.GeoDataFrame,
    bbox=None,
    latlon_proj="epsg:4326",
    min_area=None,
    simplify_tol=None,
    xy_proj="epsg:3857",
):
    """
    Extracts the shape of the area of interest. If bbox is provided, in the
    format [min_lon, min_lat, max_lon, max_lat], only keep the intersection of
    the shape with this bounding box. Then the shape we extract is simplified to
    accelerate later computations, first by removing irrelevant polygons inside
    the shape (if it's comprised of more than one), and then simplifying the
    contours.
    """
    shape_df = raw_shape_df.copy()
    if bbox:
        bbox_geodf = geopd.GeoDataFrame(geometry=[box(*bbox)], crs=latlon_proj)
        # Cannot clip because input data may be topologically invalid (case of
        # Canada).
        shape_df = geopd.overlay(shape_df, bbox_geodf, how="intersection")
    shape_df = shape_df.to_crs(xy_proj)
    shapely_geo = shape_df.geometry.iloc[0]
    if min_area is None or simplify_tol is None:
        area_bounds = shapely_geo.bounds
        # Get an upper limit of the distance that can be travelled inside the
        # area
        max_distance = np.sqrt(
            (area_bounds[0] - area_bounds[2]) ** 2
            + (area_bounds[1] - area_bounds[3]) ** 2
        )
        if simplify_tol is None:
            simplify_tol = max_distance / 1000

    if isinstance(shapely_geo, MultiPolygon):
        if min_area is None:
            min_area = max_distance**2 / 1000
        # We delete the polygons in the multipolygon which are too small and
        # just complicate the shape needlessly.
        shape_df.geometry.iloc[0] = MultiPolygon(
            [poly for poly in shapely_geo.geoms if poly.area > min_area]
        )
    # We also simplify by a given tolerance (max distance a point can be moved),
    # this could be a parameter in countries.json if needed
    shape_df.geometry = shape_df.simplify(simplify_tol)
    return shape_df
