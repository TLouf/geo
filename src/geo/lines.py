import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from shapely.geometry import LineString


def cut_line_from_endpoints(
    line: LineString, distance: float, length: float | None = None
):
    if length is None:
        length = line.length
    if length < 2 * distance:
        return LineString()

    start_p = line.interpolate(distance)
    end_p = line.interpolate(-distance)
    # Snap the line and not the point, to make absolutely sure the point --as in, its
    # exact coordinates-- is part of the linestring. If you snap the point, to then
    # check it's part of the line `.split` will interpolate from existing points, and
    # due to floating point precision issue, the point might not be considered exactly
    # on the line.
    line = shapely.ops.snap(line, start_p, 1e-6)
    line = shapely.ops.snap(line, end_p, 1e-6)
    new_coords = []
    for x, y in line.coords:
        if len(new_coords) == 0 and x == start_p.x and y == start_p.y:
            new_coords.append((x, y))
        elif len(new_coords) > 0:
            new_coords.append((x, y))
            if x == end_p.x and y == end_p.y:
                return LineString(new_coords)


def cut_lines_from_endpoints(gdf: gpd.GeoDataFrame, distance: float):
    to_cut = gdf.length > distance * 2
    cut_gdf = gdf.loc[to_cut].reset_index()
    cut_gdf["start_p"] = cut_gdf.geometry.interpolate(distance)
    cut_gdf[["start_x", "start_y"]] = cut_gdf["start_p"].get_coordinates()
    cut_gdf["end_p"] = cut_gdf.geometry.interpolate(-distance)
    cut_gdf[["end_x", "end_y"]] = cut_gdf["end_p"].get_coordinates()
    cut_gdf.geometry = cut_gdf.snap(cut_gdf["start_p"], 1e-6)
    cut_gdf.geometry = cut_gdf.snap(cut_gdf["end_p"], 1e-6)
    coords = cut_gdf.get_coordinates().join(
        cut_gdf[[f"{pos}_{dim}" for dim in ("x", "y") for pos in ("start", "end")]]
    )
    coords["is_extremity"] = (coords["x"] == coords["start_x"]) & (
        coords["y"] == coords["start_y"]
    ) | (coords["x"] == coords["end_x"]) & (coords["y"] == coords["end_y"])
    coords["segment_idx"] = coords["is_extremity"].groupby(coords.index).cumsum()
    coords = coords.loc[coords["segment_idx"] == 1, ["x", "y"]].reset_index()
    coords["i"] = coords.index
    coords = pd.concat(
        [
            coords,
            cut_gdf[["end_x", "end_y"]]
            .rename(columns={"end_x": "x", "end_y": "y"})
            .reset_index()
            .assign(i=coords.shape[0]),
        ]
    ).sort_values(["index", "i"])
    cut_gdf.geometry = shapely.linestrings(
        coords[["x", "y"]].values, indices=coords["index"]
    )
    cut_gdf = cut_gdf.set_index("index").loc[:, gdf.columns]
    empty_gdf = gdf.loc[~to_cut].copy()
    empty_gdf.geometry = [LineString() for _ in range(empty_gdf.shape[0])]
    return pd.concat([cut_gdf, empty_gdf]).sort_index()


def get_segments_arr(
    lines_coords: np.ndarray, same_line_mask: np.ndarray | None = None
):
    """Create all the segments contained in input LineStrings.

    Adapted from https://gist.github.com/jGaboardi/754e878ee4cac986132295ed43e74512.

    Parameters
    ----------
    lines_coords : np.ndarray
        2D array of (x, y) coordinates of all points of some lines.
    same_line_mask : np.ndarray, optional
        Boolean mask indicating if each pair of consecutive coordinates in
        `lines_coords` corresponds to the same input line, to avoid creating segments
        between different lines. None by default, in which case all points are
        considered to belong to the same line.

    Returns
    -------
    np.ndarray
        Array containing all the segments of the input lines.
    """
    segments_coords = np.column_stack((lines_coords[:-1], lines_coords[1:])).reshape(
        lines_coords.shape[0] - 1, 2, 2
    )
    if same_line_mask is not None:
        segments_coords = segments_coords[same_line_mask]
    segments = shapely.linestrings(segments_coords)
    return segments


def get_segments_gdf(lines_gs: gpd.GeoSeries):
    """Get a GeoDataFrame of all the segments constituting the input lines.

    Parameters
    ----------
    lines_gdf : gpd.GeoSeries
        GeoDataFrame containing input LineStrings.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame containing all the segments of the matching `line_id`.
    """
    lines_coords = lines_gs.get_coordinates()
    is_same_line = lines_coords.index[:-1] == lines_coords.index[1:]
    segments_gdf = gpd.GeoDataFrame(
        {
            "line_id": lines_coords.index[:-1][is_same_line],
            "geometry": get_segments_arr(lines_coords.values, is_same_line),
        },
        crs=lines_gs.crs,
    ).rename_axis("segment_id")
    return segments_gdf
