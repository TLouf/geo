from __future__ import annotations

import inspect
from dataclasses import _FIELD, InitVar, dataclass
from pathlib import Path

import geopandas as geopd
import pandas as pd

from .agg import levels_corr
from .grid import create_grid, extract_shape, load_ext_cells


@dataclass
class BaseGeoPaths:
    shp_file_fmt: Path
    countries_shapefile: Path
    cell_levels_corr_dir: Path


@dataclass
class BaseRegion:
    id: str
    # Setting this as dict to input general dictionary about region that may contain
    # more attributes than needed, but that could be useful later on.
    static_features: dict
    _paths: BaseGeoPaths
    _cell_size: int | float | str
    all_shapes: InitVar[geopd.GeoDataFrame]
    # not a problem to default with mutable because this initvar is never touched
    extract_shape_kwargs: InitVar[dict] = {"simplify_tol": 100}
    xy_proj: str = "epsg:3857"
    max_place_area: float = 5e9
    _cell_kind: str = "census"
    shape_geodf: geopd.GeoDataFrame | None = None
    _cell_levels_corr: pd.DataFrame | None = None
    _shape_bbox: list | None = None
    _cells_geodf: geopd.GeoDataFrame | None = None

    def __post_init__(
        self,
        all_shapes,
        extract_shape_kwargs,
    ):
        if self.shape_geodf is None:
            shapefile_val = self.static_features.get("shapefile_val", self.id)
            shapefile_col = self.static_features.get("shapefile_col", "FID")
            mask = all_shapes[shapefile_col].str.startswith(shapefile_val)
            self.shape_geodf = extract_shape(
                all_shapes.loc[mask],
                bbox=self.static_features.get("shape_bbox"),
                xy_proj=self.xy_proj,
                **extract_shape_kwargs,
            )
        print(f"shape_geodf loaded for {self.id}")


    def __repr__(self):
        field_dict = self.__dataclass_fields__
        persistent_field_keys = [
            key for key, value in field_dict.items() if value._field_type == _FIELD
        ]
        attr_str_components = []
        for key in persistent_field_keys:
            field = getattr(self, key)
            field_repr = repr(field)
            if len(field_repr) < 200:
                attr_str_components.append(f"{key}={field_repr}")

        attr_str = ", ".join(attr_str_components)
        return f"{self.__class__.__name__}({attr_str})"

    @classmethod
    def from_dict(cls, region_id, static_features, **kwargs):
        all_kwargs = {**static_features, **kwargs}
        matching_kwargs = {
            k: v
            for k, v in all_kwargs.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(region_id, static_features, **matching_kwargs)

    def update_from_dict(self, d):
        # useful when changes are made to countries.json
        for key, value in d.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        # custom to_dict to keep only parameters that can be in save path
        list_attr = [
            "readable",
            "id",
            "year_from",
            "year_to",
            "cell_size",
            "max_place_area",
            "xy_proj",
        ]
        return {
            **{attr: getattr(self, attr) for attr in list_attr},
        }

    @property
    def bbox(self):
        if self._shape_bbox is None:
            if "shape_bbox" in self.static_features:
                self._shape_bbox = self.static_features["shape_bbox"]
            else:
                self._shape_bbox = self.shape_geodf.to_crs("epsg:4326").total_bounds
        return self._shape_bbox

    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, p):
        self._paths = p

    @property
    def cell_shapefiles(self):
        return self.static_features.get("cell_shapefiles", {})

    @property
    def cell_levels_corr_files(self):
        return self.static_features.get("cell_levels_corr_files", {})

    @property
    def readable(self):
        return self.static_features["readable"]

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, _cell_size: int | float | str):
        if _cell_size != self.cell_size:
            self._cell_size = _cell_size
            cell_size_spec = self.cell_shapefiles.get(self._cell_size)
            if cell_size_spec is None:
                if self._cells_geodf is None:
                    # correspondence with `cell_levels_corr`
                    raise ValueError(
                        f"cell_size of {self.cell_size} can neither be obtained by "
                        "direct reading of a shapefile or correspondence with a "
                        "previous aggregation level: read a saved one first"
                    )
                agg_level = self._cell_size
                unit_level = self.cells_geodf.index.name
                cell_levels_corr = levels_corr(
                    self.cell_levels_corr, unit_level, agg_level
                )
                agg_level = cell_levels_corr.index.names[0]
                self.cells_geodf = (
                    self.cells_geodf.join(cell_levels_corr)
                     .dissolve(by=agg_level)
                )
            else:
                # self.cell_kind = cell_size_spec["kind"]
                # Else load according to cell_size_spec using cells_geodf's setter
                del self.cells_geodf
                self.cells_geodf = self.cells_geodf

    @property
    def cell_kind(self):
        return self.cell_shapefiles.get(self.cell_size, {}).get("kind") or self._cell_kind

    @property
    def cells_geodf(self):
        if self._cells_geodf is None:
            if isinstance(self.cell_size, str):
                cell_size_spec = self.cell_shapefiles.get(self.cell_size)
                if cell_size_spec is None:
                    raise ValueError(
                        f"cell_size of {self.cell_size} can neither be obtained by "
                        "direct reading of a shapefile or correspondence with a "
                        "previous aggregation level: read a saved one first"
                    )
                fpath = str(self.paths.shp_file_fmt).format(cell_size_spec["fname"])
                index_col = cell_size_spec["index_col"]
                self._cells_geodf = load_ext_cells(
                    fpath, index_col, xy_proj=self.xy_proj,
                )
            else:
                _, self._cells_geodf, _, _ = create_grid(
                    self.shape_geodf,
                    self.cell_size,
                    self.id,
                    xy_proj=self.xy_proj,
                    intersect=True,
                )
        return self._cells_geodf

    @cells_geodf.setter
    def cells_geodf(self, _cells_geodf):
        self._cells_geodf = _cells_geodf

    @cells_geodf.deleter
    def cells_geodf(self):
        self.cells_geodf = None


    @property
    def cell_levels_corr(self):
        if self._cell_levels_corr is None:
            corr_fpath = self.paths.cell_levels_corr_dir / self.cell_levels_corr_files[self.cell_kind]
            self._cell_levels_corr = pd.read_csv(corr_fpath)
        return self._cell_levels_corr


    @cell_levels_corr.setter
    def cell_levels_corr(self, _cell_levels_corr):
        self._cell_levels_corr = _cell_levels_corr

    def load_cell_levels_corr(self):
        corr_path = self.cell_levels_corr_files[self.cell_kind]
        self.cell_levels_corr = pd.read_csv(corr_path)
        return self.cell_levels_corr
