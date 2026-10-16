from __future__ import annotations

import math
from typing import Any, Optional

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point

from config.settings import DEFAULT_TARGET_CRS


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((float(lon) + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 + zone) if float(lat) >= 0.0 else (32700 + zone)


def select_projected_crs(
    gdf: gpd.GeoDataFrame,
    fallback_crs: Any = DEFAULT_TARGET_CRS,
) -> tuple[CRS, dict[str, Any]]:
    """
    Select a projected CRS for metric calculations.

    Preference order
    ----------------
    1. Auto-select local UTM from the geometry centroid when the input CRS is known.
    2. Fall back to `fallback_crs` when the input CRS is missing or centroid inference fails.
    """
    fallback = CRS.from_user_input(fallback_crs)
    meta: dict[str, Any] = {
        "input_crs": None if gdf.crs is None else gdf.crs.to_string(),
        "crs_selection_method": "fallback_default",
        "crs_selection_note": None,
    }

    if gdf.crs is None:
        meta["crs_selection_note"] = (
            "Input CRS was missing; assumed projected coordinates in the fallback metric CRS."
        )
        return fallback, meta

    try:
        gdf_4326 = gdf.to_crs(epsg=4326)
        centroid = gdf_4326.union_all().centroid
        if centroid.is_empty:
            raise ValueError("Empty centroid geometry.")
        utm_epsg = _utm_epsg_from_lonlat(centroid.x, centroid.y)
        meta["crs_selection_method"] = "auto_local_utm"
        return CRS.from_epsg(utm_epsg), meta
    except Exception:
        meta["crs_selection_note"] = (
            "Automatic local UTM selection failed; used the fallback projected CRS instead."
        )
        return fallback, meta


def load_and_transform_data(
    file,
    target_crs: Any | None = None,
    *,
    return_metadata: bool = False,
) -> Optional[gpd.GeoDataFrame] | tuple[Optional[gpd.GeoDataFrame], dict[str, Any]]:
    """
    Load and transform spatial data from an uploaded file.

    Supported formats
    -----------------
    - GeoPackage (.gpkg): uses the file CRS when available, otherwise falls back to the
      selected/projected working CRS.
    - Excel (.xlsx): expects 'latitude' and 'longitude' columns in WGS84 (EPSG:4326).

    Returns
    -------
    GeoDataFrame in the selected/projected CRS, or None if the file cannot be parsed.
    When `return_metadata=True`, also returns a dict with CRS-selection details.
    """
    metadata: dict[str, Any] = {
        "input_crs": None,
        "projected_crs": None,
        "projected_crs_epsg": None,
        "crs_selection_method": None,
        "crs_selection_note": None,
    }

    if file is None:
        return (None, metadata) if return_metadata else None

    name = getattr(file, "name", "")
    gdf: Optional[gpd.GeoDataFrame] = None

    if name.endswith(".gpkg"):
        gdf = gpd.read_file(file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(file)
        if "latitude" in df.columns and "longitude" in df.columns:
            geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)

    if gdf is None:
        return (None, metadata) if return_metadata else None

    if target_crs is None:
        selected_crs, meta = select_projected_crs(gdf, fallback_crs=DEFAULT_TARGET_CRS)
    else:
        selected_crs = CRS.from_user_input(target_crs)
        meta = {
            "input_crs": None if gdf.crs is None else gdf.crs.to_string(),
            "crs_selection_method": "provided_target_crs",
            "crs_selection_note": None,
        }
        if gdf.crs is None:
            meta["crs_selection_method"] = "provided_target_crs_missing_input_assumed"
            meta["crs_selection_note"] = (
                "Input CRS was missing; assumed coordinates already match the provided target CRS."
            )

    if gdf.crs is None:
        gdf = gdf.set_crs(selected_crs)

    if gdf.crs != selected_crs:
        gdf = gdf.to_crs(selected_crs)

    metadata.update(meta)
    metadata["projected_crs"] = selected_crs.to_string()
    metadata["projected_crs_epsg"] = selected_crs.to_epsg()

    out = gdf[gdf.is_valid]
    return (out, metadata) if return_metadata else out
