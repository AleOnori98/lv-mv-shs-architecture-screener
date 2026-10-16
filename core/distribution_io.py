from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
from pyproj import CRS

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


