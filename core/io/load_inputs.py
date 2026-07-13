"""Load MVP input files (CSV / GeoJSON / GPKG / YAML) and adapt to the topology engine.

The simplified MVP CSV schema differs from the legacy topology engine schema, so
:func:`customers_to_engine_gdf` maps the MVP customer columns onto the columns that
``core.distribution_service._run_low_voltage_core`` expects, while preserving a stable
``customer_id`` and a ``building_id`` (positional index) for round-tripping results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point

from core.distribution_io import select_projected_crs
from . import input_schemas as schemas


def _read_csv(file: Any) -> pd.DataFrame:
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass
    return pd.read_csv(file)


def load_customers(file: Any) -> pd.DataFrame:
    return schemas.normalize_customers(_read_csv(file))


def load_source(file: Any) -> Dict[str, Any]:
    return schemas.normalize_source(_read_csv(file))


def load_lv_conductors(file: Any) -> pd.DataFrame:
    return schemas.normalize_conductors(_read_csv(file), "LV")


def load_mv_conductors(file: Any) -> pd.DataFrame:
    return schemas.normalize_conductors(_read_csv(file), "MV")


def load_transformers(file: Any) -> pd.DataFrame:
    return schemas.normalize_transformers(_read_csv(file))


def load_standalone_packages(file: Any) -> pd.DataFrame:
    return schemas.normalize_standalone_packages(_read_csv(file))


def load_cost_parameters(file: Any) -> Dict[str, float]:
    name = str(getattr(file, "name", file) or "")
    if name.endswith((".yaml", ".yml")):
        if hasattr(file, "read"):
            file.seek(0)
            data = yaml.safe_load(file.read())
        else:
            data = yaml.safe_load(Path(file).read_text())
        return schemas.normalize_cost_parameters(dict(data or {}))
    df = _read_csv(file)
    # Accept either a key/value layout or a single-row wide layout.
    if {"parameter", "value"} <= set(df.columns):
        params = dict(zip(df["parameter"].astype(str), df["value"]))
    elif {"key", "value"} <= set(df.columns):
        params = dict(zip(df["key"].astype(str), df["value"]))
    else:
        params = df.iloc[0].to_dict()
    return schemas.normalize_cost_parameters(params)


def load_roads(file: Any) -> Optional[gpd.GeoDataFrame]:
    """Load roads from GeoJSON or GPKG (None-safe). Returns a GeoDataFrame in EPSG:4326."""
    if file is None:
        return None
    name = str(getattr(file, "name", file) or "")
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass
    gdf = gpd.read_file(file)
    if gdf.empty:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=4326)


def customers_to_engine_gdf(
    customers: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Build a projected GeoDataFrame in the legacy engine schema.

    Returns ``(gdf, crs_meta)`` where the gdf index is the ``building_id`` used by the
    topology engine and ``customer_id`` is preserved as a stable string column.
    """
    df = customers.reset_index(drop=True).copy()
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    crs, meta = select_projected_crs(gdf)
    gdf_proj = gdf.to_crs(crs)

    # Map MVP columns -> engine-expected columns.
    gdf_proj["consumer_type"] = gdf_proj["customer_type"].astype(str)
    gdf_proj["peak_kw"] = gdf_proj["peak_kw_now"].astype(float)
    gdf_proj["annual_demand_kwh"] = gdf_proj["annual_kwh_now"].astype(float)
    # priority_score 1..5: must_connect -> 5, else derive from demand_tier (clamped).
    score = gdf_proj["demand_tier"].clip(lower=1, upper=5).astype(int)
    score = score.where(~gdf_proj["must_connect_flag"], 5)
    gdf_proj["priority_score"] = score
    gdf_proj["must_connect_flag"] = gdf_proj["must_connect_flag"].astype(bool)
    # The full-LV case connects everyone; must_standalone is only used in later cases.
    gdf_proj["must_standalone_flag"] = False

    crs_meta = {
        "input_crs": "EPSG:4326",
        "projected_crs": crs.to_string(),
        "projected_crs_epsg": crs.to_epsg(),
        "crs_selection_method": meta.get("crs_selection_method"),
        "crs_selection_note": meta.get("crs_selection_note"),
    }
    return gdf_proj, crs_meta
