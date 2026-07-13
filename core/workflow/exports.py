"""Serialize MVP results to CSV / GeoJSON bytes and a single zip archive."""
from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    if gdf is None or gdf.empty:
        gdf = gpd.GeoDataFrame({"placeholder": []}, geometry=[], crs="EPSG:4326")
    return gdf.to_json().encode("utf-8")


def case_lines_gdf(case, source: Dict[str, Any]) -> gpd.GeoDataFrame:
    """LV (+ MV for the MV/LV case) lines for one candidate, in EPSG:4326."""
    rows: List[Dict[str, Any]] = []
    geoms: List[Any] = []

    def add_edges(topo, level, zone="", id_prefix=""):
        if topo is None or topo.edges_4326 is None or topo.edges_4326.empty:
            return
        for _, e in topo.edges_4326.iterrows():
            rows.append({"line_id": f"{id_prefix}{e['line_id']}", "voltage_level": level, "zone_id": zone, "length_m": float(e["length_m"])})
            geoms.append(e.geometry)

    if case.plan is not None:  # MV/LV
        add_edges(case.plan.core_topology, "LV", "core", id_prefix="core::")
        for zone in case.plan.zones:
            add_edges(zone["topology"], "LV", zone["zone_id"], id_prefix=f"{zone['zone_id']}::")
        if getattr(case.plan, "mv_topology", None) is not None:
            add_edges(case.plan.mv_topology, "MV", "trunk", id_prefix="mv::")
            poles = case.plan.mv_topology.poles_4326.set_index("pole_id").geometry.to_dict()
            for zone in case.plan.zones:
                pole_id = zone.get("mv_pole_id")
                pole_geom = poles.get(int(pole_id)) if pole_id is not None else None
                if pole_geom is None:
                    continue
                dst = Point(zone["transformer_latlon"][1], zone["transformer_latlon"][0])
                rows.append({
                    "line_id": f"mv_drop::{zone['zone_id']}",
                    "voltage_level": "MV",
                    "zone_id": zone["zone_id"],
                    "length_m": float(zone.get("mv_service_drop_m", 0.0)),
                })
                geoms.append(LineString([pole_geom, dst]))
        else:
            for zone in case.plan.zones:
                src = Point(source["longitude"], source["latitude"])
                dst = Point(zone["transformer_latlon"][1], zone["transformer_latlon"][0])
                rows.append({"line_id": f"mv::{zone['zone_id']}", "voltage_level": "MV", "zone_id": zone["zone_id"], "length_m": float(zone["mv_length_m"])})
                geoms.append(LineString([src, dst]))
    else:
        add_edges(case.topology, "LV", "")

    if not rows:
        return gpd.GeoDataFrame({"line_id": []}, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def transformers_gdf(case) -> gpd.GeoDataFrame:
    if case.plan is None or not case.plan.zones:
        return gpd.GeoDataFrame({"transformer_id": []}, geometry=[], crs="EPSG:4326")
    rows, geoms = [], []
    for zone in case.plan.zones:
        tx = zone["transformer_spec"]
        rows.append({
            "transformer_id": zone["transformer_id"], "zone_id": zone["zone_id"],
            "sn_kva": float(tx["sn_kva"]), "s_zone_kva": round(float(zone["s_zone_kva"]), 1),
            "n_customers": len(zone["customer_ids"]),
        })
        geoms.append(Point(zone["transformer_latlon"][1], zone["transformer_latlon"][0]))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def standalone_customers_gdf(case, customers_df: pd.DataFrame) -> gpd.GeoDataFrame:
    ids = set(case.standalone_customer_ids)
    sub = customers_df[customers_df["customer_id"].astype(str).isin(ids)]
    if sub.empty:
        return gpd.GeoDataFrame({"customer_id": []}, geometry=[], crs="EPSG:4326")
    geoms = [Point(lon, lat) for lon, lat in zip(sub["longitude"], sub["latitude"])]
    keep = sub[["customer_id", "customer_type", "demand_tier", "peak_kw_now"]].copy()
    return gpd.GeoDataFrame(keep, geometry=geoms, crs="EPSG:4326")


def build_exports(result, customers_df: pd.DataFrame, source: Dict[str, Any]) -> Dict[str, bytes]:
    cases = result.cases
    files: Dict[str, bytes] = {
        "comparison_table.csv": _csv_bytes(result.comparison_table),
        "customer_strategy.csv": _csv_bytes(result.customer_strategy_table),
        "decision_log.csv": _csv_bytes(result.decision_log),
        "cost_breakdown.csv": _csv_bytes(result.cost_breakdown_table),
        "validation_summary.csv": _csv_bytes(result.validation_summary),
        "pandapower_validation_results.csv": _csv_bytes(result.validation_summary),
        "full_lv_edges.geojson": _geojson_bytes(case_lines_gdf(cases["A_full_LV"], source)),
        "lv_standalone_edges.geojson": _geojson_bytes(case_lines_gdf(cases["B_LV_plus_standalone"], source)),
        "mvlv_edges.geojson": _geojson_bytes(case_lines_gdf(cases["C_MVLV"], source)),
        "mvlv_standalone_edges.geojson": _geojson_bytes(case_lines_gdf(cases["D_MVLV_plus_standalone"], source)),
        "transformers.geojson": _geojson_bytes(transformers_gdf(cases["C_MVLV"])),
        "mvlv_standalone_transformers.geojson": _geojson_bytes(transformers_gdf(cases["D_MVLV_plus_standalone"])),
        "standalone_customers.geojson": _geojson_bytes(standalone_customers_gdf(cases["B_LV_plus_standalone"], customers_df)),
        "mvlv_standalone_customers.geojson": _geojson_bytes(standalone_customers_gdf(cases["D_MVLV_plus_standalone"], customers_df)),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    files["all_outputs.zip"] = buf.getvalue()
    return files
