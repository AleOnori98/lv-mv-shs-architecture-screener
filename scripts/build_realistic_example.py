"""Build a complete, MVP-ready realistic SSA example from examples/realistic_examples/complete/.

Source signal: `complete/users.xlsx` (real coordinates + customer types + demand tiers) and
`complete/roads.gpkg` (OSM roads). This maps the user table onto the MVP customers schema,
adds the missing engineering/economic fields with realistic Sub-Saharan-Africa defaults, picks
a source location, and writes the full MVP input set next to the raw files.

Run:  python scripts/build_realistic_example.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples" / "complete"

# customer_type mapping: raw label -> MVP accepted type.
TYPE_MAP = {
    "residential": "household",
    "household": "household",
    "shop": "shop",
    "productive": "productive",
    "school": "school",
    "clinic": "clinic",
    "hospital": "hospital",
    "water_point": "water_pump",
    "water_pump": "water_pump",
    "public_service": "public_service",
    "community": "public_service",
    "mosque": "mosque",
}
# Per-type realistic SSA power factor / annual growth / grid-priority (can_be_standalone=False).
PF = {"household": 0.95, "shop": 0.92, "productive": 0.85, "water_pump": 0.80}
GROWTH = {"household": 0.06, "shop": 0.06, "productive": 0.08, "water_pump": 0.07}
GRID_PRIORITY = {"productive", "school", "clinic", "hospital", "water_pump", "public_service"}


def main() -> None:
    xl = pd.read_excel(SRC / "users.xlsx")
    df = pd.DataFrame()
    df["customer_id"] = xl["customer_id"].astype(str)
    df["latitude"] = xl["latitude"].astype(float)
    df["longitude"] = xl["longitude"].astype(float)
    ctype = xl["customer_type"].astype(str).str.lower().map(lambda t: TYPE_MAP.get(t, "other"))
    df["customer_type"] = ctype
    df["peak_kw_now"] = xl["peak_kw"].astype(float)
    df["annual_kwh_now"] = xl["annual_demand_kwh"].astype(float)
    df["power_factor"] = ctype.map(lambda t: PF.get(t, 0.90))
    df["growth_rate"] = ctype.map(lambda t: GROWTH.get(t, 0.04))
    df["demand_tier"] = pd.to_numeric(xl["demand_tier"], errors="coerce").fillna(2).astype(int)
    df["priority_weight"] = pd.to_numeric(xl.get("priority_score"), errors="coerce").fillna(1.0)
    df["must_connect_flag"] = xl["must_connect_flag"].astype(bool)
    df["can_be_standalone_flag"] = ~(ctype.isin(GRID_PRIORITY) | df["must_connect_flag"])

    # Spatial spread (projected) to size the source and report thresholds.
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])], crs="EPSG:4326")
    utm = gdf.estimate_utm_crs()
    gproj = gdf.to_crs(utm)
    centroid = gproj.geometry.union_all().centroid
    dist = gproj.geometry.distance(centroid)
    print(f"customers={len(df)}  utm={utm.to_epsg()}  spread: median={dist.median():.0f} m  p90={dist.quantile(.9):.0f} m  max={dist.max():.0f} m")

    # Source: customer centroid snapped to the nearest road, at LV 0.4 kV / MV 11 kV.
    roads = gpd.read_file(SRC / "roads.gpkg").to_crs(utm)
    src_proj = nearest_points(centroid, roads.geometry.union_all())[1]
    src_ll = gpd.GeoSeries([src_proj], crs=utm).to_crs(4326).iloc[0]

    df.to_csv(SRC / "customers.csv", index=False)
    pd.DataFrame([{
        "source_id": "SRC0", "latitude": round(float(src_ll.y), 6), "longitude": round(float(src_ll.x), 6),
        "source_type": "grid_injection", "lv_voltage_kv": 0.4, "mv_voltage_kv": 11.0,
    }]).to_csv(SRC / "source.csv", index=False)

    # Roads -> GeoJSON for the page uploader / loader convenience.
    gpd.read_file(SRC / "roads.gpkg").to_crs(4326)[["geometry"]].to_file(SRC / "roads.geojson", driver="GeoJSON")

    _write_catalogues(SRC)
    print("Wrote MVP input set to", SRC)


def _write_catalogues(out: Path) -> None:
    """Realistic SSA LV/MV conductor, transformer, stand-alone and cost catalogues."""
    pd.DataFrame([
        {"conductor_id": "LV_ABC_25", "voltage_level": "LV", "r_ohm_per_km": 1.20, "x_ohm_per_km": 0.090, "max_i_ka": 0.10, "cost_per_m": 7.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "LV_ABC_50", "voltage_level": "LV", "r_ohm_per_km": 0.64, "x_ohm_per_km": 0.085, "max_i_ka": 0.16, "cost_per_m": 10.0, "phase_count": 3, "rank": 2},
        {"conductor_id": "LV_ABC_95", "voltage_level": "LV", "r_ohm_per_km": 0.32, "x_ohm_per_km": 0.078, "max_i_ka": 0.24, "cost_per_m": 15.0, "phase_count": 3, "rank": 3},
        {"conductor_id": "LV_ABC_150", "voltage_level": "LV", "r_ohm_per_km": 0.21, "x_ohm_per_km": 0.075, "max_i_ka": 0.32, "cost_per_m": 22.0, "phase_count": 3, "rank": 4},
    ]).to_csv(out / "lv_conductors.csv", index=False)

    pd.DataFrame([
        {"conductor_id": "MV_ACSR_35", "voltage_level": "MV", "r_ohm_per_km": 0.87, "x_ohm_per_km": 0.40, "max_i_ka": 0.17, "cost_per_m": 11.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "MV_ACSR_70", "voltage_level": "MV", "r_ohm_per_km": 0.44, "x_ohm_per_km": 0.38, "max_i_ka": 0.27, "cost_per_m": 16.0, "phase_count": 3, "rank": 2},
    ]).to_csv(out / "mv_conductors.csv", index=False)

    pd.DataFrame([
        {"transformer_id": "TX_25", "sn_kva": 25, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.0, "vkr_percent": 1.3, "pfe_kw": 0.10, "i0_percent": 0.5, "cost_fixed": 3500, "cost_per_kva": 65, "max_loading_percent": 100, "rank": 1},
        {"transformer_id": "TX_50", "sn_kva": 50, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.0, "vkr_percent": 1.2, "pfe_kw": 0.15, "i0_percent": 0.4, "cost_fixed": 4500, "cost_per_kva": 58, "max_loading_percent": 100, "rank": 2},
        {"transformer_id": "TX_100", "sn_kva": 100, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.5, "vkr_percent": 1.1, "pfe_kw": 0.25, "i0_percent": 0.3, "cost_fixed": 6500, "cost_per_kva": 52, "max_loading_percent": 100, "rank": 3},
        {"transformer_id": "TX_200", "sn_kva": 200, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 5.0, "vkr_percent": 1.0, "pfe_kw": 0.45, "i0_percent": 0.3, "cost_fixed": 9000, "cost_per_kva": 46, "max_loading_percent": 100, "rank": 4},
    ]).to_csv(out / "transformers.csv", index=False)

    pd.DataFrame([
        {"package_id": "SHS_T1", "service_tier": 1, "max_peak_kw": 0.20, "annual_kwh": 300, "capex": 300, "lifetime_years": 8, "annual_om_fraction": 0.05, "replacement_fraction": 0.6},
        {"package_id": "SHS_T2", "service_tier": 2, "max_peak_kw": 0.45, "annual_kwh": 800, "capex": 650, "lifetime_years": 8, "annual_om_fraction": 0.05, "replacement_fraction": 0.6},
        {"package_id": "SHS_T3", "service_tier": 3, "max_peak_kw": 1.00, "annual_kwh": 2000, "capex": 1500, "lifetime_years": 10, "annual_om_fraction": 0.045, "replacement_fraction": 0.6},
    ]).to_csv(out / "standalone_packages.csv", index=False)

    cost = {
        "discount_rate": 0.10, "planning_horizon_years": 15,
        "lv_connection_cost_per_customer": 120.0, "metering_protection_cost_per_customer": 60.0,
        "pole_cost": 140.0, "civil_logistics_multiplier": 1.3, "loss_value_per_kwh": 0.25, "loss_load_factor": 0.35,
        "voltage_min_pu": 0.90, "voltage_max_pu": 1.05, "max_line_loading_percent": 100.0, "max_transformer_loading_percent": 100.0,
        "dmax_lv_service_zone_m": 400.0, "lmax_lv_path_m": 600.0, "remote_branch_length_threshold_m": 250.0,
        "branch_cost_per_customer_threshold": 700.0, "loss_threshold_percent": 8.0,
    }
    pd.DataFrame({"parameter": list(cost), "value": list(cost.values())}).to_csv(out / "cost_parameters.csv", index=False)


if __name__ == "__main__":
    main()
