"""Build a complete, MVP-ready example from the Mafa (NGA) field survey.

Source signal (``examples/mafa case study/``):
  * ``consumers_coordinates_nga2.xlsx`` — 240 real customers (Google-Maps + field survey):
    coordinates, ``consumer_type`` and ``demand_tier`` (priority / must-connect left blank).
  * ``category_profiles.csv`` — 8760 h load profiles per category. ``hospital`` and ``school``
    carry real measured shapes; households are implausibly low and 7 categories are flat ``1.0``
    placeholders, so demand is complemented as documented below.
  * ``Roads_Mafa_edited.gpkg`` — OSM road network.

This maps the survey onto the MVP ``customers`` schema, complements the missing
engineering/economic fields with realistic Sub-Saharan-Africa defaults, derives demand from the
real profiles where available, picks a road-snapped source, and writes the full MVP input set
(with **fewer, bigger** transformers) next to the raw files.

Run:  python scripts/build_mafa_example.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples" / "mafa case study"

# Raw survey consumer_type -> MVP accepted customer_type.
TYPE_MAP = {
    "hh-low": "household", "hh-medium": "household", "hh-high": "household",
    "mosque": "mosque", "hospital": "hospital", "school": "school", "female_school": "school",
    "enterprise": "productive", "garage": "productive", "shopping_center": "shop",
    "police_station": "public_service", "government_office": "public_service",
}
# Households are complemented by MTF demand tier (peak_kW, annual_kWh) — the raw household
# profiles are too low to be usable and do not separate Medium from High.
TIER_DEMAND = {2: (0.25, 400.0), 3: (0.50, 900.0), 4: (1.00, 1800.0), 5: (2.00, 3500.0)}
# Non-household demand for the flat ``1.0`` placeholder categories (realistic SSA estimates).
# hospital / school are overwritten below with values derived from their real profiles.
TYPE_DEMAND = {
    "mosque": (1.2, 1500.0), "enterprise": (5.0, 9000.0), "garage": (6.0, 9000.0),
    "shopping_center": (8.0, 16000.0), "police_station": (3.0, 7000.0),
    "government_office": (3.0, 6000.0), "female_school": (6.0, 7000.0),
}
PF = {"household": 0.95, "shop": 0.90, "productive": 0.85, "public_service": 0.92,
      "school": 0.90, "hospital": 0.90, "mosque": 0.95}
GROWTH = {
    # Category-level annual demand growth. Households are moderate; shops and productive
    # loads grow faster after electrification; public/institutional demand is steadier.
    "household": 0.04,
    "shop": 0.065,
    "productive": 0.075,
    "public_service": 0.035,
    "school": 0.035,
    "hospital": 0.03,
    "mosque": 0.025,
}
PRIORITY = {"hospital": 3.0, "school": 2.0, "public_service": 2.0, "mosque": 1.5,
            "productive": 1.5, "shop": 1.5, "household": 1.0}
# Institutional / community loads that must be grid-connected (never stand-alone).
MUST_CONNECT_RAW = {"hospital", "school", "female_school", "police_station",
                    "government_office", "mosque"}


def _profile_demand() -> dict[str, tuple[float, float]]:
    """(peak_kW, annual_kWh) per raw category from real profiles; flat placeholders -> override."""
    cp = pd.read_csv(SRC / "category_profiles.csv")
    out: dict[str, tuple[float, float]] = {}
    for cat in (c for c in cp.columns if c != "hour"):
        s = cp[cat].astype(float)
        if s.std() < 1e-6:                       # flat placeholder
            out[cat] = TYPE_DEMAND.get(cat, (1.0, 2000.0))
        else:                                    # real measured shape (hospital, school, HH-*)
            out[cat] = (float(s.max()), float(s.sum()))
    return out


def main() -> None:
    xl = pd.read_excel(SRC / "consumers_coordinates_nga2.xlsx")
    raw = xl["consumer_type"].astype(str).str.lower()
    ctype = raw.map(lambda t: TYPE_MAP.get(t, "other"))
    tier = pd.to_numeric(xl["demand_tier"], errors="coerce").fillna(2).astype(int)
    prof = _profile_demand()

    def demand(row_raw: str, row_type: str, row_tier: int) -> tuple[float, float]:
        if row_type == "household":
            return TIER_DEMAND.get(row_tier, TIER_DEMAND[2])
        if row_raw in ("hospital", "school"):    # honour the real measured profile
            return prof[row_raw]
        return TYPE_DEMAND.get(row_raw, (2.0, 4000.0))

    peaks, kwh = zip(*[demand(r, t, ti) for r, t, ti in zip(raw, ctype, tier)])

    df = pd.DataFrame({
        "customer_id": xl["customer_id"].astype(str),
        "latitude": xl["latitude"].astype(float),
        "longitude": xl["longitude"].astype(float),
        "customer_type": ctype,
        "peak_kw_now": np.round(peaks, 3),
        "annual_kwh_now": np.round(kwh, 0),
        "power_factor": ctype.map(lambda t: PF.get(t, 0.90)),
        "growth_rate": ctype.map(lambda t: GROWTH.get(t, 0.04)),
        "demand_tier": tier,
        "priority_weight": ctype.map(lambda t: PRIORITY.get(t, 1.0)),
        "must_connect_flag": raw.isin(MUST_CONNECT_RAW),
    })
    df["can_be_standalone_flag"] = (ctype == "household") & ~df["must_connect_flag"]

    # Spatial spread (projected) for source placement and a sanity print.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    utm = gdf.estimate_utm_crs()
    gproj = gdf.to_crs(utm)
    centroid = gproj.geometry.union_all().centroid
    dist = gproj.geometry.distance(centroid)
    print(f"customers={len(df)}  utm={utm.to_epsg()}  spread: median={dist.median():.0f} m  "
          f"p90={dist.quantile(.9):.0f} m  max={dist.max():.0f} m")
    print(f"non-coincident peak={df['peak_kw_now'].sum():.0f} kW  (hospital share="
          f"{df.loc[df['customer_type']=='hospital','peak_kw_now'].sum():.0f} kW)")

    # Source: customer centroid snapped to the nearest road, LV 0.4 kV / MV 11 kV.
    roads = gpd.read_file(SRC / "Roads_Mafa_edited.gpkg").to_crs(utm)
    src_proj = nearest_points(centroid, roads.geometry.union_all())[1]
    src_ll = gpd.GeoSeries([src_proj], crs=utm).to_crs(4326).iloc[0]

    df.to_csv(SRC / "customers.csv", index=False)
    pd.DataFrame([{
        "source_id": "SRC0", "latitude": round(float(src_ll.y), 6), "longitude": round(float(src_ll.x), 6),
        "source_type": "grid_injection", "lv_voltage_kv": 0.4, "mv_voltage_kv": 11.0,
    }]).to_csv(SRC / "source.csv", index=False)
    gpd.read_file(SRC / "Roads_Mafa_edited.gpkg").to_crs(4326)[["geometry"]].to_file(
        SRC / "roads.geojson", driver="GeoJSON")

    _write_catalogues(SRC)
    print("Wrote MVP input set to", SRC)


def _write_catalogues(out: Path) -> None:
    """SSA LV/MV conductor, (fewer/bigger) transformer, stand-alone and cost catalogues."""
    pd.DataFrame([
        {"conductor_id": "LV_ABC_25", "voltage_level": "LV", "r_ohm_per_km": 1.20, "x_ohm_per_km": 0.090, "max_i_ka": 0.10, "cost_per_m": 7.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "LV_ABC_50", "voltage_level": "LV", "r_ohm_per_km": 0.64, "x_ohm_per_km": 0.085, "max_i_ka": 0.16, "cost_per_m": 10.0, "phase_count": 3, "rank": 2},
        {"conductor_id": "LV_ABC_95", "voltage_level": "LV", "r_ohm_per_km": 0.32, "x_ohm_per_km": 0.078, "max_i_ka": 0.24, "cost_per_m": 15.0, "phase_count": 3, "rank": 3},
        {"conductor_id": "LV_ABC_150", "voltage_level": "LV", "r_ohm_per_km": 0.21, "x_ohm_per_km": 0.075, "max_i_ka": 0.32, "cost_per_m": 22.0, "phase_count": 3, "rank": 4},
        {"conductor_id": "LV_ABC_240", "voltage_level": "LV", "r_ohm_per_km": 0.13, "x_ohm_per_km": 0.072, "max_i_ka": 0.42, "cost_per_m": 32.0, "phase_count": 3, "rank": 5},
        {"conductor_id": "LV_ABC_300", "voltage_level": "LV", "r_ohm_per_km": 0.10, "x_ohm_per_km": 0.070, "max_i_ka": 0.50, "cost_per_m": 42.0, "phase_count": 3, "rank": 6},
        {"conductor_id": "LV_ABC_400", "voltage_level": "LV", "r_ohm_per_km": 0.077, "x_ohm_per_km": 0.068, "max_i_ka": 0.62, "cost_per_m": 56.0, "phase_count": 3, "rank": 7},
    ]).to_csv(out / "lv_conductors.csv", index=False)

    pd.DataFrame([
        {"conductor_id": "MV_ACSR_35", "voltage_level": "MV", "r_ohm_per_km": 0.87, "x_ohm_per_km": 0.40, "max_i_ka": 0.17, "cost_per_m": 11.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "MV_ACSR_70", "voltage_level": "MV", "r_ohm_per_km": 0.44, "x_ohm_per_km": 0.38, "max_i_ka": 0.27, "cost_per_m": 16.0, "phase_count": 3, "rank": 2},
    ]).to_csv(out / "mv_conductors.csv", index=False)

    # Fewer, bigger distribution transformers (100 / 200 / 315 / 500 kVA) — realistic ground-mount
    # MV/LV units that let a dense village be served by a handful of substations.
    pd.DataFrame([
        {"transformer_id": "TX_100", "sn_kva": 100, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.0, "vkr_percent": 1.3, "pfe_kw": 0.25, "i0_percent": 0.4, "cost_fixed": 7000, "cost_per_kva": 52, "max_loading_percent": 100, "rank": 1},
        {"transformer_id": "TX_200", "sn_kva": 200, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.5, "vkr_percent": 1.2, "pfe_kw": 0.45, "i0_percent": 0.3, "cost_fixed": 9500, "cost_per_kva": 46, "max_loading_percent": 100, "rank": 2},
        {"transformer_id": "TX_315", "sn_kva": 315, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 5.0, "vkr_percent": 1.1, "pfe_kw": 0.65, "i0_percent": 0.3, "cost_fixed": 12500, "cost_per_kva": 42, "max_loading_percent": 100, "rank": 3},
        {"transformer_id": "TX_500", "sn_kva": 500, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 5.5, "vkr_percent": 1.0, "pfe_kw": 1.00, "i0_percent": 0.25, "cost_fixed": 17000, "cost_per_kva": 38, "max_loading_percent": 100, "rank": 4},
    ]).to_csv(out / "transformers.csv", index=False)

    pd.DataFrame([
        {"package_id": "SHS_T1", "service_tier": 1, "max_peak_kw": 0.20, "annual_kwh": 300, "capex": 240, "lifetime_years": 8, "annual_om_fraction": 0.045, "replacement_fraction": 0.55},
        {"package_id": "SHS_T2", "service_tier": 2, "max_peak_kw": 0.45, "annual_kwh": 800, "capex": 520, "lifetime_years": 8, "annual_om_fraction": 0.045, "replacement_fraction": 0.55},
        {"package_id": "SHS_T3", "service_tier": 3, "max_peak_kw": 1.10, "annual_kwh": 2200, "capex": 950, "lifetime_years": 10, "annual_om_fraction": 0.035, "replacement_fraction": 0.50},
        {"package_id": "SHS_T4", "service_tier": 4, "max_peak_kw": 2.20, "annual_kwh": 4000, "capex": 2100, "lifetime_years": 10, "annual_om_fraction": 0.035, "replacement_fraction": 0.50},
    ]).to_csv(out / "standalone_packages.csv", index=False)

    # Larger zone radius + customer cap than the default so the settlement resolves into a few
    # big transformer zones (matching the bigger transformer catalogue).
    cost = {
        "discount_rate": 0.10, "planning_horizon_years": 15,
        "lv_connection_cost_per_customer": 120.0, "metering_protection_cost_per_customer": 60.0,
        "pole_cost": 140.0, "civil_logistics_multiplier": 1.3, "loss_value_per_kwh": 0.25, "loss_load_factor": 0.35,
        "voltage_min_pu": 0.90, "voltage_max_pu": 1.05, "max_line_loading_percent": 100.0, "max_transformer_loading_percent": 100.0,
        "dmax_lv_service_zone_m": 450.0, "lmax_lv_path_m": 700.0, "remote_branch_length_threshold_m": 300.0,
        "branch_cost_per_customer_threshold": 700.0, "loss_threshold_percent": 8.0,
        "max_lv_zone_radius_m": 360.0, "max_zone_customers": 140,
    }
    pd.DataFrame({"parameter": list(cost), "value": list(cost.values())}).to_csv(out / "cost_parameters.csv", index=False)


if __name__ == "__main__":
    main()
