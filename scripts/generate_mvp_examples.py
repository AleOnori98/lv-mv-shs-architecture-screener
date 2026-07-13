"""Deterministically generate the simplified-MVP example dataset under examples/mvp/.

Produces a small but representative rural settlement:
  * 1 fixed source
  * 1 dense core cluster (grid-favourable)
  * 1 remote sparse household cluster (stand-alone candidate)
  * 1 remote institutional/productive cluster (MV/LV candidate, not stand-alone eligible)
  * a few extra households near the core edge for demand-growth stress testing

Run with the project env:
    python scripts/generate_mvp_examples.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "mvp"
OUT.mkdir(parents=True, exist_ok=True)

LAT0, LON0 = 0.3476, 32.5825  # origin (source)
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


def to_latlon(x_m: float, y_m: float) -> tuple[float, float]:
    return LAT0 + y_m / M_PER_DEG_LAT, LON0 + x_m / M_PER_DEG_LON


def ring(cx, cy, n, r, seed):
    """Deterministic pseudo-random scatter of n points within radius r of (cx, cy)."""
    pts = []
    s = seed
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        rad = r * math.sqrt(((s >> 5) % 1000) / 1000.0)
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        ang = 2 * math.pi * (((s >> 7) % 1000) / 1000.0)
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


rows = []
cid = 0


def add(x, y, ctype, peak, akwh, tier, *, must_connect=False, can_sa=None, pf=0.90, growth=0.03):
    global cid
    rows.append(
        {
            "customer_id": f"C{cid:03d}",
            **dict(zip(("latitude", "longitude"), to_latlon(x, y))),
            "customer_type": ctype,
            "peak_kw_now": peak,
            "annual_kwh_now": akwh,
            "power_factor": pf,
            "growth_rate": growth,
            "demand_tier": tier,
            "priority_weight": 1.0,
            "must_connect_flag": must_connect,
            "can_be_standalone_flag": "" if can_sa is None else can_sa,
        }
    )
    cid += 1


# --- Dense core cluster: 34 households + 4 shops, near source ---
for (x, y) in ring(60, 60, 34, 110, seed=11):
    add(x, y, "household", 0.45, 900, 2)
for (x, y) in ring(40, 90, 4, 90, seed=23):
    add(x, y, "shop", 1.2, 3500, 3)

# --- Remote sparse household cluster (stand-alone candidate), ~1.3 km east ---
for (x, y) in ring(1350, 250, 10, 200, seed=37):
    add(x, y, "household", 0.30, 500, 1, can_sa=True)

# --- Remote institutional/productive cluster (MV/LV candidate), ~870 m north-west ---
add(-210, 840, "clinic", 3.5, 12000, 4, must_connect=True)
add(-170, 880, "school", 2.5, 8000, 3, can_sa=False)
add(-250, 820, "water_pump", 4.0, 14000, 4, can_sa=False)
add(-200, 900, "productive", 3.0, 11000, 3, can_sa=False)
for (x, y) in ring(-200, 860, 5, 70, seed=53):
    add(x, y, "household", 0.4, 800, 2)

# --- Additional households near core edge ---
for (x, y) in ring(180, 40, 3, 40, seed=71):
    add(x, y, "household", 0.4, 800, 2)

customers = pd.DataFrame(rows)
customers.to_csv(OUT / "customers_example.csv", index=False)

# --- Source ---
pd.DataFrame(
    [{"source_id": "SRC0", "latitude": LAT0, "longitude": LON0, "source_type": "injection", "lv_voltage_kv": 0.4, "mv_voltage_kv": 11.0}]
).to_csv(OUT / "source_example.csv", index=False)

# --- LV conductors (rank 1 = cheapest/smallest) ---
pd.DataFrame(
    [
        {"conductor_id": "LV_AL_25", "voltage_level": "LV", "r_ohm_per_km": 1.20, "x_ohm_per_km": 0.090, "max_i_ka": 0.10, "cost_per_m": 6.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "LV_AL_50", "voltage_level": "LV", "r_ohm_per_km": 0.64, "x_ohm_per_km": 0.085, "max_i_ka": 0.16, "cost_per_m": 9.0, "phase_count": 3, "rank": 2},
        {"conductor_id": "LV_AL_95", "voltage_level": "LV", "r_ohm_per_km": 0.32, "x_ohm_per_km": 0.078, "max_i_ka": 0.24, "cost_per_m": 14.0, "phase_count": 3, "rank": 3},
        {"conductor_id": "LV_AL_150", "voltage_level": "LV", "r_ohm_per_km": 0.21, "x_ohm_per_km": 0.075, "max_i_ka": 0.32, "cost_per_m": 20.0, "phase_count": 3, "rank": 4},
    ]
).to_csv(OUT / "lv_conductors_example.csv", index=False)

# --- MV conductors ---
pd.DataFrame(
    [
        {"conductor_id": "MV_AL_35", "voltage_level": "MV", "r_ohm_per_km": 0.87, "x_ohm_per_km": 0.40, "max_i_ka": 0.14, "cost_per_m": 10.0, "phase_count": 3, "rank": 1},
        {"conductor_id": "MV_AL_70", "voltage_level": "MV", "r_ohm_per_km": 0.44, "x_ohm_per_km": 0.38, "max_i_ka": 0.22, "cost_per_m": 15.0, "phase_count": 3, "rank": 2},
    ]
).to_csv(OUT / "mv_conductors_example.csv", index=False)

# --- Transformers (rank by size) ---
pd.DataFrame(
    [
        {"transformer_id": "TX_25", "sn_kva": 25, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.0, "vkr_percent": 1.2, "pfe_kw": 0.1, "i0_percent": 0.5, "cost_fixed": 3000, "cost_per_kva": 60, "max_loading_percent": 100, "rank": 1},
        {"transformer_id": "TX_50", "sn_kva": 50, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.0, "vkr_percent": 1.1, "pfe_kw": 0.15, "i0_percent": 0.4, "cost_fixed": 4000, "cost_per_kva": 55, "max_loading_percent": 100, "rank": 2},
        {"transformer_id": "TX_100", "sn_kva": 100, "vn_hv_kv": 11.0, "vn_lv_kv": 0.4, "vk_percent": 4.5, "vkr_percent": 1.0, "pfe_kw": 0.25, "i0_percent": 0.3, "cost_fixed": 6000, "cost_per_kva": 50, "max_loading_percent": 100, "rank": 3},
    ]
).to_csv(OUT / "transformers_example.csv", index=False)

# --- Stand-alone packages (Solar Home Systems / mini-PV) ---
pd.DataFrame(
    [
        {"package_id": "SHS_T1", "service_tier": 1, "max_peak_kw": 0.3, "annual_kwh": 550, "capex": 350, "lifetime_years": 8, "annual_om_fraction": 0.04, "replacement_fraction": 0.6},
        {"package_id": "SHS_T2", "service_tier": 2, "max_peak_kw": 0.6, "annual_kwh": 1100, "capex": 700, "lifetime_years": 8, "annual_om_fraction": 0.04, "replacement_fraction": 0.6},
        {"package_id": "SHS_T3", "service_tier": 3, "max_peak_kw": 1.5, "annual_kwh": 3500, "capex": 1800, "lifetime_years": 10, "annual_om_fraction": 0.035, "replacement_fraction": 0.6},
    ]
).to_csv(OUT / "standalone_packages_example.csv", index=False)

# --- Cost parameters (key/value layout) ---
cost = {
    "discount_rate": 0.08,
    "planning_horizon_years": 15,
    "lv_connection_cost_per_customer": 100.0,
    "metering_protection_cost_per_customer": 50.0,
    "pole_cost": 120.0,
    "civil_logistics_multiplier": 1.2,
    "loss_value_per_kwh": 0.20,
    "loss_load_factor": 0.30,
    "voltage_min_pu": 0.90,
    "voltage_max_pu": 1.05,
    "max_line_loading_percent": 100.0,
    "max_transformer_loading_percent": 100.0,
    "dmax_lv_service_zone_m": 700.0,
    "lmax_lv_path_m": 850.0,
    "remote_branch_length_threshold_m": 300.0,
    "branch_cost_per_customer_threshold": 550.0,
    "loss_threshold_percent": 8.0,
}
pd.DataFrame({"parameter": list(cost), "value": list(cost.values())}).to_csv(
    OUT / "cost_parameters_example.csv", index=False
)

# --- Roads (GeoJSON): trunk source->core->institutional + spur to sparse cluster ---
import geopandas as gpd
from shapely.geometry import LineString


def line(coords_m):
    return LineString([to_latlon(x, y)[::-1] for (x, y) in coords_m])  # (lon, lat)


roads = gpd.GeoDataFrame(
    {"road_id": ["trunk", "spur_inst", "spur_sparse"]},
    geometry=[
        line([(0, 0), (60, 60), (120, 120), (40, 400), (-150, 700), (-200, 860)]),
        line([(-150, 700), (-200, 860)]),
        line([(120, 120), (500, 150), (960, 200), (1350, 250)]),
    ],
    crs="EPSG:4326",
)
roads.to_file(OUT / "roads_example.geojson", driver="GeoJSON")

print(f"Wrote {len(customers)} customers and example files to {OUT}")
