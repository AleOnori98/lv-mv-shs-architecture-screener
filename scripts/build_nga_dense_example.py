"""Build the ``nga_dense`` counterfactual case study from the real NGA 1 settlement.

Purpose
-------
Provide a second, *comparable* case study to the Mafa survey so that a preliminary
sensitivity analysis can isolate the effect of **geographic dispersion** and **demand
distribution** on the identified least-cost electrification architecture (full LV vs
MV/LV) and on stand-alone (SHS) identification / coverage.

Design (defendable, deliberately simple)
----------------------------------------
* **Scale is held comparable to Mafa** (~240 connections, ~240 kW aggregate peak).
* **Structure is varied**: the settlement is *compact / monocentric* (the real nga_1
  footprint, densified) and the demand is *homogeneous / residential* (households +
  a small central market), i.e. the opposite of Mafa's dispersed, institution-dominated
  (2 x 53 kW hospitals, one remote) demand.
* Everything else is identical to Mafa: demand model (MTF ``TIER_DEMAND`` + per-type
  values), LV/MV/transformer/SHS catalogues and cost parameters. This maximises
  comparability so differences in the result are attributable to geometry + demand mix.

Source signal (``examples/nga_1/``)
  * ``customers.csv`` -- 128 real georeferenced household points (kept verbatim).
  * ``roads.geojson`` / ``source.csv`` -- real road network and grid feed-in (kept).

Run:  python scripts/build_nga_dense_example.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples" / "nga_1"
MAFA = ROOT / "examples" / "mafa case study"
OUT = ROOT / "examples" / "nga_dense"

SEED = 20260701
TARGET_CUSTOMERS = 240          # match Mafa's connection count

# --- Shared demand model (identical to build_mafa_example) --------------------
# MTF demand tier -> (peak_kW, annual_kWh) for households.
TIER_DEMAND = {2: (0.25, 400.0), 3: (0.50, 900.0), 4: (1.00, 1800.0)}
# Household tier mix: a compact, more-electrified peri-urban town (higher tiers than
# Mafa's 55/35/10) so the aggregate demand matches Mafa without any large anchor load.
HH_TIER_MIX = {2: 0.15, 3: 0.40, 4: 0.45}

# Central market / institution core (compact, no large remote anchor). Each tuple is
# (customer_type, peak_kW, annual_kWh, demand_tier, must_connect). Sized to keep the
# demand distribution homogeneous (largest single load 8 kW, vs Mafa's 53 kW hospital).
INSTITUTIONS = [
    ("public_service", 8.0, 18000.0, 4, True),    # health clinic
    ("school",         6.0,  7000.0, 3, True),
    ("school",         6.0,  7000.0, 3, True),
    ("public_service", 3.0,  6500.0, 4, True),     # admin office
    ("mosque",         1.2,  1500.0, 3, True),
    ("mosque",         1.2,  1500.0, 3, True),
    ("shop",           8.0, 16000.0, 4, False),
    ("shop",           8.0, 16000.0, 4, False),
    ("shop",           8.0, 16000.0, 4, False),
    ("shop",           8.0, 16000.0, 4, False),
    ("shop",           8.0, 16000.0, 4, False),
    ("productive",     5.0,  9000.0, 4, False),
    ("productive",     5.0,  9000.0, 4, False),
    ("productive",     5.0,  9000.0, 4, False),
]

PF = {"household": 0.95, "shop": 0.90, "productive": 0.85, "public_service": 0.92,
      "school": 0.90, "hospital": 0.90, "mosque": 0.95}
GROWTH = {"household": 0.04, "shop": 0.065, "productive": 0.075, "public_service": 0.035,
          "school": 0.035, "hospital": 0.03, "mosque": 0.025}
PRIORITY = {"hospital": 3.0, "school": 2.0, "public_service": 2.0, "mosque": 1.5,
            "productive": 1.5, "shop": 1.5, "household": 1.0}

CATALOGUES = ("lv_conductors.csv", "mv_conductors.csv", "transformers.csv",
              "standalone_packages.csv", "cost_parameters.csv")


def _infill_points(gproj: gpd.GeoSeries, n_add: int, rng: np.random.Generator) -> gpd.GeoSeries:
    """Add ``n_add`` new points by sampling near existing ones (compact densification)."""
    xs = np.asarray(gproj.geometry.x)
    ys = np.asarray(gproj.geometry.y)
    hull = gproj.geometry.union_all().convex_hull.buffer(15)  # stay within the footprint
    new_x, new_y = [], []
    while len(new_x) < n_add:
        i = rng.integers(0, len(xs))
        r = float(np.clip(rng.normal(12.0, 6.0), 5.0, 30.0))   # 5-30 m offset
        th = rng.uniform(0, 2 * np.pi)
        px, py = xs[i] + r * np.cos(th), ys[i] + r * np.sin(th)
        if hull.contains(gpd.points_from_xy([px], [py])[0]):
            new_x.append(px)
            new_y.append(py)
    return gpd.GeoSeries(gpd.points_from_xy(new_x, new_y), crs=gproj.crs)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(SRC / "customers.csv")
    gdf = gpd.GeoDataFrame(
        real, geometry=gpd.points_from_xy(real.longitude, real.latitude), crs="EPSG:4326")
    utm = gdf.estimate_utm_crs()
    gproj = gdf.to_crs(utm)

    # 1) Densify: keep the 128 real points, infill up to TARGET_CUSTOMERS.
    n_add = TARGET_CUSTOMERS - len(gproj)
    infill = _infill_points(gproj["geometry"], n_add, rng)
    all_pts = gpd.GeoSeries(pd.concat([gproj.geometry, infill], ignore_index=True), crs=utm)
    ll = all_pts.to_crs(4326)
    n = len(ll)

    # 2) Types: reassign the points closest to the centroid to the institution core.
    centroid = all_pts.union_all().centroid
    order = np.argsort(all_pts.distance(centroid).to_numpy())
    inst_idx = list(order[: len(INSTITUTIONS)])

    ctype = np.array(["household"] * n, dtype=object)
    peak = np.zeros(n)
    kwh = np.zeros(n)
    tier = np.zeros(n, dtype=int)
    must = np.zeros(n, dtype=bool)
    for slot, idx in enumerate(inst_idx):
        t, pk, kw, ti, mc = INSTITUTIONS[slot]
        ctype[idx], peak[idx], kwh[idx], tier[idx], must[idx] = t, pk, kw, ti, mc

    # 3) Households: assign an MTF tier mix, mapped to the shared TIER_DEMAND model.
    hh_idx = np.array([i for i in range(n) if i not in set(inst_idx)])
    tiers = np.array(sorted(HH_TIER_MIX))
    probs = np.array([HH_TIER_MIX[t] for t in tiers])
    draw = rng.choice(tiers, size=len(hh_idx), p=probs / probs.sum())
    for i, ti in zip(hh_idx, draw):
        pk, kw = TIER_DEMAND[int(ti)]
        peak[i], kwh[i], tier[i] = pk, kw, int(ti)

    df = pd.DataFrame({
        "customer_id": [str(i + 1) for i in range(n)],
        "latitude": np.round(ll.geometry.y.to_numpy(), 6),
        "longitude": np.round(ll.geometry.x.to_numpy(), 6),
        "customer_type": ctype,
        "peak_kw_now": np.round(peak, 3),
        "annual_kwh_now": np.round(kwh, 0),
        "power_factor": [PF.get(t, 0.90) for t in ctype],
        "growth_rate": [GROWTH.get(t, 0.04) for t in ctype],
        "demand_tier": tier,
        "priority_weight": [PRIORITY.get(t, 1.0) for t in ctype],
        "must_connect_flag": must,
    })
    df["can_be_standalone_flag"] = (df["customer_type"] == "household") & ~df["must_connect_flag"]
    df.to_csv(OUT / "customers.csv", index=False)

    # 4) Roads, source, catalogues, cost parameters -> copied verbatim for comparability.
    shutil.copy(SRC / "roads.geojson", OUT / "roads.geojson")
    shutil.copy(SRC / "source.csv", OUT / "source.csv")
    for name in CATALOGUES:
        shutil.copy(MAFA / name, OUT / name)

    # 5) Diagnostics.
    g2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326).to_crs(utm)
    d = g2.distance(g2.union_all().centroid)
    hull_km2 = g2.union_all().convex_hull.area / 1e6
    print(f"Wrote {n} customers to {OUT}")
    print(f"  total peak = {df.peak_kw_now.sum():.1f} kW   annual = {df.annual_kwh_now.sum()/1000:.1f} MWh")
    print(f"  spread from centroid: median={d.median():.0f} m  p90={d.quantile(.9):.0f} m  max={d.max():.0f} m")
    print(f"  convex-hull area={hull_km2:.3f} km2  density={n/hull_km2:.0f} cust/km2")
    print("  type mix:\n" + df.customer_type.value_counts().to_string())
    print("  household tier mix:\n" + df[df.customer_type == "household"].demand_tier.value_counts().sort_index().to_string())
    print(f"  must_connect={int(df.must_connect_flag.sum())}  can_standalone={int(df.can_be_standalone_flag.sum())}")


if __name__ == "__main__":
    main()
