"""Run Mafa and NGA-dense through the MVP workflow and produce a comparative analysis.

Outputs (under docs/):
  * docs/figures/geometry_compare.png       -- settlement geometry, same scale
  * docs/figures/npc_breakdown.png          -- NPC by component, recommended case each
  * docs/figures/architecture_kpis.png      -- zones / SHS / coverage / NPC-per-customer
  * docs/figures/case_metrics.json          -- machine-readable metrics for the writeup

Run:  python scripts/compare_case_studies.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io.example_data import load_example_inputs
from core.workflow.mvp_runner import run_simplified_mvp

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = {"mafa": "Mafa (dispersed)", "nga_dense": "NGA-dense (compact)"}
TYPE_COLORS = {
    "household": "#4C78A8", "hospital": "#E45756", "school": "#F58518",
    "mosque": "#54A24B", "productive": "#B279A2", "shop": "#EECA3B",
    "public_service": "#72B7B2",
}
COMPONENTS = [
    ("lv_line_cost", "LV lines", "#4C78A8"),
    ("mv_line_cost", "MV lines", "#E45756"),
    ("pole_cost", "Poles", "#9D755D"),
    ("transformer_cost", "Transformers", "#F58518"),
    ("connection_metering_cost", "Connection/metering", "#54A24B"),
    ("standalone_npc", "Stand-alone (SHS)", "#B279A2"),
    ("loss_pv_cost", "Losses (PV)", "#BAB0AC"),
]


def eng_from_cp(cp):
    return {
        "max_user_connection_radius_m": 40, "max_users_per_pole": 12, "max_pole_span_m": 45,
        "road_pole_spacing_m": 35,
        "planning_horizon_years": int(cp.get("planning_horizon_years", 15)),
        "discount_rate": float(cp.get("discount_rate", 0.10)),
        "voltage_min_pu": float(cp.get("voltage_min_pu", 0.90)),
        "voltage_max_pu": float(cp.get("voltage_max_pu", 1.05)),
        "max_line_loading_percent": float(cp.get("max_line_loading_percent", 100)),
        "max_transformer_loading_percent": float(cp.get("max_transformer_loading_percent", 100)),
        "dmax_lv_service_zone_m": float(cp.get("dmax_lv_service_zone_m", 700)),
        "lmax_lv_path_m": float(cp.get("lmax_lv_path_m", 850)),
    }


def spatial_stats(customers: pd.DataFrame):
    g = gpd.GeoDataFrame(customers, geometry=gpd.points_from_xy(
        customers.longitude, customers.latitude), crs=4326)
    g = g.to_crs(g.estimate_utm_crs())
    d = g.distance(g.union_all().centroid)
    hull_km2 = g.union_all().convex_hull.area / 1e6
    return {
        "n_customers": int(len(customers)),
        "total_peak_kw": round(float(customers.peak_kw_now.sum()), 1),
        "total_annual_mwh": round(float(customers.annual_kwh_now.sum()) / 1000, 1),
        "spread_median_m": round(float(d.median()), 0),
        "spread_p90_m": round(float(d.quantile(0.9)), 0),
        "spread_max_m": round(float(d.max()), 0),
        "hull_area_km2": round(float(hull_km2), 3),
        "density_per_km2": round(float(len(customers) / hull_km2), 0),
        "n_institutions": int((customers.customer_type != "household").sum()),
    }


def case_metrics(res):
    out = {"recommended": res.recommended, "cases": {}}
    total = None
    for cid, c in res.cases.items():
        oh = c.outcome_horizon
        cost = c.cost
        n_zones = len(c.plan.zones) if (c.plan is not None and getattr(c.plan, "zones", None)) else (
            c.n_transformers)
        if total is None:
            total = c.n_grid_customers + c.n_standalone_customers
        out["cases"][cid] = {
            "feasible": bool(c.feasible),
            "npc_total": round(float(cost.total_npc), 0),
            "grid_capex": round(float(cost.grid_capex), 0),
            "components": {k: round(float(getattr(cost, k)), 0) for k, _, _ in COMPONENTS},
            "n_grid": int(c.n_grid_customers),
            "n_standalone": int(c.n_standalone_customers),
            "n_transformers": int(c.n_transformers),
            "n_zones": int(n_zones),
            "lv_length_m": round(float(c.lv_length_m), 0),
            "mv_length_m": round(float(c.mv_length_m), 0),
            "n_poles": int(c.n_poles),
            "min_vm_pu": round(float(oh.min_vm_pu), 3) if oh and oh.converged else None,
            "max_line_pct": round(float(oh.max_line_loading_percent), 0) if oh and oh.converged else None,
            "max_trafo_pct": round(float(oh.max_trafo_loading_percent), 0) if oh and oh.converged else None,
        }
    out["total_customers"] = int(total or 0)
    return out


def run_all():
    metrics = {}
    for key in DATASETS:
        inp = load_example_inputs(key)
        t0 = time.time()
        res = run_simplified_mvp(
            inp["customers"], inp["roads"], inp["source"], inp["lv_conductors"],
            inp["mv_conductors"], inp["transformers"], inp["standalone_packages"],
            inp["cost_parameters"], eng_from_cp(inp["cost_parameters"]),
            progress=lambda m: None)
        m = {"spatial": spatial_stats(inp["customers"]), **case_metrics(res)}
        rec = m["cases"][m["recommended"]]
        m["grid_coverage_pct"] = round(100 * rec["n_grid"] / m["total_customers"], 1)
        m["shs_coverage_pct"] = round(100 * rec["n_standalone"] / m["total_customers"], 1)
        m["npc_per_customer"] = round(rec["npc_total"] / m["total_customers"], 0)
        m["runtime_s"] = round(time.time() - t0, 1)
        metrics[key] = m
        print(f"[{key}] {m['runtime_s']}s  recommended={m['recommended']}  "
              f"NPC={rec['npc_total']:.0f}  zones={rec['n_zones']}  SHS={rec['n_standalone']}  "
              f"cover={m['grid_coverage_pct']}%")
    (FIG / "case_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def fig_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, key in zip(axes, DATASETS):
        inp = load_example_inputs(key)
        df = inp["customers"]
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
        utm = g.estimate_utm_crs(); g = g.to_crs(utm)
        cx, cy = g.geometry.x.mean(), g.geometry.y.mean()
        if inp["roads"] is not None:
            inp["roads"].to_crs(utm).plot(ax=ax, color="0.75", linewidth=0.8, zorder=1)
        src = inp["source"]
        for t, sub in g.groupby("customer_type"):
            ax.scatter(sub.geometry.x - cx, sub.geometry.y - cy, s=16,
                       c=TYPE_COLORS.get(t, "0.3"), label=f"{t} ({len(sub)})", zorder=3, edgecolors="none")
        sg = gpd.GeoSeries([gpd.points_from_xy([src["longitude"]], [src["latitude"]])[0]], crs=4326).to_crs(utm)
        ax.scatter(sg.x.iloc[0] - cx, sg.y.iloc[0] - cy, marker="*", s=280, c="black",
                   label="grid source", zorder=4)
        ax.set_title(f"{DATASETS[key]}  —  {len(df)} customers, {df.peak_kw_now.sum():.0f} kW")
        ax.set_aspect("equal"); ax.set_xlim(-650, 650); ax.set_ylim(-650, 650)
        ax.axhline(0, color="0.92", lw=0.5); ax.axvline(0, color="0.92", lw=0.5)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax.set_xlabel("metres from settlement centroid")
    fig.suptitle("Settlement geometry at comparable scale (same axes)", fontsize=13)
    plt.tight_layout(); plt.savefig(FIG / "geometry_compare.png", dpi=115); plt.close()


def fig_npc_breakdown(metrics):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    keys = list(DATASETS)
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys))
    for comp, label, color in COMPONENTS:
        vals = np.array([metrics[k]["cases"][metrics[k]["recommended"]]["components"][comp] for k in keys])
        ax.bar(x, vals / 1000, bottom=bottoms / 1000, label=label, color=color, width=0.55)
        bottoms += vals
    for xi, k in zip(x, keys):
        rec = metrics[k]["cases"][metrics[k]["recommended"]]
        ax.text(xi, rec["npc_total"] / 1000 + 4, f"{metrics[k]['recommended']}\n{rec['npc_total']/1000:.0f}k",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([DATASETS[k] for k in keys])
    ax.set_ylabel("NPC (thousand USD)")
    ax.set_title("Least-cost architecture: NPC by component (recommended case)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.margins(y=0.15)
    plt.tight_layout(); plt.savefig(FIG / "npc_breakdown.png", dpi=115); plt.close()


def fig_kpis(metrics):
    keys = list(DATASETS)
    labels = ["Transformer\nzones", "SHS units", "Grid coverage\n(%)", "NPC per\ncustomer (USD)"]
    getters = [
        lambda m: m["cases"][m["recommended"]]["n_zones"],
        lambda m: m["cases"][m["recommended"]]["n_standalone"],
        lambda m: m["grid_coverage_pct"],
        lambda m: m["npc_per_customer"],
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    colors = ["#E45756", "#4C78A8"]
    for ax, lab, get in zip(axes, labels, getters):
        vals = [get(metrics[k]) for k in keys]
        ax.bar([DATASETS[k].split(" ")[0] for k in keys], vals, color=colors, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(lab, fontsize=11); ax.margins(y=0.2)
    fig.suptitle("Recommended-solution KPIs: dispersed vs compact (matched demand)", fontsize=13)
    plt.tight_layout(); plt.savefig(FIG / "architecture_kpis.png", dpi=115); plt.close()


if __name__ == "__main__":
    metrics = run_all()
    fig_geometry()
    fig_npc_breakdown(metrics)
    fig_kpis(metrics)
    print("Wrote figures + metrics to", FIG)
