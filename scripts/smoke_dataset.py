"""End-to-end smoke for any registered example dataset.

Usage:  python scripts/smoke_dataset.py [dataset_key]   (default: mafa)
        dataset_key in: synthetic | mafa | realistic_ssa
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io.example_data import load_example_inputs
from core.workflow.mvp_runner import run_simplified_mvp

dataset = sys.argv[1] if len(sys.argv) > 1 else "mafa"
inp = load_example_inputs(dataset)
cp = inp["cost_parameters"]
eng = {
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
print(f"DATASET: {dataset}  ({len(inp['customers'])} customers)")
t0 = time.time()
res = run_simplified_mvp(
    inp["customers"], inp["roads"], inp["source"], inp["lv_conductors"], inp["mv_conductors"],
    inp["transformers"], inp["standalone_packages"], inp["cost_parameters"], eng,
    progress=lambda m: print(f"  [{time.time()-t0:5.1f}s] {m}"),
)
print(f"TOTAL {time.time()-t0:.1f}s | RECOMMENDED {res.recommended}")
cols = ["candidate_id", "feasible", "npc_total", "n_grid_customers", "n_standalone_customers",
        "n_transformers", "must_connect_violations", "service_tier_violations", "min_voltage_horizon"]
print(res.comparison_table[cols].to_string())
print("strategy:\n", res.customer_strategy_table["recommended_status"].value_counts().to_string())
for case_id in ("C_MVLV", "D_MVLV_plus_standalone"):
    mvlv = res.cases.get(case_id)
    if mvlv is not None and mvlv.plan is not None:
        zs = mvlv.plan.zones
        print(f"{case_id} zones: {len(zs)}  transformers chosen:",
              {z["transformer_spec"]["transformer_id"]: z["s_zone_kva"] for z in zs})
for k, r in res.cases.items():
    o = r.outcome_horizon
    if o:
        print(f"{k}: conv={o.converged} feas={o.feasible} minV={o.min_vm_pu:.3f} maxV={o.max_vm_pu:.3f} "
              f"maxLine={o.max_line_loading_percent:.0f}% maxTrafo={o.max_trafo_loading_percent:.0f}% | {o.active_bottleneck}")
