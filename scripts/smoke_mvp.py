import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io.example_data import load_example_inputs
from core.workflow.mvp_runner import run_simplified_mvp

inp = load_example_inputs()
eng = {
    "max_user_connection_radius_m": 35, "max_users_per_pole": 16, "max_pole_span_m": 50,
    "road_pole_spacing_m": 40, "planning_horizon_years": 15, "discount_rate": 0.08,
    "voltage_min_pu": 0.90, "voltage_max_pu": 1.05, "max_line_loading_percent": 100,
    "max_transformer_loading_percent": 100, "dmax_lv_service_zone_m": 700, "lmax_lv_path_m": 850,
}
res = run_simplified_mvp(
    inp["customers"], inp["roads"], inp["source"], inp["lv_conductors"], inp["mv_conductors"],
    inp["transformers"], inp["standalone_packages"], inp["cost_parameters"], eng,
    progress=lambda m: print("  ..", m),
)
print("RECOMMENDED:", res.recommended, "|", res.recommendation_note)
cols = ["candidate_id", "feasible", "npc_total", "n_grid_customers", "n_standalone_customers",
        "n_transformers", "min_voltage_horizon", "max_line_loading_horizon", "horizon_growth_feasible"]
print(res.comparison_table[cols].to_string())
print("clusters:", [(c.cluster_id, c.n_customers, c.is_problematic, c.standalone_eligible_precheck, c.reasons) for c in res.clusters])
print("strategy status counts:\n", res.customer_strategy_table["recommended_status"].value_counts().to_string())

# Exercise the export + map-building paths used by the Streamlit page.
from core.workflow.exports import build_exports
from core.viz.maps import case_map
ex = build_exports(res, inp["customers"], inp["source"])
print("export files:", len(ex), "| zip bytes:", len(ex["all_outputs.zip"]))
for k in res.cases:
    case_map(res.cases[k], inp["customers"], inp["source"])
    print("map ok:", k)
print("INTEGRATION OK")
