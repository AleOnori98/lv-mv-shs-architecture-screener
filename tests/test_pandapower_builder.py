"""pandapower network construction + power flow."""
from __future__ import annotations

import pandapower as pp

from core.io.load_inputs import customers_to_engine_gdf
from core.topology.lv_topology import build_lv_topology
from core.electrical.pandapower_builder import build_lv_net
from core.electrical.validation import run_powerflow


def test_pandapower_network_builds_and_runs(inputs, engineering):
    gdf, crs_meta = customers_to_engine_gdf(inputs["customers"])
    topo = build_lv_topology(gdf, inputs["roads"], crs_meta, inputs["source"], engineering)
    cheapest = inputs["lv_conductors"].sort_values("rank").iloc[0].to_dict()
    cond_by_line = {str(e["line_id"]): cheapest for _, e in topo.edges_4326.iterrows()}

    model = build_lv_net(topo, inputs["customers"], cond_by_line, lv_kv=0.4, horizon_years=15, apply_growth=True)
    assert len(model.net.bus) == len(topo.pole_xy)
    assert len(model.net.line) == len(topo.edges_4326)
    assert len(model.net.ext_grid) == 1

    pp.runpp(model.net)
    assert model.net.converged

    outcome = run_powerflow(model, {
        "voltage_min_pu": 0.90, "voltage_max_pu": 1.05,
        "max_line_loading_percent": 100, "max_transformer_loading_percent": 100,
    })
    assert outcome.converged
    assert 0.0 < outcome.min_vm_pu <= 1.05
    assert outcome.total_load_kw > 0


def test_coincidence_factor_scales_load_and_stress(inputs, engineering):
    """A diversified design peak (<1) loads the network less than the coincident peak (1.0)."""
    gdf, crs_meta = customers_to_engine_gdf(inputs["customers"])
    topo = build_lv_topology(gdf, inputs["roads"], crs_meta, inputs["source"], engineering)
    cheapest = inputs["lv_conductors"].sort_values("rank").iloc[0].to_dict()
    cond_by_line = {str(e["line_id"]): cheapest for _, e in topo.edges_4326.iterrows()}
    constraints = {"voltage_min_pu": 0.90, "voltage_max_pu": 1.05,
                   "max_line_loading_percent": 100, "max_transformer_loading_percent": 100}

    design = build_lv_net(topo, inputs["customers"], cond_by_line, lv_kv=0.4, horizon_years=15,
                          apply_growth=True, coincidence_factor=0.6)
    stress = build_lv_net(topo, inputs["customers"], cond_by_line, lv_kv=0.4, horizon_years=15,
                          apply_growth=True, coincidence_factor=1.0)

    assert abs(design.total_load_kw - 0.6 * stress.total_load_kw) < 1e-6
    d_out = run_powerflow(design, constraints)
    s_out = run_powerflow(stress, constraints)
    # Coincident peak draws more current -> lower voltage and higher line loading.
    assert s_out.min_vm_pu <= d_out.min_vm_pu
    assert s_out.max_line_loading_percent >= d_out.max_line_loading_percent
