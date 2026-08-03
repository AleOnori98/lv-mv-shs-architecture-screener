"""End-to-end MVP workflow tests."""
from __future__ import annotations

from core.workflow.exports import case_lines_gdf


def test_example_inputs_load_successfully(inputs):
    assert len(inputs["customers"]) >= 40
    for key in ("source", "lv_conductors", "mv_conductors", "transformers", "standalone_packages", "cost_parameters"):
        assert inputs[key] is not None
    assert {"customer_id", "latitude", "longitude", "customer_type"} <= set(inputs["customers"].columns)


def test_full_lv_topology_has_all_customers_connected(mvp, inputs):
    full = mvp.cases["A_full_LV"]
    assert full.n_grid_customers == len(inputs["customers"])
    assert full.topology.unserved_customer_ids == []


def test_four_candidates_present(mvp):
    assert set(mvp.cases) == {"A_full_LV", "B_LV_plus_standalone", "C_MVLV", "D_MVLV_plus_standalone"}
    for case in mvp.cases.values():
        assert case.outcome_horizon is not None
        assert case.outcome_horizon.converged


def test_mvlv_uses_explicit_mv_tree(mvp, inputs):
    case = mvp.cases["C_MVLV"]
    assert case.plan.mv_topology is not None
    assert len(case.plan.mv_topology.edges_4326) > 0
    lines = case_lines_gdf(case, inputs["source"])
    mv_lines = lines[lines["voltage_level"] == "MV"]
    assert not mv_lines.empty
    assert all(str(line_id) in case.model.line_index_by_id for line_id in mv_lines["line_id"])
    hybrid = mvp.cases["D_MVLV_plus_standalone"]
    assert hybrid.plan.mv_topology is not None


def test_mvlv_has_single_slack_and_explicit_stepup(mvp):
    """Case C is fed through ONE slack on the plant LV bus and an explicit step-up transformer;
    the MV backbone sits below the sending voltage (a real drop, not the old flat 1.04 pu slack)."""
    for cid in ("C_MVLV", "D_MVLV_plus_standalone"):
        case = mvp.cases[cid]
        if case.model is None:  # D can be empty if everyone goes stand-alone
            continue
        net = case.model.net
        assert len(net.ext_grid) == 1
        assert case.model.stepup_trafo_index is not None
        assert "stepup_tx" in set(net.trafo["name"])
        assert float(net.ext_grid.vm_pu.iloc[0]) <= 1.001  # honest sending voltage, not 1.04
        mv_vm = net.res_bus.loc[net.bus["vn_kv"] > 1.0, "vm_pu"]
        assert not mv_vm.empty and mv_vm.max() < 1.0  # MV energised only through the step-up


def test_mvlv_records_transformer_count_search(mvp):
    """The transformer count is chosen by a criterion and its search history is exposed."""
    case = mvp.cases["C_MVLV"]
    assert case.plan.clustering_criterion in {"voltage_cap", "distance_cap", "capacity"}
    assert len(case.plan.mv_iterations) >= 1
    assert case.n_transformers == len(case.plan.zones)


def test_every_candidate_has_a_coincident_stress_outcome(mvp):
    """The coincident-peak (factor 1.0) stress PF is computed and exposed for every candidate."""
    for case in mvp.cases.values():
        assert case.outcome_stress is not None
    for col in ("coincident_within_limits", "coincident_min_voltage", "coincident_max_line_loading"):
        assert col in mvp.comparison_table.columns


def test_generation_npc_is_a_positive_component_for_grid_candidates(mvp):
    cbt = mvp.cost_breakdown_table.set_index("candidate_id")
    assert "generation_npc" in cbt.columns
    # Full LV serves everyone on the grid, so its generation NPC must be strictly positive.
    assert cbt.loc["A_full_LV", "generation_npc"] > 0


def test_tier_service_table_covers_every_customer_per_candidate(mvp, inputs):
    tst = mvp.tier_service_table
    n = len(inputs["customers"])
    for cid in mvp.cases:
        sub = tst[tst["candidate_id"] == cid]
        assert len(sub) == n
    # Stand-alone customers carry a package service-tier ceiling; grid customers do not.
    sa = tst[tst["status"] == "standalone"]
    if not sa.empty:
        assert sa["service_ceiling_tier"].notna().all()


def test_candidate_comparison_selects_feasible_lowest_npc(mvp):
    comp = mvp.comparison_table
    feasible = comp[comp["feasible"]]
    assert not feasible.empty
    best = feasible.sort_values("npc_total").iloc[0]["candidate_id"]
    assert mvp.recommended == best


def test_decision_log_and_strategy_complete(mvp, inputs):
    assert not mvp.decision_log.empty
    assert len(mvp.customer_strategy_table) == len(inputs["customers"])
    assert (mvp.customer_strategy_table["recommended_status"] != "not_served_error").all()
