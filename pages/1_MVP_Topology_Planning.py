"""Streamlit page: Simplified LV/MV + Stand-alone MVP (end-to-end pipeline)."""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from core.io import load_inputs, example_data
from core.workflow.mvp_runner import run_simplified_mvp
from core.workflow.exports import build_exports
from core.viz.maps import case_map, input_map, peak_demand_map, LEGEND
from core.viz.charts import (
    case_headroom_html,
    npc_breakdown_chart,
    npc_share_chart,
    parallel_coordinates_chart,
    shs_decision_chart,
    tier_service_chart,
)
from core.viz.tables import (
    cost_parameters_table,
    customer_type_summary,
    highlight_recommended,
    npc_components,
    roads_preview_table,
    roads_summary_table,
    shs_screening_table,
    source_table,
)

st.set_page_config(page_title="LV/MV + Stand-alone MVP", layout="wide")
st.title("Simplified LV/MV + Stand-alone MVP")
st.caption(
    "A sequential diagnostic workflow (not a co-optimization model) comparing **Full LV**, "
    "**LV + Stand-alone**, **MV/LV service zones**, and **MV/LV + Stand-alone** by least-feasible net present cost. "
    "Validation uses pandapower (balanced three-phase)."
)

CANDIDATES = {
    "A_full_LV": "A · Full LV",
    "B_LV_plus_standalone": "B · LV + Stand-alone",
    "C_MVLV": "C · MV/LV zones",
    "D_MVLV_plus_standalone": "D · MV/LV + Stand-alone",
}

# --------------------------------------------------------------------------
# Section 1 — Inputs
# --------------------------------------------------------------------------
st.header("1 · Inputs")
mode = st.radio("Data source", ["Use example dataset", "Upload my own files"], horizontal=True)

inputs = None
if mode == "Use example dataset":
    datasets = example_data.discover_example_datasets()
    labels = {k: v["label"] for k, v in datasets.items()}
    if labels:
        chosen = st.selectbox("Example project", list(labels), format_func=lambda k: labels[k])
        if st.button("Load example project", type="primary"):
            st.session_state["mvp_inputs"] = example_data.load_example_inputs(chosen)
            st.session_state["mvp_dataset"] = chosen
            st.success(f"Loaded: {labels[chosen]}")
    else:
        st.warning("No loadable project folders found in examples/.")
    inputs = st.session_state.get("mvp_inputs")
else:
    cols = st.columns(2)
    up = {
        "customers": cols[0].file_uploader("customers.csv", type=["csv"]),
        "source": cols[0].file_uploader("source.csv", type=["csv"]),
        "lv_conductors": cols[0].file_uploader("lv_conductors.csv", type=["csv"]),
        "mv_conductors": cols[0].file_uploader("mv_conductors.csv", type=["csv"]),
        "transformers": cols[1].file_uploader("transformers.csv", type=["csv"]),
        "standalone_packages": cols[1].file_uploader("standalone_packages.csv", type=["csv"]),
        "cost_parameters": cols[1].file_uploader("cost_parameters.csv / .yaml", type=["csv", "yaml", "yml"]),
        "roads": cols[1].file_uploader("roads.geojson / .gpkg (optional)", type=["geojson", "json", "gpkg"]),
    }
    required = ("customers", "source", "lv_conductors", "mv_conductors", "transformers", "standalone_packages", "cost_parameters")
    if all(up[k] is not None for k in required):
        try:
            st.session_state["mvp_inputs"] = {
                "customers": load_inputs.load_customers(up["customers"]),
                "source": load_inputs.load_source(up["source"]),
                "lv_conductors": load_inputs.load_lv_conductors(up["lv_conductors"]),
                "mv_conductors": load_inputs.load_mv_conductors(up["mv_conductors"]),
                "transformers": load_inputs.load_transformers(up["transformers"]),
                "standalone_packages": load_inputs.load_standalone_packages(up["standalone_packages"]),
                "cost_parameters": load_inputs.load_cost_parameters(up["cost_parameters"]),
                "roads": load_inputs.load_roads(up["roads"]) if up["roads"] else None,
            }
            st.session_state["mvp_dataset"] = "uploaded"
            st.success("Uploaded inputs parsed successfully.")
        except Exception as exc:
            st.error(f"Could not parse inputs: {exc}")
    inputs = st.session_state.get("mvp_inputs")

if inputs is not None:
    customers = inputs["customers"]
    roads = inputs.get("roads")
    source = inputs["source"]

    st.subheader("Loaded input overview")
    horizon_years = int(inputs["cost_parameters"].get("planning_horizon_years", 15))
    horizon_peak_kw = (
        customers["peak_kw_now"].astype(float)
        * (1.0 + customers["growth_rate"].astype(float)) ** horizon_years
    ).sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{len(customers):,}")
    k2.metric("Types", customers["customer_type"].nunique())
    k3.metric("Peak demand now", f"{customers['peak_kw_now'].sum():,.1f} kW")
    k4.metric(f"Peak demand year {horizon_years}", f"{horizon_peak_kw:,.1f} kW")
    k5.metric("Road features", f"{len(roads):,}" if roads is not None else "not provided")

    try:
        st_folium(
            input_map(customers, source, roads),
            height=520,
            use_container_width=True,
            key="input_overview_map",
            returned_objects=[],
        )
    except Exception as exc:
        st.warning(f"Input map unavailable: {exc}")

    try:
        st_folium(
            peak_demand_map(customers, source, roads),
            height=520,
            use_container_width=True,
            key="input_peak_demand_map",
            returned_objects=[],
        )
    except Exception as exc:
        st.warning(f"Peak demand map unavailable: {exc}")

    with st.expander("Customer summary and preview", expanded=False):
        st.dataframe(customer_type_summary(customers), use_container_width=True, hide_index=True)
        st.dataframe(customers.head(100), use_container_width=True, hide_index=True)

    with st.expander("Source", expanded=False):
        st.dataframe(source_table(source), use_container_width=True, hide_index=True)

    with st.expander("Conductor catalogues", expanded=False):
        lv_col, mv_col = st.columns(2)
        lv_col.dataframe(inputs["lv_conductors"], use_container_width=True, hide_index=True)
        mv_col.dataframe(inputs["mv_conductors"], use_container_width=True, hide_index=True)

    with st.expander("Transformers and stand-alone packages", expanded=False):
        tx_col, sa_col = st.columns(2)
        tx_col.dataframe(inputs["transformers"], use_container_width=True, hide_index=True)
        sa_col.dataframe(inputs["standalone_packages"], use_container_width=True, hide_index=True)

    with st.expander("Cost and engineering parameters", expanded=False):
        st.dataframe(cost_parameters_table(inputs["cost_parameters"]), use_container_width=True, hide_index=True)

    with st.expander("Road input", expanded=False):
        st.dataframe(roads_summary_table(roads), use_container_width=True, hide_index=True)
        roads_preview = roads_preview_table(roads)
        if not roads_preview.empty:
            st.dataframe(roads_preview, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Section 2 — Engineering assumptions
# --------------------------------------------------------------------------
st.header("2 · Engineering & economic assumptions")
st.caption(
    "The inputs that drive the workflow, grouped by role — defaults load from the dataset; change any value "
    "to re-run under a different design or economic scenario."
)
cp = (inputs or {}).get("cost_parameters", {})
topo_col, econ_col, pf_col = st.columns(3)

with topo_col.container(border=True):
    st.markdown("**🕸️ Topology heuristics**")
    st.caption("Road-pole placement and customer attachment.")
    max_user_connection_radius_m = st.number_input("Max user→pole radius [m]", 5.0, 200.0, 40.0, 5.0)
    max_users_per_pole = st.number_input("Max users per pole", 1, 60, 12, 1)
    max_pole_span_m = st.number_input("Max LV pole span [m]", 10.0, 200.0, 45.0, 5.0)
    road_pole_spacing_m = st.number_input("Road pole spacing [m]", 10.0, 150.0, 35.0, 5.0)

with econ_col.container(border=True):
    st.markdown("**💶 Economics & time horizon**")
    st.caption("Discounting and levelised generation cost over the planning period.")
    planning_horizon_years = st.number_input("Planning horizon [years]", 1, 40, int(cp.get("planning_horizon_years", 15)))
    discount_rate = st.number_input("Discount rate", 0.0, 0.30, float(cp.get("discount_rate", 0.08)), 0.01, format="%.2f")
    standalone_sizing_tolerance = st.number_input(
        "SHS package sizing tolerance", 0.0, 0.25,
        float(cp.get("standalone_sizing_tolerance", 0.02)), 0.01, format="%.2f",
        help="Fractional headroom added to a package's peak/energy limits before testing "
             "eligibility. The catalogue is discrete and the horizon peak is compounded, so a zero "
             "tolerance is a knife edge: a 0.25 kW household growing 4%% over 15 years needs "
             "0.45024 kW against a 0.45 kW package and misses by 0.05%%, jumping to the next "
             "package up at +61%% NPC. With all-or-nothing subtree eligibility one such customer "
             "also vetoes a whole branch.",
    )
    generation_lcoe_per_kwh = st.number_input(
        "Generation LCOE [€/kWh]", 0.0, 2.0, float(cp.get("generation_lcoe_per_kwh", 0.25)), 0.01, format="%.2f",
        help="Levelised cost of the mini-grid electricity, applied to grid-served energy over the horizon so "
             "grid architectures are compared fairly against SHS (whose generation is already in the package NPC).",
    )

with pf_col.container(border=True):
    st.markdown("**⚡ Load-flow limits**")
    st.caption("Electrical limits enforced by the power flow, plus the design coincidence factor.")
    voltage_min_pu = st.number_input("Voltage min [pu]", 0.80, 1.0, float(cp.get("voltage_min_pu", 0.90)), 0.01, format="%.2f")
    voltage_max_pu = st.number_input("Voltage max [pu]", 1.0, 1.15, float(cp.get("voltage_max_pu", 1.05)), 0.01, format="%.2f")
    max_line_loading_percent = st.number_input("Max line loading [%]", 20.0, 120.0, float(cp.get("max_line_loading_percent", 100.0)), 5.0)
    max_transformer_loading_percent = st.number_input("Max transformer loading [%]", 20.0, 120.0, float(cp.get("max_transformer_loading_percent", 100.0)), 5.0)
    coincidence_factor = st.number_input(
        "Coincidence factor (design)", 0.1, 1.0, float(cp.get("coincidence_factor", 0.6)), 0.05, format="%.2f",
        help="Fraction of the summed individual customer peaks that coincides at the network peak; used to "
             "SIZE and validate the network. The coincident-peak stress test always uses 1.0 (everyone at once).",
    )

    st.markdown("**MV/LV transformer zoning (Case C)**")
    _crit_opts = ["voltage_cap", "distance_cap", "capacity"]
    _crit_default = str(cp.get("clustering_criterion", "voltage_cap"))
    clustering_criterion = st.selectbox(
        "Transformer-count criterion",
        _crit_opts,
        index=_crit_opts.index(_crit_default) if _crit_default in _crit_opts else 0,
        help="voltage_cap: iterate the transformer count and let the power flow pick the MINIMUM that holds "
             "the LV voltage drop (backtracked). distance_cap: complete-linkage clusters guaranteed ≤ max "
             "diameter. capacity: legacy greedy packing. All three respect the transformer kVA cap.",
    )
    max_cluster_diameter_m = st.number_input(
        "Max cluster diameter [m] (distance_cap)", 100.0, 3000.0,
        float(cp.get("max_cluster_diameter_m", 500.0)), 50.0,
    )
    max_transformers = st.number_input(
        "Max transformers (safety cap)", 1, 50, int(cp.get("max_transformers", 12)), 1,
    )
    source_vm_pu = st.number_input(
        "Sending voltage at plant slack [pu]", 0.95, 1.10,
        float(cp.get("source_vm_pu", 1.0)), 0.01, format="%.2f",
        help="Slack voltage on the plant LV bus; an explicit step-up transformer produces the MV drop "
             "(replaces the earlier fixed 1.04 pu slack placed directly on the MV bus).",
    )
    design_voltage_margin_pu = st.number_input(
        "Design voltage margin [pu]", 0.0, 0.08,
        float(cp.get("design_voltage_margin_pu", 0.02)), 0.005, format="%.3f",
        help="Planning reserve the zone-count search keeps ON TOP of the minimum voltage when it "
             "accepts a transformer count. Minimising the count against the bare limit leaves the "
             "design sitting on the constraint. It is a preference, not a constraint: if no zone "
             "count can reach it (the voltage plateaus once the MV trunk, not the LV zones, is the "
             "bottleneck) the search falls back to the smallest feasible count and says so. "
             "0 reproduces the old behaviour. Does not change what counts as feasible.",
    )
    _snap_opts = ["design", "coincident"]
    _snap_default = str(cp.get("k_selection_snapshot", "design"))
    k_selection_snapshot = st.selectbox(
        "Select zone count against", _snap_opts,
        index=_snap_opts.index(_snap_default) if _snap_default in _snap_opts else 0,
        help="Which snapshot the zone-count search must satisfy: 'design' (diversified, plus the "
             "margin above) or 'coincident' (every customer at peak at once) -- the conservative "
             "extreme, which needs many more transformers.",
    )
    transformer_sizing_coincidence_factor = st.number_input(
        "Transformer sizing coincidence", 0.0, 1.0,
        float(cp.get("transformer_sizing_coincidence_factor", 1.0)), 0.05, format="%.2f",
        help="Diversity basis for transformer sizing and the zone kVA cap. 0 tracks the design "
             "coincidence factor, so the rating and the loading it is checked against agree -- but "
             "the design factor is a flat SYSTEM-WIDE number and diversity only emerges with many "
             "customers, so it under-sizes small zones. The 1.0 default sizes on the coincident "
             "zone peak: deliberately conservative (loading then peaks near 50%, so its limit never "
             "binds) and it also buys LV voltage support, because a smaller unit has a higher "
             "per-unit impedance. A size-dependent ADMD curve is the real answer.",
    )

eng = {
    # Topology heuristics
    "max_user_connection_radius_m": max_user_connection_radius_m,
    "max_users_per_pole": max_users_per_pole,
    "max_pole_span_m": max_pole_span_m,
    "road_pole_spacing_m": road_pole_spacing_m,
    # Economics & time horizon
    "planning_horizon_years": planning_horizon_years,
    "discount_rate": discount_rate,
    "generation_lcoe_per_kwh": generation_lcoe_per_kwh,
    "standalone_sizing_tolerance": standalone_sizing_tolerance,
    # Load-flow limits
    "voltage_min_pu": voltage_min_pu,
    "voltage_max_pu": voltage_max_pu,
    "max_line_loading_percent": max_line_loading_percent,
    "max_transformer_loading_percent": max_transformer_loading_percent,
    "coincidence_factor": coincidence_factor,
    # MV/LV transformer zoning (Case C)
    "clustering_criterion": clustering_criterion,
    "max_cluster_diameter_m": max_cluster_diameter_m,
    "max_transformers": max_transformers,
    "source_vm_pu": source_vm_pu,
    "design_voltage_margin_pu": design_voltage_margin_pu,
    "k_selection_snapshot": k_selection_snapshot,
    "transformer_sizing_coincidence_factor": transformer_sizing_coincidence_factor,
}

# --------------------------------------------------------------------------
# Section 3 — Run
# --------------------------------------------------------------------------
st.header("3 · Run workflow")
if st.button("Run simplified LV/MV + stand-alone MVP", type="primary", disabled=inputs is None):
    status = st.status("Running pipeline...", expanded=True)
    try:
        result = run_simplified_mvp(
            inputs["customers"], inputs["roads"], inputs["source"], inputs["lv_conductors"],
            inputs["mv_conductors"], inputs["transformers"], inputs["standalone_packages"],
            inputs["cost_parameters"], eng, progress=lambda m: status.write(m),
        )
        st.session_state["mvp_result"] = result
        st.session_state["mvp_exports"] = build_exports(result, inputs["customers"], inputs["source"])
        status.update(label="Done.", state="complete")
    except Exception as exc:
        status.update(label="Failed.", state="error")
        st.exception(exc)

# --------------------------------------------------------------------------
# Section 4 — Results
# --------------------------------------------------------------------------
result = st.session_state.get("mvp_result")
if result is not None:
    st.header("4 · Results dashboard")
    rec = result.cases[result.recommended]
    a, b, c = st.columns(3)
    a.metric("Recommended architecture", rec.architecture)
    b.metric("NPC (recommended)", f"{rec.cost.total_npc:,.0f}")
    c.metric("Feasible candidates", int(result.comparison_table["feasible"].sum()))
    st.success(result.recommendation_note)

    st.subheader("Candidate comparison (A · B · C · D)")
    st.dataframe(highlight_recommended(result.comparison_table, result.recommended), use_container_width=True)
    feasible_ids = [k for k, case in result.cases.items() if case.feasible]
    if feasible_ids:
        st.caption("Net present cost by component — feasible candidates only, recommended starred.")
        st.altair_chart(
            npc_breakdown_chart(result.cost_breakdown_table, result.recommended, CANDIDATES, feasible_ids),
            use_container_width=True,
        )

    # Enforced electrical limits, reused by the headroom bars and the stress map below.
    constraints = result.inputs.get("params", (inputs or {}).get("cost_parameters", {}))

    # ----- Per-candidate detail: every option visualized, not just the optimal -----
    st.subheader("Per-candidate detail")
    st.caption(LEGEND)
    horizon_years = int(result.inputs.get("horizon_years", (inputs or {}).get("cost_parameters", {}).get("planning_horizon_years", 15)))
    coincidence = float(result.inputs.get("params", {}).get("coincidence_factor", 0.6))
    stress_options = {
        "Now": "now",
        f"Year {horizon_years} (design)": "horizon",
        f"Year {horizon_years} coincident peak": "stress",
    }
    selected_stress_label = st.select_slider(
        "Electrical stress map scenario",
        options=list(stress_options),
        value=f"Year {horizon_years} (design)",
        help=(
            f"Now / design use the diversified peak (coincidence {coincidence:.2f}); the coincident-peak "
            "snapshot puts every customer at their year-horizon peak simultaneously (factor 1.0)."
        ),
    )
    selected_stress = stress_options[selected_stress_label]
    tabs = st.tabs([f"{'⭐ ' if k == result.recommended else ''}{lbl}" for k, lbl in CANDIDATES.items()])
    for tab, key in zip(tabs, CANDIDATES):
        case = result.cases[key]
        with tab:
            badge = "✅ feasible" if case.feasible else "❌ infeasible"
            rec_tag = " · **recommended**" if key == result.recommended else ""
            st.markdown(f"**{case.architecture}** — {badge}{rec_tag}  \n*{case.decision_summary}*")

            oh = case.outcome_horizon
            loss_txt = f"{oh.losses_kw:,.1f} kW" if (oh is not None and oh.losses_kw == oh.losses_kw) else "—"

            m = st.columns(4)
            m[0].metric("Total NPC", f"{case.cost.total_npc:,.0f}")
            m[1].metric("Grid / stand-alone", f"{case.n_grid_customers} / {case.n_standalone_customers}")
            m[2].metric("Transformers", case.n_transformers)
            m[3].metric("Losses (horizon)", loss_txt)

            m2 = st.columns(4)
            m2[0].metric("Poles", f"{case.n_poles:,}")
            m2[1].metric("Service-drop length [m]", f"{case.service_drop_total_m:,.0f}")
            m2[2].metric("LV branch length [m]", f"{case.lv_length_m:,.0f}")
            m2[3].metric("MV length [m]", f"{case.mv_length_m:,.0f}")

            if oh is not None:
                st.markdown(f"**Headroom at year {horizon_years}** — diversified design peak (coincidence {coincidence:.2f})")
                st.markdown(case_headroom_html(oh, constraints), unsafe_allow_html=True)
                if not case.feasible:
                    st.warning(f"Active bottleneck: {oh.active_bottleneck}")

            os_ = case.outcome_stress
            if os_ is not None:
                st.markdown(f"**Coincident-peak stress** — all customers at their year-{horizon_years} peak at once (factor 1.0)")
                st.markdown(case_headroom_html(os_, constraints), unsafe_allow_html=True)
                if not os_.converged:
                    st.error("⚡ Coincident peak: power flow did not converge (network cannot supply everyone at once).")
                elif not os_.feasible:
                    st.warning(f"⚡ Coincident-peak limits exceeded: {os_.active_bottleneck} (informational — does not gate feasibility).")
                else:
                    st.caption("✓ Survives the coincident peak within all limits.")
            if case.must_connect_violations:
                st.error(f"⛔ {case.must_connect_violations} must-connect customer(s) left off-grid → candidate infeasible")
            if case.service_tier_violations:
                st.warning(f"⚠ {case.service_tier_violations} customer(s) receive no service (grid or stand-alone)")

            comp = npc_components(result.cost_breakdown_table, key)
            if not comp.empty:
                st.caption("NPC composition")
                pcol, vcol = st.columns([2, 1])
                share = npc_share_chart(comp)
                if share is not None:
                    pcol.altair_chart(share, use_container_width=True)
                vcol.dataframe(comp, use_container_width=True, hide_index=True)

            try:
                st_folium(
                    case_map(
                        case,
                        inputs["customers"],
                        inputs["source"],
                        stress_scenario=selected_stress,
                        scenario_label=selected_stress_label,
                        constraints=constraints,
                    ),
                    height=520,
                    use_container_width=True,
                    key=f"map_{key}_{selected_stress}",
                    returned_objects=[],
                )
            except Exception as exc:
                st.warning(f"Map unavailable: {exc}")

    # ----- SHS screening: the economic leaf-branch decision behind Cases B and D -----
    st.subheader("SHS screening — economics per leaf branch")
    st.caption(
        "Each candidate terminal subtree (leaf branch) stays on the grid unless its stand-alone NPC "
        "is below the grid cost it would avoid. Bars show the margin (avoided − SHS NPC): green = "
        "stand-alone selected, grey = kept on grid; branches blocked before costing (must-connect or "
        "no feasible package) are listed in the table only."
    )
    sb, sd = st.tabs(["B · LV leaf branches", "D · MV/LV leaf branches"])
    for stab, cand in ((sb, "B_LV_plus_standalone"), (sd, "D_MVLV_plus_standalone")):
        with stab:
            screen = shs_screening_table(result.decision_log, cand)
            if screen.empty:
                st.info("No leaf-branch candidates were screened for this case.")
                continue
            n_sel = int((screen["outcome"] == "selected").sum())
            n_blocked = int((screen["outcome"] == "blocked").sum())
            st.caption(f"{len(screen)} candidate branches · {n_sel} selected for stand-alone · {n_blocked} blocked before costing.")
            chart = shs_decision_chart(screen)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            st.dataframe(screen, use_container_width=True, hide_index=True)

    # ----- Coincident-peak stress test: robustness when everyone peaks at once -----
    st.subheader("Coincident-peak stress test — all users at their peak simultaneously")
    st.caption(
        f"Feasibility is decided at the **diversified design peak** (coincidence {coincidence:.2f}). This table "
        f"re-solves each candidate with **every customer at their year-{horizon_years} peak at once** (factor 1.0). "
        "It is a robustness check and does **not** change the recommendation."
    )
    comp = result.comparison_table
    stress_cols = [
        "candidate_id", "architecture", "coincident_within_limits", "coincident_min_voltage",
        "coincident_max_line_loading", "coincident_max_transformer_loading", "coincident_bottleneck",
    ]
    st.dataframe(
        comp[[c for c in stress_cols if c in comp.columns]],
        use_container_width=True, hide_index=True,
    )

    # ----- Tier served & future expansion headroom -----
    st.subheader("Tier served & future expansion headroom")
    st.caption(
        "Grid customers can evolve to higher tiers as demand grows; **stand-alone customers are capped at their "
        "SHS package's service tier** — a hard ceiling for future demand evolution and network expansion. "
        "Green = grid (expandable), red = stand-alone (tier-capped)."
    )
    tier_tbl = result.tier_service_table
    ttabs = st.tabs([f"{'⭐ ' if k == result.recommended else ''}{lbl}" for k, lbl in CANDIDATES.items()])
    for ttab, key in zip(ttabs, CANDIDATES):
        with ttab:
            sub = tier_tbl[tier_tbl["candidate_id"] == key]
            tm = st.columns(3)
            tm[0].metric("Grid (expandable)", int((sub["status"] == "grid").sum()))
            tm[1].metric("Stand-alone (tier-capped)", int((sub["status"] == "standalone").sum()))
            tm[2].metric("Capped at current tier (no headroom)", int(sub["capped"].sum()))
            tchart = tier_service_chart(tier_tbl, key)
            if tchart is not None:
                st.altair_chart(tchart, use_container_width=True)

    # ----- Parallel coordinates: the feasible solutions on x, one line per key metric -----
    st.subheader("Solution trade-offs — parallel coordinates")
    st.caption(
        "**Feasible** solutions only (an unbuildable option can't win a trade-off), so the recommended ⭐ tops "
        "the Total NPC line. Each **metric is a line** across the solutions, min-max normalized *across the shown "
        "solutions* and oriented so **up = better** — e.g. the cheapest solution sits at the **top** of the Total "
        "NPC line, not the bottom. Read a solution vertically to see how it scores on every metric; follow a line "
        "to see which solution wins it. Metrics: **total NPC** (cheapest wins), **total tier capacity** (Σ future "
        "service-tier ceiling — grid connections stay expandable, SHS are capped), **coincident-peak min voltage** "
        "(electrical stability when all users peak at once; a non-converged case scores 0). Raw values are in the tooltip."
    )
    feasible_ids_pcp = [k for k in CANDIDATES if k in set(result.comparison_table.loc[result.comparison_table["feasible"], "candidate_id"])]
    if len(feasible_ids_pcp) < 2:
        only = feasible_ids_pcp[0] if feasible_ids_pcp else None
        st.info(
            f"Only one feasible solution ({CANDIDATES.get(only, only)}) — nothing to compare across solutions."
            if only else "No feasible solution to compare."
        )
    else:
        MAX_TIER = 5  # full MTF ceiling a grid connection can grow into
        tt = tier_tbl.copy()
        tt["tier_capacity"] = 0.0
        tt.loc[tt["status"] == "grid", "tier_capacity"] = MAX_TIER
        tt.loc[tt["status"] == "standalone", "tier_capacity"] = tt["service_ceiling_tier"].fillna(0.0)
        capacity_by = tt.groupby("candidate_id")["tier_capacity"].sum()

        comp = result.comparison_table
        pcp_df = comp[comp["candidate_id"].isin(feasible_ids_pcp)][
            ["candidate_id", "npc_total", "coincident_min_voltage"]
        ].copy()
        pcp_df["tier_capacity"] = pcp_df["candidate_id"].map(capacity_by).fillna(0.0)
        pcp_df["coincident_min_voltage"] = pcp_df["coincident_min_voltage"].fillna(0.0)
        pcp_df["candidate"] = pcp_df["candidate_id"].map(
            lambda k: ("⭐ " if k == result.recommended else "") + CANDIDATES.get(k, k)
        )
        order = [("⭐ " if k == result.recommended else "") + CANDIDATES[k] for k in feasible_ids_pcp]
        metrics = [
            {"col": "npc_total", "label": "Total NPC", "higher_is_better": False},
            {"col": "tier_capacity", "label": "Total tier capacity", "higher_is_better": True},
            {"col": "coincident_min_voltage", "label": "Coincident-peak min voltage", "higher_is_better": True},
        ]
        pcp = parallel_coordinates_chart(pcp_df, metrics, candidate_col="candidate", candidate_order=order)
        if pcp is not None:
            st.altair_chart(pcp, use_container_width=True)

    st.subheader("Customer strategy (recommended architecture)")
    st.dataframe(result.customer_strategy_table, use_container_width=True, height=300)
    st.subheader("Decision log")
    st.dataframe(result.decision_log, use_container_width=True, height=260)

    # --------------------------------------------------------------------------
    # Section 5 — Downloads
    # --------------------------------------------------------------------------
    st.header("5 · Downloads")
    exports = st.session_state.get("mvp_exports", {})
    dl = st.columns(3)
    for i, (name, data) in enumerate(exports.items()):
        mime = "application/zip" if name.endswith(".zip") else ("application/geo+json" if name.endswith(".geojson") else "text/csv")
        dl[i % 3].download_button(name, data=data, file_name=name, mime=mime, key=f"dl_{name}")

st.divider()
st.caption(
    "**Limitations:** balanced three-phase only; fixed source; heuristic road-pole + MST topology; "
    "branch/cluster-level (not per-customer) stand-alone decisions; heuristic capacity-based transformer "
    "zoning and placement; a single flat coincidence factor for the design peak (no size-dependent ADMD); "
    "generation costed with one flat mini-grid LCOE (no hourly dispatch); illustrative cost values."
)
