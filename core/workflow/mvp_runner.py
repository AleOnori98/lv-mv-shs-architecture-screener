"""End-to-end simplified MVP: Full LV, hybrids and MV/LV, compared by NPC.

This is a sequential diagnostic workflow, not a co-optimization model. Each step is
deterministic and logged to ``decision_log`` for traceability.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.io.load_inputs import customers_to_engine_gdf
from core.topology.lv_topology import build_lv_topology
from core.topology.branch_detection import Cluster, detect_lv_subtrees, subtree_clusters
from core.topology.mvlv_service_zones import (
    _coords_and_ids,
    build_mvlv_plan,
    build_stepup_spec,
    partition_zones,
    suggest_zone_start,
    zone_apparent_power_kva,
)
from core.electrical.pandapower_builder import build_lv_net, build_mvlv_net
from core.electrical.conductor_sizing import size_conductors, critical_lv_lines
from core.electrical.validation import run_powerflow, violation_severity
from core.economics import cost_model as cm
from core.economics.standalone import cluster_standalone
from .result_models import CaseResult, MVPResult


def _log(log: List[Dict[str, Any]], step, candidate, obj, decision, reason, metric="", value="", threshold="") -> None:
    log.append(
        {
            "step": step,
            "candidate_id": candidate,
            "object_id": obj,
            "decision": decision,
            "reason": reason,
            "metric_name": metric,
            "metric_value": value,
            "threshold": threshold,
        }
    )


def _length_by_line(topology, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if topology is None or topology.edges_4326 is None or topology.edges_4326.empty:
        return out
    for _, e in topology.edges_4326.iterrows():
        out[f"{prefix}{e['line_id']}"] = float(e["length_m"])
    return out


def _size_flat_lv(topology, customers_df, lv_conductors, constraints, T, coincidence=1.0):
    line_ids = list(_length_by_line(topology).keys())
    build_fn = lambda cond: build_lv_net(topology, customers_df, cond, lv_kv=topology.metadata.get("lv_kv", 0.4), horizon_years=T, apply_growth=True, coincidence_factor=coincidence)
    critical_fn = lambda model, outcome: critical_lv_lines(model, outcome, topology)
    return size_conductors(line_ids, lv_conductors, build_fn, constraints, critical_fn=critical_fn)


def _now_powerflow(topology, customers_df, conductor_by_line, constraints, T, lv_kv, coincidence=1.0):
    model = build_lv_net(topology, customers_df, conductor_by_line, lv_kv=lv_kv, horizon_years=T, apply_growth=False, coincidence_factor=coincidence)
    return model, run_powerflow(model, constraints)


def _stress_lv_powerflow(topology, customers_df, conductor_by_line, constraints, T, lv_kv):
    """Coincident-peak (factor 1.0) stress: every customer at their horizon peak simultaneously,
    on the network already sized for the diversified design peak."""
    model = build_lv_net(topology, customers_df, conductor_by_line, lv_kv=lv_kv, horizon_years=T, apply_growth=True, coincidence_factor=1.0)
    return model, run_powerflow(model, constraints)


def _generation_npc(ctx, grid_ids) -> float:
    """PV of mini-grid generation cost for the given grid-served customers (see cost_model)."""
    kwh_by_id = ctx["annual_kwh_by_id"]
    total_kwh = sum(kwh_by_id.get(str(c), 0.0) for c in grid_ids)
    return cm.generation_present_value(total_kwh, ctx["params"])


# ---------------------------------------------------------------------------
# Case A — Full LV
# ---------------------------------------------------------------------------

def _run_full_lv(ctx) -> CaseResult:
    cid = "A_full_LV"
    topo = build_lv_topology(ctx["gdf"], ctx["roads"], ctx["crs_meta"], ctx["source"], ctx["eng"])
    topo.metadata["lv_kv"] = ctx["lv_kv"]
    _log(ctx["log"], "topology", cid, "full_lv", "generate", "all customers connected (allow_unserved=False)", "n_served", len(topo.served_customer_ids))

    sizing = _size_flat_lv(topo, ctx["customers"], ctx["lv_conductors"], ctx["constraints"], ctx["T"], ctx["coincidence"])
    topo.bus_by_pole = sizing.model.bus_by_pole
    _log(ctx["log"], "conductor_sizing", cid, "lv_network", "pass" if sizing.passed else "fail",
         sizing.outcome.active_bottleneck, "iterations", sizing.iterations)

    now_model, now = _now_powerflow(topo, ctx["customers"], sizing.conductor_by_line, ctx["constraints"], ctx["T"], ctx["lv_kv"], ctx["coincidence"])
    stress_model, stress = _stress_lv_powerflow(topo, ctx["customers"], sizing.conductor_by_line, ctx["constraints"], ctx["T"], ctx["lv_kv"])

    cost = cm.CostBreakdown()
    cost.lv_line_cost = cm.line_cost(sizing.conductor_by_line, _length_by_line(topo), ctx["params"]["civil_logistics_multiplier"])
    cost.pole_cost = cm.pole_capex(len(topo.pole_xy), ctx["params"]["pole_cost"], ctx["params"]["civil_logistics_multiplier"])
    cost.connection_metering_cost = cm.connection_capex(len(topo.served_customer_ids), ctx["params"])
    cost.loss_pv_cost = cm.loss_present_value(sizing.outcome.losses_kw, ctx["params"])
    cost.generation_npc = _generation_npc(ctx, topo.served_customer_ids)

    assign = {c: {"status": "grid_now", "architecture": "Full LV", "cluster_id": "", "transformer_id": "", "package_id": "", "reason": "connected in Full LV"} for c in topo.served_customer_ids}

    res = CaseResult(
        candidate_id=cid, architecture="Full LV", feasible=sizing.passed, cost=cost,
        outcome_now=now, outcome_horizon=sizing.outcome, outcome_stress=stress,
        n_grid_customers=len(topo.served_customer_ids), n_standalone_customers=0,
        lv_length_m=topo.total_lv_length_m, mv_length_m=0.0,
        n_poles=len(topo.pole_xy),
        service_drop_total_m=sum(topo.service_drop_m.get(c, 0.0) for c in topo.served_customer_ids),
        n_transformers=0,
        sizing_iterations=sizing.iterations, topology=topo, model_now=now_model, model=sizing.model,
        model_stress=stress_model,
        conductor_by_line=sizing.conductor_by_line, grid_customer_ids=list(topo.served_customer_ids),
        customer_assignment=assign,
        decision_summary=("Feasible single LV network" if sizing.passed else f"Infeasible: {sizing.outcome.active_bottleneck}"),
    )
    ctx["full_lv"] = {"topology": topo, "sizing": sizing}
    return res


# ---------------------------------------------------------------------------
# Case B — LV + Stand-alone
# ---------------------------------------------------------------------------

def _avoided_grid_cost(cluster: Cluster, ctx, *, include_losses: bool) -> float:
    """Grid cost avoided if a candidate subtree's customers all move to stand-alone.

    All terms are *marginal* to that subtree:
      branch line cost (internal lines + entry edge, already priced on ``cluster``)
      + service-drop reconnection proxy (drop metres priced with the cheapest LV conductor)
      + avoided connection/metering for the subtree's customers
      + avoided poles dedicated to the subtree (the boundary pole stays, hence is excluded)
      + present value of avoided grid GENERATION for the subtree's customers (they no longer draw
        mini-grid energy once off-grid) — credited in BOTH screens so the gate matches the true
        total-NPC break-even (generation dominates NPC and was previously omitted)
      [+ present value of avoided branch I^2R losses — LV screen only].
    """
    params = ctx["params"]
    mult = float(params["civil_logistics_multiplier"])
    cheapest_lv = ctx["lv_conductors"].sort_values("rank").iloc[0].to_dict()
    service_drop_cost = float(cluster.metadata.get("service_drop_m", 0.0)) * float(cheapest_lv["cost_per_m"]) * mult
    conn = cm.connection_capex(cluster.n_customers, params)
    pole_cost = cm.pole_capex(len(cluster.pole_ids), params["pole_cost"], mult)
    generation_pv = _generation_npc(ctx, cluster.customer_ids)
    avoided = cluster.branch_cost + service_drop_cost + conn + pole_cost + generation_pv
    if include_losses:
        avoided += cm.loss_present_value(getattr(cluster, "branch_losses_kw", 0.0), params)
    return float(avoided)


def _avoided_lv_cost(cluster: Cluster, ctx) -> float:
    """Case B: avoided grid generation PV + avoided branch-loss PV (sliced from the solved
    Full-LV power flow)."""
    return _avoided_grid_cost(cluster, ctx, include_losses=True)


def _avoided_mvlv_subtree_cost(cluster: Cluster, ctx) -> float:
    """Case D: avoided grid generation PV is credited (as in Case B). Still conservative on the
    electrical side — branch losses are not sliced per subtree, and the zone transformer / MV
    service drop are credited only via the rebuild (an evacuated zone is simply not re-created)."""
    return _avoided_grid_cost(cluster, ctx, include_losses=False)


def _select_standalone_clusters(ctx, clusters, avoided_cost_fn, architecture_label, cid):
    """Screen candidate subtrees and pick stand-alone where SHS NPC < avoided grid cost.

    Shared by the LV screen (Case B) and the MV/LV screen (Case D). Each candidate exposes
    ``cluster_id``, ``customer_ids`` and ``standalone_eligible_precheck``;
    ``avoided_cost_fn(candidate, ctx)`` is the marginal grid cost avoided if that subtree goes
    stand-alone. The decision is **purely economic** — engineering thresholds are diagnostic
    only and never gate here.

    Terminal subtrees nest (a subtree contains its own sub-branches). Candidates are evaluated
    coarsest-first (descending avoided cost) and a customer already claimed by a selected
    subtree blocks any overlapping candidate, so the best-paying grouping wins and no customer
    is double-counted (subtrees form a laminar family, so overlaps are always nested).

    Returns ``(selected_customer_ids, standalone_assignments, standalone_npc_total)``.
    """
    T, rate = ctx["T"], ctx["discount"]
    selected: List[str] = []
    claimed: set = set()
    assignments: Dict[str, Dict[str, Any]] = {}
    npc_total = 0.0
    avoided_by = {id(cl): avoided_cost_fn(cl, ctx) for cl in clusters}
    for cl in sorted(clusters, key=lambda c: avoided_by[id(c)], reverse=True):
        if not cl.standalone_eligible_precheck:
            continue
        if any(c in claimed for c in cl.customer_ids):  # a coarser subtree already took these
            continue
        sa = cluster_standalone(cl.customer_ids, ctx["customers"], ctx["packages"], horizon_years=T, discount_rate=rate)
        if not sa["eligible"]:
            _log(ctx["log"], "standalone", cid, cl.cluster_id, "keep_grid", "; ".join(sa["blocking_reasons"]) or "not eligible")
            continue
        avoided = avoided_by[id(cl)]
        decision = sa["total_npc"] < avoided
        _log(ctx["log"], "standalone", cid, cl.cluster_id,
             "standalone" if decision else "keep_grid",
             "NPC_SA < avoided grid cost" if decision else "grid cheaper",
             "npc_sa_vs_avoided", f"{sa['total_npc']:.0f} vs {avoided:.0f}")
        if decision:
            selected.extend(cl.customer_ids)
            claimed.update(cl.customer_ids)
            for ch in sa["choices"]:
                assignments[ch.customer_id] = {
                    "status": "standalone", "architecture": architecture_label, "cluster_id": cl.cluster_id,
                    "transformer_id": "", "package_id": ch.package_id, "reason": "stand-alone NPC < avoided grid cost",
                }
            npc_total += sa["total_npc"]
    return selected, assignments, npc_total


def _run_lv_standalone(ctx, full_res: CaseResult, sa_customers, sa_assignments, sa_npc_total) -> CaseResult:
    cid = "B_LV_plus_standalone"
    T = ctx["T"]
    selected_sa_customers = list(sa_customers)
    grid_ids = [c for c in full_res.grid_customer_ids if c not in set(selected_sa_customers)]

    if not selected_sa_customers:
        # No stand-alone selected: identical to Full LV (still a valid candidate).
        res = CaseResult(
            candidate_id=cid, architecture="LV + Stand-alone", feasible=full_res.feasible, cost=full_res.cost,
            outcome_now=full_res.outcome_now, outcome_horizon=full_res.outcome_horizon, outcome_stress=full_res.outcome_stress,
            n_grid_customers=full_res.n_grid_customers, n_standalone_customers=0,
            lv_length_m=full_res.lv_length_m,
            n_poles=full_res.n_poles, service_drop_total_m=full_res.service_drop_total_m,
            sizing_iterations=full_res.sizing_iterations, topology=full_res.topology,
            model_now=full_res.model_now, model=full_res.model, model_stress=full_res.model_stress,
            conductor_by_line=full_res.conductor_by_line, grid_customer_ids=list(grid_ids),
            customer_assignment=dict(full_res.customer_assignment),
            decision_summary="No cluster cheaper as stand-alone; equals Full LV",
        )
        return res

    topo = build_lv_topology(ctx["gdf"], ctx["roads"], ctx["crs_meta"], ctx["source"], ctx["eng"], customer_ids=grid_ids)
    topo.metadata["lv_kv"] = ctx["lv_kv"]
    sizing = _size_flat_lv(topo, ctx["customers"], ctx["lv_conductors"], ctx["constraints"], T, ctx["coincidence"])
    topo.bus_by_pole = sizing.model.bus_by_pole
    now_model, now = _now_powerflow(topo, ctx["customers"], sizing.conductor_by_line, ctx["constraints"], T, ctx["lv_kv"], ctx["coincidence"])
    stress_model, stress = _stress_lv_powerflow(topo, ctx["customers"], sizing.conductor_by_line, ctx["constraints"], T, ctx["lv_kv"])

    cost = cm.CostBreakdown()
    cost.lv_line_cost = cm.line_cost(sizing.conductor_by_line, _length_by_line(topo), ctx["params"]["civil_logistics_multiplier"])
    cost.pole_cost = cm.pole_capex(len(topo.pole_xy), ctx["params"]["pole_cost"], ctx["params"]["civil_logistics_multiplier"])
    cost.connection_metering_cost = cm.connection_capex(len(topo.served_customer_ids), ctx["params"])
    cost.loss_pv_cost = cm.loss_present_value(sizing.outcome.losses_kw, ctx["params"])
    cost.generation_npc = _generation_npc(ctx, topo.served_customer_ids)
    cost.standalone_npc = sa_npc_total

    assign = {c: {"status": "grid_now", "architecture": "LV + Stand-alone", "cluster_id": "", "transformer_id": "", "package_id": "", "reason": "connected to LV grid"} for c in topo.served_customer_ids}
    assign.update(sa_assignments)

    return CaseResult(
        candidate_id=cid, architecture="LV + Stand-alone", feasible=sizing.passed, cost=cost,
        outcome_now=now, outcome_horizon=sizing.outcome, outcome_stress=stress,
        n_grid_customers=len(topo.served_customer_ids), n_standalone_customers=len(selected_sa_customers),
        lv_length_m=topo.total_lv_length_m,
        n_poles=len(topo.pole_xy),
        service_drop_total_m=sum(topo.service_drop_m.get(c, 0.0) for c in topo.served_customer_ids),
        sizing_iterations=sizing.iterations, topology=topo, model_now=now_model, model=sizing.model,
        model_stress=stress_model,
        conductor_by_line=sizing.conductor_by_line, grid_customer_ids=list(topo.served_customer_ids),
        standalone_customer_ids=list(selected_sa_customers), standalone_assignments=sa_assignments,
        customer_assignment=assign,
        decision_summary=f"{len(selected_sa_customers)} customers moved to stand-alone; LV grid resized",
    )


# ---------------------------------------------------------------------------
# Case C — MV/LV service zones (criterion-driven transformer count)
# ---------------------------------------------------------------------------

def _iter_record(n_zones: int, outcome, note: str) -> Dict[str, Any]:
    """One row of the transformer-count search history (for logs / UI)."""
    return {
        "n_zones": int(n_zones),
        "feasible": bool(outcome.feasible),
        "converged": bool(outcome.converged),
        "min_vm_pu": float(outcome.min_vm_pu),
        "worst_dv_pu": (float(1.0 - outcome.min_vm_pu) if outcome.converged else float("nan")),
        "bottleneck": outcome.active_bottleneck,
        "note": note,
    }


def _build_mvlv_at(ctx, groups: List[List[str]], stepup_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build plan + per-zone LV sizing + MV-rank sweep + horizon PF for a FIXED partition.

    The transformer count is whatever ``groups`` contains; the caller decides how ``groups``
    was formed (capacity packing, distance-cap linkage, or a voltage-cap k-means attempt).
    """
    T, constraints = ctx["T"], ctx["constraints"]
    plan = build_mvlv_plan(
        groups, ctx["gdf"], ctx["roads"], ctx["crs_meta"], ctx["source"], ctx["eng"],
        ctx["customers"], ctx["transformers"], horizon_years=T,
    )
    zone_cond: Dict[str, Any] = {}
    transformers_chosen: List[Dict[str, Any]] = []
    for zone in plan.zones:
        ztopo = zone["topology"]
        ztopo.metadata["lv_kv"] = ctx["lv_kv"]
        zsizing = _size_flat_lv(ztopo, ctx["customers"], ctx["lv_conductors"], constraints, T, ctx["coincidence"])
        ztopo.bus_by_pole = zsizing.model.bus_by_pole
        for lid, c in zsizing.conductor_by_line.items():
            zone_cond[f"{zone['zone_id']}::{lid}"] = c
        transformers_chosen.append({**zone["transformer_spec"], "zone_id": zone["zone_id"], "transformer_id": zone["transformer_id"]})

    transformer_by_zone = {z["zone_id"]: z["transformer_spec"] for z in plan.zones}
    final_model = final_outcome = mv_cond = None
    for rank_i in range(len(ctx["mv_conductors"])):
        mv_cond = ctx["mv_conductors"].iloc[rank_i].to_dict()
        final_model = build_mvlv_net(
            plan, ctx["customers"], core_conductor_by_line={}, zone_conductor_by_line=zone_cond,
            mv_conductor=mv_cond, transformer_by_zone=transformer_by_zone,
            lv_kv=ctx["lv_kv"], mv_kv=ctx["mv_kv"], horizon_years=T, apply_growth=True,
            stepup_transformer=stepup_spec, coincidence_factor=ctx["coincidence"],
            source_vm_pu=ctx["source_vm_pu"],
        )
        final_outcome = run_powerflow(final_model, constraints)
        if final_outcome.feasible:
            break

    return {"plan": plan, "zone_cond": zone_cond, "transformers": transformers_chosen,
            "transformer_by_zone": transformer_by_zone, "mv_cond": mv_cond,
            "model": final_model, "outcome": final_outcome}


def _voltage_cap_search(ctx, grid_ids, crs_epsg, max_kva, max_transformers, stepup_spec, seed):
    """Iterate the number of zones and let the power flow pick the MINIMUM that holds the cap.

    ``run_powerflow`` feasibility already encodes the LV voltage cap (v_min >= voltage_min_pu,
    i.e. dV <= 1 - voltage_min_pu) plus line/transformer loading, and treats Newton-Raphson
    divergence as infeasible. Seed at a physically-motivated zone count, then backtrack to the
    minimum feasible count (or climb to max_transformers if the seed is not enough).
    """
    T = ctx["T"]
    coords, _ = _coords_and_ids(grid_ids, ctx["gdf"])
    iterations: List[Dict[str, Any]] = []

    def attempt(n_zones: int) -> Dict[str, Any]:
        groups = partition_zones(
            "voltage_cap", grid_ids, ctx["gdf"], ctx["source"], crs_epsg,
            customers_df=ctx["customers"], horizon_years=T, max_zone_kva=max_kva,
            max_zone_radius_m=float(ctx["params"]["max_lv_zone_radius_m"]),
            max_zone_customers=int(ctx["params"]["max_zone_customers"]),
            n_zones=int(n_zones), seed=seed,
        )
        return _build_mvlv_at(ctx, groups, stepup_spec)

    k_seed = max(1, min(int(suggest_zone_start(coords, float(ctx["params"]["seed_diameter_m"]))), max_transformers))
    best = attempt(k_seed)
    iterations.append(_iter_record(len(best["plan"].zones), best["outcome"], "seed"))

    if best["outcome"].feasible:
        # Backtrack toward fewer zones; accept only strictly smaller feasible partitions.
        target = len(best["plan"].zones) - 1
        while target >= 1:
            cand = attempt(target)
            n = len(cand["plan"].zones)
            iterations.append(_iter_record(n, cand["outcome"], "backtrack"))
            if cand["outcome"].feasible and n < len(best["plan"].zones):
                best = cand
                target = n - 1
            else:
                break
    else:
        # Climb toward more zones until feasible or the safety cap is hit.
        target = len(best["plan"].zones) + 1
        while target <= max_transformers:
            cand = attempt(target)
            n = len(cand["plan"].zones)
            iterations.append(_iter_record(n, cand["outcome"], "climb"))
            best = cand
            if cand["outcome"].feasible:
                break
            target = n + 1
    return best, iterations


def _build_mvlv(ctx, grid_ids: List[str]) -> Dict[str, Any]:
    """Choose the transformer count by the active criterion, then size + validate the network.

    Returns the plan, sized conductors, chosen step-down transformers, the plant step-up spec,
    the design/stress PF outcomes and the transformer-count search history.
    """
    T = ctx["T"]
    crs_epsg = int(ctx["crs_meta"].get("projected_crs_epsg") or 32633)
    # Cap zone load at the largest transformer's derated rating over the sizing safety factor,
    # so even the biggest zone leaves headroom (avoids transformer overload at horizon).
    max_derated = float((ctx["transformers"]["sn_kva"] * ctx["transformers"]["max_loading_percent"] / 100.0).max())
    max_kva = max_derated / 1.2
    criterion = str(ctx["params"].get("clustering_criterion", "voltage_cap"))
    max_transformers = int(ctx["params"].get("max_transformers", 12))
    seed = 42

    # Plant step-up sized on the whole settlement's coincident apparent power (may exceed the
    # catalogue; it is a single project-procured unit, unlike the kVA-capped zone step-downs).
    s_system_kva = zone_apparent_power_kva(list(grid_ids), ctx["customers"], T)
    stepup_spec = build_stepup_spec(s_system_kva, ctx["transformers"], mv_kv=ctx["mv_kv"], lv_kv=ctx["lv_kv"])

    if criterion == "voltage_cap":
        best, iterations = _voltage_cap_search(ctx, grid_ids, crs_epsg, max_kva, max_transformers, stepup_spec, seed)
    else:
        groups = partition_zones(
            criterion, grid_ids, ctx["gdf"], ctx["source"], crs_epsg,
            customers_df=ctx["customers"], horizon_years=T, max_zone_kva=max_kva,
            max_zone_radius_m=float(ctx["params"]["max_lv_zone_radius_m"]),
            max_zone_customers=int(ctx["params"]["max_zone_customers"]),
            max_cluster_diameter_m=float(ctx["params"]["max_cluster_diameter_m"]),
            seed=seed,
        )
        if criterion == "distance_cap" and len(groups) > max_transformers:
            raise ValueError(
                f"The {float(ctx['params']['max_cluster_diameter_m']):.0f} m diameter cap requires "
                f"{len(groups)} transformers, above the safety cap of {max_transformers}. Raise "
                f"'max_transformers' or relax the diameter cap."
            )
        best = _build_mvlv_at(ctx, groups, stepup_spec)
        iterations = [_iter_record(len(best["plan"].zones), best["outcome"], criterion)]

    plan = best["plan"]
    plan.clustering_criterion = criterion
    plan.mv_iterations = iterations

    # Coincident-peak stress: same network + chosen MV conductor, every customer at horizon peak
    # simultaneously (factor 1.0). Reported alongside the design outcome, not used to gate feasibility.
    stress_model = build_mvlv_net(
        plan, ctx["customers"], core_conductor_by_line={}, zone_conductor_by_line=best["zone_cond"],
        mv_conductor=best["mv_cond"], transformer_by_zone=best["transformer_by_zone"],
        lv_kv=ctx["lv_kv"], mv_kv=ctx["mv_kv"], horizon_years=T, apply_growth=True,
        stepup_transformer=stepup_spec, coincidence_factor=1.0, source_vm_pu=ctx["source_vm_pu"],
    )
    stress_outcome = run_powerflow(stress_model, ctx["constraints"])
    return {**best, "stepup": stepup_spec, "stress_model": stress_model,
            "stress_outcome": stress_outcome, "iterations": iterations}


def _run_mvlv(ctx, grid_ids: List[str], *, cid: str = "C_MVLV", architecture: str = "MV/LV",
              sa_npc: float = 0.0, sa_customers: Optional[List[str]] = None,
              sa_assignments: Optional[Dict[str, Dict[str, Any]]] = None) -> CaseResult:
    """Build an MV/LV zoned candidate over ``grid_ids``.

    Shared by Case C (all customers on the grid) and Case D (the Case-B stand-alone customers
    removed first, then the remainder MV/LV-zoned). Pass the stand-alone NPC/customers/assignments
    to fold them into the candidate.
    """
    T = ctx["T"]
    sa_customers = list(sa_customers or [])
    sa_assignments = dict(sa_assignments or {})
    grid_ids = list(grid_ids)
    if not grid_ids:
        cost = cm.CostBreakdown(standalone_npc=float(sa_npc))
        assign = dict(sa_assignments)
        return CaseResult(candidate_id=cid, architecture=architecture, feasible=False, cost=cost,
                          n_standalone_customers=len(sa_customers),
                          standalone_customer_ids=list(sa_customers), standalone_assignments=sa_assignments,
                          customer_assignment=assign, decision_summary="No grid customers to serve via MV/LV")

    built = _build_mvlv(ctx, grid_ids)
    plan, zone_cond, transformers_chosen = built["plan"], built["zone_cond"], built["transformers"]
    mv_cond, final_model, final_outcome = built["mv_cond"], built["model"], built["outcome"]
    stress_model, stress_outcome = built["stress_model"], built["stress_outcome"]
    stepup = built["stepup"]
    criterion = plan.clustering_criterion or "voltage_cap"
    _log(ctx["log"], "mvlv_zone", cid, "partition", f"{criterion} → {len(plan.zones)} transformer zones",
         "n_zones", f"{len(plan.zones)} over {len(grid_ids)} customers")
    for it in built["iterations"]:
        _log(ctx["log"], "mv_transformer_count", cid, f"k={it['n_zones']}",
             "feasible" if it["feasible"] else ("diverged" if not it["converged"] else "infeasible"),
             it["bottleneck"], "worst_dv_pu", f"{it['worst_dv_pu']:.4f}", threshold=it["note"])
    _log(ctx["log"], "transformer_sizing", cid, "stepup", f"plant step-up {stepup['sn_kva']:.0f} kVA",
         "system coincident S × margin, may exceed catalogue", "sn_kva", f"{stepup['sn_kva']}")
    for tx in transformers_chosen:
        _log(ctx["log"], "transformer_sizing", cid, tx["zone_id"], f"select {tx['transformer_id']}",
             "smallest rating covering horizon S", "sn_kva", f"{tx['sn_kva']}")
    _log(ctx["log"], "conductor_sizing", cid, "mv_trunk", "pass" if final_outcome.feasible else "fail",
         final_outcome.active_bottleneck, "mv_conductor", str((mv_cond or {}).get("conductor_id")))

    now_model = build_mvlv_net(
        plan, ctx["customers"], core_conductor_by_line={}, zone_conductor_by_line=zone_cond,
        mv_conductor=mv_cond, transformer_by_zone={z["zone_id"]: z["transformer_spec"] for z in plan.zones},
        lv_kv=ctx["lv_kv"], mv_kv=ctx["mv_kv"], horizon_years=T, apply_growth=False,
        stepup_transformer=stepup, source_vm_pu=ctx["source_vm_pu"],
        coincidence_factor=ctx["coincidence"],
    )
    now = run_powerflow(now_model, ctx["constraints"])

    cost = cm.CostBreakdown()
    lv_len_by_line: Dict[str, float] = {}
    for zone in plan.zones:
        lv_len_by_line.update(_length_by_line(zone["topology"], prefix=f"{zone['zone_id']}::"))
    cost.lv_line_cost = cm.line_cost(zone_cond, lv_len_by_line, ctx["params"]["civil_logistics_multiplier"])
    mv_len_by_line: Dict[str, float] = {}
    if getattr(plan, "mv_topology", None) is not None:
        mv_len_by_line.update(_length_by_line(plan.mv_topology, prefix="mv::"))
        for zone in plan.zones:
            mv_len_by_line[f"mv_drop::{zone['zone_id']}"] = float(zone.get("mv_service_drop_m", 0.0))
    else:
        mv_len_by_line = {f"mv::{z['zone_id']}": z["mv_length_m"] for z in plan.zones}
    cost.mv_line_cost = cm.line_cost({lid: mv_cond for lid in mv_len_by_line}, mv_len_by_line, ctx["params"]["civil_logistics_multiplier"])
    n_poles = sum(len(z["topology"].pole_xy) for z in plan.zones)
    cost.pole_cost = cm.pole_capex(n_poles, ctx["params"]["pole_cost"], ctx["params"]["civil_logistics_multiplier"])
    cost.transformer_cost = cm.transformer_capex(transformers_chosen + [stepup])
    service_drop_total = sum(
        float(z["topology"].service_drop_m.get(c, 0.0))
        for z in plan.zones for c in z["topology"].served_customer_ids
    )
    grid_served_ids = [c for z in plan.zones for c in z["topology"].served_customer_ids]
    n_grid = len(grid_served_ids)
    cost.connection_metering_cost = cm.connection_capex(n_grid, ctx["params"])
    cost.loss_pv_cost = cm.loss_present_value(final_outcome.losses_kw, ctx["params"])
    cost.generation_npc = _generation_npc(ctx, grid_served_ids)
    cost.standalone_npc = float(sa_npc)

    assign: Dict[str, Dict[str, Any]] = {}
    for zone in plan.zones:
        for c in zone["topology"].served_customer_ids:
            assign[c] = {"status": "grid_now", "architecture": architecture, "cluster_id": zone["zone_id"],
                         "transformer_id": zone["transformer_id"], "package_id": "", "reason": "MV/LV service zone"}
    assign.update(sa_assignments)

    return CaseResult(
        candidate_id=cid, architecture=architecture, feasible=final_outcome.feasible, cost=cost,
        outcome_now=now, outcome_horizon=final_outcome, outcome_stress=stress_outcome,
        n_grid_customers=n_grid, n_standalone_customers=len(sa_customers),
        lv_length_m=sum(lv_len_by_line.values()), mv_length_m=plan.mv_total_length_m,
        n_poles=n_poles, service_drop_total_m=service_drop_total, n_transformers=len(transformers_chosen),
        topology=None, plan=plan, model_now=now_model, model=final_model, model_stress=stress_model,
        conductor_by_line=zone_cond,
        grid_customer_ids=[c for c in assign.keys() if c not in set(sa_customers)],
        standalone_customer_ids=list(sa_customers), standalone_assignments=sa_assignments,
        customer_assignment=assign, transformers=transformers_chosen,
        decision_summary=(
            f"{len(plan.zones)} MV/LV zones, {len(transformers_chosen)} transformers"
            + (f"; {len(sa_customers)} stand-alone customers" if sa_customers else "")
        ),
    )


def detect_mvlv_subtree_candidates(mvlv_res: CaseResult, ctx) -> List[Cluster]:
    """MV/LV-specific SHS screen: terminal subtrees *within each transformer zone's LV tree*.

    Each MV/LV zone owns a radial LV tree (``zone["topology"]``); its terminal subtrees are the
    candidates, so the unit of decision is a leaf branch — never the whole zone. Avoided cost is
    LV-marginal (see :func:`_avoided_mvlv_subtree_cost`); the zone transformer and MV service drop
    are credited only implicitly, through the Case-D rebuild, when a zone is fully evacuated.
    Cluster ids are namespaced by zone (e.g. ``Z03-S01``).
    """
    plan = getattr(mvlv_res, "plan", None)
    if plan is None or not getattr(plan, "zones", None):
        return []
    largest_lv_id = str(ctx["lv_conductors"].sort_values("rank").iloc[-1]["conductor_id"])
    zone_cond_all = mvlv_res.conductor_by_line or {}

    candidates: List[Cluster] = []
    for zone in plan.zones:
        zid = zone["zone_id"]
        # Zone conductors are keyed "<zone_id>::<line_id>"; strip the prefix back to bare line ids.
        zcond = {k.split("::", 1)[1]: v for k, v in zone_cond_all.items() if k.startswith(f"{zid}::")}
        candidates.extend(subtree_clusters(
            zone["topology"], ctx["customers"], conductor_by_line=zcond, cost_params=ctx["params"],
            horizon_years=ctx["T"], largest_conductor_id=largest_lv_id,
            full_model=None, id_prefix=f"{zid}-S",
        ))
    return candidates


def _run_mvlv_standalone(ctx, full_res: CaseResult, sa_customers, sa_assignments, sa_npc) -> CaseResult:
    """Case D: apply the MV/LV-specific subtree stand-alone screen, then rebuild MV/LV over the rest.

    The stand-alone set comes from :func:`detect_mvlv_subtree_candidates` (per-zone terminal
    subtrees), not from Case B's LV screen, so D never reuses B's SHS customers.
    """
    cid = "D_MVLV_plus_standalone"
    sa_customers = list(sa_customers)
    sa_set = set(sa_customers)
    grid_ids = [c for c in full_res.grid_customer_ids if c not in sa_set]

    res = _run_mvlv(
        ctx, grid_ids, cid=cid, architecture="MV/LV + Stand-alone",
        sa_npc=float(sa_npc), sa_customers=sa_customers, sa_assignments=dict(sa_assignments),
    )
    if not sa_customers:
        res.decision_summary += "; no stand-alone zone selected"
    return res


# ---------------------------------------------------------------------------
# Tables + orchestration
# ---------------------------------------------------------------------------

def _comparison_row(res: CaseResult, customers_df) -> Dict[str, Any]:
    o_n, o_h, o_s = res.outcome_now, res.outcome_horizon, res.outcome_stress
    return {
        "candidate_id": res.candidate_id,
        "architecture": res.architecture,
        "feasible": res.feasible,
        "npc_total": round(res.cost.total_npc, 0),
        # False => npc_total omits an unknown component (see npc_missing_components) and is
        # a LOWER BOUND, not comparable with a complete total.
        "npc_complete": res.cost.npc_complete,
        "npc_missing_components": "; ".join(res.cost.missing_components),
        "grid_capex": round(res.cost.grid_capex, 0),
        "generation_npc": round(res.cost.generation_npc, 0),
        "standalone_npc": round(res.cost.standalone_npc, 0),
        "lv_length_m": round(res.lv_length_m, 0),
        "mv_length_m": round(res.mv_length_m, 0),
        "n_transformers": res.n_transformers,
        "n_grid_customers": res.n_grid_customers,
        "n_standalone_customers": res.n_standalone_customers,
        "must_connect_violations": res.must_connect_violations,
        "service_tier_violations": res.service_tier_violations,
        "min_voltage_now": round(o_n.min_vm_pu, 3) if o_n else None,
        "min_voltage_horizon": round(o_h.min_vm_pu, 3) if o_h else None,
        "max_line_loading_now": round(o_n.max_line_loading_percent, 1) if o_n else None,
        "max_line_loading_horizon": round(o_h.max_line_loading_percent, 1) if o_h else None,
        "max_transformer_loading_now": round(o_n.max_trafo_loading_percent, 1) if o_n else None,
        "max_transformer_loading_horizon": round(o_h.max_trafo_loading_percent, 1) if o_h else None,
        "losses_kw": round(o_h.losses_kw, 2) if o_h else None,
        "losses_percent": round(o_h.losses_percent, 2) if o_h else None,
        "horizon_growth_feasible": bool(o_h.feasible) if o_h else None,
        "horizon_growth_bottleneck": o_h.active_bottleneck if o_h else None,
        # Coincident-peak stress (all customers at horizon peak simultaneously) — informational.
        "coincident_within_limits": bool(o_s.feasible) if o_s else None,
        "coincident_min_voltage": round(o_s.min_vm_pu, 3) if o_s else None,
        "coincident_max_line_loading": round(o_s.max_line_loading_percent, 1) if o_s else None,
        "coincident_max_transformer_loading": round(o_s.max_trafo_loading_percent, 1) if o_s else None,
        "coincident_bottleneck": o_s.active_bottleneck if o_s else None,
        "decision_summary": res.decision_summary,
    }


def _validation_row(res: CaseResult, case_name: str, outcome) -> Dict[str, Any]:
    if outcome is None:
        return {"candidate_id": res.candidate_id, "case_name": case_name, "converged": None}
    return {
        "candidate_id": res.candidate_id, "case_name": case_name,
        "converged": outcome.converged, "feasible": outcome.feasible,
        "min_voltage_pu": round(outcome.min_vm_pu, 3), "max_voltage_pu": round(outcome.max_vm_pu, 3),
        "max_line_loading_percent": round(outcome.max_line_loading_percent, 1),
        "max_transformer_loading_percent": round(outcome.max_trafo_loading_percent, 1),
        "losses_kw": round(outcome.losses_kw, 2), "losses_percent": round(outcome.losses_percent, 2),
        "active_bottleneck": outcome.active_bottleneck,
    }


def _strategy_table(customers_df, recommended: CaseResult, T) -> pd.DataFrame:
    rows = []
    by_id = customers_df.set_index("customer_id")
    for cid in customers_df["customer_id"].astype(str):
        r = by_id.loc[cid]
        a = recommended.customer_assignment.get(cid, {"status": "not_served_error", "architecture": recommended.architecture, "cluster_id": "", "transformer_id": "", "package_id": "", "reason": "no assignment"})
        rows.append({
            "customer_id": cid,
            "customer_type": r["customer_type"],
            "demand_tier": int(r["demand_tier"]),
            "peak_kw_now": round(float(r["peak_kw_now"]), 3),
            "peak_kw_horizon": round(float(r["peak_kw_now"]) * (1 + float(r["growth_rate"])) ** T, 3),
            "annual_kwh_now": round(float(r["annual_kwh_now"]), 1),
            "must_connect_flag": bool(r["must_connect_flag"]),
            "can_be_standalone_flag": bool(r["can_be_standalone_flag"]),
            "recommended_status": a["status"],
            "assigned_architecture": a["architecture"],
            "assigned_cluster_id": a["cluster_id"],
            "assigned_transformer_id": a["transformer_id"],
            "standalone_package_id": a["package_id"],
            "decision_reason": a["reason"],
        })
    return pd.DataFrame(rows)


def _tier_service_table(cases: Dict[str, CaseResult], customers_df, packages) -> pd.DataFrame:
    """Service tier served to every customer under every candidate, with expansion headroom.

    Grid-served customers keep growing with the network (their tier can evolve with future demand),
    so their service ceiling is open-ended. Stand-alone customers are capped at their SHS package's
    ``service_tier`` — a hard ceiling that matters for future demand evolution and expansion.
    Columns: candidate_id, architecture, customer_id, demand_tier, status, package_id,
    service_ceiling_tier (NaN = grid, open-ended), tier_headroom (ceiling - demand_tier; NaN for
    grid), capped (True = stand-alone with zero tier headroom).
    """
    pkg_tier = {str(p): int(t) for p, t in zip(packages["package_id"].astype(str), packages["service_tier"])}
    by_id = customers_df.set_index(customers_df["customer_id"].astype(str))
    rows: List[Dict[str, Any]] = []
    for key, case in cases.items():
        assign = case.customer_assignment or {}
        for cid in customers_df["customer_id"].astype(str):
            demand_tier = int(by_id.loc[cid, "demand_tier"])
            a = assign.get(cid)
            status = a["status"] if a else "unserved"
            pkg_id = (a or {}).get("package_id", "") or ""
            if status == "standalone":
                ceiling = pkg_tier.get(str(pkg_id))
                headroom = (ceiling - demand_tier) if ceiling is not None else None
                capped = ceiling is not None and (ceiling - demand_tier) <= 0
            else:  # grid_now / unserved: no fixed generation ceiling from a package
                ceiling = None
                headroom = None
                capped = False
            rows.append({
                "candidate_id": key,
                "architecture": case.architecture,
                "customer_id": cid,
                "demand_tier": demand_tier,
                "status": "grid" if status == "grid_now" else status,
                "package_id": pkg_id,
                "service_ceiling_tier": ceiling,
                "tier_headroom": headroom,
                "capped": bool(capped),
            })
    return pd.DataFrame(rows)


def _cost_row(res: CaseResult) -> Dict[str, Any]:
    c = res.cost
    return {
        "candidate_id": res.candidate_id,
        "lv_line_cost": round(c.lv_line_cost, 0), "mv_line_cost": round(c.mv_line_cost, 0),
        "pole_cost": round(c.pole_cost, 0), "transformer_cost": round(c.transformer_cost, 0),
        "connection_metering_cost": round(c.connection_metering_cost, 0),
        "standalone_npc": round(c.standalone_npc, 0), "loss_pv_cost": round(c.loss_pv_cost, 0),
        "generation_npc": round(c.generation_npc, 0),
        "total_npc": round(c.total_npc, 0),
        "npc_complete": c.npc_complete,
    }


def run_simplified_mvp(
    customers: pd.DataFrame,
    roads,
    source: Dict[str, Any],
    lv_conductors: pd.DataFrame,
    mv_conductors: pd.DataFrame,
    transformers: pd.DataFrame,
    standalone_packages: pd.DataFrame,
    cost_parameters: Dict[str, float],
    engineering_parameters: Dict[str, Any],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> MVPResult:
    def emit(msg: str) -> None:
        if progress:
            progress(msg)

    params = dict(cost_parameters)
    eng = dict(engineering_parameters)
    # Engineering overrides for shared keys.
    for k in ("planning_horizon_years", "discount_rate", "voltage_min_pu", "voltage_max_pu",
              "max_line_loading_percent", "max_transformer_loading_percent",
              "coincidence_factor", "generation_lcoe_per_kwh",
              "clustering_criterion", "max_cluster_diameter_m", "max_transformers", "source_vm_pu"):
        if k in eng and eng[k] is not None:
            params[k] = eng[k]
    T = int(params["planning_horizon_years"])
    constraints = {
        "voltage_min_pu": float(params["voltage_min_pu"]),
        "voltage_max_pu": float(params["voltage_max_pu"]),
        "max_line_loading_percent": float(params["max_line_loading_percent"]),
        "max_transformer_loading_percent": float(params["max_transformer_loading_percent"]),
    }

    gdf, crs_meta = customers_to_engine_gdf(customers)
    log: List[Dict[str, Any]] = []
    annual_kwh_by_id = {
        str(cid): float(kwh)
        for cid, kwh in zip(customers["customer_id"].astype(str), customers["annual_kwh_now"])
    }
    ctx = {
        "customers": customers, "gdf": gdf, "roads": roads, "crs_meta": crs_meta, "source": source,
        "eng": eng, "params": params, "constraints": constraints, "T": T,
        "discount": float(params["discount_rate"]), "lv_conductors": lv_conductors,
        "mv_conductors": mv_conductors, "transformers": transformers, "packages": standalone_packages,
        "lv_kv": float(source.get("lv_voltage_kv", 0.4)), "mv_kv": float(source.get("mv_voltage_kv", 11.0)),
        "coincidence": float(params["coincidence_factor"]), "annual_kwh_by_id": annual_kwh_by_id,
        "source_vm_pu": float(params["source_vm_pu"]),
        "log": log,
    }

    emit("Generating full LV topology and running LV conductor sizing...")
    full_res = _run_full_lv(ctx)

    emit("Building MV/LV service-zone case...")
    mvlv_res = _run_mvlv(ctx, full_res.grid_customer_ids)

    # Case B: LV screen over the terminal subtrees (leaf branches) of the Full-LV tree.
    emit("Generating Full-LV terminal subtrees (leaf branches)...")
    largest_lv_id = str(lv_conductors.sort_values("rank").iloc[-1]["conductor_id"])
    lv_clusters = detect_lv_subtrees(
        ctx["full_lv"]["topology"], customers, conductor_by_line=full_res.conductor_by_line,
        full_model=full_res.model, cost_params=params, horizon_years=T, largest_conductor_id=largest_lv_id,
    )
    for cl in lv_clusters:
        _log(log, "branch_detection", "A_full_LV", cl.cluster_id,
             "flagged" if cl.is_problematic else "clean", "; ".join(cl.reasons) or "within limits",
             "n_customers", cl.n_customers)

    emit("Building LV + stand-alone case (LV leaf-branch SHS screen)...")
    lv_sa = _select_standalone_clusters(ctx, lv_clusters, _avoided_lv_cost, "LV + Stand-alone", "B_LV_plus_standalone")
    sa_res = _run_lv_standalone(ctx, full_res, *lv_sa)

    # Case D: MV/LV screen over the terminal subtrees inside each zone's LV tree (own SHS set, not B's).
    emit("Building MV/LV + stand-alone case (per-zone leaf-branch SHS screen)...")
    mvlv_clusters = detect_mvlv_subtree_candidates(mvlv_res, ctx)
    for cl in mvlv_clusters:
        _log(log, "mvlv_subtree", "C_MVLV", cl.cluster_id,
             "flagged" if cl.is_problematic else "clean", "; ".join(cl.reasons) or "within limits",
             "n_customers", cl.n_customers)
    mvlv_sa = _select_standalone_clusters(ctx, mvlv_clusters, _avoided_mvlv_subtree_cost, "MV/LV + Stand-alone", "D_MVLV_plus_standalone")
    mvlv_sa_res = _run_mvlv_standalone(ctx, full_res, *mvlv_sa)

    cases = {
        "A_full_LV": full_res,
        "B_LV_plus_standalone": sa_res,
        "C_MVLV": mvlv_res,
        "D_MVLV_plus_standalone": mvlv_sa_res,
    }

    # Service obligations: must_connect customers must be grid-served; everyone must get some
    # service (grid or stand-alone). A must_connect violation makes a candidate infeasible.
    mc_ids = set(customers.loc[customers["must_connect_flag"], "customer_id"].astype(str))
    all_ids = set(customers["customer_id"].astype(str))
    for r in cases.values():
        grid = set(r.grid_customer_ids)
        served = grid | set(r.standalone_customer_ids)
        r.must_connect_violations = len(mc_ids - grid)
        r.service_tier_violations = len(all_ids - served)
        if r.must_connect_violations > 0:
            r.feasible = False

    # Recommendation: feasible candidate with the lowest NPC; else the least-infeasible one.
    #
    # Ranking infeasible candidates by NPC would be wrong: a candidate whose power flow
    # diverged has no computable loss cost, so its total is a lower bound and it would win
    # the comparison precisely BECAUSE it is broken. Rank them by how close they are to
    # being a usable network instead, and only use cost to break ties between candidates
    # that are equally broken and equally costed.
    feasible_ids = [k for k, r in cases.items() if r.feasible]
    if feasible_ids:
        recommended = min(feasible_ids, key=lambda k: cases[k].cost.total_npc)
        note = f"Lowest-NPC feasible candidate: {recommended}"
    else:
        def _infeasibility_rank(key: str):
            r = cases[key]
            o = r.outcome_horizon
            return (
                0 if (o is not None and o.converged) else 1,   # a solvable network first
                r.must_connect_violations,                     # then service obligations
                r.service_tier_violations,
                violation_severity(o, constraints),            # then distance from the limits
                0 if r.cost.npc_complete else 1,               # then a comparable cost
                r.cost.total_npc,
            )
        recommended = min(cases, key=_infeasibility_rank)
        rec = cases[recommended]
        bottleneck = rec.outcome_horizon.active_bottleneck if rec.outcome_horizon else "n/a"
        note = (
            f"No feasible candidate; least-infeasible by violation severity: "
            f"{recommended} ({bottleneck})"
        )
        if not rec.cost.npc_complete:
            note += f" - NPC incomplete, missing {', '.join(rec.cost.missing_components)}"

    _log(log, "comparison", recommended, "recommendation", "select", note)

    emit("Comparing candidates...")
    comparison = pd.DataFrame([_comparison_row(r, customers) for r in cases.values()])
    cost_table = pd.DataFrame([_cost_row(r) for r in cases.values()])
    val_rows = []
    for r in cases.values():
        val_rows.append(_validation_row(r, "now", r.outcome_now))
        val_rows.append(_validation_row(r, "horizon", r.outcome_horizon))
    validation = pd.DataFrame(val_rows)

    strategy = _strategy_table(customers, cases[recommended], T)
    tier_service = _tier_service_table(cases, customers, standalone_packages)

    return MVPResult(
        cases=cases, comparison_table=comparison, customer_strategy_table=strategy,
        decision_log=pd.DataFrame(log), validation_summary=validation, cost_breakdown_table=cost_table,
        tier_service_table=tier_service,
        recommended=recommended, recommendation_note=note, clusters=lv_clusters,
        inputs={"source": source, "horizon_years": T, "discount_rate": params["discount_rate"], "params": params},
    )
