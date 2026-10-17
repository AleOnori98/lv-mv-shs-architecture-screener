"""End-to-end simplified MVP: Full LV, hybrids and MV/LV, compared by NPC.

This is a sequential diagnostic workflow, not a co-optimization model. Each step is
deterministic and logged to ``decision_log`` for traceability.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import pandas as pd

from core.io.load_inputs import customers_to_engine_gdf
from core.topology.lv_topology import build_lv_topology
from core.topology.branch_detection import Cluster, detect_lv_subtrees, subtree_clusters
from core.topology.mvlv_service_zones import (
    TRANSFORMER_SIZING_SAFETY_FACTOR,
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
    """``line_id -> length_m`` for one topology, optionally namespaced by a zone prefix."""
    if topology is None or topology.edges_4326 is None or topology.edges_4326.empty:
        return {}
    edges = topology.edges_4326
    return {f"{prefix}{lid}": float(m) for lid, m in zip(edges["line_id"], edges["length_m"])}


def _size_flat_lv(ctx, topology):
    """Size the LV conductors of one topology against the design (horizon) snapshot.

    ``lv_kv`` comes from ``ctx`` like every other snapshot's; it used to be read from
    ``topology.metadata``, so the sizing build and the snapshots it was validated against
    could in principle disagree on the voltage level.
    """
    line_ids = list(_length_by_line(topology).keys())
    build_fn = lambda cond: build_lv_net(
        topology, ctx["customers"], cond, lv_kv=ctx["lv_kv"], horizon_years=ctx["T"],
        apply_growth=True, coincidence_factor=ctx["coincidence"],
    )
    critical_fn = lambda model, outcome: critical_lv_lines(model, outcome, topology)
    return size_conductors(line_ids, ctx["lv_conductors"], build_fn, ctx["constraints"],
                           critical_fn=critical_fn)


def _lv_powerflow(ctx, topology, conductor_by_line, *, apply_growth: bool, coincidence: float):
    """Solve one LV snapshot. The three snapshots differ only in these two arguments:

    design   apply_growth=True,  coincidence=ctx["coincidence"]   (gates feasibility)
    now      apply_growth=False, coincidence=ctx["coincidence"]   (reference)
    stress   apply_growth=True,  coincidence=1.0                  (every customer at peak)
    """
    model = build_lv_net(
        topology, ctx["customers"], conductor_by_line, lv_kv=ctx["lv_kv"],
        horizon_years=ctx["T"], apply_growth=apply_growth, coincidence_factor=coincidence,
    )
    return model, run_powerflow(model, ctx["constraints"])


def build_context(
    *,
    customers: pd.DataFrame,
    params: Dict[str, Any],
    lv_conductors: pd.DataFrame,
    packages: pd.DataFrame,
    horizon_years: int,
    discount_rate: float,
    **extra: Any,
) -> Dict[str, Any]:
    """Assemble the per-run context the cost/screen helpers read.

    Every helper that prices something takes this dict, so the derived lookups
    (``annual_kwh_by_id``, ``growth_by_id``, the resolved transformer sizing factor) belong
    in one place. Hand-rolling the dict -- which the tests used to do -- means a new key
    added to the pipeline silently breaks callers with a ``KeyError`` far from the cause,
    so both the runner and the tests build it here.
    """
    cid = customers["customer_id"].astype(str)
    ctx: Dict[str, Any] = {
        "customers": customers,
        "params": params,
        "lv_conductors": lv_conductors,
        "packages": packages,
        "T": int(horizon_years),
        "discount": float(discount_rate),
        "annual_kwh_by_id": {
            str(c): float(v) for c, v in zip(cid, customers["annual_kwh_now"])
        },
        "growth_by_id": {
            str(c): float(v) for c, v in zip(cid, customers["growth_rate"])
        },
        "coincidence": float(params["coincidence_factor"]),
        # Catalogue lookups that are invariant for the whole run.
        "cheapest_lv_cost_per_m": cm.cheapest_lv_cost_per_m(lv_conductors),
        # Diversity basis for transformer sizing / the zone kVA cap; 0 means "track the
        # design coincidence factor" (see transformer_sizing_coincidence_factor).
        "tx_coincidence": (
            float(params["transformer_sizing_coincidence_factor"])
            if float(params.get("transformer_sizing_coincidence_factor", 0.0) or 0.0) > 0.0
            else float(params["coincidence_factor"])
        ),
        "log": [],
    }
    ctx.update(extra)
    return ctx


def _energy_weighted_growth(ctx, customer_ids) -> float:
    """Energy-weighted mean annual demand growth of a customer set.

    One growth rate for the whole set, weighted by each customer's share of the energy, so
    the levelised energy costs track the trajectory of the demand that actually drives them.
    """
    kwh_by_id, growth_by_id = ctx["annual_kwh_by_id"], ctx["growth_by_id"]
    total = 0.0
    weighted = 0.0
    for c in customer_ids:
        kwh = kwh_by_id.get(str(c), 0.0)
        total += kwh
        weighted += kwh * growth_by_id.get(str(c), 0.0)
    return (weighted / total) if total > 0 else 0.0


def _generation_npc(ctx, grid_ids) -> float:
    """PV of mini-grid generation cost for the given grid-served customers (see cost_model).

    Uses each customer's own growth rate exactly (not the weighted mean), since the sum of
    per-customer PVs is exact and just as cheap here.
    """
    kwh_by_id, growth_by_id = ctx["annual_kwh_by_id"], ctx["growth_by_id"]
    total = 0.0
    for c in grid_ids:
        key = str(c)
        kwh = kwh_by_id.get(key, 0.0)
        if kwh:
            total += cm.generation_present_value(
                kwh, ctx["params"], growth_rate=growth_by_id.get(key, 0.0)
            )
    return total


def _loss_npc(ctx, losses_kw, grid_ids) -> float:
    """PV of loss energy for a candidate, levelised on that candidate's demand growth."""
    return cm.loss_present_value(
        losses_kw, ctx["params"], growth_rate=_energy_weighted_growth(ctx, grid_ids)
    )


def _service_drop_cost(ctx, total_drop_m: float) -> float:
    """CAPEX of the service drops, on the same proxy the avoided-cost side credits.

    The cheapest-conductor price is read from the context: this runs once per candidate AND
    once per screened cluster, and the catalogue does not change during a run.
    """
    return cm.service_drop_capex(
        total_drop_m, ctx["cheapest_lv_cost_per_m"],
        ctx["params"]["civil_logistics_multiplier"],
    )


# ---------------------------------------------------------------------------
# Case A — Full LV
# ---------------------------------------------------------------------------

def _assignment(status: str, architecture: str, *, cluster_id: str = "",
                transformer_id: str = "", package_id: str = "", reason: str = "") -> Dict[str, Any]:
    """One per-customer assignment row. The six keys are the schema that the customer
    strategy and tier-service tables index unconditionally, so they live in one place."""
    return {"status": status, "architecture": architecture, "cluster_id": cluster_id,
            "transformer_id": transformer_id, "package_id": package_id, "reason": reason}


def _excluding(ids, excluded) -> List[str]:
    """``ids`` minus ``excluded``, order preserved, with the exclusion set hoisted."""
    drop = set(excluded)
    return [c for c in ids if c not in drop]


def _run_flat_lv(ctx, *, cid: str, architecture: str, customer_ids: Optional[List[str]] = None,
                 sa_customers: Optional[List[str]] = None,
                 sa_assignments: Optional[Dict[str, Dict[str, Any]]] = None,
                 sa_npc: float = 0.0, grid_reason: str = "connected to LV grid") -> CaseResult:
    """Build one flat-LV candidate: topology, conductor sizing, three snapshots, costs.

    Shared by Case A (every customer) and Case B (the stand-alone customers removed and
    their NPC folded in). These were two copies of the same ~35 lines, and this diff's
    history shows why that matters: the service-drop component had to be added to both, and
    anything landing in only one makes the A-vs-B NPC comparison invalid.
    """
    sa_customers = list(sa_customers or [])
    topo = build_lv_topology(ctx["gdf"], ctx["roads"], ctx["crs_meta"], ctx["source"], ctx["eng"],
                             customer_ids=customer_ids)
    topo.metadata["lv_kv"] = ctx["lv_kv"]

    sizing = _size_flat_lv(ctx, topo)
    topo.bus_by_pole = sizing.model.bus_by_pole
    _log(ctx["log"], "conductor_sizing", cid, "lv_network", "pass" if sizing.passed else "fail",
         sizing.outcome.active_bottleneck, "iterations", sizing.iterations)

    now_model, now = _lv_powerflow(ctx, topo, sizing.conductor_by_line,
                                   apply_growth=False, coincidence=ctx["coincidence"])
    stress_model, stress = _lv_powerflow(ctx, topo, sizing.conductor_by_line,
                                         apply_growth=True, coincidence=1.0)

    served = list(topo.served_customer_ids)
    params = ctx["params"]
    mult = params["civil_logistics_multiplier"]
    cost = cm.CostBreakdown(
        lv_line_cost=cm.line_cost(sizing.conductor_by_line, _length_by_line(topo), mult),
        pole_cost=cm.pole_capex(len(topo.pole_xy), params["pole_cost"], mult),
        connection_metering_cost=cm.connection_capex(len(served), params),
        service_drop_cost=_service_drop_cost(ctx, topo.total_service_drop_m),
        loss_pv_cost=_loss_npc(ctx, sizing.outcome.losses_kw, served),
        generation_npc=_generation_npc(ctx, served),
        standalone_npc=float(sa_npc),
    )

    assign = {c: _assignment("grid_now", architecture, reason=grid_reason) for c in served}
    assign.update(sa_assignments or {})

    return CaseResult(
        candidate_id=cid, architecture=architecture, feasible=sizing.passed, cost=cost,
        outcome_now=now, outcome_horizon=sizing.outcome, outcome_stress=stress,
        n_grid_customers=len(served), n_standalone_customers=len(sa_customers),
        lv_length_m=topo.total_lv_length_m, mv_length_m=0.0, n_transformers=0,
        n_poles=len(topo.pole_xy), service_drop_total_m=topo.total_service_drop_m,
        sizing_iterations=sizing.iterations, topology=topo, model_now=now_model,
        model=sizing.model, model_stress=stress_model,
        conductor_by_line=sizing.conductor_by_line, grid_customer_ids=served,
        standalone_customer_ids=sa_customers, standalone_assignments=dict(sa_assignments or {}),
        customer_assignment=assign,
    )


def _run_full_lv(ctx) -> CaseResult:
    """Case A: every customer on one flat LV network."""
    res = _run_flat_lv(ctx, cid="A_full_LV", architecture="Full LV",
                       grid_reason="connected in Full LV")
    _log(ctx["log"], "topology", "A_full_LV", "full_lv", "generate",
         "all customers connected (allow_unserved=False)", "n_served", res.n_grid_customers)
    res.decision_summary = ("Feasible single LV network" if res.feasible
                            else f"Infeasible: {res.outcome_horizon.active_bottleneck}")
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
    # Same helper, same proxy, same metres as the candidate NPC now charges.
    service_drop_cost = _service_drop_cost(ctx, float(cluster.metadata.get("service_drop_m", 0.0)))
    conn = cm.connection_capex(cluster.n_customers, params)
    pole_cost = cm.pole_capex(len(cluster.pole_ids), params["pole_cost"], mult)
    generation_pv = _generation_npc(ctx, cluster.customer_ids)
    avoided = cluster.branch_cost + service_drop_cost + conn + pole_cost + generation_pv
    if include_losses:
        avoided += _loss_npc(ctx, getattr(cluster, "branch_losses_kw", 0.0), cluster.customer_ids)
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


class StandaloneSelection(NamedTuple):
    """What one SHS screen decided: the customers, their assignments and their total NPC.

    A named value rather than a bare 3-tuple because it was threaded through four
    signatures in two different orders, and both call sites star-unpacked it -- so
    reordering the tuple would have bound an assignments dict into the NPC slot silently.
    """

    customer_ids: List[str]
    assignments: Dict[str, Dict[str, Any]]
    npc: float


def _select_standalone_clusters(ctx, clusters, avoided_cost_fn, architecture_label, cid) -> StandaloneSelection:
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
    # Only eligible candidates can be selected, and the ordering only matters among those,
    # so ineligible ones are not priced at all.
    eligible = [cl for cl in clusters if cl.standalone_eligible_precheck]
    avoided_by = {id(cl): avoided_cost_fn(cl, ctx) for cl in eligible}
    for cl in sorted(eligible, key=lambda c: avoided_by[id(c)], reverse=True):
        if any(c in claimed for c in cl.customer_ids):  # a coarser subtree already took these
            continue
        sa = cluster_standalone(
            cl.customer_ids, ctx["customers"], ctx["packages"], horizon_years=T,
            discount_rate=rate,
            sizing_tolerance=float(ctx["params"].get("standalone_sizing_tolerance", 0.0) or 0.0),
        )
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
                assignments[ch.customer_id] = _assignment(
                    "standalone", architecture_label, cluster_id=cl.cluster_id,
                    package_id=ch.package_id, reason="stand-alone NPC < avoided grid cost",
                )
            npc_total += sa["total_npc"]
    return StandaloneSelection(selected, assignments, npc_total)


def _run_lv_standalone(ctx, full_res: CaseResult, selection: "StandaloneSelection") -> CaseResult:
    """Case B: the LV screen's stand-alone customers removed, the LV grid rebuilt on the rest."""
    cid = "B_LV_plus_standalone"
    if not selection.customer_ids:
        # Nothing was cheaper off-grid: B is Case A, re-labelled. Still a valid candidate.
        res = replace(full_res, candidate_id=cid, architecture="LV + Stand-alone",
                      decision_summary="No cluster cheaper as stand-alone; equals Full LV")
        res.customer_assignment = dict(full_res.customer_assignment)
        return res

    grid_ids = _excluding(full_res.grid_customer_ids, selection.customer_ids)
    res = _run_flat_lv(
        ctx, cid=cid, architecture="LV + Stand-alone", customer_ids=grid_ids,
        sa_customers=selection.customer_ids, sa_assignments=selection.assignments,
        sa_npc=selection.npc,
    )
    res.decision_summary = (
        f"{len(selection.customer_ids)} customers moved to stand-alone; LV grid resized"
    )
    return res


# ---------------------------------------------------------------------------
# Case C — MV/LV service zones (criterion-driven transformer count)
# ---------------------------------------------------------------------------

def _iter_record(n_zones: int, outcome, note: str, accepted: Optional[bool] = None,
                 v_target: Optional[float] = None) -> Dict[str, Any]:
    """One row of the transformer-count search history (for logs / UI).

    ``feasible`` is the candidate's own feasibility test; ``accepted`` is the stricter
    question the k-search asks (feasible AND holding the design voltage margin). They
    differ whenever a partition only just scrapes past the limit.
    """
    return {
        "n_zones": int(n_zones),
        "feasible": bool(outcome.feasible),
        "converged": bool(outcome.converged),
        "accepted": (bool(outcome.feasible) if accepted is None else bool(accepted)),
        "min_vm_pu": float(outcome.min_vm_pu),
        "v_target_pu": (float(v_target) if v_target is not None else float("nan")),
        "worst_dv_pu": (float(1.0 - outcome.min_vm_pu) if outcome.converged else float("nan")),
        "bottleneck": outcome.active_bottleneck,
        "note": note,
    }


def _k_search_failure(n_zones: int, note: str, error: str) -> Dict[str, Any]:
    """History row for a zone count that could not even be partitioned/built."""
    return {
        "n_zones": int(n_zones), "feasible": False, "converged": False, "accepted": False,
        "min_vm_pu": float("nan"), "v_target_pu": float("nan"), "worst_dv_pu": float("nan"),
        "bottleneck": f"could not build {n_zones} zones: {error}", "note": note,
    }


def _k_search_stop(n_zones: int, reason: str) -> Dict[str, Any]:
    """History row recording why the k-search stopped climbing (not a failed build)."""
    return {
        "n_zones": int(n_zones), "feasible": False, "converged": True, "accepted": False,
        "min_vm_pu": float("nan"), "v_target_pu": float("nan"), "worst_dv_pu": float("nan"),
        "bottleneck": reason, "note": "search_stopped",
    }


def _mvlv_net(ctx, plan, zone_cond, mv_cond, transformer_by_zone, stepup_spec,
              *, apply_growth: bool, coincidence: float):
    """Build one MV/LV pandapower model. The three snapshots differ only in these two args."""
    return build_mvlv_net(
        plan, ctx["customers"], zone_conductor_by_line=zone_cond,
        mv_conductor=mv_cond, transformer_by_zone=transformer_by_zone,
        lv_kv=ctx["lv_kv"], mv_kv=ctx["mv_kv"], horizon_years=ctx["T"],
        apply_growth=apply_growth, stepup_transformer=stepup_spec,
        coincidence_factor=coincidence, source_vm_pu=ctx["source_vm_pu"],
    )


def _build_mvlv_at(ctx, groups: List[List[str]], stepup_spec: Dict[str, Any],
                   *, with_stress: bool = False) -> Dict[str, Any]:
    """Build plan + per-zone LV sizing + MV-rank sweep + horizon PF for a FIXED partition.

    The transformer count is whatever ``groups`` contains; the caller decides how ``groups``
    was formed (capacity packing, distance-cap linkage, or a voltage-cap k-means attempt).
    """
    T, constraints = ctx["T"], ctx["constraints"]
    # Memoised across k attempts: with a nested partition family, k -> k+1 splits one zone
    # and leaves the others byte-identical, so their topology and conductor sizing are too.
    zone_cache = ctx.setdefault("zone_cache", {})
    sizing_cache = ctx.setdefault("zone_sizing_cache", {})

    plan = build_mvlv_plan(
        groups, ctx["gdf"], ctx["roads"], ctx["crs_meta"], ctx["source"], ctx["eng"],
        ctx["customers"], ctx["transformers"], horizon_years=T,
        coincidence_factor=ctx["tx_coincidence"], zone_cache=zone_cache,
    )
    zone_cond: Dict[str, Any] = {}
    transformers_chosen: List[Dict[str, Any]] = []
    for zone in plan.zones:
        ztopo = zone["topology"]
        ztopo.metadata["lv_kv"] = ctx["lv_kv"]
        # Each zone's LV tree is rooted at its own transformer and sized against its own
        # slack, so the sizing depends only on the zone's customer set (and the catalogue /
        # constraints, fixed for the run) -- never on k or on the neighbouring zones.
        cache_key = frozenset(zone["customer_ids"])
        cached = sizing_cache.get(cache_key)
        if cached is None:
            zsizing = _size_flat_lv(ctx, ztopo)
            cached = {"conductors": dict(zsizing.conductor_by_line),
                      "bus_by_pole": dict(zsizing.model.bus_by_pole)}
            sizing_cache[cache_key] = cached
        # Set on the HIT path too. The cache hands back the same LVTopology instance to every
        # attempt and to both the Case C and Case D plans, so assigning bus_by_pole only on a
        # miss left it holding whatever the last build wrote -- correct today only because
        # nothing reads it for zone candidates (detect_mvlv_subtree_candidates passes
        # full_model=None), which is an invariant nothing states.
        ztopo.bus_by_pole = cached["bus_by_pole"]
        for lid, c in cached["conductors"].items():
            zone_cond[f"{zone['zone_id']}::{lid}"] = c
        transformers_chosen.append({**zone["transformer_spec"], "zone_id": zone["zone_id"], "transformer_id": zone["transformer_id"]})

    transformer_by_zone = {z["zone_id"]: z["transformer_spec"] for z in plan.zones}
    final_model = final_outcome = mv_cond = None
    # Positional order IS rank order: normalize_conductors sorts by `rank` and resets the
    # index, so iloc[i] is the i-th cheapest. Sweep cheapest-first until the network passes.
    for rank_i in range(len(ctx["mv_conductors"])):
        mv_cond = ctx["mv_conductors"].iloc[rank_i].to_dict()
        final_model = _mvlv_net(ctx, plan, zone_cond, mv_cond, transformer_by_zone, stepup_spec,
                                apply_growth=True, coincidence=ctx["coincidence"])
        final_outcome = run_powerflow(final_model, constraints)
        if final_outcome.feasible:
            break

    built = {"plan": plan, "zone_cond": zone_cond, "transformers": transformers_chosen,
             "transformer_by_zone": transformer_by_zone, "mv_cond": mv_cond,
             "model": final_model, "outcome": final_outcome,
             "stress_model": None, "stress_outcome": None}
    if with_stress:
        built.update(_mvlv_stress(ctx, built, stepup_spec))
    return built


def _mvlv_stress(ctx, built: Dict[str, Any], stepup_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Coincident-peak snapshot: the same network with every customer at horizon peak at once.

    Kept separate from :func:`_build_mvlv_at` because it costs an extra power flow per
    partition and is only needed for the candidate that is finally reported -- unless
    ``k_selection_snapshot="coincident"``, in which case the zone-count search selects on it
    and every attempt needs it. That option used to raise ``KeyError: 'stress_outcome'``: the
    search read a key only added one level up, after the search had already finished.
    """
    model = _mvlv_net(
        ctx, built["plan"], built["zone_cond"], built["mv_cond"], built["transformer_by_zone"],
        stepup_spec, apply_growth=True, coincidence=1.0,
    )
    return {"stress_model": model, "stress_outcome": run_powerflow(model, ctx["constraints"])}


def _voltage_cap_search(ctx, grid_ids, crs_epsg, max_kva, max_transformers, stepup_spec, seed):
    """Search the number of zones for the smallest count that holds the LV voltage target.

    Three properties, each fixing a specific defect in the previous search:

    * **The partition family is nested.** ``partition_zones`` cuts one Ward dendrogram at k
      (:func:`~core.topology.mvlv_service_zones.partition_hierarchical_k`), so k+1 refines
      k instead of reshuffling it. With the old k-means-per-k the worst voltage drop was not
      even monotone in k (measured on Mafa: k=4 worse than k=3), which is incompatible with
      a search that stops at the first failure.

    * **Acceptance keeps a declared margin.** A partition is *accepted* only at
      ``min_vm_pu >= voltage_min_pu + design_voltage_margin_pu``. Minimising k against the
      bare limit spends every last volt of headroom, leaving the chosen design sitting on
      the constraint (0.913 pu against a 0.90 limit, measured), so any error in the
      diversity assumption or the load estimate flips it infeasible.

      The margin is a **preference, not a constraint**: adding transformers shortens LV
      feeders but lengthens the MV trunk and adds another transformer impedance, so the
      achievable voltage *plateaus* -- on Mafa it saturates near 0.905 pu and no zone count
      reaches a 0.92 target. When the margin is unreachable the search falls back to the
      smallest merely-feasible count and says so, rather than climbing to the ceiling and
      returning the most expensive (and possibly infeasible) partition.

    * **The climb is bounded by geometry and by progress.** k never exceeds the customer
      count (the old loop was bounded only by ``max_transformers`` and raised out of the
      partitioner for a settlement smaller than that cap), and the climb stops once extra
      transformers stop buying voltage.

    Candidate feasibility itself is unchanged -- the margin governs which k the *designer*
    picks, not what counts as a feasible network.
    """
    T = ctx["T"]
    params = ctx["params"]
    coords, _ = _coords_and_ids(grid_ids, ctx["gdf"])
    iterations: List[Dict[str, Any]] = []

    # Never ask for more zones than there are customers to put in them.
    k_ceiling = max(1, min(int(max_transformers), len(grid_ids)))

    # Hard bound on SEARCH EFFORT, not just on the k range. Each attempt is a full plan
    # build, a per-zone sizing loop and an MV rank sweep, and building a pandapower net is
    # O(n^2) in elements (every bus/line/load is an individual create_* call doing an O(n)
    # pandas setitem). Bounding only k lets a case where the voltage margin is unreachable,
    # and where nothing is yet feasible for the plateau test to latch onto, walk the whole
    # range: measured at >20 minutes for a 240-customer settlement. The fallback selection
    # already copes with finding nothing, so truncating costs at worst a larger zone count
    # than the true optimum -- and the truncation is recorded in the search history.
    max_attempts = max(1, int(params.get("max_zone_count_attempts", 8) or 8))

    v_min = float(params["voltage_min_pu"])
    margin = max(0.0, float(params.get("design_voltage_margin_pu", 0.0) or 0.0))
    v_target = v_min + margin
    on_coincident = str(params.get("k_selection_snapshot", "design")) == "coincident"

    built_by_k: Dict[int, Dict[str, Any]] = {}
    attempts_made = 0

    def attempt(n_zones: int) -> Optional[Dict[str, Any]]:
        """Build at ``n_zones``; None when that count cannot be built or the budget is spent."""
        nonlocal attempts_made
        if attempts_made >= max_attempts:
            iterations.append(_k_search_stop(
                n_zones,
                f"stopped searching: reached the {max_attempts}-attempt budget "
                f"(max_zone_count_attempts)",
            ))
            return None
        try:
            groups = partition_zones(
                "voltage_cap", grid_ids, ctx["gdf"], ctx["source"], crs_epsg,
                customers_df=ctx["customers"], horizon_years=T, max_zone_kva=max_kva,
                max_zone_radius_m=float(params["max_lv_zone_radius_m"]),
                max_zone_customers=int(params["max_zone_customers"]),
                n_zones=int(n_zones), seed=seed,
                coincidence_factor=ctx["tx_coincidence"],
            )
            attempts_made += 1
            built = _build_mvlv_at(ctx, groups, stepup_spec, with_stress=on_coincident)
        except ValueError as exc:
            # Too few distinct locations for this many zones, or a degenerate geometry.
            iterations.append(_k_search_failure(n_zones, "unbuildable", str(exc)))
            return None
        built_by_k.setdefault(len(built["plan"].zones), built)
        return built

    def selection_outcome(built):
        """The snapshot the ACCEPTANCE test reads (the design one unless selecting on stress)."""
        return built["stress_outcome"] if on_coincident else built["outcome"]

    def is_feasible(built) -> bool:
        """Feasible in the candidate's own terms (design snapshot within every limit)."""
        o = built["outcome"]
        return o is not None and o.feasible

    def accepts(built) -> bool:
        """Feasible AND clear of the voltage target on the selection snapshot."""
        if not is_feasible(built):
            return False
        o = selection_outcome(built)
        if o is None or not o.converged:
            return False
        return float(o.min_vm_pu) >= v_target

    def record(built, note):
        iterations.append(_iter_record(
            len(built["plan"].zones), built["outcome"], note,
            accepted=accepts(built), v_target=v_target,
        ))

    # Seed from geometry only. Warm-starting Case D from Case C's accepted count is
    # tempting (it cut ~25% off the runtime) but wrong: the seed is only ever climbed FROM,
    # so seeding D at C's k=10 meant D never tried a smaller count, and a case that is
    # feasible at k=5 was reported infeasible. Cost is not worth a missed solution.
    k_seed = max(1, min(int(suggest_zone_start(coords, float(params["seed_diameter_m"]))), k_ceiling))
    seeded = attempt(k_seed)
    if seeded is None:
        raise ValueError(
            f"Could not build an MV/LV partition at any zone count up to {k_ceiling} for "
            f"{len(grid_ids)} customers (seed k={k_seed}). Check for duplicate customer "
            f"coordinates or relax 'max_transformers'."
        )
    record(seeded, "seed")

    if accepts(seeded):
        # Backtrack toward fewer zones while the target still holds.
        target = len(seeded["plan"].zones) - 1
        while target >= 1:
            cand = attempt(target)
            if cand is None:
                break
            record(cand, "backtrack")
            if not accepts(cand):
                break
            target = len(cand["plan"].zones) - 1
    else:
        # Climb toward more zones until the target is met, the ceiling is hit, or -- once a
        # feasible partition is already in hand -- extra transformers stop buying voltage.
        #
        # The plateau test is gated on having something feasible for a measured reason: on
        # the Mafa case the worst voltage sits flat at 0.884 pu for k=4..7 and then JUMPS to
        # 0.905 at k=10, when a split finally relieves the one binding feeder. Flat is not
        # the same as saturated, so stopping on flatness alone would discard the only
        # feasible designs. The plateau break is a cost optimisation and must never cost
        # feasibility.
        vm_since_feasible: List[float] = []
        target = len(seeded["plan"].zones) + 1
        while target <= k_ceiling:
            cand = attempt(target)
            if cand is None:
                break
            n = len(cand["plan"].zones)
            record(cand, "climb")
            if accepts(cand):
                break
            o = cand["outcome"]
            if any(is_feasible(b) for b in built_by_k.values()):
                vm_since_feasible.append(
                    float(o.min_vm_pu) if (o is not None and o.converged) else float("nan")
                )
                if _voltage_has_plateaued(vm_since_feasible):
                    iterations.append(_k_search_stop(
                        n,
                        "stopped climbing: extra transformers no longer improve the worst "
                        "voltage (bottleneck is the MV trunk / step-up, not the LV zones)",
                    ))
                    break
            if o is None or not o.converged:
                # Divergence means grossly too few zones, not "one short": stepping by one
                # pays for a full build per step through a prefix that cannot converge
                # (k=1,2,3 on Mafa). Double instead, then refine by one once it solves.
                target = min(k_ceiling, max(n + 1, n * 2))
            else:
                target = max(n + 1, target + 1)  # guard: a cap-split can stall the climb

    # Selection: smallest accepted count; else smallest merely-feasible count; else the
    # attempt that came closest to being usable (for the infeasible-candidate report).
    accepted_ks = sorted(k for k, b in built_by_k.items() if accepts(b))
    feasible_ks = sorted(k for k, b in built_by_k.items() if is_feasible(b))
    if accepted_ks:
        chosen_k = accepted_ks[0]
        note = f"smallest zone count holding v_min >= {v_target:.3f} pu"
    elif feasible_ks:
        chosen_k = feasible_ks[0]
        solved = [selection_outcome(b) for b in built_by_k.values()]
        best_vm = max((float(o.min_vm_pu) for o in solved if o is not None and o.converged),
                      default=float("nan"))
        note = (
            f"design voltage margin of {margin:.3f} pu unreachable at any zone count "
            f"(best {best_vm:.3f} pu); fell back to the smallest feasible count"
        )
    else:
        chosen_k = min(
            built_by_k,
            key=lambda k: violation_severity(built_by_k[k]["outcome"], ctx["constraints"]),
        )
        note = "no zone count is feasible; reporting the least-violating partition"
    best = built_by_k[chosen_k]
    best["selection_note"] = note
    return best, iterations


def _voltage_has_plateaued(vm_seen: List[float], *, patience: int = 3, eps: float = 2e-3) -> bool:
    """True when the last ``patience`` zone counts stopped improving the worst voltage.

    Adding a transformer shortens the LV feeders inside the zone it splits but lengthens
    the MV trunk and inserts another transformer impedance, so the achievable minimum
    voltage saturates. Past that point the climb is buying nothing and paying for a
    transformer, an MV spur and its losses each step, so it stops.
    """
    finite = [v for v in vm_seen if v == v]  # drop NaN (non-converged attempts)
    if len(finite) <= patience:
        return False
    best_before = max(finite[:-patience])
    return max(finite[-patience:]) <= best_before + eps


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
    # Divide by the sizing safety factor so a zone sitting exactly on the cap is still
    # coverable by the largest catalogue transformer (select_transformer targets 1.2*S).
    # This is the SAME margin, applied once -- not a second, compounding one.
    max_kva = max_derated / TRANSFORMER_SIZING_SAFETY_FACTOR
    criterion = str(ctx["params"].get("clustering_criterion", "voltage_cap"))
    max_transformers = int(ctx["params"].get("max_transformers", 12))
    seed = 42

    # Plant step-up sized on the whole settlement's coincident apparent power (may exceed the
    # catalogue; it is a single project-procured unit, unlike the kVA-capped zone step-downs).
    s_system_kva = zone_apparent_power_kva(
        list(grid_ids), ctx["customers"], T, ctx["tx_coincidence"]
    )
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
            seed=seed, coincidence_factor=ctx["tx_coincidence"],
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
    plan.selection_note = best.get("selection_note", "")

    # Coincident-peak stress, reported alongside the design outcome and never gating it. The
    # search already solved it when selecting on it; otherwise solve it once, for the winner.
    if best.get("stress_outcome") is None:
        best.update(_mvlv_stress(ctx, best, stepup_spec))
    return {**best, "stepup": stepup_spec, "iterations": iterations}


def _run_mvlv(ctx, grid_ids: List[str], *, cid: str = "C_MVLV", architecture: str = "MV/LV",
              sa_npc: float = 0.0, sa_customers: Optional[List[str]] = None,
              sa_assignments: Optional[Dict[str, Dict[str, Any]]] = None) -> CaseResult:
    """Build an MV/LV zoned candidate over ``grid_ids``.

    Shared by Case C (all customers on the grid) and Case D (the Case-B stand-alone customers
    removed first, then the remainder MV/LV-zoned). Pass the stand-alone NPC/customers/assignments
    to fold them into the candidate.
    """
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
         getattr(plan, "selection_note", "") or f"{criterion} partition",
         "n_zones", f"{len(plan.zones)} over {len(grid_ids)} customers")
    for it in built["iterations"]:
        if it["note"] in ("search_stopped", "unbuildable"):
            verdict = it["note"]
        elif not it["converged"]:
            verdict = "diverged"
        elif it["accepted"]:
            verdict = "accepted"
        elif it["feasible"]:
            # Within the limits but not clear of the design voltage margin: a partition
            # sitting on the constraint is not what the search should settle for.
            verdict = "feasible_no_margin"
        else:
            verdict = "infeasible"
        _log(ctx["log"], "mv_transformer_count", cid, f"k={it['n_zones']}", verdict,
             it["bottleneck"], "min_vm_pu",
             f"{it['min_vm_pu']:.4f} (target {it['v_target_pu']:.4f})", threshold=it["note"])
    _log(ctx["log"], "transformer_sizing", cid, "stepup", f"plant step-up {stepup['sn_kva']:.0f} kVA",
         "system coincident S × margin, may exceed catalogue", "sn_kva", f"{stepup['sn_kva']}")
    for tx in transformers_chosen:
        _log(ctx["log"], "transformer_sizing", cid, tx["zone_id"], f"select {tx['transformer_id']}",
             "smallest rating covering horizon S", "sn_kva", f"{tx['sn_kva']}")
    _log(ctx["log"], "conductor_sizing", cid, "mv_trunk", "pass" if final_outcome.feasible else "fail",
         final_outcome.active_bottleneck, "mv_conductor", str((mv_cond or {}).get("conductor_id")))

    # Reuse the stored transformer_by_zone rather than re-deriving it: if zone transformer
    # selection ever stops being a pure function of plan.zones, re-deriving would build the
    # "now" snapshot against different transformers than the horizon one beside it.
    now_model = _mvlv_net(ctx, plan, zone_cond, mv_cond, built["transformer_by_zone"], stepup,
                          apply_growth=False, coincidence=ctx["coincidence"])
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
    service_drop_total = sum(z["topology"].total_service_drop_m for z in plan.zones)
    grid_served_ids = [c for z in plan.zones for c in z["topology"].served_customer_ids]
    n_grid = len(grid_served_ids)
    cost.connection_metering_cost = cm.connection_capex(n_grid, ctx["params"])
    cost.service_drop_cost = _service_drop_cost(ctx, service_drop_total)
    cost.loss_pv_cost = _loss_npc(ctx, final_outcome.losses_kw, grid_served_ids)
    cost.generation_npc = _generation_npc(ctx, grid_served_ids)
    cost.standalone_npc = float(sa_npc)

    assign: Dict[str, Dict[str, Any]] = {}
    for zone in plan.zones:
        for c in zone["topology"].served_customer_ids:
            assign[c] = _assignment("grid_now", architecture, cluster_id=zone["zone_id"],
                                    transformer_id=zone["transformer_id"],
                                    reason="MV/LV service zone")
    assign.update(sa_assignments)

    return CaseResult(
        candidate_id=cid, architecture=architecture, feasible=final_outcome.feasible, cost=cost,
        outcome_now=now, outcome_horizon=final_outcome, outcome_stress=stress_outcome,
        n_grid_customers=n_grid, n_standalone_customers=len(sa_customers),
        lv_length_m=sum(lv_len_by_line.values()), mv_length_m=plan.mv_total_length_m,
        n_poles=n_poles, service_drop_total_m=service_drop_total, n_transformers=len(transformers_chosen),
        topology=None, plan=plan, model_now=now_model, model=final_model, model_stress=stress_model,
        conductor_by_line=zone_cond,
        grid_customer_ids=list(grid_served_ids),
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
    largest_lv_id = cm.largest_lv_conductor_id(ctx["lv_conductors"])
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


def _run_mvlv_standalone(ctx, full_res: CaseResult, selection: StandaloneSelection) -> CaseResult:
    """Case D: apply the MV/LV-specific subtree stand-alone screen, then rebuild MV/LV over the rest.

    The stand-alone set comes from :func:`detect_mvlv_subtree_candidates` (per-zone terminal
    subtrees), not from Case B's LV screen, so D never reuses B's SHS customers.
    """
    cid = "D_MVLV_plus_standalone"
    res = _run_mvlv(
        ctx, _excluding(full_res.grid_customer_ids, selection.customer_ids),
        cid=cid, architecture="MV/LV + Stand-alone",
        sa_npc=selection.npc, sa_customers=selection.customer_ids,
        sa_assignments=selection.assignments,
    )
    if not selection.customer_ids:
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
    """One row per candidate, one column per cost component.

    Keyed off ``cm.COMPONENTS`` so a component added to ``CostBreakdown`` reaches this
    table automatically; it used to be a hand-written list that had to be edited in step.
    """
    c = res.cost
    return {
        "candidate_id": res.candidate_id,
        **{name: round(float(getattr(c, name)), 0) for name in cm.COMPONENTS},
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
              "clustering_criterion", "max_cluster_diameter_m", "max_transformers", "source_vm_pu",
              "design_voltage_margin_pu", "k_selection_snapshot",
              "transformer_sizing_coincidence_factor", "standalone_sizing_tolerance",
              "max_zone_count_attempts"):
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
    ctx = build_context(
        customers=customers, params=params, lv_conductors=lv_conductors,
        packages=standalone_packages, horizon_years=T, discount_rate=float(params["discount_rate"]),
        gdf=gdf, roads=roads, crs_meta=crs_meta, source=source, eng=eng, constraints=constraints,
        mv_conductors=mv_conductors, transformers=transformers,
        lv_kv=float(source.get("lv_voltage_kv", 0.4)),
        mv_kv=float(source.get("mv_voltage_kv", 11.0)),
        source_vm_pu=float(params["source_vm_pu"]),
    )
    log = ctx["log"]

    emit("Generating full LV topology and running LV conductor sizing...")
    full_res = _run_full_lv(ctx)

    emit("Building MV/LV service-zone case...")
    mvlv_res = _run_mvlv(ctx, full_res.grid_customer_ids)

    # Case B: LV screen over the terminal subtrees (leaf branches) of the Full-LV tree.
    emit("Generating Full-LV terminal subtrees (leaf branches)...")
    largest_lv_id = cm.largest_lv_conductor_id(lv_conductors)
    lv_clusters = detect_lv_subtrees(
        full_res.topology, customers, conductor_by_line=full_res.conductor_by_line,
        full_model=full_res.model, cost_params=params, horizon_years=T, largest_conductor_id=largest_lv_id,
    )
    for cl in lv_clusters:
        _log(log, "branch_detection", "A_full_LV", cl.cluster_id,
             "flagged" if cl.is_problematic else "clean", "; ".join(cl.reasons) or "within limits",
             "n_customers", cl.n_customers)

    emit("Building LV + stand-alone case (LV leaf-branch SHS screen)...")
    lv_sa = _select_standalone_clusters(ctx, lv_clusters, _avoided_lv_cost, "LV + Stand-alone", "B_LV_plus_standalone")
    sa_res = _run_lv_standalone(ctx, full_res, lv_sa)

    # Case D: MV/LV screen over the terminal subtrees inside each zone's LV tree (own SHS set, not B's).
    emit("Building MV/LV + stand-alone case (per-zone leaf-branch SHS screen)...")
    mvlv_clusters = detect_mvlv_subtree_candidates(mvlv_res, ctx)
    for cl in mvlv_clusters:
        _log(log, "mvlv_subtree", "C_MVLV", cl.cluster_id,
             "flagged" if cl.is_problematic else "clean", "; ".join(cl.reasons) or "within limits",
             "n_customers", cl.n_customers)
    mvlv_sa = _select_standalone_clusters(ctx, mvlv_clusters, _avoided_mvlv_subtree_cost, "MV/LV + Stand-alone", "D_MVLV_plus_standalone")
    mvlv_sa_res = _run_mvlv_standalone(ctx, full_res, mvlv_sa)

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
