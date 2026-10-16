"""Build SHS candidate groups from the *terminal subtrees* of a radial LV tree.

Candidates are the leaf-side subtrees produced by :func:`terminal_subtrees` — each maximal
branch that detaches at a fork — rather than threshold-defined "remote clusters". The same
builder serves the Full-LV tree (Case B) and each MV/LV zone's LV tree (Case D). Engineering
thresholds (``dmax``, ``lmax``, branch cost/customer, undervoltage, largest-conductor use) are
computed here only as **diagnostic** ``reasons``; they no longer gate the SHS decision, which is
made purely on economics in the workflow (SHS NPC vs avoided grid cost).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .subtree_candidates import terminal_subtrees


@dataclass
class Cluster:
    """One screened SHS candidate: a terminal subtree with its economics and diagnostics."""

    cluster_id: str
    customer_ids: List[str]
    pole_ids: List[int]
    branch_edge_ids: List[str]
    branch_root: int
    boundary_node: int
    n_customers: int
    peak_kw_now: float
    annual_kwh_now: float
    branch_cost: float
    branch_cost_per_customer: float
    branch_losses_kw: float
    min_voltage: float
    uses_largest_conductor: bool
    standalone_eligible_precheck: bool
    centroid_xy: tuple
    is_problematic: bool = False
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def subtree_clusters(
    topology,
    customers_df: pd.DataFrame,
    *,
    conductor_by_line: Dict[str, Dict[str, Any]],
    cost_params: Dict[str, float],
    horizon_years: int,
    largest_conductor_id: str,
    full_model=None,
    bus_by_pole: Optional[Dict[int, int]] = None,
    id_prefix: str = "CL",
) -> List[Cluster]:
    """Scored SHS candidates = the terminal subtrees of ``topology`` (a radial LV tree).

    Reused by the Full-LV screen (Case B) and, per zone, by the MV/LV screen (Case D).
    Each subtree's branch cost includes its internal lines **and** its entry edge (both are
    avoided when the subtree leaves the grid). Pass a solved ``full_model`` (+ ``bus_by_pole``)
    to fill voltage / loading / branch-loss diagnostics; omit it for zone candidates, which are
    costed without per-zone power-flow slicing. ``id_prefix`` namespaces the cluster ids
    (e.g. ``"CL"`` for LV, ``"Z03-S"`` for a zone).
    """
    g = topology.graph
    root = topology.slack_pole_id
    by_id = customers_df.set_index("customer_id")
    mult = float(cost_params["civil_logistics_multiplier"])
    edge_len_by_lid = {str(d.get("line_id")): float(d["length_m"]) for *_uv, d in g.edges(data=True)}

    res_bus = full_model.net.res_bus if full_model is not None else None
    res_line = full_model.net.res_line if full_model is not None else None
    line_idx = full_model.line_index_by_id if full_model is not None else {}
    bus_by_pole = bus_by_pole or {}

    clusters: List[Cluster] = []
    for i, st in enumerate(terminal_subtrees(g, root)):
        poles = st.poles
        edge_ids = sorted(set(st.internal_edge_ids) | ({st.entry_edge_id} if st.entry_edge_id else set()))

        cust_ids = [cid for pid in poles for cid in topology.pole_customers.get(pid, [])]
        if not cust_ids:
            continue
        rows = by_id.loc[[c for c in cust_ids if c in by_id.index]]

        branch_cost = sum(
            edge_len_by_lid.get(lid, 0.0) * float(conductor_by_line[lid]["cost_per_m"]) * mult
            for lid in edge_ids if lid in conductor_by_line
        )
        cost_per_cust = branch_cost / max(len(cust_ids), 1)
        service_drop_m = sum(float(topology.service_drop_m.get(c, 0.0)) for c in cust_ids)

        min_v, branch_losses_kw = float("nan"), 0.0
        if res_bus is not None and len(res_bus):
            buses = [bus_by_pole[p] for p in poles if p in bus_by_pole]
            if buses:
                min_v = float(res_bus.loc[buses, "vm_pu"].min())
        if res_line is not None and len(res_line):
            idxs = [line_idx[lid] for lid in edge_ids if lid in line_idx]
            if idxs:
                branch_losses_kw = max(0.0, float(res_line.loc[idxs, "pl_mw"].sum()) * 1000.0)

        uses_largest = any(str((conductor_by_line.get(lid) or {}).get("conductor_id")) == str(largest_conductor_id) for lid in edge_ids)
        elig = bool(len(rows) and (~rows["must_connect_flag"]).all() and rows["can_be_standalone_flag"].all())
        cxy = (
            sum(topology.pole_xy[p][0] for p in poles) / len(poles),
            sum(topology.pole_xy[p][1] for p in poles) / len(poles),
        )

        cluster = Cluster(
            cluster_id=f"{id_prefix}{i:02d}", customer_ids=sorted(cust_ids), pole_ids=sorted(poles),
            branch_edge_ids=edge_ids, branch_root=int(st.branch_root), boundary_node=int(st.boundary_node),
            n_customers=len(cust_ids), peak_kw_now=float(rows["peak_kw_now"].sum()),
            annual_kwh_now=float(rows["annual_kwh_now"].sum()),
            branch_cost=branch_cost, branch_cost_per_customer=cost_per_cust,
            branch_losses_kw=branch_losses_kw, min_voltage=min_v,
            uses_largest_conductor=uses_largest, standalone_eligible_precheck=elig, centroid_xy=cxy,
            metadata={"service_drop_m": service_drop_m},
        )
        _score(cluster, cost_params)
        clusters.append(cluster)
    return clusters


def detect_lv_subtrees(
    topology,
    customers_df: pd.DataFrame,
    *,
    conductor_by_line: Dict[str, Dict[str, Any]],
    full_model,
    cost_params: Dict[str, float],
    horizon_years: int,
    largest_conductor_id: str,
) -> List[Cluster]:
    """Terminal-subtree SHS candidates for the Full-LV tree (Case B)."""
    return subtree_clusters(
        topology, customers_df, conductor_by_line=conductor_by_line, cost_params=cost_params,
        horizon_years=horizon_years, largest_conductor_id=largest_conductor_id,
        full_model=full_model, bus_by_pole=getattr(topology, "bus_by_pole", {}), id_prefix="CL",
    )


def _score(cluster: Cluster, cost_params: Dict[str, float]) -> None:
    """Annotate diagnostic flags (reporting only, never gates): high branch cost/customer,
    undervoltage and largest-conductor use. Distance/length (dmax/lmax) flags were removed."""
    reasons: List[str] = []
    if cluster.branch_cost_per_customer > float(cost_params["branch_cost_per_customer_threshold"]):
        reasons.append(f"branch cost/customer {cluster.branch_cost_per_customer:.0f} > {cost_params['branch_cost_per_customer_threshold']:.0f}")
    if cluster.min_voltage == cluster.min_voltage and cluster.min_voltage < float(cost_params["voltage_min_pu"]):
        reasons.append(f"undervoltage {cluster.min_voltage:.3f} pu")
    if cluster.uses_largest_conductor:
        reasons.append("uses largest LV conductor")
    cluster.reasons = reasons
    cluster.is_problematic = bool(reasons)
