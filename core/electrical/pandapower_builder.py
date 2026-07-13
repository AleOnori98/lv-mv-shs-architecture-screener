"""Build pandapower networks for the MVP cases (balanced 3-phase approximation).

Replaces the legacy PyPSA validation backend. Two builders are provided:
  * :func:`build_lv_net` — flat LV network (Full LV and LV+Stand-alone cases).
  * :func:`build_mvlv_net` — LV core plus MV trunk + transformer service zones.

Loads are aggregated to poles. Horizon demand applies per-customer growth:
``P_h = P_now * (1 + growth) ** T``; reactive power from each customer's power factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandapower as pp


@dataclass
class PPModel:
    net: Any
    bus_by_pole: Dict[int, int] = field(default_factory=dict)
    mv_bus_by_pole: Dict[int, int] = field(default_factory=dict)
    zone_bus_by_pole: Dict[str, Dict[int, int]] = field(default_factory=dict)
    line_index_by_id: Dict[str, int] = field(default_factory=dict)
    trafo_index_by_zone: Dict[str, int] = field(default_factory=dict)
    lv_line_ids: List[str] = field(default_factory=list)
    mv_line_ids: List[str] = field(default_factory=list)
    total_load_kw: float = 0.0


def customer_lookup(customers_df) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for _, r in customers_df.iterrows():
        out[str(r["customer_id"])] = {
            "peak_kw_now": float(r["peak_kw_now"]),
            "power_factor": float(r["power_factor"]),
            "growth_rate": float(r["growth_rate"]),
        }
    return out


def _pq_kw(cids, lookup, horizon_years: int, apply_growth: bool, coincidence_factor: float = 1.0) -> tuple[float, float]:
    """Aggregate active/reactive power at a pole.

    ``coincidence_factor`` scales the summed individual peaks: 1.0 = every customer at their peak
    simultaneously (the coincident-peak stress case), < 1.0 = a diversified design peak.
    """
    cf = float(coincidence_factor)
    p_kw = 0.0
    q_kw = 0.0
    for cid in cids:
        c = lookup.get(str(cid))
        if c is None:
            continue
        p = c["peak_kw_now"] * ((1.0 + c["growth_rate"]) ** horizon_years if apply_growth else 1.0) * cf
        pf = min(max(c["power_factor"], 0.5), 1.0)
        p_kw += p
        q_kw += p * math.tan(math.acos(pf))
    return p_kw, q_kw


def _add_line(net, from_bus, to_bus, length_km, conductor, name) -> int:
    return pp.create_line_from_parameters(
        net,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=max(float(length_km), 1e-4),
        r_ohm_per_km=float(conductor["r_ohm_per_km"]),
        x_ohm_per_km=float(conductor["x_ohm_per_km"]),
        c_nf_per_km=0.0,
        max_i_ka=float(conductor["max_i_ka"]),
        name=name,
    )


def build_lv_net(
    topology,
    customers_df,
    conductor_by_line: Dict[str, Dict[str, Any]],
    *,
    lv_kv: float,
    horizon_years: int,
    apply_growth: bool,
    coincidence_factor: float = 1.0,
) -> PPModel:
    """Flat LV pandapower network: a bus per pole, ext_grid at the slack pole."""
    net = pp.create_empty_network(name="LV")
    lookup = customer_lookup(customers_df)
    model = PPModel(net=net)

    for pid in topology.pole_xy:
        bus = pp.create_bus(net, vn_kv=float(lv_kv), name=f"pole_{int(pid)}")
        model.bus_by_pole[int(pid)] = bus

    pp.create_ext_grid(net, bus=model.bus_by_pole[topology.slack_pole_id], vm_pu=1.0, name="source")

    edges = topology.edges_4326
    if edges is not None and not edges.empty:
        for _, e in edges.iterrows():
            lid = str(e["line_id"])
            cond = conductor_by_line[lid]
            idx = _add_line(net, model.bus_by_pole[int(e["u"])], model.bus_by_pole[int(e["v"])], float(e["length_m"]) / 1000.0, cond, lid)
            model.line_index_by_id[lid] = idx
            model.lv_line_ids.append(lid)

    for pid, cids in topology.pole_customers.items():
        p_kw, q_kw = _pq_kw(cids, lookup, horizon_years, apply_growth, coincidence_factor)
        if p_kw <= 0:
            continue
        pp.create_load(net, bus=model.bus_by_pole[int(pid)], p_mw=p_kw / 1000.0, q_mvar=q_kw / 1000.0, name=f"load_{int(pid)}")
        model.total_load_kw += p_kw
    return model


def build_mvlv_net(
    plan,
    customers_df,
    *,
    core_conductor_by_line: Dict[str, Dict[str, Any]],
    zone_conductor_by_line: Dict[str, Dict[str, Any]],
    mv_conductor: Dict[str, Any],
    transformer_by_zone: Dict[str, Dict[str, Any]],
    lv_kv: float,
    mv_kv: float,
    horizon_years: int,
    apply_growth: bool,
    coincidence_factor: float = 1.0,
    source_vm_pu: float = 1.04,
) -> PPModel:
    """LV core fed from source + MV trunk to transformer service zones.

    ``plan`` is an :class:`core.topology.mvlv_service_zones.MVLVPlan`. The source/slack is set
    at ``source_vm_pu`` (a realistic high sending-end voltage) to offset the downstream MV-line
    and transformer voltage drop across the service zones.
    """
    net = pp.create_empty_network(name="MVLV")
    lookup = customer_lookup(customers_df)
    model = PPModel(net=net)

    # --- Core LV (fed directly from the source at LV) ---
    core = plan.core_topology
    core_bus_by_pole: Dict[int, int] = {}
    if core is not None:
        for pid in core.pole_xy:
            core_bus_by_pole[int(pid)] = pp.create_bus(net, vn_kv=float(lv_kv), name=f"core_pole_{int(pid)}")
        pp.create_ext_grid(net, bus=core_bus_by_pole[core.slack_pole_id], vm_pu=float(source_vm_pu), name="source")
        source_lv_bus = core_bus_by_pole[core.slack_pole_id]
        if core.edges_4326 is not None and not core.edges_4326.empty:
            for _, e in core.edges_4326.iterrows():
                lid = f"core::{e['line_id']}"
                cond = core_conductor_by_line[lid]
                idx = _add_line(net, core_bus_by_pole[int(e["u"])], core_bus_by_pole[int(e["v"])], float(e["length_m"]) / 1000.0, cond, lid)
                model.line_index_by_id[lid] = idx
                model.lv_line_ids.append(lid)
        for pid, cids in core.pole_customers.items():
            p_kw, q_kw = _pq_kw(cids, lookup, horizon_years, apply_growth, coincidence_factor)
            if p_kw > 0:
                pp.create_load(net, bus=core_bus_by_pole[int(pid)], p_mw=p_kw / 1000.0, q_mvar=q_kw / 1000.0)
                model.total_load_kw += p_kw
        model.bus_by_pole.update(core_bus_by_pole)
    else:
        # No core customers: still need a source MV bus reference.
        source_lv_bus = None

    # --- MV source bus ---
    mv_topology = getattr(plan, "mv_topology", None)
    if mv_topology is not None:
        for pid in mv_topology.pole_xy:
            model.mv_bus_by_pole[int(pid)] = pp.create_bus(net, vn_kv=float(mv_kv), name=f"mv_pole_{int(pid)}")
        mv_source_bus = model.mv_bus_by_pole[mv_topology.slack_pole_id]
        if mv_topology.edges_4326 is not None and not mv_topology.edges_4326.empty:
            for _, e in mv_topology.edges_4326.iterrows():
                mv_lid = f"mv::{e['line_id']}"
                idx = _add_line(
                    net,
                    model.mv_bus_by_pole[int(e["u"])],
                    model.mv_bus_by_pole[int(e["v"])],
                    float(e["length_m"]) / 1000.0,
                    mv_conductor,
                    mv_lid,
                )
                model.line_index_by_id[mv_lid] = idx
                model.mv_line_ids.append(mv_lid)
    else:
        mv_source_bus = pp.create_bus(net, vn_kv=float(mv_kv), name="mv_source")
    # The MV bus is fed from the same source point (modelled as a slack at the sending voltage).
    # When there is no LV core, this is the only slack; otherwise it is a second reference at the
    # same source location (no explicit step-up transformer is modelled — an MVP simplification).
    pp.create_ext_grid(net, bus=mv_source_bus, vm_pu=float(source_vm_pu),
                       name="source" if source_lv_bus is None else "mv_source_slack")

    # --- Service zones: MV line -> transformer -> local LV tree ---
    for zone in plan.zones:
        zid = zone["zone_id"]
        tx = transformer_by_zone[zid]
        hv_bus = pp.create_bus(net, vn_kv=float(mv_kv), name=f"{zid}_hv")
        lv_bus = pp.create_bus(net, vn_kv=float(lv_kv), name=f"{zid}_lv")
        if mv_topology is not None and zone.get("mv_pole_id") is not None:
            mv_lid = f"mv_drop::{zid}"
            idx = _add_line(
                net,
                model.mv_bus_by_pole[int(zone["mv_pole_id"])],
                hv_bus,
                float(zone.get("mv_service_drop_m", 0.0)) / 1000.0,
                mv_conductor,
                mv_lid,
            )
        else:
            mv_lid = f"mv::{zid}"
            idx = _add_line(net, mv_source_bus, hv_bus, float(zone["mv_length_m"]) / 1000.0, mv_conductor, mv_lid)
        model.line_index_by_id[mv_lid] = idx
        model.mv_line_ids.append(mv_lid)
        # Transformer.
        t_idx = pp.create_transformer_from_parameters(
            net,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            sn_mva=float(tx["sn_kva"]) / 1000.0,
            vn_hv_kv=float(tx["vn_hv_kv"]),
            vn_lv_kv=float(tx["vn_lv_kv"]),
            vk_percent=float(tx["vk_percent"]),
            vkr_percent=float(tx["vkr_percent"]),
            pfe_kw=float(tx["pfe_kw"]),
            i0_percent=float(tx["i0_percent"]),
            name=f"{zid}_tx",
        )
        model.trafo_index_by_zone[zid] = t_idx

        ztopo = zone["topology"]
        zbus_by_pole: Dict[int, int] = {}
        for pid in ztopo.pole_xy:
            zbus_by_pole[int(pid)] = pp.create_bus(net, vn_kv=float(lv_kv), name=f"{zid}_pole_{int(pid)}")
        model.zone_bus_by_pole[zid] = zbus_by_pole
        # Connect transformer LV bus to the zone slack pole with a near-zero bus tie.
        pp.create_switch(net, bus=lv_bus, element=zbus_by_pole[ztopo.slack_pole_id], et="b", closed=True)
        if ztopo.edges_4326 is not None and not ztopo.edges_4326.empty:
            for _, e in ztopo.edges_4326.iterrows():
                lid = f"{zid}::{e['line_id']}"
                cond = zone_conductor_by_line[lid]
                li = _add_line(net, zbus_by_pole[int(e["u"])], zbus_by_pole[int(e["v"])], float(e["length_m"]) / 1000.0, cond, lid)
                model.line_index_by_id[lid] = li
                model.lv_line_ids.append(lid)
        for pid, cids in ztopo.pole_customers.items():
            p_kw, q_kw = _pq_kw(cids, lookup, horizon_years, apply_growth, coincidence_factor)
            if p_kw > 0:
                pp.create_load(net, bus=zbus_by_pole[int(pid)], p_mw=p_kw / 1000.0, q_mvar=q_kw / 1000.0)
                model.total_load_kw += p_kw

    return model
