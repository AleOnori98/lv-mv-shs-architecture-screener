"""Build pandapower networks for the MVP cases (balanced 3-phase approximation).

Two builders:
  * :func:`build_lv_net` — flat LV network (Full LV and LV+Stand-alone cases).
  * :func:`build_mvlv_net` — MV trunk feeding transformer service zones.

Loads are aggregated to poles. Horizon demand applies per-customer growth:
``P_h = P_now * (1 + growth) ** T``; reactive power from each customer's power factor.

**Elements are created in bulk.** Each ``pp.create_*`` call appends a row to a pandas
table, which is O(n) in the rows already there, so creating elements one at a time makes a
build O(n^2) -- and building networks, not solving them, is where this tool spends its
time (measured: 62% of wall clock in these two functions, 3% in ``runpp``). The plural
``create_buses`` / ``create_lines_from_parameters`` / ``create_loads`` APIs write one
table each instead, which is flat in n (measured 7x at 150 elements, 28x at 1800) and
produces a bit-identical network: same insertion order, so the ``*_by_pole`` and
``line_index_by_id`` maps are unchanged, and identical power-flow results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandapower as pp


@dataclass
class PPModel:
    net: Any
    bus_by_pole: Dict[int, int] = field(default_factory=dict)
    mv_bus_by_pole: Dict[int, int] = field(default_factory=dict)
    zone_bus_by_pole: Dict[str, Dict[int, int]] = field(default_factory=dict)
    line_index_by_id: Dict[str, int] = field(default_factory=dict)
    trafo_index_by_zone: Dict[str, int] = field(default_factory=dict)
    stepup_trafo_index: Optional[int] = None
    lv_line_ids: List[str] = field(default_factory=list)
    mv_line_ids: List[str] = field(default_factory=list)
    total_load_kw: float = 0.0


def customer_lookup(customers_df) -> Dict[str, Tuple[float, float, float]]:
    """``customer_id -> (peak_kw_now, growth_rate, tan(phi))``.

    ``tan(acos(pf))`` is precomputed here because it is invariant for the whole run while
    :func:`_pq_kw` runs once per pole per build -- it was thousands of trig calls per run.
    The lookup is also invariant, so callers should build it once and pass it in.
    """
    return {
        str(cid): (float(p), float(g), math.tan(math.acos(min(max(float(pf), 0.5), 1.0))))
        for cid, p, pf, g in zip(
            customers_df["customer_id"], customers_df["peak_kw_now"],
            customers_df["power_factor"], customers_df["growth_rate"],
        )
    }


def _pq_kw(cids, lookup, horizon_years: int, apply_growth: bool, coincidence_factor: float = 1.0) -> Tuple[float, float]:
    """Aggregate active/reactive power at a pole [kW, kVAr].

    ``coincidence_factor`` scales the summed individual peaks: 1.0 = every customer at their
    peak simultaneously (the coincident-peak stress case), < 1.0 = a diversified design peak.
    """
    cf = float(coincidence_factor)
    p_kw = 0.0
    q_kw = 0.0
    for cid in cids:
        c = lookup.get(str(cid))
        if c is None:
            continue
        peak_now, growth, tan_phi = c
        p = peak_now * ((1.0 + growth) ** horizon_years if apply_growth else 1.0) * cf
        p_kw += p
        q_kw += p * tan_phi
    return p_kw, q_kw


class _LineBatch:
    """Collects line parameters so a whole network's lines land in one table write.

    Every line still carries its own conductor; batching only changes how the rows reach
    pandapower. ``flush`` returns nothing and records the assigned indices on the model, so
    ``line_index_by_id`` keeps mapping each ``line_id`` to its own row.
    """

    __slots__ = ("from_bus", "to_bus", "length_km", "r", "x", "max_i", "names")

    def __init__(self) -> None:
        self.from_bus: List[int] = []
        self.to_bus: List[int] = []
        self.length_km: List[float] = []
        self.r: List[float] = []
        self.x: List[float] = []
        self.max_i: List[float] = []
        self.names: List[str] = []

    def add(self, from_bus: int, to_bus: int, length_km: float, conductor: Dict[str, Any], name: str) -> None:
        self.from_bus.append(int(from_bus))
        self.to_bus.append(int(to_bus))
        # pandapower rejects a zero-length line; the floor matches the per-line builder.
        self.length_km.append(max(float(length_km), 1e-4))
        self.r.append(float(conductor["r_ohm_per_km"]))
        self.x.append(float(conductor["x_ohm_per_km"]))
        self.max_i.append(float(conductor["max_i_ka"]))
        self.names.append(str(name))

    def flush(self, net, model: "PPModel", *, mv: bool) -> None:
        if not self.names:
            return
        idx = pp.create_lines_from_parameters(
            net,
            from_buses=self.from_bus,
            to_buses=self.to_bus,
            length_km=self.length_km,
            r_ohm_per_km=self.r,
            x_ohm_per_km=self.x,
            c_nf_per_km=0.0,          # shunt capacitance neglected
            max_i_ka=self.max_i,
            name=self.names,
        )
        for lid, i in zip(self.names, idx):
            model.line_index_by_id[lid] = int(i)
        (model.mv_line_ids if mv else model.lv_line_ids).extend(self.names)


class _LoadBatch:
    """Collects pole loads so a whole network's loads land in one table write."""

    __slots__ = ("buses", "p_mw", "q_mvar", "names", "total_kw")

    def __init__(self) -> None:
        self.buses: List[int] = []
        self.p_mw: List[float] = []
        self.q_mvar: List[float] = []
        self.names: List[str] = []
        self.total_kw = 0.0

    def add(self, bus: int, p_kw: float, q_kw: float, name: Optional[str] = None) -> None:
        # Poles with no (or non-positive) demand get no load element, as before.
        if p_kw <= 0:
            return
        self.buses.append(int(bus))
        self.p_mw.append(p_kw / 1000.0)
        self.q_mvar.append(q_kw / 1000.0)
        self.names.append(name or "")
        self.total_kw += p_kw

    def flush(self, net, model: "PPModel") -> None:
        if self.buses:
            pp.create_loads(net, buses=self.buses, p_mw=self.p_mw, q_mvar=self.q_mvar,
                            name=self.names)
        # total_load_kw divides into the reported losses_percent, so it must track the
        # loads actually created.
        model.total_load_kw += self.total_kw


def _create_buses(net, poles: Iterable[int], vn_kv: float, name_prefix: str) -> Dict[int, int]:
    """One bus per pole, in iteration order; returns ``pole_id -> bus index``."""
    pole_ids = [int(p) for p in poles]
    if not pole_ids:
        return {}
    idx = pp.create_buses(net, len(pole_ids), vn_kv=float(vn_kv),
                          name=[f"{name_prefix}{p}" for p in pole_ids])
    return {p: int(i) for p, i in zip(pole_ids, idx)}


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
    """Flat LV pandapower network: a bus per pole, ext_grid at the slack pole.

    The LV cases are fed from an ideal 0.4 kV slack at 1.0 pu with no plant transformer --
    unlike the MV/LV cases, which pay for an explicit step-up.
    """
    net = pp.create_empty_network(name="LV")
    lookup = customer_lookup(customers_df)
    model = PPModel(net=net)

    model.bus_by_pole = _create_buses(net, topology.pole_xy, lv_kv, "pole_")
    pp.create_ext_grid(net, bus=model.bus_by_pole[topology.slack_pole_id], vm_pu=1.0, name="source")

    lines = _LineBatch()
    edges = topology.edges_4326
    if edges is not None and not edges.empty:
        for lid, u, v, length_m in zip(edges["line_id"], edges["u"], edges["v"], edges["length_m"]):
            lid = str(lid)
            lines.add(model.bus_by_pole[int(u)], model.bus_by_pole[int(v)],
                      float(length_m) / 1000.0, conductor_by_line[lid], lid)
    lines.flush(net, model, mv=False)

    loads = _LoadBatch()
    for pid, cids in topology.pole_customers.items():
        p_kw, q_kw = _pq_kw(cids, lookup, horizon_years, apply_growth, coincidence_factor)
        loads.add(model.bus_by_pole[int(pid)], p_kw, q_kw, f"load_{int(pid)}")
    loads.flush(net, model)
    return model


def build_mvlv_net(
    plan,
    customers_df,
    *,
    zone_conductor_by_line: Dict[str, Dict[str, Any]],
    mv_conductor: Dict[str, Any],
    transformer_by_zone: Dict[str, Dict[str, Any]],
    lv_kv: float,
    mv_kv: float,
    horizon_years: int,
    apply_growth: bool,
    stepup_transformer: Dict[str, Any],
    coincidence_factor: float = 1.0,
    source_vm_pu: float = 1.0,
) -> PPModel:
    """MV trunk feeding transformer service zones, energised through an explicit step-up.

    ``plan`` is an :class:`core.topology.mvlv_service_zones.MVLVPlan`. A single slack
    (``ext_grid`` at ``source_vm_pu``, the sending voltage) sits on the plant LV bus; an
    explicit **step-up transformer** lifts it to the MV backbone, which steps back down at
    each service zone.
    """
    net = pp.create_empty_network(name="MVLV")
    lookup = customer_lookup(customers_df)
    model = PPModel(net=net)

    # Plant LV bus: the one and only slack. Every customer sits in a transformer service
    # zone, so there is no LV network fed directly at the plant.
    source_lv_bus = pp.create_bus(net, vn_kv=float(lv_kv), name="source_lv")

    # The one and only slack: the plant LV bus at the sending voltage.
    pp.create_ext_grid(net, bus=source_lv_bus, vm_pu=float(source_vm_pu), name="source")

    # --- MV source bus (energised from the plant LV bus through the step-up transformer) ---
    # Trunk lines and the per-zone MV drops share one batch, flushed after the zone loop.
    mv_lines = _LineBatch()
    mv_topology = getattr(plan, "mv_topology", None)
    if mv_topology is not None:
        model.mv_bus_by_pole = _create_buses(net, mv_topology.pole_xy, mv_kv, "mv_pole_")
        mv_source_bus = model.mv_bus_by_pole[mv_topology.slack_pole_id]
        edges = mv_topology.edges_4326
        if edges is not None and not edges.empty:
            for lid, u, v, length_m in zip(edges["line_id"], edges["u"], edges["v"], edges["length_m"]):
                mv_lines.add(model.mv_bus_by_pole[int(u)], model.mv_bus_by_pole[int(v)],
                             float(length_m) / 1000.0, mv_conductor, f"mv::{lid}")
    else:
        mv_source_bus = pp.create_bus(net, vn_kv=float(mv_kv), name="mv_source")

    # --- Explicit step-up transformer: plant LV bus -> MV backbone ---
    su = stepup_transformer
    model.stepup_trafo_index = pp.create_transformer_from_parameters(
        net,
        hv_bus=mv_source_bus,
        lv_bus=source_lv_bus,
        sn_mva=float(su["sn_kva"]) / 1000.0,
        vn_hv_kv=float(su.get("vn_hv_kv", mv_kv)),
        vn_lv_kv=float(su.get("vn_lv_kv", lv_kv)),
        vk_percent=float(su["vk_percent"]),
        vkr_percent=float(su["vkr_percent"]),
        pfe_kw=float(su.get("pfe_kw", 0.0)),
        i0_percent=float(su.get("i0_percent", 0.0)),
        name="stepup_tx",
    )

    # --- Service zones: MV line -> transformer -> local LV tree ---
    #
    # Built table by table rather than zone by zone: one create_buses / create_switches /
    # create_transformers_from_parameters / create_lines / create_loads call for the whole
    # settlement. Element index VALUES therefore differ from a zone-by-zone build, but every
    # reference goes through the bus maps below, and a power flow does not depend on bus
    # numbering -- results are unchanged.
    zones = list(plan.zones)
    if zones:
        zids = [z["zone_id"] for z in zones]
        hv_buses = pp.create_buses(net, len(zones), vn_kv=float(mv_kv),
                                   name=[f"{z}_hv" for z in zids])
        lv_buses = pp.create_buses(net, len(zones), vn_kv=float(lv_kv),
                                   name=[f"{z}_lv" for z in zids])

        # All zones' pole buses in one write, then sliced back per zone.
        pole_ids_by_zone = [[int(pid) for pid in z["topology"].pole_xy] for z in zones]
        flat_names = [f"{zid}_pole_{pid}" for zid, pids in zip(zids, pole_ids_by_zone) for pid in pids]
        flat_idx = (pp.create_buses(net, len(flat_names), vn_kv=float(lv_kv), name=flat_names)
                    if flat_names else [])
        at = 0
        for zid, pids in zip(zids, pole_ids_by_zone):
            model.zone_bus_by_pole[zid] = {
                pid: int(i) for pid, i in zip(pids, flat_idx[at:at + len(pids)])
            }
            at += len(pids)

        zone_lines = _LineBatch()
        zone_loads = _LoadBatch()
        for zone, zid, hv_bus in zip(zones, zids, hv_buses):
            # MV feed: a drop off the trunk pole where there is a trunk, else straight
            # from the plant MV bus.
            if mv_topology is not None and zone.get("mv_pole_id") is not None:
                mv_lines.add(model.mv_bus_by_pole[int(zone["mv_pole_id"])], hv_bus,
                             float(zone.get("mv_service_drop_m", 0.0)) / 1000.0,
                             mv_conductor, f"mv_drop::{zid}")
            else:
                mv_lines.add(mv_source_bus, hv_bus, float(zone["mv_length_m"]) / 1000.0,
                             mv_conductor, f"mv::{zid}")

            ztopo = zone["topology"]
            zbus = model.zone_bus_by_pole[zid]
            edges = ztopo.edges_4326
            if edges is not None and not edges.empty:
                for lid, u, v, length_m in zip(edges["line_id"], edges["u"], edges["v"], edges["length_m"]):
                    key = f"{zid}::{lid}"
                    zone_lines.add(zbus[int(u)], zbus[int(v)], float(length_m) / 1000.0,
                                   zone_conductor_by_line[key], key)
            for pid, cids in ztopo.pole_customers.items():
                p_kw, q_kw = _pq_kw(cids, lookup, horizon_years, apply_growth, coincidence_factor)
                zone_loads.add(zbus[int(pid)], p_kw, q_kw)

        # Step-down transformers, and the zero-impedance tie from each LV winding to its
        # zone's slack pole.
        specs = [transformer_by_zone[zid] for zid in zids]
        t_idx = pp.create_transformers_from_parameters(
            net,
            hv_buses=list(hv_buses),
            lv_buses=list(lv_buses),
            sn_mva=[float(t["sn_kva"]) / 1000.0 for t in specs],
            vn_hv_kv=[float(t["vn_hv_kv"]) for t in specs],
            vn_lv_kv=[float(t["vn_lv_kv"]) for t in specs],
            vk_percent=[float(t["vk_percent"]) for t in specs],
            vkr_percent=[float(t["vkr_percent"]) for t in specs],
            pfe_kw=[float(t["pfe_kw"]) for t in specs],
            i0_percent=[float(t["i0_percent"]) for t in specs],
            name=[f"{zid}_tx" for zid in zids],
        )
        model.trafo_index_by_zone = {zid: int(i) for zid, i in zip(zids, t_idx)}
        pp.create_switches(
            net,
            buses=list(lv_buses),
            elements=[model.zone_bus_by_pole[zid][int(z["topology"].slack_pole_id)]
                      for zid, z in zip(zids, zones)],
            et="b", closed=True,
        )
        zone_lines.flush(net, model, mv=False)
        zone_loads.flush(net, model)

    mv_lines.flush(net, model, mv=True)

    # Single-slack invariant: exactly one reference (the plant LV bus); the MV layer is
    # energised only through the step-up transformer, never by a second ext_grid.
    n_slack = int(len(net.ext_grid))
    if n_slack != 1:
        raise ValueError(f"build_mvlv_net expected exactly one slack (ext_grid); got {n_slack}")

    return model
