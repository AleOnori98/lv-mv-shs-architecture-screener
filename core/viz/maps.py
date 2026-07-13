"""Folium maps for MVP inputs and candidate results."""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable

import folium
import pandas as pd
from branca.element import Element
from shapely.geometry import MultiPoint

from core.workflow.exports import case_lines_gdf, transformers_gdf

# Customer served-status -> marker style. "unserved" = neither grid-connected nor stand-alone
# (e.g. a zone member the local LV engine could not attach); previously mis-drawn as "grid".
CUSTOMER_STATUS_STYLE = {
    "grid": {"fill": "#2ca02c", "edge": "#2ca02c", "radius": 2.8, "weight": 1.0, "label": "grid"},
    "standalone": {"fill": "#d62728", "edge": "#d62728", "radius": 3.5, "weight": 1.0, "label": "stand-alone"},
    "unserved": {"fill": "#9ca3af", "edge": "#111111", "radius": 4.2, "weight": 1.6, "label": "UNSERVED - not connected"},
}

# Distinct fills for per-zone area hulls (cycled by zone order).
ZONE_AREA_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#17becf", "#e377c2", "#bcbd22", "#393b79", "#d62728",
)


CUSTOMER_TYPE_COLORS = {
    "household": "#2ca02c",
    "shop": "#1f77b4",
    "productive": "#ff7f0e",
    "school": "#9467bd",
    "clinic": "#d62728",
    "hospital": "#8c564b",
    "mosque": "#17becf",
    "public_service": "#e377c2",
    "water_pump": "#bcbd22",
    "other": "#7f7f7f",
}
FALLBACK_CUSTOMER_COLORS = ("#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#17becf", "#7f7f7f")


def _customer_color(customer_type: str, index: int = 0) -> str:
    key = str(customer_type or "other").strip().lower()
    return CUSTOMER_TYPE_COLORS.get(key, FALLBACK_CUSTOMER_COLORS[index % len(FALLBACK_CUSTOMER_COLORS)])


def _stress_color(value: float | None, *, red_if_nan: bool = False) -> str:
    if value is None or value != value:
        return "#d62728" if red_if_nan else "#6b7280"
    if value < 0.50:
        return "#2ca02c"
    if value < 0.75:
        return "#f2b701"
    if value <= 1.0:
        return "#f28e2b"
    return "#d62728"


def _line_loading_color(loading_percent: float | None, max_loading_percent: float, *, nonconverged: bool = False) -> str:
    if nonconverged:
        return "#d62728"
    if max_loading_percent <= 0:
        max_loading_percent = 100.0
    stress = None if loading_percent is None else float(loading_percent) / float(max_loading_percent)
    return _stress_color(stress)


def _bus_voltage_color(vm_pu: float | None, v_min: float, v_max: float, *, nonconverged: bool = False) -> str:
    if nonconverged:
        return "#d62728"
    if vm_pu is None or vm_pu != vm_pu:
        return "#6b7280"
    v = float(vm_pu)
    if v < v_min or v > v_max:
        return "#d62728"
    if v <= v_min + 0.02 or v >= v_max - 0.01:
        return "#f28e2b"
    if v <= v_min + 0.05 or v >= v_max - 0.025:
        return "#f2b701"
    return "#2ca02c"


def _map_model_and_outcome(case, scenario: str):
    scenario = str(scenario or "horizon").lower()
    if scenario == "now":
        return getattr(case, "model_now", None), getattr(case, "outcome_now", None)
    if scenario in ("stress", "coincident"):
        return getattr(case, "model_stress", None), getattr(case, "outcome_stress", None)
    return getattr(case, "model", None), getattr(case, "outcome_horizon", None)


def _line_voltage_info(model, outcome, line_id: str) -> dict[str, float] | None:
    if model is None or getattr(model, "net", None) is None:
        return None
    if outcome is not None and not bool(getattr(outcome, "converged", False)):
        return {"nonconverged": True}
    idx = getattr(model, "line_index_by_id", {}).get(str(line_id))
    net = model.net
    if idx is None or "res_bus" not in net or "res_line" not in net or idx not in net.line.index:
        return None
    try:
        line = net.line.loc[idx]
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        from_v = float(net.res_bus.at[from_bus, "vm_pu"])
        to_v = float(net.res_bus.at[to_bus, "vm_pu"])
        loading = float(net.res_line.at[idx, "loading_percent"]) if idx in net.res_line.index else float("nan")
    except Exception:
        return None
    return {
        "from_vm_pu": from_v,
        "to_vm_pu": to_v,
        "min_vm_pu": min(from_v, to_v),
        "drop_pu": abs(from_v - to_v),
        "loading_percent": loading,
        "nonconverged": False,
    }


def _voltage_tooltip(line_id: str, level: str, info: dict[str, float] | None) -> str:
    base = f"{html.escape(str(level))} {html.escape(str(line_id))}"
    if info is None:
        return base
    if info.get("nonconverged"):
        return f"{base} | power flow did not converge"
    loading = info["loading_percent"]
    loading_text = f" | loading {loading:.1f}%" if loading == loading else ""
    return (
        f"{base} | V {info['from_vm_pu']:.3f}->{info['to_vm_pu']:.3f} pu"
        f" | drop {info['drop_pu'] * 100:.2f}%"
        f" | min {info['min_vm_pu']:.3f} pu"
        f"{loading_text}"
    )


def _constraints(params: Dict[str, Any] | None) -> Dict[str, float]:
    params = params or {}
    return {
        "voltage_min_pu": float(params.get("voltage_min_pu", 0.90)),
        "voltage_max_pu": float(params.get("voltage_max_pu", 1.05)),
        "max_line_loading_percent": float(params.get("max_line_loading_percent", 100.0)),
    }


def _bus_voltage(model, outcome, bus_idx: int) -> float | None:
    if model is None or outcome is None or not bool(getattr(outcome, "converged", False)):
        return None
    net = model.net
    if "res_bus" not in net or bus_idx not in net.res_bus.index:
        return None
    try:
        return float(net.res_bus.at[bus_idx, "vm_pu"])
    except Exception:
        return None


def _add_voltage_buses(m: folium.Map, case, model, outcome, params: Dict[str, Any] | None) -> None:
    """Add colored bus markers for the selected validation scenario."""
    c = _constraints(params)
    nonconverged = bool(outcome is not None and not bool(getattr(outcome, "converged", False)))
    layer = folium.FeatureGroup(name="Bus voltage stress", show=True)
    n = 0

    def add_marker(lat: float, lon: float, bus_idx: int | None, label: str, radius: float = 3.2) -> None:
        nonlocal n
        vm = _bus_voltage(model, outcome, int(bus_idx)) if bus_idx is not None else None
        color = _bus_voltage_color(vm, c["voltage_min_pu"], c["voltage_max_pu"], nonconverged=nonconverged)
        if nonconverged:
            tooltip = f"{html.escape(label)} | power flow did not converge"
        elif vm is None:
            tooltip = f"{html.escape(label)} | voltage unavailable"
        else:
            tooltip = f"{html.escape(label)} | V {vm:.3f} pu"
        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=radius,
            color="#111111",
            weight=0.6,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            tooltip=tooltip,
        ).add_to(layer)
        n += 1

    if getattr(case, "plan", None) is not None:
        mv_topo = getattr(case.plan, "mv_topology", None)
        if mv_topo is not None and getattr(mv_topo, "poles_4326", None) is not None:
            for _, p in mv_topo.poles_4326.iterrows():
                pid = int(p["pole_id"])
                add_marker(
                    p.geometry.y,
                    p.geometry.x,
                    getattr(model, "mv_bus_by_pole", {}).get(pid) if model is not None else None,
                    f"MV bus pole {pid}",
                    radius=3.6,
                )
        for zone in case.plan.zones:
            zid = zone["zone_id"]
            ztopo = zone["topology"]
            zone_buses = getattr(model, "zone_bus_by_pole", {}).get(zid, {}) if model is not None else {}
            if getattr(ztopo, "poles_4326", None) is not None:
                for _, p in ztopo.poles_4326.iterrows():
                    pid = int(p["pole_id"])
                    add_marker(
                        p.geometry.y,
                        p.geometry.x,
                        zone_buses.get(pid),
                        f"{zid} LV bus pole {pid}",
                    )
            if model is not None and getattr(model, "net", None) is not None:
                lat, lon = zone["transformer_latlon"]
                hv = model.net.bus.index[model.net.bus["name"].eq(f"{zid}_hv")]
                lv = model.net.bus.index[model.net.bus["name"].eq(f"{zid}_lv")]
                add_marker(lat, lon, int(hv[0]) if len(hv) else None, f"{zid} transformer HV bus", radius=4.2)
                add_marker(lat, lon, int(lv[0]) if len(lv) else None, f"{zid} transformer LV bus", radius=4.2)
    elif getattr(case, "topology", None) is not None and getattr(case.topology, "poles_4326", None) is not None:
        for _, p in case.topology.poles_4326.iterrows():
            pid = int(p["pole_id"])
            add_marker(
                p.geometry.y,
                p.geometry.x,
                getattr(model, "bus_by_pole", {}).get(pid) if model is not None else None,
                f"LV bus pole {pid}",
            )

    if n:
        layer.add_to(m)


def _add_stress_legend(m: folium.Map, scenario_label: str) -> None:
    legend = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: white; padding: 10px 12px; border: 1px solid #c8ced6; '
        'border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.25); '
        'font-size: 12px; line-height: 1.45;">'
        f'<div style="font-weight: 700; margin-bottom: 4px;">Electrical stress: {html.escape(scenario_label)}</div>'
        '<div><span style="display:inline-block;width:18px;border-top:4px solid #2ca02c;margin:0 6px 3px 0;"></span>low stress</div>'
        '<div><span style="display:inline-block;width:18px;border-top:4px solid #f2b701;margin:0 6px 3px 0;"></span>medium stress</div>'
        '<div><span style="display:inline-block;width:18px;border-top:4px solid #f28e2b;margin:0 6px 3px 0;"></span>near limit</div>'
        '<div><span style="display:inline-block;width:18px;border-top:4px solid #d62728;margin:0 6px 3px 0;"></span>violation / no convergence</div>'
        '<div style="margin-top:6px;">Lines = loading; dots = bus voltage</div>'
        '<div><span style="display:inline-block;width:18px;border-top:1px solid #111;margin:0 6px 3px 0;"></span>service drop</div>'
        '<div style="margin-top:6px;">Customers:'
        '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#2ca02c;margin:0 3px 0 6px;"></span>grid'
        '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#d62728;margin:0 3px 0 8px;"></span>stand-alone'
        '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#9ca3af;border:1px solid #111;margin:0 3px 0 8px;"></span>unserved</div>'
        '<div><span style="display:inline-block;width:12px;height:10px;background:#1f77b4;opacity:0.30;'
        'border:1px solid #1f77b4;margin:0 6px 2px 0;"></span>service-zone area (MV-fed LV island)</div>'
        "</div>"
    )
    m.get_root().html.add_child(Element(legend))


def _case_topologies(case) -> list:
    if getattr(case, "plan", None) is not None:
        topos = []
        if case.plan.core_topology is not None:
            topos.append(case.plan.core_topology)
        topos.extend(z["topology"] for z in case.plan.zones)
        return topos
    if getattr(case, "topology", None) is not None:
        return [case.topology]
    return []


def _geometry_line_sequences(geometry) -> list[list[tuple[float, float]]]:
    """Return line coordinate sequences from LineString/MultiLineString-like geometries."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"LineString", "LinearRing"}:
        return [list(geometry.coords)]
    if geom_type == "Polygon":
        return [list(geometry.exterior.coords)]
    if geom_type.startswith("Multi") or geom_type == "GeometryCollection":
        seqs: list[list[tuple[float, float]]] = []
        for part in getattr(geometry, "geoms", []):
            seqs.extend(_geometry_line_sequences(part))
        return seqs
    return []


def _latlon_sequence(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(float(y), float(x)) for x, y, *_ in coords]


def _fit_bounds(m: folium.Map, customers_df: pd.DataFrame, source: Dict[str, Any]) -> None:
    points = [[float(source["latitude"]), float(source["longitude"])]]
    if not customers_df.empty:
        points.extend(
            [float(row["latitude"]), float(row["longitude"])]
            for _, row in customers_df[["latitude", "longitude"]].dropna().iterrows()
        )
    if len(points) > 1:
        m.fit_bounds(points, padding=(20, 20))


def _add_input_legend(m: folium.Map, types: list[str]) -> None:
    items = []
    for i, customer_type in enumerate(types):
        label = html.escape(str(customer_type))
        color = _customer_color(customer_type, i)
        items.append(
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};margin-right:6px;"></span>{label}</div>'
        )
    items.extend(
        [
            '<div><span style="display:inline-block;width:18px;border-top:3px solid #5b6472;'
            'margin:0 6px 3px 0;"></span>roads</div>',
            '<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            'background:#ffd400;border:2px solid #111;margin-right:6px;"></span>source</div>',
        ]
    )
    legend = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: white; padding: 10px 12px; border: 1px solid #c8ced6; '
        'border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.25); '
        'font-size: 12px; line-height: 1.45;">'
        '<div style="font-weight: 700; margin-bottom: 4px;">Input layers</div>'
        + "".join(items)
        + "</div>"
    )
    m.get_root().html.add_child(Element(legend))


def _add_peak_demand_legend(m: folium.Map, min_peak: float, max_peak: float) -> None:
    if max_peak <= 0:
        labels = ["0 kW"]
    elif min_peak == max_peak:
        labels = [f"{max_peak:.2f} kW"]
    else:
        mid_peak = (min_peak + max_peak) / 2.0
        labels = [f"{min_peak:.2f} kW", f"{mid_peak:.2f} kW", f"{max_peak:.2f} kW"]
    sizes = [7, 12, 18][-len(labels):]
    items = "".join(
        f'<div style="display:flex;align-items:center;margin-top:4px;">'
        f'<span style="display:inline-block;width:{size}px;height:{size}px;border-radius:50%;'
        f'background:#1f6feb;border:1px solid #0b3d91;opacity:0.72;margin-right:8px;"></span>{label}</div>'
        for size, label in zip(sizes, labels)
    )
    legend = (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: white; padding: 10px 12px; border: 1px solid #c8ced6; '
        'border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.25); '
        'font-size: 12px; line-height: 1.45;">'
        '<div style="font-weight: 700; margin-bottom: 4px;">Peak load demand</div>'
        + items
        + '<div style="margin-top:6px;"><span style="display:inline-block;width:18px;'
        'border-top:3px solid #5b6472;margin:0 6px 3px 0;"></span>roads</div>'
        + '<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        'background:#ffd400;border:2px solid #111;margin-right:6px;"></span>source</div>'
        + "</div>"
    )
    m.get_root().html.add_child(Element(legend))


def _add_roads(m: folium.Map, roads) -> None:
    if roads is None or getattr(roads, "empty", True):
        return
    road_layer = folium.FeatureGroup(name="Roads", show=True)
    for i, row in roads.iterrows():
        tooltip = str(row.get("road_id", f"road {i}")) if hasattr(row, "get") else f"road {i}"
        for coords in _geometry_line_sequences(row.geometry):
            latlon = _latlon_sequence(coords)
            if len(latlon) >= 2:
                folium.PolyLine(
                    latlon,
                    color="#5b6472",
                    weight=2.2,
                    opacity=0.65,
                    tooltip=tooltip,
                ).add_to(road_layer)
    road_layer.add_to(m)


def _add_source(m: folium.Map, source: Dict[str, Any]) -> None:
    center = [float(source["latitude"]), float(source["longitude"])]
    source_label = html.escape(str(source.get("source_id", "Source")))
    source_type = html.escape(str(source.get("source_type", "injection")))
    source_tooltip = (
        f"{source_label} ({source_type})"
        f" | LV {float(source.get('lv_voltage_kv', 0.4)):.2f} kV"
        f" | MV {float(source.get('mv_voltage_kv', 11.0)):.1f} kV"
    )
    folium.CircleMarker(
        center,
        radius=10,
        color="#111111",
        weight=2.5,
        fill=True,
        fill_color="#ffd400",
        fill_opacity=0.95,
        tooltip=source_tooltip,
    ).add_to(m)
    folium.Marker(
        center,
        tooltip=source_tooltip,
        icon=folium.Icon(color="black", icon="star", prefix="fa"),
    ).add_to(m)


def _add_service_drops(m: folium.Map, case, customers_df: pd.DataFrame) -> None:
    by_customer = customers_df.set_index(customers_df["customer_id"].astype(str))
    layer = folium.FeatureGroup(name="Service drops: pole to customer", show=True)
    n_drops = 0
    for topo in _case_topologies(case):
        if topo is None or topo.poles_4326 is None or topo.poles_4326.empty:
            continue
        pole_geom = topo.poles_4326.set_index("pole_id").geometry.to_dict()
        for cid, pole_id in topo.customer_pole.items():
            cid = str(cid)
            if cid not in by_customer.index:
                continue
            pole = pole_geom.get(int(pole_id))
            if pole is None:
                continue
            row = by_customer.loc[cid]
            customer_lat = float(row["latitude"])
            customer_lon = float(row["longitude"])
            drop_m = float(topo.service_drop_m.get(cid, 0.0))
            folium.PolyLine(
                [(float(pole.y), float(pole.x)), (customer_lat, customer_lon)],
                color="#111111",
                weight=1.0,
                opacity=0.55,
                tooltip=f"{html.escape(cid)} service drop | {drop_m:.1f} m",
            ).add_to(layer)
            n_drops += 1
    if n_drops:
        layer.add_to(m)


def input_map(customers_df: pd.DataFrame, source: Dict[str, Any], roads=None) -> folium.Map:
    """Map raw loaded inputs before the MVP workflow is run."""
    center = [float(source["latitude"]), float(source["longitude"])]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    _add_roads(m, roads)

    customer_types = sorted(customers_df["customer_type"].dropna().astype(str).unique().tolist())
    layers = {
        customer_type: folium.FeatureGroup(name=f"Customers: {customer_type}", show=True)
        for customer_type in customer_types
    }
    type_index = {customer_type: i for i, customer_type in enumerate(customer_types)}
    for _, row in customers_df.iterrows():
        customer_type = str(row.get("customer_type", "other"))
        color = _customer_color(customer_type, type_index.get(customer_type, 0))
        tooltip = (
            f"{html.escape(str(row.get('customer_id', 'customer')))}"
            f" | {html.escape(customer_type)}"
            f" | peak {float(row.get('peak_kw_now', 0.0)):.2f} kW"
        )
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=3.8,
            color=color,
            weight=1.0,
            fill=True,
            fill_color=color,
            fill_opacity=0.86,
            tooltip=tooltip,
        ).add_to(layers.setdefault(customer_type, folium.FeatureGroup(name=f"Customers: {customer_type}", show=True)))
    for layer in layers.values():
        layer.add_to(m)

    _add_source(m, source)

    _fit_bounds(m, customers_df, source)
    _add_input_legend(m, customer_types)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def peak_demand_map(customers_df: pd.DataFrame, source: Dict[str, Any], roads=None) -> folium.Map:
    """Map customer peak load as same-color bubbles scaled by demand."""
    center = [float(source["latitude"]), float(source["longitude"])]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    _add_roads(m, roads)

    peaks = pd.to_numeric(customers_df["peak_kw_now"], errors="coerce").fillna(0.0).clip(lower=0.0)
    min_peak = float(peaks.min()) if not peaks.empty else 0.0
    max_peak = float(peaks.max()) if not peaks.empty else 0.0
    if max_peak > min_peak:
        scaled = ((peaks - min_peak) / (max_peak - min_peak)).pow(0.5)
        radii = 4.0 + scaled * 14.0
    else:
        radii = pd.Series(8.0, index=customers_df.index)

    demand_layer = folium.FeatureGroup(name="Peak load demand", show=True)
    for idx, row in customers_df.iterrows():
        peak_kw = float(peaks.loc[idx])
        tooltip = f"{html.escape(str(row.get('customer_id', 'customer')))} | peak {peak_kw:.2f} kW"
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=float(radii.loc[idx]),
            color="#0b3d91",
            weight=1.0,
            fill=True,
            fill_color="#1f6feb",
            fill_opacity=0.58,
            tooltip=tooltip,
        ).add_to(demand_layer)
    demand_layer.add_to(m)

    _add_source(m, source)
    _fit_bounds(m, customers_df, source)
    _add_peak_demand_legend(m, min_peak, max_peak)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def classify_customers(grid_ids: Iterable[str], standalone_ids: Iterable[str],
                       all_ids: Iterable[str]) -> Dict[str, str]:
    """Map each customer id to ``grid`` / ``standalone`` / ``unserved`` for map colouring.

    ``unserved`` = served by neither the grid nor a stand-alone system (a service obligation
    the workflow also records as a violation), distinct from a grid customer and not green.
    """
    grid = {str(c) for c in grid_ids}
    standalone = {str(c) for c in standalone_ids}
    out: Dict[str, str] = {}
    for cid in (str(c) for c in all_ids):
        out[cid] = "standalone" if cid in standalone else ("grid" if cid in grid else "unserved")
    return out


def _add_zone_areas(m: folium.Map, case, customers_df: pd.DataFrame) -> None:
    """Faint per-zone hull behind the network so each MV-fed LV island reads as one zone.

    Drawn before the lines/dots so it sits in the background; toggleable via the layer control.
    """
    plan = getattr(case, "plan", None)
    if plan is None or not getattr(plan, "zones", None):
        return
    by_customer = customers_df.set_index(customers_df["customer_id"].astype(str))
    layer = folium.FeatureGroup(name="Service-zone areas", show=True)
    added = 0
    for i, zone in enumerate(plan.zones):
        served = [str(c) for c in zone["topology"].served_customer_ids]
        pts = [(float(by_customer.loc[cid]["longitude"]), float(by_customer.loc[cid]["latitude"]))
               for cid in served if cid in by_customer.index]
        lat, lon = zone["transformer_latlon"]
        pts.append((float(lon), float(lat)))
        if not pts:
            continue
        try:
            hull = MultiPoint(pts).convex_hull.buffer(0.0008)  # ~80 m padding in degrees
        except Exception:
            continue
        color = ZONE_AREA_COLORS[i % len(ZONE_AREA_COLORS)]
        tip = f"{zone['zone_id']} · {zone['transformer_id']} · {len(served)} customers"
        for seq in _geometry_line_sequences(hull):
            folium.Polygon(_latlon_sequence(seq), color=color, weight=1.2, opacity=0.55,
                           fill=True, fill_color=color, fill_opacity=0.08, tooltip=tip).add_to(layer)
        added += 1
    if added:
        layer.add_to(m)


def case_map(
    case,
    customers_df: pd.DataFrame,
    source: Dict[str, Any],
    *,
    stress_scenario: str = "horizon",
    scenario_label: str = "horizon",
    constraints: Dict[str, Any] | None = None,
) -> folium.Map:
    center = [source["latitude"], source["longitude"]]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron", control_scale=True)
    model, outcome = _map_model_and_outcome(case, stress_scenario)
    c = _constraints(constraints)
    nonconverged = bool(outcome is not None and not bool(getattr(outcome, "converged", False)))

    # Faint per-zone hulls first so they sit behind the network.
    _add_zone_areas(m, case, customers_df)

    # Lines colored by solved loading stress; bus markers below show voltage stress.
    lines = case_lines_gdf(case, source)
    for _, row in lines.iterrows():
        info = _line_voltage_info(model, outcome, str(row["line_id"]))
        color = _line_loading_color(
            None if info is None else info.get("loading_percent"),
            c["max_line_loading_percent"],
            nonconverged=nonconverged,
        )
        weight = 2.6 if row["voltage_level"] == "LV" else 4.2
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.86,
            tooltip=_voltage_tooltip(str(row["line_id"]), str(row["voltage_level"]), info),
        ).add_to(m)

    _add_service_drops(m, case, customers_df)
    _add_voltage_buses(m, case, model, outcome, constraints)

    # Poles.
    pole_layer = folium.FeatureGroup(name="Poles", show=False)
    topos = []
    if case.plan is not None:
        if case.plan.core_topology is not None:
            topos.append(case.plan.core_topology)
        if getattr(case.plan, "mv_topology", None) is not None:
            topos.append(case.plan.mv_topology)
        topos.extend(z["topology"] for z in case.plan.zones)
    elif case.topology is not None:
        topos.append(case.topology)
    for topo in topos:
        for _, p in topo.poles_4326.iterrows():
            folium.CircleMarker([p.geometry.y, p.geometry.x], radius=1.6, color="#444", fill=True, fill_opacity=0.7).add_to(pole_layer)
    pole_layer.add_to(m)

    # Customers by served status (grid / stand-alone / unserved), each a toggleable layer.
    status = classify_customers(case.grid_customer_ids, case.standalone_customer_ids, customers_df["customer_id"])
    status_layers = {
        s: folium.FeatureGroup(name=f"Customers: {CUSTOMER_STATUS_STYLE[s]['label'].split(' - ')[0]}", show=True)
        for s in CUSTOMER_STATUS_STYLE
    }
    used = set()
    for _, cust in customers_df.iterrows():
        cid = str(cust["customer_id"])
        s = status.get(cid, "unserved")
        style = CUSTOMER_STATUS_STYLE[s]
        folium.CircleMarker(
            [float(cust["latitude"]), float(cust["longitude"])],
            radius=style["radius"], color=style["edge"], weight=style["weight"],
            fill=True, fill_color=style["fill"], fill_opacity=0.9,
            tooltip=f"{cid} - {style['label']}",
        ).add_to(status_layers[s])
        used.add(s)
    for s, layer in status_layers.items():
        if s in used:
            layer.add_to(m)

    # Transformers + source.
    for _, t in transformers_gdf(case).iterrows():
        folium.Marker([t.geometry.y, t.geometry.x], tooltip=f"{t['transformer_id']} ({t['sn_kva']:.0f} kVA)",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m)
    folium.Marker(center, tooltip="Source", icon=folium.Icon(color="black", icon="star", prefix="fa")).add_to(m)

    _add_stress_legend(m, scenario_label)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


LEGEND = (
    "Customers: green grid-connected, red stand-alone, grey unserved (not connected). "
    "Shaded areas: MV-fed LV service zones (each is a separate LV island joined only by the MV trunk). "
    "Network lines: green/yellow/orange/red = line-loading stress. "
    "Electrical bus dots: green/yellow/orange/red = voltage stress. "
    "Thin black lines: pole-to-customer service drops. Orange markers: transformers. Black marker: source."
)
