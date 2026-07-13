"""Altair charts and headroom bars for the results dashboard.

All builders take the dataframes the workflow already produces (``cost_breakdown_table`` and
the parsed SHS screening table) and return either an Altair chart or a small HTML snippet, so
the Streamlit page stays thin and these stay unit-testable without a running app.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import altair as alt
import pandas as pd

from .tables import NPC_COMPONENTS

# Consistent status colours across the dashboard.
STATUS_COLORS = {"grid": "#2ca02c", "standalone": "#d62728", "unserved": "#9ca3af"}


def npc_breakdown_chart(cost_breakdown_table: pd.DataFrame, recommended: str,
                        label_map: Dict[str, str], feasible_ids: Iterable[str]) -> alt.Chart:
    """Stacked bar of NPC contributions for the **feasible** candidates, recommended one starred."""
    feasible = list(feasible_ids)
    present = [c for c in NPC_COMPONENTS if c in cost_breakdown_table.columns]
    df = cost_breakdown_table[cost_breakdown_table["candidate_id"].isin(feasible)].melt(
        id_vars="candidate_id", value_vars=present, var_name="component", value_name="cost",
    )
    df = df[df["cost"] != 0.0]
    df["component"] = df["component"].map(NPC_COMPONENTS)

    def _label(key: str) -> str:
        return ("⭐ " if key == recommended else "") + label_map.get(key, key)

    df["candidate"] = df["candidate_id"].map(_label)
    order = [_label(k) for k in label_map if k in feasible]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("candidate:N", sort=order, title=None),
            y=alt.Y("cost:Q", title="NPC contribution", stack=True),
            color=alt.Color("component:N", title="Cost component",
                            sort=list(NPC_COMPONENTS.values())),
            tooltip=[alt.Tooltip("candidate:N"), alt.Tooltip("component:N"),
                     alt.Tooltip("cost:Q", format=",.0f")],
        )
        .properties(height=360)
    )


def npc_share_chart(components_df: pd.DataFrame) -> Optional[alt.Chart]:
    """Donut of one candidate's NPC composition (from :func:`core.viz.tables.npc_components`)."""
    if components_df is None or components_df.empty:
        return None
    return (
        alt.Chart(components_df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("npc:Q", stack=True),
            color=alt.Color("component:N", title=None, sort=list(NPC_COMPONENTS.values())),
            tooltip=[alt.Tooltip("component:N"), alt.Tooltip("npc:Q", format=",.0f"),
                     alt.Tooltip("share_%:Q", title="share %")],
        )
        .properties(height=260)
    )


def shs_decision_chart(screening_df: pd.DataFrame) -> Optional[alt.Chart]:
    """Horizontal bars of the economic margin (avoided grid cost − SHS NPC) per leaf branch.

    A positive margin means stand-alone is cheaper (selected, green); negative means the grid
    is cheaper (kept, grey). Candidates blocked before costing have no margin and are omitted.
    Returns ``None`` when nothing has a numeric margin.
    """
    d = screening_df[screening_df["margin"].notna()]
    if d.empty:
        return None
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("margin:Q", title="Avoided grid cost − SHS NPC   (> 0 ⇒ stand-alone cheaper)"),
            y=alt.Y("cluster_id:N", sort="-x", title="Candidate leaf branch"),
            color=alt.Color(
                "outcome:N", title="Decision",
                scale=alt.Scale(domain=["selected", "kept"], range=["#2ca02c", "#6b7280"]),
            ),
            tooltip=[alt.Tooltip("cluster_id:N", title="branch"),
                     alt.Tooltip("n_customers:Q", title="customers"),
                     alt.Tooltip("shs_npc:Q", title="SHS NPC", format=",.0f"),
                     alt.Tooltip("avoided_cost:Q", title="avoided", format=",.0f"),
                     alt.Tooltip("margin:Q", format=",.0f"),
                     alt.Tooltip("flags:N", title="diagnostics")],
        )
        .properties(height=alt.Step(18))
    )


# --------------------------------------------------------------------------------------------
# Tier served: how many customers get which service tier, and who is capped by an SHS package.
# --------------------------------------------------------------------------------------------

def tier_service_chart(tier_frame: pd.DataFrame, candidate_id: str) -> Optional[alt.Chart]:
    """Stacked bar of customers per current demand tier, split by service status.

    Grid customers (green) can still grow with future demand; stand-alone customers (red) are
    capped at their SHS package's service tier — the future-expansion limit this chart surfaces.
    """
    if tier_frame is None or tier_frame.empty:
        return None
    d = tier_frame[tier_frame["candidate_id"] == candidate_id]
    if d.empty:
        return None
    grouped = d.groupby(["demand_tier", "status"]).size().reset_index(name="customers")
    present = [s for s in STATUS_COLORS if s in set(grouped["status"])]
    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("demand_tier:O", title="Current demand tier"),
            y=alt.Y("customers:Q", title="Customers", stack=True),
            color=alt.Color("status:N", title="Service",
                            scale=alt.Scale(domain=present, range=[STATUS_COLORS[s] for s in present])),
            tooltip=[alt.Tooltip("demand_tier:O", title="tier"),
                     alt.Tooltip("status:N", title="service"),
                     alt.Tooltip("customers:Q")],
        )
        .properties(height=280)
    )


# --------------------------------------------------------------------------------------------
# Parallel coordinates: the four solutions on the x-axis, one line per decision metric.
# --------------------------------------------------------------------------------------------

def parallel_coordinates_chart(candidates_df: pd.DataFrame, metrics: List[Dict[str, Any]],
                               *, candidate_col: str = "candidate",
                               candidate_order: Optional[List[str]] = None) -> Optional[alt.Chart]:
    """Parallel-coordinates plot with the **candidates on the x-axis** and **one line per metric**.

    ``candidates_df`` has one row per candidate (column ``candidate_col`` = its display label plus
    one column per metric). ``metrics`` is an ordered list of ``{"col", "label", "higher_is_better"}``.
    Each metric is min-max normalized *across the candidates* and oriented so that **up = better**,
    so for any solution you can read how it ranks on every metric, and each metric line shows which
    solution wins it. Raw values stay in the tooltip. Returns ``None`` if nothing to plot.
    """
    if candidates_df is None or candidates_df.empty or not metrics:
        return None
    order = candidate_order or list(candidates_df[candidate_col])

    records = []
    for m in metrics:
        vals = pd.to_numeric(candidates_df[m["col"]], errors="coerce")
        lo, hi = float(vals.min()), float(vals.max())
        for _, row in candidates_df.iterrows():
            raw = pd.to_numeric(pd.Series([row[m["col"]]]), errors="coerce").iloc[0]
            if raw != raw:  # NaN
                norm = None
            elif hi > lo:
                frac = (float(raw) - lo) / (hi - lo)
                norm = frac if m.get("higher_is_better", True) else 1.0 - frac
            else:
                norm = 0.5
            records.append({
                "candidate": row[candidate_col],
                "metric": m["label"],
                "norm": norm,
                "raw": None if raw != raw else float(raw),
            })
    long = pd.DataFrame(records)
    if long["norm"].dropna().empty:
        return None
    metric_order = [m["label"] for m in metrics]
    return (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.5, opacity=0.9)
        .encode(
            x=alt.X("candidate:N", sort=order, title=None,
                    axis=alt.Axis(labelAngle=0, labelLimit=160)),
            y=alt.Y("norm:Q", title="Normalized across solutions — up = better",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title="Metric", sort=metric_order),
            detail="metric:N",
            tooltip=[alt.Tooltip("candidate:N", title="solution"),
                     alt.Tooltip("metric:N"), alt.Tooltip("raw:Q", title="value", format=",.3f")],
        )
        .properties(height=380)
    )


# --------------------------------------------------------------------------------------------
# Headroom bars: how much of an enforced electrical limit a candidate consumes at the horizon.
# --------------------------------------------------------------------------------------------

def _headroom_color(util: Optional[float]) -> str:
    if util is None:
        return "#6b7280"
    if util >= 1.0:
        return "#d62728"   # at or over the limit (violation)
    if util >= 0.85:
        return "#f28e2b"
    if util >= 0.70:
        return "#f2b701"
    return "#2ca02c"


def _bar(label: str, text: str, util: Optional[float]) -> str:
    width = 100.0 if util is None else max(0.0, min(util, 1.0)) * 100.0
    color = _headroom_color(util)
    warn = " ⚠" if (util is not None and util >= 1.0) else ""
    return (
        '<div style="margin:6px 0;">'
        '<div style="display:flex;justify-content:space-between;font-size:0.8rem;opacity:0.9;">'
        f'<span>{label}</span><span><b>{text}{warn}</b></span></div>'
        '<div style="background:#2a2a2a;border-radius:4px;height:10px;overflow:hidden;">'
        f'<div style="width:{width:.0f}%;height:100%;background:{color};"></div></div></div>'
    )


def _finite(x: Any) -> bool:
    try:
        return x is not None and not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def case_headroom_html(outcome, constraints: Dict[str, Any]) -> str:
    """HTML headroom bars for one candidate's binding (horizon) power-flow outcome.

    Voltage is a *floor* limit (fraction of the permitted drop from 1.0 pu that is used);
    line and transformer loading are *ceiling* limits (value / limit). Bars only appear for
    quantities the candidate actually has (e.g. no transformer bar for a pure-LV candidate).
    """
    if outcome is None:
        return ""
    v_min = float(constraints.get("voltage_min_pu", 0.90))
    ll_max = float(constraints.get("max_line_loading_percent", 100.0))
    tl_max = float(constraints.get("max_transformer_loading_percent", 100.0))
    bars = []

    v = outcome.min_vm_pu
    if _finite(v):
        util = (1.0 - float(v)) / (1.0 - v_min) if (1.0 - v_min) > 0 else None
        bars.append(_bar(f"Min voltage (≥ {v_min:.2f} pu)", f"{float(v):.3f} pu", util))

    ll = outcome.max_line_loading_percent
    if _finite(ll):
        bars.append(_bar(f"Max line loading (≤ {ll_max:.0f}%)", f"{float(ll):.0f}%", float(ll) / ll_max if ll_max else None))

    tl = outcome.max_trafo_loading_percent
    if _finite(tl) and float(tl) > 0:
        bars.append(_bar(f"Max transformer loading (≤ {tl_max:.0f}%)", f"{float(tl):.0f}%", float(tl) / tl_max if tl_max else None))

    return "".join(bars)
