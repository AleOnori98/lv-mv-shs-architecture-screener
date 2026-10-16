"""Small table-presentation helpers for the Streamlit dashboard."""
from __future__ import annotations

import math

from typing import Any, Dict

import pandas as pd


def highlight_recommended(comparison: pd.DataFrame, recommended: str) -> pd.io.formats.style.Styler:
    def _row_style(row):
        if row["candidate_id"] == recommended:
            return ["background-color: #1b3a2b; color: #d1f7d1"] * len(row)
        if not row["feasible"]:
            return ["background-color: #3a1b1b; color: #f7d1d1"] * len(row)
        return [""] * len(row)

    return comparison.style.apply(_row_style, axis=1)


def shs_screening_table(decision_log: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    """Per-leaf-branch SHS economics for one screen (B or D), parsed from the decision log.

    The workflow logs one ``standalone`` row per screened subtree with metric ``"<SHS NPC> vs
    <avoided>"``; subtree size and diagnostic flags come from the matching ``branch_detection`` /
    ``mvlv_subtree`` rows. Returns columns ``cluster_id, n_customers, shs_npc, avoided_cost,
    margin, outcome, reason, flags`` where ``outcome`` is ``selected`` / ``kept`` / ``blocked``
    (blocked = ineligible or no feasible package, hence no economics).
    """
    log = decision_log
    if log is None or log.empty:
        return pd.DataFrame()
    diag = log[log["step"].isin(["branch_detection", "mvlv_subtree"])]
    n_by = dict(zip(diag["object_id"], pd.to_numeric(diag["metric_value"], errors="coerce")))
    flags_by = dict(zip(diag["object_id"], diag["reason"]))

    rows = []
    sub = log[(log["step"] == "standalone") & (log["candidate_id"] == candidate_id)]
    for r in sub.itertuples(index=False):
        shs_npc = avoided = margin = float("nan")
        metric = str(getattr(r, "metric_value", "") or "")
        if " vs " in metric:
            lhs, rhs = metric.split(" vs ", 1)
            try:
                shs_npc, avoided = float(lhs), float(rhs)
                margin = avoided - shs_npc
            except ValueError:
                pass
        priced = margin == margin  # not NaN
        outcome = "selected" if r.decision == "standalone" else ("kept" if priced else "blocked")
        n = n_by.get(r.object_id)
        rows.append({
            "cluster_id": r.object_id,
            "n_customers": int(n) if n == n and n is not None else None,
            "shs_npc": round(shs_npc) if shs_npc == shs_npc else None,
            "avoided_cost": round(avoided) if avoided == avoided else None,
            "margin": round(margin) if priced else None,
            "outcome": outcome,
            "reason": r.reason,
            "flags": flags_by.get(r.object_id, ""),
        })
    return pd.DataFrame(rows)


def customer_type_summary(customers: pd.DataFrame) -> pd.DataFrame:
    """Compact demand and service-flag summary by normalized customer type."""
    summary = (
        customers.groupby("customer_type", dropna=False)
        .agg(
            customers=("customer_id", "count"),
            peak_kw_now=("peak_kw_now", "sum"),
            annual_kwh_now=("annual_kwh_now", "sum"),
            must_connect=("must_connect_flag", "sum"),
            standalone_eligible=("can_be_standalone_flag", "sum"),
        )
        .reset_index()
        .sort_values(["customers", "customer_type"], ascending=[False, True])
    )
    summary["peak_kw_now"] = summary["peak_kw_now"].round(2)
    summary["annual_kwh_now"] = summary["annual_kwh_now"].round(1)
    for col in ("must_connect", "standalone_eligible"):
        summary[col] = summary[col].astype(int)
    return summary


# NPC component columns in cost_breakdown_table -> human labels (also display/stack order).
NPC_COMPONENTS = {
    "lv_line_cost": "LV line",
    "mv_line_cost": "MV line",
    "pole_cost": "Poles",
    "transformer_cost": "Transformers",
    "connection_metering_cost": "Connection & metering",
    "service_drop_cost": "Service drops",
    "standalone_npc": "Stand-alone NPC",
    "loss_pv_cost": "Loss PV",
    "generation_npc": "Generation (LCOE)",
}


def npc_components(cost_breakdown_table: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    """Non-zero NPC components for one candidate: columns ``component, npc, share_%`` (desc).

    Components that are **unknown** (NaN -- e.g. the loss cost of a candidate whose power
    flow diverged) are left out rather than plotted as a NaN bar or folded into the share
    denominator. The candidate's ``npc_complete`` / ``npc_missing_components`` fields in the
    comparison table are what report the omission.
    """
    row = cost_breakdown_table[cost_breakdown_table["candidate_id"] == candidate_id]
    if row.empty:
        return pd.DataFrame(columns=["component", "npc", "share_%"])
    row = row.iloc[0]
    items = [
        (NPC_COMPONENTS[c], float(row[c]))
        for c in NPC_COMPONENTS
        if c in row.index and not math.isnan(float(row[c])) and float(row[c]) != 0.0
    ]
    df = pd.DataFrame(items, columns=["component", "npc"]).sort_values("npc", ascending=False)
    total = float(df["npc"].sum())
    df["share_%"] = (df["npc"] / total * 100.0).round(1) if total else 0.0
    df["npc"] = df["npc"].round(0)
    return df.reset_index(drop=True)


def source_table(source: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([source])


def cost_parameters_table(cost_parameters: Dict[str, Any]) -> pd.DataFrame:
    rows = [{"parameter": key, "value": value} for key, value in sorted(cost_parameters.items())]
    return pd.DataFrame(rows)


def roads_summary_table(roads) -> pd.DataFrame:
    if roads is None or getattr(roads, "empty", True):
        return pd.DataFrame([{"metric": "provided", "value": False}])
    geom_counts = roads.geometry.geom_type.value_counts().to_dict()
    rows = [
        {"metric": "provided", "value": True},
        {"metric": "features", "value": len(roads)},
        {"metric": "crs", "value": str(getattr(roads, "crs", "") or "unknown")},
    ]
    rows.extend({"metric": f"geometry_{name}", "value": count} for name, count in geom_counts.items())
    return pd.DataFrame(rows)


def roads_preview_table(roads, rows: int = 25) -> pd.DataFrame:
    if roads is None or getattr(roads, "empty", True):
        return pd.DataFrame()
    preview = roads.head(rows).copy()
    preview["geometry_type"] = preview.geometry.geom_type
    preview["geometry_wkt"] = preview.geometry.apply(lambda geom: geom.wkt[:180] if geom is not None else "")
    return pd.DataFrame(preview.drop(columns="geometry", errors="ignore"))
