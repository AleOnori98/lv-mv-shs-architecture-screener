"""Tests for the results-dashboard presentation helpers (charts + screening table)."""
from __future__ import annotations

from types import SimpleNamespace

import altair as alt
import pandas as pd

from core.viz.charts import case_headroom_html, npc_breakdown_chart, npc_share_chart, shs_decision_chart
from core.viz.tables import npc_components, shs_screening_table

LOG_COLS = ["step", "candidate_id", "object_id", "decision", "reason", "metric_name", "metric_value", "threshold"]


def _synthetic_log() -> pd.DataFrame:
    rows = [
        ("branch_detection", "A_full_LV", "CL00", "flagged", "uses largest LV conductor", "n_customers", 5, ""),
        ("branch_detection", "A_full_LV", "CL01", "clean", "within limits", "n_customers", 2, ""),
        ("standalone", "B_LV_plus_standalone", "CL00", "standalone", "NPC_SA < avoided grid cost", "npc_sa_vs_avoided", "8000 vs 12000", ""),
        ("standalone", "B_LV_plus_standalone", "CL01", "keep_grid", "grid cheaper", "npc_sa_vs_avoided", "3000 vs 1000", ""),
        ("standalone", "B_LV_plus_standalone", "CL02", "keep_grid", "no package satisfies tier/peak/energy requirement", "", "", ""),
    ]
    return pd.DataFrame(rows, columns=LOG_COLS)


def test_shs_screening_table_parses_decisions():
    df = shs_screening_table(_synthetic_log(), "B_LV_plus_standalone").set_index("cluster_id")
    assert list(df.index) == ["CL00", "CL01", "CL02"]

    assert df.loc["CL00", "outcome"] == "selected"
    assert df.loc["CL00", "shs_npc"] == 8000 and df.loc["CL00", "avoided_cost"] == 12000
    assert df.loc["CL00", "margin"] == 4000
    assert df.loc["CL00", "n_customers"] == 5
    assert df.loc["CL00", "flags"] == "uses largest LV conductor"

    assert df.loc["CL01", "outcome"] == "kept" and df.loc["CL01", "margin"] == -2000

    assert df.loc["CL02", "outcome"] == "blocked"
    assert pd.isna(df.loc["CL02", "margin"]) and pd.isna(df.loc["CL02", "shs_npc"])


def test_shs_screening_table_empty_for_unknown_candidate():
    assert shs_screening_table(_synthetic_log(), "D_MVLV_plus_standalone").empty
    assert shs_screening_table(pd.DataFrame(columns=LOG_COLS), "B_LV_plus_standalone").empty


def test_shs_decision_chart_skips_blocked_and_handles_empty():
    df = shs_screening_table(_synthetic_log(), "B_LV_plus_standalone")
    chart = shs_decision_chart(df)
    assert isinstance(chart, alt.Chart)
    assert set(chart.data["cluster_id"]) == {"CL00", "CL01"}  # blocked CL02 omitted
    assert shs_decision_chart(df[df["outcome"] == "blocked"]) is None


def test_npc_breakdown_chart_filters_feasible_and_stars_recommended():
    cbt = pd.DataFrame([
        {"candidate_id": "A_full_LV", "lv_line_cost": 100, "mv_line_cost": 0, "transformer_cost": 0, "standalone_npc": 0},
        {"candidate_id": "B_LV_plus_standalone", "lv_line_cost": 80, "mv_line_cost": 0, "transformer_cost": 0, "standalone_npc": 20},
        {"candidate_id": "C_MVLV", "lv_line_cost": 200, "mv_line_cost": 50, "transformer_cost": 30, "standalone_npc": 0},
    ])
    labels = {"A_full_LV": "A · Full LV", "B_LV_plus_standalone": "B · LV + SA", "C_MVLV": "C · MV/LV"}
    chart = npc_breakdown_chart(cbt, "B_LV_plus_standalone", labels, ["A_full_LV", "B_LV_plus_standalone"])
    cands = set(chart.data["candidate"])
    assert "⭐ B · LV + SA" in cands and "A · Full LV" in cands
    assert not any("MV/LV" in c for c in cands)   # infeasible C excluded
    assert (chart.data["cost"] == 0).sum() == 0   # zero components dropped


def test_npc_components_and_share_chart():
    cbt = pd.DataFrame([{"candidate_id": "B", "lv_line_cost": 80, "mv_line_cost": 0, "standalone_npc": 20, "loss_pv_cost": 0}])
    comp = npc_components(cbt, "B")
    assert list(comp["component"]) == ["LV line", "Stand-alone NPC"]   # zeros dropped, sorted desc
    assert comp.loc[comp["component"] == "LV line", "share_%"].iloc[0] == 80.0
    assert isinstance(npc_share_chart(comp), alt.Chart)
    assert npc_share_chart(npc_components(cbt, "missing")) is None


def test_case_headroom_html_colors_and_omits_missing_bars():
    constraints = {"voltage_min_pu": 0.90, "max_line_loading_percent": 100.0, "max_transformer_loading_percent": 100.0}

    ok = case_headroom_html(SimpleNamespace(min_vm_pu=0.94, max_line_loading_percent=83.0, max_trafo_loading_percent=78.0), constraints)
    assert "Min voltage" in ok and "Max line loading" in ok and "Max transformer loading" in ok
    assert "⚠" not in ok

    violation = case_headroom_html(SimpleNamespace(min_vm_pu=0.88, max_line_loading_percent=120.0, max_trafo_loading_percent=float("nan")), constraints)
    assert "⚠" in violation                       # 0.88 pu is below the 0.90 floor
    assert "Max transformer loading" not in violation  # NaN transformer loading -> bar omitted (pure-LV case)

    assert case_headroom_html(None, constraints) == ""


def test_screening_helpers_run_on_real_result(mvp):
    for cand in ("B_LV_plus_standalone", "D_MVLV_plus_standalone"):
        screen = shs_screening_table(mvp.decision_log, cand)
        assert not screen.empty
        assert set(["cluster_id", "shs_npc", "avoided_cost", "margin", "outcome"]).issubset(screen.columns)
        assert set(screen["outcome"]).issubset({"selected", "kept", "blocked"})

    feasible = [k for k, c in mvp.cases.items() if c.feasible]
    chart = npc_breakdown_chart(mvp.cost_breakdown_table, mvp.recommended, {k: k for k in mvp.cases}, feasible)
    assert isinstance(chart, alt.Chart)

    # New per-candidate breakdown metrics are populated by the workflow.
    full = mvp.cases["A_full_LV"]
    assert full.n_poles > 0 and full.service_drop_total_m >= 0.0
