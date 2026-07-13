"""Tests for the economically-driven leaf-branch SHS screens (Cases B and D).

Covers the Phase 6-7 behaviour: candidates are terminal subtrees, the gate is purely
economic (thresholds are diagnostic), and Case D draws its SHS set from per-zone subtrees.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.workflow.mvp_runner import _select_standalone_clusters, detect_mvlv_subtree_candidates


def _selection_ctx(inputs):
    return {"customers": inputs["customers"], "packages": inputs["standalone_packages"],
            "T": 15, "discount": 0.08, "log": []}


# --- The shared selection helper is purely economic, with no threshold gate ---------------

def test_select_standalone_screen_is_purely_economic(mvp, inputs):
    b = mvp.cases["B_LV_plus_standalone"]
    ids = list(b.standalone_customer_ids)[:2]
    if not ids:
        pytest.skip("fixture selected no LV stand-alone customers to reuse")
    ctx = _selection_ctx(inputs)
    cand = SimpleNamespace(cluster_id="X", customer_ids=ids, standalone_eligible_precheck=True)

    # Avoided cost dominates SHS NPC -> selected.
    sel, assign, npc = _select_standalone_clusters(ctx, [cand], lambda c, x: 1e15, "Arch", "CID")
    assert set(sel) == set(ids) and npc > 0
    assert all(a["architecture"] == "Arch" and a["cluster_id"] == "X" and a["status"] == "standalone"
               for a in assign.values())

    # Zero avoided cost -> grid cheaper -> kept.
    sel2, _, npc2 = _select_standalone_clusters(ctx, [cand], lambda c, x: 0.0, "Arch", "CID")
    assert sel2 == [] and npc2 == 0.0

    # Eligibility precheck (feasibility) is the only non-economic gate.
    blocked = SimpleNamespace(cluster_id="Y", customer_ids=ids, standalone_eligible_precheck=False)
    sel3, _, _ = _select_standalone_clusters(ctx, [blocked], lambda c, x: 1e15, "Arch", "CID")
    assert sel3 == []


def test_select_standalone_greedy_no_double_claim(mvp, inputs):
    # Nested candidates: a parent subtree and its sub-branch share customers. The higher-avoided
    # (parent) is evaluated first, claims the customers, and the overlapping child is skipped.
    ids = list(mvp.cases["B_LV_plus_standalone"].standalone_customer_ids)[:2]
    if len(ids) < 2:
        pytest.skip("need >= 2 LV stand-alone customers")
    ctx = _selection_ctx(inputs)
    parent = SimpleNamespace(cluster_id="P", customer_ids=ids, standalone_eligible_precheck=True)
    child = SimpleNamespace(cluster_id="C", customer_ids=ids[:1], standalone_eligible_precheck=True)
    sel, assign, _ = _select_standalone_clusters(
        ctx, [child, parent], lambda c, x: 1e15 if c.cluster_id == "P" else 1e14, "Arch", "CID")
    assert set(sel) == set(ids)
    assert all(a["cluster_id"] == "P" for a in assign.values())  # parent took both; child skipped


# --- Per-zone MV/LV subtree detection -----------------------------------------------------

def test_detect_mvlv_subtree_candidates_empty_without_plan(inputs):
    ctx = {"customers": inputs["customers"], "params": inputs["cost_parameters"],
           "T": 15, "lv_conductors": inputs["lv_conductors"]}
    assert detect_mvlv_subtree_candidates(SimpleNamespace(plan=None, conductor_by_line={}), ctx) == []
    assert detect_mvlv_subtree_candidates(SimpleNamespace(plan=SimpleNamespace(zones=[]), conductor_by_line={}), ctx) == []


# --- Case D draws SHS from per-zone subtrees, not from Case B -----------------------------

def test_case_d_uses_its_own_subtree_screen(mvp):
    b = mvp.cases["B_LV_plus_standalone"]
    d = mvp.cases["D_MVLV_plus_standalone"]

    for a in d.standalone_assignments.values():
        assert a["architecture"] == "MV/LV + Stand-alone"
        assert a["cluster_id"].startswith("Z") and "-S" in a["cluster_id"]   # per-zone subtree id
    for a in b.standalone_assignments.values():
        assert a["cluster_id"].startswith("CL")                              # Full-LV subtree id

    assert d.n_standalone_customers == len(d.standalone_customer_ids)
    assert set(d.standalone_assignments) == set(d.standalone_customer_ids)


def test_thresholds_are_diagnostic_not_gating(mvp):
    # LV candidates are generated for every terminal subtree, irrespective of dmax/lmax/voltage;
    # selection is logged with the economic reason only.
    assert len(mvp.clusters) >= 1
    log = mvp.decision_log
    chosen = log[(log["step"] == "standalone") & (log["decision"] == "standalone")]
    if not chosen.empty:
        assert (chosen["reason"] == "NPC_SA < avoided grid cost").all()
