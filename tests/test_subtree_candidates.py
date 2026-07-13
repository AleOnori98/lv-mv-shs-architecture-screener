"""Unit tests for terminal-subtree generation and scoring (Phase 6).

These use tiny hand-built radial graphs so the expected subtrees are obvious, with no
power-flow or example data involved.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest

from core.topology.subtree_candidates import terminal_subtrees
from core.topology.branch_detection import subtree_clusters
from core.workflow.mvp_runner import detect_mvlv_subtree_candidates


def _toy_graph() -> nx.Graph:
    #            0 --L01-- 1 --L12-- 2 --L23-- 3(leaf, cust c3)
    #                                  \--L24-- 4 --L45-- 5(leaf, cust c5)
    g = nx.Graph()
    g.add_edge(0, 1, line_id="L01", length_m=10.0)
    g.add_edge(1, 2, line_id="L12", length_m=20.0)
    g.add_edge(2, 3, line_id="L23", length_m=30.0)
    g.add_edge(2, 4, line_id="L24", length_m=40.0)
    g.add_edge(4, 5, line_id="L45", length_m=50.0)
    return g


def _toy_topology() -> SimpleNamespace:
    return SimpleNamespace(
        graph=_toy_graph(), slack_pole_id=0,
        pole_xy={i: (float(i), 0.0) for i in range(6)},
        pole_customers={3: ["c3"], 5: ["c5"]},
        service_drop_m={"c3": 5.0, "c5": 7.0},
    )


def _toy_customers() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["c3", "c5"],
        "peak_kw_now": [1.0, 1.0],
        "growth_rate": [0.0, 0.0],
        "annual_kwh_now": [100.0, 100.0],
        "must_connect_flag": [False, False],
        "can_be_standalone_flag": [True, True],
    })


CP = {"civil_logistics_multiplier": 1.0, "dmax_lv_service_zone_m": 1e9, "lmax_lv_path_m": 1e9,
      "branch_cost_per_customer_threshold": 1e9, "voltage_min_pu": 0.9}
COND = {lid: {"cost_per_m": 1.0, "conductor_id": "LV1"} for lid in ["L01", "L12", "L23", "L24", "L45"]}


# --- terminal_subtrees -------------------------------------------------------------------

def test_terminal_subtrees_split_at_fork():
    sts = terminal_subtrees(_toy_graph(), 0)
    assert sorted(tuple(s.poles) for s in sts) == [(3,), (4, 5)]
    by_root = {s.branch_root: s for s in sts}
    assert by_root[3].boundary_node == 2 and by_root[3].entry_edge_id == "L23" and by_root[3].internal_edge_ids == []
    assert by_root[4].boundary_node == 2 and by_root[4].entry_edge_id == "L24" and by_root[4].internal_edge_ids == ["L45"]


def test_terminal_subtrees_pure_chain_has_none():
    g = nx.Graph()
    g.add_edge(0, 1, line_id="a", length_m=1.0)
    g.add_edge(1, 2, line_id="b", length_m=1.0)
    assert terminal_subtrees(g, 0) == []  # no fork -> the trunk is never an SHS candidate


def test_terminal_subtrees_root_fork():
    g = nx.Graph()
    g.add_edge(0, 1, line_id="a", length_m=1.0)
    g.add_edge(0, 2, line_id="b", length_m=1.0)
    sts = terminal_subtrees(g, 0)
    assert sorted(tuple(s.poles) for s in sts) == [(1,), (2,)]
    assert all(s.boundary_node == 0 for s in sts)


def test_terminal_subtrees_empty_graph():
    assert terminal_subtrees(nx.Graph(), 0) == []


# --- subtree_clusters (scoring + marginal cost) ------------------------------------------

def test_subtree_clusters_builds_costed_candidates():
    cls = subtree_clusters(_toy_topology(), _toy_customers(), conductor_by_line=COND,
                           cost_params=CP, horizon_years=15, largest_conductor_id="BIG", full_model=None)
    assert len(cls) == 2
    by_cust = {tuple(c.customer_ids): c for c in cls}

    tail = by_cust[("c5",)]
    assert sorted(tail.pole_ids) == [4, 5]
    assert sorted(tail.branch_edge_ids) == ["L24", "L45"]        # internal L45 + entry edge L24
    assert tail.branch_cost == pytest.approx(40.0 + 50.0)        # mult=1, cost_per_m=1
    assert tail.metadata["service_drop_m"] == 7.0
    assert tail.standalone_eligible_precheck is True
    assert tail.branch_losses_kw == 0.0                          # no power flow supplied

    stub = by_cust[("c3",)]
    assert stub.branch_edge_ids == ["L23"] and stub.branch_cost == pytest.approx(30.0)


def test_subtree_clusters_precheck_blocks_must_connect():
    df = _toy_customers()
    df.loc[df["customer_id"] == "c5", "must_connect_flag"] = True
    cls = subtree_clusters(_toy_topology(), df, conductor_by_line=COND, cost_params=CP,
                           horizon_years=15, largest_conductor_id="BIG", full_model=None)
    tail = next(c for c in cls if c.customer_ids == ["c5"])
    assert tail.standalone_eligible_precheck is False


# --- per-zone detection namespaces candidates by zone ------------------------------------

def test_detect_mvlv_subtree_candidates_namespaced_per_zone():
    zone = {"zone_id": "Z00", "topology": _toy_topology()}
    mvlv_res = SimpleNamespace(plan=SimpleNamespace(zones=[zone]),
                               conductor_by_line={f"Z00::{lid}": COND[lid] for lid in COND})
    lv_df = pd.DataFrame({"rank": [0, 1], "conductor_id": ["LV0", "LV1"], "cost_per_m": [1.0, 2.0]})
    ctx = {"customers": _toy_customers(), "params": CP, "T": 15, "lv_conductors": lv_df}

    cands = detect_mvlv_subtree_candidates(mvlv_res, ctx)
    assert len(cands) == 2
    assert all(c.cluster_id.startswith("Z00-S") for c in cands)
    assert {tuple(c.customer_ids) for c in cands} == {("c3",), ("c5",)}
