"""The transformer-count criterion and the transformer-sizing coincidence basis.

Covers the two §3.4/§3.5 defects:
  * the k-search assumed feasibility was monotone in k while k-means re-partitioned from
    scratch at every k, so the zones at k+1 bore no relation to those at k;
  * the climb loop was bounded only by ``max_transformers`` and crashed for a settlement
    with fewer customers than that cap;
  * transformers were sized on the coincident peak but validated at the diversified one,
    so the rating was 1/cf larger than the snapshot it was checked against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.topology.mvlv_service_zones import (
    partition_hierarchical_k,
    partition_kmeans,
    select_transformer,
    zone_apparent_power_kva,
)


def _three_blobs(per=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal((0.0, 0.0), 8.0, size=(per, 2)),
        rng.normal((400.0, 0.0), 8.0, size=(per, 2)),
        rng.normal((0.0, 400.0), 8.0, size=(per, 2)),
    ])


def _labels_to_sets(labels):
    out = {}
    for i, lab in enumerate(np.asarray(labels)):
        out.setdefault(int(lab), set()).add(i)
    return list(out.values())


def test_hierarchical_cut_returns_exactly_k_clusters():
    coords = _three_blobs()
    for k in range(1, 9):
        labels = partition_hierarchical_k(coords, k)
        assert len(np.unique(labels)) == k, k
        assert labels.shape == (coords.shape[0],)


def test_hierarchical_cut_is_nested_over_k():
    """The property the k-search depends on: k+1 REFINES k, it does not reshuffle it.

    Every cluster at k+1 must be a subset of some cluster at k. k-means does not have
    this property, which is why the worst voltage drop was not monotone in k.
    """
    coords = _three_blobs()
    for k in range(1, 10):
        coarse = _labels_to_sets(partition_hierarchical_k(coords, k))
        finer = _labels_to_sets(partition_hierarchical_k(coords, k + 1))
        for child in finer:
            assert any(child <= parent for parent in coarse), (
                f"cluster {sorted(child)} at k={k+1} is not contained in any k={k} cluster"
            )


def _uniform_village(n=60, extent=600.0, seed=11):
    """A continuous point cloud, like a real settlement -- not pre-separated blobs.

    Well-separated blobs are the easy case: k-means finds the obvious structure and
    happens to look nested. A uniform cloud is where its centroids genuinely move as k
    changes, which is the geometry the k-search actually meets.
    """
    return np.random.default_rng(seed).uniform(0.0, extent, size=(n, 2))


def _count_non_nested(partition_at):
    violations = 0
    for k in range(2, 10):
        coarse = _labels_to_sets(partition_at(k))
        for child in _labels_to_sets(partition_at(k + 1)):
            if not any(child <= parent for parent in coarse):
                violations += 1
    return violations


def test_kmeans_is_not_nested_but_the_ward_cut_is():
    """Documents WHY the partitioner changed, on the same point cloud.

    k-means re-solves from scratch at every k, so going from k to k+1 moves customers
    between unrelated zones; the Ward cut only ever splits one zone.
    """
    coords = _uniform_village()
    kmeans_violations = _count_non_nested(lambda k: partition_kmeans(coords, k, seed=7))
    ward_violations = _count_non_nested(lambda k: partition_hierarchical_k(coords, k))
    assert kmeans_violations > 0
    assert ward_violations == 0


def test_hierarchical_cut_recovers_well_separated_groups():
    labels = partition_hierarchical_k(_three_blobs(), 3)
    sizes = sorted(len(g) for g in _labels_to_sets(labels))
    assert sizes == [12, 12, 12]


def test_hierarchical_cut_rejects_more_clusters_than_points():
    with pytest.raises(ValueError, match="only 4 customers"):
        partition_hierarchical_k(np.random.default_rng(0).random((4, 2)) * 100, 7)


def test_hierarchical_cut_degenerate_inputs():
    assert partition_hierarchical_k(np.zeros((1, 2)), 1).tolist() == [0]
    assert partition_hierarchical_k(np.zeros((5, 2)), 1).tolist() == [0] * 5


# --------------------------------------------------------------------------------------
# Transformer sizing coincidence basis
# --------------------------------------------------------------------------------------
def _customers(n=10, peak=5.0, pf=0.95, growth=0.0):
    return pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "peak_kw_now": [peak] * n,
        "growth_rate": [growth] * n,
        "power_factor": [pf] * n,
    })


def test_zone_apparent_power_scales_with_the_coincidence_basis():
    df = _customers(n=10, peak=5.0)
    ids = list(df["customer_id"])
    coincident = zone_apparent_power_kva(ids, df, 0, 1.0)
    diversified = zone_apparent_power_kva(ids, df, 0, 0.6)
    assert diversified == pytest.approx(0.6 * coincident)
    # 10 x 5 kW at pf 0.95 -> 50 kW / 0.95 ~ 52.6 kVA coincident
    assert coincident == pytest.approx(50.0 / 0.95, rel=1e-6)


def test_zone_apparent_power_applies_growth_then_diversity():
    df = _customers(n=4, peak=1.0, pf=1.0, growth=0.05)
    ids = list(df["customer_id"])
    assert zone_apparent_power_kva(ids, df, 10, 0.5) == pytest.approx(4 * 1.05 ** 10 * 0.5)


def test_sizing_basis_sets_whether_the_loading_limit_can_bind():
    """Quantifies the trade-off behind ``transformer_sizing_coincidence_factor``.

    Sizing on the coincident peak while validating at the diversified one makes the
    transformer 1/cf larger than the snapshot it is checked against, so its loading peaks
    well below the limit and ``max_transformer_loading_percent`` cannot bind. Tracking the
    design factor instead makes the constraint live -- but picks a physically smaller unit,
    whose per-unit impedance is higher, which costs LV voltage. Neither is free; the
    parameter makes the choice explicit rather than hard-coded.
    """
    catalogue = pd.DataFrame({
        "transformer_id": ["TX_100", "TX_200", "TX_315", "TX_500"],
        "sn_kva": [100.0, 200.0, 315.0, 500.0],
        "max_loading_percent": [100.0] * 4,
        "rank": [1, 2, 3, 4],
    })
    df = _customers(n=40, peak=5.0, pf=1.0)      # 200 kW of coincident peak
    ids = list(df["customer_id"])
    cf = 0.6

    s_design = zone_apparent_power_kva(ids, df, 0, cf)          # 120 kVA
    consistent = select_transformer(s_design, catalogue)         # target 144 -> TX_200
    load_consistent = 100.0 * s_design / float(consistent["sn_kva"])

    s_coincident = zone_apparent_power_kva(ids, df, 0, 1.0)      # 200 kVA
    old = select_transformer(s_coincident, catalogue)             # target 240 -> TX_315
    load_old = 100.0 * s_design / float(old["sn_kva"])            # checked at the DESIGN load

    assert float(consistent["sn_kva"]) < float(old["sn_kva"])
    assert load_consistent == pytest.approx(60.0)
    assert load_old == pytest.approx(38.1, abs=0.2)
    assert load_consistent > load_old


# --------------------------------------------------------------------------------------
# The climb loop used to be bounded only by max_transformers, so a settlement with fewer
# customers than that cap raised "k clusters requested but only n customers" out of
# partition_kmeans and crashed the whole run instead of returning a result.
# --------------------------------------------------------------------------------------
def test_k_search_handles_a_settlement_smaller_than_max_transformers(inputs):
    """A tiny, hard-to-serve settlement must still produce four candidates, not an exception."""
    from core.workflow.mvp_runner import run_simplified_mvp

    customers = inputs["customers"].head(6).copy()
    # Push them far apart so no small zone count can hold the voltage, forcing the climb
    # to run all the way into the ceiling -- the exact path that used to raise.
    customers["latitude"] = customers["latitude"].to_numpy() + np.arange(len(customers)) * 0.004
    customers["must_connect_flag"] = False

    params = dict(inputs["cost_parameters"])
    params["max_transformers"] = 12          # deliberately larger than the 6 customers

    res = run_simplified_mvp(
        customers, inputs["roads"], inputs["source"], inputs["lv_conductors"],
        inputs["mv_conductors"], inputs["transformers"], inputs["standalone_packages"],
        params, {"max_user_connection_radius_m": 40, "max_users_per_pole": 12,
                 "max_pole_span_m": 45, "road_pole_spacing_m": 35},
    )

    assert set(res.cases) == {"A_full_LV", "B_LV_plus_standalone", "C_MVLV", "D_MVLV_plus_standalone"}
    ks = [int(r["object_id"].removeprefix("k=")) for _, r in res.decision_log.iterrows()
          if r["step"] == "mv_transformer_count"]
    assert ks, "the k-search recorded no attempts"
    assert max(ks) <= len(customers), f"asked for more zones ({max(ks)}) than customers"


def test_k_search_records_the_margin_it_selected_against(mvp):
    log = mvp.decision_log
    rows = log[log["step"] == "mv_transformer_count"]
    assert not rows.empty
    # Every attempt reports the voltage target it was judged against, so the accepted
    # zone count is auditable rather than implicit.
    assert rows["metric_value"].str.contains("target").all()
    assert rows["decision"].isin(
        {"accepted", "feasible_no_margin", "infeasible", "diverged"}
    ).all()


def test_smaller_transformer_has_higher_impedance_so_sizing_basis_costs_voltage():
    """Why tracking the design coincidence factor is not simply "the fix".

    Catalogue impedance is quoted in percent of the unit's own rating, so the ohmic
    impedance of a smaller unit is LARGER (vk% / sn). Sizing a zone on the diversified
    peak therefore picks a unit that drops more volts at the same load -- measured on the
    Mafa case as ~0.010 pu, enough to flip Case D from feasible to infeasible. The
    conservatism was partly buying voltage support, not purely wasted capex.
    """
    tx_200 = {"sn_kva": 200.0, "vk_percent": 4.5}
    tx_315 = {"sn_kva": 315.0, "vk_percent": 5.0}

    def z_pu_at(tx, base_kva):
        """Impedance referred to a common base: vk% scaled by base/rating."""
        return tx["vk_percent"] / 100.0 * base_kva / tx["sn_kva"]

    base = 120.0  # the diversified zone load both units would serve
    assert z_pu_at(tx_200, base) > z_pu_at(tx_315, base)
    assert z_pu_at(tx_200, base) / z_pu_at(tx_315, base) == pytest.approx(1.418, abs=0.01)


def test_default_sizing_basis_is_the_declared_conservative_one():
    """The default stays on the coincident peak, deliberately.

    coincidence_factor is a single flat SYSTEM-WIDE number and diversity only emerges with
    many aggregated customers, so applying it to a 10-customer zone understates that
    zone's peak and would under-size the small zones. The honest fix is a size-dependent
    ADMD curve; until then the basis is explicit and reported rather than hidden.
    """
    from core.io.input_schemas import normalize_cost_parameters

    params = normalize_cost_parameters({})
    assert params["transformer_sizing_coincidence_factor"] == 1.0
    assert params["coincidence_factor"] == 0.6
