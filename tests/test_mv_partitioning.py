"""Unit tests for the criterion-driven MV partitioning helpers.

Pure-geometry checks (no power flow): the distance-cap diameter guarantee, the
kVA-cap enforcer that keeps zone transformers procurable under every criterion,
deterministic k-means, the voltage-cap seed, and the plant step-up sizing.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from core.topology.mvlv_service_zones import (
    cluster_diameters_m,
    enforce_kva_cap,
    partition_complete_linkage,
    partition_kmeans,
    size_stepup_kva,
    suggest_zone_start,
    zone_apparent_power_kva,
)


def _synthetic(per_blob: int = 20, spacing_m: float = 1000.0, spread_m: float = 40.0, seed: int = 0):
    """Three tight blobs `spacing_m` apart (projected metres)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [spacing_m, 0.0], [0.0, spacing_m]])
    pts = np.vstack([c + rng.normal(0.0, spread_m, size=(per_blob, 2)) for c in centers])
    ids = [f"c{i}" for i in range(len(pts))]
    gdf = gpd.GeoDataFrame(
        {"customer_id": ids}, geometry=[Point(float(x), float(y)) for x, y in pts], crs="EPSG:32633"
    )
    customers = pd.DataFrame(
        {"customer_id": ids, "peak_kw_now": 1.0, "growth_rate": 0.0, "power_factor": 0.95}
    )
    return pts, ids, gdf, customers


def test_complete_linkage_guarantees_diameter_cap():
    pts, _, _, _ = _synthetic()
    cap = 300.0
    labels = partition_complete_linkage(pts, cap)
    diams = cluster_diameters_m(pts, labels)
    assert diams, "expected at least one cluster"
    assert max(diams.values()) <= cap + 1e-6
    # The three 1 km-separated blobs cannot share a cluster under a 300 m cap.
    assert len(diams) >= 3


def test_complete_linkage_loosens_with_a_wider_cap():
    pts, _, _, _ = _synthetic()
    tight = len(np.unique(partition_complete_linkage(pts, 300.0)))
    loose = len(np.unique(partition_complete_linkage(pts, 5000.0)))
    assert loose < tight  # a 5 km cap merges the blobs into fewer clusters


def test_partition_kmeans_returns_exactly_k_nonempty_clusters():
    pts, _, _, _ = _synthetic()
    labels = partition_kmeans(pts, 4, seed=42)
    assert len(np.unique(labels)) == 4
    # Deterministic given the seed.
    assert np.array_equal(labels, partition_kmeans(pts, 4, seed=42))


def test_enforce_kva_cap_never_leaves_a_multi_customer_zone_over_cap():
    _, ids, gdf, customers = _synthetic(per_blob=20)
    total_kva = zone_apparent_power_kva(ids, customers, horizon_years=0)
    cap = total_kva / 8.0  # force several splits
    out = enforce_kva_cap([ids], gdf, customers, horizon_years=0, max_zone_kva=cap, seed=42)
    assert sum(len(g) for g in out) == len(ids)  # partition, nothing lost or duplicated
    for g in out:
        s = zone_apparent_power_kva(g, customers, horizon_years=0)
        assert s <= cap + 1e-6 or len(g) == 1  # singletons may exceed an unrealistically tight cap


def test_suggest_zone_start_reflects_geometry():
    pts, _, _, _ = _synthetic()
    # Blobs 1 km apart: a 500 m "hopeless" diameter cannot merge them -> >= 3 zones.
    assert suggest_zone_start(pts, hopeless_diameter_m=500.0) >= 3
    # A very large hopeless diameter collapses everything to a single seed zone.
    assert suggest_zone_start(pts, hopeless_diameter_m=100000.0) == 1


def test_size_stepup_kva_snaps_to_catalogue_then_rounds_up():
    catalogue = pd.DataFrame({"sn_kva": [100.0, 200.0, 315.0, 500.0]})
    # Within the catalogue: pick the next standard size covering target = S * margin.
    assert size_stepup_kva(150.0, 1.2, catalogue) == 200.0  # 150 * 1.2 = 180 -> 200
    # Above the largest catalogue entry: round up to the next 50 kVA (project-procured unit).
    assert size_stepup_kva(500.0, 1.2, catalogue) == 600.0  # 500 * 1.2 = 600


def test_partition_kmeans_rejects_more_clusters_than_points():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        partition_kmeans(pts, 5)
