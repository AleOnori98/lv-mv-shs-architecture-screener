"""Wrapper around the existing topology engine, normalized for the MVP pipeline.

This module does NOT re-implement pole placement or MST routing — it calls
``core.distribution_service._run_low_voltage_core`` (the same engine used by the legacy
app) and repackages the result into an :class:`LVTopology` with stable ``customer_id``
keys, a rooted pole graph, and per-customer service-drop lengths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

from core.distribution_service import _run_low_voltage_core
from . import graph_utils as gu


@dataclass
class LVTopology:
    poles_4326: gpd.GeoDataFrame
    edges_4326: gpd.GeoDataFrame
    buildings_4326: gpd.GeoDataFrame
    graph: nx.Graph
    slack_pole_id: int
    pole_xy: Dict[int, tuple]
    customer_pole: Dict[str, int]
    pole_customers: Dict[int, List[str]]
    service_drop_m: Dict[str, float]
    served_customer_ids: List[str]
    unserved_customer_ids: List[str]
    center: tuple
    crs_epsg: int
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_lv_length_m(self) -> float:
        """Total backbone length [m] (excludes service drops, which carry no impedance)."""
        return float(self.edges_4326["length_m"].sum()) if not self.edges_4326.empty else 0.0

    @property
    def total_service_drop_m(self) -> float:
        """Total pole-to-customer drop length [m] over the served customers.

        A property so the metres that are CHARGED and the metres that are REPORTED are the
        same expression; they were written out six times, twice within one function.
        """
        return float(sum(self.service_drop_m.get(c, 0.0) for c in self.served_customer_ids))


def build_lv_topology(
    customers_engine_gdf: gpd.GeoDataFrame,
    roads_gdf: Optional[gpd.GeoDataFrame],
    crs_meta: Dict[str, Any],
    source: Dict[str, Any],
    eng: Dict[str, Any],
    *,
    customer_ids: Optional[List[str]] = None,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
    edge_weight_mode: str = "euclidean",
) -> LVTopology:
    """Generate one LV topology (optionally restricted to ``customer_ids``)."""
    gdf = customers_engine_gdf
    if customer_ids is not None:
        wanted = set(customer_ids)
        gdf = gdf[gdf["customer_id"].isin(wanted)].copy()
    if gdf.empty:
        raise ValueError("No customers available to build an LV topology.")

    result = _run_low_voltage_core(
        gdf_buildings=gdf,
        gdf_roads=roads_gdf,
        crs_meta=crs_meta,
        sampling_distance=float(eng["road_pole_spacing_m"]),
        user_distance=float(eng["max_user_connection_radius_m"]),
        max_associations=int(eng["max_users_per_pole"]),
        # Default MVP behaviour is Euclidean backbone weighting with road-seeded pole placement.
        edge_weight_mode=edge_weight_mode,
        allow_unserved_isolated=bool(allow_unserved_isolated),
        min_cluster_size=int(min_cluster_size),
        max_pole_span_m=eng.get("max_pole_span_m"),
    )
    return _package_topology(result, crs_meta, source)


def _package_topology(result: Dict[str, Any], crs_meta: Dict[str, Any], source: Dict[str, Any]) -> LVTopology:
    crs_epsg = int(crs_meta.get("projected_crs_epsg") or 32633)
    poles_4326: gpd.GeoDataFrame = result["gdf_poles_4326"].copy()
    edges_4326: gpd.GeoDataFrame = result["gdf_edges_4326"]
    buildings_4326: gpd.GeoDataFrame = result["gdf_buildings_4326"].copy()
    associations: pd.DataFrame = result["associations_df"]

    if "pole_id" not in poles_4326.columns:
        poles_4326["pole_id"] = poles_4326.index.astype(int)
    poles_4326["pole_id"] = poles_4326["pole_id"].astype(int)

    # Projected pole coordinates for distance work.
    poles_proj = poles_4326.to_crs(epsg=crs_epsg)
    pole_xy = {int(p): (float(g.x), float(g.y)) for p, g in zip(poles_proj["pole_id"], poles_proj.geometry)}

    edges_df = edges_4326 if (edges_4326 is not None and not edges_4326.empty) else pd.DataFrame(columns=["line_id", "u", "v", "length_m"])
    graph = gu.build_pole_graph(edges_df)
    for pid in pole_xy:  # ensure single-pole / isolated poles exist as nodes
        graph.add_node(int(pid))

    # Slack pole = nearest pole to the source point (projected).
    src_pt = gpd.GeoSeries.from_xy([source["longitude"]], [source["latitude"]], crs="EPSG:4326").to_crs(epsg=crs_epsg).iloc[0]
    slack_pole_id = gu.nearest_node(pole_xy, (float(src_pt.x), float(src_pt.y)))

    # building_id -> customer_id (engine index keys).
    bid_to_cid = buildings_4326["customer_id"].astype(str).to_dict()
    buildings_proj = buildings_4326.to_crs(epsg=crs_epsg)
    bid_to_geom = buildings_proj.geometry.to_dict()

    customer_pole: Dict[str, int] = {}
    pole_customers: Dict[int, List[str]] = {}
    service_drop_m: Dict[str, float] = {}
    if associations is not None and not associations.empty:
        for pid, bid in associations[["pole_id", "building_id"]].itertuples(index=False):
            pid, bid = int(pid), int(bid)
            cid = bid_to_cid.get(bid)
            if cid is None:
                continue
            customer_pole[cid] = pid
            pole_customers.setdefault(pid, []).append(cid)
            geom = bid_to_geom.get(bid)
            xy = pole_xy.get(pid)
            service_drop_m[cid] = float(geom.distance(Point(*xy))) if (geom is not None and xy) else 0.0

    served = sorted(customer_pole.keys())
    all_cids = set(buildings_4326["customer_id"].astype(str))
    unserved = sorted(all_cids - set(served))

    return LVTopology(
        poles_4326=poles_4326,
        edges_4326=edges_4326 if edges_4326 is not None else gpd.GeoDataFrame(),
        buildings_4326=buildings_4326,
        graph=graph,
        slack_pole_id=int(slack_pole_id),
        pole_xy=pole_xy,
        customer_pole=customer_pole,
        pole_customers=pole_customers,
        service_drop_m=service_drop_m,
        served_customer_ids=served,
        unserved_customer_ids=unserved,
        center=tuple(result.get("center") or (source["latitude"], source["longitude"])),
        crs_epsg=crs_epsg,
        metrics=dict(result.get("metrics", {})),
        metadata=dict(result.get("metadata", {})),
        raw=result,
    )
