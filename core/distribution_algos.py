from __future__ import annotations

import io
from typing import Dict, List, Tuple, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import math
from shapely.geometry import Point, LineString

from config.settings import (
    DEFAULT_EDGE_WEIGHT_MODE,
    DEFAULT_PRIORITY_SCORE,
    GEOMETRY_EPSILON_M,
    MAX_PRIORITY_SCORE,
    MIN_PRIORITY_SCORE,
    PRE_MST_DEDUPLICATION_TOLERANCE_M,
    ROAD_ATTRACTION_BUFFER_M,
    ROAD_EDGE_SAMPLE_SPACING_M,
    ROAD_OFFROAD_PENALTY_FACTOR,
)


def _priority_score_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    if "priority_score" not in gdf.columns:
        return pd.Series(int(DEFAULT_PRIORITY_SCORE), index=gdf.index, dtype=int)
    numeric = pd.to_numeric(gdf["priority_score"], errors="coerce")
    rounded = pd.Series(np.nan, index=gdf.index, dtype=float)
    mask = numeric.notna()
    rounded.loc[mask] = np.where(
        numeric.loc[mask] >= 0,
        np.floor(numeric.loc[mask] + 0.5),
        np.ceil(numeric.loc[mask] - 0.5),
    )
    return (
        rounded.fillna(int(DEFAULT_PRIORITY_SCORE))
        .clip(lower=int(MIN_PRIORITY_SCORE), upper=int(MAX_PRIORITY_SCORE))
        .astype(int)
    )


# ---------------------------------------------------------------------
# Sampling & association algorithms
# ---------------------------------------------------------------------

def sample_points_along_line(line: LineString, sampling_distance: float) -> List[Point]:
    """
    Sample points along a line geometry at regular intervals.

    Parameters
    ----------
    line : shapely LineString
    sampling_distance : float
        Distance between successive sample points (in CRS units, e.g. meters).
    """
    points: List[Point] = []
    current_distance = 0.0
    while current_distance < line.length:
        points.append(line.interpolate(current_distance))
        current_distance += sampling_distance
    return points


def collect_sampled_points(gdf_roads: gpd.GeoDataFrame, sampling_distance: float) -> List[Point]:
    """
    Collect candidate pole locations from road geometries.

    Strategy
    --------
    - Take start and end point of each road segment.
    - If the segment is long, sample additional points every `sampling_distance`.
    """
    sampled_points: List[Point] = []
    for _, row in gdf_roads.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type == "LineString":
            line_geoms = [geometry]
        elif geometry.geom_type == "MultiLineString":
            line_geoms = list(geometry.geoms)
        else:
            continue

        for line in line_geoms:
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            # Always add endpoints
            sampled_points.append(Point(coords[0]))
            sampled_points.append(Point(coords[-1]))

            # Additional sampling along the line
            if line.length > sampling_distance:
                sampled_points.extend(sample_points_along_line(line, sampling_distance))

    return sampled_points


def _iter_road_lines(gdf_roads: gpd.GeoDataFrame | None) -> List[LineString]:
    if gdf_roads is None or gdf_roads.empty:
        return []

    lines: List[LineString] = []
    for geometry in gdf_roads.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            lines.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            lines.extend([line for line in geometry.geoms if line is not None and not line.is_empty])
    return lines


def _dedupe_consecutive_points(points: List[Point]) -> List[Point]:
    deduped: List[Point] = []
    for point in points:
        if point is None or point.is_empty:
            continue
        if deduped and float(deduped[-1].distance(point)) <= float(GEOMETRY_EPSILON_M):
            continue
        deduped.append(point)
    return deduped


def _line_substring(line: LineString, start_m: float, end_m: float) -> LineString:
    """Return a LineString section between two chainage distances."""
    if line.is_empty:
        return line

    start = max(0.0, min(float(start_m), float(line.length)))
    end = max(0.0, min(float(end_m), float(line.length)))
    if end < start:
        start, end = end, start

    if abs(end - start) <= float(GEOMETRY_EPSILON_M):
        point = line.interpolate(start)
        return LineString([point, point])

    coords = list(line.coords)
    out: List[Point] = [line.interpolate(start)]
    traversed = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        p_a = Point(a)
        p_b = Point(b)
        seg_len = float(p_a.distance(p_b))
        if seg_len <= float(GEOMETRY_EPSILON_M):
            continue
        seg_start = traversed
        seg_end = traversed + seg_len
        if seg_end > start + float(GEOMETRY_EPSILON_M) and seg_start < end - float(GEOMETRY_EPSILON_M):
            if start < seg_start < end:
                out.append(p_a)
            if start < seg_end < end:
                out.append(p_b)
        traversed = seg_end

    out.append(line.interpolate(end))
    out = _dedupe_consecutive_points(out)
    if len(out) == 1:
        out.append(out[0])
    return LineString(out)


def _road_node_key(point: Point) -> tuple[float, float]:
    # Millimetre precision is enough to merge exact road endpoints in projected CRS.
    return (round(float(point.x), 3), round(float(point.y), 3))


def _points_from_intersection_geometry(geometry: Any) -> List[Point]:
    if geometry is None or geometry.is_empty:
        return []
    geom_type = str(getattr(geometry, "geom_type", ""))
    if geom_type == "Point":
        return [geometry]
    if geom_type == "MultiPoint":
        return [point for point in geometry.geoms if point is not None and not point.is_empty]
    if geom_type in {"LineString", "LinearRing"}:
        coords = list(geometry.coords)
        if not coords:
            return []
        return [Point(coords[0]), Point(coords[-1])]
    if geom_type in {"MultiLineString", "GeometryCollection"}:
        points: List[Point] = []
        for part in geometry.geoms:
            points.extend(_points_from_intersection_geometry(part))
        return points
    return []


def _intersection_points_by_road_line(
    lines: List[LineString],
    *,
    crs: Any = None,
) -> Dict[int, List[Point]]:
    points_by_idx: Dict[int, List[Point]] = {idx: [] for idx in range(len(lines))}
    if len(lines) < 2:
        return points_by_idx

    try:
        lines_gdf = gpd.GeoDataFrame({"line_idx": list(range(len(lines)))}, geometry=lines, crs=crs)
        spatial_index = lines_gdf.sindex
        for idx, line in enumerate(lines):
            hits = spatial_index.query(line, predicate="intersects")
            for hit in hits:
                hit_idx = int(hit)
                if hit_idx <= idx:
                    continue
                other = lines[hit_idx]
                if line is other:
                    continue
                intersection = line.intersection(other)
                points = _points_from_intersection_geometry(intersection)
                if not points:
                    continue
                points_by_idx[idx].extend(points)
                points_by_idx[hit_idx].extend(points)
    except Exception:
        # Spatial-index availability varies across GeoPandas/Shapely builds.
        # The caller still gets sampled endpoints and near-node snapping below.
        return points_by_idx
    return points_by_idx


def _build_sampled_road_graph(
    gdf_roads: gpd.GeoDataFrame | None,
    *,
    sample_spacing_m: float = ROAD_EDGE_SAMPLE_SPACING_M,
    snap_tolerance_m: float = 5.0,
) -> tuple[nx.Graph | None, Dict[tuple[float, float], Point]]:
    """
    Build a lightweight road graph for planning-level route densification.

    Roads are discretized at `sample_spacing_m` while preserving original
    vertices and detected line intersections. Exact duplicate sampled points
    become shared graph nodes, and a small snap tolerance connects near-miss road
    nodes that are visually connected but not topologically identical.
    """
    lines = _iter_road_lines(gdf_roads)
    if not lines:
        return None, {}

    spacing = max(float(sample_spacing_m), float(GEOMETRY_EPSILON_M))
    graph = nx.Graph()
    node_points: Dict[tuple[float, float], Point] = {}
    intersection_points_by_line = _intersection_points_by_road_line(
        lines,
        crs=None if gdf_roads is None else gdf_roads.crs,
    )

    def _add_node(point: Point) -> tuple[float, float]:
        key = _road_node_key(point)
        if key not in node_points:
            node_points[key] = Point(float(point.x), float(point.y))
            graph.add_node(key)
        return key

    for line_idx, line in enumerate(lines):
        if line.length <= float(GEOMETRY_EPSILON_M):
            continue

        distances = list(np.arange(0.0, float(line.length), spacing, dtype=float))
        if not distances or abs(float(distances[-1]) - float(line.length)) > float(GEOMETRY_EPSILON_M):
            distances.append(float(line.length))
        distances.extend(float(line.project(Point(coord))) for coord in line.coords)
        for point in intersection_points_by_line.get(line_idx, []):
            if point is None or point.is_empty:
                continue
            distances.append(float(line.project(point)))
        distances = sorted(
            {
                round(max(0.0, min(float(distance), float(line.length))), 6)
                for distance in distances
            }
        )

        sampled = _dedupe_consecutive_points([line.interpolate(float(distance)) for distance in distances])
        if len(sampled) < 2:
            continue

        prev_key = _add_node(sampled[0])
        prev_point = node_points[prev_key]
        for point in sampled[1:]:
            key = _add_node(point)
            point = node_points[key]
            length = float(prev_point.distance(point))
            if length > float(GEOMETRY_EPSILON_M):
                graph.add_edge(prev_key, key, weight=length)
            prev_key = key
            prev_point = point

    snap_tolerance = max(0.0, float(snap_tolerance_m))
    if snap_tolerance > float(GEOMETRY_EPSILON_M) and len(node_points) > 1:
        try:
            keys = list(node_points.keys())
            nodes_gdf = gpd.GeoDataFrame(
                {"node_key": keys},
                geometry=[node_points[key] for key in keys],
                crs=None if gdf_roads is None else gdf_roads.crs,
            )
            spatial_index = nodes_gdf.sindex
            for idx, geom in enumerate(nodes_gdf.geometry):
                hits = spatial_index.query(geom.buffer(snap_tolerance), predicate="intersects")
                for hit in hits:
                    hit_idx = int(hit)
                    if hit_idx <= idx:
                        continue
                    other = nodes_gdf.geometry.iloc[hit_idx]
                    distance = float(geom.distance(other))
                    if distance <= snap_tolerance and distance > float(GEOMETRY_EPSILON_M):
                        graph.add_edge(keys[idx], keys[hit_idx], weight=distance)
        except Exception:
            # Spatial-index availability varies across GeoPandas/Shapely builds.
            # Exact endpoint matching above is still a safe fallback.
            pass

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None, {}
    return graph, node_points


def build_sampled_road_routing_graph(
    gdf_roads: gpd.GeoDataFrame | None,
    *,
    sample_spacing_m: float = ROAD_EDGE_SAMPLE_SPACING_M,
    snap_tolerance_m: float = 5.0,
) -> tuple[nx.Graph | None, Dict[tuple[float, float], Point]]:
    """
    Public wrapper for the planning-level road graph used by LV and MV routing.
    """
    return _build_sampled_road_graph(
        gdf_roads,
        sample_spacing_m=float(sample_spacing_m),
        snap_tolerance_m=float(snap_tolerance_m),
    )


def _nearest_road_node(
    point: Point,
    node_points: Dict[tuple[float, float], Point],
) -> tuple[tuple[float, float] | None, Point | None, float]:
    if not node_points:
        return None, None, math.inf
    best_key: tuple[float, float] | None = None
    best_point: Point | None = None
    best_distance = math.inf
    for key, candidate in node_points.items():
        distance = float(point.distance(candidate))
        if distance < best_distance:
            best_key = key
            best_point = candidate
            best_distance = distance
    return best_key, best_point, best_distance


def nearest_road_node(
    point: Point,
    node_points: Dict[tuple[float, float], Point],
) -> tuple[tuple[float, float] | None, Point | None, float]:
    """
    Public wrapper for snapping terminals to a sampled road-routing graph.
    """
    return _nearest_road_node(point, node_points)


def _road_route_between_points(
    start: Point,
    end: Point,
    road_graph: nx.Graph | None,
    road_node_points: Dict[tuple[float, float], Point],
    *,
    max_endpoint_snap_m: float,
) -> LineString | None:
    if road_graph is None or not road_node_points:
        return None

    start_key, start_snap, start_snap_m = _nearest_road_node(start, road_node_points)
    end_key, end_snap, end_snap_m = _nearest_road_node(end, road_node_points)
    if start_key is None or end_key is None or start_snap is None or end_snap is None:
        return None
    if start_snap_m > float(max_endpoint_snap_m) or end_snap_m > float(max_endpoint_snap_m):
        return None

    try:
        path_keys = nx.shortest_path(road_graph, start_key, end_key, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    route_points = [start]
    route_points.extend([road_node_points[key] for key in path_keys])
    route_points.append(end)
    route_points = _dedupe_consecutive_points(route_points)
    if len(route_points) < 2:
        return None
    route = LineString(route_points)
    if route.length <= float(GEOMETRY_EPSILON_M):
        return None
    return route


def associate_buildings_to_poles(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_poles: gpd.GeoDataFrame,
    user_distance: float,
    max_associations: int,
) -> pd.DataFrame:
    """
    Associate buildings to poles based on spatial proximity.

    The algorithm loops over poles, builds a buffer of radius `user_distance`,
    finds currently unassociated buildings within that buffer, and associates
    up to `max_associations` buildings.

    Ranking
    -------
    Buildings are ranked by:
    1. higher `priority_score`
    2. shorter distance to the pole
    3. lower building index for deterministic tie-breaking

    With the default `priority_score = 3` for all buildings, this reduces to
    the original distance-first association rule.

    Returns
    -------
    DataFrame with columns ['pole_id', 'building_id'].
    """
    building_association: Dict[int, List[int]] = {idx: [] for idx in gdf_poles.index}
    associated_buildings = set()

    for pole in gdf_poles.itertuples():
        buffer = pole.geometry.buffer(user_distance)

        nearby_buildings = gdf_buildings[
            ~gdf_buildings.index.isin(associated_buildings)
            & gdf_buildings.geometry.within(buffer)
        ].copy()

        if nearby_buildings.empty:
            continue

        nearby_buildings["distance_to_pole"] = nearby_buildings.geometry.distance(
            pole.geometry
        )
        nearby_buildings["priority_score"] = _priority_score_series(nearby_buildings)
        nearby_buildings["_building_idx"] = nearby_buildings.index.astype(int)
        nearby_buildings = nearby_buildings.sort_values(
            ["priority_score", "distance_to_pole", "_building_idx"],
            ascending=[False, True, True],
        )

        num_associated = 0
        for building_id in nearby_buildings.index:
            if num_associated >= max_associations:
                break
            if building_id not in associated_buildings:
                building_association[pole.Index].append(building_id)
                associated_buildings.add(building_id)
                num_associated += 1

        if len(building_association[pole.Index]) > max_associations:
            building_association[pole.Index] = building_association[pole.Index][
                :max_associations
            ]

    records = [
        (pole_id, building_id)
        for pole_id, buildings in building_association.items()
        for building_id in buildings
    ]
    return pd.DataFrame(records, columns=["pole_id", "building_id"])


def place_poles_for_unassociated_buildings(
    gdf_unassociated_buildings: gpd.GeoDataFrame,
    user_distance: float,
    max_associations: int,
    *,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
) -> Tuple[gpd.GeoDataFrame, List[Tuple[Point, Point]], gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Place new poles for unassociated buildings based on proximity.

    If allow_unserved_isolated is True, clusters smaller than min_cluster_size
    are left unserved and returned as 'remaining' buildings, unless the
    residual cluster contains a `must_connect_flag=True` user. A global stop is
    only allowed when no remaining residual cluster contains any protected
    must-connect user.

    Important
    ---------
    Free-placement poles must satisfy the final user-to-pole radius after the
    pole geometry is created. Buildings outside that hard radius remain
    unassigned for later iterations or final standalone treatment.

    Priority behavior
    -----------------
    Cluster selection and final building assignment use `priority_score`.
    When all buildings keep the default `priority_score = 3`, the behavior
    collapses back to the original cluster-size / nearest-distance heuristic.
    """
    new_poles: List[Point] = []
    all_associations: List[Tuple[Point, Point]] = []

    gdf_remaining = gdf_unassociated_buildings.copy()
    unserved_reason_by_idx: dict[int, str] = {}
    must_connect_protection_applied_count = 0
    selective_coverage_stop_allowed = False

    while not gdf_remaining.empty:
        largest_cluster = None
        best_cluster_key: tuple[float, int, float, int] | None = None
        best_cluster_size = 0
        best_cluster_has_must_connect = False
        best_protected_cluster = None
        best_protected_cluster_key: tuple[float, int, float, int] | None = None
        best_protected_cluster_size = 0

        # Find the highest-priority cluster of unassociated buildings.
        for building in gdf_remaining.itertuples():
            buffer = building.geometry.buffer(user_distance)
            intersecting = gdf_remaining[gdf_remaining.geometry.intersects(buffer)]
            if intersecting.empty:
                continue

            cluster_priorities = _priority_score_series(intersecting)
            cluster_priority_score = float(cluster_priorities.sum())
            cluster_size = int(len(intersecting))
            cluster_max_priority = float(cluster_priorities.max())
            cluster_min_idx = int(min(intersecting.index.tolist()))
            cluster_key = (
                cluster_priority_score,
                cluster_size,
                cluster_max_priority,
                -cluster_min_idx,
            )
            cluster_has_must_connect = bool(
                "must_connect_flag" in intersecting.columns
                and intersecting["must_connect_flag"].fillna(False).astype(bool).any()
            )
            if best_cluster_key is None or cluster_key > best_cluster_key:
                best_cluster_key = cluster_key
                best_cluster_size = cluster_size
                best_cluster_has_must_connect = cluster_has_must_connect
                largest_cluster = intersecting
            if cluster_has_must_connect and (
                best_protected_cluster_key is None or cluster_key > best_protected_cluster_key
            ):
                best_protected_cluster_key = cluster_key
                best_protected_cluster_size = cluster_size
                best_protected_cluster = intersecting

        if largest_cluster is None:
            break

        selected_cluster = largest_cluster
        selected_cluster_size = best_cluster_size
        selected_cluster_has_must_connect = best_cluster_has_must_connect

        # `min_cluster_size` remains a count-based selective-coverage rule.
        # A cluster is dropped only when it is smaller than the threshold AND
        # there are no remaining protected residual clusters still requiring
        # service. This prevents a small unprotected cluster from causing a
        # global stop that strands must-connect users elsewhere.
        if allow_unserved_isolated and best_cluster_size < int(min_cluster_size) and not best_cluster_has_must_connect:
            if best_protected_cluster is not None:
                selected_cluster = best_protected_cluster
                selected_cluster_size = best_protected_cluster_size
                selected_cluster_has_must_connect = True
            else:
                selective_coverage_stop_allowed = True
                stop_reason = "below_min_cluster_size" if int(min_cluster_size) > 1 else "standalone_candidate"
                for idx in gdf_remaining.index.tolist():
                    unserved_reason_by_idx.setdefault(int(idx), stop_reason)
                break

        if allow_unserved_isolated and selected_cluster_size < int(min_cluster_size) and selected_cluster_has_must_connect:
            must_connect_protection_applied_count += 1

        # Operate on the selected residual cluster.
        largest_cluster = selected_cluster
        buffers = [b.geometry.buffer(user_distance) for b in largest_cluster.itertuples()]
        merged_buffer = buffers[0]
        for buf in buffers[1:]:
            merged_buffer = merged_buffer.union(buf)

        pole_location: Point = merged_buffer.centroid

        cluster_priority_lookup = _priority_score_series(largest_cluster).to_dict()
        must_connect_idx: set[int] = set()
        if "must_connect_flag" in largest_cluster.columns:
            must_connect_idx = set(
                int(idx)
                for idx, is_required in largest_cluster["must_connect_flag"].fillna(False).astype(bool).items()
                if bool(is_required)
            )

        def _rank_cluster_buildings(final_pole_location: Point) -> list[tuple[int, Point, int]]:
            ranked_buildings = [
                (b.Index, b.geometry, int(cluster_priority_lookup.get(b.Index, int(DEFAULT_PRIORITY_SCORE))))
                for b in largest_cluster.itertuples()
            ]
            ranked_buildings.sort(
                key=lambda x: (-x[2], final_pole_location.distance(x[1]), int(x[0]))
            )
            return ranked_buildings

        # Sort buildings by priority first, then by final service-drop distance.
        buildings_in_cluster = _rank_cluster_buildings(pole_location)

        feasible = [
            (building_idx, building_geom, priority_score)
            for building_idx, building_geom, priority_score in buildings_in_cluster
            if float(pole_location.distance(building_geom)) <= float(user_distance) + float(GEOMETRY_EPSILON_M)
        ]

        if not feasible:
            if selected_cluster_has_must_connect and must_connect_idx:
                # Must-connect users are exempt from min_cluster_size exclusion,
                # so if the centroid is not feasible we anchor the pole directly
                # on a protected user before giving up.
                anchor = next(
                    (
                        (building_idx, building_geom, priority_score)
                        for building_idx, building_geom, priority_score in buildings_in_cluster
                        if int(building_idx) in must_connect_idx
                    ),
                    buildings_in_cluster[0],
                )
                pole_location = Point(anchor[1].x, anchor[1].y)
                buildings_in_cluster = _rank_cluster_buildings(pole_location)
                feasible = [
                    (building_idx, building_geom, priority_score)
                    for building_idx, building_geom, priority_score in buildings_in_cluster
                    if float(pole_location.distance(building_geom)) <= float(user_distance) + float(GEOMETRY_EPSILON_M)
                ]
            elif allow_unserved_isolated:
                # In selective-coverage mode, keep the building unserved rather
                # than breaking the configured radius or looping forever.
                # Drop the lowest-priority candidate first.
                dropped_building_id = int(buildings_in_cluster[-1][0])
                unserved_reason_by_idx.setdefault(dropped_building_id, "not_assigned")
                gdf_remaining = gdf_remaining[~gdf_remaining.index.isin([buildings_in_cluster[-1][0]])]
                continue
            else:
                # In full-coverage mode, fall back to placing a pole directly on a
                # high-priority building point so at least one feasible assignment
                # satisfies the radius.
                pole_location = Point(buildings_in_cluster[0][1].x, buildings_in_cluster[0][1].y)
                buildings_in_cluster = _rank_cluster_buildings(pole_location)
                feasible = [
                    (building_idx, building_geom, priority_score)
                    for building_idx, building_geom, priority_score in buildings_in_cluster
                    if float(pole_location.distance(building_geom)) <= float(user_distance) + float(GEOMETRY_EPSILON_M)
                ]

        if not feasible:
            infeasible_ids = [int(building_idx) for building_idx, _, _ in buildings_in_cluster]
            for building_idx in infeasible_ids:
                if int(building_idx) in must_connect_idx:
                    unserved_reason_by_idx[int(building_idx)] = "physically_infeasible"
            raise ValueError(
                "Free-placement failed to create a feasible pole assignment within the configured user radius. "
                "A protected must-connect cluster could not be served after direct pole fallback."
            )

        new_poles.append(pole_location)

        # Associate only customers that remain feasible under the final pole geometry.
        closest = feasible[:max_associations]
        for building_idx, building_geom, _priority_score in closest:
            all_associations.append((pole_location, building_geom))

        associated_indices = [b[0] for b in closest]
        gdf_remaining = gdf_remaining[~gdf_remaining.index.isin(associated_indices)]

    gdf_new_poles = gpd.GeoDataFrame({"geometry": new_poles}, crs=gdf_unassociated_buildings.crs)
    final_unserved = gdf_remaining.copy()
    if not final_unserved.empty:
        final_unserved["reason_code"] = [
            unserved_reason_by_idx.get(
                int(idx),
                "standalone_candidate" if allow_unserved_isolated else "unserved_or_standalone",
            )
            for idx in final_unserved.index.tolist()
        ]
    return gdf_new_poles, all_associations, final_unserved, {
        "must_connect_protection_applied_count": int(must_connect_protection_applied_count),
        "selective_coverage_stop_allowed": bool(selective_coverage_stop_allowed),
    }



# ---------------------------------------------------------------------
# Graph + MST + exports
# ---------------------------------------------------------------------
def _sample_points_along_edge(line: LineString, spacing_m: float) -> List[Point]:
    if line.is_empty:
        return []

    spacing = max(float(spacing_m), float(GEOMETRY_EPSILON_M))
    if line.length <= spacing:
        return [line.interpolate(0.0), line.interpolate(line.length)]

    distances = np.arange(0.0, float(line.length), spacing, dtype=float)
    points = [line.interpolate(float(distance)) for distance in distances]
    end_point = line.interpolate(line.length)
    if not points or not points[-1].equals(end_point):
        points.append(end_point)
    return points


def compute_edge_weight_attributes(
    start: Point,
    end: Point,
    *,
    edge_weight_mode: str = DEFAULT_EDGE_WEIGHT_MODE,
    roads_geometry: Any | None = None,
    road_attraction_buffer_m: float = ROAD_ATTRACTION_BUFFER_M,
    road_offroad_penalty_factor: float = ROAD_OFFROAD_PENALTY_FACTOR,
    road_edge_sample_spacing_m: float = ROAD_EDGE_SAMPLE_SPACING_M,
) -> Dict[str, Any]:
    """
    Compute MST edge attributes while keeping physical length separate from
    penalized selection weight.

    `length_m` stays Euclidean/projected. `edge_weight` is what the MST uses.
    """
    line = LineString([start, end])
    euclidean_length_m = float(start.distance(end))

    offroad_fraction: float | None = None
    road_distance_mean_m: float | None = None
    road_following_flag: bool | None = None

    if roads_geometry is not None and not getattr(roads_geometry, "is_empty", False):
        sample_points = _sample_points_along_edge(
            line,
            spacing_m=float(road_edge_sample_spacing_m),
        )
        if sample_points:
            distances = np.array(
                [float(roads_geometry.distance(point)) for point in sample_points],
                dtype=float,
            )
            offroad_fraction = float(
                np.mean(distances > float(road_attraction_buffer_m))
            )
            road_distance_mean_m = float(np.mean(distances))
            road_following_flag = bool(offroad_fraction < 0.5)

    use_road_penalty = edge_weight_mode == "road_attracted" and offroad_fraction is not None
    edge_weight = float(euclidean_length_m)
    if use_road_penalty:
        edge_weight = float(
            euclidean_length_m
            * (
                1.0
                + float(road_offroad_penalty_factor) * float(offroad_fraction)
            )
        )

    return {
        "weight": float(edge_weight),
        "edge_weight": float(edge_weight),
        "euclidean_length_m": float(euclidean_length_m),
        "edge_weight_mode": "road_attracted" if use_road_penalty else "euclidean",
        "offroad_fraction": offroad_fraction,
        "road_buffer_m": (
            float(road_attraction_buffer_m)
            if roads_geometry is not None and not getattr(roads_geometry, "is_empty", False)
            else None
        ),
        "road_offroad_penalty_factor": (
            float(road_offroad_penalty_factor)
            if roads_geometry is not None and not getattr(roads_geometry, "is_empty", False)
            else None
        ),
        "road_edge_sample_spacing_m": (
            float(road_edge_sample_spacing_m)
            if roads_geometry is not None and not getattr(roads_geometry, "is_empty", False)
            else None
        ),
        "road_distance_mean_m": road_distance_mean_m,
        "road_following_flag": road_following_flag,
    }


def create_graph_and_mst(
    gdf_poles: gpd.GeoDataFrame,
    *,
    edge_weight_mode: str = DEFAULT_EDGE_WEIGHT_MODE,
    gdf_roads: gpd.GeoDataFrame | None = None,
    road_attraction_buffer_m: float = ROAD_ATTRACTION_BUFFER_M,
    road_offroad_penalty_factor: float = ROAD_OFFROAD_PENALTY_FACTOR,
    road_edge_sample_spacing_m: float = ROAD_EDGE_SAMPLE_SPACING_M,
) -> nx.Graph:
    """
    Create a complete graph over poles and return its Minimum Spanning Tree.

    Modes
    -----
    - `euclidean`: MST selection uses Euclidean pole-to-pole distance.
    - `road_attracted`: MST selection penalizes candidate edges whose sampled
      points lie far from uploaded roads. This is not true road routing.

    IMPORTANT:
    - Graph node IDs are *stable pole_id* (not row index).
    - This prevents downstream identity mismatches (PF, exports, promotion).
    """
    if gdf_poles.empty:
        raise ValueError("create_graph_and_mst: empty pole set.")

    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    # must be projected CRS for Euclidean distance in meters
    if getattr(poles.crs, "is_geographic", False):
        raise ValueError("create_graph_and_mst expects projected CRS in meters, not geographic.")

    pids = poles["pole_id"].to_numpy(dtype=int)
    xs = poles.geometry.x.to_numpy(dtype=float)
    ys = poles.geometry.y.to_numpy(dtype=float)

    G = nx.Graph()
    for pid in pids:
        G.add_node(int(pid))

    roads_geometry = None
    if gdf_roads is not None and not gdf_roads.empty:
        non_empty_roads = gdf_roads[gdf_roads.geometry.notna() & ~gdf_roads.geometry.is_empty]
        if not non_empty_roads.empty:
            roads_geometry = non_empty_roads.geometry.union_all()

    # complete graph (OK for a few hundred)
    n = len(pids)
    for i in range(n):
        for j in range(i + 1, n):
            start = Point(xs[i], ys[i])
            end = Point(xs[j], ys[j])
            edge_attrs = compute_edge_weight_attributes(
                start,
                end,
                edge_weight_mode=edge_weight_mode,
                roads_geometry=roads_geometry,
                road_attraction_buffer_m=float(road_attraction_buffer_m),
                road_offroad_penalty_factor=float(road_offroad_penalty_factor),
                road_edge_sample_spacing_m=float(road_edge_sample_spacing_m),
            )
            G.add_edge(int(pids[i]), int(pids[j]), **edge_attrs)

    return nx.minimum_spanning_tree(G, weight="weight")


def mst_edges_as_latlon(
    gdf_poles_4326: gpd.GeoDataFrame, mst: nx.Graph
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert MST edges to ((lat1, lon1), (lat2, lon2)) pairs for plotting.

    MST nodes are stable pole_id values.
    """
    poles = gdf_poles_4326.copy()
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    geom_by_pid = poles.set_index("pole_id").geometry.to_dict()

    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for u_pid, v_pid in mst.edges():
        p1 = geom_by_pid.get(int(u_pid))
        p2 = geom_by_pid.get(int(v_pid))
        if p1 is None or p2 is None:
            continue
        out.append(((float(p1.y), float(p1.x)), (float(p2.y), float(p2.x))))
    return out


def build_mst_edges_gdf(
    gdf_poles: gpd.GeoDataFrame,
    mst: nx.Graph,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame for MST edges with stable edge attributes.

    Notes
    -----
    - `u` and `v` are canonicalized as the sorted pole_id pair so exports are
      deterministic for a given undirected tree.
    - `line_id` follows the existing `L0`, `L1`, ... convention used elsewhere
      in the app, but is assigned after sorting edge pairs for stability.
    - `length_m` is always the physical Euclidean/projected edge length.
    - `edge_weight` is the MST selection weight and may exceed `length_m` in
      road-attracted mode.
    """
    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    geom_by_pid = poles.set_index("pole_id").geometry.to_dict()

    edge_rows: list[dict[str, Any]] = []
    for u_pid, v_pid, data in mst.edges(data=True):
        u_pid = int(u_pid)
        v_pid = int(v_pid)
        start = geom_by_pid.get(u_pid)
        end = geom_by_pid.get(v_pid)
        if start is None or end is None:
            raise ValueError(
                "MST edge export failed because an edge endpoint is missing from the final pole table. "
                f"Edge=({u_pid}, {v_pid})"
            )

        u_out, v_out = sorted((u_pid, v_pid))
        edge_geometry = data.get("geometry")
        if not isinstance(edge_geometry, LineString) or edge_geometry.is_empty:
            edge_geometry = LineString([geom_by_pid[u_out], geom_by_pid[v_out]])
        else:
            coords = list(edge_geometry.coords)
            first = Point(coords[0])
            last = Point(coords[-1])
            endpoint_mismatch = min(
                float(start.distance(first) + end.distance(last)),
                float(start.distance(last) + end.distance(first)),
            )
            if endpoint_mismatch > 100.0:
                edge_geometry = LineString([geom_by_pid[u_out], geom_by_pid[v_out]])

        euclidean_length_m = float(data.get("euclidean_length_m", edge_geometry.length))
        edge_weight = float(data.get("edge_weight", data.get("weight", euclidean_length_m)))
        edge_rows.append(
            {
                "u": int(u_out),
                "v": int(v_out),
                "u_pole_id": int(u_out),
                "v_pole_id": int(v_out),
                "weight": float(edge_weight),
                "edge_weight": float(edge_weight),
                "edge_weight_mode": str(data.get("edge_weight_mode", "euclidean")),
                "euclidean_length_m": float(euclidean_length_m),
                "length_m": float(euclidean_length_m),
                "length_km": float(euclidean_length_m) / 1000.0,
                "offroad_fraction": data.get("offroad_fraction"),
                "road_buffer_m": data.get("road_buffer_m"),
                "road_offroad_penalty_factor": data.get("road_offroad_penalty_factor"),
                "road_edge_sample_spacing_m": data.get("road_edge_sample_spacing_m"),
                "road_distance_mean_m": data.get("road_distance_mean_m"),
                "road_following_flag": data.get("road_following_flag"),
                "densification_route_mode": data.get("densification_route_mode"),
                "densification_parent_u": data.get("densification_parent_u"),
                "densification_parent_v": data.get("densification_parent_v"),
                "route_length_m": data.get("route_length_m"),
                "chord_length_m": data.get("chord_length_m"),
                "geometry": edge_geometry,
            }
        )

    edge_rows.sort(key=lambda row: (int(row["u"]), int(row["v"])))
    for idx, row in enumerate(edge_rows):
        row["line_id"] = f"L{idx}"

    edge_cols = [
        "line_id",
        "u",
        "v",
        "u_pole_id",
        "v_pole_id",
        "weight",
        "edge_weight",
        "edge_weight_mode",
        "euclidean_length_m",
        "length_m",
        "length_km",
        "offroad_fraction",
        "road_buffer_m",
        "road_offroad_penalty_factor",
        "road_edge_sample_spacing_m",
        "road_distance_mean_m",
        "road_following_flag",
        "densification_route_mode",
        "densification_parent_u",
        "densification_parent_v",
        "route_length_m",
        "chord_length_m",
        "geometry",
    ]
    return gpd.GeoDataFrame(edge_rows, columns=edge_cols, crs=poles.crs)


def save_mst_to_geojson(
    gdf_poles: gpd.GeoDataFrame,
    mst: nx.Graph,
    *,
    gdf_edges: gpd.GeoDataFrame | None = None,
) -> Tuple[io.BytesIO, io.BytesIO]:
    """
    Export MST nodes/edges to GeoJSON. Nodes include:
      - pole_id
      - pole_type
      - pole_origin

    Edges include:
      - line_id
      - u, v / u_pole_id, v_pole_id (stable endpoints)
      - weight / edge_weight
      - length_m, length_km
      - optional road-attraction diagnostics such as `offroad_fraction`
    """
    poles = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)
    if "pole_type" not in poles.columns:
        poles["pole_type"] = "base"
    if "pole_origin" not in poles.columns:
        poles["pole_origin"] = "base"

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    optional_node_cols = [
        "densification_route_mode",
        "densification_parent_u",
        "densification_parent_v",
    ]
    nodes_records = []
    for pole in poles.itertuples():
        record = {
            "pole_id": int(getattr(pole, "pole_id")),
            "pole_type": str(getattr(pole, "pole_type")),
            "pole_origin": str(getattr(pole, "pole_origin")),
            "geometry": pole.geometry,
        }
        for column in optional_node_cols:
            if column in poles.columns:
                record[column] = getattr(pole, column, None)
        nodes_records.append(record)
    gdf_nodes = gpd.GeoDataFrame(nodes_records, crs=poles.crs)

    if gdf_edges is None:
        gdf_edges = build_mst_edges_gdf(poles, mst)

    nodes_buf = io.BytesIO()
    gdf_nodes.to_file(nodes_buf, driver="GeoJSON")
    nodes_buf.seek(0)

    edges_buf = io.BytesIO()
    gdf_edges.to_file(edges_buf, driver="GeoJSON")
    edges_buf.seek(0)

    return nodes_buf, edges_buf

# ---------------------------------------------------------------------
# MST post-processing
# ---------------------------------------------------------------------
def densify_mst_edges(
    gdf_poles: gpd.GeoDataFrame,
    mst: nx.Graph,
    max_pole_span_m: float,
    *,
    gdf_roads: gpd.GeoDataFrame | None = None,
    road_route_sample_spacing_m: float = ROAD_EDGE_SAMPLE_SPACING_M,
    road_route_snap_tolerance_m: float = 5.0,
) -> Tuple[gpd.GeoDataFrame, nx.Graph]:
    """
    Split MST edges longer than `max_pole_span_m` by inserting intermediate poles.

    IMPORTANT:
    - Graph node IDs are stable pole_id (not row index).
    - Inserted poles get new unique pole_id values.
    - When roads are available, long edges are first routed along the sampled
      road graph and densified along that route. Disconnected or unsuitable road
      paths safely fall back to the original straight-line densification.
    """
    if max_pole_span_m is None or max_pole_span_m <= 0:
        out = gdf_poles.copy().reset_index(drop=True)
        if "pole_id" not in out.columns:
            out["pole_id"] = out.index.astype(int)
        if "pole_type" not in out.columns:
            out["pole_type"] = "base"
        if "pole_origin" not in out.columns:
            out["pole_origin"] = "base"
        return out, mst

    base = gdf_poles.copy().reset_index(drop=True)
    if "pole_id" not in base.columns:
        base["pole_id"] = base.index.astype(int)
    if "pole_type" not in base.columns:
        base["pole_type"] = "base"
    if "pole_origin" not in base.columns:
        base["pole_origin"] = "base"

    base["pole_id"] = pd.to_numeric(base["pole_id"], errors="coerce")
    base = base.dropna(subset=["pole_id"]).copy()
    base["pole_id"] = base["pole_id"].astype(int)

    # Map pole_id -> geometry
    geom_by_pid: Dict[int, Point] = base.set_index("pole_id").geometry.to_dict()

    # Ensure MST nodes match pole_id set
    pole_ids_set = set(int(x) for x in base["pole_id"].tolist())
    mst_nodes_set = set(int(n) for n in mst.nodes())
    if mst_nodes_set != pole_ids_set:
        missing = sorted(list(pole_ids_set - mst_nodes_set))[:20]
        extra = sorted(list(mst_nodes_set - pole_ids_set))[:20]
        raise ValueError(
            "Inconsistent MST vs poles (node set mismatch). "
            f"Missing-in-mst (first 20): {missing} | Extra-in-mst (first 20): {extra}"
        )

    G2 = nx.Graph()
    for pid in pole_ids_set:
        G2.add_node(int(pid))

    new_rows: List[Dict[str, Any]] = []
    next_pole_id = int(base["pole_id"].max()) + 1
    has_long_edges = any(
        float(data.get("euclidean_length_m", geom_by_pid[int(u)].distance(geom_by_pid[int(v)]))) > float(max_pole_span_m)
        for u, v, data in mst.edges(data=True)
    )
    if has_long_edges and gdf_roads is not None and not gdf_roads.empty:
        road_graph, road_node_points = _build_sampled_road_graph(
            gdf_roads,
            sample_spacing_m=float(road_route_sample_spacing_m),
            snap_tolerance_m=float(road_route_snap_tolerance_m),
        )
    else:
        road_graph, road_node_points = None, {}

    def _edge_attrs(
        *,
        parent_data: dict[str, Any],
        parent_u: int,
        parent_v: int,
        segment_geometry: LineString,
        start_point: Point,
        end_point: Point,
        route_mode: str,
    ) -> dict[str, Any]:
        route_length_m = float(segment_geometry.length)
        chord_length_m = float(start_point.distance(end_point))
        physical_length_m = route_length_m if route_mode == "road_following" else chord_length_m
        attrs: dict[str, Any] = {
            "weight": float(physical_length_m),
            "edge_weight": float(physical_length_m),
            "euclidean_length_m": float(physical_length_m),
            "route_length_m": float(route_length_m),
            "chord_length_m": float(chord_length_m),
            "edge_weight_mode": str(parent_data.get("edge_weight_mode", "euclidean")),
            "densification_route_mode": str(route_mode),
            "densification_parent_u": int(parent_u),
            "densification_parent_v": int(parent_v),
            "geometry": segment_geometry,
        }
        if route_mode == "road_following":
            attrs.update(
                {
                    "offroad_fraction": 0.0,
                    "road_distance_mean_m": 0.0,
                    "road_following_flag": True,
                    "road_edge_sample_spacing_m": float(road_route_sample_spacing_m),
                }
            )
        return attrs

    for u_pid, v_pid, data in mst.edges(data=True):
        u_pid = int(u_pid)
        v_pid = int(v_pid)
        p_u: Point = geom_by_pid[u_pid]
        p_v: Point = geom_by_pid[v_pid]
        d = float(data.get("euclidean_length_m", p_u.distance(p_v)))

        if d <= max_pole_span_m:
            passthrough = dict(data)
            passthrough["euclidean_length_m"] = float(d)
            passthrough.setdefault("route_length_m", float(d))
            passthrough.setdefault("chord_length_m", float(p_u.distance(p_v)))
            passthrough.setdefault("densification_route_mode", "not_densified")
            passthrough.setdefault("geometry", LineString([p_u, p_v]))
            G2.add_edge(u_pid, v_pid, **passthrough)
            continue

        route_mode = "straight_line_fallback"
        line = LineString([p_u, p_v])
        max_endpoint_snap_m = max(
            float(max_pole_span_m),
            2.0 * float(ROAD_ATTRACTION_BUFFER_M),
            float(road_route_snap_tolerance_m),
        )
        road_line = _road_route_between_points(
            p_u,
            p_v,
            road_graph,
            road_node_points,
            max_endpoint_snap_m=float(max_endpoint_snap_m),
        )
        if road_line is not None:
            line = road_line
            route_mode = "road_following"

        route_length_m = float(line.length)
        n_segments = max(2, int(math.ceil(route_length_m / max_pole_span_m)))

        prev_pid = u_pid
        prev_point = p_u
        prev_distance = 0.0

        # Insert intermediate support poles between the original endpoints.
        for k in range(1, n_segments):
            current_distance = route_length_m * (k / n_segments)
            pt_k: Point = line.interpolate(current_distance)
            segment_geometry = _line_substring(line, prev_distance, current_distance)

            new_pid = int(next_pole_id)
            next_pole_id += 1

            new_rows.append(
                {
                    "geometry": pt_k,
                    "pole_id": new_pid,
                    "pole_type": "support",
                    "pole_origin": "inserted",
                    "densification_route_mode": route_mode,
                    "densification_parent_u": int(u_pid),
                    "densification_parent_v": int(v_pid),
                }
            )

            G2.add_node(new_pid)
            G2.add_edge(
                prev_pid,
                new_pid,
                **_edge_attrs(
                    parent_data=dict(data),
                    parent_u=int(u_pid),
                    parent_v=int(v_pid),
                    segment_geometry=segment_geometry,
                    start_point=prev_point,
                    end_point=pt_k,
                    route_mode=route_mode,
                ),
            )

            prev_pid = new_pid
            prev_point = pt_k
            prev_distance = current_distance

        segment_geometry = _line_substring(line, prev_distance, route_length_m)
        G2.add_edge(
            prev_pid,
            v_pid,
            **_edge_attrs(
                parent_data=dict(data),
                parent_u=int(u_pid),
                parent_v=int(v_pid),
                segment_geometry=segment_geometry,
                start_point=prev_point,
                end_point=p_v,
                route_mode=route_mode,
            ),
        )

    if new_rows:
        gdf_support = gpd.GeoDataFrame(new_rows, geometry="geometry", crs=base.crs)
    else:
        gdf_support = gpd.GeoDataFrame(
            columns=[
                "geometry",
                "pole_id",
                "pole_type",
                "pole_origin",
                "densification_route_mode",
                "densification_parent_u",
                "densification_parent_v",
            ],
            geometry="geometry",
            crs=base.crs,
        )
    densified = pd.concat([base, gdf_support], ignore_index=True)
    densified = gpd.GeoDataFrame(densified, crs=base.crs).reset_index(drop=True)

    return densified, G2

def deduplicate_poles_with_remap(
    gdf_poles: gpd.GeoDataFrame,
    associations_df: pd.DataFrame,
    *,
    tol_m: float = PRE_MST_DEDUPLICATION_TOLERANCE_M,
    prefer_serving: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[int, int]]:
    """
    Deduplicate poles that are within `tol_m` meters.

    Keeps ONE canonical pole per cluster, drops the rest, and returns:
      - cleaned poles GeoDataFrame
      - remapped associations_df (pole_id updated to canonical)
      - mapping old_pole_id -> canonical_pole_id

    Notes:
    - Requires projected CRS (meters). Call BEFORE reprojecting to 4326.
    - If prefer_serving=True, clusters keep a pole that already serves at least one building, if present.
      Otherwise keep the smallest pole_id.
    """
    if gdf_poles.empty:
        return gdf_poles.copy(), associations_df.copy(), {}

    poles = gdf_poles.copy().reset_index(drop=True)

    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)

    # Defensive: must be projected CRS
    if getattr(poles.crs, "is_geographic", False):
        raise ValueError("deduplicate_poles_with_remap expects projected CRS in meters, not geographic.")

    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    # Build serving set (optional preference)
    serving_ids: set[int] = set()
    if prefer_serving and associations_df is not None and not associations_df.empty and "pole_id" in associations_df.columns:
        serving_ids = set(pd.to_numeric(associations_df["pole_id"], errors="coerce").dropna().astype(int).unique())

    # Simple greedy clustering by distance threshold (O(N^2) but OK for ~hundreds poles)
    used = np.zeros(len(poles), dtype=bool)
    mapping: dict[int, int] = {}

    # Precompute coordinates for speed
    xs = poles.geometry.x.to_numpy(dtype=float)
    ys = poles.geometry.y.to_numpy(dtype=float)
    pids = poles["pole_id"].to_numpy(dtype=int)

    # For reproducibility: process poles ordered by pole_id
    order = np.argsort(pids)

    keep_rows: list[int] = []

    for idx in order:
        if used[idx]:
            continue

        # find all poles within tol of this seed
        dx = xs - xs[idx]
        dy = ys - ys[idx]
        d2 = dx * dx + dy * dy
        cluster_idx = np.where((~used) & (d2 <= float(tol_m) ** 2))[0]

        cluster_pids = [int(pids[i]) for i in cluster_idx]

        # choose canonical
        if prefer_serving:
            serving_in_cluster = [pid for pid in cluster_pids if pid in serving_ids]
            if serving_in_cluster:
                canonical = int(min(serving_in_cluster))
            else:
                canonical = int(min(cluster_pids))
        else:
            canonical = int(min(cluster_pids))

        # choose which *row* to keep for the canonical geometry (if canonical exists in cluster)
        if canonical in cluster_pids:
            keep_i = cluster_idx[cluster_pids.index(canonical)]
        else:
            keep_i = int(cluster_idx[0])

        keep_rows.append(int(keep_i))

        # map all cluster members -> canonical
        for pid in cluster_pids:
            mapping[int(pid)] = canonical

        used[cluster_idx] = True

    cleaned = poles.iloc[sorted(set(keep_rows))].copy().reset_index(drop=True)

    # Optionally "snap" canonical geometry to centroid of cluster (more realistic)
    # Here we keep the chosen geometry to avoid moving poles unexpectedly.

    # Remap associations
    assoc_out = associations_df.copy() if associations_df is not None else pd.DataFrame(columns=["pole_id", "building_id"])
    if not assoc_out.empty and "pole_id" in assoc_out.columns:
        assoc_out["pole_id"] = pd.to_numeric(assoc_out["pole_id"], errors="coerce")
        assoc_out = assoc_out.dropna(subset=["pole_id"]).copy()
        assoc_out["pole_id"] = assoc_out["pole_id"].astype(int).map(lambda pid: mapping.get(int(pid), int(pid)))
        assoc_out = assoc_out.dropna(subset=["pole_id"]).copy()
        assoc_out["pole_id"] = assoc_out["pole_id"].astype(int)
        assoc_out = assoc_out.drop_duplicates(subset=["pole_id", "building_id"])

    return cleaned, assoc_out, mapping


def merge_graph_nodes_with_remap(
    graph: nx.Graph,
    mapping: dict[int, int],
) -> nx.Graph:
    """
    Collapse graph nodes according to an old->canonical pole_id mapping.

    - Node IDs remain stable pole_id values.
    - Self-loops introduced by merges are dropped.
    - Parallel edges created by merges are collapsed, keeping the shortest weight.
    """
    if not mapping:
        return graph.copy()

    merged = nx.Graph()

    for node in graph.nodes():
        merged.add_node(int(mapping.get(int(node), int(node))))

    for u, v, data in graph.edges(data=True):
        u2 = int(mapping.get(int(u), int(u)))
        v2 = int(mapping.get(int(v), int(v)))
        if u2 == v2:
            continue

        weight = float(data.get("weight", 0.0))
        edge_attrs = dict(data)
        edge_attrs["weight"] = weight
        if merged.has_edge(u2, v2):
            existing = float(merged[u2][v2].get("weight", weight))
            if weight < existing:
                merged[u2][v2].clear()
                merged[u2][v2].update(edge_attrs)
            else:
                merged[u2][v2]["weight"] = min(existing, weight)
        else:
            merged.add_edge(u2, v2, **edge_attrs)

    return merged
