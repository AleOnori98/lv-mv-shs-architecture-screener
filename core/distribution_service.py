from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List

import geopandas as gpd
import math
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from config.settings import (
    CRITICAL_PRIORITY_THRESHOLD,
    DEFAULT_DEMAND_TIER,
    DEFAULT_EDGE_WEIGHT_MODE,
    DEFAULT_PRIORITY_SCORE,
    GEOMETRY_EPSILON_M,
    HIGH_PRIORITY_THRESHOLD,
    MAX_PRIORITY_SCORE,
    MIN_PRIORITY_SCORE,
    OPTIONAL_USER_ATTRIBUTE_COLUMNS,
    POST_DENSIFICATION_MERGE_TOLERANCE_M,
    PRE_MST_DEDUPLICATION_TOLERANCE_M,
    PROMOTE_SUPPORT_POLES_TO_SERVING,
    ROAD_ATTRACTION_BUFFER_M,
    ROAD_EDGE_SAMPLE_SPACING_M,
    ROAD_OFFROAD_PENALTY_FACTOR,
)
from .distribution_io import load_and_transform_data
from .distribution_algos import (
    collect_sampled_points,
    associate_buildings_to_poles,
    place_poles_for_unassociated_buildings,
    compute_edge_weight_attributes,
    create_graph_and_mst,
    densify_mst_edges,
    mst_edges_as_latlon,
    build_mst_edges_gdf,
    save_mst_to_geojson,
    deduplicate_poles_with_remap,
    merge_graph_nodes_with_remap,
)


REQUIRED_USER_IDENTITY_COLUMNS = ("customer_id", "consumer_type")


def _validate_required_user_identity_columns(gdf_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf_buildings.copy()
    missing = [column for column in REQUIRED_USER_IDENTITY_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(
            "Users file must include the required identity columns "
            f"{list(REQUIRED_USER_IDENTITY_COLUMNS)}. Missing: {missing}. "
            "customer_id is the primary external customer key and consumer_type is used "
            "to aggregate validation loads from category_profiles.csv."
        )

    for column in REQUIRED_USER_IDENTITY_COLUMNS:
        values = out[column].map(lambda value: "" if pd.isna(value) else str(value).strip())
        blank_mask = values == ""
        if blank_mask.any():
            bad_rows = out.index[blank_mask].astype(str).tolist()[:10]
            raise ValueError(
                f"Users file column '{column}' contains blank values. "
                f"Example row indices: {bad_rows}"
            )
        out[column] = values

    duplicate_mask = out["customer_id"].duplicated(keep=False)
    if duplicate_mask.any():
        examples = sorted(out.loc[duplicate_mask, "customer_id"].astype(str).unique().tolist())[:10]
        raise ValueError(
            "Users file column 'customer_id' must be unique. "
            f"Duplicated customer_id examples: {examples}"
        )

    return out


def _round_half_up(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rounded = pd.Series(np.nan, index=values.index, dtype=float)
    mask = numeric.notna()
    rounded.loc[mask] = np.where(
        numeric.loc[mask] >= 0,
        np.floor(numeric.loc[mask] + 0.5),
        np.ceil(numeric.loc[mask] - 0.5),
    )
    return rounded


def _normalize_priority_score_series(values: pd.Series) -> pd.Series:
    rounded = _round_half_up(values)
    return (
        rounded.fillna(int(DEFAULT_PRIORITY_SCORE))
        .clip(lower=int(MIN_PRIORITY_SCORE), upper=int(MAX_PRIORITY_SCORE))
        .astype(int)
    )


def _normalize_bool_like(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except TypeError:
        pass

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return bool(default)

    text = str(value).strip().casefold()
    if text == "":
        return bool(default)

    truthy = {"true", "t", "yes", "y", "1", "si"}
    falsy = {"false", "f", "no", "n", "0"}
    if text in truthy:
        return True
    if text in falsy:
        return False
    return bool(default)


def _normalize_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    return values.apply(lambda value: _normalize_bool_like(value, default=default)).astype(bool)


def _normalize_optional_user_columns(
    gdf_buildings: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    out = gdf_buildings.copy()
    original_columns = set(out.columns)
    priority_source_column = "priority_score" if "priority_score" in original_columns else None

    detected = {
        "has_customer_id": "customer_id" in original_columns,
        "has_consumer_type": "consumer_type" in original_columns,
        "has_priority_score": priority_source_column is not None,
        "has_must_connect_flag": "must_connect_flag" in original_columns,
        "has_must_standalone_flag": "must_standalone_flag" in original_columns,
        "has_demand_tier": "demand_tier" in original_columns,
        "priority_score_source_column": "priority_score" if priority_source_column is not None else "default",
        "unsupported_priority_weight_detected": "priority_weight" in original_columns,
        "unsupported_can_be_standalone_flag_detected": "can_be_standalone_flag" in original_columns,
    }

    if priority_source_column is None:
        out["priority_score"] = int(DEFAULT_PRIORITY_SCORE)
    else:
        out["priority_score"] = _normalize_priority_score_series(out[priority_source_column])

    if "must_connect_flag" in out.columns:
        out["must_connect_flag"] = _normalize_bool_series(out["must_connect_flag"], default=False)
    else:
        out["must_connect_flag"] = False

    if "must_standalone_flag" in out.columns:
        out["must_standalone_flag"] = _normalize_bool_series(out["must_standalone_flag"], default=False)
    else:
        out["must_standalone_flag"] = False

    conflict_mask = out["must_connect_flag"] & out["must_standalone_flag"]
    if conflict_mask.any():
        examples = out.loc[conflict_mask, "customer_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Users file contains contradictory constraints: the same customer cannot have both "
            "must_connect_flag=True and must_standalone_flag=True. "
            f"Example customer_id values: {examples}"
        )

    tiers = pd.to_numeric(out["demand_tier"], errors="coerce") if "demand_tier" in out.columns else None
    if tiers is None:
        out["demand_tier"] = int(DEFAULT_DEMAND_TIER)
    else:
        out["demand_tier"] = tiers.fillna(int(DEFAULT_DEMAND_TIER)).round().clip(lower=0).astype(int)

    if "customer_type" in out.columns:
        out["customer_type"] = out["customer_type"].fillna("").astype(str).str.strip()

    for column in ("peak_kw", "annual_demand_kwh"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    consumer_type_counts = out["consumer_type"].astype(str).value_counts().sort_index()
    detected["consumer_types"] = consumer_type_counts.index.astype(str).tolist()
    detected["consumer_type_counts"] = {
        str(consumer_type): int(count) for consumer_type, count in consumer_type_counts.items()
    }

    return out, detected


def _tier_count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = series.value_counts().sort_index()
    return {str(int(tier)): int(count) for tier, count in counts.items()}


def _compute_priority_tier_metrics(
    gdf_buildings: gpd.GeoDataFrame,
    *,
    served_ids: list[int],
    unserved_ids: set[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    served_index = pd.Index(served_ids)
    unserved_index = pd.Index(sorted(unserved_ids))

    if "priority_score" in gdf_buildings.columns:
        priority_scores = _normalize_priority_score_series(gdf_buildings["priority_score"]).astype(float)
    else:
        priority_scores = pd.Series(float(DEFAULT_PRIORITY_SCORE), index=gdf_buildings.index, dtype=float)

    if "must_connect_flag" in gdf_buildings.columns:
        must_connect = _normalize_bool_series(gdf_buildings["must_connect_flag"], default=False)
    else:
        must_connect = pd.Series(False, index=gdf_buildings.index, dtype=bool)

    if "must_standalone_flag" in gdf_buildings.columns:
        must_standalone = _normalize_bool_series(gdf_buildings["must_standalone_flag"], default=False)
    else:
        must_standalone = pd.Series(False, index=gdf_buildings.index, dtype=bool)

    if "demand_tier" in gdf_buildings.columns:
        tiers = pd.to_numeric(gdf_buildings["demand_tier"], errors="coerce").fillna(int(DEFAULT_DEMAND_TIER)).round().clip(lower=0).astype(int)
    else:
        tiers = pd.Series(int(DEFAULT_DEMAND_TIER), index=gdf_buildings.index, dtype=int)

    priority_score_total = float(priority_scores.sum())
    priority_score_served = float(priority_scores.loc[served_index].sum()) if len(served_index) > 0 else 0.0
    priority_score_unserved = float(priority_scores.loc[unserved_index].sum()) if len(unserved_index) > 0 else 0.0
    priority_coverage_fraction = (
        priority_score_served / priority_score_total if abs(priority_score_total) > 0.0 else 0.0
    )

    high_priority_mask = priority_scores >= float(HIGH_PRIORITY_THRESHOLD)
    critical_priority_mask = priority_scores >= float(CRITICAL_PRIORITY_THRESHOLD)
    high_priority_total = int(high_priority_mask.sum())
    high_priority_served = int(high_priority_mask.loc[served_index].sum()) if len(served_index) > 0 else 0
    high_priority_unserved = int(high_priority_mask.loc[unserved_index].sum()) if len(unserved_index) > 0 else 0
    critical_priority_total = int(critical_priority_mask.sum())
    critical_priority_served = int(critical_priority_mask.loc[served_index].sum()) if len(served_index) > 0 else 0
    critical_priority_unserved = int(critical_priority_mask.loc[unserved_index].sum()) if len(unserved_index) > 0 else 0

    must_connect_total = int(must_connect.sum())
    must_connect_served = int(must_connect.loc[served_index].sum()) if len(served_index) > 0 else 0
    must_connect_unserved = int(must_connect.loc[unserved_index].sum()) if len(unserved_index) > 0 else 0
    must_connect_violations = int(must_connect_unserved)
    must_standalone_total = int(must_standalone.sum())
    must_standalone_served = int(must_standalone.loc[served_index].sum()) if len(served_index) > 0 else 0
    must_standalone_unserved = int(must_standalone.loc[unserved_index].sum()) if len(unserved_index) > 0 else 0

    tier_counts_total = _tier_count_dict(tiers)
    tier_counts_served = _tier_count_dict(tiers.loc[served_index]) if len(served_index) > 0 else {}
    tier_counts_unserved = _tier_count_dict(tiers.loc[unserved_index]) if len(unserved_index) > 0 else {}

    tier3plus_total = int((tiers >= 3).sum())
    tier3plus_served = int((tiers.loc[served_index] >= 3).sum()) if len(served_index) > 0 else 0
    tier3plus_connected_fraction = (tier3plus_served / tier3plus_total) if tier3plus_total > 0 else 0.0

    metrics: dict[str, float] = {
        "priority_score_total": priority_score_total,
        "priority_score_served": priority_score_served,
        "priority_score_unserved": priority_score_unserved,
        "priority_coverage_fraction": priority_coverage_fraction,
        "high_priority_total": float(high_priority_total),
        "high_priority_served": float(high_priority_served),
        "high_priority_unserved": float(high_priority_unserved),
        "critical_priority_total": float(critical_priority_total),
        "critical_priority_served": float(critical_priority_served),
        "critical_priority_unserved": float(critical_priority_unserved),
        "must_connect_total": float(must_connect_total),
        "must_connect_served": float(must_connect_served),
        "must_connect_unserved": float(must_connect_unserved),
        "must_connect_violations": float(must_connect_violations),
        "must_standalone_total": float(must_standalone_total),
        "must_standalone_served": float(must_standalone_served),
        "must_standalone_unserved": float(must_standalone_unserved),
        "tier3plus_total": float(tier3plus_total),
        "tier3plus_served": float(tier3plus_served),
        "tier3plus_connected_fraction": tier3plus_connected_fraction,
    }

    metadata = {
        "high_priority_threshold": float(HIGH_PRIORITY_THRESHOLD),
        "critical_priority_threshold": float(CRITICAL_PRIORITY_THRESHOLD),
        "priority_score_total": priority_score_total,
        "priority_score_served": priority_score_served,
        "priority_score_unserved": priority_score_unserved,
        "priority_coverage_fraction": priority_coverage_fraction,
        "high_priority_total": high_priority_total,
        "high_priority_served": high_priority_served,
        "high_priority_unserved": high_priority_unserved,
        "critical_priority_total": critical_priority_total,
        "critical_priority_served": critical_priority_served,
        "critical_priority_unserved": critical_priority_unserved,
        "must_connect_total": must_connect_total,
        "must_connect_served": must_connect_served,
        "must_connect_unserved": must_connect_unserved,
        "must_connect_violations": must_connect_violations,
        "must_standalone_total": must_standalone_total,
        "must_standalone_served": must_standalone_served,
        "must_standalone_unserved": must_standalone_unserved,
        "tier_counts_total": tier_counts_total,
        "tier_counts_served": tier_counts_served,
        "tier_counts_unserved": tier_counts_unserved,
        "tier3plus_total": tier3plus_total,
        "tier3plus_served": tier3plus_served,
        "tier3plus_connected_fraction": tier3plus_connected_fraction,
    }
    return metrics, metadata


def _split_must_standalone_customers(
    gdf_buildings: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """
    Split customers that are explicitly excluded from grid topology design.

    `must_standalone_flag=True` is a hard user-supplied planning constraint. It
    is intentionally stronger than priority or residual-cluster rules: these
    customers are kept for traceability and exported as unserved/standalone, but
    they do not enter pole placement, MV/LV clustering, or transformer sizing.
    """
    out = gdf_buildings.copy()
    if "must_standalone_flag" in out.columns:
        standalone_mask = _normalize_bool_series(out["must_standalone_flag"], default=False)
    else:
        standalone_mask = pd.Series(False, index=out.index, dtype=bool)
        out["must_standalone_flag"] = False

    grid_eligible = out.loc[~standalone_mask].copy()
    forced_standalone = out.loc[standalone_mask].copy()
    if not forced_standalone.empty:
        forced_standalone["reason_code"] = "must_standalone_flag"

    forced_customer_ids = (
        forced_standalone["customer_id"].astype(str).head(20).tolist()
        if "customer_id" in forced_standalone.columns
        else forced_standalone.index.astype(str).head(20).tolist()
    )
    metadata = {
        "must_standalone_flag_role": "hard_grid_exclusion_by_design",
        "must_standalone_excluded_count": int(len(forced_standalone)),
        "must_standalone_excluded_customer_ids_sample": forced_customer_ids,
    }
    return grid_eligible, forced_standalone, metadata


def _promote_inserted_poles_to_serving(
    *,
    poles_all: gpd.GeoDataFrame,
    associations_df: pd.DataFrame,
    gdf_buildings: gpd.GeoDataFrame,
    served_ids: list[int],
    user_distance: float,
    max_associations: int,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Optional post-densification reassignment step.

    Inserted poles created only to cap long spans may be reclassified as serving
    poles if they reduce customer service-drop distance while still respecting:
    - the hard user-to-pole radius
    - the max users per pole capacity

    This step is intentionally distance-only. We do not apply `priority_score`
    or `must_connect_flag` here, so span-densification cleanup does not add a
    second hidden social-priority decision after the main topology selection.
    """
    current_assign: Dict[int, int] = {}
    if not associations_df.empty:
        for pid, bid in associations_df[["pole_id", "building_id"]].itertuples(index=False):
            current_assign[int(bid)] = int(pid)

    pole_counts: Dict[int, int] = {int(pid): 0 for pid in poles_all["pole_id"].astype(int).tolist()}
    for bid, pid in current_assign.items():
        pole_counts[pid] = pole_counts.get(pid, 0) + 1

    pole_geom_by_id = poles_all.set_index("pole_id").geometry.to_dict()
    served_buildings = gdf_buildings.loc[served_ids]
    bldg_geom = served_buildings.geometry.to_dict()
    inserted = poles_all[poles_all["pole_origin"].astype(str) == "inserted"].copy()

    promoted_pole_ids: set[int] = set()
    for row in inserted.itertuples(index=False):
        pid = int(getattr(row, "pole_id"))
        pgeom = pole_geom_by_id.get(pid)
        if pgeom is None:
            continue

        candidates: List[Tuple[float, int]] = []
        for bid, geom in bldg_geom.items():
            d = float(pgeom.distance(geom))
            if d <= float(user_distance) + float(GEOMETRY_EPSILON_M):
                candidates.append((d, int(bid)))

        if not candidates:
            continue

        candidates.sort(key=lambda x: (x[0], x[1]))
        for d_new, bid in candidates:
            if pole_counts.get(pid, 0) >= int(max_associations):
                break

            cur_pid = current_assign.get(bid)
            if cur_pid is None:
                current_assign[bid] = pid
                pole_counts[pid] = pole_counts.get(pid, 0) + 1
                promoted_pole_ids.add(pid)
                continue

            cur_geom = pole_geom_by_id.get(int(cur_pid))
            if cur_geom is None:
                continue
            d_cur = float(cur_geom.distance(bldg_geom[bid]))
            if d_new < d_cur:
                pole_counts[cur_pid] = max(0, pole_counts.get(cur_pid, 0) - 1)
                current_assign[bid] = pid
                pole_counts[pid] = pole_counts.get(pid, 0) + 1
                promoted_pole_ids.add(pid)

    if len(current_assign) > 0:
        out = pd.DataFrame(
            [(pid, bid) for bid, pid in current_assign.items()],
            columns=["pole_id", "building_id"],
        ).drop_duplicates()
        out["pole_id"] = out["pole_id"].astype(int)
        out["building_id"] = out["building_id"].astype(int)
    else:
        out = pd.DataFrame(columns=["pole_id", "building_id"])

    return out, promoted_pole_ids


def _repair_final_assignments_within_radius(
    *,
    poles_all: gpd.GeoDataFrame,
    associations_df: pd.DataFrame,
    gdf_buildings: gpd.GeoDataFrame,
    served_ids: list[int],
    user_distance: float,
    max_associations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Final safety repair for assignments shifted by post-processing.

    Pole deduplication and road-routed support insertion can move the final pole
    table just enough that an association violates the configured service-drop
    radius. This pass only repairs already-invalid assignments by moving them to
    the nearest feasible final pole with spare capacity. If no feasible pole
    exists, the normal hard consistency check still fails afterwards.
    """
    if associations_df is None or associations_df.empty:
        return pd.DataFrame(columns=["pole_id", "building_id"]), {
            "final_radius_reassignment_count": 0,
            "final_radius_unresolved_count": 0,
            "final_radius_unresolved_building_ids": [],
        }

    poles = poles_all.copy().reset_index(drop=True)
    if "pole_id" not in poles.columns:
        poles["pole_id"] = poles.index.astype(int)
    poles["pole_id"] = pd.to_numeric(poles["pole_id"], errors="coerce")
    poles = poles.dropna(subset=["pole_id"]).copy()
    poles["pole_id"] = poles["pole_id"].astype(int)

    current_assign: Dict[int, int] = {}
    for pid, bid in associations_df[["pole_id", "building_id"]].dropna().itertuples(index=False):
        current_assign[int(bid)] = int(pid)

    pole_geom_by_id = poles.set_index("pole_id").geometry.to_dict()
    pole_counts: Dict[int, int] = {int(pid): 0 for pid in pole_geom_by_id.keys()}
    for pid in current_assign.values():
        pole_counts[int(pid)] = pole_counts.get(int(pid), 0) + 1

    limit = float(user_distance) + float(GEOMETRY_EPSILON_M)
    reassigned = 0
    unresolved: list[int] = []

    for bid in sorted(int(idx) for idx in served_ids):
        if bid not in current_assign or bid not in gdf_buildings.index:
            continue
        building_geom = gdf_buildings.geometry.loc[bid]
        current_pid = int(current_assign[bid])
        current_geom = pole_geom_by_id.get(current_pid)
        current_distance = (
            math.inf if current_geom is None else float(current_geom.distance(building_geom))
        )
        if current_distance <= limit:
            continue

        candidates: list[tuple[float, int]] = []
        for pid, pole_geom in pole_geom_by_id.items():
            pid = int(pid)
            if pole_counts.get(pid, 0) >= int(max_associations):
                continue
            distance = float(pole_geom.distance(building_geom))
            if distance <= limit:
                candidates.append((distance, pid))

        if not candidates:
            unresolved.append(int(bid))
            continue

        candidates.sort(key=lambda item: (item[0], item[1]))
        new_pid = int(candidates[0][1])
        pole_counts[current_pid] = max(0, pole_counts.get(current_pid, 0) - 1)
        pole_counts[new_pid] = pole_counts.get(new_pid, 0) + 1
        current_assign[bid] = new_pid
        reassigned += 1

    out = pd.DataFrame(
        [(pid, bid) for bid, pid in current_assign.items()],
        columns=["pole_id", "building_id"],
    ).drop_duplicates()
    out["pole_id"] = out["pole_id"].astype(int)
    out["building_id"] = out["building_id"].astype(int)
    return out, {
        "final_radius_reassignment_count": int(reassigned),
        "final_radius_unresolved_count": int(len(unresolved)),
        "final_radius_unresolved_building_ids": unresolved[:20],
    }


def _run_low_voltage_core(
    *,
    gdf_buildings: gpd.GeoDataFrame,
    gdf_roads: gpd.GeoDataFrame | None,
    crs_meta: dict[str, Any],
    sampling_distance: float,
    user_distance: float,
    max_associations: int,
    centroid_hint: Tuple[float, float] | None = None,
    edge_weight_mode: str = DEFAULT_EDGE_WEIGHT_MODE,
    allow_unserved_isolated: bool = False,
    min_cluster_size: int = 1,
    max_pole_span_m: float | None = None,
) -> Dict[str, Any]:

    if gdf_buildings is None or gdf_buildings.empty:
        raise ValueError("Users file could not be loaded or is empty.")
    if gdf_buildings.geometry.isna().any():
        raise ValueError("Users file contains missing geometries.")
    if getattr(gdf_buildings.crs, "is_geographic", False):
        raise ValueError("Buildings are still in a geographic CRS; expected projected CRS in meters.")

    gdf_buildings = _validate_required_user_identity_columns(gdf_buildings.copy())
    gdf_buildings, user_field_flags = _normalize_optional_user_columns(gdf_buildings)
    gdf_all_buildings = gdf_buildings.copy()
    gdf_buildings, gdf_forced_standalone, must_standalone_metadata = _split_must_standalone_customers(
        gdf_all_buildings
    )
    if gdf_buildings.empty:
        raise ValueError(
            "All users are marked must_standalone_flag=True, so no grid topology can be generated. "
            "Leave at least one grid-eligible customer or skip the grid-topology step for this case."
        )
    crs_meta = dict(crs_meta or {})
    if gdf_roads is not None and gdf_roads.crs != gdf_buildings.crs:
        gdf_roads = gdf_roads.to_crs(gdf_buildings.crs)
    roads_available_for_weighting = bool(gdf_roads is not None and not gdf_roads.empty)
    road_attracted_requested_but_unavailable = (
        str(edge_weight_mode) == "road_attracted" and not roads_available_for_weighting
    )
    effective_edge_weight_mode = (
        "road_attracted"
        if str(edge_weight_mode) == "road_attracted" and roads_available_for_weighting
        else "euclidean"
    )
    scenario_family = (
        "road_attracted"
        if effective_edge_weight_mode == "road_attracted"
        else "direct_euclidean"
    )
    customer_id_by_building_id = gdf_all_buildings["customer_id"].to_dict()
    bldg_geom_by_id = gdf_all_buildings.geometry.to_dict()
    bldg_wkb_to_id = {geom.wkb: idx for idx, geom in bldg_geom_by_id.items()}

    # ------------------------------------------------------------------
    # 2) Candidate poles from roads
    # ------------------------------------------------------------------
    if gdf_roads is not None and not gdf_roads.empty:
        sampled_points = collect_sampled_points(gdf_roads, sampling_distance)
        gdf_associated_poles = (
            gpd.GeoDataFrame({"geometry": sampled_points}, crs=gdf_buildings.crs)
            if sampled_points
            else gpd.GeoDataFrame(geometry=[], crs=gdf_buildings.crs)
        )
    else:
        gdf_associated_poles = gpd.GeoDataFrame(geometry=[], crs=gdf_buildings.crs)

    # ------------------------------------------------------------------
    # 3) Associate buildings to road poles
    # ------------------------------------------------------------------
    if not gdf_associated_poles.empty:
        gdf_associated_poles = gdf_associated_poles.reset_index(drop=True)

        associations_df = associate_buildings_to_poles(
            gdf_buildings=gdf_buildings,
            gdf_poles=gdf_associated_poles,
            user_distance=user_distance,
            max_associations=max_associations,
        )

        if not associations_df.empty:
            associations_df = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()

            kept_old_ids = sorted(associations_df["pole_id"].astype(int).unique().tolist())
            gdf_associated_poles = gdf_associated_poles.iloc[kept_old_ids].reset_index(drop=True)

            remap = {old: new for new, old in enumerate(kept_old_ids)}
            associations_df["pole_id"] = associations_df["pole_id"].map(remap).astype(int)
            associations_df["building_id"] = associations_df["building_id"].astype(int)
        else:
            gdf_associated_poles = gdf_associated_poles.iloc[0:0].copy()
            associations_df = pd.DataFrame(columns=["pole_id", "building_id"])
    else:
        associations_df = pd.DataFrame(columns=["pole_id", "building_id"])

    # ------------------------------------------------------------------
    # 4) New poles for remaining buildings
    # ------------------------------------------------------------------
    gdf_unassociated = gdf_buildings[
        ~gdf_buildings.index.isin(associations_df.get("building_id", []))
    ]

    gdf_new_poles, new_associations, gdf_remaining, placement_metadata = place_poles_for_unassociated_buildings(
        gdf_unassociated_buildings=gdf_unassociated,
        user_distance=user_distance,
        max_associations=max_associations,
        allow_unserved_isolated=allow_unserved_isolated,
        min_cluster_size=min_cluster_size,
    )
    gdf_new_poles = gdf_new_poles.reset_index(drop=True)

    gdf_final_poles = pd.concat([gdf_associated_poles, gdf_new_poles], ignore_index=True).reset_index(drop=True)
    gdf_final_poles = gpd.GeoDataFrame(gdf_final_poles, crs=gdf_buildings.crs)

    if gdf_final_poles.empty:
        raise ValueError("No poles could be placed; check input data and parameters.")

    # stable ids + origin (create, then deduplicate in projected CRS)
    gdf_final_poles["pole_id"] = gdf_final_poles.index.astype(int)
    gdf_final_poles["pole_origin"] = "base"
    gdf_final_poles["pole_type"] = "base"  # temporary; serving/non-serving set after associations

    # ------------------------------------------------------------------
    # 4a) Deduplicate near-coincident poles (prevents degenerate MST edges)
    # ------------------------------------------------------------------
    # Recommended tolerance: 0.5–1.0 m depending on input cleanliness
    DEDUP_TOL_M = PRE_MST_DEDUPLICATION_TOLERANCE_M

    gdf_final_poles, associations_df, pole_id_remap = deduplicate_poles_with_remap(
        gdf_final_poles,
        associations_df,
        tol_m=DEDUP_TOL_M,
        prefer_serving=True,
    )

    # After dropping poles, ensure pole_id are still unique ints
    gdf_final_poles["pole_id"] = pd.to_numeric(gdf_final_poles["pole_id"], errors="coerce")
    gdf_final_poles = gdf_final_poles.dropna(subset=["pole_id"]).copy()
    gdf_final_poles["pole_id"] = gdf_final_poles["pole_id"].astype(int)

    if gdf_final_poles["pole_id"].duplicated().any():
        raise ValueError("Deduplication produced duplicate pole_id values; this should not happen.")


    # ------------------------------------------------------------------
    # 4b) Append new associations
    # ------------------------------------------------------------------
    if len(new_associations) > 0:
        new_df = pd.DataFrame(new_associations, columns=["pole", "building_geom"])

        new_pole_wkb_to_local = {geom.wkb: i for i, geom in enumerate(gdf_new_poles.geometry)}
        offset = len(gdf_associated_poles)

        def _safe_lookup_new_pole_local_id(p: Point) -> int:
            k = p.wkb
            if k in new_pole_wkb_to_local:
                return int(new_pole_wkb_to_local[k])
            if len(gdf_new_poles) == 0:
                raise ValueError("Internal error: new_associations provided but gdf_new_poles is empty.")
            dists = gdf_new_poles.geometry.distance(p)
            return int(dists.idxmin())

        def _safe_lookup_building_id(g: Point) -> int:
            k = g.wkb
            if k in bldg_wkb_to_id:
                return int(bldg_wkb_to_id[k])
            dists = gdf_buildings.geometry.distance(g)
            return int(dists.idxmin())

        new_df["pole_id"] = new_df["pole"].apply(lambda p: offset + _safe_lookup_new_pole_local_id(p))
        new_df["building_id"] = new_df["building_geom"].apply(_safe_lookup_building_id)

        new_assoc_ids = new_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        new_assoc_ids["pole_id"] = new_assoc_ids["pole_id"].astype(int)
        new_assoc_ids["building_id"] = new_assoc_ids["building_id"].astype(int)

        associations_df = pd.concat([associations_df[["pole_id", "building_id"]], new_assoc_ids], ignore_index=True)

    # cleanup associations
    if not associations_df.empty:
        associations_df = associations_df[["pole_id", "building_id"]].dropna().drop_duplicates()
        associations_df["pole_id"] = associations_df["pole_id"].astype(int)
        associations_df["building_id"] = associations_df["building_id"].astype(int)

        valid_base_pole_ids = set(pd.to_numeric(gdf_final_poles["pole_id"], errors="coerce").dropna().astype(int).tolist())
        missing_base_poles = sorted(
            pid for pid in associations_df["pole_id"].astype(int).tolist() if int(pid) not in valid_base_pole_ids
        )
        if missing_base_poles:
            raise ValueError(
                "Pole ID mismatch in base associations vs base pole set. "
                f"Example missing pole_ids (up to 20): {missing_base_poles[:20]}"
            )

    # base serving ids and types
    base_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()
    gdf_final_poles["pole_type"] = gdf_final_poles["pole_id"].apply(
        lambda pid: "serving" if int(pid) in base_serving_ids else "non_serving"
    )

    # ------------------------------------------------------------------
    # 5) Served vs unserved buildings
    # ------------------------------------------------------------------
    if allow_unserved_isolated:
        gdf_algorithm_unserved = gdf_remaining.copy()
        if "reason_code" not in gdf_algorithm_unserved.columns:
            gdf_algorithm_unserved["reason_code"] = (
                "below_min_cluster_size" if int(min_cluster_size) > 1 else "standalone_candidate"
            )
    else:
        gdf_algorithm_unserved = gdf_buildings.iloc[0:0].copy()
        gdf_algorithm_unserved["reason_code"] = pd.Series(dtype=object)

    unserved_parts = [part for part in (gdf_algorithm_unserved, gdf_forced_standalone) if not part.empty]
    if unserved_parts:
        gdf_unserved = gpd.GeoDataFrame(
            pd.concat(unserved_parts, ignore_index=False),
            geometry="geometry",
            crs=gdf_all_buildings.crs,
        )
    else:
        gdf_unserved = gdf_all_buildings.iloc[0:0].copy()
        gdf_unserved["reason_code"] = pd.Series(dtype=object)

    if not gdf_unserved.empty and "must_connect_flag" in gdf_unserved.columns:
        must_connect_unserved_mask = gdf_unserved["must_connect_flag"].fillna(False).astype(bool)
        invalid_selective_reasons = {"below_min_cluster_size", "standalone_candidate"}
        if "reason_code" in gdf_unserved.columns:
            gdf_unserved.loc[
                must_connect_unserved_mask
                & gdf_unserved["reason_code"].isin(invalid_selective_reasons),
                "reason_code",
            ] = "must_connect_unserved_inconsistency"

    unserved_ids = set(gdf_unserved.index)
    served_ids = [idx for idx in gdf_all_buildings.index if idx not in unserved_ids]
    gdf_served = gdf_all_buildings.loc[served_ids] if served_ids else gdf_all_buildings.iloc[0:0]

    num_buildings = int(len(gdf_all_buildings))
    num_served = int(len(gdf_served))
    num_unserved = int(len(gdf_unserved))

    # Ensure associations only contain served buildings (defensive)
    if not associations_df.empty:
        associations_df = associations_df[associations_df["building_id"].isin(served_ids)].copy()

    # ------------------------------------------------------------------
    # 6) MST + densification
    # ------------------------------------------------------------------
    mst_base: nx.Graph = create_graph_and_mst(
        gdf_final_poles,
        edge_weight_mode=effective_edge_weight_mode,
        gdf_roads=gdf_roads,
        road_attraction_buffer_m=float(ROAD_ATTRACTION_BUFFER_M),
        road_offroad_penalty_factor=float(ROAD_OFFROAD_PENALTY_FACTOR),
        road_edge_sample_spacing_m=float(ROAD_EDGE_SAMPLE_SPACING_M),
    )

    gdf_poles_densified, mst = densify_mst_edges(
        gdf_poles=gdf_final_poles,
        mst=mst_base,
        max_pole_span_m=float(max_pole_span_m or 0.0),
        gdf_roads=gdf_roads,
        road_route_sample_spacing_m=float(ROAD_EDGE_SAMPLE_SPACING_M),
    )

    # ------------------------------------------------------------------
    # 6b) Support-pole promotion / customer reassignment step.
    # Inserted poles from span densification may become serving poles if the
    # optional promotion rule is enabled and reassignment improves service-drop
    # distance without breaking capacity or radius constraints.
    # ------------------------------------------------------------------
    poles_all = gdf_poles_densified.copy().reset_index(drop=True)
    if "pole_id" not in poles_all.columns:
        poles_all["pole_id"] = poles_all.index.astype(int)
    if "pole_type" not in poles_all.columns:
        poles_all["pole_type"] = "base"
    if "pole_origin" not in poles_all.columns:
        poles_all["pole_origin"] = "base"
    promoted_pole_ids: set[int] = set()
    if PROMOTE_SUPPORT_POLES_TO_SERVING:
        associations_df, promoted_pole_ids = _promote_inserted_poles_to_serving(
            poles_all=gpd.GeoDataFrame(poles_all, crs=gdf_buildings.crs),
            associations_df=associations_df,
            gdf_buildings=gdf_all_buildings,
            served_ids=served_ids,
            user_distance=user_distance,
            max_associations=max_associations,
        )

    # Update pole_type after promotion
    final_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()

    def _final_type(row) -> str:
        pid = int(row["pole_id"])
        origin = str(row.get("pole_origin", "base"))
        if pid in final_serving_ids:
            return "serving"
        if origin == "inserted":
            return "support"
        return "non_serving"

    poles_all["pole_type"] = poles_all.apply(_final_type, axis=1)

    gdf_poles_densified = gpd.GeoDataFrame(poles_all, crs=gdf_buildings.crs)

    # ------------------------------------------------------------------
    # 6c) Post-densification cleanup for near-coincident final poles.
    # Densification can insert support poles very close to existing poles,
    # especially where long spans terminate near intersections. If left as-is,
    # this creates ultra-short electrical segments that the PF step correctly
    # filters out as numerical near-shorts, which can orphan a load-bearing pole.
    #
    # We merge only very-close final poles here, then collapse the MST node IDs
    # with the same remap so topology and associations stay consistent.
    # ------------------------------------------------------------------
    FINAL_MERGE_TOL_M = POST_DENSIFICATION_MERGE_TOLERANCE_M
    gdf_poles_densified, associations_df, final_pole_id_remap = deduplicate_poles_with_remap(
        gdf_poles_densified,
        associations_df,
        tol_m=FINAL_MERGE_TOL_M,
        prefer_serving=True,
    )
    mst = merge_graph_nodes_with_remap(mst, final_pole_id_remap)

    # Refresh final edge attributes after node merges so exported physical
    # lengths, road-adherence diagnostics, and MST selection weights all
    # reflect the final pole geometry rather than pre-merge spans.
    final_geom_by_pid = gdf_poles_densified.set_index("pole_id").geometry.to_dict()
    roads_geometry = None
    if roads_available_for_weighting:
        non_empty_roads = gdf_roads[gdf_roads.geometry.notna() & ~gdf_roads.geometry.is_empty]
        if not non_empty_roads.empty:
            roads_geometry = non_empty_roads.geometry.union_all()
    for u_pid, v_pid in mst.edges():
        p_u = final_geom_by_pid.get(int(u_pid))
        p_v = final_geom_by_pid.get(int(v_pid))
        if p_u is None or p_v is None:
            continue
        existing_edge_attrs = dict(mst[int(u_pid)][int(v_pid)])
        refreshed_edge_attrs = compute_edge_weight_attributes(
            p_u,
            p_v,
            edge_weight_mode=effective_edge_weight_mode,
            roads_geometry=roads_geometry,
            road_attraction_buffer_m=float(ROAD_ATTRACTION_BUFFER_M),
            road_offroad_penalty_factor=float(ROAD_OFFROAD_PENALTY_FACTOR),
            road_edge_sample_spacing_m=float(ROAD_EDGE_SAMPLE_SPACING_M),
        )
        if str(existing_edge_attrs.get("densification_route_mode", "")) == "road_following":
            for attr in (
                "geometry",
                "euclidean_length_m",
                "route_length_m",
                "chord_length_m",
                "densification_route_mode",
                "densification_parent_u",
                "densification_parent_v",
                "offroad_fraction",
                "road_distance_mean_m",
                "road_following_flag",
                "road_edge_sample_spacing_m",
            ):
                if attr in existing_edge_attrs:
                    refreshed_edge_attrs[attr] = existing_edge_attrs[attr]
            refreshed_edge_attrs["weight"] = float(
                existing_edge_attrs.get("weight", refreshed_edge_attrs["weight"])
            )
            refreshed_edge_attrs["edge_weight"] = float(
                existing_edge_attrs.get("edge_weight", refreshed_edge_attrs["edge_weight"])
            )
        else:
            for attr in (
                "geometry",
                "route_length_m",
                "chord_length_m",
                "densification_route_mode",
                "densification_parent_u",
                "densification_parent_v",
            ):
                if attr in existing_edge_attrs:
                    refreshed_edge_attrs[attr] = existing_edge_attrs[attr]
        mst[int(u_pid)][int(v_pid)].update(refreshed_edge_attrs)

    associations_df, final_radius_repair_metadata = _repair_final_assignments_within_radius(
        poles_all=gdf_poles_densified,
        associations_df=associations_df,
        gdf_buildings=gdf_all_buildings,
        served_ids=served_ids,
        user_distance=user_distance,
        max_associations=max_associations,
    )

    final_serving_ids = set(associations_df["pole_id"].unique()) if not associations_df.empty else set()
    gdf_poles_densified["pole_type"] = gdf_poles_densified.apply(_final_type, axis=1)

    # Stable MST edges as pole_id pairs (for PF / validation).
    #
    # IMPORTANT:
    # `mst` already uses stable pole_id values as graph node IDs.
    # Do not remap through GeoDataFrame row indices here: after deduplication the
    # row index is no longer guaranteed to match pole_id, and that can corrupt the
    # exported PF topology (load-bearing poles appear disconnected even though the
    # internal MST is connected).
    mst_edges_pole_ids = [(int(u), int(v)) for u, v in mst.edges()]

    # ------------------------------------------------------------------
    # 6d) Consistency check: every served building must reference a graph pole,
    # and every serving pole must belong to the electrical tree.
    # ------------------------------------------------------------------
    if not associations_df.empty:
        graph_pole_ids = {int(n) for n in mst.nodes()}
        assoc_pole_ids = set(pd.to_numeric(associations_df["pole_id"], errors="coerce").dropna().astype(int).tolist())

        missing_poles = sorted(pid for pid in assoc_pole_ids if pid not in graph_pole_ids)
        if missing_poles:
            raise ValueError(
                "Topology consistency error: some buildings reference poles that are not present in the MST graph. "
                f"Example pole_ids (up to 20): {missing_poles[:20]}"
            )

        allow_single_node_serving = int(len(graph_pole_ids)) == 1
        orphan_serving_poles = sorted(
            pid
            for pid in assoc_pole_ids
            if int(mst.degree(pid)) < 1 and not allow_single_node_serving
        )
        if orphan_serving_poles:
            raise ValueError(
                "Topology consistency error: some poles have assigned buildings but no electrical connection in the MST. "
                f"Example pole_ids (up to 20): {orphan_serving_poles[:20]}"
            )

        pole_geom_by_id_post = gdf_poles_densified.set_index("pole_id").geometry.to_dict()
        missing_geom_poles = sorted(pid for pid in assoc_pole_ids if int(pid) not in pole_geom_by_id_post)
        if missing_geom_poles:
            raise ValueError(
                "Topology consistency error: some buildings reference poles missing from the final pole table. "
                f"Example pole_ids (up to 20): {missing_geom_poles[:20]}"
            )

    short_final_edges = sorted(
        float(data.get("euclidean_length_m", data.get("weight", 0.0)))
        for _, _, data in mst.edges(data=True)
        if float(data.get("euclidean_length_m", data.get("weight", 0.0))) < FINAL_MERGE_TOL_M
    )
    if short_final_edges:
        raise ValueError(
            "Topology consistency error: final MST still contains ultra-short pole-to-pole segments after cleanup. "
            f"Shortest remaining segment = {short_final_edges[0]:.2f} m; expected >= {FINAL_MERGE_TOL_M:.2f} m."
        )

    # ------------------------------------------------------------------
    # 7) Service drops length (recomputed using densified poles)
    # ------------------------------------------------------------------
    service_drop_length_m = 0.0
    service_drop_by_building: dict[int, float] = {}
    max_service_drop_m = 0.0
    if not associations_df.empty:
        pole_geom_by_id = gdf_poles_densified.set_index("pole_id").geometry.to_dict()
        for pole_id, building_id in associations_df[["pole_id", "building_id"]].itertuples(index=False):
            pole_geom = pole_geom_by_id.get(int(pole_id))
            bldg = bldg_geom_by_id.get(int(building_id))
            if pole_geom is None or bldg is None:
                continue
            drop_m = float(pole_geom.distance(bldg))
            service_drop_length_m += drop_m
            service_drop_by_building[int(building_id)] = drop_m
            max_service_drop_m = max(max_service_drop_m, drop_m)

    if max_service_drop_m > float(user_distance) + float(GEOMETRY_EPSILON_M):
        raise ValueError(
            "Topology consistency error: final customer assignment exceeds the configured max user-pole radius. "
            f"Max service-drop = {max_service_drop_m:.3f} m, configured limit = {float(user_distance):.3f} m."
        )

    service_drop_length_km = service_drop_length_m / 1000.0

    # ------------------------------------------------------------------
    # 8) Backbone length + totals
    # ------------------------------------------------------------------
    backbone_length_m = float(
        sum(
            float(data.get("euclidean_length_m", data.get("weight", 0.0)))
            for _, _, data in mst.edges(data=True)
        )
    )
    total_mst_edge_weight = float(
        sum(
            float(data.get("edge_weight", data.get("weight", 0.0)))
            for _, _, data in mst.edges(data=True)
        )
    )
    offroad_weighted_length_m = None
    road_following_ratio = None
    offroad_backbone_km = None
    if roads_available_for_weighting and backbone_length_m > 0.0:
        offroad_weighted_length_m = float(
            sum(
                float(data.get("euclidean_length_m", data.get("weight", 0.0)))
                * float(data.get("offroad_fraction", 0.0) or 0.0)
                for _, _, data in mst.edges(data=True)
            )
        )
        road_following_ratio = float(
            1.0 - (offroad_weighted_length_m / backbone_length_m)
        )
        offroad_backbone_km = float(offroad_weighted_length_m / 1000.0)
    backbone_length_km = backbone_length_m / 1000.0
    total_network_length_km = backbone_length_km + service_drop_length_km

    total_poles = int(len(gdf_poles_densified))
    serving_poles = int(associations_df["pole_id"].nunique()) if not associations_df.empty else 0
    support_poles = int((gdf_poles_densified["pole_type"] == "support").sum())
    densification_route_modes = [
        str(data.get("densification_route_mode", "unknown"))
        for _, _, data in mst.edges(data=True)
        if str(data.get("densification_route_mode", "not_densified")) != "not_densified"
    ]
    densification_route_counts = {
        str(mode): int(count)
        for mode, count in pd.Series(densification_route_modes, dtype=object).value_counts().items()
    } if densification_route_modes else {}
    road_routed_densified_length_km = float(
        sum(
            float(data.get("euclidean_length_m", data.get("weight", 0.0)))
            for _, _, data in mst.edges(data=True)
            if str(data.get("densification_route_mode", "")) == "road_following"
        )
        / 1000.0
    )
    straight_fallback_densified_length_km = float(
        sum(
            float(data.get("euclidean_length_m", data.get("weight", 0.0)))
            for _, _, data in mst.edges(data=True)
            if str(data.get("densification_route_mode", "")) == "straight_line_fallback"
        )
        / 1000.0
    )
    priority_metrics, priority_metadata = _compute_priority_tier_metrics(
        gdf_all_buildings,
        served_ids=served_ids,
        unserved_ids=unserved_ids,
    )
    if int(priority_metadata.get("must_standalone_served", 0)) > 0:
        raise ValueError(
            "Topology consistency error: must_standalone_flag=True customers were assigned to the grid."
        )

    must_connect_unserved = int(priority_metadata.get("must_connect_unserved", 0))
    if must_connect_unserved > 0:
        must_connect_reason_counts: dict[str, int] = {}
        if not gdf_unserved.empty and "must_connect_flag" in gdf_unserved.columns:
            must_connect_rows = gdf_unserved[
                gdf_unserved["must_connect_flag"].fillna(False).astype(bool)
            ].copy()
            if "reason_code" in must_connect_rows.columns:
                must_connect_reason_counts = {
                    str(reason): int(count)
                    for reason, count in must_connect_rows["reason_code"]
                    .fillna("unknown")
                    .astype(str)
                    .value_counts()
                    .items()
                }

        selective_only_reasons = {
            "below_min_cluster_size",
            "standalone_candidate",
            "must_connect_unserved_inconsistency",
        }
        if must_connect_reason_counts and set(must_connect_reason_counts).issubset(selective_only_reasons):
            raise ValueError(
                "Topology consistency error: must-connect users remain unserved even though must-connect "
                "clusters are exempt from min_cluster_size exclusion. "
                f"Reason counts: {must_connect_reason_counts}"
            )

        raise ValueError(
            "Topology consistency error: must-connect users remain unserved after topology generation. "
            f"Reason counts: {must_connect_reason_counts or {'unknown': must_connect_unserved}}"
        )

    if not associations_df.empty:
        associations_df = associations_df.copy()
        associations_df["building_id_internal"] = associations_df["building_id"].astype(int)
        associations_df["customer_id"] = (
            associations_df["building_id"]
            .map(customer_id_by_building_id)
            .fillna("")
            .astype(str)
        )
        associations_df["service_drop_m"] = (
            associations_df["building_id"]
            .map(service_drop_by_building)
            .astype(float)
        )
        building_attribute_lookup = gdf_all_buildings.drop(columns=["geometry"], errors="ignore")
        for column in OPTIONAL_USER_ATTRIBUTE_COLUMNS:
            if column in building_attribute_lookup.columns:
                associations_df[column] = associations_df["building_id"].map(building_attribute_lookup[column])
        associations_df = associations_df[
            [
                "pole_id",
                "building_id",
                "building_id_internal",
                "customer_id",
                "service_drop_m",
                *[column for column in OPTIONAL_USER_ATTRIBUTE_COLUMNS if column in associations_df.columns],
            ]
        ].copy()
    else:
        associations_df = pd.DataFrame(
            columns=[
                "pole_id",
                "building_id",
                "building_id_internal",
                "customer_id",
                "service_drop_m",
                *OPTIONAL_USER_ATTRIBUTE_COLUMNS,
            ]
        )

    if not associations_df.empty:
        served_join_cols = ["pole_id", "service_drop_m"]
        gdf_served = gdf_served.join(
            associations_df.set_index("building_id")[served_join_cols],
            how="left",
        )

    # ------------------------------------------------------------------
    # 9) Reproject outputs to EPSG:4326 for plotting
    # ------------------------------------------------------------------
    gdf_buildings_4326 = gdf_all_buildings.to_crs(epsg=4326)
    gdf_poles_4326 = gdf_poles_densified.to_crs(epsg=4326)
    gdf_edges_projected = build_mst_edges_gdf(gdf_poles_densified, mst)
    gdf_edges_4326 = gdf_edges_projected.to_crs(epsg=4326)
    gdf_roads_4326 = gdf_roads.to_crs(epsg=4326) if gdf_roads is not None else None
    gdf_served_4326 = gdf_served.to_crs(epsg=4326) if num_served > 0 else gdf_buildings_4326.iloc[0:0]
    gdf_unserved_4326 = gdf_unserved.to_crs(epsg=4326) if num_unserved > 0 else gdf_buildings_4326.iloc[0:0]

    # ------------------------------------------------------------------
    # 10) Map center
    # ------------------------------------------------------------------
    if centroid_hint and any(centroid_hint):
        center = centroid_hint
    else:
        if not gdf_poles_4326.empty:
            c = gdf_poles_4326.union_all().centroid
            center = (c.y, c.x)
        else:
            c = gdf_buildings_4326.union_all().centroid
            center = (c.y, c.x)

    # ------------------------------------------------------------------
    # 11) Edge list for plotting + downloads
    # ------------------------------------------------------------------
    mst_edges_latlon = mst_edges_as_latlon(gdf_poles_4326, mst)
    nodes_geojson, edges_geojson = save_mst_to_geojson(gdf_poles_4326, mst, gdf_edges=gdf_edges_4326)
    associations_csv = associations_df.sort_values(["pole_id", "building_id"]).to_csv(index=False).encode("utf-8")
    projected_crs = gdf_buildings.crs
    metadata = {
        "projected_crs": None if projected_crs is None else projected_crs.to_string(),
        "projected_crs_epsg": None if projected_crs is None else projected_crs.to_epsg(),
        "input_crs": crs_meta.get("input_crs"),
        "crs_selection_method": crs_meta.get("crs_selection_method"),
        "crs_selection_note": crs_meta.get("crs_selection_note"),
        "edge_weight_mode": effective_edge_weight_mode,
        "scenario_family": scenario_family,
        "edge_weight_formula": (
            "euclidean_length_m"
            if effective_edge_weight_mode == "euclidean"
            else "euclidean_length_m * (1 + road_offroad_penalty_factor * offroad_fraction)"
        ),
        "roads_available_for_weighting": roads_available_for_weighting,
        "road_attracted_requested_but_unavailable": road_attracted_requested_but_unavailable,
        "road_attraction_buffer_m": float(ROAD_ATTRACTION_BUFFER_M),
        "road_offroad_penalty_factor": float(ROAD_OFFROAD_PENALTY_FACTOR),
        "road_edge_sample_spacing_m": float(ROAD_EDGE_SAMPLE_SPACING_M),
        "road_following_ratio": road_following_ratio,
        "offroad_backbone_km": offroad_backbone_km,
        "total_mst_edge_weight": total_mst_edge_weight,
        "total_input_customers": int(len(gdf_all_buildings)),
        "total_grid_eligible_customers": int(len(gdf_buildings)),
        "note": (
            "Road-attracted mode penalizes candidate MST edges far from roads but does not perform true road-network routing."
        ),
        "must_connect_note": (
            "Residual clusters containing must_connect_flag=True users are exempt from exclusion due only to min_cluster_size."
        ),
        **user_field_flags,
        **must_standalone_metadata,
        "priority_score_default": int(DEFAULT_PRIORITY_SCORE),
        "priority_score_min": int(MIN_PRIORITY_SCORE),
        "priority_score_max": int(MAX_PRIORITY_SCORE),
        "priority_score_rounding": "nearest_integer",
        "priority_score_out_of_range_policy": "clip_to_1_5",
        "distance_units": "m",
        "length_units": "m/km",
        "pre_mst_deduplication_tolerance_m": float(PRE_MST_DEDUPLICATION_TOLERANCE_M),
        "post_densification_merge_tolerance_m": float(POST_DENSIFICATION_MERGE_TOLERANCE_M),
        "geometry_epsilon_m": float(GEOMETRY_EPSILON_M),
        "support_pole_promotion_enabled": bool(PROMOTE_SUPPORT_POLES_TO_SERVING),
        "support_pole_reassignment_priority_aware": False,
        "support_pole_promoted_count": int(len(promoted_pole_ids)),
        "densification_road_routing_enabled": bool(gdf_roads is not None and not gdf_roads.empty),
        "densification_route_counts": densification_route_counts,
        "road_routed_densified_length_km": float(road_routed_densified_length_km),
        "straight_fallback_densified_length_km": float(straight_fallback_densified_length_km),
        **final_radius_repair_metadata,
        "demand_tier_topology_role": "descriptive_only",
        "must_connect_flag_role": "protects_against_standalone_exclusion_due_to_min_cluster_size",
        "must_standalone_flag_role": "hard_grid_exclusion_by_design",
        "min_cluster_size_role": "count_based",
        "must_connect_protection_applied_count": int(
            placement_metadata.get("must_connect_protection_applied_count", 0)
        ),
        "selective_coverage_stop_allowed": bool(
            placement_metadata.get("selective_coverage_stop_allowed", False)
        ),
        "max_service_drop_m": float(max_service_drop_m),
        **priority_metadata,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "topology_params": {
            "sampling_distance_m": float(sampling_distance),
            "user_distance_m": float(user_distance),
            "max_associations": int(max_associations),
            "edge_weight_mode": effective_edge_weight_mode,
            "allow_unserved_isolated": bool(allow_unserved_isolated),
            "min_cluster_size": int(min_cluster_size),
            "max_pole_span_m": None if max_pole_span_m is None else float(max_pole_span_m),
        },
    }

    return {
        "metrics": {
            "total_network_length_km": total_network_length_km,
            "backbone_length_km": backbone_length_km,
            "service_drop_length_km": service_drop_length_km,
            "total_mst_edge_weight": total_mst_edge_weight,
            "road_following_ratio": road_following_ratio,
            "offroad_backbone_km": offroad_backbone_km,
            "road_routed_densified_length_km": float(road_routed_densified_length_km),
            "straight_fallback_densified_length_km": float(straight_fallback_densified_length_km),
            "num_poles_total": total_poles,
            "num_poles_serving": serving_poles,
            "num_poles_support": support_poles,
            "num_buildings": num_buildings,
            "num_grid_eligible": int(len(gdf_buildings)),
            "num_forced_standalone": int(len(gdf_forced_standalone)),
            "num_served": num_served,
            "num_unserved": num_unserved,
            **priority_metrics,
            "tier_counts_total": priority_metadata["tier_counts_total"],
            "tier_counts_served": priority_metadata["tier_counts_served"],
            "tier_counts_unserved": priority_metadata["tier_counts_unserved"],
        },
        "gdf_buildings_4326": gdf_buildings_4326,
        "gdf_poles_4326": gdf_poles_4326,
        "gdf_edges_4326": gdf_edges_4326,
        "gdf_roads_4326": gdf_roads_4326,
        "gdf_served_4326": gdf_served_4326,
        "gdf_unserved_4326": gdf_unserved_4326,
        "mst_edges_latlon": mst_edges_latlon,
        "mst_edges_pole_ids": mst_edges_pole_ids,
        "associations_df": associations_df,
        "metadata": metadata,
        "downloads": {
            "nodes_geojson": nodes_geojson,
            "edges_geojson": edges_geojson,
            "associations_csv": associations_csv,
        },
        "center": center,
    }
