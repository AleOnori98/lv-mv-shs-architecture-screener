"""Input schemas, accepted values, and default-filling for the simplified MVP.

Each schema declares required columns and the defaults applied when an optional
column is missing. Keeping this in one place makes the CSV contract explicit and
keeps :mod:`core.io.load_inputs` small.
"""
from __future__ import annotations

import difflib
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd


# Accepted customer_type values (Section 4.1).
CUSTOMER_TYPES = (
    "household",
    "shop",
    "productive",
    "school",
    "clinic",
    "hospital",
    "mosque",
    "public_service",
    "water_pump",
    "other",
)

# Customer types that must NOT default to standalone-eligible (Section 4.1 / 5).
GRID_PRIORITY_TYPES = ("clinic", "hospital", "school", "productive", "water_pump")


def _as_bool(series: pd.Series, default: bool) -> pd.Series:
    truthy = {"true", "t", "yes", "y", "1", "si"}
    falsy = {"false", "f", "no", "n", "0"}

    def conv(value: Any) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().casefold()
        if text == "":
            return default
        if text in truthy:
            return True
        if text in falsy:
            return False
        return default

    return series.apply(conv).astype(bool)


def normalize_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and fill MVP defaults for the customers table."""
    out = df.copy()
    required = {"customer_id", "latitude", "longitude", "customer_type"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"customers.csv missing required columns: {sorted(missing)}")

    out["customer_id"] = out["customer_id"].astype(str).str.strip()
    if out["customer_id"].duplicated().any():
        dups = out.loc[out["customer_id"].duplicated(), "customer_id"].unique().tolist()[:10]
        raise ValueError(f"customers.csv has duplicate customer_id values: {dups}")

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    if out[["latitude", "longitude"]].isna().any().any():
        raise ValueError("customers.csv has missing/invalid latitude or longitude values.")

    out["customer_type"] = out["customer_type"].fillna("other").astype(str).str.strip().str.lower()

    # Numeric loads / parameters with defaults.
    out["peak_kw_now"] = pd.to_numeric(out.get("peak_kw_now"), errors="coerce").fillna(0.3)
    out["annual_kwh_now"] = pd.to_numeric(out.get("annual_kwh_now"), errors="coerce")
    out["annual_kwh_now"] = out["annual_kwh_now"].fillna(out["peak_kw_now"] * 8760.0 * 0.2)
    out["power_factor"] = pd.to_numeric(out.get("power_factor"), errors="coerce").fillna(0.90).clip(0.5, 1.0)
    out["growth_rate"] = pd.to_numeric(out.get("growth_rate"), errors="coerce").fillna(0.03)
    out["demand_tier"] = (
        pd.to_numeric(out.get("demand_tier"), errors="coerce").fillna(2).round().clip(lower=0).astype(int)
    )
    out["priority_weight"] = pd.to_numeric(out.get("priority_weight"), errors="coerce").fillna(1.0)

    out["must_connect_flag"] = _as_bool(out.get("must_connect_flag", pd.Series(index=out.index, dtype=object)), False)
    out = out.drop(columns=["future_connection_candidate"], errors="ignore")

    # can_be_standalone defaults: True for normal households, False for grid-priority types.
    if "can_be_standalone_flag" in df.columns:
        out["can_be_standalone_flag"] = _as_bool(out["can_be_standalone_flag"], True)
    else:
        out["can_be_standalone_flag"] = ~out["customer_type"].isin(GRID_PRIORITY_TYPES)
    # must_connect customers can never be standalone.
    out.loc[out["must_connect_flag"], "can_be_standalone_flag"] = False

    return out.reset_index(drop=True)


def _require(df: pd.DataFrame, name: str, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def normalize_source(df: pd.DataFrame) -> Dict[str, Any]:
    _require(df, "source.csv", {"source_id", "latitude", "longitude"})
    row = df.iloc[0]
    return {
        "source_id": str(row["source_id"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "source_type": str(row.get("source_type", "injection")),
        "lv_voltage_kv": float(row.get("lv_voltage_kv", 0.4) or 0.4),
        "mv_voltage_kv": float(row.get("mv_voltage_kv", 11.0) or 11.0),
    }


def normalize_conductors(df: pd.DataFrame, level: str) -> pd.DataFrame:
    _require(df, f"{level} conductors", {"conductor_id", "r_ohm_per_km", "x_ohm_per_km", "max_i_ka", "cost_per_m", "rank"})
    out = df.copy()
    for col in ("r_ohm_per_km", "x_ohm_per_km", "max_i_ka", "cost_per_m", "rank"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["phase_count"] = pd.to_numeric(out.get("phase_count", 3), errors="coerce").fillna(3).astype(int)
    if "voltage_level" not in out.columns:
        out["voltage_level"] = level
    out = out.sort_values("rank").reset_index(drop=True)
    return out


def normalize_transformers(df: pd.DataFrame) -> pd.DataFrame:
    _require(
        df,
        "transformers.csv",
        {"transformer_id", "sn_kva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "rank"},
    )
    out = df.copy()
    for col in ("sn_kva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "rank"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["pfe_kw"] = pd.to_numeric(out.get("pfe_kw", 0.0), errors="coerce").fillna(0.0)
    out["i0_percent"] = pd.to_numeric(out.get("i0_percent", 0.0), errors="coerce").fillna(0.0)
    out["cost_fixed"] = pd.to_numeric(out.get("cost_fixed", 0.0), errors="coerce").fillna(0.0)
    out["cost_per_kva"] = pd.to_numeric(out.get("cost_per_kva", 0.0), errors="coerce").fillna(0.0)
    out["max_loading_percent"] = pd.to_numeric(out.get("max_loading_percent", 100.0), errors="coerce").fillna(100.0)
    return out.sort_values("rank").reset_index(drop=True)


def normalize_standalone_packages(df: pd.DataFrame) -> pd.DataFrame:
    _require(
        df,
        "standalone_packages.csv",
        {"package_id", "service_tier", "max_peak_kw", "annual_kwh", "capex", "lifetime_years"},
    )
    out = df.copy()
    out["service_tier"] = pd.to_numeric(out["service_tier"], errors="coerce").fillna(1).astype(int)
    for col in ("max_peak_kw", "annual_kwh", "capex", "lifetime_years"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["annual_om_fraction"] = pd.to_numeric(out.get("annual_om_fraction", 0.03), errors="coerce").fillna(0.03)
    out["replacement_fraction"] = pd.to_numeric(out.get("replacement_fraction", 0.8), errors="coerce").fillna(0.8)
    return out.reset_index(drop=True)


# cost_parameters defaults (Section 4.8) — used to fill any missing keys.
COST_PARAMETER_DEFAULTS: Dict[str, float] = {
    "discount_rate": 0.08,
    "planning_horizon_years": 15,
    "lv_connection_cost_per_customer": 100.0,
    "metering_protection_cost_per_customer": 50.0,
    "pole_cost": 120.0,
    "civil_logistics_multiplier": 1.2,
    "loss_value_per_kwh": 0.20,
    "loss_load_factor": 0.30,  # engineering default: snapshot->annual loss energy scaling
    # Levelised cost of the mini-grid generation (EUR/kWh). Applied to grid-served annual energy
    # so the grid architectures are compared fairly against stand-alone systems, whose own
    # generation cost is already embedded in the SHS package NPC.
    "generation_lcoe_per_kwh": 0.25,
    # Diversity/coincidence factor for the DESIGN power flow: the fraction of the sum of individual
    # customer peaks that coincides at the network peak. The additional "coincident peak" stress
    # test always uses 1.0 (every customer at their individual peak simultaneously).
    "coincidence_factor": 0.6,
    "voltage_min_pu": 0.90,
    "voltage_max_pu": 1.05,
    "max_line_loading_percent": 100.0,
    "max_transformer_loading_percent": 100.0,
    "dmax_lv_service_zone_m": 800.0,
    "lmax_lv_path_m": 1000.0,
    "remote_branch_length_threshold_m": 300.0,
    "branch_cost_per_customer_threshold": 600.0,
    "loss_threshold_percent": 8.0,
    # MV/LV service-zone partition (Case C): max LV radius and customers per transformer zone.
    "max_lv_zone_radius_m": 250.0,
    "max_zone_customers": 60,
    # MV transformer-count criterion (Case C). "voltage_cap": iterate the number of zones and
    # let the power flow pick the minimum that holds the LV voltage drop (1 - voltage_min_pu);
    # "distance_cap": complete-linkage clusters guaranteed <= max_cluster_diameter_m; "capacity":
    # legacy greedy farthest-first packing. All three respect the transformer kVA cap.
    "clustering_criterion": "voltage_cap",
    "max_cluster_diameter_m": 500.0,
    "max_transformers": 12,
    "seed_diameter_m": 2000.0,
    # Planning reserve the voltage_cap k-search must keep ON TOP of voltage_min_pu when it
    # accepts a transformer count. Minimising k against the bare limit spends every volt of
    # headroom, leaving the chosen design sitting on the constraint, so any error in the
    # diversity assumption or load estimate flips it infeasible. 0.0 reproduces the old
    # behaviour. It does NOT change what counts as a feasible candidate.
    "design_voltage_margin_pu": 0.02,
    # Which snapshot the k-search selects against: "design" (diversified, + the margin
    # above) or "coincident" (every customer at peak at once -- the conservative extreme).
    "k_selection_snapshot": "design",
    # Diversity basis for transformer sizing and the zone kVA cap. 0 => track
    # coincidence_factor (the design power flow's basis); any value in (0, 1] is used
    # directly. Default 1.0 = size on the coincident zone peak.
    #
    # This is a DELIBERATE, declared conservatism, not an oversight -- it used to be a
    # hard-coded 1.0 with no stated rationale, which is what made it look like a bug:
    # sizing at 1.0 while validating at 0.6 makes the rating 1/0.6 larger than the
    # snapshot it is checked against (2.0x with the 1.2 margin), so transformer loading
    # peaks near 50% and max_transformer_loading_percent can never bind.
    #
    # Setting it to 0 (track the design factor) is nevertheless NOT the right default:
    # coincidence_factor here is a single flat SYSTEM-WIDE number, and diversity only
    # emerges with many aggregated customers. Applied to a 10-customer zone it understates
    # that zone's real peak, so it would under-size exactly the small zones. Measured on
    # the Mafa case: tracking the design factor shrinks the units, raises their per-unit
    # impedance and costs ~0.010 pu of LV voltage, which pushed Case D from feasible
    # (8 zones, 0.916 pu) to infeasible (10 zones, 0.8998 pu) -- so the oversizing was
    # partly buying voltage support, not purely wasted money.
    #
    # The principled fix is a size-dependent ADMD / diversity curve (cf as a function of
    # the number of aggregated customers) rather than either flat number; until then this
    # stays explicit, single-sourced and reported, with the loading headroom visible so the
    # slack is auditable instead of hidden.
    "transformer_sizing_coincidence_factor": 1.0,
    # Sending voltage at the plant LV slack; an explicit step-up transformer produces the real
    # drop onto the MV backbone (replaces the earlier fixed 1.04 pu slack on the MV bus).
    "source_vm_pu": 1.0,
}

# Non-numeric parameters passed through verbatim by ``normalize_cost_parameters``.
_STRING_PARAMETERS = frozenset({"clustering_criterion", "k_selection_snapshot"})
_ALLOWED_CLUSTERING_CRITERIA = frozenset({"voltage_cap", "distance_cap", "capacity"})
_ALLOWED_K_SNAPSHOTS = frozenset({"design", "coincident"})


# Parameters that were removed from the engine but may still appear in older input files.
# They are accepted silently (so existing datasets keep loading) but are not warned about,
# because the user cannot act on them beyond deleting the row.
RETIRED_PARAMETERS = frozenset({
    "dmax_lv_service_zone_m",
    "lmax_lv_path_m",
    "remote_branch_length_threshold_m",
    "loss_threshold_percent",
})


def normalize_cost_parameters(params: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    """Fill defaults and coerce types for the cost/engineering parameter table.

    Unrecognised keys used to be dropped in silence, so a typo in ``cost_parameters.csv``
    (``discount_rte``) was accepted and the default used instead -- the run looked fine and
    answered a different question. Unknown keys now raise a warning naming the key and the
    closest known parameter; pass ``strict=True`` to turn that into a hard error.
    """
    out = dict(COST_PARAMETER_DEFAULTS)
    unknown: list[str] = []
    uncoercible: list[str] = []
    for key, value in (params or {}).items():
        if key not in out:
            if key not in RETIRED_PARAMETERS:
                unknown.append(str(key))
            continue
        if value is None or str(value).strip() == "":
            continue
        if key in _STRING_PARAMETERS:
            out[key] = str(value).strip()
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            uncoercible.append(str(key))

    if uncoercible:
        msg = (
            "cost_parameters: non-numeric value(s) ignored for "
            + ", ".join(sorted(uncoercible))
            + " (the default was used instead)"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    if unknown:
        hints = []
        for key in sorted(unknown):
            near = difflib.get_close_matches(key, out.keys(), n=1, cutoff=0.7)
            hints.append(f"{key!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        msg = (
            "cost_parameters: unrecognised key(s) ignored: "
            + ", ".join(hints)
            + ". Known parameters: "
            + ", ".join(sorted(out.keys()))
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    crit = str(out.get("clustering_criterion", "voltage_cap"))
    if crit not in _ALLOWED_CLUSTERING_CRITERIA:
        raise ValueError(
            f"clustering_criterion must be one of {sorted(_ALLOWED_CLUSTERING_CRITERIA)}; got {crit!r}"
        )
    snap = str(out.get("k_selection_snapshot", "design"))
    if snap not in _ALLOWED_K_SNAPSHOTS:
        raise ValueError(
            f"k_selection_snapshot must be one of {sorted(_ALLOWED_K_SNAPSHOTS)}; got {snap!r}"
        )
    return out
