"""CAPEX and present-value-of-losses costing for a candidate architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .npc import annuity_factor


@dataclass
class CostBreakdown:
    lv_line_cost: float = 0.0
    mv_line_cost: float = 0.0
    pole_cost: float = 0.0
    transformer_cost: float = 0.0
    connection_metering_cost: float = 0.0
    standalone_npc: float = 0.0
    loss_pv_cost: float = 0.0
    generation_npc: float = 0.0

    @property
    def grid_capex(self) -> float:
        return self.lv_line_cost + self.mv_line_cost + self.pole_cost + self.transformer_cost + self.connection_metering_cost

    @property
    def total_npc(self) -> float:
        return self.grid_capex + self.standalone_npc + self.loss_pv_cost + self.generation_npc


def line_cost(conductor_by_line: Dict[str, Dict[str, Any]], length_by_line: Dict[str, float], civil_multiplier: float) -> float:
    total = 0.0
    for lid, cond in conductor_by_line.items():
        length_m = float(length_by_line.get(lid, 0.0))
        total += length_m * float(cond["cost_per_m"])
    return total * float(civil_multiplier)


def pole_capex(n_poles: int, pole_cost: float, civil_multiplier: float) -> float:
    return float(n_poles) * float(pole_cost) * float(civil_multiplier)


def connection_capex(n_grid_customers: int, cost_params: Dict[str, float]) -> float:
    per = float(cost_params["lv_connection_cost_per_customer"]) + float(cost_params["metering_protection_cost_per_customer"])
    return float(n_grid_customers) * per


def transformer_capex(transformers: list[Dict[str, Any]]) -> float:
    total = 0.0
    for tx in transformers:
        total += float(tx["cost_fixed"]) + float(tx["cost_per_kva"]) * float(tx["sn_kva"])
    return total


def generation_present_value(annual_kwh_total: float, cost_params: Dict[str, float]) -> float:
    """PV of the mini-grid generation cost for grid-served energy over the planning horizon.

    ``annual_kwh_total`` is the grid-served annual energy (sum of ``annual_kwh_now`` over grid
    customers). It is levelised at ``generation_lcoe_per_kwh`` and discounted with the horizon
    annuity factor. Stand-alone customers are excluded here — their generation cost is already
    inside the SHS package NPC — so grid and stand-alone are compared on the same footing.
    """
    lcoe = float(cost_params.get("generation_lcoe_per_kwh", 0.0) or 0.0)
    kwh = float(annual_kwh_total or 0.0)
    if lcoe <= 0.0 or kwh <= 0.0:
        return 0.0
    return lcoe * kwh * annuity_factor(float(cost_params["discount_rate"]), int(cost_params["planning_horizon_years"]))


def loss_present_value(losses_kw: float, cost_params: Dict[str, float]) -> float:
    """PV of annual loss energy cost over the horizon (snapshot losses scaled to a year)."""
    if losses_kw is None or losses_kw != losses_kw:  # NaN guard
        return 0.0
    annual_kwh = float(losses_kw) * 8760.0 * float(cost_params["loss_load_factor"])
    annual_cost = annual_kwh * float(cost_params["loss_value_per_kwh"])
    return annual_cost * annuity_factor(float(cost_params["discount_rate"]), int(cost_params["planning_horizon_years"]))
