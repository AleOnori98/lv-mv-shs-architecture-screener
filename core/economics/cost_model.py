"""CAPEX and present-value-of-losses costing for a candidate architecture."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from .npc import annuity_factor, growth_annuity_factor


@dataclass
class CostBreakdown:
    lv_line_cost: float = 0.0
    mv_line_cost: float = 0.0
    pole_cost: float = 0.0
    transformer_cost: float = 0.0
    connection_metering_cost: float = 0.0
    service_drop_cost: float = 0.0
    standalone_npc: float = 0.0
    loss_pv_cost: float = 0.0
    generation_npc: float = 0.0

    _COMPONENTS = (
        "lv_line_cost", "mv_line_cost", "pole_cost", "transformer_cost",
        "connection_metering_cost", "service_drop_cost",
        "standalone_npc", "loss_pv_cost", "generation_npc",
    )
    _CAPEX_COMPONENTS = (
        "lv_line_cost", "mv_line_cost", "pole_cost", "transformer_cost",
        "connection_metering_cost", "service_drop_cost",
    )

    @property
    def missing_components(self) -> List[str]:
        """Components whose value is unknown (NaN), e.g. losses after a diverged power flow.

        A diverged candidate has no computable loss cost. Reporting that as 0.0 would make
        the broken candidate look *cheaper* than a working one, so the NaN is preserved and
        surfaced here instead of being silently absorbed into the total.
        """
        return [n for n in self._COMPONENTS if math.isnan(float(getattr(self, n)))]

    @property
    def npc_complete(self) -> bool:
        """True when every cost component is known, so ``total_npc`` is comparable."""
        return not self.missing_components

    def _sum(self, names) -> float:
        return float(sum(v for v in (float(getattr(self, n)) for n in names) if not math.isnan(v)))

    @property
    def grid_capex(self) -> float:
        return self._sum(self._CAPEX_COMPONENTS)

    @property
    def total_npc(self) -> float:
        """Sum of the KNOWN components. Check ``npc_complete`` before comparing two totals:
        an incomplete total is a lower bound, never a like-for-like cost."""
        return self._sum(self._COMPONENTS)


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


def service_drop_capex(total_drop_m: float, cheapest_lv_cost_per_m: float, civil_multiplier: float) -> float:
    """CAPEX of the service drops from each pole to its customers.

    Drops are priced with the cheapest LV conductor as a proxy -- they are not separate
    pandapower lines, so they carry no impedance in the power flow, only cost.

    This component exists because the SHS screen already CREDITED avoided service drops in
    ``_avoided_grid_cost`` while no candidate's NPC ever charged them: the gate was paid for
    a saving the objective did not recognise, which biased every screen toward going
    off-grid. Credit and charge now use the same proxy on the same metres.
    """
    return float(total_drop_m) * float(cheapest_lv_cost_per_m) * float(civil_multiplier)


def cheapest_lv_cost_per_m(lv_conductors) -> float:
    """``cost_per_m`` of the lowest-rank (cheapest) LV conductor in the catalogue."""
    return float(lv_conductors.sort_values("rank").iloc[0]["cost_per_m"])


def transformer_capex(transformers: list[Dict[str, Any]]) -> float:
    total = 0.0
    for tx in transformers:
        total += float(tx["cost_fixed"]) + float(tx["cost_per_kva"]) * float(tx["sn_kva"])
    return total


def generation_present_value(
    annual_kwh_total: float,
    cost_params: Dict[str, float],
    *,
    growth_rate: float = 0.0,
) -> float:
    """PV of the mini-grid generation cost for grid-served energy over the planning horizon.

    ``annual_kwh_total`` is the grid-served annual energy TODAY (sum of ``annual_kwh_now``
    over grid customers); ``growth_rate`` is the energy-weighted annual growth of that same
    set. The stream is levelised at ``generation_lcoe_per_kwh`` over its actual growth
    trajectory (:func:`~core.economics.npc.growth_annuity_factor`).

    Stand-alone customers are excluded -- their generation cost is already inside the SHS
    package NPC -- so grid and stand-alone are compared on the same footing.

    Previously this held present-year energy flat for the whole horizon while the loss cost
    was taken from the HORIZON power flow, also held flat: two different years of the same
    growing demand added into one NPC. Both now use the same growth-aware discounting.
    """
    lcoe = float(cost_params.get("generation_lcoe_per_kwh", 0.0) or 0.0)
    kwh = float(annual_kwh_total or 0.0)
    if lcoe <= 0.0 or kwh <= 0.0:
        return 0.0
    return lcoe * kwh * growth_annuity_factor(
        float(cost_params["discount_rate"]), float(growth_rate),
        int(cost_params["planning_horizon_years"]),
    )


def loss_energy_price(cost_params: Dict[str, float]) -> float:
    """Price of one kWh of loss energy.

    Loss energy is *generated* energy that never reaches a customer, so by default it is
    priced at ``generation_lcoe_per_kwh`` -- the same price as every other generated kWh.
    Pricing it separately would put two different prices on one commodity (the old
    behaviour: losses at 0.20 while generation was 0.25).

    ``loss_value_per_kwh`` remains available as a deliberate override for the case where
    loss energy really is worth something else (e.g. a tariff-based opportunity cost);
    set it to 0 / leave it unset to track the generation LCOE.
    """
    override = float(cost_params.get("loss_value_per_kwh", 0.0) or 0.0)
    if override > 0.0:
        return override
    return float(cost_params.get("generation_lcoe_per_kwh", 0.0) or 0.0)


def loss_present_value(
    losses_kw: float,
    cost_params: Dict[str, float],
    *,
    growth_rate: float = 0.0,
) -> float:
    """PV of annual loss energy cost over the horizon (snapshot losses scaled to a year).

    ``losses_kw`` is the horizon loss snapshot and ``growth_rate`` the energy-weighted
    demand growth, so the year-T snapshot can be back-cast and levelised over the actual
    trajectory rather than charged flat for every year.

    Returns **NaN** when the loss snapshot is unknown (a diverged power flow). It is NOT
    zero: charging nothing for the losses of a network that could not be solved makes a
    diverged candidate cheaper than a working one. ``CostBreakdown`` propagates the NaN
    through ``missing_components`` / ``npc_complete``.
    """
    if losses_kw is None or math.isnan(float(losses_kw)):
        return float("nan")
    r = float(cost_params["discount_rate"])
    T = int(cost_params["planning_horizon_years"])
    price = loss_energy_price(cost_params)

    # Losses scale with current squared, so loss energy grows at roughly (1+g)^2 - 1.
    g_load = float(growth_rate)
    g_loss = (1.0 + g_load) ** 2 - 1.0

    # ``losses_kw`` is the HORIZON snapshot (kept so a diverged horizon power flow still
    # propagates as NaN). Back-cast it to year 0 and levelise over the growth trajectory,
    # instead of charging the year-T loss for all T years.
    losses_now_kw = float(losses_kw) / ((1.0 + g_loss) ** T) if T > 0 else float(losses_kw)
    annual_kwh_now = losses_now_kw * 8760.0 * float(cost_params["loss_load_factor"])
    return annual_kwh_now * price * growth_annuity_factor(r, g_loss, T)
