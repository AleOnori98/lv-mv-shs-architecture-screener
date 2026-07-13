"""Discounting helpers and stand-alone net-present-cost calculation."""
from __future__ import annotations

from typing import Any, Dict


def annuity_factor(rate: float, years: int) -> float:
    """PV of 1 unit paid at the end of each year for ``years`` years."""
    r = float(rate)
    n = int(years)
    if r <= 0:
        return float(n)
    return (1.0 - (1.0 + r) ** (-n)) / r


def pv(amount: float, rate: float, t: int) -> float:
    return float(amount) / ((1.0 + float(rate)) ** int(t))


def standalone_npc(package: Dict[str, Any], horizon_years: int, discount_rate: float) -> float:
    """NPC = capex + PV(O&M) + PV(replacements) - PV(salvage)."""
    capex = float(package["capex"])
    om_annual = float(package.get("annual_om_fraction", 0.03)) * capex
    repl_cost = float(package.get("replacement_fraction", 0.6)) * capex
    life = max(int(package.get("lifetime_years", 8)), 1)
    T = int(horizon_years)
    r = float(discount_rate)

    npc = capex
    npc += om_annual * annuity_factor(r, T)
    # Replacements at each full lifetime within the horizon (not at year T itself).
    t = life
    last_install = 0
    while t < T:
        npc += pv(repl_cost, r, t)
        last_install = t
        t += life
    # Salvage credit for the residual life of the last installed asset at year T.
    residual = (last_install + life - T) / life if (last_install + life) > T else 0.0
    salvage = residual * repl_cost
    npc -= pv(salvage, r, T)
    return float(npc)
