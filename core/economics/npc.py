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


def growth_annuity_factor(rate: float, growth: float, years: int) -> float:
    """PV of a stream that is 1 unit in year 1 and grows at ``growth`` per year.

    ``sum_{t=1..T} (1+g)^(t-1) / (1+r)^t``.

    This is what levelises a GROWING quantity, and it is the piece that was missing from
    the energy costs. Generation was priced off ``annual_kwh_now`` held flat for the whole
    horizon while the loss cost was priced off the HORIZON loss snapshot, also held flat --
    two different years of the same growing demand summed into one NPC. A flat annuity
    under-counts a growing stream; anchoring everything at the horizon over-counts it by
    assuming year-T demand from day one. This factor prices the actual trajectory.

    Reduces to :func:`annuity_factor` when ``growth`` is 0, and is finite even when
    ``growth`` exceeds ``rate`` (the horizon is finite).
    """
    r = float(rate)
    g = float(growth)
    n = int(years)
    if n <= 0:
        return 0.0
    if r <= 0.0 and g == 0.0:
        return float(n)
    if abs(r - g) < 1e-12:
        # Degenerate case: each term is 1/(1+r), so the sum is n/(1+r).
        return n / (1.0 + r)
    ratio = (1.0 + g) / (1.0 + r)
    return (1.0 - ratio ** n) / (1.0 + r) / (1.0 - ratio)


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
