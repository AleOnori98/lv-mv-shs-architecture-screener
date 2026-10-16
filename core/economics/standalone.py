"""Stand-alone eligibility and package selection per customer / cluster."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .npc import standalone_npc


@dataclass
class StandaloneChoice:
    customer_id: str
    eligible: bool
    package_id: Optional[str]
    npc: float
    reason: str


def _horizon_peak(row, horizon_years: int) -> float:
    """Peak demand at the planning horizon [kW]."""
    return float(row["peak_kw_now"]) * ((1.0 + float(row["growth_rate"])) ** int(horizon_years))


def choose_package_for_customer(
    row: pd.Series,
    packages: pd.DataFrame,
    *,
    horizon_years: int,
    discount_rate: float,
    size_on_horizon: bool = True,
    sizing_tolerance: float = 0.0,
) -> StandaloneChoice:
    """Cheapest package covering this customer, or an ineligible choice with the reason.

    ``sizing_tolerance`` widens each package's peak and energy headroom by that fraction
    before testing it. The catalogue is discrete and the horizon peak is a compounded
    number, so without it the test is a knife edge: a Mafa household at 0.25 kW growing 4%
    over 15 years needs 0.45024 kW against ``SHS_T2``'s 0.45 kW limit and misses by 0.05%,
    which pushes it to ``SHS_T3`` at an NPC of 1329 instead of 823 -- a 61% cost jump from a
    rounding-scale margin. With all-or-nothing subtree eligibility one such customer also
    vetoes an entire branch.

    ``size_on_horizon`` applies to BOTH peak and energy. Sizing the peak at the horizon
    while testing energy at today's consumption compared a grown quantity against an
    ungrown one and let packages qualify on energy they would be short of by year T.
    """
    cid = str(row["customer_id"])
    if bool(row["must_connect_flag"]):
        return StandaloneChoice(cid, False, None, 0.0, "must_connect_flag=True")
    if not bool(row["can_be_standalone_flag"]):
        return StandaloneChoice(cid, False, None, 0.0, "can_be_standalone_flag=False")

    growth = (1.0 + float(row["growth_rate"])) ** int(horizon_years) if size_on_horizon else 1.0
    peak = float(row["peak_kw_now"]) * growth
    akwh = float(row["annual_kwh_now"]) * growth
    tier = int(row["demand_tier"])
    headroom = 1.0 + max(0.0, float(sizing_tolerance))

    feasible = packages[
        (packages["service_tier"] >= tier)
        & (packages["max_peak_kw"] * headroom >= peak)
        & (packages["annual_kwh"] * headroom >= akwh)
    ].copy()
    if feasible.empty:
        return StandaloneChoice(cid, False, None, 0.0, "no package satisfies tier/peak/energy requirement")

    feasible["_npc"] = feasible.apply(lambda p: standalone_npc(p.to_dict(), horizon_years, discount_rate), axis=1)
    best = feasible.sort_values("_npc").iloc[0]
    return StandaloneChoice(cid, True, str(best["package_id"]), float(best["_npc"]), "cheapest eligible package")


def cluster_standalone(
    customer_ids: List[str],
    customers_df: pd.DataFrame,
    packages: pd.DataFrame,
    *,
    horizon_years: int,
    discount_rate: float,
    size_on_horizon: bool = True,
    sizing_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Per-customer SHS selection for a cluster; eligible only if every customer is."""
    by_id = customers_df.set_index("customer_id")
    choices: List[StandaloneChoice] = []
    for cid in customer_ids:
        if cid not in by_id.index:
            continue
        row = pd.Series({**by_id.loc[cid].to_dict(), "customer_id": cid})
        choices.append(
            choose_package_for_customer(
                row,
                packages,
                horizon_years=horizon_years,
                discount_rate=discount_rate,
                size_on_horizon=size_on_horizon,
                sizing_tolerance=sizing_tolerance,
            )
        )
    eligible = bool(choices) and all(c.eligible for c in choices)
    total_npc = sum(c.npc for c in choices) if eligible else float("inf")
    blockers = [c.reason for c in choices if not c.eligible]
    return {
        "eligible": eligible,
        "total_npc": total_npc,
        "choices": choices,
        "n_customers": len(choices),
        "npc_per_customer": (total_npc / len(choices)) if (eligible and choices) else float("inf"),
        "blocking_reasons": sorted(set(blockers)),
    }
