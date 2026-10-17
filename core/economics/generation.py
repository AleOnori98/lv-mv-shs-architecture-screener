"""The generation-cost seam: what it costs to generate the energy the grid serves.

Today this is a flat levelised cost per kWh. The intended replacement is **MicroGridsPy**,
which sizes and dispatches the generation plant (PV, storage, backup) against the
settlement's *aggregated* demand and returns the resulting levelised cost. This module
exists so that substitution is one new class rather than a rework of the screens.

Three properties make the boundary hold.

**Aggregate, not per customer.** :class:`ServedEnergy` describes the whole grid-served
demand. A sizing/dispatch model has economies of scale and a storage/dispatch interaction,
so its cost is *not* a sum over customers; anything that asks "what does this one customer's
generation cost?" cannot be answered by it. ``components`` is carried alongside the
aggregate for the one model that genuinely is separable (the flat LCOE), which keeps today's
per-customer growth discounting exact.

**Avoided cost is a difference, not a slice.** The SHS screen asks what is saved by taking a
leaf branch off-grid. Under a flat LCOE that is the branch's own energy, which is why the
code used to compute it directly. Under a sizing model it is
``present_value(all) - present_value(all without the branch)`` -- a difference of two plant
designs, materially smaller than the slice. :func:`avoided_present_value` is that difference
and is *algebraically identical* to the slice for a separable model, so adopting it now
changes no number and makes the screen correct under substitution.

**Losses are visible.** :class:`ServedEnergy` carries ``losses_kwh`` separately from
delivered energy. A dispatch model must generate delivered + losses, so it needs both; the
flat model prices losses as their own NPC component (see :mod:`.cost_model`) and therefore
ignores the field. When a real generation model lands, that component folds into the
trajectory instead of being priced separately -- the fold is the substitution's one
behavioural change, and it is explicit here rather than buried.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Sequence, Tuple

from .npc import growth_annuity_factor


@dataclass(frozen=True)
class ServedEnergy:
    """The grid-served demand a generation plant must supply, over the planning horizon.

    ``annual_kwh`` and ``growth_rate`` describe year-1 delivered energy and its
    energy-weighted annual growth. ``losses_kwh`` is year-1 network loss energy, which a
    plant must also generate. ``peak_kw`` is the diversified design peak, which a sizing
    model needs and a flat LCOE does not.

    ``components`` holds the per-customer ``(annual_kwh, growth_rate)`` pairs. A separable
    model should use them: aggregating growth first is not the same number (Jensen --
    measured at 0.24% on the Mafa case), and the flat LCOE is exact per customer.

    ``hourly_kw`` is the hook for the aggregated 8760-hour series MicroGridsPy sizes
    against. The repository already ships that shape (``examples/nga_1/Aggregated
    Demand.csv``) and per-category profiles to build it from; nothing reads them yet.
    """

    annual_kwh: float
    growth_rate: float = 0.0
    losses_kwh: float = 0.0
    peak_kw: float = 0.0
    n_customers: int = 0
    components: Tuple[Tuple[float, float], ...] = ()
    hourly_kw: Sequence[float] | None = None

    @property
    def is_empty(self) -> bool:
        return self.annual_kwh <= 0.0 and not self.components


class GenerationCostModel(Protocol):
    """Present value of generating a candidate's grid-served energy over the horizon."""

    def present_value(self, served: ServedEnergy, *, horizon_years: int,
                      discount_rate: float) -> float:
        ...


@dataclass(frozen=True)
class FlatLcoeGeneration:
    """Generation priced at one levelised cost per delivered kWh.

    A proxy for a generation model, not a model: the cost per kWh does not respond to how
    much energy is served or to the shape of the load, so it has **no economies of scale**.
    On Mafa this is *not* what drives the SHS screen: synthetic scale-economy curves leave
    the answer unchanged at the dataset price, which the grid price level relative to the
    SHS cost per kWh decides. See ``docs/CONVERGENCE_EXPERIMENT.md``.

    Separable in customers, so ``present_value`` sums each customer's own growth trajectory
    exactly rather than discounting an aggregate growth rate.
    """

    lcoe_per_kwh: float

    def present_value(self, served: ServedEnergy, *, horizon_years: int,
                      discount_rate: float) -> float:
        lcoe = float(self.lcoe_per_kwh)
        if lcoe <= 0.0 or served.is_empty:
            return 0.0
        if served.components:
            return lcoe * sum(
                kwh * growth_annuity_factor(discount_rate, growth, horizon_years)
                for kwh, growth in served.components
            )
        return lcoe * served.annual_kwh * growth_annuity_factor(
            discount_rate, served.growth_rate, horizon_years
        )


def served_energy_for(
    customer_ids,
    annual_kwh_by_id: Dict[str, float],
    growth_by_id: Dict[str, float],
    *,
    losses_kwh: float = 0.0,
    peak_kw: float = 0.0,
) -> ServedEnergy:
    """Aggregate a customer set into one :class:`ServedEnergy`.

    Customers with no recorded energy contribute nothing and are left out of
    ``components``, so a separable model does not pay for an empty trajectory.
    """
    parts: list[Tuple[float, float]] = []
    total = 0.0
    weighted = 0.0
    for cid in customer_ids:
        key = str(cid)
        kwh = float(annual_kwh_by_id.get(key, 0.0))
        if kwh <= 0.0:
            continue
        growth = float(growth_by_id.get(key, 0.0))
        parts.append((kwh, growth))
        total += kwh
        weighted += kwh * growth
    return ServedEnergy(
        annual_kwh=total,
        growth_rate=(weighted / total) if total > 0 else 0.0,
        losses_kwh=float(losses_kwh),
        peak_kw=float(peak_kw),
        n_customers=len(parts),
        components=tuple(parts),
    )


def avoided_present_value(
    model: GenerationCostModel,
    served_with: ServedEnergy,
    served_without: ServedEnergy,
    *,
    horizon_years: int,
    discount_rate: float,
) -> float:
    """Generation cost avoided by removing some customers from the grid-served set.

    The **difference between two plant designs**, not the departing customers' own slice.
    Identical to the slice for a separable model (so this changes nothing today), and the
    only correct definition once generation is sized and dispatched: a plant serving less
    energy is not cheaper by that energy's average cost.
    """
    pv = lambda s: model.present_value(s, horizon_years=horizon_years, discount_rate=discount_rate)
    return pv(served_with) - pv(served_without)


def model_from_parameters(cost_params: Dict[str, float]) -> GenerationCostModel:
    """The generation model a run should use, from its cost parameters.

    One place to switch implementations. A MicroGridsPy-backed model would be selected
    here, on a parameter naming the generation model rather than a bare LCOE.
    """
    return FlatLcoeGeneration(float(cost_params.get("generation_lcoe_per_kwh", 0.0) or 0.0))
