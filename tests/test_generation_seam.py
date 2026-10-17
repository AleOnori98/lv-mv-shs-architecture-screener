"""The generation-cost seam: the boundary a sizing/dispatch model (MicroGridsPy) plugs into.

The properties tested here are the ones that make the substitution a one-class change:
aggregate input, avoided cost as a difference rather than a slice, and exactness of the
current separable model.
"""
from __future__ import annotations

import pytest

from core.economics.generation import (
    FlatLcoeGeneration,
    GenerationCostModel,
    ServedEnergy,
    avoided_present_value,
    model_from_parameters,
    served_energy_for,
)
from core.economics.npc import growth_annuity_factor

KWH = {"a": 1000.0, "b": 500.0, "c": 0.0}
GROWTH = {"a": 0.04, "b": 0.02, "c": 0.05}
T, R, LCOE = 15, 0.10, 0.25


def _pv(model, served):
    return model.present_value(served, horizon_years=T, discount_rate=R)


def test_served_energy_aggregates_and_keeps_components():
    s = served_energy_for(["a", "b", "c"], KWH, GROWTH)
    assert s.annual_kwh == pytest.approx(1500.0)
    # Energy-weighted, so the larger customer's growth dominates.
    assert s.growth_rate == pytest.approx((1000 * 0.04 + 500 * 0.02) / 1500)
    # The zero-energy customer contributes nothing and is not a component.
    assert s.n_customers == 2
    assert s.components == ((1000.0, 0.04), (500.0, 0.02))


def test_flat_model_is_exact_per_customer_not_on_aggregated_growth():
    """Aggregating growth first is a different number (Jensen); the flat model must not."""
    model = FlatLcoeGeneration(LCOE)
    s = served_energy_for(["a", "b"], KWH, GROWTH)

    exact = LCOE * sum(k * growth_annuity_factor(R, g, T) for k, g in s.components)
    on_aggregate = LCOE * s.annual_kwh * growth_annuity_factor(R, s.growth_rate, T)

    assert _pv(model, s) == pytest.approx(exact)
    assert on_aggregate != pytest.approx(exact, rel=1e-6)   # they really do differ


def test_model_falls_back_to_the_aggregate_when_components_are_absent():
    """A model given only an aggregate (as a dispatch model would be) still prices it."""
    model = FlatLcoeGeneration(LCOE)
    s = ServedEnergy(annual_kwh=1500.0, growth_rate=0.03)
    assert _pv(model, s) == pytest.approx(LCOE * 1500.0 * growth_annuity_factor(R, 0.03, T))


def test_avoided_cost_is_a_difference_and_equals_the_slice_only_because_flat_is_separable():
    """The property that lets the SHS screen survive substitution.

    The screen must credit cost(all) - cost(all without the branch). For a flat LCOE that
    happens to equal the branch's own slice, which is why the code could get away with the
    slice; for a sized-and-dispatched plant it does not, because serving less energy is not
    cheaper by that energy's average cost.
    """
    model = FlatLcoeGeneration(LCOE)
    whole = served_energy_for(["a", "b"], KWH, GROWTH)
    without_a = served_energy_for(["b"], KWH, GROWTH)

    difference = avoided_present_value(model, whole, without_a, horizon_years=T, discount_rate=R)
    slice_of_a = _pv(model, served_energy_for(["a"], KWH, GROWTH))
    assert difference == pytest.approx(slice_of_a, rel=1e-12)


def test_a_non_separable_model_breaks_the_slice_identity():
    """Guards the reason the seam exists, using a model with economies of scale."""

    class _ScaleModel:
        """Unit cost falls with volume, as a sized plant's does."""

        def present_value(self, served, *, horizon_years, discount_rate):
            if served.annual_kwh <= 0:
                return 0.0
            return served.annual_kwh ** 0.9

    model: GenerationCostModel = _ScaleModel()
    whole = served_energy_for(["a", "b"], KWH, GROWTH)
    without_a = served_energy_for(["b"], KWH, GROWTH)

    difference = avoided_present_value(model, whole, without_a, horizon_years=T, discount_rate=R)
    slice_of_a = _pv(model, served_energy_for(["a"], KWH, GROWTH))
    # The slice OVERSTATES the saving -- which is exactly how a flat LCOE over-selects SHS.
    assert difference < slice_of_a
    assert difference > 0


def test_disabled_or_empty_generation_costs_nothing():
    assert _pv(FlatLcoeGeneration(0.0), served_energy_for(["a"], KWH, GROWTH)) == 0.0
    assert _pv(FlatLcoeGeneration(LCOE), served_energy_for([], KWH, GROWTH)) == 0.0
    assert _pv(FlatLcoeGeneration(LCOE), served_energy_for(["c"], KWH, GROWTH)) == 0.0


def test_model_is_selected_from_the_cost_parameters():
    model = model_from_parameters({"generation_lcoe_per_kwh": 0.31})
    assert isinstance(model, FlatLcoeGeneration)
    assert model.lcoe_per_kwh == pytest.approx(0.31)


def test_losses_are_carried_for_a_dispatch_model_and_ignored_by_the_flat_one():
    """A plant must generate delivered + losses. The flat model prices losses as their own
    NPC component, so it must NOT also charge them here (that would double-count)."""
    model = FlatLcoeGeneration(LCOE)
    plain = served_energy_for(["a"], KWH, GROWTH)
    with_losses = served_energy_for(["a"], KWH, GROWTH, losses_kwh=200.0)
    assert with_losses.losses_kwh == 200.0
    assert _pv(model, with_losses) == pytest.approx(_pv(model, plain))
