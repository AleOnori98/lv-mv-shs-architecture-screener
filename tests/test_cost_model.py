"""Cost model + NPC aggregation."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.economics.npc import annuity_factor, standalone_npc
from core.economics import cost_model as cm
from core.workflow.mvp_runner import _avoided_lv_cost


def test_annuity_factor_matches_closed_form():
    af = annuity_factor(0.10, 10)
    assert abs(af - 6.1446) < 1e-3


def test_annuity_factor_zero_rate():
    assert annuity_factor(0.0, 12) == 12.0


def test_cost_breakdown_totals():
    c = cm.CostBreakdown(lv_line_cost=1000, mv_line_cost=500, pole_cost=300, transformer_cost=2000,
                         connection_metering_cost=400, standalone_npc=600, loss_pv_cost=250,
                         generation_npc=1500)
    assert c.grid_capex == 1000 + 500 + 300 + 2000 + 400
    assert c.total_npc == c.grid_capex + 600 + 250 + 1500


def test_generation_present_value_levelises_grid_energy():
    params = {"generation_lcoe_per_kwh": 0.25, "discount_rate": 0.10, "planning_horizon_years": 10}
    # 10,000 kWh/yr levelised at 0.25 EUR/kWh over the horizon annuity.
    expected = 0.25 * 10000.0 * annuity_factor(0.10, 10)
    assert abs(cm.generation_present_value(10000.0, params) - expected) < 1e-6
    # Disabled (zero LCOE) or no grid energy -> no generation cost.
    assert cm.generation_present_value(10000.0, {**params, "generation_lcoe_per_kwh": 0.0}) == 0.0
    assert cm.generation_present_value(0.0, params) == 0.0


def test_line_cost_uses_length_and_civil_multiplier():
    cond = {"L0": {"cost_per_m": 10.0}}
    length = {"L0": 100.0}
    assert cm.line_cost(cond, length, 1.2) == 100.0 * 10.0 * 1.2


def test_loss_present_value_non_negative():
    params = {"loss_load_factor": 0.3, "loss_value_per_kwh": 0.2, "discount_rate": 0.08, "planning_horizon_years": 15}
    assert cm.loss_present_value(5.0, params) > 0
    assert cm.loss_present_value(float("nan"), params) == 0.0


def test_avoided_lv_cost_includes_service_drops_and_losses():
    params = {
        "lv_connection_cost_per_customer": 100.0,
        "metering_protection_cost_per_customer": 50.0,
        "pole_cost": 120.0,
        "civil_logistics_multiplier": 1.2,
        "loss_load_factor": 0.3,
        "loss_value_per_kwh": 0.2,
        "discount_rate": 0.08,
        "planning_horizon_years": 15,
    }
    # Service drops are precomputed onto the candidate (metadata["service_drop_m"]); all poles in
    # the subtree are avoided (the boundary pole is upstream of the subtree, so it is not counted).
    cluster = SimpleNamespace(
        customer_ids=["C1", "C2"],
        n_customers=2,
        pole_ids=[1, 2, 3],
        branch_cost=100.0,
        branch_losses_kw=1.0,
        metadata={"service_drop_m": 30.0},
    )
    ctx = {
        "lv_conductors": pd.DataFrame(
            [
                {"conductor_id": "small", "rank": 1, "cost_per_m": 2.0},
                {"conductor_id": "large", "rank": 2, "cost_per_m": 5.0},
            ]
        ),
        "params": params,
    }

    expected_without_loss = (
        100.0  # branch cost (internal lines + entry edge)
        + 30.0 * 2.0 * 1.2  # service drops with cheapest LV conductor
        + 2 * (100.0 + 50.0)  # connection and metering
        + 3 * 120.0 * 1.2  # all subtree poles avoided
    )
    expected = expected_without_loss + cm.loss_present_value(1.0, params)

    assert _avoided_lv_cost(cluster, ctx) == expected
