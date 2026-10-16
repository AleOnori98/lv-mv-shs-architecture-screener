"""Regression tests for the feasibility/costing gate.

These cover the two defects that made a BROKEN candidate look better than a working one:
  * a non-raising non-convergence falling through to the limit checks (NaN comparisons are
    always False, so every limit "passed" and the outcome read as feasible);
  * diverged losses being NaN-guarded to a zero cost, so an infeasible candidate was
    systematically cheaper and the least-infeasible fallback preferred the worst candidate.
"""
from __future__ import annotations

import math

import pandapower as pp
import pytest

from core.economics import cost_model as cm
from core.electrical.validation import run_powerflow, violation_severity

CONSTRAINTS = {
    "voltage_min_pu": 0.90, "voltage_max_pu": 1.05,
    "max_line_loading_percent": 100.0, "max_transformer_loading_percent": 100.0,
}


class _Model:
    """Minimal stand-in for PPModel (run_powerflow only needs .net and .total_load_kw)."""

    def __init__(self, net, total_load_kw=10.0):
        self.net = net
        self.total_load_kw = total_load_kw


def _two_bus_net(load_mw=0.01, length_km=0.1):
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=0.4)
    b1 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, length_km=length_km,
        r_ohm_per_km=0.64, x_ohm_per_km=0.085, c_nf_per_km=0.0, max_i_ka=0.16, name="L1",
    )
    pp.create_load(net, bus=b1, p_mw=load_mw, q_mvar=0.0)
    return net


def test_healthy_network_is_feasible_and_converged():
    out = run_powerflow(_Model(_two_bus_net()), CONSTRAINTS)
    assert out.converged and out.feasible
    assert out.active_bottleneck == "within limits"
    assert not math.isnan(out.losses_kw)


def test_non_raising_non_convergence_is_not_reported_feasible(monkeypatch):
    """The regression: runpp returning without convergence must not read as 'within limits'.

    Previously `converged = bool(net.converged)` was computed and discarded, the limit
    checks ran against NaN results (every comparison False) and the outcome came back
    feasible with bottleneck "within limits".
    """
    net = _two_bus_net()

    def _fake_runpp(n, **kwargs):
        n["converged"] = False  # solved nothing, raised nothing

    monkeypatch.setattr(pp, "runpp", _fake_runpp)
    out = run_powerflow(_Model(net), CONSTRAINTS)

    assert out.converged is False
    assert out.feasible is False
    assert out.active_bottleneck == "power flow did not converge"
    assert math.isnan(out.losses_kw)


def test_raising_non_convergence_still_reported_as_such(monkeypatch):
    def _boom(n, **kwargs):
        raise RuntimeError("Power Flow nr did not converge")

    monkeypatch.setattr(pp, "runpp", _boom)
    out = run_powerflow(_Model(_two_bus_net()), CONSTRAINTS)
    assert not out.converged and not out.feasible
    assert "did not converge" in out.active_bottleneck
    assert math.isnan(out.losses_kw)


def test_unknown_losses_cost_nan_not_zero():
    """A diverged candidate must not be handed a free pass on its loss cost."""
    params = {"loss_load_factor": 0.35, "loss_value_per_kwh": 0.25,
              "discount_rate": 0.10, "planning_horizon_years": 15}
    assert math.isnan(cm.loss_present_value(float("nan"), params))
    assert cm.loss_present_value(20.0, params) > 0.0


def test_cost_breakdown_flags_unknown_component_and_excludes_it():
    known = cm.CostBreakdown(lv_line_cost=100.0, generation_npc=50.0, loss_pv_cost=10.0)
    assert known.npc_complete and known.missing_components == []
    assert known.total_npc == pytest.approx(160.0)

    unknown = cm.CostBreakdown(lv_line_cost=100.0, generation_npc=50.0, loss_pv_cost=float("nan"))
    assert not unknown.npc_complete
    assert unknown.missing_components == ["loss_pv_cost"]
    # The total stays finite and sortable, but it is a LOWER BOUND, not a comparable cost.
    assert unknown.total_npc == pytest.approx(150.0)
    assert not math.isnan(unknown.grid_capex)


def test_loss_energy_tracks_generation_lcoe_unless_overridden():
    base = {"generation_lcoe_per_kwh": 0.25}
    assert cm.loss_energy_price({**base, "loss_value_per_kwh": 0.0}) == pytest.approx(0.25)
    assert cm.loss_energy_price(base) == pytest.approx(0.25)
    assert cm.loss_energy_price({**base, "loss_value_per_kwh": 0.40}) == pytest.approx(0.40)


def test_violation_severity_orders_broken_networks():
    from core.electrical.validation import PFOutcome

    def mk(**kw):
        d = dict(converged=True, feasible=False, min_vm_pu=0.95, max_vm_pu=1.0,
                 max_line_loading_percent=50.0, max_trafo_loading_percent=50.0,
                 losses_kw=1.0, losses_percent=1.0, total_load_kw=100.0, active_bottleneck="")
        d.update(kw)
        return PFOutcome(**d)

    assert violation_severity(mk(), CONSTRAINTS) == pytest.approx(0.0)
    mild = violation_severity(mk(min_vm_pu=0.88), CONSTRAINTS)
    bad = violation_severity(mk(min_vm_pu=0.80), CONSTRAINTS)
    assert 0.0 < mild < bad
    # Overloads contribute as a fraction of the limit, comparable with pu voltage breaches.
    assert violation_severity(mk(max_line_loading_percent=150.0), CONSTRAINTS) == pytest.approx(0.5)
    # Nothing is worse than a network that cannot be solved at all.
    assert violation_severity(mk(converged=False), CONSTRAINTS) == float("inf")
    assert violation_severity(None, CONSTRAINTS) == float("inf")


# --------------------------------------------------------------------------------------
# Input-contract hygiene: an unrecognised parameter used to be dropped in silence, so a
# typo answered a different question with no indication anything was wrong.
# --------------------------------------------------------------------------------------
def test_unknown_cost_parameter_warns_and_suggests():
    from core.io.input_schemas import normalize_cost_parameters

    with pytest.warns(UserWarning, match="discount_rate"):
        out = normalize_cost_parameters({"discount_rte": 0.42})
    assert out["discount_rate"] == 0.08  # default used, not the typo'd value


def test_retired_cost_parameters_stay_silent():
    """Old datasets still carry retired keys; they load without noise the user can't act on."""
    import warnings as _w
    from core.io.input_schemas import normalize_cost_parameters

    with _w.catch_warnings():
        _w.simplefilter("error")
        normalize_cost_parameters({"dmax_lv_service_zone_m": 450.0, "lmax_lv_path_m": 700.0})


def test_strict_mode_rejects_unknown_parameter():
    from core.io.input_schemas import normalize_cost_parameters

    with pytest.raises(ValueError, match="unrecognised key"):
        normalize_cost_parameters({"nonsense_knob": 1.0}, strict=True)
