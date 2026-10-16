"""Run pandapower power flow and summarize feasibility against MVP constraints."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandapower as pp


@dataclass
class PFOutcome:
    converged: bool
    feasible: bool
    min_vm_pu: float
    max_vm_pu: float
    max_line_loading_percent: float
    max_trafo_loading_percent: float
    losses_kw: float
    losses_percent: float
    total_load_kw: float
    active_bottleneck: str
    overloaded_line_ids: List[str] = field(default_factory=list)
    undervoltage_bus_indices: List[int] = field(default_factory=list)
    error: str = ""


def _not_converged(total_load_kw: float, error: str = "") -> PFOutcome:
    """Outcome for a power flow that did not solve.

    Every numeric result is NaN: nothing about the network state is known, so no limit
    check may be evaluated (a NaN comparison is always False and would otherwise read as
    "within limits"). ``losses_kw`` stays NaN so the cost model can report the loss
    component as *unknown* rather than silently charging zero.
    """
    return PFOutcome(
        converged=False, feasible=False, min_vm_pu=float("nan"), max_vm_pu=float("nan"),
        max_line_loading_percent=float("nan"), max_trafo_loading_percent=float("nan"),
        losses_kw=float("nan"), losses_percent=float("nan"), total_load_kw=float(total_load_kw),
        active_bottleneck="power flow did not converge", error=error,
    )


def run_powerflow(model, constraints: Dict[str, float]) -> PFOutcome:
    net = model.net
    v_min = float(constraints["voltage_min_pu"])
    v_max = float(constraints["voltage_max_pu"])
    max_line = float(constraints["max_line_loading_percent"])
    max_trafo = float(constraints["max_transformer_loading_percent"])

    try:
        pp.runpp(net, calculate_voltage_angles=False, init="flat")
    except Exception as exc:  # diverged / singular
        return _not_converged(model.total_load_kw, str(exc))

    # pandapower normally raises on non-convergence, but the flag is authoritative: a
    # non-raising non-convergence must never fall through to the limit checks below.
    if not bool(getattr(net, "converged", False)):
        return _not_converged(model.total_load_kw, "runpp returned without convergence")

    res_bus = net.res_bus
    res_line = net.res_line
    min_vm = float(res_bus["vm_pu"].min()) if len(res_bus) else 1.0
    max_vm = float(res_bus["vm_pu"].max()) if len(res_bus) else 1.0
    max_ll = float(res_line["loading_percent"].max()) if len(res_line) else 0.0
    max_tl = float(net.res_trafo["loading_percent"].max()) if len(net.res_trafo) else 0.0

    # A converged solve with NaN state (singular island, unsupplied bus) is not a usable
    # result either -- treat it as non-convergence rather than comparing NaN to a limit.
    if any(math.isnan(v) for v in (min_vm, max_vm, max_ll, max_tl)):
        return _not_converged(model.total_load_kw, "converged with non-finite bus/branch results")

    line_losses_kw = float(res_line["pl_mw"].sum()) * 1000.0 if len(res_line) else 0.0
    trafo_losses_kw = float(net.res_trafo["pl_mw"].sum()) * 1000.0 if len(net.res_trafo) else 0.0
    losses_kw = line_losses_kw + trafo_losses_kw
    total_kw = float(model.total_load_kw) or 1e-9
    losses_pct = 100.0 * losses_kw / total_kw

    overloaded = []
    if len(res_line):
        for idx, loading in res_line["loading_percent"].items():
            if float(loading) > max_line:
                overloaded.append(str(net.line.at[idx, "name"]))
    undervolt = [int(i) for i, v in res_bus["vm_pu"].items() if float(v) < v_min] if len(res_bus) else []

    # Feasibility + active bottleneck.
    reasons = []
    if min_vm < v_min:
        reasons.append(f"undervoltage {min_vm:.3f} pu < {v_min:.2f}")
    if max_vm > v_max:
        reasons.append(f"overvoltage {max_vm:.3f} pu > {v_max:.2f}")
    if max_ll > max_line:
        reasons.append(f"line loading {max_ll:.0f}% > {max_line:.0f}%")
    if max_tl > max_trafo:
        reasons.append(f"transformer loading {max_tl:.0f}% > {max_trafo:.0f}%")
    feasible = not reasons
    bottleneck = "within limits" if feasible else "; ".join(reasons)

    return PFOutcome(
        converged=True, feasible=feasible, min_vm_pu=min_vm, max_vm_pu=max_vm,
        max_line_loading_percent=max_ll, max_trafo_loading_percent=max_tl,
        losses_kw=losses_kw, losses_percent=losses_pct, total_load_kw=float(model.total_load_kw),
        active_bottleneck=bottleneck, overloaded_line_ids=overloaded, undervoltage_bus_indices=undervolt,
    )


def violation_severity(outcome: PFOutcome, constraints: Dict[str, float]) -> float:
    """How far outside its limits a converged outcome sits (0.0 = within limits).

    Sum of the per-limit breaches in comparable units: voltage breaches in pu, loading
    breaches as a fraction of the limit. Used to rank *infeasible* candidates, where NPC
    is not a meaningful ordering (a candidate that did not converge has no cost either).
    ``inf`` for a non-converged outcome: nothing worse than an unsolvable network.
    """
    if outcome is None or not outcome.converged:
        return float("inf")
    v_min = float(constraints["voltage_min_pu"])
    v_max = float(constraints["voltage_max_pu"])
    max_line = float(constraints["max_line_loading_percent"])
    max_trafo = float(constraints["max_transformer_loading_percent"])
    return (
        max(0.0, v_min - outcome.min_vm_pu)
        + max(0.0, outcome.max_vm_pu - v_max)
        + max(0.0, outcome.max_line_loading_percent - max_line) / 100.0
        + max(0.0, outcome.max_trafo_loading_percent - max_trafo) / 100.0
    )
