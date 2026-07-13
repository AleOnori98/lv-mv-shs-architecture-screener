"""Iterative conductor sizing: start cheapest, upgrade critical lines until PF passes.

Generic over the network builder so the same loop sizes the Full-LV network, each
MV/LV service zone, and the MV/LV core. Critical lines are the overloaded lines plus
the lines on the path from the slack to any undervoltage bus; if none can be isolated
the loop falls back to upgrading every line (transparent, deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import networkx as nx
import pandas as pd

from .validation import PFOutcome, run_powerflow


@dataclass
class SizingResult:
    conductor_by_line: Dict[str, Dict[str, Any]]
    model: Any
    outcome: PFOutcome
    iterations: int
    passed: bool
    history: List[Dict[str, Any]] = field(default_factory=list)


def critical_lv_lines(model, outcome: PFOutcome, topology) -> Set[str]:
    """Line ids to upgrade: overloaded lines + lines feeding undervoltage buses."""
    crit: Set[str] = set(outcome.overloaded_line_ids)
    if not outcome.undervoltage_bus_indices:
        return crit
    pole_by_bus = {bus: pole for pole, bus in model.bus_by_pole.items()}
    line_id_by_pair = {}
    for u, v, data in topology.graph.edges(data=True):
        line_id_by_pair[frozenset((int(u), int(v)))] = str(data.get("line_id"))
    for bus in outcome.undervoltage_bus_indices:
        pole = pole_by_bus.get(int(bus))
        if pole is None or pole not in topology.graph or topology.slack_pole_id not in topology.graph:
            continue
        try:
            path = nx.shortest_path(topology.graph, topology.slack_pole_id, pole, weight="length_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for a, b in zip(path[:-1], path[1:]):
            lid = line_id_by_pair.get(frozenset((int(a), int(b))))
            if lid is not None:
                crit.add(lid)
    return crit


def size_conductors(
    line_ids: List[str],
    conductors_df: pd.DataFrame,
    build_model_fn: Callable[[Dict[str, Dict[str, Any]]], Any],
    constraints: Dict[str, float],
    *,
    critical_fn: Optional[Callable[[Any, PFOutcome], Set[str]]] = None,
    max_iters: int = 8,
) -> SizingResult:
    n_cond = len(conductors_df)
    ranks: Dict[str, int] = {lid: 0 for lid in line_ids}
    history: List[Dict[str, Any]] = []
    model = None
    outcome = None

    if not line_ids:
        model = build_model_fn({})
        outcome = run_powerflow(model, constraints)
        return SizingResult({}, model, outcome, 0, outcome.feasible, history)

    diverged_streak = 0
    for it in range(1, max_iters + 1):
        cond_by_line = {lid: conductors_df.iloc[ranks[lid]].to_dict() for lid in line_ids}
        model = build_model_fn(cond_by_line)
        outcome = run_powerflow(model, constraints)
        history.append(
            {
                "iteration": it,
                "feasible": outcome.feasible,
                "converged": outcome.converged,
                "min_vm_pu": outcome.min_vm_pu,
                "max_line_loading_percent": outcome.max_line_loading_percent,
                "bottleneck": outcome.active_bottleneck,
            }
        )
        if outcome.feasible:
            return SizingResult(cond_by_line, model, outcome, it, True, history)

        # Persistent non-convergence (e.g. voltage collapse on an over-large network)
        # cannot be fixed by reconductoring -> bail early instead of upgrading all lines
        # for the full iteration budget.
        diverged_streak = diverged_streak + 1 if not outcome.converged else 0
        if diverged_streak >= 2:
            break

        crit = critical_fn(model, outcome) if critical_fn else set()
        if not crit:
            crit = set(line_ids)  # transparent fallback: upgrade all
        upgraded = False
        for lid in crit:
            if ranks.get(lid, n_cond - 1) < n_cond - 1:
                ranks[lid] += 1
                upgraded = True
        if not upgraded:
            break  # no larger conductor available anywhere

    cond_by_line = {lid: conductors_df.iloc[ranks[lid]].to_dict() for lid in line_ids}
    return SizingResult(cond_by_line, model, outcome, len(history), bool(outcome and outcome.feasible), history)
