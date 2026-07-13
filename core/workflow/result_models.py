"""Lightweight dataclasses for MVP case results and the overall comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class CaseResult:
    candidate_id: str
    architecture: str
    feasible: bool
    cost: Any                      # CostBreakdown
    outcome_now: Any = None        # PFOutcome at current demand (diversified/design coincidence)
    outcome_horizon: Any = None    # PFOutcome at horizon demand (diversified/design coincidence)
    outcome_stress: Any = None     # PFOutcome at horizon demand, coincident peak (factor 1.0)
    n_grid_customers: int = 0
    n_standalone_customers: int = 0
    lv_length_m: float = 0.0
    mv_length_m: float = 0.0
    n_poles: int = 0
    service_drop_total_m: float = 0.0
    n_transformers: int = 0
    must_connect_violations: int = 0
    service_tier_violations: int = 0
    decision_summary: str = ""
    sizing_iterations: int = 0
    # Artifacts for maps / exports / strategy table.
    topology: Any = None
    plan: Any = None
    model_now: Any = None
    model: Any = None
    model_stress: Any = None
    conductor_by_line: Dict[str, Any] = field(default_factory=dict)
    standalone_assignments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    grid_customer_ids: List[str] = field(default_factory=list)
    standalone_customer_ids: List[str] = field(default_factory=list)
    customer_assignment: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    transformers: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MVPResult:
    cases: Dict[str, CaseResult]
    comparison_table: pd.DataFrame
    customer_strategy_table: pd.DataFrame
    decision_log: pd.DataFrame
    validation_summary: pd.DataFrame
    cost_breakdown_table: pd.DataFrame
    tier_service_table: pd.DataFrame
    recommended: str
    recommendation_note: str
    clusters: List[Any] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    exports: Dict[str, Any] = field(default_factory=dict)
