"""Map rendering tests for served-status visualization."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.viz import maps


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "A", "latitude": 0.001, "longitude": 0.001},
            {"customer_id": "B", "latitude": 0.002, "longitude": 0.002},
            {"customer_id": "C", "latitude": 0.003, "longitude": 0.003},
        ]
    )


def test_classify_customers_marks_unserved_separately():
    status = maps.classify_customers(grid_ids=["A"], standalone_ids=["B"], all_ids=["A", "B", "C"])
    assert status == {"A": "grid", "B": "standalone", "C": "unserved"}


def test_case_map_draws_unserved_customers_distinctly(monkeypatch):
    monkeypatch.setattr(
        maps,
        "case_lines_gdf",
        lambda case, source: pd.DataFrame(columns=["line_id", "voltage_level", "geometry"]),
    )
    monkeypatch.setattr(
        maps,
        "transformers_gdf",
        lambda case: pd.DataFrame(columns=["transformer_id", "sn_kva", "geometry"]),
    )
    case = SimpleNamespace(
        grid_customer_ids=["A"],
        standalone_customer_ids=["B"],
        plan=None,
        topology=None,
        model=None,
        model_now=None,
        outcome_horizon=None,
        outcome_now=None,
    )
    source = {"latitude": 0.0, "longitude": 0.0}

    html = maps.case_map(case, _customers(), source).get_root().render()

    assert "Customers: grid" in html
    assert "Customers: stand-alone" in html
    assert "Customers: UNSERVED" in html
    assert "C - UNSERVED - not connected" in html
