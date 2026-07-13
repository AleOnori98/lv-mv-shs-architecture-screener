"""Shared fixtures for the simplified-MVP tests."""
from __future__ import annotations

import pytest

from core.io.example_data import load_example_inputs
from core.workflow.mvp_runner import run_simplified_mvp

ENGINEERING = {
    "max_user_connection_radius_m": 35, "max_users_per_pole": 16, "max_pole_span_m": 50,
    "road_pole_spacing_m": 40, "planning_horizon_years": 15, "discount_rate": 0.08,
    "voltage_min_pu": 0.90, "voltage_max_pu": 1.05, "max_line_loading_percent": 100,
    "max_transformer_loading_percent": 100, "dmax_lv_service_zone_m": 700, "lmax_lv_path_m": 850,
}


@pytest.fixture(scope="session")
def engineering():
    return dict(ENGINEERING)


@pytest.fixture(scope="session")
def inputs():
    return load_example_inputs()


@pytest.fixture(scope="session")
def mvp(inputs):
    return run_simplified_mvp(
        inputs["customers"], inputs["roads"], inputs["source"], inputs["lv_conductors"],
        inputs["mv_conductors"], inputs["transformers"], inputs["standalone_packages"],
        inputs["cost_parameters"], ENGINEERING,
    )
