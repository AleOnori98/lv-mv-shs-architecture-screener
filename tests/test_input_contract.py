"""The customers.csv contract: documented defaults and documented accepted values.

Both were documented but unenforced. Defaults were promised for missing columns yet only
worked for missing *values*, and CUSTOMER_TYPES listed the accepted types while nothing
checked them -- so a typo passed through as an unknown type and, because standalone
eligibility defaults to "not a grid-priority type", silently made an institutional load
SHS-eligible.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from core.io.input_schemas import CUSTOMER_TYPES, GRID_PRIORITY_TYPES, normalize_customers

REQUIRED_ONLY = pd.DataFrame({
    "customer_id": ["a", "b"],
    "latitude": [11.92, 11.93],
    "longitude": [13.59, 13.60],
    "customer_type": ["household", "clinic"],
})


def _quiet(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return normalize_customers(df)


def test_defaults_fill_when_optional_columns_are_absent_entirely():
    """The regression: this raised AttributeError on a file with only the required columns."""
    out = _quiet(REQUIRED_ONLY)
    assert list(out["peak_kw_now"]) == [0.3, 0.3]
    assert out["annual_kwh_now"].tolist() == pytest.approx([0.3 * 8760 * 0.2] * 2)
    assert list(out["power_factor"]) == [0.90, 0.90]
    assert list(out["growth_rate"]) == [0.03, 0.03]
    assert list(out["demand_tier"]) == [2, 2]
    assert list(out["priority_weight"]) == [1.0, 1.0]
    assert not out["must_connect_flag"].any()


def test_defaults_still_fill_individual_missing_values():
    df = REQUIRED_ONLY.assign(peak_kw_now=[1.5, None], power_factor=[0.8, None])
    out = _quiet(df)
    assert list(out["peak_kw_now"]) == [1.5, 0.3]
    assert list(out["power_factor"]) == [0.8, 0.90]


def test_power_factor_is_clamped():
    out = _quiet(REQUIRED_ONLY.assign(power_factor=[0.1, 2.0]))
    assert list(out["power_factor"]) == [0.5, 1.0]


def test_grid_priority_types_are_not_standalone_eligible_by_default():
    out = _quiet(REQUIRED_ONLY)           # household + clinic
    assert bool(out.loc[0, "can_be_standalone_flag"]) is True
    assert bool(out.loc[1, "can_be_standalone_flag"]) is False
    assert "clinic" in GRID_PRIORITY_TYPES


def test_unknown_customer_type_warns_because_it_changes_eligibility():
    """A typo'd institutional type is the dangerous case: it defaults to SHS-eligible."""
    df = REQUIRED_ONLY.assign(customer_type=["household", "hosptial"])
    with pytest.warns(UserWarning, match="unrecognised customer_type"):
        out = normalize_customers(df)
    # The row is kept as-is (unusual local categories must still load) ...
    assert out.loc[1, "customer_type"] == "hosptial"
    # ... which is exactly why the warning matters: it is now SHS-eligible.
    assert bool(out.loc[1, "can_be_standalone_flag"]) is True


def test_known_customer_types_are_silent():
    df = REQUIRED_ONLY.assign(customer_type=["household", "hospital"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalize_customers(df)
    assert bool(out.loc[1, "can_be_standalone_flag"]) is False
    assert set(df["customer_type"]) <= set(CUSTOMER_TYPES)


def test_must_connect_forces_grid_service():
    df = REQUIRED_ONLY.assign(must_connect_flag=[True, False], can_be_standalone_flag=[True, True])
    out = _quiet(df)
    assert bool(out.loc[0, "can_be_standalone_flag"]) is False  # must_connect wins
    assert bool(out.loc[1, "can_be_standalone_flag"]) is True


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"customer_id": ["a"], "latitude": [1.0], "longitude": [2.0]}),          # no type
    pd.DataFrame({"customer_id": ["a", "a"], "latitude": [1.0, 2.0],                      # duplicate id
                  "longitude": [3.0, 4.0], "customer_type": ["household"] * 2}),
    pd.DataFrame({"customer_id": ["a"], "latitude": [None], "longitude": [2.0],           # bad coords
                  "customer_type": ["household"]}),
])
def test_hard_stops_still_raise(bad):
    with pytest.raises(ValueError):
        normalize_customers(bad)
