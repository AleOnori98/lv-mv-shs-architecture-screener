"""Stand-alone eligibility and NPC."""
from __future__ import annotations

import pandas as pd

from core.economics.standalone import choose_package_for_customer, cluster_standalone
from core.economics.npc import standalone_npc


def _packages():
    return pd.DataFrame([
        {"package_id": "T1", "service_tier": 1, "max_peak_kw": 0.3, "annual_kwh": 600, "capex": 350, "lifetime_years": 8, "annual_om_fraction": 0.04, "replacement_fraction": 0.6},
        {"package_id": "T2", "service_tier": 2, "max_peak_kw": 0.6, "annual_kwh": 1200, "capex": 700, "lifetime_years": 8, "annual_om_fraction": 0.04, "replacement_fraction": 0.6},
    ])


def _cust(**kw):
    base = {"customer_id": "X", "peak_kw_now": 0.25, "annual_kwh_now": 500, "power_factor": 0.9,
            "growth_rate": 0.03, "demand_tier": 1, "must_connect_flag": False, "can_be_standalone_flag": True}
    base.update(kw)
    return pd.Series(base)


def test_standalone_eligibility_blocks_must_connect():
    choice = choose_package_for_customer(_cust(must_connect_flag=True), _packages(), horizon_years=15, discount_rate=0.08)
    assert not choice.eligible
    assert "must_connect" in choice.reason


def test_standalone_eligibility_blocks_non_standalone_type():
    choice = choose_package_for_customer(_cust(can_be_standalone_flag=False), _packages(), horizon_years=15, discount_rate=0.08)
    assert not choice.eligible


def test_standalone_npc_positive():
    pkg = _packages().iloc[0].to_dict()
    npc = standalone_npc(pkg, horizon_years=15, discount_rate=0.08)
    assert npc > pkg["capex"]  # capex + O&M + replacements - salvage > capex


def test_cluster_requires_every_customer_eligible():
    customers = pd.DataFrame([_cust(customer_id="A"), _cust(customer_id="B", must_connect_flag=True)])
    out = cluster_standalone(["A", "B"], customers, _packages(), horizon_years=15, discount_rate=0.08)
    assert out["eligible"] is False
