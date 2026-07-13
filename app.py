"""Landing page for the Simplified LV/MV + Stand-alone MVP.

The full workflow lives on the sidebar page **"MVP Topology Planning"**.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LV/MV + Stand-alone MVP", layout="wide", initial_sidebar_state="expanded")

st.title("LV/MV + Stand-alone Mini-Grid MVP")
st.markdown(
    """
A sequential **diagnostic workflow** for rural mini-grid architecture selection. It reuses a
road-pole + MST topology engine and validates networks with **pandapower** (balanced
three-phase), then compares three architectures by **least-feasible net present cost (NPC)**:

1. **Full LV** — every customer on a single low-voltage network.
2. **LV + Stand-alone** — remote clusters cheaper to serve off-grid become solar stand-alone systems.
3. **MV/LV service zones** — problematic remote clusters served by local MV/LV transformers.

> This is a diagnostic comparison, **not** a full co-optimization model.
"""
)

st.info("Open **MVP Topology Planning** in the sidebar, then **Load example dataset → Run**.")

c1, c2, c3 = st.columns(3)
c1.markdown("#### Inputs\nCustomers, source, LV/MV conductors, transformers, stand-alone packages, "
            "cost parameters, and optional roads. Schemas + defaults in `core/io/input_schemas.py`.")
c2.markdown("#### Pipeline\nGenerate LV topology → size conductors → detect remote clusters → "
            "stand-alone NPC → build MV/LV zones → validate → compare.")
c3.markdown("#### Outputs\nRecommended architecture, comparison + cost + validation tables, "
            "customer strategy, decision log, candidate maps, and downloadable GeoJSON/CSV/ZIP.")

st.caption(
    "Limitations: balanced three-phase only · fixed source · heuristic topology and transformer "
    "placement · cluster-level stand-alone decisions · single flat coincidence factor + flat generation "
    "LCOE · scenario-based future expansion · illustrative costs."
)
