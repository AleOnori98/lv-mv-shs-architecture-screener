"""Central configuration for the LV/MV + Stand-alone MVP.

Only constants consumed by the retained topology engine (`core/distribution_*`) live
here. MVP-level economic/engineering defaults live with the input schemas
(`core/io/input_schemas.py`) and the Streamlit sidebar.
"""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class PathManager:
    """Centralised path helper (e.g. ``PathManager.EXAMPLES / "mvp"``)."""

    ROOT = ROOT_DIR
    EXAMPLES = ROOT / "examples"


# Default projected CRS fallback, used only when automatic local-UTM selection fails.
DEFAULT_TARGET_CRS = 32633

# Topology numerical tolerances (metres).
PRE_MST_DEDUPLICATION_TOLERANCE_M = 0.75
POST_DENSIFICATION_MERGE_TOLERANCE_M = 10.0
GEOMETRY_EPSILON_M = 1e-6

# Topology post-processing behaviour.
PROMOTE_SUPPORT_POLES_TO_SERVING = True

# Backbone edge-weight model.
DEFAULT_EDGE_WEIGHT_MODE = "euclidean"
ROAD_ATTRACTION_BUFFER_M = 20.0
ROAD_OFFROAD_PENALTY_FACTOR = 1.0
ROAD_EDGE_SAMPLE_SPACING_M = 10.0

# Optional user attributes preserved across topology exports/visualisation.
OPTIONAL_USER_ATTRIBUTE_COLUMNS = (
    "consumer_type",
    "priority_score",
    "must_connect_flag",
    "must_standalone_flag",
    "demand_tier",
    "customer_type",
    "peak_kw",
    "annual_demand_kwh",
)

# Priority / demand-tier bounds used by the association ranking.
DEFAULT_PRIORITY_SCORE = 3
MIN_PRIORITY_SCORE = 1
MAX_PRIORITY_SCORE = 5
DEFAULT_DEMAND_TIER = 0
HIGH_PRIORITY_THRESHOLD = 4
CRITICAL_PRIORITY_THRESHOLD = 5
