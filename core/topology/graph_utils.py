"""Graph helpers for the pole tree.

The engine returns the LV network as edges (``line_id``, ``u``, ``v``, ``length_m``).
These three helpers turn that into a graph, find the slack pole, and root the tree --
everything the terminal-subtree enumeration in :mod:`.subtree_candidates` needs.
"""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import pandas as pd


def build_pole_graph(edges_df: pd.DataFrame) -> nx.Graph:
    """Undirected weighted graph of poles; weight = physical edge length in metres."""
    g = nx.Graph()
    for _, row in edges_df.iterrows():
        g.add_edge(
            int(row["u"]),
            int(row["v"]),
            line_id=str(row["line_id"]),
            length_m=float(row["length_m"]),
        )
    return g


def nearest_node(points_xy: Dict[int, Tuple[float, float]], target_xy: Tuple[float, float]) -> int:
    """Nearest pole id to a target (x, y) using squared Euclidean distance."""
    tx, ty = target_xy
    best, best_d = None, float("inf")
    for pid, (x, y) in points_xy.items():
        d = (x - tx) ** 2 + (y - ty) ** 2
        if d < best_d:
            best, best_d = pid, d
    if best is None:
        raise ValueError("No poles available to match the source location.")
    return int(best)


def parents_from_root(graph: nx.Graph, root: int) -> Dict[int, int]:
    """Parent pole for each node in the rooted tree (root maps to itself)."""
    parents = {root: root}
    for parent, child in nx.bfs_edges(graph, root):
        parents[child] = parent
    return parents


