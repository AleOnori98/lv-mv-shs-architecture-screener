"""Graph helpers shared by the MVP topology steps.

The LV network produced by the legacy engine is a tree of poles connected by edges
(``line_id``, ``u``, ``v``, ``length_m``). These helpers root that tree at the source
pole and expose the distances / subtree queries used by branch detection.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

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


def path_lengths_from_root(graph: nx.Graph, root: int) -> Dict[int, float]:
    """Cumulative tree distance (metres) from the root to every reachable pole."""
    if root not in graph:
        # Isolated root (e.g. single-pole network): distance 0 to itself only.
        return {root: 0.0}
    return nx.single_source_dijkstra_path_length(graph, root, weight="length_m")


def parents_from_root(graph: nx.Graph, root: int) -> Dict[int, int]:
    """Parent pole for each node in the rooted tree (root maps to itself)."""
    parents = {root: root}
    for parent, child in nx.bfs_edges(graph, root):
        parents[child] = parent
    return parents


def descendants_of(graph: nx.Graph, root: int, node: int) -> List[int]:
    """All poles in the subtree rooted at ``node`` (towards the leaves), including node."""
    parents = parents_from_root(graph, root)
    children: Dict[int, List[int]] = {}
    for child, parent in parents.items():
        if child != parent:
            children.setdefault(parent, []).append(child)
    out: List[int] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def leaves(graph: nx.Graph, root: int) -> List[int]:
    """Leaf poles (degree 1, excluding the root)."""
    return [int(n) for n in graph.nodes if graph.degree(n) == 1 and n != root]


def trace_terminal_branch(graph: nx.Graph, root: int, leaf: int) -> Tuple[int, List[int]]:
    """Trace upstream from a leaf to the first branching node (degree > 2) or the root.

    Returns ``(boundary_node, branch_path_nodes)`` where ``branch_path_nodes`` is the
    chain from just below the boundary down to the leaf.
    """
    parents = parents_from_root(graph, root)
    path = [leaf]
    cur = leaf
    while True:
        parent = parents.get(cur, cur)
        if parent == cur:  # reached root
            return cur, path[::-1]
        if graph.degree(parent) > 2 or parent == root:
            return parent, path[::-1]
        path.append(parent)
        cur = parent


def subtree_length_m(graph: nx.Graph, nodes: Iterable[int]) -> float:
    """Total length of edges whose both endpoints lie in ``nodes``."""
    nodeset = set(int(n) for n in nodes)
    total = 0.0
    for u, v, data in graph.edges(data=True):
        if u in nodeset and v in nodeset:
            total += float(data["length_m"])
    return total
