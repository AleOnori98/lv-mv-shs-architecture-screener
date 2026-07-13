"""Enumerate leaf-side *terminal subtrees* of a radial pole tree.

A terminal subtree detaches from the network at a **branching node** — a node that, once
the tree is rooted at the source, has more than one child (a true fork). For each such
fork ``B`` and each child ``c``, the candidate subtree is every pole at or below ``c``
(towards the leaves); the single edge ``(B, c)`` is the *entry edge* that ties the whole
subtree to the rest of the grid. Cutting the subtree off frees its internal lines, its
entry edge, its dedicated poles and its customers' service drops.

Subtrees nest (a remote hamlet contains its individual stubs); the economic SHS screen
resolves the overlap greedily, so generation itself stays free of thresholds. Trunks and
pass-through chains (nodes with a single child) never spawn a candidate, so the un-cuttable
backbone — and, for a single zone, the whole zone — is excluded by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from . import graph_utils as gu


@dataclass
class Subtree:
    poles: List[int]                 # all poles in the subtree (branch_root down to its leaves)
    branch_root: int                 # child node c where the subtree begins
    boundary_node: int               # fork node B just upstream of branch_root
    entry_edge: Tuple[int, int]      # (B, c): the line connecting the subtree to the grid
    entry_edge_id: Optional[str]     # line_id of the entry edge
    internal_edge_ids: List[str]     # line_ids of edges fully inside the subtree


def terminal_subtrees(graph: nx.Graph, root: int) -> List[Subtree]:
    """Leaf-side subtrees that detach at a fork node (a node with >= 2 children).

    Deterministic; runs one rooted traversal and then collects descendants iteratively
    (no per-candidate re-rooting). Returns an empty list for an empty graph or a pure
    chain with no forks.
    """
    if graph.number_of_nodes() == 0 or root not in graph:
        return []

    parents = gu.parents_from_root(graph, root)
    children: Dict[int, List[int]] = {}
    for child, parent in parents.items():
        if child != parent:
            children.setdefault(parent, []).append(child)
    # line_id of the edge linking each node to its parent (its uplink towards the source).
    parent_edge_id = {child: str(graph[child][parent]["line_id"]) for child, parent in parents.items() if child != parent}

    def descendants(node: int) -> List[int]:
        out, stack = [], [node]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(children.get(cur, []))
        return out

    subtrees: List[Subtree] = []
    for boundary, kids in children.items():
        if len(kids) < 2:  # only true forks spawn candidates; trunk / pass-through nodes do not
            continue
        for c in sorted(kids):
            poles = descendants(c)
            internal_ids = sorted({parent_edge_id[p] for p in poles if p != c})
            subtrees.append(Subtree(
                poles=sorted(int(p) for p in poles),
                branch_root=int(c),
                boundary_node=int(boundary),
                entry_edge=(int(boundary), int(c)),
                entry_edge_id=parent_edge_id.get(c),
                internal_edge_ids=internal_ids,
            ))
    return subtrees
