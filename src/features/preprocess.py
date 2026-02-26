"""
src.features.preprocess
=======================
Preprocessing: convert directed cascade observations to undirected graphs
and validate tree structure before feature extraction.

Functions (TODO)
-----------
to_undirected  : Drop edge directions from a CascadeResult's observed graph.
validate_tree  : Assert the undirected cascade is a valid connected tree.
filter_trivial : Remove cascades too small to be informative (size < min_size).
"""
from __future__ import annotations

import networkx as nx
from src.data.cascade import CascadeResult


def to_undirected(result: CascadeResult) -> nx.Graph:
    """Return the undirected observed graph for a cascade.

    This simulates partial observability: the ML model sees *which* nodes
    were infected but not the direction of transmission.

    Parameters
    ----------
    result : CascadeResult

    Returns
    -------
    nx.Graph
        Undirected subgraph of infected nodes.
    """
    return result.observed_graph


def filter_trivial(
    results: list[CascadeResult],
    min_size: int = 3,
) -> list[CascadeResult]:
    """Remove cascades that are too small to extract meaningful features.

    Parameters
    ----------
    results : list[CascadeResult]
    min_size : int
        Minimum number of infected nodes to keep (default 3).

    Returns
    -------
    list[CascadeResult]
    """
    return [r for r in results if r.size >= min_size]


# TODO: validate_tree — checks the undirected observed graph forms a tree
