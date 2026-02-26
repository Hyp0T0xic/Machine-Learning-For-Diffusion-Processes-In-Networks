"""
src.baselines.centrality
========================
Classical graph-theory baselines for source detection.

These methods predict the source without any ML training — they rank nodes
by a graph-theoretic heuristic on the undirected observed cascade.

Methods
-------
jordan_center     : Predict the Jordan center (minimises eccentricity).
                    Known to perform well for tree-like cascades [Shah 2011].
degree_rank       : Predict the highest-degree node.
closeness_rank    : Predict the node with highest closeness centrality.
betweenness_rank  : Predict the node with highest betweenness centrality.
predict_all       : Run all baselines and return ranked node lists.
"""
from __future__ import annotations

import networkx as nx

from src.data.cascade import CascadeResult


def jordan_center(result: CascadeResult) -> list[int]:
    """Return nodes sorted by Jordan centrality (ascending eccentricity).

    The node(s) with minimum eccentricity are predicted as the source.
    Ties are broken arbitrarily.

    Parameters
    ----------
    result : CascadeResult

    Returns
    -------
    list[int]
        All nodes sorted by predicted likelihood of being the source
        (most likely first).
    """
    G = result.observed_graph
    if G.number_of_nodes() == 0:
        return []
    try:
        ecc = nx.eccentricity(G)
        return sorted(G.nodes(), key=lambda n: ecc[n])
    except nx.NetworkXError:
        # Graph may not be connected
        components = list(nx.connected_components(G))
        ranked = []
        for comp in sorted(components, key=len, reverse=True):
            sub = G.subgraph(comp)
            ecc = nx.eccentricity(sub)
            ranked.extend(sorted(comp, key=lambda n: ecc[n]))
        return ranked


def degree_rank(result: CascadeResult) -> list[int]:
    """Rank nodes by degree (highest first)."""
    G = result.observed_graph
    return sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)


def closeness_rank(result: CascadeResult) -> list[int]:
    """Rank nodes by closeness centrality (highest first)."""
    G = result.observed_graph
    c = nx.closeness_centrality(G)
    return sorted(G.nodes(), key=lambda n: c[n], reverse=True)


def betweenness_rank(result: CascadeResult) -> list[int]:
    """Rank nodes by betweenness centrality (highest first)."""
    G = result.observed_graph
    b = nx.betweenness_centrality(G, normalized=True)
    return sorted(G.nodes(), key=lambda n: b[n], reverse=True)


def predict_all(result: CascadeResult) -> dict[str, list[int]]:
    """Run all baselines and return ranked node lists.

    Parameters
    ----------
    result : CascadeResult

    Returns
    -------
    dict[str, list[int]]
        Keys: "jordan", "degree", "closeness", "betweenness".
    """
    return {
        "jordan": jordan_center(result),
        "degree": degree_rank(result),
        "closeness": closeness_rank(result),
        "betweenness": betweenness_rank(result),
    }
