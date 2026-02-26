"""
src.features.extract
====================
Compute per-node structural features on undirected cascade graphs.

Each cascade is represented as an undirected graph; for every node we wish
to produce a feature vector that can be used by an ML classifier to rank
nodes by their likelihood of being the true source.

Planned features
----------------
Centrality-based:
    degree_centrality, betweenness_centrality, closeness_centrality,
    eigenvector_centrality, page_rank

Distance-based:
    eccentricity, distance_to_centroid, jordan_center_distance

Structural:
    local_clustering, neighbour_avg_degree (assortativity proxy),
    subtree_size (in a BFS tree rooted at the node)

Functions (TODO)
----------------
extract_node_features : Return a feature dict for every node in a cascade.
build_feature_matrix  : Stack features into a numpy array X and label vector y.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult


def extract_node_features(result: CascadeResult) -> dict[int, dict[str, float]]:
    """Compute structural node features on the undirected cascade graph.

    Parameters
    ----------
    result : CascadeResult

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping node_id → {feature_name: value}.
    """
    G = result.observed_graph
    if G.number_of_nodes() == 0:
        return {}

    degree_c = nx.degree_centrality(G)
    closeness_c = nx.closeness_centrality(G)
    # Betweenness is expensive — skip for large graphs in early experiments
    betweenness_c = nx.betweenness_centrality(G, normalized=True)

    features: dict[int, dict[str, float]] = {}
    for node in G.nodes():
        features[node] = {
            "degree_centrality": degree_c.get(node, 0.0),
            "closeness_centrality": closeness_c.get(node, 0.0),
            "betweenness_centrality": betweenness_c.get(node, 0.0),
            "degree": float(G.degree(node)),
        }

    # TODO: eccentricity, jordan_center_distance, subtree_size, page_rank
    return features


def build_feature_matrix(
    results: list[CascadeResult],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Stack per-node features from all cascades into X, y arrays.

    Parameters
    ----------
    results : list[CascadeResult]

    Returns
    -------
    X : np.ndarray, shape (N_samples, N_features)
    y : np.ndarray, shape (N_samples,)   — 1 if node is source, else 0
    index : list[tuple[int, int]]         — (cascade_idx, node_id) per row
    """
    rows_X, rows_y, index = [], [], []
    for cascade_idx, result in enumerate(results):
        node_features = extract_node_features(result)
        for node, feats in node_features.items():
            rows_X.append(list(feats.values()))
            rows_y.append(1 if node == result.source else 0)
            index.append((cascade_idx, node))
    if not rows_X:
        return np.empty((0, 0)), np.empty(0), []
    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=int), index
