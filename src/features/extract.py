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

    # 1. Standard Centralities
    degree_c = nx.degree_centrality(G)
    closeness_c = nx.closeness_centrality(G)
    # Betweenness is expensive but acceptable for small graphs (e.g. N=20)
    betweenness_c = nx.betweenness_centrality(G, normalized=True)
    clustering = nx.clustering(G)

    # 2. Distance to Jordan Center
    # Handle disconnected graph components explicitly
    jordan_centers = []
    eccentricities = {}
    try:
        eccentricities = nx.eccentricity(G)
        jordan_centers = nx.center(G, e=eccentricities)
    except nx.NetworkXError:
        # Disconnected graph
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            try:
                sub_ecc = nx.eccentricity(sub)
                eccentricities.update(sub_ecc)
                jordan_centers.extend(nx.center(sub, e=sub_ecc))
            except Exception:
                pass

    # Precompute shortest paths for all pairs to answer queries quickly
    all_pairs_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # 3. Global Context Features
    cascade_size = G.number_of_nodes()
    # A leaf in an undirected strictly connected tree has degree 1
    # For isolated nodes, degree is 0. We'll count degree <= 1 as leaves.
    num_leaves = sum(1 for n in G.nodes() if G.degree(n) <= 1)

    features: dict[int, dict[str, float]] = {}
    
    for node in G.nodes():
        # Distance to closest Jordan Center
        if jordan_centers and node in all_pairs_lengths:
            dist_to_jc = min((all_pairs_lengths[node].get(jc, float('inf')) for jc in jordan_centers), default=float('inf'))
            if dist_to_jc == float('inf'): dist_to_jc = cascade_size  # Fallback
        else:
            dist_to_jc = cascade_size
            
        # Two-hop neighborhood count (including self)
        if node in all_pairs_lengths:
            two_hop_count = sum(1 for d in all_pairs_lengths[node].values() if d <= 2)
        else:
            two_hop_count = 1

        # Subtree depth (if we root the observed graph at 'node')
        # This is exactly the eccentricity of the node.
        subtree_depth = eccentricities.get(node, cascade_size)

        features[node] = {
            "degree": float(G.degree(node)),
            "degree_centrality": degree_c.get(node, 0.0),
            "closeness_centrality": closeness_c.get(node, 0.0),
            "betweenness_centrality": betweenness_c.get(node, 0.0),
            "eccentricity": float(eccentricities.get(node, cascade_size)),
            "jordan_center_dist": float(dist_to_jc),
            "clustering": clustering.get(node, 0.0),
            "two_hop_count": float(two_hop_count),
            "subtree_depth": float(subtree_depth),
            "cascade_size": float(cascade_size),
            "num_leaves": float(num_leaves),
        }

    return features


def build_feature_matrix(
    results: list[CascadeResult],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[str]]:
    """Stack per-node features from all cascades into X, y arrays.

    Parameters
    ----------
    results : list[CascadeResult]

    Returns
    -------
    X : np.ndarray, shape (N_samples, N_features)
    y : np.ndarray, shape (N_samples,)   — 1 if node is source, else 0
    index : list[tuple[int, int]]         — (cascade_idx, node_id) per row
    feature_names : list[str]             — ordered list of feature names
    """
    rows_X, rows_y, index = [], [], []
    feature_names = []
    
    for cascade_idx, result in enumerate(results):
        node_features = extract_node_features(result)
        for node, feats in node_features.items():
            if not feature_names:
                feature_names = list(feats.keys())
            rows_X.append([feats[k] for k in feature_names])
            rows_y.append(1 if node == result.source else 0)
            index.append((cascade_idx, node))
            
    if not rows_X:
        return np.empty((0, 0)), np.empty(0), [], []
        
    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=int), index, feature_names
