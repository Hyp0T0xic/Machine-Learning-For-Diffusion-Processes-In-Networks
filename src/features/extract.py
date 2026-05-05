"""
src.features.extract
====================
Compute per-node structural features on undirected cascade graphs.

Each cascade is represented as an undirected graph; for every node we wish
to produce a feature vector that can be used by an ML classifier to rank
nodes by their likelihood of being the true source.

Active features
---------------
- degree_zscore
- largest_comp_fraction_zscore
- nodes_within_2_hops_zscore
- jordan_center_dist
- cascade_diameter
- cascade_leaves
- endpoint_balance

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

    cascade_size = G.number_of_nodes()

    # 1. Standard Centralities
    # closeness_c = nx.closeness_centrality(G)

    # 2. Distance to Jordan Center
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

    all_pairs_lengths = dict(nx.all_pairs_shortest_path_length(G))

    if eccentricities:
        cascade_diameter = max(eccentricities.values())
    else:
        cascade_diameter = cascade_size

    # Identify peripheral nodes for robust endpoint balance
    peripheral_nodes = [n for n, e in eccentricities.items() if e == cascade_diameter] if eccentricities else []

    # Count leaf nodes (degree-1 nodes) — global cascade feature
    cascade_leaves = float(sum(1 for n in G.nodes() if G.degree(n) == 1))

    # 3. Calculate raw absolute features first
    raw_features: dict[int, dict[str, float]] = {}
    
    for node in G.nodes():
        # Distance to closest Jordan Center
        if jordan_centers and node in all_pairs_lengths:
            dist_to_jc = min((all_pairs_lengths[node].get(jc, float('inf')) for jc in jordan_centers), default=float('inf'))
            if dist_to_jc == float('inf'): dist_to_jc = cascade_size  # Fallback
        else:
            dist_to_jc = cascade_size

        # Custom Feature: Largest Component Fraction After Removal
        sub = G.subgraph(set(G.nodes()) - {node})
        largest_comp_size = max(len(c) for c in nx.connected_components(sub)) if sub.number_of_nodes() > 0 else 0
        comp_fraction = largest_comp_size / (cascade_size - 1) if cascade_size > 1 else 0.0

        # Endpoint Balance (max dist minus min dist to peripheral nodes)
        if peripheral_nodes and node in all_pairs_lengths:
            dists = [all_pairs_lengths[node].get(p, cascade_size) for p in peripheral_nodes]
            endpoint_balance = float(max(dists) - min(dists)) if dists else 0.0
        else:
            endpoint_balance = 0.0

        # Nodes within 2 hops
        if node in all_pairs_lengths:
            nodes_within_2_hops = sum(1 for d in all_pairs_lengths[node].values() if d <= 2)
        else:
            nodes_within_2_hops = 1

        raw_features[node] = {
            "degree": float(G.degree(node)),
            "largest_comp_fraction": float(comp_fraction),
            "nodes_within_2_hops": float(nodes_within_2_hops),
            "jordan_center_dist": float(dist_to_jc),
            "cascade_diameter": float(cascade_diameter),
            "cascade_leaves": cascade_leaves,
            "endpoint_balance": float(endpoint_balance),
        }

    # 4. Z-score normalization for the ranking features
    features: dict[int, dict[str, float]] = {}
    
    # We want to z-score structural features; keep geometric ones absolute
    zscore_keys = ["degree", "largest_comp_fraction", "nodes_within_2_hops"]
    
    # Calculate mean and std for each feature
    stats = {}
    for key in zscore_keys:
        vals = [raw_features[n][key] for n in G.nodes()]
        stats[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    for node in G.nodes():
        node_feats = {}
        for key in zscore_keys:
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            val = raw_features[node][key]
            # If std is 0 (all nodes have identical value), z-score is 0
            node_feats[f"{key}_zscore"] = float((val - mean) / std) if std > 1e-8 else 0.0
            
        # Keep absolute features
        node_feats["jordan_center_dist"] = raw_features[node]["jordan_center_dist"]
        node_feats["cascade_diameter"] = raw_features[node]["cascade_diameter"]
        node_feats["cascade_leaves"] = raw_features[node]["cascade_leaves"]
        node_feats["endpoint_balance"] = raw_features[node]["endpoint_balance"]
        
        features[node] = node_feats

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
    
    total_cascades = len(results)
    print(f"Extracting features for {total_cascades} cascades...")
    
    for cascade_idx, result in enumerate(results):
        if cascade_idx > 0 and cascade_idx % 500 == 0:
            print(f"  Processed {cascade_idx}/{total_cascades} cascades...")
            
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
