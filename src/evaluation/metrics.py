"""
src.evaluation.metrics
======================
Performance metrics for source-detection experiments.

The key metrics for this task are:

* **Top-k accuracy** — the true source appears in the top-k predicted nodes.
* **Distance to true source** — hop distance between the predicted node
  (rank-1) and the true source in the undirected observed graph. Captures
  "how far off" a wrong prediction is.
* **Mean Reciprocal Rank (MRR)** — 1/rank of the true source, averaged over
  cascades. Measures ranking quality.

Functions
---------
top_k_accuracy       : Fraction of cascades where source is in top-k.
distance_to_source   : Hop distance between rank-1 prediction and true source.
mean_reciprocal_rank : MRR over a list of ranking results.
evaluate_ranker      : Full evaluation of one ranking function.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult


def _rank_of_source(ranked_nodes: list[int], source: int) -> int:
    """Return the 1-based rank of the true source in a ranking (inf if absent)."""
    try:
        return ranked_nodes.index(source) + 1
    except ValueError:
        return len(ranked_nodes) + 1


def top_k_accuracy(
    results: list[CascadeResult],
    rankings: list[list[int]],
    k: int = 1,
) -> float:
    """Fraction of cascades where the true source appears in the top-k ranked nodes.

    Parameters
    ----------
    results : list[CascadeResult]
    rankings : list[list[int]]
        One ranked node list per cascade (most-likely first).
    k : int

    Returns
    -------
    float in [0, 1]
    """
    assert len(results) == len(rankings)
    hits = sum(
        1 for result, ranked in zip(results, rankings)
        if result.source in ranked[:k]
    )
    return hits / len(results) if results else 0.0


def distance_to_source(
    result: CascadeResult,
    ranked_nodes: list[int],
) -> int | float:
    """Hop distance in the undirected cascade graph between rank-1 prediction and true source.

    Returns 0 for a correct prediction, float('inf') if the graph is disconnected.

    Parameters
    ----------
    result : CascadeResult
    ranked_nodes : list[int]
        Ranked predictions (most-likely first).
    """
    if not ranked_nodes:
        return float("inf")
    predicted = ranked_nodes[0]
    if predicted == result.source:
        return 0
    G = result.observed_graph
    try:
        return nx.shortest_path_length(G, source=predicted, target=result.source)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def mean_reciprocal_rank(
    results: list[CascadeResult],
    rankings: list[list[int]],
) -> float:
    """Mean Reciprocal Rank (MRR) of the true source across all cascades.

    Parameters
    ----------
    results : list[CascadeResult]
    rankings : list[list[int]]

    Returns
    -------
    float in (0, 1]
    """
    assert len(results) == len(rankings)
    rr_values = [
        1.0 / _rank_of_source(ranked, result.source)
        for result, ranked in zip(results, rankings)
    ]
    return float(np.mean(rr_values)) if rr_values else 0.0


def evaluate_ranker(
    results: list[CascadeResult],
    rankings: list[list[int]],
    ks: list[int] | None = None,
) -> dict:
    """Full evaluation report for one source-ranking method.

    Parameters
    ----------
    results : list[CascadeResult]
    rankings : list[list[int]]
    ks : list[int]
        Top-k values to compute (default: [1, 3, 5]).

    Returns
    -------
    dict
        Keys: top_k (dict), mrr, mean_distance, median_distance.
    """
    ks = ks or [1, 3, 5]
    distances = [distance_to_source(r, ranked) for r, ranked in zip(results, rankings)]
    finite_distances = [d for d in distances if d != float("inf")]
    return {
        "top_k": {k: top_k_accuracy(results, rankings, k) for k in ks},
        "mrr": mean_reciprocal_rank(results, rankings),
        "mean_distance": float(np.mean(finite_distances)) if finite_distances else float("inf"),
        "median_distance": float(np.median(finite_distances)) if finite_distances else float("inf"),
        "n_cascades": len(results),
    }
