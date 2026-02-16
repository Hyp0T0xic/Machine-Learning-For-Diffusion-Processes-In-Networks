"""
Simulation orchestrator for running diffusion cascades across networks.

Provides source selection, experiment runner, cascade statistics, and
JSON serialisation for reproducible experiments.
"""

from __future__ import annotations

import json
import random as stdlib_random
from pathlib import Path

import networkx as nx
import numpy as np

from src.models import CascadeResult, create_model, r0_to_params


def select_sources(
    G: nx.Graph,
    n_sources: int = 5,
    seed: int = 42,
) -> list[int]:
    """Select random source nodes from the graph.

    Prefers nodes with degree ≥ 1 (connected) to avoid trivial cascades.

    Parameters
    ----------
    G : nx.Graph
        Contact network.
    n_sources : int
        Number of source nodes to select.
    seed : int
        Random seed.

    Returns
    -------
    list[int]
        Node IDs of selected sources.
    """
    rng = stdlib_random.Random(seed)
    candidates = [n for n, d in G.degree() if d >= 1]
    if not candidates:
        candidates = list(G.nodes())
    n_sources = min(n_sources, len(candidates))
    return rng.sample(candidates, n_sources)


def run_experiment(
    G: nx.Graph,
    model_name: str,
    model_params: dict,
    sources: list[int],
    n_runs: int = 1,
    seed: int = 42,
    network_name: str = "",
) -> list[CascadeResult]:
    """Run a diffusion model from each source node.

    Parameters
    ----------
    G : nx.Graph
        Contact network.
    model_name : str
        Model identifier (``"IC"``, ``"SI"``, ``"SIR"``).
    model_params : dict
        Parameters forwarded to the model constructor.
    sources : list[int]
        Source node IDs.
    n_runs : int
        Number of stochastic runs per source.
    seed : int
        Base random seed (incremented per run).
    network_name : str
        Label stored in each CascadeResult for downstream use.

    Returns
    -------
    list[CascadeResult]
    """
    model = create_model(model_name, **model_params)
    results: list[CascadeResult] = []
    run_seed = seed

    for src in sources:
        for run_idx in range(n_runs):
            result = model.run(G, source=src, seed=run_seed)
            result.network_name = network_name
            results.append(result)
            run_seed += 1

    return results


def compute_cascade_stats(result: CascadeResult, G: nx.Graph | None = None) -> dict:
    """Compute summary statistics for a single cascade.

    Parameters
    ----------
    result : CascadeResult
        Outcome of a diffusion run.
    G : nx.Graph, optional
        Original contact network (for coverage calculation).

    Returns
    -------
    dict
        Keys: size, depth, actual_r0, coverage_pct, avg_path_from_source.
    """
    size = result.size
    depth = result.depth
    actual_r0 = result.actual_r0()

    coverage_pct = 0.0
    if G is not None:
        coverage_pct = round(100.0 * size / G.number_of_nodes(), 2)

    # Average shortest path from source within the infection tree
    avg_path = 0.0
    if result.cascade_edges:
        tree = result.infection_tree
        try:
            lengths = nx.single_source_shortest_path_length(tree, result.source)
            if len(lengths) > 1:
                avg_path = round(
                    np.mean([v for k, v in lengths.items() if k != result.source]),
                    2,
                )
        except nx.NetworkXError:
            pass

    return {
        "source": result.source,
        "model": result.model_name,
        "network": result.network_name,
        "size": size,
        "depth": depth,
        "actual_r0": round(actual_r0, 3),
        "coverage_pct": coverage_pct,
        "avg_path_from_source": avg_path,
    }


def save_cascades(
    results: list[CascadeResult],
    output_dir: str | Path = "data/cascades",
    filename: str = "cascades.json",
) -> Path:
    """Serialize cascade results to JSON.

    Parameters
    ----------
    results : list[CascadeResult]
        Cascade outcomes to save.
    output_dir : str or Path
        Output directory (created if absent).
    filename : str
        Name of the JSON file.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    data = [r.to_dict() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def load_cascades(path: str | Path) -> list[CascadeResult]:
    """Load cascade results from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    list[CascadeResult]
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CascadeResult.from_dict(d) for d in data]
