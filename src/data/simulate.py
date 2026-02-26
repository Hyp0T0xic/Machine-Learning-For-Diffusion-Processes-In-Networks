"""
src.data.simulate
=================
Experiment runner for batch cascade simulation.

Provides helpers to select source nodes, run a diffusion model from multiple
sources across a range of parameters, compute per-cascade statistics, and
serialise/deserialise results to JSON.

Functions
---------
select_sources       : Sample seed nodes from a contact network.
run_experiment       : Run a model from each source; return CascadeResult list.
compute_cascade_stats: Summarise one cascade (size, depth, R₀, coverage…).
save_cascades        : Serialise a list of CascadeResults to JSON.
load_cascades        : Deserialise cascades from a JSON file.
"""

from __future__ import annotations

import json
import random as stdlib_random
from pathlib import Path

import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult, create_model, r0_to_params


def select_sources(
    G: nx.Graph,
    n_sources: int = 5,
    seed: int = 42,
) -> list[int]:
    """Sample source nodes from a contact network.

    Prefers connected nodes (degree ≥ 1) to avoid trivial zero-size cascades.

    Parameters
    ----------
    G : nx.Graph
    n_sources : int
    seed : int

    Returns
    -------
    list[int]
        Node IDs of the selected sources.
    """
    rng = stdlib_random.Random(seed)
    candidates = [n for n, d in G.degree() if d >= 1] or list(G.nodes())
    return rng.sample(candidates, min(n_sources, len(candidates)))


def run_experiment(
    G: nx.Graph,
    model_name: str,
    model_params: dict,
    sources: list[int],
    n_runs: int = 1,
    seed: int = 42,
    network_name: str = "",
) -> list[CascadeResult]:
    """Run a diffusion model from each source and return all results.

    Parameters
    ----------
    G : nx.Graph
        Contact network.
    model_name : str
        ``"IC"``, ``"SI"``, or ``"SIR"``.
    model_params : dict
        Kwargs forwarded to the model constructor.
    sources : list[int]
        Source node IDs.
    n_runs : int
        Stochastic repetitions per source.
    seed : int
        Base random seed (incremented per run for independence).
    network_name : str
        Stored in each result for downstream filtering.
    """
    model = create_model(model_name, **model_params)
    results: list[CascadeResult] = []
    run_seed = seed
    for src in sources:
        for _ in range(n_runs):
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
    G : nx.Graph, optional
        Contact network — used to compute coverage percentage.

    Returns
    -------
    dict
        Keys: source, model, network, size, depth, actual_r0,
        coverage_pct, avg_path_from_source.
    """
    size = result.size
    depth = result.depth
    actual_r0 = result.actual_r0()
    coverage_pct = round(100.0 * size / G.number_of_nodes(), 2) if G else 0.0
    avg_path = 0.0
    if result.cascade_edges:
        tree = result.infection_tree
        try:
            lengths = nx.single_source_shortest_path_length(tree, result.source)
            non_source = [v for k, v in lengths.items() if k != result.source]
            if non_source:
                avg_path = round(float(np.mean(non_source)), 2)
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
    output_dir: str | Path = "data/raw",
    filename: str = "cascades.json",
) -> Path:
    """Serialise cascade results to a JSON file.

    Parameters
    ----------
    results : list[CascadeResult]
    output_dir : str or Path
    filename : str

    Returns
    -------
    Path
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    return path


def load_cascades(path: str | Path) -> list[CascadeResult]:
    """Deserialise cascade results from a JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    list[CascadeResult]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CascadeResult.from_dict(d) for d in data]
