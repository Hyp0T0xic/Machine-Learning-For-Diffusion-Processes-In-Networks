"""
Network generation module for diffusion cascade source detection.

Provides functions to generate Erdős–Rényi, Barabási–Albert, and Complete
graphs, compute statistics, and save to GraphML format.
"""

import os
from pathlib import Path

import networkx as nx
import numpy as np


def generate_er_network(n: int = 100, p: float = 0.05, seed: int = 42) -> nx.Graph:
    """Generate a connected Erdős–Rényi random graph G(n, p).

    Retries with incremented seed until the graph is connected.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge creation probability.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    nx.Graph
        A connected undirected ER graph.
    """
    current_seed = seed
    max_attempts = 100

    for attempt in range(max_attempts):
        G = nx.erdos_renyi_graph(n, p, seed=current_seed)
        if nx.is_connected(G):
            G.graph["name"] = "Erdos-Renyi"
            G.graph["model"] = "ER"
            G.graph["p"] = p
            G.graph["seed"] = current_seed
            return G
        current_seed += 1

    # Fallback: take the largest connected component
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G.graph["name"] = "Erdos-Renyi"
    G.graph["model"] = "ER"
    G.graph["p"] = p
    G.graph["seed"] = seed
    G.graph["note"] = "largest_connected_component"
    return G


def generate_ba_network(n: int = 100, m: int = 3, seed: int = 42) -> nx.Graph:
    """Generate a Barabási–Albert preferential-attachment graph.

    The BA model always produces a connected graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges each new node attaches to existing nodes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        A connected undirected BA graph.
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    G.graph["name"] = "Barabasi-Albert"
    G.graph["model"] = "BA"
    G.graph["m"] = m
    G.graph["seed"] = seed
    return G


def generate_complete_graph(n: int = 100) -> nx.Graph:
    """Generate a complete graph K_n.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    nx.Graph
        The complete undirected graph on n nodes.
    """
    G = nx.complete_graph(n)
    G.graph["name"] = "Complete"
    G.graph["model"] = "K_n"
    return G


def generate_all_networks(
    n: int = 100,
    er_p: float = 0.05,
    ba_m: int = 3,
    seed: int = 42,
) -> dict[str, nx.Graph]:
    """Generate all three network types.

    Parameters
    ----------
    n : int
        Number of nodes for every network.
    er_p : float
        Edge probability for the Erdős–Rényi model.
    ba_m : int
        Attachment parameter for the Barabási–Albert model.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, nx.Graph]
        Keys are ``"ER"``, ``"BA"``, ``"Complete"``.
    """
    return {
        "ER": generate_er_network(n, er_p, seed),
        "BA": generate_ba_network(n, ba_m, seed),
        "Complete": generate_complete_graph(n),
    }


def compute_network_stats(G: nx.Graph) -> dict:
    """Compute summary statistics for a network.

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.

    Returns
    -------
    dict
        Dictionary with keys: name, nodes, edges, avg_degree, diameter,
        clustering, density.
    """
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)

    # Diameter is finite only for connected graphs
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = float("inf")

    return {
        "name": G.graph.get("name", "Unknown"),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": round(avg_degree, 2),
        "diameter": diameter,
        "clustering": round(nx.average_clustering(G), 4),
        "density": round(nx.density(G), 4),
    }


def save_networks(
    networks: dict[str, nx.Graph],
    output_dir: str | Path = "data/networks",
) -> list[Path]:
    """Save networks as GraphML files.

    Parameters
    ----------
    networks : dict[str, nx.Graph]
        Mapping of short name → graph.
    output_dir : str or Path
        Directory to write files into (created if absent).

    Returns
    -------
    list[Path]
        Paths to the saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for key, G in networks.items():
        path = output_dir / f"{key.lower()}_network.graphml"
        nx.write_graphml(G, str(path))
        saved.append(path)

    return saved
