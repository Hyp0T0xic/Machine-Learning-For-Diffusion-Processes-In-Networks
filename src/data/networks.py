"""
src.data.networks
=================
Generate, characterise, and persist the contact networks used as substrates
for epidemic simulation.

Three network types are supported:

* **Erdős–Rényi (ER)**  — random graph G(n, p); uniform, baseline structure.
* **Barabási–Albert (BA)** — preferential attachment; scale-free with hubs.
* **Complete (K_n)**       — every node connected to every other; negative control.

Functions
---------
generate_er_network     : Create a connected ER random graph.
generate_ba_network     : Create a BA preferential-attachment graph.
generate_complete_graph : Create the complete graph K_n.
generate_all_networks   : Convenience wrapper — returns all three.
compute_network_stats   : Summary statistics (degree, diameter, clustering…).
save_networks           : Persist graphs to GraphML files.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


def generate_er_network(n: int = 100, p: float = 0.05, seed: int = 42) -> nx.Graph:
    """Generate a connected Erdős–Rényi random graph G(n, p).

    Retries with an incremented seed until the graph is connected.
    Falls back to the largest connected component after 100 attempts.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge creation probability.
    seed : int
        Base random seed for reproducibility.
    """
    current_seed = seed
    for _ in range(100):
        G = nx.erdos_renyi_graph(n, p, seed=current_seed)
        if nx.is_connected(G):
            G.graph.update({"name": "Erdos-Renyi", "model": "ER", "p": p, "seed": current_seed})
            return G
        current_seed += 1
    # Fallback — largest connected component
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G.graph.update({"name": "Erdos-Renyi", "model": "ER", "p": p, "seed": seed,
                    "note": "largest_connected_component"})
    return G


def generate_ba_network(n: int = 100, m: int = 3, seed: int = 42) -> nx.Graph:
    """Generate a Barabási–Albert preferential-attachment graph.

    The BA model always produces a connected graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges each new node attaches to.
    seed : int
        Random seed.
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    G.graph.update({"name": "Barabasi-Albert", "model": "BA", "m": m, "seed": seed})
    return G


def generate_complete_graph(n: int = 100) -> nx.Graph:
    """Generate the complete graph K_n.

    Every node is connected to every other node. Used as a negative control:
    maximal structural symmetry makes source detection near-impossible.

    Parameters
    ----------
    n : int
        Number of nodes.
    """
    G = nx.complete_graph(n)
    G.graph.update({"name": "Complete", "model": "K_n"})
    return G


def generate_all_networks(
    n: int = 100,
    er_p: float = 0.05,
    ba_m: int = 3,
    seed: int = 42,
) -> dict[str, nx.Graph]:
    """Generate all three contact networks with a single call.

    Returns
    -------
    dict[str, nx.Graph]
        Keys: ``"ER"``, ``"BA"``, ``"Complete"``.
    """
    return {
        "ER": generate_er_network(n, er_p, seed),
        "BA": generate_ba_network(n, ba_m, seed),
        "Complete": generate_complete_graph(n),
    }


def compute_network_stats(G: nx.Graph) -> dict:
    """Compute summary statistics for an undirected graph.

    Returns
    -------
    dict
        Keys: name, nodes, edges, avg_degree, diameter, clustering, density.
    """
    degrees = [d for _, d in G.degree()]
    return {
        "name": G.graph.get("name", "Unknown"),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": round(float(np.mean(degrees)), 2),
        "diameter": nx.diameter(G) if nx.is_connected(G) else float("inf"),
        "clustering": round(nx.average_clustering(G), 4),
        "density": round(nx.density(G), 4),
    }


def save_networks(
    networks: dict[str, nx.Graph],
    output_dir: str | Path = "data/networks",
) -> list[Path]:
    """Persist networks as GraphML files.

    Parameters
    ----------
    networks : dict[str, nx.Graph]
        Mapping of short name → graph.
    output_dir : str or Path
        Target directory (created if absent).

    Returns
    -------
    list[Path]
        Paths to the written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, G in networks.items():
        path = output_dir / f"{key.lower()}_network.graphml"
        nx.write_graphml(G, str(path))
        saved.append(path)
    return saved
