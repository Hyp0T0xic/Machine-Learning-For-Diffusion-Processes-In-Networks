"""cascade dataclass + independent-cascade model + r0→param mapping"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# data structure


@dataclass
class CascadeResult:
    """Stores the outcome of a single diffusion run.

    Attributes
    ----------
    source : int
        True patient-zero node (ground-truth label).
    model_name : str
        Always "IC" in this project.
    params : dict
        Transmission parameters used (e.g. {"p": 0.3}).
    infection_times : dict[int, int]
        Mapping node → timestep at which it was infected.
    cascade_edges : list[tuple[int, int]]
        Directed (infector, infected) transmission edges.
    network_name : str
        Label of the contact network (e.g. "ER", "BA", "Complete").
    """

    source: int
    model_name: str
    params: dict
    infection_times: dict[int, int]
    cascade_edges: list[tuple[int, int]]
    network_name: str = ""

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of infected nodes."""
        return len(self.infection_times)

    @property
    def depth(self) -> int:
        """Maximum hop distance from the source in the infection tree."""
        if not self.cascade_edges:
            return 0
        tree = nx.DiGraph(self.cascade_edges)
        if self.source not in tree:
            return 0
        lengths = nx.single_source_shortest_path_length(tree, self.source)
        return max(lengths.values()) if lengths else 0

    @property
    def observed_graph(self) -> nx.Graph:
        """Undirected subgraph of infected nodes — the ML model's input.

        Edge directions are dropped to simulate partial observability.
        """
        G = nx.Graph()
        # Sort nodes to destroy insertion order (which leaks the true source)
        nodes = sorted(self.infection_times.keys())
        G.add_nodes_from(nodes)
        for u, v in self.cascade_edges:
            G.add_edge(u, v)
        return G

    @property
    def infection_tree(self) -> nx.DiGraph:
        """Directed infection tree (full ground-truth, not visible to ML)."""
        T = nx.DiGraph()
        T.add_nodes_from(self.infection_times.keys())
        T.add_edges_from(self.cascade_edges)
        return T

    def actual_r0(self) -> float:
        """Empirical R₀: mean secondary infections per spreading node."""
        if self.size <= 1:
            return 0.0
        tree = self.infection_tree
        out_degrees = [tree.out_degree(n) for n in tree.nodes()]
        spreading = [d for d in out_degrees if d > 0]
        return float(np.mean(spreading)) if spreading else 0.0

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "source": int(self.source),
            "model_name": self.model_name,
            "params": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.params.items()
            },
            "network_name": self.network_name,
            "infection_times": {str(k): int(v) for k, v in self.infection_times.items()},
            "cascade_edges": [[int(u), int(v)] for u, v in self.cascade_edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CascadeResult":
        """Deserialise from a dictionary (produced by ``to_dict``)."""
        return cls(
            source=int(d["source"]),
            model_name=d["model_name"],
            params=d["params"],
            network_name=d.get("network_name", ""),
            infection_times={int(k): int(v) for k, v in d["infection_times"].items()},
            cascade_edges=[(int(u), int(v)) for u, v in d["cascade_edges"]],
        )


# r0 → parameter mapping


def r0_to_params(
    r0: float,
    avg_degree: float,
    model: str = "IC",
    **_kwargs,
) -> dict:
    """Map a target R₀ to IC transmission probability via p = R₀ / ⟨k⟩.

    Parameters
    ----------
    r0 : float
        Target basic reproduction number (e.g. 0.5, 1.0, 2.0, 3.0).
    avg_degree : float
        Average degree ⟨k⟩ of the contact network.
    model : str
        Kept for backwards compatibility; must be ``"IC"`` (case-insensitive).

    Returns
    -------
    dict
        ``{"p": min(r0 / avg_degree, 1.0)}``.
    """
    if model.upper() != "IC":
        raise ValueError(
            f"Unknown model: {model!r}. Only 'IC' is supported."
        )
    return {"p": min(r0 / avg_degree, 1.0)}


# model implementations


class IndependentCascade:
    """Independent Cascade (IC) model.

    Each infected node gets **one chance** to infect each susceptible
    neighbour. The attempt succeeds with probability *p*; the edge is then
    burned regardless of outcome.

    Best models: information/rumour spreading.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def run(self, G: nx.Graph, source: int, seed: int | None = None, max_size: int | None = None) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []
        newly_infected = [source]
        t = 0
        while newly_infected:
            t += 1
            next_wave = []
            for node in newly_infected:
                for neighbor in G.neighbors(node):
                    if neighbor not in infection_times and rng.random() < self.p:
                        infection_times[neighbor] = t
                        cascade_edges.append((node, neighbor))
                        next_wave.append(neighbor)
                        if max_size is not None and len(infection_times) >= max_size:
                            break
                if max_size is not None and len(infection_times) >= max_size:
                    break
            if max_size is not None and len(infection_times) >= max_size:
                break
            newly_infected = next_wave
        return CascadeResult(
            source=source, model_name="IC", params={"p": self.p},
            infection_times=infection_times, cascade_edges=cascade_edges,
        )


# factory


def create_model(name: str = "IC", **params) -> IndependentCascade:
    """Instantiate a diffusion model by name. Only ``"IC"`` is supported."""
    if name.upper() != "IC":
        raise ValueError(f"Unknown model: {name!r}. Only 'IC' is supported.")
    return IndependentCascade(**params)
