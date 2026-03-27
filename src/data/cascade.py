"""
src.data.cascade
================
Core data structures and epidemic model implementations.

Classes
-------
CascadeResult       : Stores the outcome of one diffusion run (source,
                      infection times, cascade edges, derived properties).
IndependentCascade  : IC model — one-shot transmission with probability *p*.
SIModel             : SI model — persistent transmission, no recovery.
SIRModel            : SIR model — transmission + recovery with probability γ.

Functions
---------
r0_to_params : Map target R₀ to model-specific transmission parameters.
create_model : Factory — return a model instance by name string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# ── Data structure ───────────────────────────────────────────────────────────


@dataclass
class CascadeResult:
    """Stores the outcome of a single diffusion run.

    Attributes
    ----------
    source : int
        True patient-zero node (ground-truth label).
    model_name : str
        One of "IC", "SI", "SIR".
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
        G.add_nodes_from(self.infection_times.keys())
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


# ── R₀ → parameter mapping ─────────────────────────────────────────────────


def r0_to_params(
    r0: float,
    avg_degree: float,
    model: str = "IC",
    gamma: float = 0.2,
) -> dict:
    """Map a target R₀ to model-specific transmission parameters.

    Parameters
    ----------
    r0 : float
        Target basic reproduction number (e.g. 0.5, 1.0, 2.0, 3.0).
    avg_degree : float
        Average degree ⟨k⟩ of the contact network.
    model : str
        One of ``"IC"``, ``"SI"``, ``"SIR"`` (case-insensitive).
    gamma : float
        Recovery probability — only used by SIR.

    Returns
    -------
    dict
        Parameter dict suitable for the corresponding model constructor.

    Notes
    -----
    Mappings:
        IC  : p    = R₀ / ⟨k⟩
        SI  : β    = R₀ / ⟨k⟩
        SIR : β    = R₀ · γ / ⟨k⟩
    """
    model = model.upper()
    if model == "IC":
        return {"p": min(r0 / avg_degree, 1.0)}
    elif model == "SI":
        return {"beta": min(r0 / avg_degree, 1.0)}
    elif model == "SIR":
        return {"beta": min(r0 * gamma / avg_degree, 1.0), "gamma": gamma}
    else:
        raise ValueError(f"Unknown model: {model!r}. Choose from IC, SI, SIR.")


# ── Model implementations ──────────────────────────────────────────────────


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


class SIModel:
    """Susceptible-Infected (SI) model.

    Infected nodes remain infectious forever and attempt transmission every
    timestep with probability *β*. Terminates when no new infections occur.

    Models: worst-case spread with no containment.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def run(
        self, G: nx.Graph, source: int, seed: int | None = None, max_steps: int = 200
    ) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []
        infected = {source}
        for t in range(1, max_steps + 1):
            new_infections: dict[int, int] = {}
            for node in infected:
                for neighbor in G.neighbors(node):
                    if neighbor not in infection_times and neighbor not in new_infections:
                        if rng.random() < self.beta:
                            new_infections[neighbor] = node
            if not new_infections:
                break
            for neighbor, infector in new_infections.items():
                infection_times[neighbor] = t
                cascade_edges.append((infector, neighbor))
                infected.add(neighbor)
        return CascadeResult(
            source=source, model_name="SI", params={"beta": self.beta},
            infection_times=infection_times, cascade_edges=cascade_edges,
        )


class SIRModel:
    """Susceptible-Infected-Recovered (SIR) model.

    Infected nodes transmit with probability *β* and recover (becoming
    immune) with probability *γ* each timestep.

    Best models: epidemics (COVID-19, influenza).
    """

    def __init__(self, beta: float = 0.1, gamma: float = 0.2):
        self.beta = beta
        self.gamma = gamma

    def run(
        self, G: nx.Graph, source: int, seed: int | None = None, max_steps: int = 200
    ) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []
        infected = {source}
        recovered: set[int] = set()
        for t in range(1, max_steps + 1):
            # Transmission
            new_infections: dict[int, int] = {}
            for node in infected:
                for neighbor in G.neighbors(node):
                    if (
                        neighbor not in infection_times
                        and neighbor not in recovered
                        and neighbor not in new_infections
                        and rng.random() < self.beta
                    ):
                        new_infections[neighbor] = node
            # Recovery
            newly_recovered = {node for node in infected if rng.random() < self.gamma}
            recovered |= newly_recovered
            infected -= newly_recovered
            for neighbor, infector in new_infections.items():
                infection_times[neighbor] = t
                cascade_edges.append((infector, neighbor))
                infected.add(neighbor)
            if not infected:
                break
        return CascadeResult(
            source=source, model_name="SIR",
            params={"beta": self.beta, "gamma": self.gamma},
            infection_times=infection_times, cascade_edges=cascade_edges,
        )


# ── Factory ─────────────────────────────────────────────────────────────────


def create_model(name: str, **params) -> IndependentCascade | SIModel | SIRModel:
    """Instantiate a diffusion model by name.

    Parameters
    ----------
    name : str
        One of ``"IC"``, ``"SI"``, ``"SIR"`` (case-insensitive).
    **params
        Forwarded to the model constructor.
    """
    name = name.upper()
    registry = {"IC": IndependentCascade, "SI": SIModel, "SIR": SIRModel}
    if name not in registry:
        raise ValueError(f"Unknown model: {name!r}. Choose from {list(registry)}.")
    return registry[name](**params)
