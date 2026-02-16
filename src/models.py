"""
Diffusion models for cascade simulation on contact networks.

Implements three epidemic/information-spreading models:
- Independent Cascade (IC)
- Susceptible-Infected (SI)
- Susceptible-Infected-Recovered (SIR)

Each model exposes a `run(G, source, seed)` → CascadeResult interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class CascadeResult:
    """Stores the outcome of a single diffusion run."""

    source: int
    model_name: str
    params: dict
    infection_times: dict[int, int]          # node → timestep
    cascade_edges: list[tuple[int, int]]     # (parent, child) directed
    network_name: str = ""

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of infected nodes."""
        return len(self.infection_times)

    @property
    def depth(self) -> int:
        """Maximum generation distance from the source."""
        if not self.cascade_edges:
            return 0
        tree = nx.DiGraph(self.cascade_edges)
        if self.source not in tree:
            return 0
        lengths = nx.single_source_shortest_path_length(tree, self.source)
        return max(lengths.values()) if lengths else 0

    @property
    def observed_graph(self) -> nx.Graph:
        """Undirected subgraph induced by infected nodes (ML input)."""
        G = nx.Graph()
        G.add_nodes_from(self.infection_times.keys())
        for u, v in self.cascade_edges:
            G.add_edge(u, v)
        return G

    @property
    def infection_tree(self) -> nx.DiGraph:
        """Directed tree of the infection cascade."""
        T = nx.DiGraph()
        T.add_nodes_from(self.infection_times.keys())
        T.add_edges_from(self.cascade_edges)
        return T

    def actual_r0(self) -> float:
        """Average number of secondary infections per infected node."""
        if self.size <= 1:
            return 0.0
        tree = self.infection_tree
        out_degrees = [tree.out_degree(n) for n in tree.nodes()]
        # Exclude leaves from average (they had no chance or didn't spread)
        # Standard epidemiological R0 = avg secondary cases from all infectors
        spreading_nodes = [d for d in out_degrees if d > 0]
        if not spreading_nodes:
            return 0.0
        return float(np.mean(spreading_nodes))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "source": int(self.source),
            "model_name": self.model_name,
            "params": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in self.params.items()},
            "network_name": self.network_name,
            "infection_times": {str(k): int(v) for k, v in self.infection_times.items()},
            "cascade_edges": [[int(u), int(v)] for u, v in self.cascade_edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CascadeResult":
        """Deserialize from a dictionary."""
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
        Target basic reproduction number.
    avg_degree : float
        Average degree ⟨k⟩ of the contact network.
    model : str
        One of ``"IC"``, ``"SI"``, ``"SIR"``.
    gamma : float
        Recovery probability (only used by SIR).

    Returns
    -------
    dict
        Parameter dictionary suitable for the corresponding model class.
    """
    model = model.upper()
    if model == "IC":
        p = min(r0 / avg_degree, 1.0)
        return {"p": p}
    elif model == "SI":
        beta = min(r0 / avg_degree, 1.0)
        return {"beta": beta}
    elif model == "SIR":
        beta = min(r0 * gamma / avg_degree, 1.0)
        return {"beta": beta, "gamma": gamma}
    else:
        raise ValueError(f"Unknown model: {model}")


# ── Model implementations ──────────────────────────────────────────────────


class IndependentCascade:
    """Independent Cascade model.

    Each infected node gets ONE chance to infect each susceptible neighbor.
    The attempt succeeds with probability *p*. Once attempted, the edge is
    effectively "burned" regardless of outcome.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def run(
        self,
        G: nx.Graph,
        source: int,
        seed: int | None = None,
    ) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []

        # Nodes that were newly infected in the previous wave
        newly_infected = [source]
        t = 0

        while newly_infected:
            t += 1
            next_wave = []
            for node in newly_infected:
                for neighbor in G.neighbors(node):
                    if neighbor not in infection_times:
                        if rng.random() < self.p:
                            infection_times[neighbor] = t
                            cascade_edges.append((node, neighbor))
                            next_wave.append(neighbor)
            newly_infected = next_wave

        return CascadeResult(
            source=source,
            model_name="IC",
            params={"p": self.p},
            infection_times=infection_times,
            cascade_edges=cascade_edges,
        )


class SIModel:
    """Susceptible-Infected model.

    Infected nodes remain infectious forever. At each discrete timestep,
    every infected node independently attempts to transmit to each
    susceptible neighbor with probability β. The process terminates when
    no new infections occur in a timestep.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def run(
        self,
        G: nx.Graph,
        source: int,
        seed: int | None = None,
        max_steps: int = 200,
    ) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []
        infected = {source}

        for t in range(1, max_steps + 1):
            new_infections: dict[int, int] = {}  # neighbor → infector
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
            source=source,
            model_name="SI",
            params={"beta": self.beta},
            infection_times=infection_times,
            cascade_edges=cascade_edges,
        )


class SIRModel:
    """Susceptible-Infected-Recovered model.

    Infected nodes can recover (transition to Recovered/immune) with
    probability γ each timestep. Recovered nodes cannot spread or be
    reinfected.
    """

    def __init__(self, beta: float = 0.1, gamma: float = 0.2):
        self.beta = beta
        self.gamma = gamma

    def run(
        self,
        G: nx.Graph,
        source: int,
        seed: int | None = None,
        max_steps: int = 200,
    ) -> CascadeResult:
        rng = random.Random(seed)
        infection_times: dict[int, int] = {source: 0}
        cascade_edges: list[tuple[int, int]] = []
        infected = {source}
        recovered: set[int] = set()

        for t in range(1, max_steps + 1):
            # --- Transmission phase ---
            new_infections: dict[int, int] = {}
            for node in infected:
                for neighbor in G.neighbors(node):
                    if (
                        neighbor not in infection_times
                        and neighbor not in recovered
                        and neighbor not in new_infections
                    ):
                        if rng.random() < self.beta:
                            new_infections[neighbor] = node

            # --- Recovery phase ---
            newly_recovered = set()
            for node in infected:
                if rng.random() < self.gamma:
                    newly_recovered.add(node)

            # --- Update state ---
            recovered |= newly_recovered
            infected -= newly_recovered

            for neighbor, infector in new_infections.items():
                infection_times[neighbor] = t
                cascade_edges.append((infector, neighbor))
                infected.add(neighbor)

            # Stop when no active infections remain
            if not infected:
                break

        return CascadeResult(
            source=source,
            model_name="SIR",
            params={"beta": self.beta, "gamma": self.gamma},
            infection_times=infection_times,
            cascade_edges=cascade_edges,
        )


# ── Factory ─────────────────────────────────────────────────────────────────


def create_model(name: str, **params) -> IndependentCascade | SIModel | SIRModel:
    """Convenience factory to create a model by name.

    Parameters
    ----------
    name : str
        One of ``"IC"``, ``"SI"``, ``"SIR"`` (case-insensitive).
    **params
        Keyword arguments forwarded to the model constructor.
    """
    name = name.upper()
    if name == "IC":
        return IndependentCascade(**params)
    elif name == "SI":
        return SIModel(**params)
    elif name == "SIR":
        return SIRModel(**params)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose from IC, SI, SIR.")
