"""
src.visualization.networks
==========================
Network comparison visualizations.

Creates a 1×3 subplot figure with nodes coloured by degree centrality
and sized proportionally to their degree, on a shared colour scale.

Functions
---------
visualize_networks : Side-by-side plot of ER, BA, and Complete graphs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_networks(
    networks: dict[str, nx.Graph],
    stats: dict[str, dict] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (20, 7),
    seed: int = 42,
) -> plt.Figure:
    """Visualize contact networks side-by-side with degree-centrality colouring.

    Parameters
    ----------
    networks : dict[str, nx.Graph]
        Keys: ``"ER"``, ``"BA"``, ``"Complete"``.
    stats : dict[str, dict], optional
        Pre-computed stats from ``compute_network_stats``; added to subtitles.
    save_path : str or Path, optional
        If given, the figure is saved here at 200 dpi.
    figsize : tuple[int, int]
    seed : int
        Seed for spring layout reproducibility.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        "Contact Network Structures for Diffusion Cascade Source Detection",
        fontsize=16, fontweight="bold", y=1.02,
    )
    order = ["ER", "BA", "Complete"]
    all_centralities: list[float] = []
    network_data: list[dict] = []
    for key in order:
        G = networks[key]
        centrality = nx.degree_centrality(G)
        values = list(centrality.values())
        all_centralities.extend(values)
        network_data.append({"key": key, "G": G, "centrality": centrality, "values": values})
    vmin, vmax = min(all_centralities), max(all_centralities)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for idx, data in enumerate(network_data):
        ax = axes[idx]
        G, centrality, key = data["G"], data["centrality"], data["key"]
        pos = (
            nx.circular_layout(G) if key == "Complete"
            else nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(G.number_of_nodes()))
        )
        degrees = dict(G.degree())
        max_deg = max(degrees.values(), default=1) or 1
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=[centrality[n] for n in G.nodes()],
            node_size=[30 + 250 * (degrees[n] / max_deg) for n in G.nodes()],
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors="white", linewidths=0.5,
        )
        title = G.graph.get("name", key)
        if stats and key in stats:
            s = stats[key]
            title += f"\nN={s['nodes']}  ⟨k⟩={s['avg_degree']}  diam={s['diameter']}"
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08).set_label(
        "Degree Centrality", fontsize=11
    )
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    return fig
