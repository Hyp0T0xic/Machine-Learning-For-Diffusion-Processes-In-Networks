"""
Visualization module for network comparison plots.

Creates a 1×3 subplot figure with nodes colored by degree centrality
and sized proportionally to degree.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


def visualize_networks(
    networks: dict[str, "nx.Graph"],
    stats: dict[str, dict] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (20, 7),
    seed: int = 42,
) -> plt.Figure:
    """Visualize networks side-by-side with degree-centrality colouring.

    Parameters
    ----------
    networks : dict[str, nx.Graph]
        Mapping of short name → graph (expected keys: "ER", "BA", "Complete").
    stats : dict[str, dict], optional
        Pre-computed stats per network (from ``compute_network_stats``).
        If *None*, subtitles show only the network name.
    save_path : str or Path, optional
        If given, the figure is saved to this path.
    figsize : tuple[int, int]
        Figure size in inches.
    seed : int
        Seed for spring layout reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        "Network Structures for Diffusion Cascade Source Detection",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Collect all degree-centrality values for a shared colour scale
    all_centralities: list[float] = []
    network_data: list[dict] = []

    order = ["ER", "BA", "Complete"]
    for key in order:
        G = networks[key]
        centrality = nx.degree_centrality(G)
        values = list(centrality.values())
        all_centralities.extend(values)
        network_data.append({"key": key, "G": G, "centrality": centrality, "values": values})

    vmin = min(all_centralities)
    vmax = max(all_centralities)
    cmap = plt.cm.YlOrRd  # warm gradient: yellow → orange → red

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, data in enumerate(network_data):
        ax = axes[idx]
        G = data["G"]
        centrality = data["centrality"]
        key = data["key"]

        # ----- Layout -----
        if key == "Complete":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(G.number_of_nodes()))

        # ----- Node properties -----
        node_colors = [centrality[n] for n in G.nodes()]
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [30 + 250 * (degrees[n] / max_deg) for n in G.nodes()]

        # ----- Draw -----
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.5,
        )

        # ----- Title -----
        title = G.graph.get("name", key)
        if stats and key in stats:
            s = stats[key]
            title += (
                f"\nN={s['nodes']}  "
                f"⟨k⟩={s['avg_degree']}  "
                f"diam={s['diameter']}"
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    # ----- Shared colorbar -----
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08)
    cbar.set_label("Degree Centrality", fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    return fig
