"""
scripts/experiments/plot_ba_network.py
Visualize the BA network structure, highlighting hubs and degree distribution.
"""
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.data.networks import generate_ba_network


def main() -> None:
    G = generate_ba_network(n=200, m=3, seed=42)

    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    avg_deg = np.mean(list(degrees.values()))

    # --- Figure: 1 row, 2 columns (network + degree distribution) ---
    fig, (ax_net, ax_hist) = plt.subplots(1, 2, figsize=(20, 10),
                                           gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0d0d1a")

    # ── LEFT: Network visualization ──
    ax_net.set_facecolor("#1a1a2e")

    # Use spring layout for nice visual separation
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # Color nodes by degree (hubs are bright, periphery is dark)
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=max_deg)
    node_colors = [cmap(norm(degrees[n])) for n in G.nodes()]

    # Size nodes by degree
    node_sizes = [30 + 15 * degrees[n] for n in G.nodes()]

    # Draw edges first (thin, transparent)
    nx.draw_networkx_edges(G, pos, ax=ax_net, edge_color="#333355",
                           alpha=0.15, width=0.5)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_color=node_colors,
                           node_size=node_sizes, edgecolors="none", alpha=0.9)

    # Label the top 5 hubs
    top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:5]
    hub_labels = {n: f"{n}\n(deg={degrees[n]})" for n in top_hubs}
    nx.draw_networkx_labels(G, pos, labels=hub_labels, ax=ax_net,
                            font_size=8, font_color="white", font_weight="bold")

    ax_net.set_title(
        f"Barabasi-Albert Network (N={G.number_of_nodes()}, m=3)\n"
        f"Edges={G.number_of_edges()}, Avg Degree={avg_deg:.1f}, Max Degree={max_deg}",
        color="white", fontweight="bold", fontsize=13
    )
    ax_net.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, shrink=0.6, pad=0.02)
    cbar.set_label("Node Degree", color="lightgray", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="lightgray")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="lightgray")

    # ── RIGHT: Degree distribution (log-log) ──
    ax_hist.set_facecolor("#1a1a2e")

    deg_vals = list(degrees.values())
    unique_degs, counts = np.unique(deg_vals, return_counts=True)

    ax_hist.scatter(unique_degs, counts, color="#ffb703", s=60,
                    edgecolors="black", linewidth=0.5, zorder=3)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Degree (k)", color="lightgray", fontsize=12)
    ax_hist.set_ylabel("Count", color="lightgray", fontsize=12)
    ax_hist.set_title("Degree Distribution (log-log)\nScale-free power law",
                      color="white", fontweight="bold", fontsize=13)
    ax_hist.tick_params(colors="lightgray")
    ax_hist.grid(True, alpha=0.2, color="gray")
    for sp in ax_hist.spines.values():
        sp.set_edgecolor("#444")

    plt.tight_layout()
    out_path = Path("results/figures/networks/ba_network_structure.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
