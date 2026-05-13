"""er network figure on white background: spring-layout graph + poisson-style degree histogram"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.networks import generate_er_network
from src.visualization.theme import METHOD_COLORS


def main() -> None:
    G = generate_er_network(n=200, p=6/199, seed=42)

    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    avg_deg = np.mean(list(degrees.values()))

    plt.style.use("default")
    fig, (ax_net, ax_hist) = plt.subplots(1, 2, figsize=(20, 10),
                                           gridspec_kw={"width_ratios": [2, 1]})

    # left panel: network
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # plasma colormap reads well on white too — bright = high degree
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=max_deg)
    node_colors = [cmap(norm(degrees[n])) for n in G.nodes()]
    node_sizes = [30 + 15 * degrees[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax_net, edge_color="grey",
                           alpha=0.25, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_color=node_colors,
                           node_size=node_sizes, edgecolors="none", alpha=0.9)

    # label the top-degree nodes (er has no true hubs, but flag the extremes anyway)
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
    node_labels = {n: f"{n}\n(deg={degrees[n]})" for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_net,
                            font_size=8, font_color="black", font_weight="bold")

    ax_net.set_title(
        f"Erdos-Renyi Network (N={G.number_of_nodes()}, p=6/(N-1))\n"
        f"Edges={G.number_of_edges()}, Avg Degree={avg_deg:.1f}, Max Degree={max_deg}",
        fontweight="bold", fontsize=13
    )
    ax_net.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, shrink=0.6, pad=0.02)
    cbar.set_label("Node Degree", fontsize=11)

    # right panel: degree distribution (linear because er ~ binomial/poisson)
    ax_hist.hist(list(degrees.values()), bins=range(0, max_deg + 2),
                 color=METHOD_COLORS["RF (IC-ER)"], edgecolor="black", alpha=0.85)
    ax_hist.set_xlabel("Degree (k)", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_title("Degree Distribution\nPoisson-like (Bell curve)",
                      fontweight="bold", fontsize=13)
    ax_hist.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    out_path = _REPO_ROOT / "results/figures/networks/er_network_structure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
