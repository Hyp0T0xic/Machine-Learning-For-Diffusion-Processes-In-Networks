#!/usr/bin/env python
"""1x2 side-by-side comparison of ER and BA networks at N=200, <k>=6.

Saves results/figures/networks/network_comparison.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data.networks import generate_er_network, generate_ba_network, compute_network_stats
from src.visualization.theme import ACCENT_COLORS

# ── parameters matching the experiments ──────────────────────────────────────
N = 200
ER_P = 6 / (N - 1)   # gives <k> ≈ 6
BA_M = 3              # gives <k> = 6 exactly
SEED = 42
LAYOUT_K = 1.2 / np.sqrt(N)   # spring-layout repulsion

CMAP = plt.cm.YlOrRd
NODE_SIZE_BASE = 20
NODE_SIZE_SCALE = 280


def _draw_panel(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    vmin: float,
    vmax: float,
    label: str,
    stats: dict,
) -> None:
    centrality = nx.degree_centrality(G)
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) or 1

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5, edge_color="#888888")
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[centrality[n] for n in G.nodes()],
        node_size=[NODE_SIZE_BASE + NODE_SIZE_SCALE * (degrees[n] / max_deg) for n in G.nodes()],
        cmap=CMAP,
        vmin=vmin, vmax=vmax,
        edgecolors="white", linewidths=0.4,
    )
    ax.set_title(
        f"{label}\n"
        rf"$N={stats['nodes']}$  $\langle k\rangle={stats['avg_degree']}$  "
        rf"$\mathrm{{diam}}={stats['diameter']}$",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.axis("off")


def _draw_degree_dist(ax: plt.Axes, G: nx.Graph, color: str, label: str) -> None:
    degrees = sorted(d for _, d in G.degree())
    max_deg = max(degrees)
    bins = np.arange(0, max_deg + 2) - 0.5
    ax.hist(degrees, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Degree $k$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{label} — degree distribution", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")


def main() -> None:
    er = generate_er_network(n=N, p=ER_P, seed=SEED)
    ba = generate_ba_network(n=N, m=BA_M, seed=SEED)

    er_stats = compute_network_stats(er)
    ba_stats = compute_network_stats(ba)

    print(f"ER  {er_stats}")
    print(f"BA  {ba_stats}")

    # shared colour normalisation across both networks
    all_centralities = list(nx.degree_centrality(er).values()) + \
                       list(nx.degree_centrality(ba).values())
    vmin, vmax = min(all_centralities), max(all_centralities)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pos_er = nx.spring_layout(er, seed=SEED, k=LAYOUT_K)
    pos_ba = nx.spring_layout(ba, seed=SEED, k=LAYOUT_K)

    # ── figure layout: 2 rows × 2 cols ───────────────────────────────────────
    #   row 0 : network spring layouts (tall panels)
    #   row 1 : degree distribution histograms (shorter panels)
    fig = plt.figure(figsize=(14, 12), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.8, 1],
        hspace=0.35, wspace=0.15,
    )

    ax_er_net = fig.add_subplot(gs[0, 0])
    ax_ba_net = fig.add_subplot(gs[0, 1])
    ax_er_deg = fig.add_subplot(gs[1, 0])
    ax_ba_deg = fig.add_subplot(gs[1, 1])

    for ax in (ax_er_net, ax_ba_net):
        ax.set_facecolor("white")

    _draw_panel(ax_er_net, er, pos_er, vmin, vmax, "Erdős–Rényi", er_stats)
    _draw_panel(ax_ba_net, ba, pos_ba, vmin, vmax, "Barabási–Albert", ba_stats)

    er_color = ACCENT_COLORS[2]   # teal  (matches RF IC-ER palette)
    ba_color = ACCENT_COLORS[5]   # green (matches RF IC-BA palette)
    _draw_degree_dist(ax_er_deg, er, er_color, "Erdős–Rényi")
    _draw_degree_dist(ax_ba_deg, ba, ba_color, "Barabási–Albert")

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_er_net, ax_ba_net],
                        orientation="horizontal", fraction=0.03, pad=0.02)
    cbar.set_label("Normalised degree centrality", fontsize=11)

    out = _REPO_ROOT / "results" / "figures" / "networks" / "network_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
