#!/usr/bin/env python
"""Generate IC cascade comparison plots for BA and ER networks at R0=0.5, 2.0, 5.0.

Each subplot shows a single cascade of exactly 25 nodes.  The script samples
many valid cascades per R0 and picks the most *structurally representative*
one: deepest chain for subcritical, shallowest/bushiest for explosive.

Node labels are suppressed for a cleaner figure suitable for a report.
"""
from __future__ import annotations

import sys
import random
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from src.data.networks import generate_ba_network, generate_er_network
from src.data.cascade import IndependentCascade, r0_to_params, CascadeResult

# ── Config ───────────────────────────────────────────────────────────────
N = 200
CASCADE_SIZE = 25
R0_VALUES = [0.5, 2.0, 5.0]
N_CANDIDATES = 200          # sample this many valid cascades, then pick best
TOPOLOGIES = ["BA", "ER"]

TITLE_MAP = {
    0.5: r"Subcritical ($R_0$=0.5)",
    2.0: r"Supercritical ($R_0$=2.0)",
    5.0: r"Explosive ($R_0$=5.0)",
}

# For low R0 we want the *deepest* cascade (long chain);
# for high R0 we want the *shallowest* (bushy star).
PREFER_DEEP = {0.5: True, 2.0: False, 5.0: False}


# ── Layout helper ────────────────────────────────────────────────────────

def _bfs_hierarchy_layout(
    tree: nx.DiGraph, source: int, width: float = 2.0, y_gap: float = 1.0
) -> dict[int, tuple[float, float]]:
    levels: dict[int, list[int]] = defaultdict(list)
    visited = {source}
    queue = [(source, 0)]
    levels[0].append(source)
    while queue:
        node, depth = queue.pop(0)
        for child in tree.successors(node):
            if child not in visited:
                visited.add(child)
                levels[depth + 1].append(child)
                queue.append((child, depth + 1))
    for node in tree.nodes():
        if node not in visited:
            levels[max(levels.keys(), default=0) + 1].append(node)
    pos = {}
    for depth, nodes in levels.items():
        n = len(nodes)
        y = depth * y_gap
        for i, node in enumerate(nodes):
            x = 0.0 if n == 1 else -width / 2 + i * width / (n - 1)
            pos[node] = (x, y)
    return pos


# ── Clean cascade drawer (no labels) ────────────────────────────────────

def draw_cascade_clean(result: CascadeResult, ax: plt.Axes) -> None:
    """Draw a cascade tree without node-id labels for a clean report figure."""
    tree = result.infection_tree
    if tree.number_of_nodes() == 0:
        return

    pos = _bfs_hierarchy_layout(tree, result.source)
    times = result.infection_times
    max_t = max(max(times.values(), default=1), 1)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=max_t)

    node_order = list(tree.nodes())
    out_degrees = dict(tree.out_degree())
    max_out = max(out_degrees.values(), default=1) or 1

    node_colors = []
    node_sizes = []
    for n in node_order:
        if n == result.source:
            node_colors.append("red")
            node_sizes.append(500)
        else:
            node_colors.append(cmap(norm(times.get(n, max_t))))
            node_sizes.append(100 + 200 * (out_degrees.get(n, 0) / max_out))

    nx.draw_networkx_edges(
        tree, pos, ax=ax, edge_color="#cc3333",
        arrows=True, arrowsize=12, alpha=0.6, width=1.2,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(
        tree, pos, ax=ax, nodelist=node_order,
        node_color=node_colors, node_size=node_sizes,
        edgecolors="black",
        linewidths=[2.0 if n == result.source else 0.5 for n in node_order],
    )
    ax.axis("off")


# ── Network factory ──────────────────────────────────────────────────────

def get_network(topology: str):
    if topology == "BA":
        return generate_ba_network(n=N, m=3, seed=42)
    return generate_er_network(n=N, p=6 / (N - 1), seed=42)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    results_dir = _REPO_ROOT / "results" / "figures" / "cascades"
    results_dir.mkdir(parents=True, exist_ok=True)

    for topo in TOPOLOGIES:
        print(f"\n{'='*60}")
        print(f"  Generating examples for {topo} network")
        print(f"{'='*60}")
        G = get_network(topo)
        avg_deg = float(np.mean([d for _, d in G.degree()]))

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.patch.set_facecolor("white")

        for col, r0 in enumerate(R0_VALUES):
            ax = axes[col]
            p = r0_to_params(r0, avg_deg, model="IC")["p"]
            model = IndependentCascade(p=p)

            print(f"  R0={r0}  p={p:.4f}  — sampling up to {N_CANDIDATES} candidates …")

            # Collect candidate cascades
            candidates: list[CascadeResult] = []
            seed = 0
            attempts = 0
            while len(candidates) < N_CANDIDATES and attempts < 500_000:
                source = random.Random(seed).choice(list(G.nodes()))
                cascade = model.run(G, source=source, seed=seed, max_size=CASCADE_SIZE)
                attempts += 1
                if cascade.size == CASCADE_SIZE:
                    candidates.append(cascade)
                seed += 1

            print(f"    collected {len(candidates)} valid cascades in {attempts} attempts")

            # Pick the most representative
            if PREFER_DEEP[r0]:
                best = max(candidates, key=lambda c: c.depth)
            else:
                best = min(candidates, key=lambda c: c.depth)

            print(f"    selected: depth={best.depth}, source={best.source}")

            draw_cascade_clean(best, ax=ax)
            ax.set_title(
                f"{TITLE_MAP[r0]}\nDepth = {best.depth}  |  {best.size} nodes",
                fontsize=16, fontweight="bold",
            )

        # Shared legend at bottom
        cmap = plt.cm.YlOrRd
        legend_handles = [
            mpatches.Patch(color="red", label="Source (Patient Zero)"),
            mpatches.Patch(color=cmap(0.3), label="Early infection"),
            mpatches.Patch(color=cmap(0.9), label="Late infection"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=3,
            fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.02),
        )

        topo_full = "Barabási–Albert" if topo == "BA" else "Erdős–Rényi"
        fig.suptitle(
            f"IC Cascade Examples on {topo_full} Network ($N={N}$)",
            fontsize=24, fontweight="bold", y=1.05,
        )
        plt.tight_layout()

        out_path = results_dir / f"examples_ic_{topo.lower()}_custom.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
