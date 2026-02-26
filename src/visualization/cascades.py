"""
src.visualization.cascades
==========================
Hierarchical cascade-tree visualizations.

Shows infection flow from the source (bottom) upward, with nodes coloured
by infection time and sized by out-degree in the cascade tree.

Functions
---------
plot_cascade_tree     : Single cascade as a hierarchical directed tree.
plot_comparison_grid  : Side-by-side cascade trees across multiple networks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult


# ── Layout helpers ──────────────────────────────────────────────────────────


def _bfs_hierarchy_layout(
    tree: nx.DiGraph,
    source: int,
    width: float = 2.0,
    y_gap: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """BFS-based hierarchical layout with source at the bottom."""
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


def _hierarchy_layout(tree: nx.DiGraph, source: int) -> dict:
    """Try graphviz dot layout, fall back to BFS."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(tree, prog="dot", root=source)
        max_y = max(p[1] for p in pos.values()) if pos else 0
        return {n: (x, max_y - y) for n, (x, y) in pos.items()}
    except Exception:
        return _bfs_hierarchy_layout(tree, source)


# ── Public API ──────────────────────────────────────────────────────────────


def plot_cascade_tree(
    result: CascadeResult,
    G: nx.Graph | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure | None:
    """Plot a single cascade as a hierarchical directed tree.

    Source is placed at the bottom; infection propagates upward.
    Nodes are coloured by infection time (YlOrRd) and the source is
    highlighted in red.

    Parameters
    ----------
    result : CascadeResult
    G : nx.Graph, optional
        Original contact network (reserved for future edge rendering).
    ax : plt.Axes, optional
        Axes to draw on; if None a new figure is created.
    save_path : str or Path, optional
        If provided the figure is saved here.
    figsize : tuple[int, int]
    """
    tree = result.infection_tree
    if tree.number_of_nodes() == 0:
        return None
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    pos = _hierarchy_layout(tree, result.source)
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
    labels = {n: f"{n}\nt={times.get(n, '?')}" for n in node_order}
    nx.draw_networkx_labels(tree, pos, labels, ax=ax, font_size=6)
    param_str = ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in result.params.items()
    )
    ax.set_title(
        f"{result.model_name} on {result.network_name} | "
        f"Source={result.source}  Size={result.size}  Depth={result.depth}\n"
        f"Params: {param_str}  R₀(actual)={result.actual_r0():.2f}",
        fontsize=10, fontweight="bold", pad=12,
    )
    ax.legend(handles=[
        mpatches.Patch(color="red", label="Source (Patient Zero)"),
        mpatches.Patch(color=cmap(0.3), label="Early infection"),
        mpatches.Patch(color=cmap(0.9), label="Late infection"),
    ], loc="upper right", fontsize=8)
    ax.axis("off")
    if save_path is not None and created_fig:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig if created_fig else None


def plot_comparison_grid(
    results_by_network: dict[str, CascadeResult],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (20, 7),
) -> plt.Figure:
    """Compare cascade trees side-by-side across different networks.

    Parameters
    ----------
    results_by_network : dict[str, CascadeResult]
        Network name → CascadeResult mapping.
    save_path : str or Path, optional
    figsize : tuple[int, int]
    """
    n = len(results_by_network)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (net_name, result) in zip(axes, results_by_network.items()):
        plot_cascade_tree(result, ax=ax)
        ax.set_title(
            f"{net_name}\nSize={result.size}  R₀={result.actual_r0():.2f}",
            fontsize=11, fontweight="bold",
        )
    first = next(iter(results_by_network.values()))
    fig.suptitle(
        f"Cascade Comparison — Source={first.source}  Model={first.model_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
