"""
Cascade visualization module.

Creates hierarchical tree layouts showing infection flow from the source
(bottom) upward, with nodes colored by infection time and sized by
out-degree in the cascade.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from .models import CascadeResult


# ── Layout helpers ──────────────────────────────────────────────────────────


def _bfs_hierarchy_layout(
    tree: nx.DiGraph,
    source: int,
    width: float = 2.0,
    y_gap: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """Custom BFS-based hierarchical layout with source at the bottom.

    Parameters
    ----------
    tree : nx.DiGraph
        Directed infection tree.
    source : int
        Root node (placed at the bottom).
    width : float
        Horizontal spread.
    y_gap : float
        Vertical distance between levels.

    Returns
    -------
    dict[int, tuple[float, float]]
        Node positions {node: (x, y)}.
    """
    # BFS to determine levels
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

    # Include any disconnected infected nodes
    for node in tree.nodes():
        if node not in visited:
            max_level = max(levels.keys()) + 1 if levels else 0
            levels[max_level].append(node)

    max_depth = max(levels.keys()) if levels else 0
    pos = {}

    for depth, nodes in levels.items():
        n = len(nodes)
        y = depth * y_gap  # source at y=0 (bottom), deeper levels go up
        for i, node in enumerate(nodes):
            if n == 1:
                x = 0.0
            else:
                x = -width / 2 + i * width / (n - 1)
            pos[node] = (x, y)

    return pos


def _get_hierarchy_layout(tree: nx.DiGraph, source: int) -> dict:
    """Try graphviz dot layout first, fall back to BFS layout."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(tree, prog="dot", root=source)
        # Flip Y axis so source is at the bottom
        ys = [p[1] for p in pos.values()]
        max_y = max(ys) if ys else 0
        pos = {n: (x, max_y - y) for n, (x, y) in pos.items()}
        return pos
    except (ImportError, Exception):
        return _bfs_hierarchy_layout(tree, source)


# ── Main plotting functions ─────────────────────────────────────────────────


def plot_cascade_tree(
    result: CascadeResult,
    G: nx.Graph | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure | None:
    """Plot a single cascade as a hierarchical tree.

    Source is at the bottom, infection spreads upward. Nodes are colored by
    infection time and the source is highlighted in bright red.

    Parameters
    ----------
    result : CascadeResult
        Cascade to visualize.
    G : nx.Graph, optional
        Original network (unused but kept for future edge-drawing).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If *None*, a new figure is created.
    save_path : str or Path, optional
        If given, save the figure to this path.
    figsize : tuple
        Figure size if a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    tree = result.infection_tree
    if tree.number_of_nodes() == 0:
        return None

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    pos = _get_hierarchy_layout(tree, result.source)

    # ── Node colors by infection time ────────────────────────────────
    times = result.infection_times
    max_t = max(times.values()) if times else 1
    max_t = max(max_t, 1)  # avoid division by zero

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=max_t)

    node_order = list(tree.nodes())
    node_colors = []
    node_sizes = []

    out_degrees = dict(tree.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    max_out = max(max_out, 1)

    for n in node_order:
        t = times.get(n, max_t)
        if n == result.source:
            node_colors.append("red")
            node_sizes.append(500)
        else:
            node_colors.append(cmap(norm(t)))
            node_sizes.append(100 + 200 * (out_degrees.get(n, 0) / max_out))

    # ── Draw edges (directed arrows) ────────────────────────────────
    nx.draw_networkx_edges(
        tree,
        pos,
        ax=ax,
        edge_color="#cc3333",
        arrows=True,
        arrowsize=12,
        alpha=0.6,
        width=1.2,
        connectionstyle="arc3,rad=0.05",
    )

    # ── Draw nodes ──────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        tree,
        pos,
        ax=ax,
        nodelist=node_order,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=[2.0 if n == result.source else 0.5 for n in node_order],
    )

    # ── Labels:  node_id (t=time) ───────────────────────────────────
    labels = {n: f"{n}\nt={times.get(n, '?')}" for n in node_order}
    nx.draw_networkx_labels(
        tree, pos, labels, ax=ax, font_size=6, font_color="black"
    )

    # ── Legend ───────────────────────────────────────────────────────
    param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in result.params.items())
    title = (
        f"{result.model_name} Cascade on {result.network_name}\n"
        f"Source={result.source}  Size={result.size}  Depth={result.depth}\n"
        f"Params: {param_str}  R₀(actual)={result.actual_r0():.2f}"
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)

    # Custom legend items
    legend_items = [
        mpatches.Patch(color="red", label="Source (Patient Zero)"),
        mpatches.Patch(color=cmap(0.3), label="Early infection"),
        mpatches.Patch(color=cmap(0.9), label="Late infection"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
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
    """Compare cascades from the same source across different networks.

    Parameters
    ----------
    results_by_network : dict[str, CascadeResult]
        Mapping of network name → CascadeResult (one per network).
    save_path : str or Path, optional
        If given, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(results_by_network)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (net_name, result) in zip(axes, results_by_network.items()):
        plot_cascade_tree(result, ax=ax)
        ax.set_title(
            f"{net_name}\nSize={result.size}  R₀={result.actual_r0():.2f}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Cascade Comparison — Source={list(results_by_network.values())[0].source}  "
        f"Model={list(results_by_network.values())[0].model_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
