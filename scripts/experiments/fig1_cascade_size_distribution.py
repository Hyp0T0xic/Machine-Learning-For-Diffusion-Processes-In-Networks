"""
Figure 1 — Cascade Size Distribution vs. R₀

Runs N_SIMS=200 simulations of IC on ER (⟨k⟩=6) and BA (m=3)
across a grid of R₀ values (0.5 to 4.0 in steps of 0.25). Records final
cascade size as a fraction of N for each run.

Produces a 1×2 grid of violin plots:
  columns = topology (ER / BA)

Horizontal shaded bands mark the "too small" (<5% of N, structurally trivial)
and "too large" (>90% of N, saturated/washed-out) regions. Vertical dashed
lines mark the chosen operating R₀ bounds.

Defends: R₀ sweep range, max_size cutoff, and the identifiability window.
BA saturation at lower R₀ than ER is a direct empirical demonstration of the
Pastor-Satorras & Vespignani zero-threshold result on scale-free networks.

Usage
-----
    python -m scripts.experiments.fig1_cascade_size_distribution
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cascade import r0_to_params, create_model
from src.data.networks import generate_er_network, generate_ba_network

# ── config ────────────────────────────────────────────────────────────────────

N_NODES      = 200
BA_M         = 3
ER_P         = 6 / (N_NODES - 1)   # ⟨k⟩ ≈ 6, matching BA

R0_GRID      = np.round(np.arange(0.5, 4.01, 0.25), 2)   # 15 values: 0.50 … 4.00
N_SIMS       = 200                                          # runs per (model, topology, R₀)
SEED         = 42

SMALL_THRESH  = 0.05    # fraction of N below which cascades are structurally trivial
LARGE_THRESH  = 0.90    # fraction of N above which cascades are saturated / washed-out
OPERATING_R0  = (0.5, 3.0)   # training sweep bounds — drawn as vertical dashed lines

MODELS = ["IC"]
TOPOS  = ["ER", "BA"]

OUT_DIR = Path("results/figures/ml_evaluation")


# ── helpers ───────────────────────────────────────────────────────────────────


def _avg_degree(G: nx.Graph) -> float:
    return sum(d for _, d in G.degree()) / G.number_of_nodes()


def collect_sizes(
    G: nx.Graph,
    model_name: str,
    r0: float,
    n_sims: int,
    seed: int,
) -> list[float]:
    """Simulate n_sims cascades; return sizes as fraction of N."""
    N      = G.number_of_nodes()
    params = r0_to_params(r0, _avg_degree(G), model=model_name)
    model  = create_model(model_name, **params)
    nodes  = list(G.nodes())
    rng    = random.Random(seed)
    sizes  = []
    for i in range(n_sims):
        src    = rng.choice(nodes)
        result = model.run(G, source=src, seed=seed * 10_000 + i)
        sizes.append(result.size / N)
    return sizes


# ── simulation ────────────────────────────────────────────────────────────────


def run_all(G_er: nx.Graph, G_ba: nx.Graph) -> dict[tuple[str, str], list[list[float]]]:
    """Collect cascade size distributions for every (model, topology, R₀) combo."""
    graphs = {"ER": G_er, "BA": G_ba}
    data: dict[tuple[str, str], list[list[float]]] = {}
    total = len(MODELS) * len(TOPOS) * len(R0_GRID)
    done  = 0

    for model_name in MODELS:
        for topo in TOPOS:
            G = graphs[topo]
            r0_arrays: list[list[float]] = []
            for r0 in R0_GRID:
                done += 1
                print(
                    f"  [{done:>3}/{total}]  {model_name} on {topo}  R0={r0:.2f}",
                    flush=True,
                )
                r0_arrays.append(collect_sizes(G, model_name, r0, N_SIMS, seed=SEED))
            data[(model_name, topo)] = r0_arrays

    return data


# ── plotting ──────────────────────────────────────────────────────────────────


def _violin_panel(
    ax: plt.Axes,
    r0_arrays: list[list[float]],
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    """Draw one violin panel with bands and operating-R₀ markers."""
    positions = list(range(len(R0_GRID)))

    parts = ax.violinplot(
        r0_arrays,
        positions=positions,
        widths=0.72,
        showmedians=True,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor("white")
        body.set_edgecolor("black")
        body.set_linewidth(0.75)
        body.set_alpha(1.0)

    parts["cmedians"].set_edgecolor("black")
    parts["cmedians"].set_linewidth(1.4)

    # horizontal threshold bands
    ax.axhspan(0, SMALL_THRESH, color="0.78", alpha=0.55, zorder=0,
               label=f"<{int(SMALL_THRESH*100)}% N  (trivial)")
    ax.axhspan(LARGE_THRESH, 1.0, color="0.50", alpha=0.30, zorder=0,
               label=f">{int(LARGE_THRESH*100)}% N  (saturated)")

    # operating-R₀ vertical markers
    r0_low_pos  = int(np.argmin(np.abs(R0_GRID - OPERATING_R0[0])))
    r0_high_pos = int(np.argmin(np.abs(R0_GRID - OPERATING_R0[1])))

    ax.axvline(r0_low_pos,  color="black", linestyle="--", linewidth=1.1, zorder=4,
               label=rf"$R_0={OPERATING_R0[0]}$ (op. bound)")
    ax.axvline(r0_high_pos, color="black", linestyle=":",  linewidth=1.1, zorder=4,
               label=rf"$R_0={OPERATING_R0[1]}$ (op. bound)")

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.75, len(R0_GRID) - 0.25)

    # show every other x-tick to prevent crowding
    tick_idx = list(range(0, len(R0_GRID), 2))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{R0_GRID[i]:.2f}" for i in tick_idx], fontsize=7)

    if show_xlabel:
        ax.set_xlabel(r"$R_0$", fontsize=10)

    if show_ylabel:
        ax.set_ylabel("cascade size / N", fontsize=9)

    ax.set_title(title, fontsize=9, pad=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))

    if show_legend:
        ax.legend(fontsize=7, frameon=True, loc="upper left")


def plot_figure(data: dict[tuple[str, str], list[list[float]]]) -> Path:
    """Assemble and save the 3×2 grid figure."""
    plt.style.use("default")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(TOPOS),
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )

    model_name = MODELS[0]
    for col_idx, topo in enumerate(TOPOS):
        ax = axes[col_idx]
        _violin_panel(
            ax,
            r0_arrays  = data[(model_name, topo)],
            title      = f"{model_name} — {topo}",
            show_xlabel= True,
            show_ylabel= col_idx == 0,
            show_legend= col_idx == 0,
        )

    fig.suptitle(
        rf"IC — Cascade Size Distribution vs. $R_0$"
        rf"  (N={N_NODES}, {N_SIMS} runs per condition, "
        rf"$\langle k \rangle \approx 6$)",
        fontsize=12,
    )
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "fig1_cascade_size_distribution_r0.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Building networks  N={N_NODES}  ER_p={ER_P:.4f}  BA_m={BA_M}")  # noqa: RUF001
    G_er = generate_er_network(N_NODES, ER_P, seed=SEED)
    G_ba = generate_ba_network(N_NODES, BA_M, seed=SEED)
    print(f"  ER avg_degree = {_avg_degree(G_er):.3f}")
    print(f"  BA avg_degree = {_avg_degree(G_ba):.3f}")

    n_combos = len(MODELS) * len(TOPOS) * len(R0_GRID)
    print(
        f"\nRunning {N_SIMS} sims x {len(R0_GRID)} R0 values "
        f"x {len(MODELS)} models x {len(TOPOS)} topologies  "
        f"= {N_SIMS * n_combos:,} total cascade runs"
    )
    data = run_all(G_er, G_ba)

    print("\nPlotting ...")
    out_file = plot_figure(data)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
