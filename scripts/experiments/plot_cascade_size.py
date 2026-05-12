"""
Quick check for the "size >= 25" cutoff.

For a few R0 values, simulate IC cascades on BA and ER graphs and count
how many tries it takes to get a cascade of at least 25 nodes.
Saves a small bar plot at the end.
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cascade import r0_to_params, IndependentCascade
from src.data.networks import generate_ba_network, generate_er_network

# basic config
N_NODES      = 200
BA_M         = 3
ER_P         = 6 / (N_NODES - 1)
R0_VALUES    = [0.5, 1.0, 2.0, 3.0, 5.0]
CASCADE_SIZE = 25
N_COLLECT    = 1000
SEED         = 42

OUT_DIR = Path("results/figures/ml_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def attempts_per_cascade(G, r0: float, n_collect: int, seed: int) -> list[int]:
    """
    Run IC on G until n_collect cascades of size >= CASCADE_SIZE are found.
    Returns the number of attempts required for each successful cascade.
    """
    avg_deg = sum(d for _, d in G.degree()) / G.number_of_nodes()
    p = r0_to_params(r0, avg_deg, model="IC")["p"]
    model = IndependentCascade(p=p)

    rng = random.Random(seed)
    nodes = list(G.nodes())

    per_cascade = []
    attempts = 0
    collected = 0

    while collected < n_collect:
        source = rng.choice(nodes)
        cascade = model.run(G, source)  # new RNG in model if you pass seed=None
        attempts += 1

        if cascade.size >= CASCADE_SIZE:
            per_cascade.append(attempts)
            attempts = 0
            collected += 1

    return per_cascade


def _bar_panel(ax, x, means, stds, x_labels, title):
    width = 0.35
    offsets = [-width / 2, width / 2]
    # BA = white fill, ER = diagonal hatch -- distinct in black and white
    styles = [
        ("BA", dict(facecolor="white",  edgecolor="black", hatch="",    linewidth=0.8)),
        ("ER", dict(facecolor="white",  edgecolor="black", hatch="////", linewidth=0.8)),
    ]

    for offset, (net, bar_kw), mean_arr, std_arr in zip(offsets, styles, means, stds):
        bars = ax.bar(x + offset, mean_arr, width, label=net, **bar_kw)
        ax.errorbar(x + offset, mean_arr, yerr=std_arr,
                    fmt="none", color="black", capsize=4, linewidth=1.2, capthick=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(r"total simulation runs ($\pm$ std)")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    G_ba = generate_ba_network(N_NODES, BA_M, seed=SEED)
    G_er = generate_er_network(N_NODES, ER_P, seed=SEED)

    ba_totals, ba_stds = [], []
    er_totals, er_stds = [], []

    for r0 in R0_VALUES:
        ba = attempts_per_cascade(G_ba, r0, N_COLLECT, SEED)
        ba_total = sum(ba)
        ba_totals.append(ba_total)
        ba_stds.append(np.std(ba) * np.sqrt(N_COLLECT))  # std of the total, not per-cascade

        er = attempts_per_cascade(G_er, r0, N_COLLECT, SEED)
        er_total = sum(er)
        er_totals.append(er_total)
        er_stds.append(np.std(er) * np.sqrt(N_COLLECT))

        print(f"R0={r0:.1f} | BA: {ba_total:>10,} runs  |  ER: {er_total:>10,} runs")

    ba_totals = np.array(ba_totals)
    ba_stds   = np.array(ba_stds)
    er_totals = np.array(er_totals)
    er_stds   = np.array(er_stds)

    x        = np.arange(len(R0_VALUES))
    x_labels = [f"R0={r}" for r in R0_VALUES]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Simulation runs required to collect cascades of size $\\geq {CASCADE_SIZE}$ (IC)",
        fontsize=12,
    )

    _bar_panel(
        ax, x,
        [ba_totals, er_totals],
        [ba_stds,   er_stds],
        x_labels,
        f"$N = {N_COLLECT}$ cascades",
    )

    plt.tight_layout()
    out_file = OUT_DIR / f"cascade_size_{CASCADE_SIZE}_justification.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
