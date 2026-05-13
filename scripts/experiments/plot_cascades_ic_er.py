#!/usr/bin/env python
"""grid of example ic-er cascades, one row per r0 in {0.5, 1.0, 2.0, 3.0, 5.0}"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.networks import generate_er_network
from src.data.cascade import r0_to_params, IndependentCascade
from src.visualization.cascades import plot_cascade_tree

CASCADE_SIZE = 25
N_EXAMPLES = 3


def main() -> None:
    G = generate_er_network(n=200, p=0.03, seed=42)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(42)

    r0_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    plt.style.use("default")
    fig, axes = plt.subplots(len(r0_values), N_EXAMPLES,
                             figsize=(8 * N_EXAMPLES, 8 * len(r0_values)))
    fig.patch.set_facecolor("white")

    for row, r0 in enumerate(r0_values):
        p = r0_to_params(r0, avg_deg, model="IC")["p"]
        model = IndependentCascade(p=p)

        found = 0
        seed = 42
        while found < N_EXAMPLES:
            source = rng.choice(nodes)
            c = model.run(G, source=source, seed=seed, max_size=CASCADE_SIZE)
            seed += 1
            if c.size >= CASCADE_SIZE:
                ax = axes[row, found]
                ax.set_facecolor("white")
                c.network_name = f"ER(200, 0.03)"
                plot_cascade_tree(c, ax=ax)

                src_deg = G.degree(c.source)
                ax.set_title(
                    f"R0={r0} | Example {found+1}\n"
                    f"Source={c.source} (degree={src_deg}), Size={c.size}",
                    color="black", fontweight="bold", fontsize=11
                )
                found += 1

    fig.suptitle(
        f"IC on ER — Cascade Examples (size={CASCADE_SIZE})\n"
        f"R0=0.5 (weak) | R0=1.0 (critical) | R0=2.0 | R0=3.0 (strong) | R0=5.0 (explosive)",
        color="black", fontweight="bold", fontsize=18, y=1.02
    )
    plt.tight_layout()

    out_path = _REPO_ROOT / "results/figures/cascades/examples_ic_er.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IC on ER examples -> {out_path}")


if __name__ == "__main__":
    main()
