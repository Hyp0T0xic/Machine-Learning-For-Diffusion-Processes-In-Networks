"""
scripts/experiments/plot_cascades_ic_ba.py
Show example cascades for IC on BA across different R0 values.
"""
from pathlib import Path
import random
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.networks import generate_ba_network
from src.data.cascade import r0_to_params, IndependentCascade
from src.visualization.cascades import plot_cascade_tree

CASCADE_SIZE = 25
N_EXAMPLES = 3


def main() -> None:
    G = generate_ba_network(n=200, m=3, seed=42)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(42)

    r0_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    fig, axes = plt.subplots(len(r0_values), N_EXAMPLES,
                             figsize=(8 * N_EXAMPLES, 8 * len(r0_values)))
    fig.patch.set_facecolor("#0d0d1a")

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
                ax.set_facecolor("#1a1a2e")
                c.network_name = f"BA(200,3)"
                plot_cascade_tree(c, ax=ax)

                src_deg = G.degree(c.source)
                ax.set_title(
                    f"R0={r0} | Example {found+1}\n"
                    f"Source={c.source} (degree={src_deg}), Size={c.size}",
                    color="white", fontweight="bold", fontsize=11
                )
                found += 1

    fig.suptitle(
        f"IC on BA — Cascade Examples (size={CASCADE_SIZE})\n"
        f"R0=0.5 (weak) | R0=1.0 (critical) | R0=2.0 | R0=3.0 (strong) | R0=5.0 (explosive)",
        color="white", fontweight="bold", fontsize=16, y=1.02
    )
    plt.tight_layout()

    out_path = Path("results/figures/cascades/examples_ic_ba.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IC on BA examples -> {out_path}")


if __name__ == "__main__":
    main()
