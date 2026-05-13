#!/usr/bin/env python
"""draw 5 example outbreak-tree cascades from the dataset"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from validation.outbreaksvalidation.scripts.validate_outbreak_trees import load_all_trees
from src.visualization.cascades import plot_cascade_tree

N_EXAMPLES = 5


def main() -> None:
    print("loading outbreaktrees cascades ...")
    cascades = load_all_trees()
    
    if not cascades:
        print("No cascades found or loaded.")
        return

    # Sort cascades by size so we can pick a diverse set
    cascades.sort(key=lambda c: c.size)
    
    # Pick 5 evenly spaced examples
    if len(cascades) > N_EXAMPLES:
        indices = [int(i) for i in range(0, len(cascades), len(cascades) // N_EXAMPLES)][:N_EXAMPLES]
        selected_cascades = [cascades[i] for i in indices]
    else:
        selected_cascades = cascades

    plt.style.use("default")
    fig, axes = plt.subplots(1, len(selected_cascades),
                             figsize=(8 * len(selected_cascades), 8))
    fig.patch.set_facecolor("white")

    if len(selected_cascades) == 1:
        axes = [axes]

    for i, c in enumerate(selected_cascades):
        ax = axes[i]
        ax.set_facecolor("white")
        c.network_name = f"Outbreak Tree"
        
        plot_cascade_tree(c, ax=ax)

        ax.set_title(
            f"Biological Example {i+1}\n"
            f"Source={c.source}, Size={c.size}, Depth={c.depth}",
            color="black", fontweight="bold", fontsize=11
        )

    fig.suptitle(
        f"Real-World Biological Transmission Tree Examples (OutbreakTrees)",
        color="black", fontweight="bold", fontsize=18, y=1.02
    )
    plt.tight_layout()

    out_path = _REPO_ROOT / "validation/outbreaksvalidation/figures/cascades/examples_outbreak_trees.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved OutbreakTrees examples -> {out_path}")


if __name__ == "__main__":
    main()
