"""
scripts/experiments/plot_outbreak_trees_cascades.py
Show example cascades for real biological transmission trees (OutbreakTrees).
"""
from pathlib import Path
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import our custom loaders
from scripts.experiments.validate_outbreak_trees import load_all_trees
from src.visualization.cascades import plot_cascade_tree

N_EXAMPLES = 5

def main() -> None:
    # 1. Load cascades
    print(f"Loading OutbreakTrees cascades...")
    # Get all trees that match our criteria
    cascades = load_all_trees() 
    
    if not cascades:
        print("No cascades found or loaded.")
        return

    # Sort cascades by size so we can pick a diverse set
    cascades.sort(key=lambda c: c.size)
    
    # Pick a few evenly spaced examples from the loaded ones
    if len(cascades) > N_EXAMPLES:
        indices = [int(i) for i in range(0, len(cascades), len(cascades) // N_EXAMPLES)][:N_EXAMPLES]
        selected_cascades = [cascades[i] for i in indices]
    else:
        selected_cascades = cascades

    fig, axes = plt.subplots(1, len(selected_cascades),
                             figsize=(8 * len(selected_cascades), 8))
    fig.patch.set_facecolor("#0d0d1a")

    # If only 1 cascade, axes is not an array
    if len(selected_cascades) == 1:
        axes = [axes]

    for i, c in enumerate(selected_cascades):
        ax = axes[i]
        ax.set_facecolor("#1a1a2e")
        c.network_name = f"Outbreak Tree"
        
        plot_cascade_tree(c, ax=ax)

        ax.set_title(
            f"Biological Example {i+1}\n"
            f"Source={c.source}, Size={c.size}, Depth={c.depth}",
            color="white", fontweight="bold", fontsize=11
        )

    fig.suptitle(
        f"Real-World Biological Transmission Tree Examples (OutbreakTrees)",
        color="white", fontweight="bold", fontsize=16, y=1.02
    )
    plt.tight_layout()

    out_path = Path("results/figures/cascades/examples_outbreak_trees.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved OutbreakTrees examples -> {out_path}")


if __name__ == "__main__":
    main()
