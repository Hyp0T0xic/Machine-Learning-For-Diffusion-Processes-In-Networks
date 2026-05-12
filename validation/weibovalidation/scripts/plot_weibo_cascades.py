"""
scripts/experiments/plot_weibo_cascades.py
Show example cascades for real Weibo diffusion trees.
"""
from pathlib import Path
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from validation.weibovalidation.scripts.validate_weibo import load_weibo_cascades
from src.visualization.cascades import plot_cascade_tree

N_EXAMPLES = 5

def main() -> None:
    # 1. Load cascades
    print(f"Loading Weibo cascades...")
    # Get a bit more than we need so we can pick nice examples
    cascades = load_weibo_cascades(max_count=50) 
    
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
        c.network_name = f"Weibo Reposts"
        
        plot_cascade_tree(c, ax=ax)

        ax.set_title(
            f"Weibo Example {i+1}\n"
            f"Source={c.source}, Size={c.size}, Depth={c.depth}",
            color="white", fontweight="bold", fontsize=11
        )

    fig.suptitle(
        f"Real-World Weibo Cascade Examples",
        color="white", fontweight="bold", fontsize=16, y=1.02
    )
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent.parent / "figures" / "examples_weibo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Weibo examples -> {out_path}")


if __name__ == "__main__":
    main()
