#!/usr/bin/env python
"""
validation/weibovalidation/scripts/plot_weibo_cascades.py
Show example cascades for real Weibo diffusion trees.
Updated to match the "Natural Range" selection (cascades near the mean).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the loader from the natural range validation script
from validation.weibovalidation.scripts.validate_weibo_natural_range import select_median_centred_cascades
from src.visualization.cascades import plot_cascade_tree

N_EXAMPLES = 5
# Use the same config as the natural range experiment
MIN_SIZE = 25
N_TOTAL_FOR_SELECTION = 101

def main() -> None:
    # 1. Load cascades using the same logic as the Natural Range experiment
    print(f"Loading Weibo cascades (Natural Range logic)...")
    
    # We take the 101 cascades nearest to the mean (per updated script)
    cascades, pop_info = select_median_centred_cascades(
        min_size=MIN_SIZE,
        n_around=N_TOTAL_FOR_SELECTION
    )
    
    if not cascades:
        print("No cascades found or loaded.")
        return

    # Sort cascades by size
    cascades.sort(key=lambda c: c.size)
    
    # Pick 5 evenly spaced examples from the 101 selected
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
        c.network_name = f"Weibo Reposts"
        
        plot_cascade_tree(c, ax=ax)

        ax.set_title(
            f"Weibo Natural Range Example {i+1}\n"
            f"Source={c.source}, Size={c.size}, Depth={c.depth}",
            color="black", fontweight="bold", fontsize=11
        )

    fig.suptitle(
        f"Real-World Weibo Cascade Examples (size range {pop_info['selected_min']}–{pop_info['selected_max']}, "
        f"median={pop_info['selected_median']:.0f}, $n={len(selected_cascades)}$)",
        color="black", fontweight="bold", fontsize=18, y=1.02
    )
    plt.tight_layout()

    out_path = _REPO_ROOT / "validation/weibovalidation/figures/cascades/examples_weibo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Weibo examples -> {out_path}")


if __name__ == "__main__":
    main()
