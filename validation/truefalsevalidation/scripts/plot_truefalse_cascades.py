#!/usr/bin/env python
"""
validation/truefalsevalidation/scripts/plot_truefalse_cascades.py
Show example cascades for real FalseNews (True/False) diffusion trees.
Updated to use a clean white background, 5-example layout, and standardized titles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Add local scripts to path
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.visualization.cascades import plot_cascade_tree

DATA_DIR = _REPO_ROOT / "validation/truefalsevalidation/data"
N_EXAMPLES = 5

def main() -> None:
    # 1. Load cascades
    print(f"Loading FalseNews cascades...")
    # NOTE: Since raw data is unavailable on this machine, we skip loading
    # In a real environment, you'd use load_falsenews_cascades()
    try:
        from load_falsenews import load_falsenews_cascades
        cascades, _ = load_falsenews_cascades(target_size=None) 
    except Exception as e:
        print(f"Skipping cascade loading (raw data missing): {e}")
        cascades = []

    # Try to get metadata from JSON for the title
    json_path = DATA_DIR / "validate_falsenews_natural_range_median.json"
    pop_info = {}
    if json_path.is_file():
        with open(json_path, "r") as f:
            pop_info = json.load(f)
    
    sel_min = pop_info.get("selected_min", "88")
    sel_max = pop_info.get("selected_max", "92")
    sel_med = pop_info.get("selected_median", 90.0)
    n_casc   = pop_info.get("n_cascades", 101)

    if not cascades:
        print("No cascades found. Visualization skipped.")
        return

    # Sort cascades by size
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
        c.network_name = f"TrueFalse News"
        
        plot_cascade_tree(c, ax=ax)

        ax.set_title(
            f"True/False Example {i+1}\n"
            f"Source={c.source}, Size={c.size}, Depth={c.depth}",
            color="black", fontweight="bold", fontsize=11
        )

    fig.suptitle(
        f"Real-World FalseNews Cascade Examples (size range {sel_min}–{sel_max}, "
        f"median={sel_med:.0f}, $n={n_casc}$)",
        color="black", fontweight="bold", fontsize=18, y=1.02
    )
    plt.tight_layout()

    out_path = _REPO_ROOT / "validation/truefalsevalidation/figures/cascades/examples_truefalse.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved TrueFalse examples -> {out_path}")


if __name__ == "__main__":
    main()
