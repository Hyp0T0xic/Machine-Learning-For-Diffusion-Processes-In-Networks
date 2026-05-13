#!/usr/bin/env python
"""
Plot FalseNews natural-range [42, 320] validation results from saved JSON.
Reads:  data/validate_falsenews_natural_range.json
Writes: figures/falsenews_natural_range.png

Usage: python validation/truefalsevalidation/scripts/plot_falsenews_natural_range.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR   = _REPO_ROOT / "validation/truefalsevalidation/data"
OUT_DIR    = _REPO_ROOT / "validation/truefalsevalidation/figures"

COLORS = [
    "#D99685", "#E38EA0", "#4DB6AC", "#9CB067", "#C0A064",
    "#7FB382", "#A3A1D8", "#DA92B7", "#C594D1", "#56B4BE",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "validate_falsenews_natural_range.json") as f:
        data = json.load(f)

    metrics      = data["metrics"]
    method_order = data["method_order"]
    labels       = data["method_labels"]
    n_cascades   = data["n_cascades"]
    n_seeds      = data["n_seeds"]
    min_size     = data["min_size"]
    max_size     = data["max_size"]
    size_mean    = data["size_mean"]
    size_median  = data["size_median"]

    present = [m for m in method_order if m in metrics]
    x = np.arange(len(present))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, key, std_key, title in zip(
        axes,
        ["top_1", "top_3"],
        ["top_1_std", "top_3_std"],
        [
            f"top-1 accuracy -- FalseNews natural range [{min_size}–{max_size}] nodes ($n={n_cascades}$)",
            f"top-3 accuracy -- FalseNews natural range [{min_size}–{max_size}] nodes ($n={n_cascades}$)",
        ],
    ):
        vals = [100 * metrics[m][key] for m in present]
        stds = [100 * metrics[m][std_key] for m in present]
        for xi, (val, std) in enumerate(zip(vals, stds)):
            ax.bar(xi, val, facecolor=COLORS[xi % len(COLORS)],
                   edgecolor="black", linewidth=0.5)
            lbl = f"{val:.1f}±{std:.1f}%" if std > 0.05 else f"{val:.1f}%"
            ax.text(xi, val + 0.5, lbl, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[m] for m in present],
            rotation=30, ha="right", fontsize=9,
        )
        k = 1 if key == "top_1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 115)

    fig.text(
        0.5, 0.01,
        f"Cascade sizes: mean={size_mean:.0f}, median={size_median:.0f} nodes  |  "
        f"averaged over {n_seeds} seeds",
        ha="center", fontsize=8, color="gray",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_file = OUT_DIR / "falsenews_natural_range.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
