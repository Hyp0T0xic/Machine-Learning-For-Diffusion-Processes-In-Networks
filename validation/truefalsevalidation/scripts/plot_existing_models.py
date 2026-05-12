#!/usr/bin/env python
"""
Plot pre-trained model accuracy on FalseNews from saved JSON.
Reads: data/validate_existing_models.json
Writes: figures/existing_models_on_falsenews.png

Usage: python validation/truefalsevalidation/plot_existing_models.py
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

    with open(DATA_DIR / "validate_existing_models.json") as f:
        data = json.load(f)

    metrics      = data["metrics"]
    method_order = data["method_order"]
    labels       = data["method_labels"]
    n_cascades   = data["n_cascades"]

    KEEP = {"RF (IC-BA)", "RF (IC-ER)", "jordan", "degree", "random"}
    present = [m for m in method_order if m in metrics and m in KEEP]
    x = np.arange(len(present))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, key, title in zip(axes, ["top_1", "top_3"], [
        f"top-1 accuracy -- existing models on FalseNews ($n={n_cascades}$)",
        f"top-3 accuracy -- existing models on FalseNews ($n={n_cascades}$)",
    ]):
        vals = [100 * metrics[m][key] for m in present]
        for xi, val in enumerate(vals):
            ax.bar(xi, val, facecolor=COLORS[xi % len(COLORS)],
                   edgecolor="black", linewidth=0.5)
            ax.text(xi, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[m] for m in present],
            rotation=30, ha="right", fontsize=9,
        )
        k = 1 if key == "top_1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out_file = OUT_DIR / "existing_models_on_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
