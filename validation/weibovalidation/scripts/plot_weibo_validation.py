#!/usr/bin/env python
"""
Plot Weibo real-world validation results from saved JSON.
Reads: data/validate_weibo.json
Writes: figures/weibo_validation_large.png

Usage: python validation/weibovalidation/scripts/plot_weibo_validation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR   = _REPO_ROOT / "validation/weibovalidation/data"
OUT_DIR    = _REPO_ROOT / "validation/weibovalidation/figures"

COLORS = ["#082a54", "#a559aa", "#59a89c", "#f0c571", "#e02b35", "#cecece"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "validate_weibo.json") as f:
        data = json.load(f)

    metrics      = data["metrics"]
    method_order = data["method_order"]
    labels       = data["method_labels"]
    n_cascades   = data["n_cascades"]
    n_seeds      = data["n_seeds"]

    present = [m for m in method_order if m in metrics]
    x = np.arange(len(present))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, key, std_key, title in zip(
        axes,
        ["top_1", "top_3"],
        ["top_1_std", "top_3_std"],
        [
            f"top-1 accuracy -- Weibo real-world validation ($n={n_cascades}$)",
            f"top-3 accuracy -- Weibo real-world validation ($n={n_cascades}$)",
        ],
    ):
        vals = [metrics[m][key] for m in present]
        stds = [metrics[m][std_key] for m in present]
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
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out_file = OUT_DIR / "weibo_validation_large.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
