#!/usr/bin/env python
"""
Plot IC on ER synthetic testset evaluation results as grouped bar charts across R0.
Uses consistent color mapping: RF variants are greener.

Usage: python scripts/experiments/plot_metrics_ic_er_bars.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.visualization.theme import ACCENT_COLORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = _REPO_ROOT / "results/metrics/testset_evaluation_ic_er.json"
OUT_DIR   = _REPO_ROOT / "results/figures/ml_evaluation"

# Consistent color mapping
METHOD_COLORS = {
    "random_forest": ACCENT_COLORS[5],  # Green
    "jordan":        ACCENT_COLORS[0],  # Salmon
    "closeness":     ACCENT_COLORS[1],  # Pink
    "degree":        ACCENT_COLORS[4],  # Gold
    "random":        ACCENT_COLORS[6],  # Periwinkle
}

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.is_file():
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH) as f:
        data = json.load(f)

    metrics_by_r0 = data["metrics_by_r0"]
    method_order  = data["method_order"]
    labels        = data["method_labels"]
    meta          = data.get("metadata", {})
    n_seeds       = meta.get("n_seeds", 5)

    r0_keys = sorted(metrics_by_r0.keys(), key=float)
    n_groups = len(r0_keys)

    KEEP = {"random_forest", "jordan", "closeness", "degree", "random"}
    present = [m for m in method_order if m in KEEP]
    n_methods = len(present)

    x = np.arange(n_groups)
    total_width = 0.8
    bar_width = total_width / n_methods
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_methods)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, title in zip(
        axes,
        ["top1", "top3"],
        [
            f"top-1 accuracy grouped bars -- IC on ER (over {n_seeds} seeds)",
            f"top-3 accuracy grouped bars -- IC on ER (over {n_seeds} seeds)",
        ],
    ):
        for xi, m in enumerate(present):
            vals = []
            for r0_val in r0_keys:
                met = metrics_by_r0[r0_val].get(m, {})
                k_mean = f"{key}_mean"
                if k_mean in met:
                    vals.append(met[k_mean])
                else:
                    k_old = "top_1" if key == "top1" else "top_3"
                    vals.append(met.get(k_old, 0.0))

            color = METHOD_COLORS.get(m, ACCENT_COLORS[9])
            ax.bar(x + offsets[xi], vals, width=bar_width, facecolor=color,
                   edgecolor="black", linewidth=0.5, label=labels.get(m, m))

        ax.set_xticks(x)
        ax.set_xticklabels([f"$R_0={k}$" for k in r0_keys], fontsize=10)
        ax.set_xlabel("Transmission Rate ($R_0$)")
        k = 1 if key == "top1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", axis="y", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out_file = OUT_DIR / "ic_er_accuracy_bars_vs_r0.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
