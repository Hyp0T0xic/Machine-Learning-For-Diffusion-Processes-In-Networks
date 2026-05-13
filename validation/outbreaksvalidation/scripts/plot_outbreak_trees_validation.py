#!/usr/bin/env python
"""
Plot OutbreakTrees real-world validation results from saved JSON.
Generates:
  1. outbreak_trees_accuracy.png (Bar charts)
  2. outbreak_trees_hops.png (Grouped bar chart)
  3. outbreak_trees_hops_cumulative.png (Cumulative line chart)

Reads:  data/validate_outbreak_trees.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.visualization.theme import ACCENT_COLORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = _REPO_ROOT / "validation/outbreaksvalidation/data"
OUT_DIR  = _REPO_ROOT / "validation/outbreaksvalidation/figures"

HOP_BINS  = ["0", "1", "2", "3", "4"]
HOP_LABEL = ["0", "1", "2", "3", "4+"]

# Consistent color mapping: RF (BA) is Green, RF (ER) is Teal
METHOD_COLORS = {
    "RF (IC-BA)":    ACCENT_COLORS[5],  # Green
    "RF (IC-ER)":    ACCENT_COLORS[2],  # Teal
    "jordan":        ACCENT_COLORS[0],  # Salmon
    "closeness":     ACCENT_COLORS[1],  # Pink
    "degree":        ACCENT_COLORS[4],  # Gold
    "random":        ACCENT_COLORS[6],  # Periwinkle
    "betweenness":   ACCENT_COLORS[8],  # Purple
}


def _accuracy_panel(ax, present, metrics, labels, mean_key, std_key, title):
    vals = [metrics[m][mean_key] for m in present]
    stds = [metrics[m][std_key]  for m in present]
    x = np.arange(len(present))
    for xi, (val, std) in enumerate(zip(vals, stds)):
        m = present[xi]
        color = METHOD_COLORS.get(m, ACCENT_COLORS[xi % len(ACCENT_COLORS)])
        ax.bar(xi, val, facecolor=color,
               edgecolor="black", linewidth=0.5, label=labels[m])
        lbl = f"{val:.1f}±{std:.1f}%" if std > 0.05 else f"{val:.1f}%"
        ax.text(xi, val + 0.5, lbl, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels[m] for m in present],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(fontsize=7, loc="upper left")


def _hop_panel(ax, present, hop_dist, labels, title):
    n_methods = len(present)
    bar_w = 0.8 / max(n_methods, 1)
    x = np.arange(len(HOP_BINS))
    for i, m in enumerate(present):
        pcts = hop_dist.get(m, {}).get("percentages", {})
        ys = [pcts.get(b, 0.0) for b in HOP_BINS]
        offset = (i - (n_methods - 1) / 2) * bar_w
        
        color = METHOD_COLORS.get(m, ACCENT_COLORS[i % len(ACCENT_COLORS)])
            
        ax.bar(x + offset, ys, bar_w,
               label=labels[m], facecolor=color,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(HOP_LABEL)
    ax.set_xlabel("hop distance (rank-1 prediction → true source)")
    ax.set_ylabel("% of cascades")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=True)


def _hop_cumulative_panel(ax, present, hop_dist, labels, title):
    x = np.arange(len(HOP_BINS))
    for i, m in enumerate(present):
        pcts = hop_dist.get(m, {}).get("percentages", {})
        ys = [pcts.get(b, 0.0) for b in HOP_BINS]
        y_cum = np.cumsum(ys)
        y_cum = np.minimum(y_cum, 100.0)
        
        color = METHOD_COLORS.get(m, ACCENT_COLORS[i % len(ACCENT_COLORS)])
        ax.plot(x, y_cum, marker="o", linewidth=2.5, markersize=7,
                label=labels[m], color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_LABEL)
    ax.set_xlabel("hop distance $H$")
    ax.set_ylabel(r"cumulative % of cascades ($\leq H$ hops)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right", frameon=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "validate_outbreak_trees.json"
    if not json_path.is_file():
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    metrics       = data["metrics"]
    hop_dist      = data.get("hop_distribution", {})
    method_order  = data["method_order"]
    labels        = data["method_labels"]
    n_cascades    = data.get("n_cascades", 75)
    n_seeds       = data.get("n_seeds", 5)
    sel_min       = data.get("min_cascade_size", 10)
    sel_max       = data.get("max_cascade_size", 300)

    # 1. Methods for Accuracy plots (exclude betweenness, closeness)
    EXCLUDE_ACC = {"betweenness", "closeness"}
    present_acc = [m for m in method_order if m in metrics and m.lower() not in EXCLUDE_ACC]

    # 2. Methods for Hop plots (exclude betweenness, closeness, degree, random, jordan)
    EXCLUDE_HOPS = {"betweenness", "closeness", "degree", "random", "jordan"}
    present_hops = [m for m in method_order if m in metrics and m.lower() not in EXCLUDE_HOPS]

    range_title = f"cascade size range {sel_min}–{sel_max}, $n={n_cascades}$"

    # 1. Plot Accuracy Bar Charts (1x2)
    plt.style.use("default")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    _accuracy_panel(
        axes1[0], present_acc, metrics, labels,
        "top1_mean", "top1_std",
        f"top-1 accuracy ({range_title})",
    )
    _accuracy_panel(
        axes1[1], present_acc, metrics, labels,
        "top3_mean", "top3_std",
        f"top-3 accuracy ({range_title})",
    )
    
    mean_hops_line = "Mean hops: " + "  ".join(
        f"{labels[m]}={metrics[m]['mean_hops']:.2f}" for m in present_acc
    )
    fig1.text(
        0.5, 0.02,
        f"Averaged over {n_seeds} seeds\n{mean_hops_line}",
        ha="center", fontsize=8, color="gray",
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    acc_out = OUT_DIR / "outbreak_trees_accuracy.png"
    fig1.savefig(acc_out, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved -> {acc_out}")

    # 2. Plot Hop Distance Distribution (Grouped Bars)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    _hop_panel(
        ax2, present_hops, hop_dist, labels,
        f"hop distance distribution ({range_title})",
    )
    plt.tight_layout()
    hop_out = OUT_DIR / "outbreak_trees_hops.png"
    fig2.savefig(hop_out, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {hop_out}")

    # 3. Plot Hop Distance Distribution (Cumulative Curve)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    _hop_cumulative_panel(
        ax3, present_hops, hop_dist, labels,
        f"cumulative hop distance -- OutbreakTrees ({range_title})",
    )
    plt.tight_layout()
    hop_cum_out = OUT_DIR / "outbreak_trees_hops_cumulative.png"
    fig3.savefig(hop_cum_out, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved -> {hop_cum_out}")


if __name__ == "__main__":
    main()
