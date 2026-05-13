#!/usr/bin/env python
"""
Plot FalseNews natural-range (101 cascades around median) results.

Three panels:
  1. Top-1 accuracy per method
  2. Top-3 accuracy per method
  3. Hop-distance distribution per method (grouped bars, one bar per
     method per hop bin)

JSON layout expected (matches validate_existing_models2.py / validate_weibo.py
from commit 2b0b737):
  metrics:           per-method top1/top3/mrr (mean+std) and mean_hops
  hop_distribution:  per-method {counts, percentages}  — top-level

Reads:  data/validate_falsenews_natural_range_median.json
Writes: figures/falsenews_natural_range_median.png

Usage: python validation/truefalsevalidation/scripts/plot_falsenews_natural_range_median.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.visualization.theme import ACCENT_COLORS, METHOD_COLORS

DATA_DIR   = _REPO_ROOT / "validation/truefalsevalidation/data"
OUT_DIR    = _REPO_ROOT / "validation/truefalsevalidation/figures"

HOP_BINS  = ["0", "1", "2", "3", "4"]
HOP_LABEL = ["0", "1", "2", "3", "4+"]


def _accuracy_panel(ax, present, metrics, labels, mean_key, std_key, title):
    vals = [metrics[m][mean_key] for m in present]
    stds = [metrics[m][std_key]  for m in present]
    x = np.arange(len(present))
    for xi, (val, std) in enumerate(zip(vals, stds)):
        m = present[xi]
        color = METHOD_COLORS.get(m, ACCENT_COLORS[xi % len(ACCENT_COLORS)])
        ax.bar(xi, val, facecolor=color,
               edgecolor="black", linewidth=0.5)
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
    ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "validate_falsenews_natural_range_median.json") as f:
        data = json.load(f)

    metrics      = data["metrics"]
    hop_dist     = data.get("hop_distribution", {})
    method_order = data["method_order"]
    labels       = data["method_labels"]
    n_cascades   = data["n_cascades"]
    n_seeds      = data["n_seeds"]
    sel_min      = data.get("selected_min")
    sel_max      = data.get("selected_max")
    sel_median   = data.get("selected_median")
    sel_mean     = data.get("selected_mean")
    pop_median   = data.get("pop_size_median")
    pop_size     = data.get("pop_size")

    present = [m for m in method_order if m in metrics]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _accuracy_panel(
        axes[0], present, metrics, labels,
        "top1_mean", "top1_std",
        f"top-1 accuracy -- FalseNews natural range "
        f"[{sel_min}–{sel_max}] nodes ($n={n_cascades}$)",
    )
    _accuracy_panel(
        axes[1], present, metrics, labels,
        "top3_mean", "top3_std",
        f"top-3 accuracy -- FalseNews natural range "
        f"[{sel_min}–{sel_max}] nodes ($n={n_cascades}$)",
    )
    _hop_panel(
        axes[2], present, hop_dist, labels,
        "hop distance distribution -- FalseNews natural range  (per method)",
    )

    mean_hops_line = "Mean hops: " + "  ".join(
        f"{labels[m]}={metrics[m]['mean_hops']:.2f}" for m in present
    )
    fig.text(
        0.5, 0.025,
        f"Selected sizes: median={sel_median:.0f}, mean={sel_mean:.0f}  |  "
        f"population median={pop_median:.0f} over {pop_size:,} cascades  |  "
        f"averaged over {n_seeds} seeds\n{mean_hops_line}",
        ha="center", fontsize=8, color="gray",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_file = OUT_DIR / "falsenews_natural_range_median.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
