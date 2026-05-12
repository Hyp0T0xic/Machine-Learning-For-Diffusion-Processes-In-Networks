#!/usr/bin/env python
"""
Plot R0 distribution and accuracy-by-R0 from saved JSON.
Reads: data/r0_analysis_falsenews.json
Writes: figures/r0_distribution_falsenews.png
        figures/r0_vs_accuracy_falsenews.png

Usage: python validation/truefalsevalidation/plot_r0_analysis.py
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

R0_BINS = [(0.0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, float("inf"))]


def _plot_r0_distribution(data: dict) -> None:
    r0_values   = data["r0_values"]
    target_size = data["target_size"]

    int_bins   = [1, 2, 3, 4, 5, float("inf")]
    bin_labels = ["1-2", "2-3", "3-4", "4-5", "5+"]
    counts = [
        sum(1 for r in r0_values if lo <= r < hi)
        for lo, hi in zip(int_bins[:-1], int_bins[1:])
    ]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    x    = np.arange(len(bin_labels))
    bars = ax.bar(x, counts, width=0.6,
                  color=COLORS[:len(bin_labels)], edgecolor="black", linewidth=0.5)

    total = len(r0_values)
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.01,
            f"$n={count}$\n{pct:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$R_0 \\in [{lbl})$" for lbl in bin_labels])
    ax.set_xlabel("estimated $R_0$ bin (mean secondary infections per spreading node)")
    ax.set_ylabel("number of cascades")
    ax.set_title(f"$R_0$ distribution -- FalseNews cascades ($n={len(r0_values)}$, size={target_size})")

    plt.tight_layout()
    out_file = OUT_DIR / "r0_distribution_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def _plot_accuracy_by_r0(data: dict) -> None:
    avg_metrics  = data["avg_metrics"]
    r0_labels    = data["r0_labels"]
    method_order = data["method_order"]
    labels       = data["method_labels"]
    n_seeds      = data["n_seeds"]

    active_labels = [lbl for lbl in r0_labels if avg_metrics.get(lbl)]
    if not active_labels:
        print("No data to plot.")
        return

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    x     = np.arange(len(active_labels))
    bar_w = 0.12
    n_m   = len(method_order)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_w, (n_m - 1) / 2 * bar_w, n_m)

    for ax, key, title in zip(axes, ["top_1", "top_3"], [
        f"top-1 accuracy -- FalseNews by estimated $R_0$ (mean over {n_seeds} seeds)",
        f"top-3 accuracy -- FalseNews by estimated $R_0$ (mean over {n_seeds} seeds)",
    ]):
        for i, method in enumerate(method_order):
            vals = []
            for label in active_labels:
                m = avg_metrics.get(label, {}).get(method)
                vals.append(100 * m[key] if m else 0)

            ax.bar(
                x + offsets[i], vals, bar_w,
                label=labels[method],
                facecolor=COLORS[i % len(COLORS)],
                edgecolor="black", linewidth=0.5,
            )
            for xi, val in enumerate(vals):
                if val > 0:
                    ax.text(x[xi] + offsets[i], val + 0.5, f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(active_labels)
        k = 1 if key == "top_1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        if key == "top_1":
            ax.legend(fontsize=9, frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    out_file = OUT_DIR / "r0_vs_accuracy_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "r0_analysis_falsenews.json") as f:
        data = json.load(f)

    _plot_r0_distribution(data)
    _plot_accuracy_by_r0(data)


if __name__ == "__main__":
    main()
