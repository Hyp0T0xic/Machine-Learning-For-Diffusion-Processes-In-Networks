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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = _REPO_ROOT / "validation/truefalsevalidation/data"
OUT_DIR    = _REPO_ROOT / "validation/truefalsevalidation/figures"

COLORS = ["#082a54", "#a559aa", "#59a89c", "#f0c571", "#e02b35", "#cecece"]

R0_BINS = [(0.0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, float("inf"))]


def _plot_r0_distribution(data: dict) -> None:
    r0_values   = data["r0_values"]
    r0_labels   = data["r0_labels"]
    target_size = data["target_size"]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(r0_values, bins=40, facecolor=COLORS[0], edgecolor="black", linewidth=0.5)

    for (lo, _), label in zip(R0_BINS[1:], r0_labels[1:]):
        ax.axvline(lo, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xlabel("estimated $R_0$ (mean secondary infections per spreading node)")
    ax.set_ylabel("number of cascades")
    ax.set_title(f"$R_0$ distribution -- FalseNews cascades ($n={len(r0_values)}$, size={target_size})")

    x_prev = 0
    for (lo, hi), label in zip(R0_BINS, r0_labels):
        x_mid = (x_prev + min(hi, max(r0_values))) / 2
        count = sum(1 for r in r0_values if lo <= r < hi)
        ax.text(x_mid, ax.get_ylim()[1] * 0.92, f"{label}\n$n={count}$",
                ha="center", va="top", fontsize=8)
        x_prev = lo

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
