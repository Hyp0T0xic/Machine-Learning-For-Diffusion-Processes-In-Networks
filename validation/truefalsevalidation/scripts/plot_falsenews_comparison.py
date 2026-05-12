#!/usr/bin/env python
"""
Plot FalseNews RF comparison and feature importances from saved JSON.
Reads: data/train_rf_falsenews.json
Writes: figures/falsenews_comparison.png
        figures/rf_feature_importance_falsenews.png

Usage: python validation/truefalsevalidation/plot_falsenews_comparison.py
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

DATA_DIR = _REPO_ROOT / "validation/truefalsevalidation/data"
OUT_DIR  = _REPO_ROOT / "validation/truefalsevalidation/figures"


def _plot_comparison(data: dict) -> None:
    metrics      = data["avg_metrics"]
    method_order = data["method_order"]
    labels       = data["method_labels"]
    target_size  = data["target_size"]

    KEEP = {"rf_falsenews", "RF (IC-BA)", "RF (IC-ER)", "jordan", "degree", "random"}
    present = [m for m in method_order if m in metrics and m in KEEP]
    x = np.arange(len(present))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, key, title in zip(axes, ["top_1", "top_3"], [
        f"top-1 accuracy -- FalseNews validation (size={target_size})",
        f"top-3 accuracy -- FalseNews validation (size={target_size})",
    ]):
        vals = [100 * metrics[m][key] for m in present]
        for xi, val in enumerate(vals):
            ax.bar(xi, val, facecolor=ACCENT_COLORS[xi % len(ACCENT_COLORS)],
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
    out_file = OUT_DIR / "falsenews_comparison.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def _plot_feature_importances(data: dict) -> None:
    avg_importances = data.get("avg_importances", {})
    if not avg_importances:
        return

    sorted_imp = sorted(avg_importances.items(), key=lambda x: x[1])
    features, scores = zip(*sorted_imp)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(features))
    bar_colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(features))]
    ax.barh(y_pos, scores, align="center", color=bar_colors,
            edgecolor="black", linewidth=0.7)
    for y, score in zip(y_pos, scores):
        ax.text(score + max(scores) * 0.01, y, f"{score:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel(f"mean decrease in impurity (avg over {data['n_seeds']} seeds)")
    ax.set_title(f"RF feature importances -- FalseNews (size={data['target_size']})")

    plt.tight_layout()
    out_file = OUT_DIR / "rf_feature_importance_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "train_rf_falsenews.json") as f:
        data = json.load(f)

    _plot_comparison(data)
    _plot_feature_importances(data)


if __name__ == "__main__":
    main()
