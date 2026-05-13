#!/usr/bin/env python
"""
Plot Weibo real-world validation results from saved JSON.
Uses consistent color mapping: RF variants are greener.

Usage: python validation/weibovalidation/scripts/plot_weibo_validation.py
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

DATA_DIR   = _REPO_ROOT / "validation/weibovalidation/data"
OUT_DIR    = _REPO_ROOT / "validation/weibovalidation/figures"

# Consistent color mapping: RF (BA) is Green, RF (ER) is Olive/Teal
METHOD_COLORS = {
    "RF (IC-BA)":    ACCENT_COLORS[5],  # Green
    "RF (IC-ER)":    ACCENT_COLORS[2],  # Teal
    "Jordan Center": ACCENT_COLORS[0],  # Salmon
    "closeness":     ACCENT_COLORS[1],  # Pink
    "betweenness":   ACCENT_COLORS[8],  # Purple
    "Degree":        ACCENT_COLORS[4],  # Gold
    "random":        ACCENT_COLORS[6],  # Periwinkle
}

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "validate_weibo.json"
    if not json_path.is_file():
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    metrics    = data["metrics"]
    labels_map = data.get("method_labels", {})
    n_cascades = data.get("n_cascades", "N/A")

    canonical_methods = [
        "RF (IC-BA)",
        "RF (IC-ER)",
        "Jordan Center",
        "closeness",
        "Degree",
        "random",
    ]

    default_labels = {
        "RF (IC-BA)": "Random Forest (IC-BA)",
        "RF (IC-ER)": "Random Forest (IC-ER)",
        "Jordan Center": "Jordan Centre",
        "closeness": "Closeness",
        "Degree": "Degree",
        "random": "Random",
    }

    x = np.arange(len(canonical_methods))

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, key, title in zip(
        axes,
        ["top_1", "top_3"],
        [
            f"top-1 accuracy -- Weibo real-world validation ($n={n_cascades}$)",
            f"top-3 accuracy -- Weibo real-world validation ($n={n_cascades}$)",
        ],
    ):
        vals = []
        stds = []
        for m in canonical_methods:
            m_metrics = metrics.get(m, metrics.get(m.lower(), {}))
            
            k_target = "top1_mean" if key == "top_1" else "top3_mean"
            k_std_target = "top1_std" if key == "top_1" else "top3_std"
            k_old_std = f"{key}_std"

            if k_target in m_metrics:
                vals.append(m_metrics[k_target])
                stds.append(m_metrics.get(k_std_target, 0.0))
            else:
                vals.append(m_metrics.get(key, 0.0))
                stds.append(m_metrics.get(k_old_std, 0.0))

        for xi, (val, std) in enumerate(zip(vals, stds)):
            m = canonical_methods[xi]
            color = METHOD_COLORS.get(m, ACCENT_COLORS[9])
            ax.bar(xi, val, facecolor=color, edgecolor="black", linewidth=0.5)
            if val > 0:
                lbl = f"{val:.1f}±{std:.1f}%" if std > 0.05 else f"{val:.1f}%"
                ax.text(xi, val + 0.5, lbl, ha="center", va="bottom", fontsize=8)
            else:
                ax.text(xi, 1.0, "N/A", ha="center", va="bottom", fontsize=7, color="gray")

        ax.set_xticks(x)
        display_labels = [labels_map.get(m, default_labels.get(m, m)) for m in canonical_methods]
        ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=9)
        
        k = 1 if key == "top_1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5, axis='y')

    plt.tight_layout()
    out_file = OUT_DIR / "weibo_validation_large.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
