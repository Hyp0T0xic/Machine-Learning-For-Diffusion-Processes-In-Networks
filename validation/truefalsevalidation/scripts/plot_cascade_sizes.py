#!/usr/bin/env python
"""falsenews cascade-size histograms (log and linear), split by TRUE / FALSE / MIXED veracity"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH   = (
    _REPO_ROOT
    / "FalseNews_Code_Data" / "FalseNews_Code_Data" / "data" / "raw_data_anon.csv"
)
OUT_DIR = _REPO_ROOT / "validation/truefalsevalidation/figures"

COLORS = [
    "#D99685", "#E38EA0", "#4DB6AC", "#9CB067", "#C0A064",
    "#7FB382", "#A3A1D8", "#DA92B7", "#C594D1", "#56B4BE",
]

STUDY_MIN_SIZE = 25


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CSV_PATH.name} ...")
    df = pd.read_csv(
        CSV_PATH,
        usecols=["cascade_id", "veracity"],
        low_memory=False,
    )
    df["veracity"] = df["veracity"].astype(str).str.upper()

    sizes = df.groupby("cascade_id").agg(
        size=("veracity", "count"),
        veracity=("veracity", "first"),
    ).reset_index()

    total       = len(sizes)
    study_mask  = sizes["size"] >= STUDY_MIN_SIZE
    study_count = study_mask.sum()

    print(f"  Total cascades : {total:,}")
    print(f"  Size >= {STUDY_MIN_SIZE}    : {study_count:,} ({100*study_count/total:.1f}%)")
    print(f"  Median size    : {sizes['size'].median():.0f}")
    print(f"  Mean size      : {sizes['size'].mean():.1f}")
    print(f"  Max size       : {sizes['size'].max():,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cascade size distribution — FalseNews dataset", fontsize=13)

    # ── Left: full distribution (log-log) ─────────────────────────────────────
    ax = axes[0]
    log_bins = np.logspace(0, np.log10(sizes["size"].max()), 60)
    for (label, color) in [("FALSE", COLORS[0]), ("TRUE", COLORS[2]), ("MIXED", COLORS[4])]:
        data = sizes.loc[sizes["veracity"] == label, "size"].values
        ax.hist(data, bins=log_bins, facecolor=color, edgecolor="black",
                linewidth=0.4, alpha=0.85, label=f"{label} (n={len(data):,})")

    ax.axvline(STUDY_MIN_SIZE, color="black", linewidth=1.4, linestyle="--",
               label=f"study threshold (size={STUDY_MIN_SIZE})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cascade size (nodes, log scale)")
    ax.set_ylabel("number of cascades (log scale)")
    ax.set_title(f"All cascades ($n={total:,}$)")
    ax.legend(fontsize=8, frameon=True)

    # ── Right: study subset (size >= 25) ──────────────────────────────────────
    ax = axes[1]
    study = sizes[study_mask]
    log_bins2 = np.logspace(np.log10(STUDY_MIN_SIZE), np.log10(sizes["size"].max()), 50)
    for (label, color) in [("FALSE", COLORS[0]), ("TRUE", COLORS[2]), ("MIXED", COLORS[4])]:
        data = study.loc[study["veracity"] == label, "size"].values
        ax.hist(data, bins=log_bins2, facecolor=color, edgecolor="black",
                linewidth=0.4, alpha=0.85, label=f"{label} (n={len(data):,})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cascade size (nodes, log scale)")
    ax.set_ylabel("number of cascades (log scale)")
    ax.set_title(f"Study subset: size ≥ {STUDY_MIN_SIZE} ($n={study_count:,}$, {100*study_count/total:.1f}% of all)")
    ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    out_file = OUT_DIR / "cascade_size_distribution_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")

    # ── Linear-scale version ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cascade size distribution — FalseNews dataset (linear scale)", fontsize=13)

    ax = axes[0]
    lin_bins = np.linspace(1, sizes["size"].max(), 80)
    for (label, color) in [("FALSE", COLORS[0]), ("TRUE", COLORS[2]), ("MIXED", COLORS[4])]:
        data = sizes.loc[sizes["veracity"] == label, "size"].values
        ax.hist(data, bins=lin_bins, facecolor=color, edgecolor="black",
                linewidth=0.4, alpha=0.85, label=f"{label} (n={len(data):,})")
    ax.axvline(STUDY_MIN_SIZE, color="black", linewidth=1.4, linestyle="--",
               label=f"study threshold (size={STUDY_MIN_SIZE})")
    ax.set_xlabel("cascade size (nodes)")
    ax.set_ylabel("number of cascades")
    ax.set_title(f"All cascades ($n={total:,}$)")
    ax.legend(fontsize=8, frameon=True)

    ax = axes[1]
    lin_bins2 = np.linspace(STUDY_MIN_SIZE, sizes["size"].max(), 80)
    for (label, color) in [("FALSE", COLORS[0]), ("TRUE", COLORS[2]), ("MIXED", COLORS[4])]:
        data = study.loc[study["veracity"] == label, "size"].values
        ax.hist(data, bins=lin_bins2, facecolor=color, edgecolor="black",
                linewidth=0.4, alpha=0.85, label=f"{label} (n={len(data):,})")
    ax.set_xlabel("cascade size (nodes)")
    ax.set_ylabel("number of cascades")
    ax.set_title(f"Study subset: size ≥ {STUDY_MIN_SIZE} ($n={study_count:,}$, {100*study_count/total:.1f}% of all)")
    ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    out_file_lin = OUT_DIR / "cascade_size_distribution_falsenews_linear.png"
    fig.savefig(out_file_lin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file_lin}")


if __name__ == "__main__":
    main()
