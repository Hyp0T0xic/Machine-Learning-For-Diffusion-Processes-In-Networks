#!/usr/bin/env python
"""
validation/plot_edge_distribution.py
=====================================
Plot the distribution of edge counts per cascade in the FalseNews dataset,
broken down by veracity (TRUE, FALSE, MIXED) and combined.

Usage
-----
    python validation/plot_edge_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = (
    _REPO_ROOT
    / "FalseNews_Code_Data"
    / "FalseNews_Code_Data"
    / "data"
    / "raw_data_anon.csv"
)
OUT_DIR = _REPO_ROOT / "validation/results/figures"


def load_edge_counts() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        usecols=["cascade_id", "veracity", "parent_tid"],
        dtype={"cascade_id": "int64", "parent_tid": "int64"},
        low_memory=False,
    )

    # normalise veracity — pandas may read True/False as booleans
    df["veracity"] = df["veracity"].astype(str).str.upper()
    df["veracity"] = df["veracity"].replace({"TRUE": "TRUE", "FALSE": "FALSE", "MIXED": "MIXED"})

    # edges = rows where the node has a parent inside the cascade
    cascade_tids = df.groupby("cascade_id")["cascade_id"].transform("count")  # unused, just for ref
    # build set of tids per cascade for parent-in-cascade check
    tid_sets = df.groupby("cascade_id").apply(
        lambda g: set(g.index)  # placeholder — use parent_tid != -1 as proxy
    )

    # for edge count: parent_tid != -1 means the node was infected by someone
    # (all such parents are in the cascade given the pre-filtering)
    grp = df.groupby(["cascade_id", "veracity"])
    stats = grp["parent_tid"].agg(
        n_nodes="count",
        n_edges=lambda s: (s != -1).sum(),
    ).reset_index()

    stats = stats[stats["n_edges"] > 1].reset_index(drop=True)
    return stats


def plot(stats: pd.DataFrame, out_dir: Path) -> None:
    veracity_colors = {
        "TRUE":  "#06d6a0",
        "FALSE": "#e63946",
        "MIXED": "#ffb703",
    }

    # shared bin edges across all groups
    max_edges = stats["n_edges"].max()
    bins = np.linspace(0, max_edges, 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        "Edge Count Distribution per Cascade — FalseNews Dataset",
        color="white", fontweight="bold", fontsize=13,
    )

    plot_order = [("ALL", "#a8dadc"), ("TRUE", "#06d6a0"), ("FALSE", "#e63946"), ("MIXED", "#ffb703")]

    for ax, (label, color) in zip(axes.flat, plot_order):
        ax.set_facecolor("#1a1a2e")

        if label == "ALL":
            data = stats["n_edges"].values
        else:
            data = stats.loc[stats["veracity"] == label, "n_edges"].values

        if len(data) == 0:
            ax.set_title(f"{label} (no data)", color="white")
            continue

        ax.hist(data, bins=bins, color=color, edgecolor="black", linewidth=0.4, alpha=0.85)

        median_val = np.median(data)
        mean_val   = np.mean(data)
        ax.axvline(median_val, color="white",  linewidth=1.2, linestyle="--", label=f"Median={median_val:.0f}")
        ax.axvline(mean_val,   color="#ffb703", linewidth=1.2, linestyle=":",  label=f"Mean={mean_val:.0f}")

        ax.set_title(
            f"{label}  (n={len(data):,} cascades)",
            color="white", fontweight="bold",
        )
        ax.set_xlabel("Edges per cascade", color="lightgray")
        ax.set_ylabel("Number of cascades", color="lightgray")
        ax.tick_params(colors="lightgray")
        ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = out_dir / "edge_distribution_falsenews.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def print_summary(stats: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print(f"  Edge count summary per veracity group")
    print(f"{'='*65}")
    print(f"{'Veracity':<10}  {'Cascades':>9}  {'Min':>6}  {'Median':>8}  "
          f"{'Mean':>8}  {'Max':>6}")
    print("-" * 65)

    for label in ["ALL", "TRUE", "FALSE", "MIXED"]:
        if label == "ALL":
            data = stats["n_edges"]
        else:
            data = stats.loc[stats["veracity"] == label, "n_edges"]
        if len(data) == 0:
            continue
        print(f"{label:<10}  {len(data):>9,}  {data.min():>6.0f}  "
              f"{data.median():>8.1f}  {data.mean():>8.1f}  {data.max():>6.0f}")

    print("=" * 65)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {CSV_PATH.name} ...")
    stats = load_edge_counts()
    print(f"  {len(stats):,} cascades loaded")
    print_summary(stats)
    plot(stats, OUT_DIR)


if __name__ == "__main__":
    main()
