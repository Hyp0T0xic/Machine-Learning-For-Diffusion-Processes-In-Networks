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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
OUT_DIR = _REPO_ROOT / "validation/truefalsevalidation/figures"


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
    plt.style.use("default")
    max_edges = stats["n_edges"].max()
    bins = np.linspace(0, max_edges, 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Edge count distribution per cascade — FalseNews dataset", fontsize=13)

    # different gray shades to distinguish veracity groups in b&w
    plot_order = ["ALL", "TRUE", "FALSE", "MIXED"]

    for ax, label in zip(axes.flat, plot_order):
        if label == "ALL":
            data = stats["n_edges"].values
        else:
            data = stats.loc[stats["veracity"] == label, "n_edges"].values

        if len(data) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        ax.hist(data, bins=bins, facecolor="white", edgecolor="black", linewidth=0.4)

        median_val = np.median(data)
        mean_val   = np.mean(data)
        ax.axvline(median_val, color="black", linewidth=1.2, linestyle="--", label=f"median={median_val:.0f}")
        ax.axvline(mean_val,   color="black", linewidth=1.2, linestyle=":",  label=f"mean={mean_val:.0f}")

        ax.set_title(f"{label}  ($n={len(data):,}$ cascades)")
        ax.set_xlabel("edges per cascade")
        ax.set_ylabel("number of cascades")
        ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    out_file = out_dir / "edge_distribution_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
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
