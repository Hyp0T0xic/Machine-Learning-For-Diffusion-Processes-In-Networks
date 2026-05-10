#!/usr/bin/env python
"""
validation/r0_analysis_falsenews.py
=====================================
Analyse the empirical R0 distribution of the FalseNews validation cascades,
then compare source-identification accuracy across R0 bins for the
FalseNews-trained RF and all centrality baselines.

Output plots are styled to match rf_vs_baselines_er_ic_size25.png:
  - r0_distribution_falsenews.png   : histogram of estimated R0
  - r0_vs_accuracy_falsenews.png    : grouped-bar accuracy by R0 bin
                                       (same 2-panel top-1 / top-3 layout)

Usage
-----
    python validation/r0_analysis_falsenews.py
"""
from __future__ import annotations

import sys
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from validation.load_falsenews import load_falsenews_cascades
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Config ------------------------------------------------------------------

TARGET_SIZE = 25
SEEDS       = [42, 123, 456, 789, 1024]
OUT_DIR     = _REPO_ROOT / "validation/results/figures"

# Pre-filtered CSV: all cascades already have >= 25 reachable nodes
CSV_PATH = (
    _REPO_ROOT
    / "FalseNews_Code_Data"
    / "FalseNews_Code_Data"
    / "data"
    / "raw_data_anon_filtered_min25_reachable.csv"
)

# R0 bins mirroring simulated values 0.5, 1, 2, 3, 5
R0_BINS   = [(0.0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, float("inf"))]
R0_LABELS = ["R0 ≈ 0.5", "R0 ≈ 1", "R0 ≈ 2", "R0 ≈ 3", "R0 ≈ 5"]

METHOD_LABELS = {
    "rf_falsenews": "RF (FalseNews)",
    "jordan":       "Jordan Centre",
    "closeness":    "Closeness",
    "betweenness":  "Betweenness",
    "degree":       "Degree",
    "random":       "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())

PALETTE = {
    "rf_falsenews": "#06d6a0",
    "jordan":       "#e63946",
    "closeness":    "#f4a261",
    "betweenness":  "#2ec4b6",
    "degree":       "#a8dadc",
    "random":       "#888888",
}


# -- Helpers -----------------------------------------------------------------

def _bin_label(r0: float) -> str | None:
    for (lo, hi), label in zip(R0_BINS, R0_LABELS):
        if lo <= r0 < hi:
            return label
    return None


def _evaluate_random(results, seed=42):
    rng = random.Random(seed)
    rankings = []
    for r in results:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        rankings.append(nodes)
    return evaluate_ranker(results, rankings, ks=[1, 3])


# -- Core pipeline -----------------------------------------------------------

def _run_seed(cascades, r0_per_cascade, X, y, groups, feature_names, seed):
    """Train RF on 80 %, evaluate all methods on 20 %, broken down by R0 bin."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    rf = SourceRandomForest(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=5,
        max_features=3,
        random_state=seed,
    )
    rf.fit(X[train_idx], y[train_idx], feature_names)

    test_cascade_indices = sorted(set(groups[i] for i in test_idx))
    test_cascades  = [cascades[i] for i in test_cascade_indices]
    test_r0s       = [r0_per_cascade[i] for i in test_cascade_indices]

    # group test cascades by R0 bin
    bin_cascades: dict[str, list] = defaultdict(list)
    for c, r0 in zip(test_cascades, test_r0s):
        label = _bin_label(r0)
        if label is not None:
            bin_cascades[label].append(c)

    metrics_by_bin: dict[str, dict[str, dict]] = {}

    for label in R0_LABELS:
        subset = bin_cascades.get(label, [])
        if not subset:
            continue

        bin_metrics: dict[str, dict] = {}

        rf_rankings = [rf.rank_nodes(c) for c in subset]
        bin_metrics["rf_falsenews"] = evaluate_ranker(subset, rf_rankings, ks=[1, 3])

        cols: dict[str, list] = defaultdict(list)
        for c in subset:
            for m_name, ranking in predict_all(c).items():
                cols[m_name].append(ranking)
        for m_name, rankings in cols.items():
            bin_metrics[m_name] = evaluate_ranker(subset, rankings, ks=[1, 3])

        bin_metrics["random"] = _evaluate_random(subset, seed=seed)

        metrics_by_bin[label] = bin_metrics

    return metrics_by_bin, rf.feature_importances


def _aggregate(all_seed_metrics):
    """Average top-1 / top-3 across seeds per (R0 bin, method)."""
    avg: dict[str, dict[str, dict]] = {}
    for label in R0_LABELS:
        avg[label] = {}
        for method in METHOD_ORDER:
            t1_vals, t3_vals = [], []
            for sm in all_seed_metrics:
                if label in sm and method in sm[label]:
                    t1_vals.append(sm[label][method]["top_k"][1])
                    t3_vals.append(sm[label][method]["top_k"][3])
            if t1_vals:
                avg[label][method] = {
                    "top_k":     {1: np.mean(t1_vals), 3: np.mean(t3_vals)},
                    "top_k_std": {1: np.std(t1_vals),  3: np.std(t3_vals)},
                }
    return avg


def _print_results(avg_metrics, bin_counts):
    print(f"\n{'='*100}")
    print(f"  FALSENEWS — ACCURACY BY R0 BIN  |  SIZE = {TARGET_SIZE}  |  "
          f"AVERAGED OVER {len(SEEDS)} SEEDS")
    print(f"{'='*100}")
    header = f"{'Method':<20}  " + "   ".join(
        f"{lbl:<12}" for lbl in R0_LABELS
    )
    print(header)
    sub = " " * 22 + "  ".join("Top1   Top3 " for _ in R0_LABELS)
    print(sub)
    print("-" * 100)
    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<20}  "
        for label in R0_LABELS:
            m = avg_metrics.get(label, {}).get(method)
            if m:
                t1 = 100 * m["top_k"][1]
                t3 = 100 * m["top_k"][3]
                row += f"{t1:>4.1f}% {t3:>4.1f}%  "
            else:
                row += "  -      -    "
        print(row)
    print("=" * 100)
    print("\n  Cascade counts per R0 bin (full dataset):")
    for label, n in bin_counts.items():
        print(f"    {label:<15}  n={n}")


# -- Plots -------------------------------------------------------------------

def _plot_r0_distribution(r0_values, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    ax.hist(r0_values, bins=40, color="#06d6a0", edgecolor="black", linewidth=0.5)

    # overlay bin boundaries
    for (lo, _), label in zip(R0_BINS[1:], R0_LABELS[1:]):
        ax.axvline(lo, color="#ffb703", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xlabel("Estimated R0 (mean secondary infections per spreading node)",
                  color="lightgray")
    ax.set_ylabel("Number of cascades", color="lightgray")
    ax.set_title(
        f"R0 Distribution — FalseNews Cascades (n={len(r0_values)}, size={TARGET_SIZE})",
        color="white", fontweight="bold",
    )
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # annotate bins
    x_prev = 0
    for (lo, hi), label in zip(R0_BINS, R0_LABELS):
        x_mid = (x_prev + min(hi, max(r0_values))) / 2
        count = sum(1 for r in r0_values if lo <= r < hi)
        ax.text(x_mid, ax.get_ylim()[1] * 0.92, f"{label}\nn={count}",
                ha="center", va="top", color="lightgray", fontsize=8)
        x_prev = lo

    plt.tight_layout()
    out_file = out_dir / "r0_distribution_falsenews.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R0 distribution plot -> {out_file}")
    return out_file


def _plot_accuracy_by_r0(avg_metrics, out_dir):
    # only include bins that have data for at least one method
    active_labels = [lbl for lbl in R0_LABELS if avg_metrics.get(lbl)]
    if not active_labels:
        print("No data to plot.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#0d0d1a")

    x      = np.arange(len(active_labels))
    bar_w  = 0.12
    n_m    = len(METHOD_ORDER)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_w, (n_m - 1) / 2 * bar_w, n_m)

    for ax, k_measure, title in zip(axes, [1, 3], [
        f"Top-1 Accuracy — FalseNews by Estimated R0 "
        f"(mean over {len(SEEDS)} seeds, size={TARGET_SIZE})",
        f"Top-3 Accuracy — FalseNews by Estimated R0 "
        f"(mean over {len(SEEDS)} seeds, size={TARGET_SIZE})",
    ]):
        ax.set_facecolor("#1a1a2e")

        for i, method in enumerate(METHOD_ORDER):
            vals, errs = [], []
            for label in active_labels:
                m = avg_metrics.get(label, {}).get(method)
                if m:
                    vals.append(100 * m["top_k"][k_measure])
                    errs.append(100 * m["top_k_std"][k_measure])
                else:
                    vals.append(0)
                    errs.append(0)

            ax.bar(
                x + offsets[i], vals, bar_w,
                yerr=errs,
                label=METHOD_LABELS[method],
                color=PALETTE[method],
                edgecolor="black", linewidth=0.5,
                capsize=2,
                error_kw={"ecolor": "white", "alpha": 0.6},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(active_labels, color="lightgray")
        ax.set_ylabel(f"Top-{k_measure} Accuracy (%)", color="lightgray")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.tick_params(colors="lightgray")
        if k_measure == 1:
            ax.legend(
                fontsize=9, facecolor="#222", edgecolor="#444",
                labelcolor="white", loc="upper left",
                bbox_to_anchor=(1.02, 1),
            )
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = out_dir / "r0_vs_accuracy_falsenews.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy-by-R0 plot -> {out_file}")
    return out_file


# -- Entry point -------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # min_size=1 because the CSV is pre-filtered to >= 25 reachable nodes
    cascades, _ = load_falsenews_cascades(
        csv_path=CSV_PATH, target_size=TARGET_SIZE, min_size=1,
    )
    print(f"\nLoaded {len(cascades)} FalseNews cascades\n")

    # --- R0 distribution ----------------------------------------------------
    r0_per_cascade = [c.actual_r0() for c in cascades]

    print("  R0 summary statistics:")
    r0_arr = np.array(r0_per_cascade)
    print(f"    min={r0_arr.min():.2f}  max={r0_arr.max():.2f}  "
          f"mean={r0_arr.mean():.2f}  median={np.median(r0_arr):.2f}  "
          f"std={r0_arr.std():.2f}")

    bin_counts = {}
    for (lo, hi), label in zip(R0_BINS, R0_LABELS):
        bin_counts[label] = int(np.sum((r0_arr >= lo) & (r0_arr < hi)))
    print("\n  Cascade counts per R0 bin:")
    for label, n in bin_counts.items():
        print(f"    {label:<15}  n={n}")

    _plot_r0_distribution(r0_per_cascade, OUT_DIR)

    # --- Accuracy by R0 bin -------------------------------------------------
    print(f"\nRunning {len(SEEDS)}-seed cross-validation ...")

    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]

    all_seed_metrics: list[dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        metrics, _ = _run_seed(
            cascades, r0_per_cascade, X, y, groups, feature_names, seed,
        )
        all_seed_metrics.append(metrics)

    avg_metrics = _aggregate(all_seed_metrics)
    _print_results(avg_metrics, bin_counts)
    _plot_accuracy_by_r0(avg_metrics, OUT_DIR)


if __name__ == "__main__":
    main()
