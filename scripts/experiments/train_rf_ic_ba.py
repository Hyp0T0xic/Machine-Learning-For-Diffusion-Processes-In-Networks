#!/usr/bin/env python
"""
scripts/experiments/train_rf_ic_ba.py
=====================================
Train a Random Forest on node structural features to predict Patient Zero
in IC cascades on 200-node Barabasi-Albert graphs.

Runs across MULTIPLE network seeds and CASCADE SIZES, averaging the results
for statistical robustness.

Usage
-----
    python -m scripts.experiments.train_rf_ic_ba
"""
from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.data.cascade import r0_to_params, IndependentCascade, CascadeResult
from src.data.networks import generate_ba_network
from src.features.preprocess import filter_trivial
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Configuration -----------------------------------------------------------

N_NODES       = 200
BA_M          = 3
R0_VALUES     = [0.5, 1.0, 2.0, 3.0, 5.0]
CASCADE_SIZES = [30]         # run both sizes for comparison
N_TARGET      = 1500         # cascades to collect per R0
SEEDS         = [42, 43, 44]
OUT_DIR       = Path("results/figures/ml_evaluation")

METHOD_LABELS = {
    "random_forest": "Random Forest",
    "jordan":      "Jordan Centre",
    "closeness":   "Closeness",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "random":      "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


def generate_data(seed: int, cascade_size: int) -> tuple[list[CascadeResult], list[float]]:
    """Simulate IC cascades for a single seed and cascade size."""
    G = generate_ba_network(n=N_NODES, m=BA_M, seed=seed)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(seed)

    all_cascades: list[CascadeResult] = []
    cascade_r0s: list[float] = []

    print(f"    Generating {N_TARGET} cascades of size {cascade_size} per R0 ...")

    sim_seed = seed * 100_000
    for r0 in R0_VALUES:
        p = r0_to_params(r0, avg_deg, model="IC")["p"]
        model = IndependentCascade(p=p)

        collected = 0
        attempts = 0
        while collected < N_TARGET:
            source = rng.choice(nodes)
            cascade = model.run(G, source=source, seed=sim_seed, max_size=cascade_size)
            sim_seed += 1
            attempts += 1

            if cascade.size >= cascade_size:
                all_cascades.append(cascade)
                cascade_r0s.append(r0)
                collected += 1

        print(f"      R0={r0:.1f}  p={p:.6f}  collected {collected} "
              f"(attempts={attempts}, hit-rate={collected/attempts:.2%})")

    return all_cascades, cascade_r0s


def evaluate_random_baseline(results: list[CascadeResult], seed: int = 42) -> dict:
    """Evaluate random guessing by simulating rankings."""
    rng = random.Random(seed)
    random_rankings = []
    for r in results:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        random_rankings.append(nodes)
    return evaluate_ranker(results, random_rankings, ks=[1, 3])


def run_single_seed(seed: int, cascade_size: int) -> tuple[dict, dict]:
    """Run the full pipeline for one seed and cascade size."""
    # 1. Generate Data
    cascades, r0s = generate_data(seed, cascade_size)

    # 2. Extract Features
    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]

    # 3. Train/Test Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]

    # Train model
    rf = SourceRandomForest(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=5,
        max_features='log2',
        random_state=seed
    )
    rf.fit(X_train, y_train, feature_names)

    # Setup test set cascades
    test_cascade_indices = sorted(set(groups[i] for i in test_idx))
    test_cascades = [cascades[i] for i in test_cascade_indices]
    test_r0s = [r0s[i] for i in test_cascade_indices]

    # 4. Evaluate all methods
    metrics_by_r0: dict[float, dict[str, dict]] = defaultdict(dict)

    for eval_r0 in R0_VALUES:
        r0_indices = [i for i, r in enumerate(test_r0s) if r == eval_r0]
        if not r0_indices:
            continue

        subset_cascades = [test_cascades[i] for i in r0_indices]

        # Random Forest Rankings
        rf_rankings = [rf.rank_nodes(c) for c in subset_cascades]
        metrics_by_r0[eval_r0]["random_forest"] = evaluate_ranker(subset_cascades, rf_rankings, ks=[1, 3])

        # Baselines
        cols = defaultdict(list)
        for c in subset_cascades:
            preds = predict_all(c)
            for m_name, ranking in preds.items():
                cols[m_name].append(ranking)
        for m_name in cols:
            metrics_by_r0[eval_r0][m_name] = evaluate_ranker(subset_cascades, cols[m_name], ks=[1, 3])

        # Random
        metrics_by_r0[eval_r0]["random"] = evaluate_random_baseline(subset_cascades, seed=seed)

    return dict(metrics_by_r0), rf.feature_importances


def aggregate_metrics(all_seed_metrics: list[dict]) -> dict:
    """Average metrics across multiple seeds."""
    avg_metrics: dict[float, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for r0 in R0_VALUES:
        for method in METHOD_ORDER:
            t1_vals = []
            t3_vals = []
            for seed_metrics in all_seed_metrics:
                if r0 in seed_metrics and method in seed_metrics[r0]:
                    t1_vals.append(seed_metrics[r0][method]["top_k"][1])
                    t3_vals.append(seed_metrics[r0][method]["top_k"][3])
            if t1_vals:
                avg_metrics[r0][method] = {
                    "top_k": {1: np.mean(t1_vals), 3: np.mean(t3_vals)},
                    "top_k_std": {1: np.std(t1_vals), 3: np.std(t3_vals)},
                }
    return dict(avg_metrics)


def aggregate_importances(all_seed_importances: list[dict]) -> dict:
    """Average feature importances across seeds."""
    avg = defaultdict(float)
    for imp in all_seed_importances:
        for feat, val in imp.items():
            avg[feat] += val / len(all_seed_importances)
    return dict(avg)


def print_results(avg_metrics: dict, cascade_size: int) -> None:
    """Print averaged results table."""
    print(f"\n{'='*105}")
    print(f"  CASCADE SIZE = {cascade_size} | AVERAGED OVER {len(SEEDS)} SEEDS")
    print(f"{'='*105}")
    print(f"{'Method':<18}  " + "   ".join(f"R0={r:>3.1f}" for r in R0_VALUES))
    print(" " * 18 + "  " + "  ".join("Top1   Top3 " for _ in R0_VALUES))
    print("-" * 105)

    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<18}  "
        for r0 in R0_VALUES:
            m = avg_metrics.get(r0, {}).get(method)
            if m:
                t1 = 100 * m["top_k"][1]
                t3 = 100 * m["top_k"][3]
                row += f"{t1:>4.1f}% {t3:>4.1f}%  "
            else:
                row += "   -      -   "
        print(row)
    print("=" * 105)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cascade_size in CASCADE_SIZES:
        print(f"\n\n{'#'*80}")
        print(f"  RUNNING CASCADE SIZE = {cascade_size}")
        print(f"{'#'*80}")

        all_seed_metrics: list[dict] = []
        all_seed_importances: list[dict] = []

        for i, seed in enumerate(SEEDS):
            print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
            metrics, importances = run_single_seed(seed, cascade_size)
            all_seed_metrics.append(metrics)
            all_seed_importances.append(importances)

        avg_metrics = aggregate_metrics(all_seed_metrics)
        avg_importances = aggregate_importances(all_seed_importances)

        print_results(avg_metrics, cascade_size)

        _plot_accuracy(avg_metrics, cascade_size)
        _plot_feature_importances(avg_importances, cascade_size)


def _plot_accuracy(avg_metrics: dict, cascade_size: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#0d0d1a")

    palette = {
        "random_forest": "#ffb703",
        "jordan":       "#e63946",
        "closeness":    "#f4a261",
        "betweenness":  "#2ec4b6",
        "degree":       "#a8dadc",
        "random":       "#888888",
    }

    r0_list = [r for r in R0_VALUES if r in avg_metrics]
    x = np.arange(len(r0_list))
    bar_w = 0.12
    offsets = np.linspace(-(len(METHOD_ORDER) - 1) / 2 * bar_w,
                          (len(METHOD_ORDER) - 1) / 2 * bar_w,
                          len(METHOD_ORDER))

    for ax, k_measure, title in zip(axes, [1, 3],
            [f"Top-1 Accuracy (Mean over {len(SEEDS)} seeds, size={cascade_size})",
             f"Top-3 Accuracy (Mean over {len(SEEDS)} seeds, size={cascade_size})"]):
        ax.set_facecolor("#1a1a2e")
        for i, method in enumerate(METHOD_ORDER):
            vals = []
            errs = []
            for r0 in r0_list:
                m = avg_metrics.get(r0, {}).get(method)
                if m:
                    vals.append(100 * m["top_k"][k_measure])
                    errs.append(100 * m["top_k_std"][k_measure])
                else:
                    vals.append(0)
                    errs.append(0)
            ax.bar(x + offsets[i], vals, bar_w, yerr=errs,
                   label=METHOD_LABELS[method], color=palette[method],
                   edgecolor="black", linewidth=0.5, capsize=2,
                   error_kw={"ecolor": "white", "alpha": 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels([f"R0={r}" for r in r0_list], color="lightgray")
        ax.set_ylabel(f"Top-{k_measure} Accuracy (%)", color="lightgray")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.tick_params(colors="lightgray")
        if k_measure == 1:
            ax.legend(fontsize=9, facecolor="#222", edgecolor="#444", labelcolor="white",
                      loc="upper left", bbox_to_anchor=(1.02, 1))
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = OUT_DIR / f"rf_vs_baselines_ba_ic_size{cascade_size}.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot   -> {out_file}")


def _plot_feature_importances(avg_importances: dict, cascade_size: int) -> None:
    if not avg_importances:
        return

    sorted_imp = sorted(avg_importances.items(), key=lambda x: x[1])
    features, scores = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, align='center', color="#2ec4b6", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, color="lightgray")
    ax.set_xlabel(f"Mean Decrease in Impurity (Gini, avg over {len(SEEDS)} seeds)", color="lightgray")
    ax.set_title(f"Random Forest - Feature Importances (BA, size={cascade_size})",
                 color="white", fontweight="bold")
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = OUT_DIR / f"rf_feature_importance_ba_ic_size{cascade_size}.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved importance plot -> {out_file}")


if __name__ == "__main__":
    main()
