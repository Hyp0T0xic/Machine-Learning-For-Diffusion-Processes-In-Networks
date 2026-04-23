#!/usr/bin/env python
"""
scripts/train_rf_ic_ba.py
================================
Train a Random Forest on node structural features to predict Patient Zero
in IC cascades on a 200-node Barabási–Albert graph.

Evaluates the RF against standard centrality baselines and plots the results.

Usage
-----
    python -m scripts.train_rf_ic_ba
"""
from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.cascade import r0_to_params, IndependentCascade, CascadeResult
from src.data.networks import generate_ba_network
from src.features.preprocess import filter_trivial
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# ── Configuration ─────────────────────────────────────────────────────────────

N_NODES      = 200
BA_M         = 3
R0_VALUES    = [0.5, 1.0, 2.0, 3.0, 5.0]
CASCADE_SIZE = 20        # exact cascade size to collect
N_TARGET     = 500       # cascades to collect per R0
BASE_SEED    = 42
OUT_DIR      = Path("results/figures/ml_evaluation")

METHOD_LABELS = {
    "random_forest": "Random Forest",
    "jordan":      "Jordan Centre",
    "closeness":   "Closeness",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "random":      "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


def generate_data() -> tuple[list[CascadeResult], list[float]]:
    """Simulate IC cascades, collecting exactly N_TARGET with size == CASCADE_SIZE per R₀.

    Each simulation is stopped early once CASCADE_SIZE infected nodes are
    reached.  Cascades that die out before hitting the target are discarded,
    so lower R₀ values require more attempts.
    """
    G = generate_ba_network(n=N_NODES, m=BA_M, seed=BASE_SEED)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(BASE_SEED)

    all_cascades: list[CascadeResult] = []
    cascade_r0s: list[float] = []

    print(f"Generating cascades: Barabasi-Albert (N={N_NODES}, m={BA_M}), "
          f"{N_TARGET} cascades of size {CASCADE_SIZE} per R0 ...")
    print(f"Graph properties: N={G.number_of_nodes()}, edges={G.number_of_edges()}, avg_deg={avg_deg:.2f}")

    seed = BASE_SEED
    for r0 in R0_VALUES:
        p = r0_to_params(r0, avg_deg, model="IC")["p"]
        model = IndependentCascade(p=p)

        collected = 0
        attempts = 0
        while collected < N_TARGET:
            source = rng.choice(nodes)
            cascade = model.run(G, source=source, seed=seed, max_size=CASCADE_SIZE)
            seed += 1
            attempts += 1

            if cascade.size >= CASCADE_SIZE:
                all_cascades.append(cascade)
                cascade_r0s.append(r0)
                collected += 1

        print(f"  R0={r0:.1f}  p={p:.6f}  collected {collected}/{N_TARGET} "
              f"(attempts={attempts}, hit-rate={collected/attempts:.2%})")

    return all_cascades, cascade_r0s


def evaluate_random_baseline(results: list[CascadeResult], seed: int = 42) -> dict:
    """Evaluate random guessing by simulating rankings."""
    rng = random.Random(seed)
    # A random ranking is just shuffling the infected nodes
    random_rankings = []
    for r in results:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        random_rankings.append(nodes)
        
    return evaluate_ranker(results, random_rankings, ks=[1, 3])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate & Filter Data
    cascades, r0s = generate_data()
    n_total = len(cascades)
    print(f"\nTotal evaluable cascades: {n_total}")
    
    # 2. Extract Features
    print("Extracting features ...")
    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]  # cascade_idx acts as grouping variable
    
    # Check class imbalance
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution  : source={sum(y==1)}, non-source={sum(y==0)}")
    
    # 3. Train/Test Split
    # We use GroupKFold cross-validation to ensure all nodes of a cascade stay together.
    # We do a simple 1-fold (80/20 train/test split)
    print("Training Random Forest ...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    
    X_train, y_train = X[train_idx], y[train_idx]
    
    # Train model
    rf = SourceRandomForest(n_estimators=100, max_depth=10, random_state=BASE_SEED)
    rf.fit(X_train, y_train, feature_names)
    
    # Setup test set cascades
    test_cascade_indices = set(groups[i] for i in test_idx)
    test_cascades = [cascades[i] for i in test_cascade_indices]
    test_r0s = [r0s[i] for i in test_cascade_indices]
    
    print(f"Test evaluation on {len(test_cascades)} unseen cascades")
    
    # 4. Evaluate all methods
    print("Evaluating models ...\n")
    # results_by_r0[r0][method] = dict of metrics
    metrics_by_r0: dict[float, dict[str, dict]] = defaultdict(dict)
    
    # Group test cascades by R0
    for eval_r0 in R0_VALUES:
        r0_indices = [i for i, r in enumerate(test_r0s) if r == eval_r0]
        if not r0_indices: continue
        
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
        metrics_by_r0[eval_r0]["random"] = evaluate_random_baseline(subset_cascades)

    # 5. Print Results
    print("=" * 105)
    print(f"{'Method':<18}  " + "   ".join(f"R0={r:>3.1f}" for r in R0_VALUES))
    print(" " * 18 + "  " + "  ".join("Top1   Top3 " for _ in R0_VALUES))
    print("-" * 105)

    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<18}  "
        for r0 in R0_VALUES:
            m = metrics_by_r0[r0].get(method)
            if m:
                t1 = 100 * m["top_k"][1]
                t3 = 100 * m["top_k"][3]
                row += f"{t1:>4.1f}% {t3:>4.1f}%  "
            else:
                row += "   —      —   "
        print(row)
    print("=" * 105)

    # 6. Plot Accuracy Comparison 
    _plot_accuracy(metrics_by_r0)
    
    # 7. Plot Feature Importances
    _plot_feature_importances(rf)


def _plot_accuracy(metrics_by_r0: dict[float, dict[str, dict]]) -> None:
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
    
    r0_list = [r for r in R0_VALUES if r in metrics_by_r0]
    x = np.arange(len(r0_list))
    bar_w = 0.12
    offsets = np.linspace(-(len(METHOD_ORDER) - 1) / 2 * bar_w,
                          (len(METHOD_ORDER) - 1) / 2 * bar_w,
                          len(METHOD_ORDER))
    
    for ax, k_measure, title in zip(axes, [1, 3], ["Top-1 Accuracy", "Top-3 Accuracy"]):
        ax.set_facecolor("#1a1a2e")
        for i, method in enumerate(METHOD_ORDER):
            vals = [100 * metrics_by_r0[r0][method]["top_k"][k_measure] if method in metrics_by_r0[r0] else 0 for r0 in r0_list]
            ax.bar(x + offsets[i], vals, bar_w, label=METHOD_LABELS[method], color=palette[method], edgecolor="black", linewidth=0.5)
            
        ax.set_xticks(x)
        ax.set_xticklabels([f"R0={r}" for r in r0_list], color="lightgray")
        ax.set_ylabel(f"Top-{k_measure} Accuracy (%)", color="lightgray")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.tick_params(colors="lightgray")
        if k_measure == 1:
            ax.legend(fontsize=9, facecolor="#222", edgecolor="#444", labelcolor="white", loc="upper left", bbox_to_anchor=(1.02, 1))
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = OUT_DIR / "rf_vs_baselines_ba_ic.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot   -> {out_file}")


def _plot_feature_importances(rf: SourceRandomForest) -> None:
    imp = rf.feature_importances
    if not imp: return
    
    sorted_imp = sorted(imp.items(), key=lambda x: x[1])
    features, scores = zip(*sorted_imp)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, align='center', color="#2ec4b6", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, color="lightgray")
    ax.set_xlabel("Mean Decrease in Impurity (Gini)", color="lightgray")
    ax.set_title("Random Forest - Feature Importances (BA)", color="white", fontweight="bold")
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
        
    plt.tight_layout()
    out_file = OUT_DIR / "rf_feature_importance_ba_ic.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved importance plot -> {out_file}")


if __name__ == "__main__":
    main()
