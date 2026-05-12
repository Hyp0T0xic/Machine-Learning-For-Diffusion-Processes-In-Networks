#!/usr/bin/env python
"""
Extract advanced metrics averaged across all 5 random seeds on their respective test sets for IC on ER.
Computes Top-1, Top-3, MRR, Mean Hops for RF + all baselines, plus Hop distribution and 2x2 Confusion Matrix for RF.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.experiments.train_rf_ic_er import generate_data, R0_VALUES, SEEDS
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import distance_to_source, mean_reciprocal_rank, evaluate_ranker

OUT_DIR = _REPO_ROOT / "results" / "figures" / "ml_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "random_forest": "Random Forest",
    "jordan":      "Jordan Centre",
    "closeness":   "Closeness",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "random":      "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


def evaluate_random_baseline(results: list, seed: int = 42) -> dict:
    """Evaluate random guessing by simulating rankings."""
    rng = random.Random(seed)
    random_rankings = []
    for r in results:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        random_rankings.append(nodes)
    return evaluate_ranker(results, random_rankings, ks=[1, 3])


def main() -> None:
    print("=" * 95)
    print(f"ADVANCED TEST-SET EVALUATION across {len(SEEDS)} SEEDS: IC on ER (Size=25)")
    print("=" * 95)

    cascade_size = 25
    N_NODES = 200

    # Accumulate results across seeds: seed_metrics[method][r0] -> list of dicts
    seed_metrics = {m: defaultdict(list) for m in METHOD_ORDER}

    # Accumulate global counts for the 2x2 confusion matrix and hop distributions across all seeds (RF only)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    all_hop_distances = []

    for s_idx, seed in enumerate(SEEDS):
        print(f"\n[{s_idx+1}/{len(SEEDS)}] Running full extraction and evaluation for SEED {seed}...")

        # 1. Generate data
        cascades, r0s = generate_data(seed=seed, cascade_size=cascade_size)

        # 2. Build feature matrix
        X, y, index, feature_names = build_feature_matrix(cascades)
        groups = [idx[0] for idx in index]

        # 3. Deterministic Train/Test Split
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

        # Fit model for this seed on the fly to evaluate its exact test set
        X_train, y_train = X[train_idx], y[train_idx]
        rf = SourceRandomForest(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=5,
            max_features=3,
            random_state=seed
        )
        rf.fit(X_train, y_train, feature_names)

        # Setup test set cascades
        test_cascade_indices = sorted(set(groups[i] for i in test_idx))
        test_cascades = [cascades[i] for i in test_cascade_indices]
        test_r0s = [r0s[i] for i in test_cascade_indices]

        # Evaluate per R0
        for eval_r0 in R0_VALUES:
            subset = [c for c, r in zip(test_cascades, test_r0s) if r == eval_r0]
            if not subset:
                continue

            # --- Random Forest ---
            rf_rankings = [rf.rank_nodes(c) for c in subset]
            rf_eval = evaluate_ranker(subset, rf_rankings, ks=[1, 3])
            seed_metrics["random_forest"][eval_r0].append(rf_eval)

            # RF-specific: confusion matrix + hop distances
            for cascade, ranked in zip(subset, rf_rankings):
                true_src = cascade.source
                pred_src = ranked[0]
                if pred_src == true_src:
                    total_tp += 1
                    total_tn += (N_NODES - 1)
                else:
                    total_fp += 1
                    total_fn += 1
                    total_tn += (N_NODES - 2)

                dist = distance_to_source(cascade, ranked)
                if dist != float("inf"):
                    all_hop_distances.append(dist)

            # --- Centrality Baselines ---
            baseline_cols = defaultdict(list)
            for c in subset:
                preds = predict_all(c)
                for m_name, ranking in preds.items():
                    baseline_cols[m_name].append(ranking)
            for m_name in baseline_cols:
                bl_eval = evaluate_ranker(subset, baseline_cols[m_name], ks=[1, 3])
                seed_metrics[m_name][eval_r0].append(bl_eval)

            # --- Random Baseline ---
            rand_eval = evaluate_random_baseline(subset, seed=seed)
            seed_metrics["random"][eval_r0].append(rand_eval)

    # Average metrics across the 5 seeds and print results table
    print(f"\n{'='*110}")
    print(f"AVERAGED RESULTS TABLE across {len(SEEDS)} SEEDS: Detailed Test-Set Metrics (IC on ER)")
    print(f"{'='*110}")

    for method in METHOD_ORDER:
        print(f"\n  --- {METHOD_LABELS[method]} ---")
        print(f"  {'R0 Value':<9} | {'Top-1 Acc (%)':<15} | {'Top-3 Acc (%)':<15} | {'MRR':<13} | {'Mean Hops':<10}")
        print("  " + "-" * 75)
        for r0 in R0_VALUES:
            if r0 in seed_metrics[method]:
                s_list = seed_metrics[method][r0]
                t1_vals = [100 * m["top_k"][1] for m in s_list]
                t3_vals = [100 * m["top_k"][3] for m in s_list]
                mrr_vals = [m["mrr"] for m in s_list]
                hop_vals = [m["mean_distance"] for m in s_list]
                t1_m, t1_s = np.mean(t1_vals), np.std(t1_vals)
                t3_m, t3_s = np.mean(t3_vals), np.std(t3_vals)
                mrr_m, mrr_s = np.mean(mrr_vals), np.std(mrr_vals)
                h_m = np.mean(hop_vals)
                print(f"  R0 = {r0:<4.1f} | {t1_m:>5.1f}±{t1_s:<5.1f}   | {t3_m:>5.1f}±{t3_s:<5.1f}   | {mrr_m:>5.3f}±{mrr_s:<5.3f} | {h_m:>6.2f}")

    # Visualizations
    print("\nGenerating visual diagnostics...")

    # Plot A: Hop Distribution Bar Chart (RF only)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    max_hop_bin = 4
    hop_counts = defaultdict(int)
    for h in all_hop_distances:
        bin_h = int(min(h, max_hop_bin))
        hop_counts[bin_h] += 1

    total_valid = len(all_hop_distances)
    bins = list(range(max_hop_bin + 1))
    percentages = [hop_counts[b] / total_valid * 100 if total_valid > 0 else 0 for b in bins]
    labels = [str(b) for b in bins[:-1]] + [f"{max_hop_bin}+"]

    bars = ax.bar(labels, percentages, color="#ffb703", edgecolor="black", width=0.6)

    for bar_obj, pct in zip(bars, percentages):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 1.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Shortest Path Hop Distance to True Source")
    ax.set_ylabel("Percentage of Test Cascades (%)")
    ax.set_title(f"Hop Distance Distribution (IC on ER, Averaged over {len(SEEDS)} seeds, Total $n={total_valid}$)")
    ax.set_ylim(0, max(percentages) + 12 if percentages else 100)

    plt.tight_layout()
    hop_file = OUT_DIR / "testset_hop_distribution_ic_er.png"
    fig.savefig(hop_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Hop Distribution -> {hop_file}")

    # Plot B: 2x2 Binary Confusion Matrix (RF only)
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_2x2 = np.array([[total_tn, total_fp],
                       [total_fn, total_tp]])

    total_decisions = total_tn + total_fp + total_fn + total_tp

    # Use a custom annotation matrix
    annot_texts = [
        [f"True Negative (TN)\n{total_tn:,}\n({total_tn/total_decisions*100:.2f}%)",
         f"False Positive (FP)\n{total_fp:,}\n({total_fp/total_decisions*100:.2f}%)"],
        [f"False Negative (FN)\n{total_fn:,}\n({total_fn/total_decisions*100:.2f}%)",
         f"True Positive (TP)\n{total_tp:,}\n({total_tp/total_decisions*100:.2f}%)"]
    ]

    im = ax.imshow(cm_2x2, cmap="Oranges", norm=matplotlib.colors.LogNorm(vmin=1, vmax=cm_2x2.max()))

    # Add borders and text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if cm_2x2[i, j] > cm_2x2.max() * 0.1 else "black"
            ax.text(j, i, annot_texts[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Non-Source", "Predicted Source"])
    ax.set_yticklabels(["Actual Non-Source", "Actual Source"])
    ax.set_title(f"2x2 Source Detection Confusion Matrix\n(IC on ER across {len(SEEDS)} seeds, Total Decisions={total_decisions:,})", pad=15)

    plt.tight_layout()
    cm_file = OUT_DIR / "testset_confusion_matrix_ic_er.png"
    fig.savefig(cm_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 2x2 Confusion Matrix -> {cm_file}")

    # Export all metrics and distributions to a structured JSON file
    json_dir = _REPO_ROOT / "results" / "metrics"
    json_dir.mkdir(parents=True, exist_ok=True)

    json_data = {
        "metadata": {
            "topology": "ic_er",
            "cascade_size": cascade_size,
            "n_seeds": len(SEEDS),
            "n_nodes": N_NODES,
            "total_decisions": total_decisions
        },
        "method_labels": METHOD_LABELS,
        "method_order": METHOD_ORDER,
        "metrics_by_r0": {},
        "confusion_matrix_2x2": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn
        },
        "hop_distribution": {
            "counts": {str(k): v for k, v in hop_counts.items()},
            "percentages": {str(b): p for b, p in zip(bins, percentages)}
        }
    }

    for r0 in R0_VALUES:
        r0_key = str(r0)
        json_data["metrics_by_r0"][r0_key] = {}
        for method in METHOD_ORDER:
            if r0 in seed_metrics[method]:
                s_list = seed_metrics[method][r0]
                t1_vals = [100 * m["top_k"][1] for m in s_list]
                t3_vals = [100 * m["top_k"][3] for m in s_list]
                mrr_vals = [m["mrr"] for m in s_list]
                hop_vals = [m["mean_distance"] for m in s_list]
                json_data["metrics_by_r0"][r0_key][method] = {
                    "top1_mean": float(np.mean(t1_vals)),
                    "top1_std": float(np.std(t1_vals)),
                    "top3_mean": float(np.mean(t3_vals)),
                    "top3_std": float(np.std(t3_vals)),
                    "mrr_mean": float(np.mean(mrr_vals)),
                    "mrr_std": float(np.std(mrr_vals)),
                    "mean_hops": float(np.mean(hop_vals)),
                }

    json_path = json_dir / "testset_evaluation_ic_er.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved JSON Results -> {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
