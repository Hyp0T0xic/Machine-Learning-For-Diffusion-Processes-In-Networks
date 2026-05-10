#!/usr/bin/env python
"""
Train RF on FalseNews cascades and compare to IC-trained models + baselines.
Feature matrix built once, only the train/test split varies across seeds.

Usage: python validation/train_rf_falsenews.py
"""
from __future__ import annotations

import sys
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import joblib

from validation.load_falsenews import load_falsenews_cascades
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Config ------------------------------------------------------------------

TARGET_SIZE   = 25
SEEDS         = [42, 123, 456, 789, 1024]
OUT_DIR       = Path("validation/results/figures")
MODEL_DIR     = Path("validation/results/models")

# these never saw FalseNews data so no leakage concern
EXISTING_MODELS = {
    "RF (IC-BA)": Path("results/models/ic_ba/rf_model_size25.pkl"),
    "RF (IC-ER)": Path("results/models/ic_er/rf_model_size25.pkl"),
}

METHOD_LABELS = {
    "rf_falsenews": "RF (FalseNews)",
    "RF (IC-BA)":   "RF (IC-BA)",
    "RF (IC-ER)":   "RF (IC-ER)",
    "jordan":       "Jordan Centre",
    "closeness":    "Closeness",
    "betweenness":  "Betweenness",
    "degree":       "Degree",
    "random":       "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


def _evaluate_random(results, seed=42):
    rng = random.Random(seed)
    rankings = []
    for r in results:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        rankings.append(nodes)
    return evaluate_ranker(results, rankings, ks=[1, 3])


def _run_seed(cascades, X, y, groups, feature_names, existing_models, seed):
    """One fold: train on 80%, eval everything on held-out 20%."""

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    # same hyperparams as IC scripts
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
    test_cascades = [cascades[i] for i in test_cascade_indices]

    # eval all methods on same test set for fair comparison
    metrics: dict[str, dict] = {}

    rf_rankings = [rf.rank_nodes(c) for c in test_cascades]
    metrics["rf_falsenews"] = evaluate_ranker(
        test_cascades, rf_rankings, ks=[1, 3],
    )

    for name, model in existing_models.items():
        rankings = [model.rank_nodes(c) for c in test_cascades]
        metrics[name] = evaluate_ranker(test_cascades, rankings, ks=[1, 3])

    baseline_cols: dict[str, list] = defaultdict(list)
    for c in test_cascades:
        preds = predict_all(c)
        for m_name, ranking in preds.items():
            baseline_cols[m_name].append(ranking)
    for m_name, rankings in baseline_cols.items():
        metrics[m_name] = evaluate_ranker(test_cascades, rankings, ks=[1, 3])

    metrics["random"] = _evaluate_random(test_cascades, seed=seed)

    return metrics, rf.feature_importances, rf


def _aggregate_metrics(all_seed_metrics):
    avg: dict[str, dict] = {}
    for method in METHOD_ORDER:
        t1, t3, mrr = [], [], []
        for sm in all_seed_metrics:
            if method in sm:
                t1.append(sm[method]["top_k"][1])
                t3.append(sm[method]["top_k"][3])
                mrr.append(sm[method]["mrr"])
        if t1:
            avg[method] = {
                "top_k":     {1: np.mean(t1), 3: np.mean(t3)},
                "top_k_std": {1: np.std(t1),  3: np.std(t3)},
                "mrr":       float(np.mean(mrr)),
                "mrr_std":   float(np.std(mrr)),
            }
    return avg


def _aggregate_importances(all_imp):
    avg = defaultdict(float)
    for imp in all_imp:
        for feat, val in imp.items():
            avg[feat] += val / len(all_imp)
    return dict(avg)


def _print_results(avg_metrics):
    print(f"\n{'='*90}")
    print(f"  FALSENEWS VALIDATION  |  SIZE = {TARGET_SIZE}  |  "
          f"AVERAGED OVER {len(SEEDS)} SEEDS")
    print(f"{'='*90}")
    print(f"{'Method':<25}  {'Top-1':>12}  {'Top-3':>12}  {'MRR':>8}")
    print("-" * 90)
    for method in METHOD_ORDER:
        m = avg_metrics.get(method)
        if m:
            t1     = 100 * m["top_k"][1]
            t3     = 100 * m["top_k"][3]
            t1_std = 100 * m["top_k_std"][1]
            t3_std = 100 * m["top_k_std"][3]
            print(f"{METHOD_LABELS[method]:<25}  "
                  f"{t1:>5.1f}±{t1_std:>4.1f}%  "
                  f"{t3:>5.1f}±{t3_std:>4.1f}%  "
                  f"{m['mrr']:>8.4f}")
    print("=" * 90)


# -- Plots -------------------------------------------------------------------

def _plot_comparison(avg_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")

    palette = {
        "rf_falsenews": "#06d6a0",
        "RF (IC-BA)":   "#ffb703",
        "RF (IC-ER)":   "#e76f51",
        "jordan":       "#e63946",
        "closeness":    "#f4a261",
        "betweenness":  "#2ec4b6",
        "degree":       "#a8dadc",
        "random":       "#888888",
    }

    present = [m for m in METHOD_ORDER if m in avg_metrics]
    x = np.arange(len(present))

    for ax, k, title in zip(axes, [1, 3], [
        f"Top-1 Accuracy — FalseNews Validation (size={TARGET_SIZE})",
        f"Top-3 Accuracy — FalseNews Validation (size={TARGET_SIZE})",
    ]):
        ax.set_facecolor("#1a1a2e")
        vals = [100 * avg_metrics[m]["top_k"][k] for m in present]
        errs = [100 * avg_metrics[m]["top_k_std"][k] for m in present]
        colors = [palette.get(m, "#888") for m in present]
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor="black",
               linewidth=0.5, capsize=3,
               error_kw={"ecolor": "white", "alpha": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in present],
            color="lightgray", rotation=30, ha="right", fontsize=9,
        )
        ax.set_ylabel(f"Top-{k} Accuracy (%)", color="lightgray")
        ax.set_title(title, color="white", fontweight="bold", fontsize=10)
        ax.set_ylim(0, 105)
        ax.tick_params(colors="lightgray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = OUT_DIR / "falsenews_comparison.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot -> {out_file}")
    return out_file


def _plot_feature_importances(avg_importances):
    if not avg_importances:
        return None

    sorted_imp = sorted(avg_importances.items(), key=lambda x: x[1])
    features, scores = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, align="center", color="#06d6a0", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, color="lightgray")
    ax.set_xlabel(
        f"Mean Decrease in Impurity (avg over {len(SEEDS)} seeds)",
        color="lightgray",
    )
    ax.set_title(
        f"RF Feature Importances — FalseNews (size={TARGET_SIZE})",
        color="white", fontweight="bold",
    )
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = OUT_DIR / "rf_feature_importance_falsenews.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved importance plot -> {out_file}")
    return out_file


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cascades, metadata = load_falsenews_cascades(target_size=TARGET_SIZE)
    print(f"\nTraining on {len(cascades)} FalseNews cascades\n")

    # build once — data is fixed, only the split changes per seed
    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]

    existing_models = {}
    for name, path in EXISTING_MODELS.items():
        if path.exists():
            existing_models[name] = joblib.load(path)
            print(f"  Loaded existing model: {name}")
        else:
            print(f"  WARNING: model not found — {path}")

    all_seed_metrics: list[dict]  = []
    all_seed_importances: list[dict] = []
    all_models: list[SourceRandomForest] = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        metrics, importances, model = _run_seed(
            cascades, X, y, groups, feature_names, existing_models, seed,
        )
        all_seed_metrics.append(metrics)
        all_seed_importances.append(importances)
        all_models.append(model)

    avg_metrics     = _aggregate_metrics(all_seed_metrics)
    avg_importances = _aggregate_importances(all_seed_importances)

    _print_results(avg_metrics)
    _plot_comparison(avg_metrics)
    _plot_feature_importances(avg_importances)

    model_path = MODEL_DIR / "rf_model_falsenews_size25.pkl"
    joblib.dump(all_models[0], model_path)
    print(f"\nSaved model -> {model_path}")


if __name__ == "__main__":
    main()
