#!/usr/bin/env python
"""
Test pre-trained RF models (IC on BA/ER) against real FalseNews cascades.
Compares to centrality baselines and random. No training here, just eval.

Usage: python validation/validate_existing_models.py
"""
from __future__ import annotations

import sys
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib

from validation.truefalsevalidation.load_falsenews import load_falsenews_cascades
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Config ------------------------------------------------------------------

TARGET_SIZE = 25
MODEL_PATHS = {
    "RF (IC-BA)": _REPO_ROOT / "results/models/ic_ba/rf_model_size25.pkl",
    "RF (IC-ER)": _REPO_ROOT / "results/models/ic_er/rf_model_size25.pkl",
}
OUT_DIR = _REPO_ROOT / "validation/results/figures"

METHOD_LABELS = {
    "RF (IC-BA)":  "RF (trained IC-BA)",
    "RF (IC-ER)":  "RF (trained IC-ER)",
    "jordan":      "Jordan Centre",
    "closeness":   "Closeness",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "random":      "Random",
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cascades, metadata = load_falsenews_cascades(target_size=TARGET_SIZE)
    print(f"\nEvaluating on {len(cascades)} FalseNews cascades "
          f"(size={TARGET_SIZE})\n")

    # load saved models
    models = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded model: {name}  ({path})")
        else:
            print(f"  WARNING: model not found — {path}")

    metrics: dict[str, dict] = {}

    # rank with each pre-trained RF
    for name, model in models.items():
        print(f"\n  Ranking cascades with {name} ...")
        rankings = [model.rank_nodes(c) for c in cascades]
        metrics[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])

    # centrality baselines
    print("\n  Ranking cascades with baselines ...")
    baseline_cols: dict[str, list] = defaultdict(list)
    for c in cascades:
        preds = predict_all(c)
        for m_name, ranking in preds.items():
            baseline_cols[m_name].append(ranking)
    for m_name, rankings in baseline_cols.items():
        metrics[m_name] = evaluate_ranker(cascades, rankings, ks=[1, 3])

    metrics["random"] = _evaluate_random(cascades)

    # results table
    print(f"\n{'='*80}")
    print(f"  EXISTING MODELS ON FALSENEWS DATA  |  CASCADE SIZE = {TARGET_SIZE}")
    print(f"  {len(cascades)} cascades evaluated")
    print(f"{'='*80}")
    print(f"{'Method':<25}  {'Top-1':>8}  {'Top-3':>8}  "
          f"{'MRR':>8}  {'Mean Dist':>10}")
    print("-" * 80)
    for method in METHOD_ORDER:
        m = metrics.get(method)
        if m:
            print(f"{METHOD_LABELS[method]:<25}  "
                  f"{100*m['top_k'][1]:>7.1f}%  "
                  f"{100*m['top_k'][3]:>7.1f}%  "
                  f"{m['mrr']:>8.4f}  "
                  f"{m['mean_distance']:>10.2f}")
    print("=" * 80)

    _plot_accuracy(metrics, len(cascades))


HATCHES = {
    "RF (IC-BA)":  "",
    "RF (IC-ER)":  "////",
    "jordan":      "\\\\\\\\",
    "closeness":   "xxxx",
    "betweenness": "----",
    "degree":      "||||",
    "random":      "....",
}


def _plot_accuracy(metrics: dict, n_cascades: int) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    methods_present = [m for m in METHOD_ORDER if m in metrics]
    x = np.arange(len(methods_present))

    for ax, k, title in zip(axes, [1, 3], [
        f"top-1 accuracy — existing models on FalseNews ($n={n_cascades}$)",
        f"top-3 accuracy — existing models on FalseNews ($n={n_cascades}$)",
    ]):
        vals = [100 * metrics[m]["top_k"][k] for m in methods_present]
        for xi, (val, method) in enumerate(zip(vals, methods_present)):
            ax.bar(xi, val, facecolor="white", hatch=HATCHES.get(method, ""),
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in methods_present],
            rotation=30, ha="right", fontsize=9,
        )
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out_file = OUT_DIR / "existing_models_on_falsenews.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot -> {out_file}")


if __name__ == "__main__":
    main()
