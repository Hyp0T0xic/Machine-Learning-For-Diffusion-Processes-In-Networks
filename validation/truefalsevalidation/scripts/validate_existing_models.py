#!/usr/bin/env python
"""evaluate pre-trained ic-ba and ic-er rfs on real falsenews cascades, dump metrics + hop dist to json"""
from __future__ import annotations

import json
import sys
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Add the local scripts directory to path for local imports
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import joblib

# Local import from the same directory
from load_falsenews import load_falsenews_cascades
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker, distance_to_source

# -- Config ------------------------------------------------------------------

TARGET_SIZE = 25
MODEL_PATHS = {
    "RF (IC-BA)": _REPO_ROOT / "results/models/ic_ba/rf_model_size25.pkl",
    "RF (IC-ER)": _REPO_ROOT / "results/models/ic_er/rf_model_size25.pkl",
}
DATA_DIR = _REPO_ROOT / "validation/truefalsevalidation/data"

METHOD_LABELS = {
    "RF (IC-BA)":  "Random Forest (IC-BA)",
    "RF (IC-ER)":  "Random Forest (IC-ER)",
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
    return rankings, evaluate_ranker(results, rankings, ks=[1, 3])


def _compute_hop_distribution(distances: list[float], max_hop_bin: int = 4) -> dict:
    hop_counts = defaultdict(int)
    for h in distances:
        if h == float("inf"):
            continue
        bin_h = int(min(h, max_hop_bin))
        hop_counts[bin_h] += 1

    total_valid = len(distances)
    bins = list(range(max_hop_bin + 1))
    percentages = [hop_counts[b] / total_valid * 100 if total_valid > 0 else 0 for b in bins]
    return {
        "counts": {str(b): hop_counts[b] for b in bins},
        "percentages": {str(b): round(p, 2) for b, p in zip(bins, percentages)},
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cascades, metadata = load_falsenews_cascades(target_size=TARGET_SIZE)
    print(f"\nEvaluating on {len(cascades)} FalseNews cascades "
          f"(size={TARGET_SIZE})\n")

    models = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded model: {name}  ({path})")
        else:
            print(f"  WARNING: model not found -- {path}")

    metrics: dict[str, dict] = {}
    all_distances: dict[str, list[float]] = {}

    for name, model in models.items():
        print(f"  Ranking cascades with {name} ...")
        rankings = [model.rank_nodes(c) for c in cascades]
        metrics[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])
        all_distances[name] = [distance_to_source(c, r) for c, r in zip(cascades, rankings)]

    print("  Ranking cascades with baselines ...")
    baseline_cols: dict[str, list] = defaultdict(list)
    for c in cascades:
        preds = predict_all(c)
        for m_name, ranking in preds.items():
            baseline_cols[m_name].append(ranking)
            
    for m_name, rankings in baseline_cols.items():
        metrics[m_name] = evaluate_ranker(cascades, rankings, ks=[1, 3])
        all_distances[m_name] = [distance_to_source(c, r) for c, r in zip(cascades, rankings)]

    print("  Evaluating random baseline ...")
    rand_rankings, rand_eval = _evaluate_random(cascades)
    metrics["random"] = rand_eval
    all_distances["random"] = [distance_to_source(c, r) for c, r in zip(cascades, rand_rankings)]

    # results table
    print(f"\n{'='*90}")
    print(f"  EXISTING MODELS ON FALSENEWS DATA  |  CASCADE SIZE = {TARGET_SIZE}")
    print(f"  {len(cascades)} cascades evaluated")
    print(f"{'='*90}")
    print(f"{'Method':<25} | {'Top-1 Acc (%)':>14} | {'Top-3 Acc (%)':>14} | {'MRR':>10} | {'Mean Hops':>10}")
    print("-" * 90)
    for method in METHOD_ORDER:
        m = metrics.get(method)
        if m:
            print(f"{METHOD_LABELS[method]:<25} | "
                  f"{100*m['top_k'][1]:>13.1f}% | "
                  f"{100*m['top_k'][3]:>13.1f}% | "
                  f"{m['mrr']:>10.4f} | "
                  f"{m['mean_distance']:>10.2f}")
    print("=" * 90)

    # save to JSON
    json_data = {
        "n_cascades": len(cascades),
        "target_size": TARGET_SIZE,
        "method_labels": METHOD_LABELS,
        "method_order": METHOD_ORDER,
        "metrics": {},
        "hop_distribution": {},
    }
    for method, m in metrics.items():
        json_data["metrics"][method] = {
            "top_1": float(m["top_k"][1]),
            "top_3": float(m["top_k"][3]),
            "mrr": float(m["mrr"]),
            "mean_distance": float(m["mean_distance"]),
        }
        json_data["hop_distribution"][method] = _compute_hop_distribution(all_distances[method])

    out_path = DATA_DIR / "validate_existing_models.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
