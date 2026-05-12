#!/usr/bin/env python
"""
Test pre-trained RF models (IC on BA/ER) against real FalseNews cascades.
Compares to centrality baselines and random. No training here, just eval.
Saves metrics to JSON for separate plotting.

Usage: python validation/truefalsevalidation/validate_existing_models.py
"""
from __future__ import annotations

import json
import sys
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

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
DATA_DIR = _REPO_ROOT / "validation/truefalsevalidation/data"

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

    for name, model in models.items():
        print(f"\n  Ranking cascades with {name} ...")
        rankings = [model.rank_nodes(c) for c in cascades]
        metrics[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])

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

    # save to JSON
    json_data = {
        "n_cascades": len(cascades),
        "target_size": TARGET_SIZE,
        "method_labels": METHOD_LABELS,
        "method_order": METHOD_ORDER,
        "metrics": {},
    }
    for method, m in metrics.items():
        json_data["metrics"][method] = {
            "top_1": float(m["top_k"][1]),
            "top_3": float(m["top_k"][3]),
            "mrr": float(m["mrr"]),
            "mean_distance": float(m["mean_distance"]),
        }

    out_path = DATA_DIR / "validate_existing_models.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
