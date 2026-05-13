#!/usr/bin/env python
"""train a fresh rf on falsenews cascades, compare to ic-ba/er + baselines, dump metrics + importances to json"""
from __future__ import annotations

import json
import sys
import random
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import joblib

from validation.truefalsevalidation.load_falsenews import load_falsenews_cascades
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Config ------------------------------------------------------------------

TARGET_SIZE   = 25
SEEDS         = [42, 123, 456, 789, 1024]
DATA_DIR      = _REPO_ROOT / "validation/truefalsevalidation/data"
MODEL_DIR     = _REPO_ROOT / "validation/truefalsevalidation/models"

EXISTING_MODELS = {
    "RF (IC-BA)": _REPO_ROOT / "results/models/ic_ba/rf_model_size25.pkl",
    "RF (IC-ER)": _REPO_ROOT / "results/models/ic_er/rf_model_size25.pkl",
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
    test_cascades = [cascades[i] for i in test_cascade_indices]

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
                  f"{t1:>5.1f}+/-{t1_std:>4.1f}%  "
                  f"{t3:>5.1f}+/-{t3_std:>4.1f}%  "
                  f"{m['mrr']:>8.4f}")
    print("=" * 90)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cascades, metadata = load_falsenews_cascades(target_size=TARGET_SIZE)
    print(f"\nTraining on {len(cascades)} FalseNews cascades\n")

    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]

    existing_models = {}
    for name, path in EXISTING_MODELS.items():
        if path.exists():
            existing_models[name] = joblib.load(path)
            print(f"  Loaded existing model: {name}")
        else:
            print(f"  WARNING: model not found -- {path}")

    all_seed_metrics: list[dict] = []
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

    # save to JSON
    json_data = {
        "target_size": TARGET_SIZE,
        "n_seeds": len(SEEDS),
        "method_labels": METHOD_LABELS,
        "method_order": METHOD_ORDER,
        "avg_metrics": {},
        "avg_importances": {k: float(v) for k, v in avg_importances.items()},
    }
    for method, m in avg_metrics.items():
        json_data["avg_metrics"][method] = {
            "top_1": float(m["top_k"][1]),
            "top_3": float(m["top_k"][3]),
            "top_1_std": float(m["top_k_std"][1]),
            "top_3_std": float(m["top_k_std"][3]),
            "mrr": float(m["mrr"]),
            "mrr_std": float(m["mrr_std"]),
        }

    out_path = DATA_DIR / "train_rf_falsenews.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")

    model_path = MODEL_DIR / "rf_model_falsenews_size25.pkl"
    joblib.dump(all_models[0], model_path)
    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
