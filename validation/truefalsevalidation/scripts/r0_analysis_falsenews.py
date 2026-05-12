#!/usr/bin/env python
"""
Analyse the empirical R0 distribution of the FalseNews validation cascades,
then compare source-identification accuracy across R0 bins for the
FalseNews-trained RF and all centrality baselines.
Saves results to JSON for separate plotting.

Usage: python validation/truefalsevalidation/r0_analysis_falsenews.py
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
from sklearn.model_selection import StratifiedGroupKFold

from validation.truefalsevalidation.load_falsenews import load_falsenews_cascades
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker

# -- Config ------------------------------------------------------------------

TARGET_SIZE = 25
SEEDS       = [42, 123, 456, 789, 1024]
DATA_DIR    = _REPO_ROOT / "validation/truefalsevalidation/data"

CSV_PATH = (
    _REPO_ROOT
    / "FalseNews_Code_Data"
    / "FalseNews_Code_Data"
    / "data"
    / "raw_data_anon_filtered_min25_reachable.csv"
)

R0_BINS   = [(0.0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, float("inf"))]
R0_LABELS = ["R0~0.5", "R0~1", "R0~2", "R0~3", "R0~5"]

METHOD_LABELS = {
    "rf_falsenews": "RF (FalseNews)",
    "jordan":       "Jordan Centre",
    "closeness":    "Closeness",
    "betweenness":  "Betweenness",
    "degree":       "Degree",
    "random":       "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


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


def _run_seed(cascades, r0_per_cascade, X, y, groups, feature_names, seed):
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
    test_r0s      = [r0_per_cascade[i] for i in test_cascade_indices]

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
    print(f"  FALSENEWS -- ACCURACY BY R0 BIN  |  SIZE = {TARGET_SIZE}  |  "
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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cascades, _ = load_falsenews_cascades(
        csv_path=CSV_PATH, target_size=TARGET_SIZE, min_size=1,
    )
    print(f"\nLoaded {len(cascades)} FalseNews cascades\n")

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

    # save to JSON
    json_avg = {}
    for label, methods in avg_metrics.items():
        json_avg[label] = {}
        for method, m in methods.items():
            json_avg[label][method] = {
                "top_1": float(m["top_k"][1]),
                "top_3": float(m["top_k"][3]),
                "top_1_std": float(m["top_k_std"][1]),
                "top_3_std": float(m["top_k_std"][3]),
            }

    json_data = {
        "target_size": TARGET_SIZE,
        "n_seeds": len(SEEDS),
        "r0_labels": R0_LABELS,
        "method_labels": METHOD_LABELS,
        "method_order": METHOD_ORDER,
        "r0_values": [float(v) for v in r0_per_cascade],
        "bin_counts": bin_counts,
        "avg_metrics": json_avg,
    }

    out_path = DATA_DIR / "r0_analysis_falsenews.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
