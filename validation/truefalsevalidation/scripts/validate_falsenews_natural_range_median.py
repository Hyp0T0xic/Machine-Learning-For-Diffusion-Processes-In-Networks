#!/usr/bin/env python
"""
FalseNews natural-range validation centred on the median: select 101
cascades whose untruncated size is closest to the population median
(50 below, the median, and 50 above) and evaluate all rankers.

JSON output matches the standardized format used by validate_existing_models2.py
and validate_weibo.py (commit 2b0b737) — per-method `metrics` plus a
top-level `hop_distribution` dict keyed by method.

Reads:  FalseNews raw CSV (via load_falsenews_cascades)
        results/models/ic_ba/rf_model_size25.pkl
        results/models/ic_er/rf_model_size25.pkl
Writes: validation/truefalsevalidation/data/validate_falsenews_natural_range_median.json

Usage: python validation/truefalsevalidation/scripts/validate_falsenews_natural_range_median.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np

from validation.truefalsevalidation.scripts.load_falsenews import load_falsenews_cascades
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker, distance_to_source

# ── Config ────────────────────────────────────────────────────────────────────

MIN_CASCADE_SIZE = 25
NO_TRUNCATE_SIZE = 1_000_000   # effectively no truncation
N_AROUND_MEDIAN  = 101

SEEDS    = [42, 123, 456, 789, 1024]
DATA_DIR = _REPO_ROOT / "validation" / "truefalsevalidation" / "data"

MODELS = {
    "RF (IC-BA)": _REPO_ROOT / "results" / "models" / "ic_ba" / "rf_model_size25.pkl",
    "RF (IC-ER)": _REPO_ROOT / "results" / "models" / "ic_er" / "rf_model_size25.pkl",
}

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


# ── Selection ─────────────────────────────────────────────────────────────────


def select_median_centred_cascades(n_around: int):
    """Load all FalseNews cascades untruncated; pick 101 closest to median."""
    print("[load] reading FalseNews CSV (untruncated, "
          f"min_size={MIN_CASCADE_SIZE}) ...")
    all_cascades, _ = load_falsenews_cascades(
        target_size=NO_TRUNCATE_SIZE,
        min_size=MIN_CASCADE_SIZE,
    )
    sizes = np.array([c.size for c in all_cascades], dtype=int)
    if sizes.size == 0:
        return [], {}

    median_size = float(np.median(sizes))
    print(f"[median] across {len(all_cascades):,} cascades:")
    print(f"         median = {median_size:.1f}")
    print(f"         mean   = {float(sizes.mean()):.1f}")
    print(f"         range  = {int(sizes.min())}–{int(sizes.max())}")

    order = sorted(
        range(len(all_cascades)),
        key=lambda i: (abs(all_cascades[i].size - median_size),
                       all_cascades[i].size),
    )
    chosen = order[:n_around]
    cascades = [all_cascades[i] for i in chosen]

    selected_sizes = np.array([c.size for c in cascades])
    print(f"[select] {len(cascades)} closest to median  "
          f"(range {int(selected_sizes.min())}–{int(selected_sizes.max())})")

    pop_info = {
        "pop_size":        int(len(all_cascades)),
        "pop_size_median": float(median_size),
        "pop_size_mean":   float(sizes.mean()),
        "pop_size_min":    int(sizes.min()),
        "pop_size_max":    int(sizes.max()),
        "selected_min":    int(selected_sizes.min()),
        "selected_max":    int(selected_sizes.max()),
        "selected_median": float(np.median(selected_sizes)),
        "selected_mean":   float(np.mean(selected_sizes)),
    }
    return cascades, pop_info


# ── Evaluation ────────────────────────────────────────────────────────────────


def _evaluate_random(cascades, seed=42):
    """Random guessing baseline. Returns (rankings, evaluate_ranker dict)."""
    rng = random.Random(seed)
    rankings = []
    for r in cascades:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        rankings.append(nodes)
    return rankings, evaluate_ranker(cascades, rankings, ks=[1, 3])


def _run_seed(cascades, models, seed):
    """Run all methods for one seed. Returns (metrics_dict, distances_dict)."""
    random.seed(seed)
    np.random.seed(seed)

    results: dict[str, dict] = {}
    all_distances: dict[str, list[float]] = {}

    # RF models
    for name, model in models.items():
        rankings = [model.rank_nodes(c) for c in cascades]
        results[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])
        all_distances[name] = [
            distance_to_source(c, r) for c, r in zip(cascades, rankings)
        ]

    # Centrality baselines: jordan, closeness, betweenness, degree
    baseline_cols: dict[str, list] = defaultdict(list)
    for c in cascades:
        preds = predict_all(c)
        for m_name, ranking in preds.items():
            baseline_cols[m_name].append(ranking)
    for m_name, rankings in baseline_cols.items():
        results[m_name] = evaluate_ranker(cascades, rankings, ks=[1, 3])
        all_distances[m_name] = [
            distance_to_source(c, r) for c, r in zip(cascades, rankings)
        ]

    # Random baseline
    rand_rankings, rand_eval = _evaluate_random(cascades, seed=seed)
    results["random"] = rand_eval
    all_distances["random"] = [
        distance_to_source(c, r) for c, r in zip(cascades, rand_rankings)
    ]

    return results, all_distances


def _compute_hop_distribution(distances: list[float], max_hop_bin: int = 4) -> dict:
    """Bin hop distances into 0, 1, 2, 3, 4+ and return counts + percentages."""
    hop_counts = defaultdict(int)
    for h in distances:
        if h == float("inf"):
            continue
        bin_h = int(min(h, max_hop_bin))
        hop_counts[bin_h] += 1

    total_valid = sum(hop_counts.values())
    bins = list(range(max_hop_bin + 1))
    percentages = [hop_counts[b] / total_valid * 100 if total_valid > 0 else 0
                   for b in bins]
    return {
        "counts":      {str(b): hop_counts[b] for b in bins},
        "percentages": {str(b): round(p, 2) for b, p in zip(bins, percentages)},
    }


def _aggregate_seeds(all_seed_results, all_seed_distances):
    """Average evaluate_ranker dicts across seeds + pool distances for hop dist."""
    avg: dict[str, dict] = {}
    hop_dist: dict[str, dict] = {}
    all_methods = set()
    for sr in all_seed_results:
        all_methods.update(sr.keys())

    for method in all_methods:
        seed_dicts = [sr[method] for sr in all_seed_results if method in sr]
        if not seed_dicts:
            continue
        t1_vals  = [100 * m["top_k"][1] for m in seed_dicts]
        t3_vals  = [100 * m["top_k"][3] for m in seed_dicts]
        mrr_vals = [m["mrr"]            for m in seed_dicts]
        hop_vals = [m["mean_distance"]  for m in seed_dicts]
        avg[method] = {
            "top1_mean": float(np.mean(t1_vals)),
            "top1_std":  float(np.std(t1_vals)),
            "top3_mean": float(np.mean(t3_vals)),
            "top3_std":  float(np.std(t3_vals)),
            "mrr_mean":  float(np.mean(mrr_vals)),
            "mrr_std":   float(np.std(mrr_vals)),
            "mean_hops": float(np.mean(hop_vals)),
        }

        combined_distances: list[float] = []
        for sd in all_seed_distances:
            if method in sd:
                combined_distances.extend(sd[method])
        hop_dist[method] = _compute_hop_distribution(combined_distances)

    return avg, hop_dist


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FALSENEWS NATURAL-RANGE VALIDATION (101 cascades around the median)")
    print("=" * 90)

    print("\n[1/3] Selecting cascades ...")
    cascades, pop_info = select_median_centred_cascades(N_AROUND_MEDIAN)
    if not cascades:
        print("No cascades available — aborting.")
        return

    print("\n[2/3] Loading models ...")
    models = {}
    for name, path in MODELS.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded {METHOD_LABELS[name]} <- {path.name}")
        else:
            print(f"  MISSING: {path}")

    print(f"\n[3/3] Evaluating across {len(SEEDS)} seeds ...")
    all_seed_results: list[dict] = []
    all_seed_distances: list[dict] = []
    for i, seed in enumerate(SEEDS):
        print(f"  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        metrics, distances = _run_seed(cascades, models, seed)
        all_seed_results.append(metrics)
        all_seed_distances.append(distances)

    avg_metrics, hop_distributions = _aggregate_seeds(
        all_seed_results, all_seed_distances,
    )

    print("\n" + "=" * 90)
    print(f"RESULTS: FalseNews natural-range (median-101)  "
          f"(mean ± std over {len(SEEDS)} seeds, n={len(cascades)})")
    print("=" * 90)
    print(f"{'Method':<25} | {'Top-1 Acc (%)':>14} | {'Top-3 Acc (%)':>14}"
          f" | {'MRR':>10} | {'Mean Hops':>10}")
    print("-" * 90)
    for m in METHOD_ORDER:
        if m in avg_metrics:
            a = avg_metrics[m]
            print(f"{METHOD_LABELS[m]:<25} | "
                  f"{a['top1_mean']:>5.1f}±{a['top1_std']:>4.1f}  | "
                  f"{a['top3_mean']:>5.1f}±{a['top3_std']:>4.1f}  | "
                  f"{a['mrr_mean']:>8.4f}  | {a['mean_hops']:>8.3f}")
    print("=" * 90)

    json_data = {
        "dataset":          "FalseNews",
        "selection":        "101_around_median",
        "min_cascade_size": MIN_CASCADE_SIZE,
        "n_cascades":       len(cascades),
        "n_seeds":          len(SEEDS),
        **pop_info,
        "method_labels":    METHOD_LABELS,
        "method_order":     METHOD_ORDER,
        "metrics":          avg_metrics,
        "hop_distribution": hop_distributions,
    }

    out_path = DATA_DIR / "validate_falsenews_natural_range_median.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
