#!/usr/bin/env python
"""
Validate trained RF models (IC-BA, IC-ER) against real Weibo cascade data.
Parses Weibo repost-cascade JSON files, evaluates source identification
accuracy across multiple seeds, and saves metrics to JSON for plotting.

Usage: python validation/weibovalidation/scripts/validate_weibo.py
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

from src.data.cascade import CascadeResult
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker, distance_to_source

# ── Config ────────────────────────────────────────────────────────────────────

WEIBO_DIR        = _REPO_ROOT / "data" / "rumdect" / "rumdect" / "Weibo"
MODELS_DIR       = _REPO_ROOT / "results" / "models"
DATA_DIR         = _REPO_ROOT / "validation" / "weibovalidation" / "data"

MIN_CASCADE_SIZE = 500
MAX_CASCADE_SIZE = 2000
MAX_CASCADES     = 100
SEEDS            = [42, 123, 456, 789, 1024]

METHOD_LABELS = {
    "RF (IC-BA)":    "Random Forest (IC-BA)",
    "RF (IC-ER)":    "Random Forest (IC-ER)",
    "jordan":        "Jordan Centre",
    "closeness":     "Closeness",
    "betweenness":   "Betweenness",
    "degree":        "Degree",
    "random":        "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_weibo_cascade(json_path: Path) -> CascadeResult | None:
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            posts = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    if not posts or len(posts) < MIN_CASCADE_SIZE:
        return None

    mid_to_post = {}
    source_mid = None
    for post in posts:
        mid = str(post.get("mid") or post.get("id"))
        mid_to_post[mid] = post
        if post.get("parent") is None:
            source_mid = mid

    if source_mid is None:
        return None

    mid_to_int: dict[str, int] = {}
    counter = 0

    def get_id(mid: str) -> int:
        nonlocal counter
        if mid not in mid_to_int:
            mid_to_int[mid] = counter
            counter += 1
        return mid_to_int[mid]

    source_int = get_id(source_mid)
    infection_times: dict[int, int] = {}
    cascade_edges: list[tuple[int, int]] = []

    sorted_posts = sorted(posts, key=lambda p: p.get("t", 0))
    t0 = sorted_posts[0].get("t", 0) if sorted_posts else 0

    for post in sorted_posts:
        mid = str(post.get("mid") or post.get("id"))
        node_id = get_id(mid)
        parent_mid = post.get("parent")

        timestamp = post.get("t", 0)
        rel_time = max(0, timestamp - t0)
        infection_times[node_id] = rel_time // 3600

        if parent_mid is not None:
            parent_mid_str = str(parent_mid)
            if parent_mid_str in mid_to_int:
                parent_id = mid_to_int[parent_mid_str]
                cascade_edges.append((parent_id, node_id))

    if len(infection_times) < MIN_CASCADE_SIZE or len(infection_times) > MAX_CASCADE_SIZE:
        return None

    return CascadeResult(
        source=source_int,
        model_name="Weibo_Repost",
        params={},
        infection_times=infection_times,
        cascade_edges=cascade_edges,
        network_name="Weibo",
    )


def load_weibo_cascades(max_count: int = MAX_CASCADES) -> list[CascadeResult]:
    json_files = sorted(WEIBO_DIR.glob("*.json"))
    print(f"Found {len(json_files)} Weibo event files.")

    cascades = []
    skipped = 0
    for idx, jf in enumerate(json_files):
        if len(cascades) >= max_count:
            break
        if idx > 0 and idx % 200 == 0:
            print(f"  ... Scanned {idx}/{len(json_files)} files "
                  f"(found {len(cascades)} large cascades so far)")
        result = load_weibo_cascade(jf)
        if result is not None:
            cascades.append(result)
            print(f"  -> Found large cascade: {result.size} nodes "
                  f"(Total collected: {len(cascades)}/{max_count})")
        else:
            skipped += 1

    print(f"Loaded {len(cascades)} cascades (skipped {skipped}).")
    if cascades:
        sizes = [c.size for c in cascades]
        print(f"  Size range: {min(sizes)}-{max(sizes)}")
        print(f"  Median size: {np.median(sizes):.0f}")
    return cascades


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate_random(cascades, seed=42):
    """Evaluate random guessing baseline. Returns (rankings, eval_dict)."""
    rng = random.Random(seed)
    rankings = []
    for r in cascades:
        nodes = list(r.observed_graph.nodes())
        rng.shuffle(nodes)
        rankings.append(nodes)
    return rankings, evaluate_ranker(cascades, rankings, ks=[1, 3])


def _run_seed(cascades, models, seed):
    """Run all methods for a single seed. Returns (metrics_dict, distances_dict)."""
    random.seed(seed)
    np.random.seed(seed)

    results = {}
    all_distances: dict[str, list[float]] = {}

    # RF models
    for name, model in models.items():
        rankings = [model.rank_nodes(c) for c in cascades]
        results[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])
        all_distances[name] = [
            distance_to_source(c, r) for c, r in zip(cascades, rankings)
        ]

    # Centrality baselines (jordan, closeness, betweenness, degree)
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
    percentages = [hop_counts[b] / total_valid * 100 if total_valid > 0 else 0 for b in bins]
    return {
        "counts": {str(b): hop_counts[b] for b in bins},
        "percentages": {str(b): round(p, 2) for b, p in zip(bins, percentages)},
    }


def _aggregate_seeds(all_seed_results, all_seed_distances):
    """Average evaluate_ranker dicts across seeds into standardized format."""
    avg = {}
    hop_dist = {}
    all_methods = set()
    for sr in all_seed_results:
        all_methods.update(sr.keys())

    for method in all_methods:
        seed_dicts = [sr[method] for sr in all_seed_results if method in sr]
        if not seed_dicts:
            continue
        t1_vals = [100 * m["top_k"][1] for m in seed_dicts]
        t3_vals = [100 * m["top_k"][3] for m in seed_dicts]
        mrr_vals = [m["mrr"] for m in seed_dicts]
        hop_vals = [m["mean_distance"] for m in seed_dicts]
        avg[method] = {
            "top1_mean": float(np.mean(t1_vals)),
            "top1_std":  float(np.std(t1_vals)),
            "top3_mean": float(np.mean(t3_vals)),
            "top3_std":  float(np.std(t3_vals)),
            "mrr_mean":  float(np.mean(mrr_vals)),
            "mrr_std":   float(np.std(mrr_vals)),
            "mean_hops": float(np.mean(hop_vals)),
        }

        # Collect all distances across seeds for hop distribution
        combined_distances = []
        for sd in all_seed_distances:
            if method in sd:
                combined_distances.extend(sd[method])
        hop_dist[method] = _compute_hop_distribution(combined_distances)

    return avg, hop_dist


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WEIBO REAL-WORLD VALIDATION")
    print("=" * 60)

    print("\n[1/3] Loading Weibo cascades...")
    cascades = load_weibo_cascades()
    if not cascades:
        print("No cascades matched criteria.")
        return

    print("\n[2/3] Loading saved models...")
    model_configs = {
        "RF (IC-BA)": MODELS_DIR / "ic_ba" / "rf_model_size25.pkl",
        "RF (IC-ER)": MODELS_DIR / "ic_er" / "rf_model_size25.pkl",
    }
    models = {}
    for name, path in model_configs.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded {name} from {path.name}")
        else:
            print(f"  MISSING: {path}")

    print(f"\n[3/3] Evaluating on Weibo cascades ({len(SEEDS)} seeds)...")
    all_seed_results = []
    all_seed_distances = []
    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        metrics, distances = _run_seed(cascades, models, seed)
        all_seed_results.append(metrics)
        all_seed_distances.append(distances)

    avg_metrics, hop_distributions = _aggregate_seeds(all_seed_results, all_seed_distances)

    print("\n" + "=" * 90)
    print(f"RESULTS: Weibo Real-World Validation (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 90)
    print(f"{'Method':<25} | {'Top-1 Acc (%)':>14} | {'Top-3 Acc (%)':>14} | {'MRR':>10} | {'Mean Hops':>10}")
    print("-" * 90)
    for m in METHOD_ORDER:
        if m in avg_metrics:
            a = avg_metrics[m]
            print(f"{METHOD_LABELS[m]:<25} | {a['top1_mean']:>5.1f}±{a['top1_std']:>4.1f}  "
                  f"| {a['top3_mean']:>5.1f}±{a['top3_std']:>4.1f}  "
                  f"| {a['mrr_mean']:>8.4f}  | {a['mean_hops']:>8.3f}")

    json_data = {
        "n_cascades":       len(cascades),
        "n_seeds":          len(SEEDS),
        "min_cascade_size": MIN_CASCADE_SIZE,
        "max_cascade_size": MAX_CASCADE_SIZE,
        "method_labels":    METHOD_LABELS,
        "method_order":     METHOD_ORDER,
        "metrics":          avg_metrics,
        "hop_distribution": hop_distributions,
    }

    out_path = DATA_DIR / "validate_weibo.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
