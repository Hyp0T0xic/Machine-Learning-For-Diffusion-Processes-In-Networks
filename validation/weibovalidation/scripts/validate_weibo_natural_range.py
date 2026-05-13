#!/usr/bin/env python
"""evaluate pre-trained rf models on the 101 weibo cascades closest in size to the population median"""
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

# config

WEIBO_DIR  = _REPO_ROOT / "data" / "rumdect" / "rumdect" / "Weibo"
MODELS_DIR = _REPO_ROOT / "results" / "models"
DATA_DIR   = _REPO_ROOT / "validation" / "weibovalidation" / "data"

# Lenient floor: cascades smaller than this can't carry useful structural
# signal but anything above is admitted into the median computation.
MIN_CASCADE_SIZE = 25
# Centre-on-median selection — 101 cascades total.
N_AROUND_MEDIAN  = 101

SEEDS = [42, 123, 456, 789, 1024]

MODELS = {
    "RF (IC-BA)": MODELS_DIR / "ic_ba" / "rf_model_size25.pkl",
    "RF (IC-ER)": MODELS_DIR / "ic_er" / "rf_model_size25.pkl",
}

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


# data loading


def _fast_cascade_size(json_path: Path) -> int | None:
    """Quickly count posts in a Weibo event file; require a parent=None root.

    Returns None on parse failure or if no root post exists. Much cheaper
    than fully parsing the cascade.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not posts or not isinstance(posts, list):
        return None
    has_root = any(p.get("parent") is None for p in posts)
    if not has_root:
        return None
    return len(posts)


def _load_full_cascade(json_path: Path) -> CascadeResult | None:
    """Full parse — build a CascadeResult with infection_times + edges."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    if not posts:
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

    return CascadeResult(
        source=source_int,
        model_name="Weibo_Repost",
        params={},
        infection_times=infection_times,
        cascade_edges=cascade_edges,
        network_name="Weibo",
    )


def select_median_centred_cascades(
    min_size: int,
    n_around: int,
) -> tuple[list[CascadeResult], dict]:
    """First pass: measure every cascade's size; second pass: load 101 closest to median."""
    json_files = sorted(WEIBO_DIR.glob("*.json"))
    print(f"[scan] {len(json_files)} Weibo event files in {WEIBO_DIR}")

    sizes: list[tuple[Path, int]] = []
    for idx, jf in enumerate(json_files):
        if idx > 0 and idx % 500 == 0:
            print(f"  ... scanned {idx}/{len(json_files)}  (valid so far: {len(sizes)})")
        sz = _fast_cascade_size(jf)
        if sz is not None and sz >= min_size:
            sizes.append((jf, sz))

    print(f"[scan] {len(sizes)} valid cascades with size >= {min_size}")
    if not sizes:
        return [], {}

    size_arr = np.array([s for _, s in sizes])
    median_size = float(np.median(size_arr))
    print(f"[median] size median across {len(sizes)} cascades = {median_size:.1f}")
    print(f"         size mean   = {float(np.mean(size_arr)):.1f}")
    print(f"         size min/max = {int(size_arr.min())}/{int(size_arr.max())}")

    # rank by absolute distance from the median; break ties by size
    ranked = sorted(sizes, key=lambda ps: (abs(ps[1] - median_size), ps[1]))
    selected = ranked[:n_around]
    selected_sizes = np.array([s for _, s in selected])
    print(f"[select] taking {len(selected)} closest to median "
          f"(range {int(selected_sizes.min())}–{int(selected_sizes.max())})")

    cascades: list[CascadeResult] = []
    for i, (jf, _) in enumerate(selected):
        if i > 0 and i % 25 == 0:
            print(f"  ... loaded {i}/{len(selected)} cascades")
        c = _load_full_cascade(jf)
        if c is not None:
            cascades.append(c)
    print(f"[load] fully parsed {len(cascades)} cascades")

    pop_info = {
        "pop_size":        int(len(sizes)),
        "pop_size_median": float(np.median(size_arr)),
        "pop_size_mean":   float(np.mean(size_arr)),
        "pop_size_min":    int(size_arr.min()),
        "pop_size_max":    int(size_arr.max()),
        "selected_min":    int(selected_sizes.min()),
        "selected_max":    int(selected_sizes.max()),
        "selected_median": float(np.median(selected_sizes)),
        "selected_mean":   float(np.mean(selected_sizes)),
    }
    return cascades, pop_info


# evaluation


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


# main


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("WEIBO NATURAL-RANGE VALIDATION (101 cascades around the median)")
    print("=" * 90)

    print("\n[1/3] Selecting cascades ...")
    cascades, pop_info = select_median_centred_cascades(
        min_size=MIN_CASCADE_SIZE,
        n_around=N_AROUND_MEDIAN,
    )
    if not cascades:
        print("No cascades matched criteria — aborting.")
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
    print(f"RESULTS: Weibo natural-range (median-101)  "
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
        "dataset":          "Weibo",
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

    out_path = DATA_DIR / "validate_weibo_natural_range.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
