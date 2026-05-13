#!/usr/bin/env python
"""
Validate trained RF models (IC-BA, IC-ER) against real-world biological
transmission trees (OutbreakTrees). Compares to centrality baselines across
multiple seeds. Saves metrics to JSON for separate plotting.

Usage: python validation/outbreaksvalidation/scripts/validate_outbreak_trees.py
"""
from __future__ import annotations

import collections
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult
from src.baselines.centrality import predict_all
from src.evaluation.metrics import evaluate_ranker, distance_to_source

# ── Config ────────────────────────────────────────────────────────────────────

TREES_DIR        = _REPO_ROOT / "data" / "outbreak_trees" / "csv_exports"
MODELS_DIR       = _REPO_ROOT / "results" / "models"
DATA_DIR         = _REPO_ROOT / "validation" / "outbreaksvalidation" / "data"

MIN_CASCADE_SIZE = 15
MAX_CASCADE_SIZE = 300
MAX_CASCADES     = 400
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

def load_outbreak_tree(csv_path: Path) -> CascadeResult | None:
    raw_edges = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_edges.append((row["from"], row["to"]))

    if not raw_edges:
        return None

    id_map = {}
    def get_id(raw_val):
        if raw_val not in id_map:
            id_map[raw_val] = len(id_map)
        return id_map[raw_val]

    edges = [(get_id(f), get_id(t)) for f, t in raw_edges]

    G_dir = nx.DiGraph()
    G_dir.add_edges_from(edges)

    if not nx.is_arborescence(G_dir):
        return None

    sources = [n for n, d in G_dir.in_degree() if d == 0]
    if len(sources) != 1:
        return None
    source = sources[0]

    infection_times = {source: 0}
    queue = collections.deque([(source, 0)])
    while queue:
        curr, depth = queue.popleft()
        for child in G_dir.successors(curr):
            if child not in infection_times:
                infection_times[child] = depth + 1
                queue.append((child, depth + 1))

    if len(infection_times) != G_dir.number_of_nodes():
        return None

    if len(infection_times) < MIN_CASCADE_SIZE or len(infection_times) > MAX_CASCADE_SIZE:
        return None

    return CascadeResult(
        source=source,
        model_name="Biological_Tree",
        params={},
        infection_times=infection_times,
        cascade_edges=edges,
        network_name="OutbreakTrees",
    )


def load_all_trees(max_count: int = MAX_CASCADES) -> list[CascadeResult]:
    csv_files = sorted(TREES_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} OutbreakTrees CSV files.")

    cascades = []
    skipped = 0
    for cf in csv_files:
        if len(cascades) >= max_count:
            break
        result = load_outbreak_tree(cf)
        if result is not None:
            cascades.append(result)
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
    print("OUTBREAK TREES REAL-WORLD VALIDATION")
    print("=" * 60)

    print("\n[1/3] Loading OutbreakTrees cascades...")
    cascades = load_all_trees()
    if not cascades:
        print("No cascades matched criteria.")
        return

    print("\n[2/3] Loading saved models...")
    model_configs = {
        "RF (IC-BA)": MODELS_DIR / "ic_ba" / "rf_model_size25.pkl",
        "RF (IC-ER)": MODELS_DIR / "ic_er" / "rf_model_size25.pkl",
    }
    models = {}
    for name, p in model_configs.items():
        if p.exists():
            models[name] = joblib.load(p)
            print(f"  Loaded {name} from {p.name}")
        else:
            print(f"  [!] Missing model: {p}")

    print(f"\n[3/3] Evaluating on biological cascades ({len(SEEDS)} seeds)...")
    all_seed_results = []
    all_seed_distances = []
    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        metrics, distances = _run_seed(cascades, models, seed)
        all_seed_results.append(metrics)
        all_seed_distances.append(distances)

    avg_metrics, hop_distributions = _aggregate_seeds(all_seed_results, all_seed_distances)

    print("\n" + "=" * 90)
    print(f"RESULTS: OutbreakTrees Biological Validation (mean ± std over {len(SEEDS)} seeds)")
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

    out_path = DATA_DIR / "validate_outbreak_trees.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
