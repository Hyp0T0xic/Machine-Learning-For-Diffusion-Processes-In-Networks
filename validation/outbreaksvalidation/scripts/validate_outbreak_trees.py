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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import networkx as nx
import numpy as np

from src.data.cascade import CascadeResult
from src.baselines.centrality import jordan_center, degree_rank

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
    "Jordan Center": "Jordan Centre",
    "Degree":        "Degree",
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

def evaluate_method(cascades: list[CascadeResult], rank_fn, name: str) -> dict:
    top1_correct = 0
    top3_correct = 0
    total = 0

    for cascade in cascades:
        try:
            ranked = rank_fn(cascade)
        except Exception:
            continue

        if not ranked:
            continue

        source = cascade.source
        total += 1
        if ranked[0] == source:
            top1_correct += 1
        if source in ranked[:3]:
            top3_correct += 1

    top1_acc = (top1_correct / total * 100) if total > 0 else 0
    top3_acc = (top3_correct / total * 100) if total > 0 else 0
    return {"name": name, "top1": top1_acc, "top3": top3_acc, "total": total}


def _run_seed(cascades, models, seed):
    random.seed(seed)
    np.random.seed(seed)

    results = {}
    results["Jordan Center"] = evaluate_method(cascades, jordan_center, "Jordan Center")
    results["Degree"] = evaluate_method(cascades, degree_rank, "Degree")
    for name, model in models.items():
        results[name] = evaluate_method(cascades, model.rank_nodes, name)

    return results


def _aggregate_seeds(all_seed_results):
    avg = {}
    for method in METHOD_ORDER:
        t1_vals = [sr[method]["top1"] for sr in all_seed_results if method in sr]
        t3_vals = [sr[method]["top3"] for sr in all_seed_results if method in sr]
        if t1_vals:
            avg[method] = {
                "top_1":     float(np.mean(t1_vals)),
                "top_3":     float(np.mean(t3_vals)),
                "top_1_std": float(np.std(t1_vals)),
                "top_3_std": float(np.std(t3_vals)),
                "total":     all_seed_results[0][method]["total"],
            }
    return avg


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
    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        all_seed_results.append(_run_seed(cascades, models, seed))

    avg_metrics = _aggregate_seeds(all_seed_results)

    print("\n" + "=" * 60)
    print(f"RESULTS: OutbreakTrees Biological Validation (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 60)
    print(f"{'Method':<20} | {'Top-1 Acc':>14} | {'Top-3 Acc':>14} | {'N':>6}")
    print("-" * 65)
    for m in METHOD_ORDER:
        if m in avg_metrics:
            a = avg_metrics[m]
            print(f"{m:<20} | {a['top_1']:>5.1f}±{a['top_1_std']:>4.1f}% "
                  f"| {a['top_3']:>5.1f}±{a['top_3_std']:>4.1f}% | {a['total']:>6}")

    json_data = {
        "n_cascades":       len(cascades),
        "n_seeds":          len(SEEDS),
        "min_cascade_size": MIN_CASCADE_SIZE,
        "max_cascade_size": MAX_CASCADE_SIZE,
        "method_labels":    METHOD_LABELS,
        "method_order":     METHOD_ORDER,
        "metrics":          avg_metrics,
    }

    out_path = DATA_DIR / "validate_outbreak_trees.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
