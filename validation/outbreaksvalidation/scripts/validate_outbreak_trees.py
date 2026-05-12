"""
scripts/experiments/validate_outbreak_trees.py
Validates the trained ML models against real-world biological transmission trees (OutbreakTrees).
"""

from pathlib import Path
import json
import csv
import collections
import random
from typing import Callable

import joblib
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.cascade import CascadeResult
from src.baselines.centrality import jordan_center, degree_rank

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TREES_DIR = PROJECT_ROOT / "data" / "outbreak_trees" / "csv_exports"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
FIG_DIR = PROJECT_ROOT / "validation" / "outbreaksvalidation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cascade size filter
MIN_CASCADE_SIZE = 15
MAX_CASCADE_SIZE = 300
MAX_CASCADES = 400


# ── Load CascadeResult ──────────────────────────────────────────────────────

def load_outbreak_tree(csv_path: Path) -> CascadeResult | None:
    """Parse one OutbreakTrees CSV into a CascadeResult.
    
    Improvements:
    - Robust ID mapping (tolerant to non-integer IDs)
    - Verification that graph is a valid arborescence
    - Verification that BFS reaches all nodes
    """
    raw_edges = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_edges.append((row["from"], row["to"]))
            
    if not raw_edges:
        return None
        
    # Map potentially non-integer IDs to consistent 0..N integers
    id_map = {}
    def get_id(raw_val):
        if raw_val not in id_map:
            id_map[raw_val] = len(id_map)
        return id_map[raw_val]

    edges = [(get_id(f), get_id(t)) for f, t in raw_edges]
    
    # Create directed graph
    G_dir = nx.DiGraph()
    G_dir.add_edges_from(edges)
    
    # 1. Structural check: must be a valid arborescence (directed tree)
    if not nx.is_arborescence(G_dir):
        return None
    
    # 2. Identify the source (node with in-degree 0)
    sources = [n for n, d in G_dir.in_degree() if d == 0]
    if len(sources) != 1:
        return None
    source = sources[0]
    
    # 3. Calculate infection times via BFS
    infection_times = {source: 0}
    queue = collections.deque([(source, 0)])
    
    while queue:
        curr, depth = queue.popleft()
        for child in G_dir.successors(curr):
            if child not in infection_times:
                infection_times[child] = depth + 1
                queue.append((child, depth + 1))
                
    # 4. Verify BFS covers all nodes (reachability)
    if len(infection_times) != G_dir.number_of_nodes():
        return None

    # Filter by size
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
    """Load and filter all exported trees."""
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
        print(f"  Size range: {min(c.size for c in cascades)}-{max(c.size for c in cascades)}")
        print(f"  Median size: {np.median([c.size for c in cascades]):.0f}")
    return cascades


# ── Plotting Style ─────────────────────────────────────────────────────────

METHOD_LABELS = {
    "RF (IC-BA)":    "Random Forest (IC-BA)",
    "RF (IC-ER)":    "Random Forest (IC-ER)",
    "Jordan Center": "Jordan Centre",
    "Degree":        "Degree",
}

METHOD_ORDER = [
    "RF (IC-BA)", "RF (IC-ER)",
    "Jordan Center", "Degree",
]

PALETTE = {
    "RF (IC-BA)":    "#ffb703",
    "RF (IC-ER)":    "#f4a261",
    "Jordan Center": "#e63946",
    "Degree":        "#c084fc",
}


# ── Evaluation logic ────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]

def evaluate_method(cascades: list[CascadeResult], rank_fn, name: str) -> dict:
    """Evaluate a ranking method on all cascades."""
    top1_correct = 0
    top3_correct = 0
    total = 0

    for i, cascade in enumerate(cascades):
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
    """Run one evaluation pass with a specific random seed."""
    random.seed(seed)
    np.random.seed(seed)

    results = {}

    # Baselines (affected by random tie-breaking)
    results["Jordan Center"] = evaluate_method(cascades, jordan_center, "Jordan Center")
    results["Degree"] = evaluate_method(cascades, degree_rank, "Degree")

    # RF Models (deterministic, but include for consistency)
    for name, model in models.items():
        results[name] = evaluate_method(cascades, model.rank_nodes, name)

    return results


def _aggregate_seeds(all_seed_results):
    """Average top-1 / top-3 across seeds per method."""
    avg = {}
    for method in METHOD_ORDER:
        t1_vals = [sr[method]["top1"] for sr in all_seed_results if method in sr]
        t3_vals = [sr[method]["top3"] for sr in all_seed_results if method in sr]
        if t1_vals:
            avg[method] = {
                "top1": np.mean(t1_vals),
                "top3": np.mean(t3_vals),
                "top1_std": np.std(t1_vals),
                "top3_std": np.std(t3_vals),
                "total": all_seed_results[0][method]["total"],
            }
    return avg


COLORS = ["#082a54", "#a559aa", "#59a89c", "#f0c571", "#e02b35", "#cecece"]


def plot_results(avg_metrics: dict) -> None:
    methods = [m for m in METHOD_ORDER if m in avg_metrics]
    x = np.arange(len(methods))

    n_cascades = avg_metrics[methods[0]]["total"] if methods else 0

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, key, title in zip(axes, ["top1", "top3"], [
        f"top-1 accuracy -- OutbreakTrees biological validation ($n={n_cascades}$)",
        f"top-3 accuracy -- OutbreakTrees biological validation ($n={n_cascades}$)",
    ]):
        vals = [avg_metrics[m][key] for m in methods]
        stds = [avg_metrics[m][key + "_std"] for m in methods]
        for xi, (val, std) in enumerate(zip(vals, stds)):
            ax.bar(xi, val, facecolor=COLORS[xi % len(COLORS)],
                   edgecolor="black", linewidth=0.5)
            lbl = f"{val:.1f}±{std:.1f}%" if std > 0.05 else f"{val:.1f}%"
            ax.text(xi, val + 1.5, lbl, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m) for m in methods],
            rotation=30, ha="right", fontsize=9,
        )
        k = 1 if key == "top1" else 3
        ax.set_ylabel(f"top-{k} accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 112)

    plt.tight_layout()
    out_file = FIG_DIR / "outbreak_trees_validation.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot -> {out_file}")


def main() -> None:
    print("=" * 60)
    print("OUTBREAK TREES REAL-WORLD VALIDATION")
    print("=" * 60)

    # 1. Load data
    print("\n[1/3] Loading OutbreakTrees cascades...")
    cascades = load_all_trees()
    if not cascades:
        print("No cascades matched criteria.")
        return

    # 2. Load ML models
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

    # 3. Evaluate across seeds
    print(f"\n[3/3] Evaluating on biological cascades ({len(SEEDS)} seeds)...")
    all_seed_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        seed_results = _run_seed(cascades, models, seed)
        all_seed_results.append(seed_results)

    # 4. Aggregate and print
    avg_metrics = _aggregate_seeds(all_seed_results)

    print("\n" + "=" * 60)
    print(f"RESULTS: OutbreakTrees Biological Validation (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 60)
    print(f"{'Method':<20} | {'Top-1 Acc':>14} | {'Top-3 Acc':>14} | {'N':>6}")
    print("-" * 65)
    for m in METHOD_ORDER:
        if m in avg_metrics:
            a = avg_metrics[m]
            print(f"{m:<20} | {a['top1']:>5.1f}±{a['top1_std']:>4.1f}% "
                  f"| {a['top3']:>5.1f}±{a['top3_std']:>4.1f}% | {a['total']:>6}")

    plot_results(avg_metrics)


if __name__ == "__main__":
    main()
