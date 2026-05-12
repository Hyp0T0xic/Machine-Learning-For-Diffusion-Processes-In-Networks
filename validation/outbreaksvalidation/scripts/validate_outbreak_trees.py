"""
scripts/experiments/validate_outbreak_trees.py
Validates the trained ML models against real-world biological transmission trees (OutbreakTrees).
"""

from pathlib import Path
import json
import csv
import collections
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREES_DIR = PROJECT_ROOT / "data" / "outbreak_trees" / "csv_exports"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "ml_evaluation"
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

def evaluate_method(cascades: list[CascadeResult],
                    rank_fn: Callable[[CascadeResult], list[int]],
                    name: str) -> tuple[float, float]:
    """Return (top_1_acc, top_3_acc)."""
    top1, top3 = 0, 0
    total = 0

    for i, cascade in enumerate(cascades):
        ranked = rank_fn(cascade)
            
        if not ranked:
            continue
            
        true_src = cascade.source
        
        if name == "Jordan Center" and i < 3:
            print(f"      [DEBUG] True source: {true_src}, Top ranked: {ranked[:5]}")
            
        if ranked[0] == true_src:
            top1 += 1
        if true_src in ranked[:3]:
            top3 += 1
        total += 1

    if total == 0:
        return 0.0, 0.0
    return (top1 / total), (top3 / total)


def plot_results(metrics: dict[str, dict[str, float]]) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes:
        ax.set_facecolor("#0d0d1a")

    top1_vals = [metrics[m]["top1"] * 100 for m in METHOD_ORDER]
    top3_vals = [metrics[m]["top3"] * 100 for m in METHOD_ORDER]
    colors = [PALETTE[m] for m in METHOD_ORDER]
    x_pos = np.arange(len(METHOD_ORDER))
    
    # Panel A: Top-1
    axes[0].bar(x_pos, top1_vals, color=colors, width=0.6, zorder=3)
    axes[0].set_title("Top-1 Accuracy", color="white", pad=15)
    axes[0].set_ylabel("Accuracy (%)", color="white")
    axes[0].set_ylim(0, 105)
    
    # Panel B: Top-3
    axes[1].bar(x_pos, top3_vals, color=colors, width=0.6, zorder=3)
    axes[1].set_title("Top-3 Accuracy", color="white", pad=15)
    axes[1].set_ylim(0, 105)

    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                           rotation=45, ha="right", color="white")
        ax.grid(True, axis="y", color="#333344", linestyle="--", alpha=0.5, zorder=0)
        ax.tick_params(axis="y", colors="white")
        for sp in ax.spines.values():
            sp.set_color("#444")

    plt.tight_layout()
    out_file = FIG_DIR / "outbreak_trees_validation.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
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

    # 3. Evaluate
    print("\n[3/3] Evaluating on biological cascades...")
    metrics = {}
    
    print("  Evaluating Jordan Center...")
    t1, t3 = evaluate_method(cascades, jordan_center, "Jordan Center")
    metrics["Jordan Center"] = {"top1": t1, "top3": t3}

    print("  Evaluating Degree...")
    t1, t3 = evaluate_method(cascades, degree_rank, "Degree")
    metrics["Degree"] = {"top1": t1, "top3": t3}

    for name in ["RF (IC-BA)", "RF (IC-ER)"]:
        if name in models:
            print(f"  Evaluating {name}...")
            clf = models[name]
            t1, t3 = evaluate_method(cascades, lambda c, m=clf: m.rank_nodes(c), name)
            metrics[name] = {"top1": t1, "top3": t3}
        else:
            metrics[name] = {"top1": 0.0, "top3": 0.0}

    # Print table
    print("\n" + "=" * 60)
    print("RESULTS: OutbreakTrees Biological Validation")
    print("=" * 60)
    print(f"{'Method':<20} | {'Top-1 Acc':>10} | {'Top-3 Acc':>10} | {'N':>6}")
    print("-" * 55)
    for m in METHOD_ORDER:
        t1 = metrics[m]["top1"] * 100
        t3 = metrics[m]["top3"] * 100
        print(f"{m:<20} | {t1:>9.1f}% | {t3:>9.1f}% | {len(cascades):>6}")

    plot_results(metrics)


if __name__ == "__main__":
    main()
