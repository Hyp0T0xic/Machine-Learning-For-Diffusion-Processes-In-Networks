"""
scripts/experiments/validate_weibo.py
=====================================
Validate trained Random Forest models on real Weibo cascade data.

This script:
1. Parses Weibo repost-cascade JSON files into CascadeResult objects.
2. Loads the 4 saved RF models (IC-BA, IC-ER, SI-BA, SI-ER).
3. Evaluates each model's Top-1 and Top-3 accuracy on the real cascades.
4. Compares against Jordan Center and Degree baselines.
5. Saves a summary figure to results/figures/ml_evaluation/.

Usage
-----
    uv run python scripts/experiments/validate_weibo.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.cascade import CascadeResult
from src.baselines.centrality import jordan_center, degree_rank

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEIBO_DIR = PROJECT_ROOT / "rumdect" / "Weibo"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
FIG_DIR = PROJECT_ROOT / "validation" / "weibovalidation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cascade size filter: test on medium-large cascades
MIN_CASCADE_SIZE = 500
MAX_CASCADE_SIZE = 2000
MAX_CASCADES = 100  # Lower limit because large cascades are slow to process


# ── Weibo → CascadeResult conversion ───────────────────────────────────────

def load_weibo_cascade(json_path: Path) -> CascadeResult | None:
    """Parse one Weibo event JSON into a CascadeResult.

    The first entry (with parent=null) is the source.
    We build an undirected cascade graph from parent→child repost links.
    Node IDs are remapped to integers 0..N-1 for compatibility.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            posts = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    if not posts or len(posts) < MIN_CASCADE_SIZE:
        return None

    # Build a mapping from post mid → post data
    mid_to_post = {}
    source_mid = None
    for post in posts:
        mid = str(post.get("mid") or post.get("id"))
        mid_to_post[mid] = post
        if post.get("parent") is None:
            source_mid = mid

    if source_mid is None:
        return None

    # Build cascade edges from parent links
    # Remap string MIDs to integer node IDs
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

    # Sort posts by timestamp to assign infection times
    sorted_posts = sorted(posts, key=lambda p: p.get("t", 0))

    t0 = sorted_posts[0].get("t", 0) if sorted_posts else 0

    for post in sorted_posts:
        mid = str(post.get("mid") or post.get("id"))
        node_id = get_id(mid)
        parent_mid = post.get("parent")

        # Assign a relative infection time
        timestamp = post.get("t", 0)
        rel_time = max(0, timestamp - t0)
        # Bin into discrete steps (each step ~= 1 hour)
        infection_times[node_id] = rel_time // 3600

        if parent_mid is not None:
            parent_mid_str = str(parent_mid)
            if parent_mid_str in mid_to_int:
                parent_id = mid_to_int[parent_mid_str]
                cascade_edges.append((parent_id, node_id))

    # Filter by size
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
    """Load and filter Weibo cascades."""
    json_files = sorted(WEIBO_DIR.glob("*.json"))
    print(f"Found {len(json_files)} Weibo event files.")

    cascades = []
    skipped = 0
    for idx, jf in enumerate(json_files):
        if len(cascades) >= max_count:
            break
        if idx > 0 and idx % 200 == 0:
            print(f"  ... Scanned {idx}/{len(json_files)} files (found {len(cascades)} large cascades so far)")
            
        result = load_weibo_cascade(jf)
        if result is not None:
            cascades.append(result)
            print(f"  -> Found large cascade: {result.size} nodes (Total collected: {len(cascades)}/{max_count})")
        else:
            skipped += 1

    print(f"Loaded {len(cascades)} cascades (skipped {skipped}).")
    print(f"  Size range: {min(c.size for c in cascades)}-{max(c.size for c in cascades)}")
    print(f"  Median size: {np.median([c.size for c in cascades]):.0f}")
    return cascades


# ── Evaluation ──────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]

def evaluate_method(cascades: list[CascadeResult], rank_fn, name: str) -> dict:
    """Evaluate a ranking method on all cascades."""
    top1_correct = 0
    top3_correct = 0
    total = 0

    for i, cascade in enumerate(cascades):
        if i > 0 and i % 10 == 0:
            print(f"    [{name}] evaluated {i}/{len(cascades)} cascades...")
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


# ── Plotting (matches train_rf_ic_ba.py style) ─────────────────────────────

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


def _plot_accuracy(avg_metrics: dict) -> Path:
    """Plot grouped bars with error bars, matching the FalseNews style."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#0d0d1a")

    methods = [m for m in METHOD_ORDER if m in avg_metrics]
    x = np.arange(len(methods))
    bar_w = 0.6

    for ax, k_measure, k_std, title in zip(axes,
            ["top1", "top3"], ["top1_std", "top3_std"],
            [f"Top-1 Accuracy — Weibo Real-World Validation (mean over {len(SEEDS)} seeds)",
             f"Top-3 Accuracy — Weibo Real-World Validation (mean over {len(SEEDS)} seeds)"]):
        ax.set_facecolor("#1a1a2e")

        vals = [avg_metrics[m][k_measure] for m in methods]
        errs = [avg_metrics[m][k_std] for m in methods]
        colors = [PALETTE.get(m, "#888888") for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]

        bars = ax.bar(x, vals, bar_w, yerr=errs,
                      color=colors, edgecolor="black", linewidth=0.5,
                      capsize=4, error_kw={"ecolor": "white", "alpha": 0.6})

        # Value labels on top of bars
        for bar_obj, val, err in zip(bars, vals, errs):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + err + 1,
                    f"{val:.1f}±{err:.1f}%", ha="center", va="bottom",
                    color="white", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="lightgray", fontsize=9, rotation=15, ha="right")
        k_num = 1 if k_measure == "top1" else 3
        ax.set_ylabel(f"Top-{k_num} Accuracy (%)", color="lightgray")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.tick_params(colors="lightgray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    out_file = FIG_DIR / "weibo_validation_large.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot -> {out_file}")
    return out_file


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WEIBO REAL-WORLD VALIDATION")
    print("=" * 60)

    # 1. Load cascades (once — data is fixed)
    print("\n[1/3] Loading Weibo cascades...")
    cascades = load_weibo_cascades()

    # 2. Load models
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

    # 3. Evaluate across seeds
    print(f"\n[3/3] Evaluating on Weibo cascades ({len(SEEDS)} seeds)...")
    all_seed_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        seed_results = _run_seed(cascades, models, seed)
        all_seed_results.append(seed_results)

    # 4. Aggregate and print
    avg_metrics = _aggregate_seeds(all_seed_results)

    print("\n" + "=" * 60)
    print(f"RESULTS: Weibo Real-World Validation (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 60)
    print(f"{'Method':<20} | {'Top-1 Acc':>14} | {'Top-3 Acc':>14} | {'N':>6}")
    print("-" * 65)
    for m in METHOD_ORDER:
        if m in avg_metrics:
            a = avg_metrics[m]
            print(f"{m:<20} | {a['top1']:>5.1f}±{a['top1_std']:>4.1f}% "
                  f"| {a['top3']:>5.1f}±{a['top3_std']:>4.1f}% | {a['total']:>6}")

    # 5. Plot
    _plot_accuracy(avg_metrics)


if __name__ == "__main__":
    main()

