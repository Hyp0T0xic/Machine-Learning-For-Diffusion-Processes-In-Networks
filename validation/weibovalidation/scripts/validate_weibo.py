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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np

from src.data.cascade import CascadeResult
from src.baselines.centrality import jordan_center, degree_rank

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
    "Jordan Center": "Jordan Centre",
    "Degree":        "Degree",
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

def evaluate_method(cascades: list[CascadeResult], rank_fn, name: str) -> dict:
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
    for i, seed in enumerate(SEEDS):
        print(f"\n  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        all_seed_results.append(_run_seed(cascades, models, seed))

    avg_metrics = _aggregate_seeds(all_seed_results)

    print("\n" + "=" * 60)
    print(f"RESULTS: Weibo Real-World Validation (mean ± std over {len(SEEDS)} seeds)")
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

    out_path = DATA_DIR / "validate_weibo.json"
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
