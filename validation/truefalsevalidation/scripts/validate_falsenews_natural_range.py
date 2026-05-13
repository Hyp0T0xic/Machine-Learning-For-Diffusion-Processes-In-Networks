#!/usr/bin/env python
"""evaluate rf models on falsenews cascades in the iqr size range [42, 320] (untruncated), dump metrics to json"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np

from validation.truefalsevalidation.scripts.load_falsenews import load_falsenews_cascades
from src.baselines.centrality import jordan_center, degree_rank
from src.evaluation.metrics import evaluate_ranker

# ── Config ────────────────────────────────────────────────────────────────────

MIN_SIZE  = 42
MAX_SIZE  = 320
SEEDS     = [42, 123, 456, 789, 1024]
DATA_DIR  = _REPO_ROOT / "validation" / "truefalsevalidation" / "data"

MODELS = {
    "rf_falsenews": _REPO_ROOT / "validation" / "truefalsevalidation" / "models" / "rf_model_falsenews_size25.pkl",
    "RF (IC-BA)":   _REPO_ROOT / "results" / "models" / "ic_ba" / "rf_model_size25.pkl",
    "RF (IC-ER)":   _REPO_ROOT / "results" / "models" / "ic_er" / "rf_model_size25.pkl",
}

METHOD_LABELS = {
    "rf_falsenews": "RF (FalseNews)",
    "RF (IC-BA)":   "RF (IC-BA)",
    "RF (IC-ER)":   "RF (IC-ER)",
    "jordan":       "Jordan Centre",
    "degree":       "Degree",
    "random":       "Random",
}
METHOD_ORDER = list(METHOD_LABELS.keys())


# ── Evaluation ────────────────────────────────────────────────────────────────

def _run_seed(cascades, models, seed):
    random.seed(seed)
    metrics = {}

    for name, model in models.items():
        rankings = [model.rank_nodes(c) for c in cascades]
        metrics[name] = evaluate_ranker(cascades, rankings, ks=[1, 3])

    metrics["jordan"] = evaluate_ranker(
        cascades, [jordan_center(c) for c in cascades], ks=[1, 3],
    )
    metrics["degree"] = evaluate_ranker(
        cascades, [degree_rank(c) for c in cascades], ks=[1, 3],
    )

    rng = random.Random(seed)
    random_rankings = []
    for c in cascades:
        nodes = list(c.observed_graph.nodes())
        rng.shuffle(nodes)
        random_rankings.append(nodes)
    metrics["random"] = evaluate_ranker(cascades, random_rankings, ks=[1, 3])

    return metrics


def _aggregate(all_seed_metrics):
    avg = {}
    for method in METHOD_ORDER:
        t1, t3, mrr = [], [], []
        for sm in all_seed_metrics:
            if method in sm:
                t1.append(sm[method]["top_k"][1])
                t3.append(sm[method]["top_k"][3])
                mrr.append(sm[method]["mrr"])
        if t1:
            avg[method] = {
                "top_1":     float(np.mean(t1)),
                "top_3":     float(np.mean(t3)),
                "top_1_std": float(np.std(t1)),
                "top_3_std": float(np.std(t3)),
                "mrr":       float(np.mean(mrr)),
                "mrr_std":   float(np.std(mrr)),
            }
    return avg


def _print_results(avg, n_cascades):
    print(f"\n{'='*70}")
    print(f"  FalseNews natural range [{MIN_SIZE}, {MAX_SIZE}]  |  "
          f"n={n_cascades}  |  {len(SEEDS)} seeds")
    print(f"{'='*70}")
    print(f"{'Method':<22}  {'Top-1':>14}  {'Top-3':>14}  {'MRR':>8}")
    print("-" * 70)
    for method in METHOD_ORDER:
        m = avg.get(method)
        if m:
            print(f"{METHOD_LABELS[method]:<22}  "
                  f"{100*m['top_1']:>5.1f}±{100*m['top_1_std']:>4.1f}%  "
                  f"{100*m['top_3']:>5.1f}±{100*m['top_3_std']:>4.1f}%  "
                  f"{m['mrr']:>8.4f}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FALSENEWS NATURAL RANGE VALIDATION  [{MIN_SIZE}, {MAX_SIZE}] nodes")
    print(f"{'='*70}\n")

    print("[1/3] Loading FalseNews cascades ...")
    all_cascades, metadata = load_falsenews_cascades(
        target_size=MAX_SIZE, min_size=MIN_SIZE,
    )
    # keep only cascades that were not truncated (original size within range)
    cascades = [
        c for c, m in zip(all_cascades, metadata)
        if m["original_size"] <= MAX_SIZE
    ]
    sizes = [c.observed_graph.number_of_nodes() for c in cascades]
    print(f"  Natural-range cascades: {len(cascades):,}  "
          f"(sizes {min(sizes)}–{max(sizes)}, mean {np.mean(sizes):.0f}, "
          f"median {np.median(sizes):.0f})")

    print("\n[2/3] Loading models ...")
    models = {}
    for name, path in MODELS.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"  Loaded: {METHOD_LABELS[name]} <- {path.name}")
        else:
            print(f"  WARNING: not found -- {path}")

    print(f"\n[3/3] Evaluating ({len(SEEDS)} seeds) ...")
    all_seed_metrics = []
    for i, seed in enumerate(SEEDS):
        print(f"  -- SEED {i+1}/{len(SEEDS)}: {seed} --")
        all_seed_metrics.append(_run_seed(cascades, models, seed))

    avg = _aggregate(all_seed_metrics)
    _print_results(avg, len(cascades))

    out = {
        "min_size":     MIN_SIZE,
        "max_size":     MAX_SIZE,
        "n_cascades":   len(cascades),
        "n_seeds":      len(SEEDS),
        "size_mean":    float(np.mean(sizes)),
        "size_median":  float(np.median(sizes)),
        "method_labels": METHOD_LABELS,
        "method_order":  METHOD_ORDER,
        "metrics":       avg,
    }
    out_path = DATA_DIR / "validate_falsenews_natural_range.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
