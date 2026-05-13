#!/usr/bin/env python
"""try every non-empty subset of the 6 structural features on ic-ba cascades and log average_precision to mlflow"""
from __future__ import annotations

import random
from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import mlflow

from src.data.cascade import r0_to_params, IndependentCascade, CascadeResult
from src.data.networks import generate_ba_network
from src.features.extract import build_feature_matrix
from src.models.random_forest import SourceRandomForest
from src.evaluation.metrics import evaluate_ranker

# -- Configuration --
N_NODES = 200
BA_M = 3
R0_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
CASCADE_SIZE = 30
N_TARGET = 300  # Smaller target specifically for ablation to save time
SEED = 42

def generate_mixed_data(seed: int) -> tuple[list[CascadeResult], list[float]]:
    """Generate a mix of cascades across all R0 values."""
    G = generate_ba_network(n=N_NODES, m=BA_M, seed=seed)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(seed)

    all_cascades = []
    all_r0s = []
    sim_seed = seed * 1000

    print(f"Generating {N_TARGET} cascades of size {CASCADE_SIZE} per R0...")
    for r0 in R0_VALUES:
        p = r0_to_params(r0, avg_deg, model="IC")["p"]
        model = IndependentCascade(p=p)
        collected = 0
        attempts = 0
        while collected < N_TARGET:
            source = rng.choice(nodes)
            c = model.run(G, source=source, seed=sim_seed, max_size=CASCADE_SIZE)
            sim_seed += 1
            attempts += 1
            if c.size >= CASCADE_SIZE:
                all_cascades.append(c)
                all_r0s.append(r0)
                collected += 1
        print(f"  R0={r0:.1f} collected {collected} (attempts={attempts})")

    return all_cascades, all_r0s


def main():
    # 1. Generate Data
    cascades, r0s = generate_mixed_data(SEED)
    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]

    print(f"\nDataset size: {len(X)} nodes across {len(cascades)} cascades.")
    print(f"Total features available: {len(feature_names)} -> {feature_names}")

    expected_features = {
        "degree_zscore",
        "largest_comp_fraction_zscore",
        "nodes_within_2_hops_zscore",
        "jordan_center_dist",
        "cascade_diameter",
        "cascade_leaves",
        "endpoint_balance",
    }
    assert set(feature_names) == expected_features, f"Unexpected features: {feature_names}"

    # 2. Train / Test Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train_full, y_train = X[train_idx], y[train_idx]
    X_test_full, y_test = X[test_idx], y[test_idx]

    # Setup test set cascades explicitly
    cascade_id_to_obj = {i: c for i, c in enumerate(cascades)}
    test_cascade_ids = sorted(set(groups[i] for i in test_idx))
    test_cascades = [cascade_id_to_obj[cid] for cid in test_cascade_ids]

    # 3. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Ablation_Study_BA")

    # 4. Generate all combinations
    all_subsets = []
    for r in range(1, len(feature_names) + 1):
        all_subsets.extend(combinations(feature_names, r))
    
    print(f"\nEvaluating {len(all_subsets)} feature combinations...")
    
    results = []

    for idx, subset in enumerate(all_subsets, 1):
        subset_list = list(subset)
        subset_str = ", ".join(subset_list)
        
        # Get indices of the selected features
        feat_indices = [feature_names.index(f) for f in subset_list]
        
        # Slice datasets
        X_tr = X_train_full[:, feat_indices]
        X_te = X_test_full[:, feat_indices]

        # Adjust max_features
        # Best hyperparams dictate max_features=3, but if subset is smaller, clip it
        mf = min(3, len(subset_list))

        rf = SourceRandomForest(
            n_estimators=100,  # Reduced from 500 for ablation speed
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=5,
            max_features=mf,
            random_state=SEED,
            n_jobs=-1
        )
        
        # Train
        rf.fit(X_tr, y_train, feature_names=subset_list)

        # Evaluate Overall Performance
        y_score = rf.predict_proba(X_te)
        pr_auc = average_precision_score(y_test, y_score)
        
        rf_rankings = [rf.rank_nodes(c) for c in test_cascades]
        metrics = evaluate_ranker(test_cascades, rf_rankings, ks=[1, 3])
        
        top1 = metrics["top_k"][1] * 100
        top3 = metrics["top_k"][3] * 100
        mrr = metrics["mrr"]

        # Log to MLflow
        with mlflow.start_run(run_name=f"Subset_{idx}_{len(subset_list)}feats"):
            mlflow.log_param("features", subset_str)
            mlflow.log_param("num_features", len(subset_list))
            mlflow.log_param("max_features_used", mf)
            for f in feature_names:
                mlflow.log_param(f"has_{f}", int(f in subset_list))
            
            mlflow.log_metric("PR_AUC", pr_auc)
            mlflow.log_metric("Top1_Accuracy", top1)
            mlflow.log_metric("Top3_Accuracy", top3)
            mlflow.log_metric("MRR", mrr)

        results.append({
            "features": subset_str,
            "num_features": len(subset_list),
            "top1": top1,
            "top3": top3,
            "mrr": mrr,
            "pr_auc": pr_auc
        })
        
        print(f"[{idx}/{len(all_subsets)}] Features: {len(subset_list):d} -> Top-1: {top1:.1f}%")

    # 5. Print Top 5
    print("\n=== TOP 5 FEATURE COMBINATIONS BY TOP-1 ===")
    results.sort(key=lambda x: x["top1"], reverse=True)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. Top-1: {res['top1']:.1f}% | MRR: {res['mrr']:.3f} | PR-AUC: {res['pr_auc']:.3f} | Features: [{res['features']}]")

    print("\n=== TOP 5 FEATURE COMBINATIONS BY MRR ===")
    results.sort(key=lambda x: x["mrr"], reverse=True)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. MRR: {res['mrr']:.3f} | Top-1: {res['top1']:.1f}% | PR-AUC: {res['pr_auc']:.3f} | Features: [{res['features']}]")

if __name__ == "__main__":
    main()
