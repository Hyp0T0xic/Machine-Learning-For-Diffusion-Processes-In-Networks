#!/usr/bin/env python
"""
scripts/experiments/tune_rf_ba.py
=================================
Perform standard Machine Learning evaluation and Hyperparameter Tuning 
for the Random Forest model on the source detection task (BA network).

This script:
1. Generates a dataset of cascades.
2. Performs a Randomized Search Cross-Validation (with StratifiedGroupKFold) 
   to find the best hyperparameters, optimizing for Average Precision (PR-AUC) 
   due to the severe class imbalance (1 source vs 19 non-sources).
3. Evaluates the best model on a hold-out test set using standard ML metrics
   (ROC AUC, Precision-Recall AUC, Classification Report).
4. Plots ROC and Precision-Recall curves.
"""
from __future__ import annotations

import random
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from src.data.cascade import r0_to_params, IndependentCascade, CascadeResult
from src.data.networks import generate_ba_network
from src.features.extract import build_feature_matrix

# -- Configuration --
N_NODES = 200
BA_M = 3
R0_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
CASCADE_SIZE = 30
N_TARGET = 1500  # Smaller target per R0 for faster tuning (total 2500 cascades)
SEED = 42
OUT_DIR = Path("results/figures/ml_evaluation")


def generate_mixed_data(seed: int) -> list[CascadeResult]:
    """Generate a mix of cascades across all R0 values."""
    G = generate_ba_network(n=N_NODES, m=BA_M, seed=seed)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    nodes = list(G.nodes())
    rng = random.Random(seed)

    all_cascades = []
    sim_seed = seed * 1000

    print(f"Generating {N_TARGET} cascades of size {CASCADE_SIZE} per R0 for tuning...")
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
                collected += 1
        print(f"  R0={r0:.1f} collected {collected} (attempts={attempts})")

    return all_cascades


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Data
    cascades = generate_mixed_data(SEED)
    X, y, index, feature_names = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]  # cascade_id for grouping

    print(f"\nDataset size: {len(X)} nodes across {len(cascades)} cascades.")
    print(f"Class distribution: {sum(y==1)} sources, {sum(y==0)} non-sources.")

    # 2. Train / Test Split (Hold-out for final evaluation)
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf_outer.split(X, y, groups=groups))
    
    X_train, y_train = X[train_idx], y[train_idx]
    groups_train = [groups[i] for i in train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 3. Hyperparameter Tuning using Cross-Validation
    # We optimize for 'average_precision' (PR-AUC) because accuracy/ROC can be misleading 
    # when the dataset is heavily imbalanced (1:19 ratio).
    param_distributions = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    base_rf = RandomForestClassifier(class_weight="balanced", random_state=SEED)
    
    # Inner CV for tuning (so we don't leak groups across folds)
    sgkf_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    
    search = RandomizedSearchCV(
        base_rf,
        param_distributions=param_distributions,
        n_iter=15,  # Try 15 random combinations (increase for more thorough search)
        scoring='average_precision',
        cv=sgkf_inner,
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
        random_state=SEED
    )
    
    print("\nStarting Hyperparameter Tuning (RandomizedSearchCV)...")
    # Note: We must pass groups_train to fit so the inner CV splits properly
    search.fit(X_train, y_train, groups=groups_train)
    
    print("\n--- Tuning Results ---")
    print(f"Best PR-AUC score from CV: {search.best_score_:.4f}")
    print("Best Hyperparameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # 4. Final Evaluation on Hold-out Test Set
    best_model = search.best_estimator_
    
    print("\nEvaluating Best Model on Hold-Out Test Set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Non-Source", "Source"]))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"PR AUC Score:  {pr_auc:.4f}")

    # 5. Plot ROC and PR Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")
    
    for ax in [ax_roc, ax_pr]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="lightgray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.xaxis.label.set_color('lightgray')
        ax.yaxis.label.set_color('lightgray')
        ax.title.set_color('white')
        ax.title.set_fontweight('bold')

    RocCurveDisplay.from_predictions(
        y_test, y_proba, name="Random Forest", ax=ax_roc, color="#ffb703"
    )
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    legend = ax_roc.legend(facecolor="#222", edgecolor="#444", labelcolor="white")

    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, name="Random Forest", ax=ax_pr, color="#2ec4b6"
    )
    baseline_pr = sum(y_test) / len(y_test)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], color="gray", linestyle="--", label=f"Random Chance ({baseline_pr:.2f})")
    ax_pr.set_title("Precision-Recall Curve")
    legend = ax_pr.legend(facecolor="#222", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    out_file = OUT_DIR / "rf_ml_metrics_ba.png"
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ML metrics plots -> {out_file}")


if __name__ == "__main__":
    main()
