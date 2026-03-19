"""
src.models.random_forest
========================
Random Forest classifier for source detection.

Wraps scikit-learn's RandomForestClassifier to interface directly with
CascadeResult objects, scoring nodes and ranking them.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data.cascade import CascadeResult
from src.features.extract import extract_node_features


class SourceRandomForest:
    """Random Forest model to predict the source node of a cascade."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int | None = 10, random_state: int = 42, **kwargs):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",  # Critical: source vs non-source is highly imbalanced
            **kwargs
        )
        self._feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> SourceRandomForest:
        """Train the Random Forest on node-level features."""
        self.clf.fit(X, y)
        self._feature_names = feature_names
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of being the source (class 1) for each row in X."""
        # clf.classes_ is typically [0, 1]
        class_1_idx = list(self.clf.classes_).index(1)
        return self.clf.predict_proba(X)[:, class_1_idx]

    def rank_nodes(self, result: CascadeResult) -> list[int]:
        """Score all infected nodes in a cascade and return them ranked most-to-least likely.
        
        This mimics the signature of the centrality heuristics in src.baselines.centrality.
        """
        feats_dict = extract_node_features(result)
        if not feats_dict:
            return []
            
        nodes = list(feats_dict.keys())
        # Ensure features are ordered correctly
        X = np.array([[feats_dict[n][feat] for feat in self._feature_names] for n in nodes])
        
        probs = self.predict_proba(X)
        
        # Sort nodes by probability descending
        ranked_pairs = sorted(zip(nodes, probs), key=lambda x: x[1], reverse=True)
        return [n for n, p in ranked_pairs]

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return a dictionary of feature importances (Gini impurity decrease)."""
        if getattr(self.clf, "feature_importances_", None) is None:
            return {}
        return dict(zip(self._feature_names, self.clf.feature_importances_))
