#!/usr/bin/env python
"""replot rf_ml_metrics_ba.png on a white background using the saved production rf model (no search needed)"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from scripts.experiments.train_rf_ic_ba import generate_data
from src.features.extract import build_feature_matrix
from src.visualization.theme import METHOD_COLORS

OUT_DIR    = _REPO_ROOT / "results" / "figures" / "ml_evaluation"
MODEL_PATH = _REPO_ROOT / "results" / "models" / "ic_ba" / "rf_model_size25.pkl"
SEED         = 42
CASCADE_SIZE = 25


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        print(f"model not found: {MODEL_PATH}")
        return

    print("loading production rf ...")
    rf = joblib.load(MODEL_PATH)

    # match the same split train_rf_ic_ba used for seed=42 so the test set is consistent
    print("re-generating ic-ba cascades for seed=42 ...")
    cascades, _ = generate_data(seed=SEED, cascade_size=CASCADE_SIZE)
    X, y, index, _ = build_feature_matrix(cascades)
    groups = [idx[0] for idx in index]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(X, y, groups=groups))
    X_test, y_test = X[test_idx], y[test_idx]

    y_proba = rf.predict_proba(X_test)

    rf_color = METHOD_COLORS["RF (IC-BA)"]

    plt.style.use("default")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    RocCurveDisplay.from_predictions(
        y_test, y_proba, name="Random Forest", ax=ax_roc, color=rf_color
    )
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_title("ROC")
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, name="Random Forest", ax=ax_pr, color=rf_color
    )
    baseline_pr = sum(y_test) / len(y_test)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], color="gray", linestyle="--",
               label=f"random chance ({baseline_pr:.2f})")
    ax_pr.set_title("Precision-Recall")
    ax_pr.legend()
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = OUT_DIR / "rf_ml_metrics_ba.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
