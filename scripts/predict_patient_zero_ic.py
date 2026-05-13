#!/usr/bin/env python
"""run centrality baselines + random on ic cascades over k20 across r0 and save a 4-panel figure"""
from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

from src.data.cascade import r0_to_params, IndependentCascade
from src.baselines.centrality import predict_all
from src.visualization.theme import METHOD_COLORS

# config

N_NODES   = 20
R0_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
N_RUNS    = 200        # cascades per r0; only non-trivial (>1 node) ones count
BASE_SEED = 42
OUT_DIR   = Path("results/figures/ml_evaluation")
OUT_FILE  = OUT_DIR / "patient_zero_prediction_accuracy.png"

METHOD_LABELS = {
    "jordan":      "Jordan Centre",
    "closeness":   "Closeness",
    "betweenness": "Betweenness",
    "degree":      "Degree",
    "random":      "Random (baseline)",
}
METHOD_ORDER = list(METHOD_LABELS.keys())

def build_complete_graph(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def random_rank(result_size: int, rng: random.Random) -> int:
    """uniform random rank in [1, n]"""
    return rng.randint(1, result_size)


def evaluate_cascade(cascade, rng: random.Random) -> dict[str, dict]:
    """run every baseline on one cascade, return per-method {rank, top1, top3}"""
    true_source = cascade.source
    obs = cascade.observed_graph

    # nothing to rank if only the source ever got infected
    if obs.number_of_nodes() <= 1:
        return {}

    preds = predict_all(cascade)
    results = {}

    for method, ranked_nodes in preds.items():
        if true_source in ranked_nodes:
            rank = ranked_nodes.index(true_source) + 1
        else:
            rank = len(ranked_nodes) + 1
        results[method] = {
            "rank": rank,
            "top1": rank == 1,
            "top3": rank <= 3,
        }

    rand_rank = random_rank(obs.number_of_nodes(), rng)
    results["random"] = {
        "rank": rand_rank,
        "top1": rand_rank == 1,
        "top3": rand_rank <= 3,
    }
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    G = build_complete_graph(N_NODES)
    avg_deg = float(N_NODES - 1)
    nodes   = list(G.nodes())
    rng     = random.Random(BASE_SEED)

    # Accumulate: results[r0][method] = list of per-cascade metric dicts
    results: dict[float, dict[str, list[dict]]] = {}

    print(f"Graph  : Complete K_{N_NODES}, avg degree = {avg_deg:.0f}")
    print(f"Runs   : up to {N_RUNS} cascades per R₀ (trivial ones skipped)\n")

    for r0 in R0_VALUES:
        p = r0_to_params(r0, avg_deg, model="IC")["p"]
        model = IndependentCascade(p=p)
        per_method: dict[str, list[dict]] = defaultdict(list)
        n_evaluated = 0

        seed = BASE_SEED
        for _ in range(N_RUNS):
            source = rng.choice(nodes)
            cascade = model.run(G, source=source, seed=seed)
            seed += 1
            metrics = evaluate_cascade(cascade, rng)
            if not metrics:
                continue   # trivial cascade
            n_evaluated += 1
            for method, m in metrics.items():
                per_method[method].append(m)

        results[r0] = dict(per_method)
        print(f"R₀={r0:.1f}  p={p:.4f}  evaluated {n_evaluated}/{N_RUNS} cascades")

    print(f"\n{'method':<20}  " + "  ".join(f"r0={r:>3.1f}" for r in R0_VALUES))
    print(" " * 20 + "  " + "  ".join("top1  top3  mnrk" for _ in R0_VALUES))

    for method in METHOD_ORDER:
        row = f"{METHOD_LABELS[method]:<20}  "
        for r0 in R0_VALUES:
            ms = results[r0].get(method, [])
            if ms:
                t1   = 100 * np.mean([m["top1"] for m in ms])
                t3   = 100 * np.mean([m["top3"] for m in ms])
                mnrk = np.mean([m["rank"] for m in ms])
                row += f"{t1:>4.0f}% {t3:>4.0f}% {mnrk:>4.1f}  "
            else:
                row += "  —     —     —  "
        print(row)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        f"Patient Zero Prediction Accuracy — IC on K$_{{{N_NODES}}}$ "
        f"({N_RUNS} runs per R₀)",
        fontsize=13, fontweight="bold", color="white", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    panel_bg = "#1a1a2e"
    # only baselines here (no rf), so we pull just those colours
    palette  = {
        "jordan":      METHOD_COLORS["jordan"],
        "closeness":   METHOD_COLORS["closeness"],
        "betweenness": METHOD_COLORS["betweenness"],
        "degree":      METHOD_COLORS["degree"],
        "random":      METHOD_COLORS["random"],
    }
    x       = np.arange(len(R0_VALUES))
    bar_w   = 0.15
    offsets = np.linspace(-(len(METHOD_ORDER) - 1) / 2 * bar_w,
                          (len(METHOD_ORDER) - 1) / 2 * bar_w,
                          len(METHOD_ORDER))

    # panel a: top-1
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_facecolor(panel_bg)
    for i, method in enumerate(METHOD_ORDER):
        top1_vals = []
        for r0 in R0_VALUES:
            ms = results[r0].get(method, [])
            top1_vals.append(100 * np.mean([m["top1"] for m in ms]) if ms else 0)
        ax_a.bar(x + offsets[i], top1_vals, bar_w,
                 label=METHOD_LABELS[method], color=palette[method],
                 edgecolor="white", linewidth=0.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"R₀={r}" for r in R0_VALUES], color="lightgrey")
    ax_a.set_ylabel("Top-1 Accuracy (%)", color="lightgrey")
    ax_a.set_title("A  Top-1 Accuracy", color="lightgrey")
    ax_a.set_ylim(0, 105)
    ax_a.tick_params(colors="lightgrey")
    ax_a.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="lightgrey")
    for sp in ax_a.spines.values(): sp.set_edgecolor("#444")

    # panel b: top-3
    ax_b = fig.add_subplot(gs[1, :2])
    ax_b.set_facecolor(panel_bg)
    for i, method in enumerate(METHOD_ORDER):
        top3_vals = []
        for r0 in R0_VALUES:
            ms = results[r0].get(method, [])
            top3_vals.append(100 * np.mean([m["top3"] for m in ms]) if ms else 0)
        ax_b.bar(x + offsets[i], top3_vals, bar_w,
                 label=METHOD_LABELS[method], color=palette[method],
                 edgecolor="white", linewidth=0.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"R₀={r}" for r in R0_VALUES], color="lightgrey")
    ax_b.set_ylabel("Top-3 Accuracy (%)", color="lightgrey")
    ax_b.set_title("B  Top-3 Accuracy", color="lightgrey")
    ax_b.set_ylim(0, 105)
    ax_b.tick_params(colors="lightgrey")
    ax_b.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="lightgrey")
    for sp in ax_b.spines.values(): sp.set_edgecolor("#444")

    # panel c: mean rank (lower = better)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor(panel_bg)
    for method in METHOD_ORDER:
        mean_ranks = []
        for r0 in R0_VALUES:
            ms = results[r0].get(method, [])
            mean_ranks.append(np.mean([m["rank"] for m in ms]) if ms else np.nan)
        ax_c.plot(R0_VALUES, mean_ranks, marker="o", label=METHOD_LABELS[method],
                  color=palette[method], linewidth=2, markersize=5)

    ax_c.set_xlabel("R₀", color="lightgrey")
    ax_c.set_ylabel("Mean rank of true source", color="lightgrey")
    ax_c.set_title("C  Mean Rank (↓ better)", color="lightgrey")
    ax_c.tick_params(colors="lightgrey")
    ax_c.legend(fontsize=7, facecolor="#222", edgecolor="#444", labelcolor="lightgrey")
    for sp in ax_c.spines.values(): sp.set_edgecolor("#444")

    # panel d: how many of the n_runs cascades were actually non-trivial
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.set_facecolor(panel_bg)
    n_eval = [len(next(iter(results[r0].values()))) for r0 in R0_VALUES]
    bars = ax_d.bar(range(len(R0_VALUES)), n_eval,
                    color=plt.cm.plasma(np.linspace(0.2, 0.8, len(R0_VALUES))),
                    edgecolor="white", linewidth=0.5)
    ax_d.axhline(N_RUNS, ls="--", color="grey", lw=1, label="Total runs")
    ax_d.set_xticks(range(len(R0_VALUES)))
    ax_d.set_xticklabels([f"R₀={r}" for r in R0_VALUES], color="lightgrey", rotation=15, ha="right")
    ax_d.set_ylabel("Non-trivial cascades", color="lightgrey")
    ax_d.set_title("D  Evaluable cascades", color="lightgrey")
    ax_d.tick_params(colors="lightgrey")
    ax_d.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="lightgrey")
    for sp in ax_d.spines.values(): sp.set_edgecolor("#444")

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved → {OUT_FILE}")
    print("Done ✓")


if __name__ == "__main__":
    main()
