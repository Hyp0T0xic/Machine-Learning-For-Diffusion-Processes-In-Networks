#!/usr/bin/env python
"""
simulate.py
===========
Generate SIR cascades on contact networks and save results.

Main entry point for running diffusion cascades on all three contact networks
across a range of R₀ values, print statistics, save results, and
generate cascade-tree visualisations.

Usage
-----
    python src/simulate.py
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.networks.generator import compute_network_stats
from src.models import r0_to_params, CascadeResult
from src.diffusion.simulator import (
    select_sources,
    run_experiment,
    compute_cascade_stats,
    save_cascades,
)
from src.viz import plot_cascade_tree, plot_comparison_grid


# ── Configuration ───────────────────────────────────────────────────────────

NETWORK_DIR = "data/networks"
CASCADE_DIR = "data/raw"
VIZ_DIR = "results/figures"

NETWORK_FILES = {
    "ER": f"{NETWORK_DIR}/er_network.graphml",
    "BA": f"{NETWORK_DIR}/ba_network.graphml",
    "Complete": f"{NETWORK_DIR}/complete_network.graphml",
}

R0_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
MODEL_NAMES = ["IC", "SI", "SIR"]
N_SOURCES = 5
N_RUNS = 1          # stochastic runs per source (keep low for clarity)
SEED = 42
SIR_GAMMA = 0.2     # recovery probability for SIR


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. Load networks ────────────────────────────────────────────────
    print("Loading networks …")
    networks: dict[str, nx.Graph] = {}
    net_stats: dict[str, dict] = {}

    for name, path in NETWORK_FILES.items():
        G = nx.read_graphml(path)
        # Relabel nodes to integers (GraphML stores them as strings)
        G = nx.convert_node_labels_to_integers(G)
        networks[name] = G
        net_stats[name] = compute_network_stats(G)
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"⟨k⟩={net_stats[name]['avg_degree']}")

    # ── 2. Select sources (shared across experiments) ───────────────────
    sources_per_net: dict[str, list[int]] = {}
    for name, G in networks.items():
        sources_per_net[name] = select_sources(G, N_SOURCES, seed=SEED)
        print(f"  {name} sources: {sources_per_net[name]}")

    # ── 3. Run experiments ──────────────────────────────────────────────
    all_results: list[CascadeResult] = []
    stats_rows: list[dict] = []

    print("\n" + "=" * 100)
    header = (
        f"{'Network':<12} {'Model':<5} {'R₀':>5} {'Source':>7} "
        f"{'Size':>6} {'Depth':>6} {'R₀_act':>7} {'Cover%':>8} {'AvgPath':>8}"
    )
    print(header)
    print("-" * 100)

    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        sources = sources_per_net[net_name]

        for r0 in R0_VALUES:
            for model_name in MODEL_NAMES:
                params = r0_to_params(r0, avg_deg, model_name, gamma=SIR_GAMMA)
                results = run_experiment(
                    G,
                    model_name=model_name,
                    model_params=params,
                    sources=sources,
                    n_runs=N_RUNS,
                    seed=SEED,
                    network_name=net_name,
                )
                all_results.extend(results)

                for res in results:
                    s = compute_cascade_stats(res, G)
                    stats_rows.append(s)
                    print(
                        f"{net_name:<12} {model_name:<5} {r0:>5.1f} {s['source']:>7} "
                        f"{s['size']:>6} {s['depth']:>6} {s['actual_r0']:>7.3f} "
                        f"{s['coverage_pct']:>7.1f}% {s['avg_path_from_source']:>8.2f}"
                    )

    print("=" * 100)
    print(f"\nTotal cascades: {len(all_results)}")

    # ── 4. Save results ─────────────────────────────────────────────────
    json_path = save_cascades(all_results, output_dir=CASCADE_DIR)
    print(f"Cascade data saved → {json_path}")

    # ── 5. Visualise sample cascades ────────────────────────────────────
    print("\nRendering cascade-tree visualisations …")

    # Pick a representative R₀ for visualisations
    viz_r0 = 2.0
    viz_count = 0

    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        source = sources_per_net[net_name][0]  # first source

        for model_name in MODEL_NAMES:
            params = r0_to_params(viz_r0, avg_deg, model_name, gamma=SIR_GAMMA)
            # Find matching result
            matches = [
                r for r in all_results
                if r.network_name == net_name
                and r.model_name == model_name
                and r.source == source
                and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
            ]
            if matches:
                res = matches[0]
                if res.size > 1:
                    save_to = f"{VIZ_DIR}/{net_name}_{model_name}_r0_{viz_r0}_src{source}.png"
                    plot_cascade_tree(res, G, save_path=save_to)
                    plt.close("all")
                    viz_count += 1

    # ── 6. Comparison grids ─────────────────────────────────────────────
    print("Rendering comparison grids …")
    # For each model, pick first shared-index source and compare across networks
    for model_name in MODEL_NAMES:
        comparison: dict[str, CascadeResult] = {}
        for net_name in networks:
            avg_deg = net_stats[net_name]["avg_degree"]
            source = sources_per_net[net_name][0]
            params = r0_to_params(viz_r0, avg_deg, model_name, gamma=SIR_GAMMA)
            matches = [
                r for r in all_results
                if r.network_name == net_name
                and r.model_name == model_name
                and r.source == source
                and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
            ]
            if matches and matches[0].size > 1:
                comparison[net_name] = matches[0]

        if len(comparison) >= 2:
            grid_path = f"{VIZ_DIR}/comparison_{model_name}_r0_{viz_r0}.png"
            plot_comparison_grid(comparison, save_path=grid_path)
            plt.close("all")
            viz_count += 1

    print(f"Saved {viz_count} visualisation(s) → {VIZ_DIR}/")

    # ── 7. Summary statistics ───────────────────────────────────────────
    print("\n── Aggregate Summary ──")
    for model_name in MODEL_NAMES:
        model_stats = [s for s in stats_rows if s["model"] == model_name]
        if model_stats:
            avg_size = np.mean([s["size"] for s in model_stats])
            avg_r0 = np.mean([s["actual_r0"] for s in model_stats])
            avg_cov = np.mean([s["coverage_pct"] for s in model_stats])
            print(f"  {model_name}: avg_size={avg_size:.1f}, avg_R₀={avg_r0:.3f}, avg_coverage={avg_cov:.1f}%")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
