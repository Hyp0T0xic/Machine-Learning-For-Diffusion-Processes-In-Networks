#!/usr/bin/env python
"""
scripts/run_simulation.py
=========================
Entry point — simulate IC/SI/SIR diffusion cascades on all three contact
networks across a grid of R₀ values, print statistics, save results, and
render cascade-tree visualisations.

Usage
-----
    python scripts/run_simulation.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data.networks import compute_network_stats
from src.data.cascade import r0_to_params, CascadeResult
from src.data.simulate import (
    select_sources, run_experiment, compute_cascade_stats, save_cascades,
)
from src.visualization.cascades import plot_cascade_tree, plot_comparison_grid

# ── Configuration ────────────────────────────────────────────────────────────

NETWORK_DIR = "data/networks"
CASCADE_DIR = "data/raw"
VIZ_DIR     = "results/figures"

NETWORK_FILES = {
    "ER":       f"{NETWORK_DIR}/er_network.graphml",
    "BA":       f"{NETWORK_DIR}/ba_network.graphml",
    "Complete": f"{NETWORK_DIR}/complete_network.graphml",
}

R0_VALUES   = [0.5, 1.0, 1.5, 2.0, 3.0]
MODEL_NAMES = ["IC", "SI", "SIR"]
N_SOURCES   = 5
N_RUNS      = 1
SEED        = 42
SIR_GAMMA   = 0.2


def main() -> None:
    # ── 1. Load networks ─────────────────────────────────────────────────
    print("Loading networks …")
    networks: dict[str, nx.Graph] = {}
    net_stats: dict[str, dict] = {}
    for name, path in NETWORK_FILES.items():
        G = nx.convert_node_labels_to_integers(nx.read_graphml(path))
        networks[name] = G
        net_stats[name] = compute_network_stats(G)
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"⟨k⟩={net_stats[name]['avg_degree']}")

    # ── 2. Select source nodes ───────────────────────────────────────────
    sources_per_net = {
        name: select_sources(G, N_SOURCES, seed=SEED)
        for name, G in networks.items()
    }

    # ── 3. Run experiments ───────────────────────────────────────────────
    all_results: list[CascadeResult] = []
    stats_rows: list[dict] = []
    print("\n" + "=" * 100)
    print(f"{'Network':<12} {'Model':<5} {'R₀':>5} {'Source':>7} "
          f"{'Size':>6} {'Depth':>6} {'R₀_act':>7} {'Cover%':>8} {'AvgPath':>8}")
    print("-" * 100)

    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        for r0 in R0_VALUES:
            for model_name in MODEL_NAMES:
                params = r0_to_params(r0, avg_deg, model_name, gamma=SIR_GAMMA)
                results = run_experiment(
                    G, model_name=model_name, model_params=params,
                    sources=sources_per_net[net_name], n_runs=N_RUNS,
                    seed=SEED, network_name=net_name,
                )
                all_results.extend(results)
                for res in results:
                    s = compute_cascade_stats(res, G)
                    stats_rows.append(s)
                    print(f"{net_name:<12} {model_name:<5} {r0:>5.1f} {s['source']:>7} "
                          f"{s['size']:>6} {s['depth']:>6} {s['actual_r0']:>7.3f} "
                          f"{s['coverage_pct']:>7.1f}% {s['avg_path_from_source']:>8.2f}")

    print("=" * 100)
    print(f"\nTotal cascades: {len(all_results)}")

    # ── 4. Save cascade data ─────────────────────────────────────────────
    json_path = save_cascades(all_results, output_dir=CASCADE_DIR)
    print(f"Cascade data saved → {json_path}")

    # ── 5. Visualise sample cascades ─────────────────────────────────────
    viz_r0, viz_count = 2.0, 0
    print("\nRendering cascade-tree visualisations …")
    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        source = sources_per_net[net_name][0]
        for model_name in MODEL_NAMES:
            params = r0_to_params(viz_r0, avg_deg, model_name, gamma=SIR_GAMMA)
            matches = [
                r for r in all_results
                if r.network_name == net_name and r.model_name == model_name
                and r.source == source
                and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
            ]
            if matches and matches[0].size > 1:
                out = f"{VIZ_DIR}/{net_name}_{model_name}_r0{viz_r0}_src{source}.png"
                plot_cascade_tree(matches[0], G, save_path=out)
                plt.close("all")
                viz_count += 1

    # ── 6. Comparison grids ──────────────────────────────────────────────
    for model_name in MODEL_NAMES:
        comparison: dict[str, CascadeResult] = {}
        for net_name in networks:
            avg_deg = net_stats[net_name]["avg_degree"]
            source = sources_per_net[net_name][0]
            params = r0_to_params(viz_r0, avg_deg, model_name, gamma=SIR_GAMMA)
            matches = [
                r for r in all_results
                if r.network_name == net_name and r.model_name == model_name
                and r.source == source
                and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
            ]
            if matches and matches[0].size > 1:
                comparison[net_name] = matches[0]
        if len(comparison) >= 2:
            plot_comparison_grid(comparison, save_path=f"{VIZ_DIR}/comparison_{model_name}_r0{viz_r0}.png")
            plt.close("all")
            viz_count += 1

    print(f"Saved {viz_count} visualisation(s) → {VIZ_DIR}/")

    # ── 7. Aggregate summary ─────────────────────────────────────────────
    print("\n── Aggregate Summary ──")
    for model_name in MODEL_NAMES:
        ms = [s for s in stats_rows if s["model"] == model_name]
        if ms:
            print(f"  {model_name}: avg_size={np.mean([s['size'] for s in ms]):.1f}, "
                  f"avg_R₀={np.mean([s['actual_r0'] for s in ms]):.3f}, "
                  f"avg_coverage={np.mean([s['coverage_pct'] for s in ms]):.1f}%")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
