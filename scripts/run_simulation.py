#!/usr/bin/env python
"""run the full ic × {er, ba, complete} × r0 grid, save cascades.json + example tree figures"""
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

# config

NETWORK_DIR = "data/networks"
CASCADE_DIR = "data/raw"
VIZ_DIR     = "results/figures/cascades"

NETWORK_FILES = {
    "ER":       f"{NETWORK_DIR}/er_network.graphml",
    "BA":       f"{NETWORK_DIR}/ba_network.graphml",
    "Complete": f"{NETWORK_DIR}/complete_network.graphml",
}

R0_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]
MODEL     = "IC"
N_SOURCES = 5
N_RUNS    = 1
SEED      = 42


def main() -> None:
    print("Loading networks …")
    networks: dict[str, nx.Graph] = {}
    net_stats: dict[str, dict] = {}
    for name, path in NETWORK_FILES.items():
        G = nx.convert_node_labels_to_integers(nx.read_graphml(path))
        networks[name] = G
        net_stats[name] = compute_network_stats(G)
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"⟨k⟩={net_stats[name]['avg_degree']}")

    sources_per_net = {
        name: select_sources(G, N_SOURCES, seed=SEED)
        for name, G in networks.items()
    }

    all_results: list[CascadeResult] = []
    stats_rows: list[dict] = []
    print(f"\n{'Network':<12} {'R₀':>5} {'Source':>7} "
          f"{'Size':>6} {'Depth':>6} {'R₀_act':>7} {'Cover%':>8} {'AvgPath':>8}")

    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        for r0 in R0_VALUES:
            params = r0_to_params(r0, avg_deg, MODEL)
            results = run_experiment(
                G, model_name=MODEL, model_params=params,
                sources=sources_per_net[net_name], n_runs=N_RUNS,
                seed=SEED, network_name=net_name,
            )
            all_results.extend(results)
            for res in results:
                s = compute_cascade_stats(res, G)
                stats_rows.append(s)
                print(f"{net_name:<12} {r0:>5.1f} {s['source']:>7} "
                      f"{s['size']:>6} {s['depth']:>6} {s['actual_r0']:>7.3f} "
                      f"{s['coverage_pct']:>7.1f}% {s['avg_path_from_source']:>8.2f}")

    print(f"\nTotal cascades: {len(all_results)}")

    json_path = save_cascades(all_results, output_dir=CASCADE_DIR)
    print(f"Cascade data saved → {json_path}")

    # one example cascade per network at r0=2 from the first sampled source
    viz_r0, viz_count = 2.0, 0
    print("\nRendering cascade-tree visualisations …")
    for net_name, G in networks.items():
        avg_deg = net_stats[net_name]["avg_degree"]
        source = sources_per_net[net_name][0]
        params = r0_to_params(viz_r0, avg_deg, MODEL)
        matches = [
            r for r in all_results
            if r.network_name == net_name and r.source == source
            and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
        ]
        if matches and matches[0].size > 1:
            out = f"{VIZ_DIR}/{net_name}_{MODEL}_r0{viz_r0}_src{source}.png"
            plot_cascade_tree(matches[0], G, save_path=out)
            plt.close("all")
            viz_count += 1

    # 1×3 side-by-side comparison across networks
    comparison: dict[str, CascadeResult] = {}
    for net_name in networks:
        avg_deg = net_stats[net_name]["avg_degree"]
        source = sources_per_net[net_name][0]
        params = r0_to_params(viz_r0, avg_deg, MODEL)
        matches = [
            r for r in all_results
            if r.network_name == net_name and r.source == source
            and abs(list(r.params.values())[0] - list(params.values())[0]) < 0.001
        ]
        if matches and matches[0].size > 1:
            comparison[net_name] = matches[0]
    if len(comparison) >= 2:
        plot_comparison_grid(comparison, save_path=f"{VIZ_DIR}/comparison_{MODEL}_r0{viz_r0}.png")
        plt.close("all")
        viz_count += 1

    print(f"Saved {viz_count} visualisation(s) → {VIZ_DIR}/")

    if stats_rows:
        print(f"\nIC: avg_size={np.mean([s['size'] for s in stats_rows]):.1f}, "
              f"avg_R₀={np.mean([s['actual_r0'] for s in stats_rows]):.3f}, "
              f"avg_coverage={np.mean([s['coverage_pct'] for s in stats_rows]):.1f}%")
    print("\nDone")


if __name__ == "__main__":
    main()
