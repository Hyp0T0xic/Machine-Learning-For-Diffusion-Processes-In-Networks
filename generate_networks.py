#!/usr/bin/env python
"""
generate_networks.py
====================
Main entry point: generate, characterise, visualise, and save the three
contact-network structures used for diffusion-cascade source detection.

Usage
-----
    python generate_networks.py
"""

from networks.generator import (
    generate_all_networks,
    compute_network_stats,
    save_networks,
)
from networks.visualizer import visualize_networks


def main() -> None:
    # ── Parameters ──────────────────────────────────────────────────────
    N = 100           # nodes
    ER_P = 0.05       # Erdős–Rényi edge probability
    BA_M = 3          # Barabási–Albert attachment parameter
    SEED = 42         # reproducibility
    DATA_DIR = "data/networks"

    # ── Generate ────────────────────────────────────────────────────────
    print("Generating networks …")
    networks = generate_all_networks(n=N, er_p=ER_P, ba_m=BA_M, seed=SEED)

    # ── Statistics ──────────────────────────────────────────────────────
    stats = {}
    header = f"{'Network':<20} {'Nodes':>6} {'Edges':>7} {'⟨k⟩':>8} {'Diam':>6} {'C':>8} {'Density':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for key, G in networks.items():
        s = compute_network_stats(G)
        stats[key] = s
        print(
            f"{s['name']:<20} {s['nodes']:>6} {s['edges']:>7} "
            f"{s['avg_degree']:>8.2f} {s['diameter']:>6} "
            f"{s['clustering']:>8.4f} {s['density']:>8.4f}"
        )
    print("=" * len(header))

    # ── Save GraphML ────────────────────────────────────────────────────
    saved_paths = save_networks(networks, output_dir=DATA_DIR)
    print(f"\nGraphML files saved:")
    for p in saved_paths:
        print(f"  ✓ {p}")

    # ── Visualise ───────────────────────────────────────────────────────
    fig_path = f"{DATA_DIR}/network_comparison.png"
    print(f"\nRendering figure …")
    fig = visualize_networks(networks, stats=stats, save_path=fig_path, seed=SEED)

    # Show the figure interactively (non-blocking in scripts)
    import matplotlib.pyplot as plt
    plt.show()

    print("\nDone ✓")


if __name__ == "__main__":
    main()
