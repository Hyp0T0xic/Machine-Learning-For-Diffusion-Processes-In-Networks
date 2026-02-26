#!/usr/bin/env python
"""
scripts/generate_networks.py
============================
Entry point — generate, characterise, visualise, and save the three
contact networks used for diffusion-cascade source detection.

Usage
-----
    python scripts/generate_networks.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.data.networks import generate_all_networks, compute_network_stats, save_networks
from src.visualization.networks import visualize_networks

DATA_DIR = "data/networks"
FIG_DIR = "results/figures"


def main() -> None:
    N, ER_P, BA_M, SEED = 100, 0.05, 3, 42

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
        print(f"{s['name']:<20} {s['nodes']:>6} {s['edges']:>7} "
              f"{s['avg_degree']:>8.2f} {s['diameter']:>6} "
              f"{s['clustering']:>8.4f} {s['density']:>8.4f}")
    print("=" * len(header))

    # ── Save GraphML ────────────────────────────────────────────────────
    saved_paths = save_networks(networks, output_dir=DATA_DIR)
    print(f"\nGraphML files saved:")
    for p in saved_paths:
        print(f"  ✓ {p}")

    # ── Visualise ───────────────────────────────────────────────────────
    fig_path = f"{FIG_DIR}/network_comparison.png"
    print(f"\nRendering figure …")
    visualize_networks(networks, stats=stats, save_path=fig_path, seed=SEED)
    plt.show()
    print("\nDone ✓")


if __name__ == "__main__":
    main()
