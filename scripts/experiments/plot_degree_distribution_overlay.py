"""
Overlay the degree distributions of ER (p=0.03) and BA (m=3) networks
on a single axes, with a vertical line marking the shared average degree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.networks import generate_er_network, generate_ba_network

N    = 200
ER_P = 6 / (N - 1)
BA_M = 3
SEED = 42

OUT_DIR = Path("results/figures/networks")


def main() -> None:
    G_er = generate_er_network(n=N, p=ER_P, seed=SEED)
    G_ba = generate_ba_network(n=N, m=BA_M, seed=SEED)

    er_degrees = [d for _, d in G_er.degree()]
    ba_degrees = [d for _, d in G_ba.degree()]

    er_avg = np.mean(er_degrees)
    ba_avg = np.mean(ba_degrees)

    max_deg = max(max(er_degrees), max(ba_degrees))
    bins = range(0, max_deg + 2)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(er_degrees, bins=bins, alpha=0.6, color="white",
            edgecolor="black", linewidth=0.8, label="ER  (p=6/(N-1))")
    ax.hist(ba_degrees, bins=bins, alpha=0.4, color="0.4",
            edgecolor="black", linewidth=0.8, label=f"BA  (m={BA_M})")

    ax.axvline(er_avg, color="black", linestyle="--", linewidth=1.2,
               label=f"ER $\\langle k \\rangle={er_avg:.1f}$")
    ax.axvline(ba_avg, color="black", linestyle=":", linewidth=1.2,
               label=f"BA $\\langle k \\rangle={ba_avg:.1f}$")

    ax.set_xlabel("degree $k$", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"Degree distribution — ER vs BA  (N={N})", fontsize=12)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "degree_distribution_overlay.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()
