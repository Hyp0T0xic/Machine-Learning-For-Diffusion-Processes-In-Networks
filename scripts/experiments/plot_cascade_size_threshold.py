"""
same hit-rate estimation idea but now r0 is fixed and we vary the cascade
size threshold from 15 to 30. shows why picking 25 is reasonable -- too high
and you need an insane number of runs, too low and youre not capturing meaningful cascades.
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cascade import r0_to_params, IndependentCascade
from src.data.networks import generate_ba_network, generate_er_network

# ── config ────────────────────────────────────────────────────────────────────
N_NODES    = 200
BA_M       = 3
ER_P       = 6 / (N_NODES - 1)
N_ESTIMATE = 1000
SEED       = 42

R0_FIXED   = 0.5                    # subcritical -- this is where the cost is interesting
SIZE_RANGE = range(15, 41)          # integer thresholds, 15 to 40 inclusive

N_REPS     = 15       # more reps = more pooled probes = more stable at high sizes
MIN_HITS   = 50
MAX_PROBES = 1_000_000              # need a big cap since size=30 at r0=0.5 is very rare
Y_CAP      = None  # set dynamically from data after estimation

OUT_DIR = Path("results/figures/ml_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def probe_single(G, cascade_size: int, p: float, seed: int) -> tuple[int, int]:
    """run one rep, return (hits, probes) -- pool these across reps for stability."""
    model = IndependentCascade(p=p)
    rng   = random.Random(seed)
    nodes = list(G.nodes())

    hits   = 0
    probes = 0
    while hits < MIN_HITS and probes < MAX_PROBES:
        source  = rng.choice(nodes)
        cascade = model.run(G, source)
        probes += 1
        if cascade.size >= cascade_size:
            hits += 1

    return hits, probes


def estimate_runs(G, cascade_size: int, p: float) -> tuple[float, float]:
    """
    pool hits and probes across all reps before computing the estimate.
    averaging individual estimates is unstable when hits per rep is 0 or 1 --
    pooling treats all reps as one big run, giving a much smoother result.
    """
    total_hits   = 0
    total_probes = 0
    for rep in range(N_REPS):
        h, p_ = probe_single(G, cascade_size, p, SEED + rep * 997)
        total_hits   += h
        total_probes += p_

    if total_hits == 0:
        return np.inf, np.inf

    hit_rate  = total_hits / total_probes
    estimated = N_ESTIMATE / hit_rate
    std       = estimated / np.sqrt(total_hits)  # std shrinks as we get more hits
    return estimated, std


def main():
    np.random.seed(SEED)

    G_ba   = generate_ba_network(N_NODES, BA_M, seed=SEED)
    G_er   = generate_er_network(N_NODES, ER_P, seed=SEED)
    sizes  = list(SIZE_RANGE)

    # compute p once -- same r0 and network properties for all sizes
    avg_deg_ba = sum(d for _, d in G_ba.degree()) / G_ba.number_of_nodes()
    avg_deg_er = sum(d for _, d in G_er.degree()) / G_er.number_of_nodes()
    p_ba = r0_to_params(R0_FIXED, avg_deg_ba, model="IC")["p"]
    p_er = r0_to_params(R0_FIXED, avg_deg_er, model="IC")["p"]

    ba_est, ba_std = [], []
    er_est, er_std = [], []

    for size in sizes:
        ba_e, ba_s = estimate_runs(G_ba, size, p_ba)
        er_e, er_s = estimate_runs(G_er, size, p_er)
        ba_est.append(ba_e)
        ba_std.append(ba_s)
        er_est.append(er_e)
        er_std.append(er_s)
        print(f"size={size:>2}  BA={ba_e:>14,.0f}  ER={er_e:>14,.0f}")

    ba_est = np.array(ba_est)
    ba_std = np.array(ba_std)
    er_est = np.array(er_est)
    er_std = np.array(er_std)

    # set cap from largest finite value, rounded up to a clean number
    all_finite = np.concatenate([ba_est[np.isfinite(ba_est)], er_est[np.isfinite(er_est)]])
    Y_CAP = float(np.max(all_finite)) * 1.05  # 5% headroom above the largest value

    ba_clipped     = np.clip(ba_est, 0, Y_CAP)
    er_clipped     = np.clip(er_est, 0, Y_CAP)
    ba_std_clipped = np.where(ba_est > Y_CAP, 0, ba_std)
    er_std_clipped = np.where(er_est > Y_CAP, 0, er_std)

    ba_valid = np.isfinite(ba_est)
    er_valid = np.isfinite(er_est)
    sizes_arr = np.array(sizes)

    # ── plot ──────────────────────────────────────────────────────────────────
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    # BA = solid, ER = dashed -- distinguishable in black and white
    styles = [
        (ba_clipped, ba_std_clipped, ba_valid, "k-",  "o", "BA"),
        (er_clipped, er_std_clipped, er_valid, "k--", "s", "ER"),
    ]
    for est, std, valid, linestyle, marker, label in styles:
        r = sizes_arr[valid]
        y = est[valid]
        s = std[valid]
        ax.plot(r, y, linestyle, linewidth=1.5, marker=marker, markersize=4,
                markerfacecolor="white", markeredgecolor="black", label=label)
        ax.fill_between(r, np.maximum(y - s, 0), y + s, color="black", alpha=0.08)

    # triangles for clipped values -- open markers so they print cleanly
    for est, marker in [(ba_est, "^"), (er_est, "v")]:
        clipped = np.isfinite(est) & (est > Y_CAP)
        if clipped.any():
            ax.scatter(sizes_arr[clipped], np.full(clipped.sum(), Y_CAP),
                       marker=marker, facecolors="none", edgecolors="black", s=60, zorder=5)

    ax.axvline(x=25, color="black", linestyle=":", linewidth=1, label="chosen cutoff ($d = 25$)")

    ax.set_ylim(bottom=0, top=Y_CAP)
    ax.set_xlim(sizes[0] - 0.5, sizes[-1] + 0.5)
    ax.set_xticks(sizes[::2])  # every other tick so x axis isnt too crowded
    ax.set_xlabel("minimum cascade size $d$", fontsize=11)
    ax.set_ylabel(f"estimated simulations for $N = {N_ESTIMATE}$ cascades", fontsize=10)
    ax.set_title(f"IC model, $R_0 = {R0_FIXED}$, $n = {N_NODES}$", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    out_file = OUT_DIR / f"cascade_size_threshold_r0{R0_FIXED}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
