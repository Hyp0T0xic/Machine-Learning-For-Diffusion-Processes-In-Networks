"""
how does the number of runs needed to collect 1000 cascades of size >= 25
change as r0 varies continuously? uses hit-rate estimation instead of
actually collecting 1000 cascades - way faster for subcritical r0 values.

for each r0: run N_REPS independent estimates (different seeds) and average them.
then estimated_total = 1000 / mean_hit_rate. y axis is linear, truncated at Y_CAP
so the interesting region near the critical threshold is visible.
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
N_NODES      = 200
BA_M         = 3
ER_P         = 6 / (N_NODES - 1)
CASCADE_SIZE = 25
N_ESTIMATE   = 1000
SEED         = 42

# denser near the critical threshold where things change fast,
# tails off toward 5 where the curve is basically flat
R0_RANGE = np.sort(np.concatenate([
    np.linspace(0.4, 0.95, 35),
    np.linspace(0.95, 1.2, 30),  # very dense right around r0=1, this is the interesting bit
    np.linspace(1.2,  5.0, 25),
]))

# averaging multiple reps smooths out the stochastic noise in the hit-rate estimate
N_REPS     = 5
MIN_HITS   = 50        # stop early once we have enough successes for a reliable estimate
MAX_PROBES = 300_000   # higher cap so we can still measure hit rates at r0=0.5 ER
Y_CAP      = 20_000_000  # show the full explosion -- this is the whole point

OUT_DIR = Path("results/figures/ml_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_runs_single(G, r0: float, seed: int) -> float:
    """
    one estimate of runs needed per 1000 cascades, using hit-rate sampling.
    returns inf if we never got a single hit (too subcritical to measure).
    """
    avg_deg = sum(d for _, d in G.degree()) / G.number_of_nodes()
    p = r0_to_params(r0, avg_deg, model="IC")["p"]
    model = IndependentCascade(p=p)

    rng = random.Random(seed)
    nodes = list(G.nodes())

    hits = 0
    probes = 0
    while hits < MIN_HITS and probes < MAX_PROBES:
        source = rng.choice(nodes)
        cascade = model.run(G, source)
        probes += 1
        if cascade.size >= CASCADE_SIZE:
            hits += 1

    if hits == 0:
        return np.inf

    return N_ESTIMATE / (hits / probes)


def estimate_runs(G, r0: float) -> tuple[float, float]:
    """
    average N_REPS independent hit-rate estimates for smoother results.
    returns (mean_estimated_total, std_across_reps).
    """
    estimates = [estimate_runs_single(G, r0, SEED + rep * 997) for rep in range(N_REPS)]
    finite = [e for e in estimates if np.isfinite(e)]
    if not finite:
        return np.inf, np.inf
    return float(np.mean(finite)), float(np.std(finite))


def main():
    np.random.seed(SEED)

    G_ba = generate_ba_network(N_NODES, BA_M, seed=SEED)
    G_er = generate_er_network(N_NODES, ER_P, seed=SEED)

    ba_est, ba_std = [], []
    er_est, er_std = [], []

    n = len(R0_RANGE)
    for i, r0 in enumerate(R0_RANGE):
        ba_e, ba_s = estimate_runs(G_ba, r0)
        er_e, er_s = estimate_runs(G_er, r0)
        ba_est.append(ba_e)
        ba_std.append(ba_s)
        er_est.append(er_e)
        er_std.append(er_s)
        print(f"[{i+1:>3}/{n}]  r0={r0:.3f}  BA={ba_e:>12,.0f}  ER={er_e:>12,.0f}")

    ba_est = np.array(ba_est)
    ba_std = np.array(ba_std)
    er_est = np.array(er_est)
    er_std = np.array(er_std)

    # clip to Y_CAP for display -- values above are real but off the chart
    ba_clipped = np.clip(ba_est, 0, Y_CAP)
    er_clipped = np.clip(er_est, 0, Y_CAP)
    ba_std_clipped = np.where(ba_est > Y_CAP, 0, ba_std)  # hide std bars on clipped points
    er_std_clipped = np.where(er_est > Y_CAP, 0, er_std)

    ba_valid = np.isfinite(ba_est)
    er_valid = np.isfinite(er_est)

    # ── plot ──────────────────────────────────────────────────────────────────
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5))

    line_styles = [
        (ba_clipped, ba_std_clipped, ba_valid, "k-",   "BA"),   # solid
        (er_clipped, er_std_clipped, er_valid, "k--",  "ER"),   # dashed
    ]
    for est, std, valid, ls, label in line_styles:
        r0s = R0_RANGE[valid]
        ys  = est[valid]
        ss  = std[valid]
        ax.plot(r0s, ys, ls, linewidth=1.5, label=label)
        ax.fill_between(r0s, np.maximum(ys - ss, 0), ys + ss,
                        color="black", alpha=0.08)

    # triangles for clipped values -- open markers
    for est, marker in [(ba_est, "^"), (er_est, "v")]:
        clipped_mask = np.isfinite(est) & (est > Y_CAP)
        if clipped_mask.any():
            ax.scatter(R0_RANGE[clipped_mask], np.full(clipped_mask.sum(), Y_CAP),
                       marker=marker, facecolors="none", edgecolors="black", s=50, zorder=5)

    ax.axvline(x=0.5, color="black", linestyle=":", linewidth=1, label="$R_0 = 0.5$ (chosen training value)")

    ax.set_ylim(bottom=0, top=Y_CAP)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_xlabel("$R_0$", fontsize=11)
    ax.set_ylabel(f"estimated simulations for $N = {N_ESTIMATE}$ cascades", fontsize=10)
    ax.set_title(
        f"IC model, cascade size $\\geq {CASCADE_SIZE}$, $n = {N_NODES}$\n"
        f"triangles = values exceed {Y_CAP:,}",
        fontsize=11,
    )
    ax.legend(frameon=True, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_file = OUT_DIR / f"cascade_size_{CASCADE_SIZE}_continuous.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
