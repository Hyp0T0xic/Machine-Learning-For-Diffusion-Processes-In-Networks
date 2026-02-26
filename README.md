# Patient Zero — Diffusion Cascade Source Detection

> **Research Question:** To what extent can the original source of a diffusion process be inferred from the undirected cascade structure, and how does prediction accuracy depend on the infectiousness parameter R₀?

In many real-world settings (epidemiology, social media, information spreading) we observe *which* nodes participated in a cascade and the underlying network, but *not* the direction of transmission. This project investigates whether ML models can identify Patient Zero from this partial, undirected view of the cascade.

---

## Pipeline

```
1. Generate networks   →   2. Simulate cascades   →   3. Extract features   →   4. Train & evaluate ML
   (ER, BA, Complete)        (IC / SI / SIR)              (centrality, etc.)        (vs. classical baselines)
```

---

## Project Structure

```
.
├── configs/
│   └── default.yaml              # Hyperparameters & paths
│
├── scripts/                      # Runnable entry points
│   ├── generate_networks.py      # Phase 1 — build contact graphs
│   └── run_simulation.py         # Phase 2 — simulate cascades
│
├── src/                          # Importable library
│   ├── data/
│   │   ├── cascade.py            # CascadeResult dataclass + IC / SI / SIR models
│   │   ├── networks.py           # ER, BA, Complete graph generation
│   │   └── simulate.py           # Experiment runner, JSON I/O
│   ├── features/
│   │   ├── preprocess.py         # Undirected conversion, trivial-cascade filtering
│   │   └── extract.py            # Node-level structural features for ML
│   ├── models/                   # ML model definitions (GNN, MLP — future)
│   ├── baselines/
│   │   └── centrality.py         # Jordan centroid, degree/closeness/betweenness
│   ├── evaluation/
│   │   └── metrics.py            # Top-k accuracy, MRR, distance to source
│   ├── visualization/
│   │   ├── networks.py           # Network comparison plots
│   │   └── cascades.py           # Cascade tree & comparison-grid plots
│   └── utils.py                  # Seed setting, config loading, helpers
│
├── data/
│   ├── networks/                 # .graphml contact network files
│   ├── raw/                      # cascades.json — simulated labeled data
│   ├── processed/                # Extracted feature matrices (future)
│   └── splits/                   # Train / val / test indices (future)
│
├── results/
│   ├── figures/                  # Generated plots
│   ├── tables/                   # CSV / LaTeX result tables
│   └── logs/                     # Run logs
│
├── notebooks/
│   ├── 01_network_exploration.ipynb
│   └── 02_diffusion_exploration.ipynb
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate the three contact networks
python scripts/generate_networks.py

# 2. Simulate 225 labeled cascades across R₀ ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
python scripts/run_simulation.py
```

---

## Networks

| Network | Model | Key property | Params |
|---|---|---|---|
| **Erdős–Rényi (ER)** | G(n, p) | Uniform baseline | N=100, p=0.05 |
| **Barabási–Albert (BA)** | Preferential attachment | Scale-free with hubs | N=100, m=3 |
| **Complete (K₁₀₀)** | K_n | Max symmetry — negative control | N=100 |

## Diffusion Models

| Model | Description | R₀ mapping |
|---|---|---|
| **IC** | One-shot transmission with prob *p* | p = R₀ / ⟨k⟩ |
| **SI** | Persistent infection, no recovery | β = R₀ / ⟨k⟩ |
| **SIR** | Infection + recovery (prob γ) | β = R₀ · γ / ⟨k⟩ |

R₀ values tested: **0.5, 1.0, 1.5, 2.0, 3.0** → total **225 cascades** (3 nets × 3 models × 5 R₀ × 5 sources).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Top-k accuracy** | True source appears in top-k ranked nodes |
| **MRR** | Mean reciprocal rank of the true source |
| **Distance to source** | Hop distance between rank-1 prediction and true source |