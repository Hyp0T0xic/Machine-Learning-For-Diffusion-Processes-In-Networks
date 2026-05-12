# Patient Zero — Diffusion Cascade Source Detection

> **Research Question:** To what extent can the original source of a diffusion process be inferred from the undirected cascade structure, and how does prediction accuracy depend on the infectiousness parameter R₀?

In many real-world settings (epidemiology, social media, information spreading) we observe *which* nodes participated in a cascade and the underlying network, but *not* the direction of transmission. This project investigates whether ML models can identify Patient Zero from this partial, undirected view of the cascade.

---

## Pipeline

```
1. Generate networks   →   2. Simulate cascades   →   3. Extract features   →   4. Train & evaluate RF
   (ER, BA)                  (IC, size >= 25)           (centrality, etc.)        (vs. classical baselines)
                                                                                          |
                                                                                  5. Validate on real data
                                                                                     (Weibo, OutbreakTrees,
                                                                                      FalseNews)
```

---

## Project Structure

```
.
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── generate_networks.py              # build contact graphs
│   ├── run_simulation.py                 # simulate cascades
│   ├── predict_patient_zero_ic.py        # run inference with a trained model
│   └── experiments/
│       ├── train_rf_ic_ba.py             # train RF on IC-BA cascades
│       ├── train_rf_ic_er.py             # train RF on IC-ER cascades
│       ├── plot_cascades_ic_ba.py        # visualise example BA cascades
│       ├── plot_cascades_ic_er.py        # visualise example ER cascades
│       ├── plot_cascade_size.py          # total runs needed vs R0 (justifies size=25 cutoff)
│       ├── plot_cascade_size_continuous.py  # same but R0 as a continuous variable
│       ├── plot_cascade_size_threshold.py   # runs needed vs size threshold at fixed R0
│       ├── ablation_study.py
│       └── tune_rf_ba.py
│
├── src/                                  # importable library
│   ├── data/
│   │   ├── cascade.py                    # CascadeResult dataclass + IC model
│   │   ├── networks.py                   # ER and BA graph generation
│   │   └── simulate.py                   # experiment runner, JSON I/O
│   ├── features/
│   │   ├── preprocess.py                 # undirected conversion, trivial-cascade filtering
│   │   └── extract.py                    # node-level structural features for ML
│   ├── models/
│   │   └── random_forest.py              # RF wrapper with rank_nodes interface
│   ├── baselines/
│   │   └── centrality.py                 # Jordan centre, degree baseline
│   ├── evaluation/
│   │   └── metrics.py                    # top-k accuracy, MRR
│   └── visualization/
│       ├── networks.py
│       └── cascades.py
│
├── results/
│   ├── figures/
│   │   ├── cascades/                     # example cascade plots
│   │   ├── networks/                     # network structure plots
│   │   └── ml_evaluation/               # accuracy plots, feature importances, size justification
│   └── models/
│       ├── ic_ba/rf_model_size25.pkl
│       └── ic_er/rf_model_size25.pkl
│
├── validation/
│   ├── weibovalidation/                  # Weibo repost cascade validation
│   │   ├── scripts/
│   │   │   ├── validate_weibo.py         # evaluate RF models on real Weibo cascades
│   │   │   ├── validate_outbreak_trees.py
│   │   │   ├── plot_weibo_cascades.py
│   │   │   └── plot_outbreak_trees_cascades.py
│   │   └── figures/
│   │       ├── weibo_validation_large.png
│   │       ├── outbreak_trees_validation.png
│   │       └── cascades/
│   └── truefalsevalidation/              # FalseNews dataset validation
│       ├── load_falsenews.py
│       ├── train_rf_falsenews.py
│       ├── validate_existing_models.py
│       ├── r0_analysis_falsenews.py
│       ├── plot_edge_distribution.py
│       └── results/
│
├── data/
│   ├── networks/                         # .graphml contact network files
│   └── raw/                              # cascades.json — simulated labelled data
│
├── notebooks/
│   ├── 01_network_exploration.ipynb
│   ├── 02_diffusion_exploration.ipynb
│   ├── 05_IC_on_erdos_renyi.ipynb
│   └── 06_IC_on_barabasi_albert.ipynb
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# generate networks
python scripts/generate_networks.py

# train models (IC on BA and ER, cascade size >= 25, 1000 cascades per R0)
uv run python scripts/experiments/train_rf_ic_ba.py
uv run python scripts/experiments/train_rf_ic_er.py

# validate on real-world data
uv run python validation/weibovalidation/scripts/validate_weibo.py
uv run python validation/weibovalidation/scripts/validate_outbreak_trees.py
```

---

## Networks

| Network | Model | Key property | Params |
|---|---|---|---|
| **Erdos-Renyi (ER)** | G(n, p) | Uniform, homogeneous degree | N=200, p=6/(N-1) |
| **Barabasi-Albert (BA)** | Preferential attachment | Scale-free with hubs | N=200, m=3 |

Both networks are calibrated to average degree ~6.

## Diffusion Model

Independent Cascade (IC): each infected node attempts to infect each neighbour once with probability p = R₀ / avg_degree. Only cascades of size >= 25 are kept for training.

R₀ values: **0.5, 1.0, 2.0, 3.0, 5.0** — 1000 cascades collected per R₀ per network.

The size=25 cutoff is justified empirically in `scripts/experiments/plot_cascade_size.py` — below R₀=1 the simulation cost grows exponentially, and above size=25 the cost becomes impractical at subcritical R₀ values.

---

## ML Model

Random Forest trained per network type (IC-BA, IC-ER). Features are node-level structural properties extracted from the observed undirected cascade. The model outputs a ranking of candidate source nodes.

Baselines: **Jordan Centre**, **Degree**.

Evaluation: **Top-1 and Top-3 accuracy**, averaged across 5 seeds with stratified group cross-validation.

---

## Validation

Models trained on synthetic IC cascades are evaluated on three real-world datasets:

| Dataset | Description |
|---|---|
| **Weibo** | Real repost cascades from Weibo (Chinese microblog), filtered to 500-2000 nodes |
| **OutbreakTrees** | Biological transmission trees from real disease outbreaks |
| **FalseNews** | Misinformation propagation cascades, used for separate RF training and R₀ analysis |

Results saved to `validation/weibovalidation/figures/` and `validation/truefalsevalidation/results/`.
