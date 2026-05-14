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

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate contact networks
python scripts/generate_networks.py

# 2. Train RF models (IC on BA and ER, cascade size >= 25, 1000 cascades per R₀)
python scripts/experiments/train_rf_ic_ba.py
python scripts/experiments/train_rf_ic_er.py

# 3. Evaluate on held-out test sets
python scripts/experiments/evaluate_testset_ic_ba.py
python scripts/experiments/evaluate_testset_ic_er.py

# 4. Validate on real-world data
python validation/weibovalidation/scripts/validate_weibo.py
python validation/outbreaksvalidation/scripts/validate_outbreak_trees.py
python validation/truefalsevalidation/scripts/train_rf_falsenews.py
```

---

## Project Structure

```
.
├── configs/
│   └── default.yaml                             # network, diffusion, and path parameters
│
├── scripts/
│   ├── generate_networks.py                     # build ER and BA contact graphs → data/networks/
│   ├── run_simulation.py                        # run IC cascade simulations → data/raw/
│   ├── predict_patient_zero_ic.py               # load a trained RF and run inference
│   └── experiments/
│       ├── train_rf_ic_ba.py                    # train RF on IC-BA cascades (5 seeds, CV)
│       ├── train_rf_ic_er.py                    # train RF on IC-ER cascades (5 seeds, CV)
│       ├── evaluate_testset_ic_ba.py            # final test-set evaluation for IC-BA model
│       ├── evaluate_testset_ic_er.py            # final test-set evaluation for IC-ER model
│       ├── ablation_study.py                    # test all 63 non-empty feature subsets
│       ├── tune_rf_ba.py                        # hyperparameter grid search for RF on BA
│       ├── plot_cascades_ic_ba.py               # visualise example BA cascades
│       ├── plot_cascades_ic_er.py               # visualise example ER cascades
│       ├── plot_cascade_size.py                 # runs needed vs R₀ (justifies size=25 cutoff)
│       ├── plot_cascade_size_continuous.py      # same but with R₀ as a continuous variable
│       ├── plot_cascade_size_threshold.py       # runs needed vs size threshold at fixed R₀
│       ├── plot_metrics_ic_ba.py                # accuracy curves: RF vs baselines (BA)
│       └── plot_metrics_ic_er.py                # accuracy curves: RF vs baselines (ER)
│
├── src/                                         # importable library
│   ├── data/
│   │   ├── cascade.py                           # CascadeResult dataclass + IC model
│   │   ├── networks.py                          # ER and BA graph generation
│   │   └── simulate.py                          # experiment runner, JSON I/O
│   ├── features/
│   │   ├── preprocess.py                        # undirected conversion, trivial-cascade filtering
│   │   └── extract.py                           # 7 node-level structural features for ML
│   ├── models/
│   │   └── random_forest.py                     # RF wrapper with rank_nodes interface
│   ├── baselines/
│   │   └── centrality.py                        # Jordan centre, degree, closeness, betweenness
│   ├── evaluation/
│   │   └── metrics.py                           # top-k accuracy, MRR, hop distance to source
│   └── visualization/
│       ├── networks.py                          # network layout and degree distribution plots
│       ├── cascades.py                          # hierarchical cascade tree plots
│       └── theme.py                             # shared colour palette (METHOD_COLORS)
│
├── validation/
│   ├── weibovalidation/
│   │   └── scripts/
│   │       ├── validate_weibo.py                # evaluate IC-BA/ER models on 100 real Weibo cascades
│   │       ├── validate_weibo_natural_range.py  # estimate R₀ distribution from real cascades
│   │       ├── plot_weibo_cascades.py           # visualise example Weibo cascades
│   │       ├── plot_weibo_validation.py         # plot validation results
│   │       └── plot_weibo_natural_range.py      # plot R₀ distribution from Weibo data
│   │
│   ├── outbreaksvalidation/
│   │   └── scripts/
│   │       ├── validate_outbreak_trees.py       # evaluate models on biological outbreak trees
│   │       ├── plot_outbreak_trees_cascades.py  # visualise outbreak cascades
│   │       └── plot_outbreak_trees_validation.py # plot outbreak validation results
│   │
│   └── truefalsevalidation/
│       └── scripts/
│           ├── load_falsenews.py                # parse FalseNews CSV, wrap as CascadeResult
│           ├── train_rf_falsenews.py            # train fresh RF on FalseNews cascades
│           ├── validate_existing_models.py      # evaluate IC-BA/ER on FalseNews
│           ├── validate_existing_models2.py     # alternative FalseNews validation setup
│           ├── r0_analysis_falsenews.py         # estimate R₀ from FalseNews cascade sizes
│           ├── export_filtered_falsenews_csv.py # filter FalseNews by criteria, export CSV
│           ├── plot_r0_analysis.py              # visualise R₀ estimates
│           ├── plot_falsenews_comparison.py     # FalseNews RF vs IC baselines
│           ├── plot_falsenews_natural_range.py  # cascade size distributions
│           ├── plot_falsenews_natural_range_median.py # median version of above
│           ├── plot_cascade_sizes.py            # cascade size histograms
│           ├── plot_edge_distribution.py        # network topology analysis
│           └── plot_truefalse_cascades.py       # visualise individual FalseNews cascades
│
├── data/
│   ├── networks/                                # .graphml contact network files (generated)
│   │   ├── ba_network.graphml
│   │   └── er_network.graphml
│   ├── raw/
│   │   └── cascades.json                        # 5000 simulated cascades (1000 per R₀)
│   ├── rumdect/                                 # RumDect dataset (Weibo repost cascades)
│   │   └── rumdect/
│   │       ├── Weibo/                           # individual cascade JSON files
│   │       └── Weibo.txt                        # cascade index
│   └── outbreak_trees/                          # disease outbreak transmission trees
│
├── FalseNews_Code_Data/                         # external FalseNews dataset
│   └── FalseNews_Code_Data/
│       └── data/
│           └── raw_data_anon.csv                # anonymised misinformation cascade data
│
├── results/
│   ├── figures/
│   │   ├── cascades/                            # example cascade tree plots
│   │   ├── networks/                            # network structure plots
│   │   └── ml_evaluation/                       # accuracy plots, feature importances
│   └── models/
│       ├── ic_ba/rf_model_size25.pkl            # trained RF for IC-BA
│       └── ic_er/rf_model_size25.pkl            # trained RF for IC-ER
│
├── notebooks/
│   ├── 01_network_exploration.ipynb
│   ├── 02_diffusion_exploration.ipynb
│   ├── 03_IC_on_complete_graph.ipynb
│   ├── 04_SI_on_erdos_renyi.ipynb
│   ├── 05_IC_on_erdos_renyi.ipynb
│   └── 06_IC_on_barabasi_albert.ipynb
│
├── mlruns/                                      # MLflow experiment tracking artefacts
├── pyproject.toml
├── requirements.txt
└── requirements-notebooks.txt
```

---

## Networks

| Network | Model | Key property | Params |
|---|---|---|---|
| **Erdős–Rényi (ER)** | G(n, p) | Uniform, homogeneous degree | N=200, p=6/(N−1) |
| **Barabási–Albert (BA)** | Preferential attachment | Scale-free with hubs | N=200, m=3 |

Both networks are calibrated to average degree ≈ 6.

---

## Diffusion Model

**Independent Cascade (IC):** each infected node attempts to infect each neighbour once with probability p = R₀ / avg\_degree. Only cascades of exactly 25 nodes are kept for training (capped by `max_size` in the simulation).

R₀ values: **0.5, 1.0, 2.0, 3.0, 5.0** — 1000 cascades collected per R₀ per network type.

The size=25 cutoff is justified empirically in `scripts/experiments/plot_cascade_size.py`. Below R₀=1 the number of simulations required to collect qualifying cascades grows rapidly, making larger cutoffs computationally impractical in the subcritical regime.

---

## Features

Seven structural node-level features are extracted from the observed **undirected** cascade subgraph:

| Feature | Type | Description |
|---|---|---|
| Degree | Normalised | Node degree within the cascade |
| Largest component fraction | Normalised | Fraction of cascade nodes reachable after removing this node |
| 2-hop neighbourhood | Normalised | Nodes within 2 hops |
| Jordan centre distance | Absolute | Distance from the Jordan centre of the cascade |
| Cascade diameter | Absolute | Diameter of the cascade subgraph |
| Leaf count | Absolute | Number of degree-1 nodes |
| Endpoint balance | Absolute | Symmetry of leaf distribution around this node |

---

## ML Model

A Random Forest trained separately per network type (IC-BA, IC-ER). The model outputs a per-node source probability; nodes are ranked and the top-k candidates are returned.

**Baselines:** Jordan Centre, Degree, Closeness centrality.

**Evaluation:** Top-1 and Top-3 accuracy plus MRR, averaged across 5 seeds with stratified group cross-validation.

Experiment runs are tracked with **MLflow** (`mlruns/`).

---

## Validation

Models trained on synthetic IC cascades are evaluated on three real-world datasets:

| Dataset | Script | Description |
|---|---|---|
| **Weibo** | `validation/weibovalidation/scripts/validate_weibo.py` | Real repost cascades from Weibo (500–2000 nodes) |
| **OutbreakTrees** | `validation/outbreaksvalidation/scripts/validate_outbreak_trees.py` | Biological transmission trees from disease outbreaks |
| **FalseNews** | `validation/truefalsevalidation/scripts/train_rf_falsenews.py` | Misinformation cascades; used for fresh RF training and R₀ analysis |

Natural-range R₀ estimates for each dataset can be plotted with the corresponding `plot_*_natural_range.py` scripts.
