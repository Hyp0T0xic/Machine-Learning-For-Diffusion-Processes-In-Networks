# Patient Zero — Project Overview

> **Research Question:** To what extent can the original source of a diffusion process be inferred from the undirected cascade structure, and how does prediction accuracy depend on the infectiousness parameter R₀?

This document is a single, flowing description of everything the project involves: the problem, the simulation pipeline, the feature engineering, the baselines and ML models, the evaluation protocol, and the experimental matrix we are working through. It is intentionally written without separating what has already been implemented from what remains — treat it as the complete specification of the work.

---

## Sources / References

This section tracks the places where we need to find literature justifications: either to defend a methodological choice we already made, or to decide between alternative methods still on the table. Each bullet identifies a concrete decision and the kind of source we are looking for.

- **Network models (Erdős–Rényi, Barabási–Albert, Complete K_n).** We need references motivating why these three families are the standard "minimal" set for diffusion studies: ER as an unstructured baseline, BA for scale-free / hub-driven spreading, and K_n as the maximum-symmetry negative control where source detection should be near-impossible. Candidate starting points: Erdős & Rényi (1959), Barabási & Albert (1999), and any recent epidemic-on-networks survey (e.g. Pastor-Satorras et al., *Rev. Mod. Phys.* 2015).
- **Choice of ER parameters (N=100, p=0.05) and BA parameters (N=100, m=3).** We need to justify the density / mean-degree chosen — i.e. why ⟨k⟩ ≈ 5 for ER and ⟨k⟩ ≈ 6 for BA are reasonable sizes for observable but non-trivial cascades. Look for source-detection papers that use similar network sizes.
- **Diffusion models (IC, SI, SIR).** We need canonical references:
  - IC: Kempe, Kleinberg & Tardos (2003), *Maximizing the Spread of Influence through a Social Network*.
  - SI / SIR: Any standard mathematical epidemiology reference (e.g. Newman, *Networks: An Introduction*, 2018) or Pastor-Satorras & Vespignani for network epidemics.
- **R₀ → parameter mapping (p = R₀/⟨k⟩ for IC, β = R₀/⟨k⟩ for SI, β = R₀·γ/⟨k⟩ for SIR).** This is the homogeneous-mixing approximation of R₀. We need to cite a source for this approximation (e.g. Newman 2002 / Anderson & May 1991) and acknowledge that on heterogeneous networks (BA) the effective R₀ differs from this nominal value due to degree variance.
- **SIR recovery rate γ = 0.2.** The choice is arbitrary; we need either a paper that uses a comparable value for simulation studies, or an explicit note that the value was chosen for convenience and the results are reported relative to R₀ (not γ).
- **Structural features for source detection.** Each feature needs a source or rationale:
  - *Jordan centre / minimum-eccentricity estimator*: Shah & Zaman (2011), *Rumors in a Network: Who's the Culprit?* — the foundational reference.
  - *Closeness / betweenness / degree centrality as baselines*: any centrality-based rumor-source detection paper (e.g. Comin & da F. Costa 2011).
  - *Eccentricity, subtree depth, number of leaves, 2-hop neighbourhood size, clustering*: find papers that use these, or argue from first principles (symmetry breaking / BFS-tree geometry).
- **Random Forest as the ML model.** We need a source justifying tree-ensembles for tabular, low-dimensional structural features (Breiman 2001). Also a source or argument for why RF is a reasonable first step before more expensive alternatives (GNNs, MLPs).
- **Class imbalance handling (`class_weight="balanced"`).** Because every cascade has exactly one source and many non-sources, the positive class is rare. Need a reference on class-weight reweighting vs. resampling in imbalanced classification.
- **Cross-validation protocol (StratifiedGroupKFold with cascade-level groups).** Justify why we must group by cascade to avoid leakage of same-cascade nodes across train/test.
- **Evaluation metrics.**
  - *Top-k accuracy* and *MRR* are standard in ranking literature (information-retrieval references).
  - *Distance to source (hop-distance)* is the standard "near-miss" metric in source-detection work — Shah & Zaman (2011) and later follow-ups use it.
- **Why the "undirected cascade" assumption is realistic.** We need 1–2 citations from epidemiology / information-spreading literature where only the set of infected nodes (and not who infected whom) is observed — this underpins the whole problem framing.

---

## Problem Statement

In many real-world diffusion processes — epidemics, misinformation, viral content — we can observe *which* nodes have been reached by a cascade and we know the static contact / friendship / interaction network on which the cascade travelled. What we typically **do not** observe is the direction of each transmission: who infected whom, and therefore who started it all.

The project asks whether, given only the undirected sub-graph of participants and the underlying network, a machine-learning model can recover the original source node ("patient zero") better than classical graph-theoretic heuristics — and how that performance depends on how infectious the process is (the basic reproduction number R₀).

---

## Pipeline

```
1. Generate networks   →   2. Simulate cascades   →   3. Preprocess (drop directions)
      ↓
4. Extract node-level features   →   5. Train ML model + run baselines   →   6. Evaluate & compare
```

Every stage of the pipeline is deterministic given a seed, so every experiment is reproducible.

---

## Networks

We work on three contact-network families, each generated with `N = 100` nodes (and `N = 200` for the larger RF experiment):

| Network | Generator | Intended role |
|---|---|---|
| **Erdős–Rényi (ER)** | `G(n, p=0.05)` | Unstructured baseline; homogeneous degree distribution |
| **Barabási–Albert (BA)** | preferential attachment with `m=3` | Scale-free, hub-dominated; models realistic social / contact graphs |
| **Complete (K_n)** | every pair connected | Maximum-symmetry negative control; the source should be *fundamentally* hard to recover here |

The generator code lives in `src/data/networks.py`; batch generation and descriptive statistics (nodes, edges, ⟨k⟩, diameter, clustering, density) are produced by `scripts/generate_networks.py` and visualised in `notebooks/01_network_exploration.ipynb`.

---

## Diffusion Models

Three stochastic contagion processes are implemented in `src/data/cascade.py`. For all three, the true source is known by construction and stored in the resulting `CascadeResult` object along with infection times and the directed transmission edges.

| Model | Dynamics | R₀ → parameter |
|---|---|---|
| **Independent Cascade (IC)** | Each newly-infected node gets one chance per neighbour to transmit, with probability *p*. Failed attempts are burned. | `p = R₀ / ⟨k⟩` |
| **Susceptible-Infected (SI)** | Infected nodes stay infectious forever; each timestep, every infected-susceptible edge transmits independently with probability *β*. Runs until no new infections occur. | `β = R₀ / ⟨k⟩` |
| **Susceptible-Infected-Recovered (SIR)** | Like SI, but each infected node recovers with probability *γ* per timestep (γ = 0.2 by default) and becomes immune. | `β = R₀ · γ / ⟨k⟩` |

The R₀ sweep used across experiments is **R₀ ∈ {0.5, 1.0, 1.5, 2.0, 3.0}** (the core simulation) with larger or extended sweeps (up to R₀ = 5.0) used for specific ML experiments. Values straddle the critical point R₀ = 1, so we expect to see a qualitative change from "cascade dies out" to "cascade reaches most of the graph".

Behaviours we care about exploring:
- **IC** tends to terminate early because each edge is a single attempt.
- **SI** instead stalls only when no infected-susceptible pair happens to transmit in a whole timestep; it can reach much deeper cascades at low β.
- **SIR** is the middle case and is the only one with a natural notion of an "outbreak" vs. "early die-out".

---

## The Experimental Matrix

The core of the study is a full factorial sweep over (network × diffusion model × R₀ × source node × run). The minimum target matrix is:

- **3 networks** × **3 diffusion models** × **5 R₀ values** × **5 source nodes** × **1 run** = **225 labelled cascades** (produced by `scripts/run_simulation.py` and stored in `data/raw/cascades.json`).

For each (network, model) combination that we want to analyse at ML-level, we also need a much larger sample — on the order of 500 cascades per R₀ — in order to train and cross-validate the Random Forest. The experimental plan is to cover **IC** and **SI** on **all three network structures**:

| | ER | BA | Complete |
|---|---|---|---|
| **IC** | exploration + RF experiment | exploration + RF experiment | exploration + RF experiment (already scripted via `scripts/train_rf_ic_complete.py`) |
| **SI** | exploration + RF experiment (exploration started in `notebooks/04_SI_on_erdos_renyi.ipynb`) | exploration + RF experiment | exploration + RF experiment |

SIR is kept in the simulation pipeline for completeness but is secondary to the IC/SI comparison.

---

## Preprocessing: From Labelled Cascades to Undirected Observations

This step is what makes the problem non-trivial. Even though the simulator knows who infected whom, we deliberately discard that information before giving the cascade to the model. Two preprocessing functions live in `src/features/preprocess.py`:

- **`to_undirected()`** — strips edge directions from the cascade. The input to the model is therefore the induced subgraph on the infected nodes, with no orientation on the edges. This mirrors the real-world observation model.
- **`filter_trivial()`** — removes cascades below a minimum size (default 3 nodes), because a cascade of size 1 or 2 makes source detection either trivial or ill-defined.

---

## Node-Level Features (`src/features/extract.py`)

Eleven structural features are computed per infected node, on the undirected observed cascade:

1. `degree` — raw degree within the observed cascade
2. `degree_centrality` — degree normalised by cascade size
3. `closeness_centrality`
4. `betweenness_centrality`
5. `eccentricity` — farthest-node distance
6. `jordan_center_dist` — hop distance to the nearest Jordan centre (minimum-eccentricity node)
7. `clustering` — local clustering coefficient
8. `two_hop_count` — number of nodes reachable within 2 hops
9. `subtree_depth` — eccentricity reinterpreted as rooted-tree depth
10. `cascade_size` — global cascade size (per-cascade constant; gives the model a context signal)
11. `num_leaves` — number of degree-≤1 nodes in the cascade

`build_feature_matrix()` stacks these across every node of every cascade into a single `(X, y)` matrix, where `y = 1` for the true source and `0` otherwise. This single matrix drives both training and evaluation.

---

## Baselines (`src/baselines/centrality.py`)

Four classical, training-free source-detection heuristics:

- **Jordan centre** — predict the node(s) with minimum eccentricity. This is the canonical baseline from Shah & Zaman (2011) and is known to be optimal on regular trees.
- **Degree rank** — predict the highest-degree node in the observed cascade.
- **Closeness rank** — predict the node with the highest closeness centrality.
- **Betweenness rank** — predict the node with the highest betweenness centrality.

`predict_all()` produces a ranked node list for each of these methods, so every baseline returns the same kind of output as the ML model and can be evaluated with the same metrics.

---

## Machine-Learning Model (`src/models/random_forest.py`)

The ML ranker is a thin wrapper around `sklearn.ensemble.RandomForestClassifier`. It is configured with:

- `n_estimators = 100`, `max_depth = 10` (default)
- `class_weight = "balanced"` to correct the heavy class imbalance (≈1 source per ≈20–200 candidate nodes)
- A `rank_nodes(result)` method that scores every infected node in a cascade by its predicted probability of being the source and returns them in decreasing order

The training protocol is **grouped cross-validation**: `StratifiedGroupKFold` with the cascade ID as the group, so that all nodes from the same cascade stay together in either the train or the test fold. This prevents trivial leakage where the classifier memorises cascade-level signals (like size) rather than learning per-node source indicators.

We also expose the Gini feature importances so we can report which structural features end up being predictive under each (network, model, R₀) regime.

Beyond Random Forest, the `src/models/` package is intended to host more expressive alternatives (MLPs, GNNs) if the RF leaves room for improvement; these are not yet the focus.

---

## Evaluation (`src/evaluation/metrics.py`)

Every ranker — the RF and each baseline — is evaluated per cascade and then aggregated across cascades. The metrics are:

- **Top-k accuracy** for `k ∈ {1, 3, 5}`: fraction of cascades where the true source appears in the top-k predicted nodes.
- **Mean Reciprocal Rank (MRR)**: mean of `1 / rank_of_true_source` across cascades. Sensitive to how high the true source is in the ranking, not just whether it is in the top-k.
- **Distance to source (hop distance)**: shortest-path distance in the observed undirected cascade between the top-1 prediction and the true source. This is the "how-close-did-we-get" metric used in the source-detection literature. We report mean and median.

`evaluate_ranker()` returns all of these in a single report.

---

## Scripts and Notebooks

The runnable pieces are:

- `scripts/generate_networks.py` — builds and saves the three networks, prints the statistics table, renders comparison figures.
- `scripts/run_simulation.py` — executes the full IC/SI/SIR × R₀ × sources matrix, saves `cascades.json`, and produces cascade visualisations.
- `scripts/predict_patient_zero_ic.py` — runs only the classical baselines against IC cascades on a small complete graph, across R₀, and produces comparison figures (top-1, top-3, mean rank, coverage).
- `scripts/train_rf_ic_complete.py` — the main ML experiment script: trains the Random Forest on IC cascades on K_200, sweeps R₀, cross-validates, compares RF against all baselines, and exports an accuracy comparison figure plus a feature-importance bar chart. Parallel scripts for the remaining (network, diffusion) cells of the matrix follow the same template.

The notebooks tell the same story in tutorial form:

- `notebooks/01_network_exploration.ipynb` — generates the three networks and visualises/describes them.
- `notebooks/02_diffusion_exploration.ipynb` — runs single cascades for each (model × network) combination and plots cascade trees side-by-side.
- `notebooks/03_IC_on_complete_graph.ipynb` — step-by-step IC tutorial on K_n, with editable parameters, a 200-run cascade-size distribution, and an R₀ sweep demonstrating the phase-transition behaviour.
- `notebooks/04_SI_on_erdos_renyi.ipynb` — the SI counterpart on ER, highlighting how SI's discrete-time dynamics differ qualitatively from IC (duration vs. extinction behaviour).

The same pattern of exploratory notebook + ML-experiment script is applied to every remaining (model × network) cell: IC on ER, IC on BA, SI on BA, SI on Complete, and an SIR sanity check.

---

## Analysis Goals

Across the full IC × SI × (ER, BA, Complete) matrix we want to be able to answer, with numbers and figures:

1. **How does source detection degrade with R₀?** We expect a characteristic U-shaped or monotone curve: at R₀ ≪ 1 cascades are too small to contain structural information; at R₀ ≫ 1 the cascade covers most of the graph and the symmetry wipes out the source signal.
2. **How does it depend on the network structure?** Complete graphs should be essentially hopeless (by design); ER should be intermediate; BA — because hubs dominate — should behave differently depending on whether the source is a hub or a leaf.
3. **How does it depend on the diffusion model?** IC cascades are shallower and more tree-like; SI cascades run deeper and are noisier. We expect different features to matter in each regime.
4. **Does the Random Forest beat the classical baselines, and by how much?** We quantify this with top-k accuracy, MRR and mean distance to source for every (network, model, R₀) cell.
5. **Which features carry the signal?** The RF's Gini importances, examined per (network, model) regime, tell us whether source detection is driven by centrality, eccentricity / Jordan-centre proximity, local clustering, or cascade-level context.

---

## Configuration and Reproducibility

All experiment-wide parameters live in `configs/default.yaml`:

```yaml
network:   { n_nodes: 100, er_probability: 0.05, ba_attachment: 3, seed: 42 }
diffusion: { r0_values: [0.5, 1.0, 1.5, 2.0, 3.0],
             model_names: ["IC", "SI", "SIR"],
             n_sources: 5, n_runs: 1, sir_gamma: 0.2, seed: 42 }
paths:     { network_dir: "data/networks", cascade_dir: "data/raw",
             viz_dir: "results/figures", processed_dir: "data/processed",
             splits_dir: "data/splits", results_dir: "results" }
```

Runtime dependencies are split between `requirements.txt` (core: networkx, matplotlib, numpy) and `requirements-notebooks.txt` (adds Jupyter). The ML scripts additionally use scikit-learn and pandas.

Every stochastic step (network generation, cascade simulation, Random Forest training, CV splitting) is seeded, so every figure and table in `results/` can be regenerated end-to-end by re-running the two main scripts followed by the per-experiment RF script.
