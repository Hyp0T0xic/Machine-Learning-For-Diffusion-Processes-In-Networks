# Diffusion Cascade Source Detection — Project Summary

## Objective

Simulate stochastic diffusion cascades on different network topologies to generate **labelled training data** where the true infection source (Patient Zero) is known. This data will later be used to train ML models for source detection from undirected cascade observations.

---

## 1. Contact Networks

Three undirected network structures serve as the base contact graphs over which epidemics spread.

| Network | Model | Key Property | Parameters |
|---|---|---|---|
| **Erdős–Rényi (ER)** | Random graph G(n, p) — each possible edge exists independently with probability *p* | Uniform, baseline structure | N=100, p=0.05 |
| **Barabási–Albert (BA)** | Preferential attachment — new nodes attach to existing high-degree nodes | Scale-free with hubs | N=100, m=3 |
| **Complete (K_n)** | Every node connected to every other | Maximally symmetric (negative control) | N=100 |

### Network Statistics

| Property | ER | BA | Complete |
|---|---|---|---|
| Nodes | 100 | 100 | 100 |
| Edges | 224 | 291 | 4,950 |
| Average degree ⟨k⟩ | 4.48 | 5.82 | 99.0 |
| Diameter | 6 | 4 | 1 |
| Clustering coefficient | 0.046 | 0.058 | 1.0 |

### Why These Three?

- **ER** — Null model with no structural bias; tests whether source detection works without topology cues.
- **BA** — Realistic scale-free structure; hub nodes create identifiable "signatures" in cascades, making source detection feasible.
- **Complete** — Every node is structurally identical, so source detection should be near-impossible. Serves as a negative control.

---

## 2. Diffusion Models

Three discrete-time stochastic models simulate how infection spreads through a network.

### Independent Cascade (IC)

- Each infected node gets **one chance** to infect each susceptible neighbour.
- Transmission succeeds with probability **p**.
- After the attempt, the edge is "burned" (no retries).
- Most realistic for **information spreading** (rumours, news).

### Susceptible–Infected (SI)

- Infected nodes attempt transmission **every timestep** with probability **β**.
- Infected nodes **never recover** — they remain infectious forever.
- Process continues until no new infections occur.
- Models worst-case spread with no containment.

### Susceptible–Infected–Recovered (SIR)

- Same transmission as SI (probability **β** per timestep per edge).
- Infected nodes **recover** with probability **γ** each timestep, becoming immune.
- Recovered nodes cannot spread or be reinfected.
- Most realistic for **epidemics** (COVID, flu).

### R₀ (Basic Reproduction Number)

R₀ = average number of secondary infections caused by one infected individual. It is mapped to model parameters using the network's average degree ⟨k⟩:

| Model | Transmission Parameter | Mapping from R₀ |
|---|---|---|
| IC | p | p = R₀ / ⟨k⟩ |
| SI | β | β = R₀ / ⟨k⟩ |
| SIR | β (with fixed γ=0.2) | β = R₀ · γ / ⟨k⟩ |

### R₀ Values Tested

| R₀ | Interpretation |
|---|---|
| 0.5 | Sub-critical — epidemic dies out |
| 1.0 | Critical threshold |
| 1.5 | Moderate spread |
| 2.0 | Strong spread |
| 3.0 | Aggressive spread |

---

## 3. Experiment Design

| Parameter | Value |
|---|---|
| Networks | ER, BA, Complete |
| Models | IC, SI, SIR |
| R₀ values | 0.5, 1.0, 1.5, 2.0, 3.0 |
| Sources per network | 5 (random) |
| Runs per source | 1 |
| Random seed | 42 |
| **Total cascades** | **225** (3 × 3 × 5 × 5) |

### What Each Cascade Records

| Field | Description |
|---|---|
| `source` | True patient zero (ground truth) |
| `infection_times` | {node → timestep} for every infected node |
| `cascade_edges` | Directed list of (parent → child) transmissions |
| `observed_graph` | Undirected subgraph of infected nodes (what an ML model would see) |
| `infection_tree` | Directed tree of the full infection chain |

---

## 4. Project Structure

```
Bachelors/
├── networks/
│   ├── __init__.py           # Package re-exports
│   ├── generator.py          # ER, BA, Complete graph generation + stats
│   └── visualizer.py         # 1×3 network comparison figure
├── diffusion/
│   ├── __init__.py           # Package re-exports
│   ├── models.py             # IC, SI, SIR classes + CascadeResult
│   ├── simulator.py          # Source selection, experiment runner, JSON I/O
│   └── cascade_viz.py        # Hierarchical cascade trees + comparison grids
├── data/
│   ├── networks/             # .graphml files + network_comparison.png
│   └── cascades/
│       ├── cascades.json     # All 225 cascade results
│       └── visualizations/   # Cascade tree PNGs + comparison grids
├── notebooks/
│   ├── 01_network_exploration.ipynb
│   └── 02_diffusion_exploration.ipynb
├── generate_networks.py      # Run: creates networks + saves .graphml
├── run_diffusion.py          # Run: simulates all cascades + saves results
└── requirements.txt          # networkx, matplotlib, numpy
```

### How to Run

```bash
pip install -r requirements.txt
python generate_networks.py    # Phase 1: networks
python run_diffusion.py        # Phase 2: diffusion cascades
```

---

## 5. Key Definitions

| Term | Definition |
|---|---|
| **Cascade** | The set of nodes and edges affected by a single diffusion run from one source |
| **Source / Patient Zero** | The node where infection originates |
| **Degree** | Number of edges connected to a node |
| **Degree centrality** | Degree normalised by (N−1); ranges 0–1 |
| **Hub** | A node with unusually high degree (characteristic of BA networks) |
| **Diameter** | Longest shortest path between any two nodes in the graph |
| **Clustering coefficient** | Probability that two neighbours of a node are also neighbours of each other |
| **Scale-free** | Degree distribution follows a power law; few hubs, many low-degree nodes |
| **GraphML** | XML-based file format for storing graphs with attributes |

---

## 6. Next Steps

The `cascades.json` file provides labelled training data where:
- **Input**: undirected observed graph (which nodes were infected, but not who infected whom)
- **Label**: true source node

This will be used to train and evaluate ML models for **source detection** — predicting Patient Zero from the cascade footprint alone.
