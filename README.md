# Graph-DAG-Scheduling-Research

Research-oriented experimental framework for DAG task scheduling using neural network architectures on heterogeneous multi-CPU graph environments.

This repository contains the implementation and experimental results accompanying two research papers investigating structural generalization for workflow scheduling problems.

---

# Research Overview

Modern workflow scheduling problems are commonly represented as Directed Acyclic Graphs (DAGs), where:
- nodes represent computational tasks,
- edges represent dependency constraints,
- scheduling decisions directly depend on graph topology.

This repository investigates how different neural architectures handle graph-structured scheduling environments.

The first study explores the limitations of classical dense Multi-Layer Perceptrons (MLPs) for DAG scheduling.

The second study investigates Graph Neural Networks (GCNs and GATs) for topology-aware scheduling and structural generalization on unseen workflow graphs.

---

# Repository Structure

```bash
GRAPH-DAG-SCHEDULING-RESEARCH/
│
├── GNN_approach/
│   ├── eval/
│   ├── MCS/
│   ├── models/
│   ├── results/
│   ├── training/
│   ├── .env
│   └── run.py
│
├── MLP_approach/
│   ├── eval/
│   ├── MCS/
│   ├── models/
│   ├── results/
│   ├── training/
│   ├── .env
│   └── readme.md
│
├── papers/
│
├── README.md
└── requirements.txt
```

---

# Paper 1 . MLP-Based DAG Scheduling

## Objective

Investigate whether classical dense neural networks can learn scheduling policies for graph-structured workflow environments.

## Main Findings

- MLP schedulers successfully imitate heuristic scheduling policies on known environments.
- However, performance collapses on unseen DAG topologies.
- Flattened vector representations fail to preserve graph dependency structures.
- Structural generalization remains extremely limited.

## Core Idea

The study demonstrates that graph topology preservation is essential for robust scheduling generalization.

---

# Paper 2 . Graph Neural Network Scheduling

## Objective

Investigate whether Graph Neural Networks improve scheduling generalization on unseen DAG environments.

## Implemented Architectures

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)

## Main Findings

- GNN schedulers significantly improve structural generalization.
- GCN architectures remain stable under increasing DAG density.
- GAT architectures occasionally outperform heuristic scheduling policies on unseen environments.
- Message passing preserves dependency relationships during inference.

## Core Idea

Graph-aware representations provide substantially more coherent scheduling embeddings than flattened dense representations.

---

# Scheduling Environment

The scheduling environment simulates:
- heterogeneous CPU graph architectures,
- DAG workflow dependencies,
- task execution constraints,
- runtime scheduling decisions.

## DAG Representation

Each workflow is represented as:

```math
G = (X, E, Y)
```

Where:
- \(X\) represents node features,
- \(E\) represents graph connectivity,
- \(Y\) represents scheduling target actions.

---

# Node Features

Task nodes contain:
- task duration,
- dependency count,
- dependent count,
- execution status,
- readiness state.

---

# CPU Graph Environment

Processors are represented as graph structures where:
- nodes correspond to CPUs,
- edges represent processor neighborhood connectivity.

This allows the scheduler to reason about:
- workload locality,
- congestion,
- neighborhood activity.

---

# Heuristic Teacher Policy

Behavior cloning datasets are generated using a handcrafted heuristic scheduling policy.

The heuristic prioritizes:
- dependency unlocking,
- task duration,
- local CPU congestion reduction.

Task priority scoring:

```math
S_{task}(t) =
|\text{dependents}(t)|
+
0.5 \times \text{duration}(t)
+
\epsilon
```

CPU selection scoring:

```math
S_{cpu}(c) =
-
|\text{busy\_neighbors}(c)|
+
\epsilon
```

The heuristic generates locally efficient scheduling trajectories used for supervised imitation learning.

---

# Graph Neural Architectures

## GCN

The Graph Convolutional Network propagates scheduling information through local dependency neighborhoods.

Graph convolution update:

```math
h_v^{(l+1)}
=
\sigma
\left(
\sum_{u \in \mathcal{N}(v)}
\frac{1}{c_{uv}}
W^{(l)} h_u^{(l)}
\right)
```

---

## GAT

The Graph Attention Network dynamically weights neighborhood importance during message propagation.

Attention coefficients:

```math
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_{k \in \mathcal{N}(i)}
\exp(e_{ik})
}
```

This allows the scheduler to prioritize:
- critical dependency paths,
- bottleneck tasks,
- important workflow regions.

---

# Valid Action Masking

Scheduling environments contain extremely sparse action spaces.

At each simulation step:
- many task/CPU assignments are invalid,
- only executable actions are allowed.

The repository implements valid action masking:

```math
A_{valid}(s) \subset A
```

Inference is restricted to:

```math
a^* =
\arg\max_{a \in A_{valid}(s)}
\pi(a|s)
```

This prevents invalid scheduling assignments during inference.

---

# Experimental Results

The experiments demonstrate:

- strong GNN generalization on unseen DAG structures,
- stable scaling under increasing DAG density,
- severe MLP degradation on unknown workflow topologies,
- topology-aware scheduling improvements using graph attention mechanisms.

## Key Observations

| Model | Behavior |
|---|---|
| MLP | Strong overfitting to known graph distributions |
| GCN | Stable topology-aware generalization |
| GAT | Improved dependency prioritization |
| Heuristic | Strong local scheduling policy |

---

# Future Research Directions

This repository serves as the foundation for future research on:
- Graph Reinforcement Learning,
- PPO-based scheduling,
- Graph Transformers,
- Distributed scheduling,
- Multi-objective optimization,
- Energy-aware scheduling,
- Heterogeneous CPU/GPU scheduling.

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Running Experiments

## MLP Scheduling

```bash
cd MLP_approach

# run heuristic
python run.py

# run learning
cd training
python training.py
```

## GNN Scheduling

```bash
cd GNN_approach

# run heuristic
python run.py

# run learning
cd training
python training.py
```

---

# Papers

The associated research papers are available inside the `papers/` directory.

---

# Citation

```bibtex
@article{Lejaille2026MLP,
  title={Task Scheduling on Multi-CPU Graph Architectures using Supervised Policy Learning},
  author={Louis Lejaille},
  year={2026}
}

@article{Lejaille2026GNN,
  title={Graph Neural Networks for DAG Task Scheduling on Multi-CPU Architectures},
  author={Louis Lejaille},
  year={2026}
}
```

---

# Author

Louis Lejaille

Research interests:
- Graph Neural Networks
- Reinforcement Learning
- Distributed Systems
- Workflow Scheduling
- Representation Learning
- Systems AI