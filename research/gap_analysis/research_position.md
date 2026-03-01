# NeuroLens — Research Positioning Document

## Core Research Question
Can we derive a mathematically rigorous Riemannian-Bayesian attention mechanism
for medical image transformers that provides provable generalization guarantees
and clinically trustworthy uncertainty — operating efficiently on CPU hardware?

## The 5 Novel Contributions

| # | Contribution | Status | Closest Existing Paper | Our Differentiation |
|---|---|---|---|---|
| 1 | Geodesic Variational Attention (GVA) | 🔬 In Progress | Standard MHSA | First Riemannian attention with Bayesian weights on Stiefel manifold |
| 2 | PAC-Bayes Bound for BayesViT with FIM Prior | 🔬 In Progress | Edelman et al. 2021 (deterministic ViT bound) | First bound for Bayesian ViT with structured (non-isotropic) prior |
| 3 | Uncertainty-Accuracy Pareto Theorem | 📋 Planned | None found | Novel impossibility result using convex duality + FIM eigenspectrum |
| 4 | FRAP — Federated Riemannian Aggregation | 📋 Planned | FedAvg (Euclidean only) | First geodesic mean aggregation for manifold-valued BNN posteriors |
| 5 | Deterministic UQ via Delta Method | 📋 Planned | MC Dropout (stochastic) | Analytical uncertainty propagation through Riemannian attention layers |

## Falsification Criteria
This research would be invalidated if:
- A NeurIPS/ICML 2024-2025 paper already derives GVA (search confirmed: NOT found as of Jan 2025)
- Geodesic attention scores produce no measurable improvement in calibration vs Euclidean attention
- FIM prior provides no tighter generalization bound than isotropic prior (testable via ablation)

## Target Venues
1. NeurIPS 2025/2026 — Theory paper (GVA + PAC-Bayes theorem)
2. Medical Image Analysis — Applied system paper (NeuroLens full system)
3. MICCAI 2026 — Clinical application paper

## Weekly Research Log
Track your progress here: [link to research/experiments/]
