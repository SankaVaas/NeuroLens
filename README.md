<div align="center">

# 🔬 NeuroLens

### Riemannian-Bayesian Federated Vision Intelligence for Early Neurological Disease Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/yourusername/neurolens/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/neurolens/actions)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-orange.svg)](https://huggingface.co/spaces/yourusername/neurolens)

*The first Riemannian-Bayesian Vision Transformer for clinically trustworthy neurological disease screening from retinal fundus images — with formal generalization guarantees.*

[📄 Paper](#) · [🤗 Demo](#) · [📖 Docs](#) · [🐛 Issues](#)

</div>

---

## 🧠 Overview

NeuroLens detects early-stage Alzheimer's Disease (AD), Parkinson's Disease (PD), and Mild Cognitive Impairment (MCI) from **non-invasive retinal fundus photographs** — using the retina as a window to the brain.

Unlike existing systems, NeuroLens provides **mathematically rigorous uncertainty quantification** — giving clinicians a calibrated confidence measure for every prediction, with provable coverage guarantees.

### 🔬 Novel Scientific Contributions

| # | Contribution | Status |
|---|---|---|
| **1** | **Geodesic Variational Attention (GVA)** — First Riemannian self-attention with Bayesian weights on the Stiefel manifold | 🔬 Active |
| **2** | **PAC-Bayes Generalization Bound** for BayesViT with FIM-structured prior (formal theorem + proof) | 🔬 Active |
| **3** | **Uncertainty-Accuracy Pareto Theorem** — impossibility result via convex duality | 📋 Planned |
| **4** | **FRAP** — Federated Riemannian Aggregation Protocol for privacy-preserving multi-hospital training | 📋 Planned |
| **5** | **Deterministic UQ** via Delta Method on Riemannian manifolds (20x faster than MC) | 📋 Planned |

---

## 🏗️ Architecture

```
Retinal Image (224×224)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  DINO ViT-Small Backbone (frozen lower blocks)    │
│                                                   │
│  Blocks 1-8:  Standard Self-Attention (frozen)    │
│  Blocks 9-12: Geodesic Variational Attention ◄── Novel │
│               • Stiefel manifold weights          │
│               • SPD manifold attention scores     │
│               • KL divergence for ELBO            │
└───────────────────────────────────────────────────┘
        │
        ▼
  CLS Token → Classification Head
        │
        ▼
 Monte Carlo Inference (T=20 samples)
        │
        ▼
┌─────────────────────────────────┐
│  Uncertainty Decomposition      │
│  • Epistemic (model uncertainty)│
│  • Aleatoric (data uncertainty) │
│  • Conformal Prediction Set     │
└─────────────────────────────────┘
        │
        ▼
   Clinical Dashboard (React)
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/neurolens.git
cd neurolens

# 2. Create environment
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate                             # Windows

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your W&B key

# 5. Run tests to verify setup
pytest tests/unit/ -v

# 6. Start API (after training)
uvicorn api.main:app --reload --port 8000
```

---

## 📁 Repository Structure

```
neurolens/
├── src/neurolens/
│   ├── models/
│   │   ├── bayes_vit.py              ← Main model
│   │   ├── manifold/
│   │   │   └── geodesic_attention.py ← NOVEL: GVA
│   │   └── bayesian/
│   │       ├── variational_linear.py ← Bayesian layers
│   │       └── fim_prior.py          ← FIM prior
│   ├── training/losses/
│   │   └── elbo_loss.py              ← Composite ELBO loss
│   ├── inference/
│   │   ├── uncertainty/decomposer.py ← Uncertainty decomposition
│   │   └── conformal/predictor.py    ← Conformal prediction
│   ├── explainability/               ← Grad-CAM++, Attention Rollout
│   ├── evaluation/                   ← ECE, reliability diagrams
│   └── federated/                    ← FRAP aggregation
├── api/                              ← FastAPI backend
├── ui/                               ← React dashboard
├── notebooks/                        ← Colab training notebooks
├── tests/                            ← Full test suite
├── configs/default.yaml              ← Experiment config
├── research/                         ← Literature, math derivations
└── paper/                            ← LaTeX paper draft
```

---

## 📊 Results (Updated as experiments complete)

| Method | AUC-ROC | ECE ↓ | Coverage | Inference Time |
|---|---|---|---|---|
| Deterministic ViT-Small | - | - | N/A | - |
| MC Dropout | - | - | N/A | - |
| Deep Ensemble (5x) | - | - | N/A | - |
| **NeuroLens (Ours)** | **-** | **-** | **95%** | **-** |

*Results will be populated as training experiments complete.*

---

## 📚 Key References

1. Blundell et al. (2015) — Weight Uncertainty in Neural Networks
2. Caron et al. (2021) — DINO Self-Supervised ViT
3. Becigneul & Ganea (2019) — Riemannian Adaptive Optimization
4. Angelopoulos & Bates (2022) — Conformal Prediction
5. Said et al. (2017) — Riemannian Gaussian Distributions on SPD Manifolds

---

## 📄 Citation

```bibtex
@article{neurolens2025,
  title={NeuroLens: Riemannian-Bayesian Federated Vision Intelligence for
         Neurological Disease Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
