"""
Uncertainty Decomposition for BayesViT.

Decomposes total predictive uncertainty into:
    - Epistemic (model/knowledge uncertainty) — reducible with more data
    - Aleatoric (data/irreducible uncertainty) — inherent noise

Method: Monte Carlo integration over the variational posterior.
    p(y|x) ≈ (1/T) * sum_t p(y|x, W_t),  W_t ~ q(W)

Predictive Entropy (total):
    H[y|x] = -sum_c p(y=c|x) * log p(y=c|x)

Epistemic (Mutual Information I[y;W|x]):
    I[y;W|x] = H[y|x] - E_q[H[y|x,W]]

Aleatoric:
    E_q[H[y|x,W]] = (1/T) * sum_t H[y|x, W_t]
"""

import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class UncertaintyDecomposer:
    """
    Decomposes MC inference results into epistemic and aleatoric uncertainty.

    Args:
        epsilon: Small constant for numerical stability in log. Default 1e-8.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def decompose(self, prob_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty from Monte Carlo probability samples.

        Args:
            prob_samples: MC probability samples (T, B, C) where
                T = number of MC samples, B = batch size, C = num classes.

        Returns:
            Dictionary with:
                mean_probs      (B, C)  — mean predictive distribution
                predictive_entropy (B)  — total uncertainty H[y|x]
                epistemic       (B)     — I[y;W|x] = mutual information
                aleatoric       (B)     — E_q[H[y|x,W]]
                variance        (B, C)  — predictive variance per class
        """
        T, B, C = prob_samples.shape

        # ── Mean predictive distribution ───────────────────────────────────
        mean_probs = prob_samples.mean(dim=0)  # (B, C)

        # ── Predictive entropy H[y|x] ──────────────────────────────────────
        pred_entropy = self._entropy(mean_probs)  # (B,)

        # ── Aleatoric: mean entropy per MC sample ──────────────────────────
        per_sample_entropy = torch.stack(
            [self._entropy(prob_samples[t]) for t in range(T)], dim=0
        )  # (T, B)
        aleatoric = per_sample_entropy.mean(dim=0)  # (B,)

        # ── Epistemic: mutual information (always >= 0, clamp numerical errors)
        epistemic = (pred_entropy - aleatoric).clamp(min=0.0)  # (B,)

        # ── Predictive variance ────────────────────────────────────────────
        variance = prob_samples.var(dim=0)  # (B, C)

        return {
            "mean_probs": mean_probs,
            "predictive_entropy": pred_entropy,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "variance": variance,
        }

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H(p) = -sum p log p."""
        return -(probs * torch.log(probs + self.epsilon)).sum(dim=-1)

    def uncertainty_quality_flag(
        self,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
        epistemic_thresh: float = 0.5,
        aleatoric_thresh: float = 0.7,
    ) -> torch.Tensor:
        """
        Classify uncertainty quality for clinical dashboard.

        Returns:
            flags: Integer tensor (B,) with values:
                0 = Low uncertainty (confident, high quality)
                1 = Medium epistemic (OOD — refer for more data)
                2 = High aleatoric (poor image quality)
                3 = High total (do not use prediction)
        """
        flags = torch.zeros(epistemic.shape[0], dtype=torch.long)
        flags[epistemic > epistemic_thresh] = 1
        flags[aleatoric > aleatoric_thresh] = 2
        flags[(epistemic > epistemic_thresh) & (aleatoric > aleatoric_thresh)] = 3
        return flags
