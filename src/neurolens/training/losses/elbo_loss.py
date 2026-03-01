"""
Composite ELBO Loss for BayesViT Training.

L_total = L_CE + beta(t) * L_KL + gamma * L_ECE

Components:
    L_CE  — Cross-entropy classification loss (with class weights for imbalance)
    L_KL  — KL divergence between variational posterior and FIM prior (ELBO term)
    L_ECE — Differentiable Expected Calibration Error (optimizes calibration directly)
    beta  — Cyclically annealed KL weight (prevents posterior collapse)
    gamma — Fixed calibration weight (default 0.1)

Cyclical KL Annealing (Fu et al. 2019):
    beta cycles from 0 to 1 over T_cycle steps, repeated M times.
    This prevents the model from ignoring the KL term early in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CyclicalBetaScheduler:
    """
    Cyclical annealing schedule for KL weight beta.

    Args:
        n_epochs: Total training epochs.
        n_cycles: Number of annealing cycles. Default 4.
        ratio: Fraction of cycle spent at beta=1 (plateau). Default 0.5.
    """

    def __init__(self, n_epochs: int, n_cycles: int = 4, ratio: float = 0.5) -> None:
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.betas = self._compute_schedule()

    def _compute_schedule(self):
        betas = []
        period = self.n_epochs / self.n_cycles
        for epoch in range(self.n_epochs):
            cycle_pos = (epoch % period) / period
            if cycle_pos < self.ratio:
                beta = cycle_pos / self.ratio  # Linear ramp 0→1
            else:
                beta = 1.0  # Plateau at 1
            betas.append(beta)
        return betas

    def get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch."""
        if epoch >= len(self.betas):
            return 1.0
        return self.betas[epoch]


class DifferentiableECELoss(nn.Module):
    """
    Differentiable Expected Calibration Error loss.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    For differentiability, we use soft bin assignment via temperature-scaled
    softmax over bin centers, following Karandikar et al. 2021.

    Args:
        n_bins: Number of calibration bins. Default 15.
        temperature: Softmax temperature for soft binning. Default 0.05.
    """

    def __init__(self, n_bins: int = 15, temperature: float = 0.05) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.temperature = temperature
        # Bin centers: [1/(2B), 3/(2B), ..., (2B-1)/(2B)]
        self.register_buffer(
            "bin_centers",
            torch.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins),
        )

    def forward(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable ECE.

        Args:
            probs: Predicted probabilities (B, C).
            labels: Ground truth labels (B,).

        Returns:
            ece: Differentiable ECE scalar.
        """
        confidences, predictions = probs.max(dim=-1)  # (B,)
        accuracies = (predictions == labels).float()  # (B,)

        # Soft bin assignment: (B, n_bins)
        bin_assignments = F.softmax(
            -torch.abs(confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0))
            / self.temperature,
            dim=-1,
        )

        # Weighted accuracy and confidence per bin
        bin_weights = bin_assignments.sum(dim=0) + 1e-8  # (n_bins,)
        bin_acc = (bin_assignments * accuracies.unsqueeze(1)).sum(dim=0) / bin_weights
        bin_conf = (
            bin_assignments * confidences.unsqueeze(1)
        ).sum(dim=0) / bin_weights

        # ECE = weighted absolute calibration error
        ece = (bin_weights / bin_weights.sum() * torch.abs(bin_acc - bin_conf)).sum()
        return ece


class ELBOLoss(nn.Module):
    """
    Composite ELBO loss: L_CE + beta * L_KL + gamma * L_ECE.

    Args:
        n_classes: Number of classes.
        class_weights: Optional inverse-frequency class weights for imbalance.
        gamma: ECE loss weight. Default 0.1.
        n_bins: Number of calibration bins for ECE. Default 15.
        dataset_size: Training set size N (for KL scaling: KL/N per batch).
    """

    def __init__(
        self,
        n_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 0.1,
        n_bins: int = 15,
        dataset_size: int = 1000,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.dataset_size = dataset_size

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.ece_loss = DifferentiableECELoss(n_bins=n_bins)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        kl: torch.Tensor,
        beta: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> dict:
        """
        Compute composite loss.

        Args:
            logits: Model logits (B, C).
            labels: Ground truth (B,).
            kl: KL divergence from BayesViT forward pass.
            beta: Current KL annealing weight.
            batch_size: Current batch size (for KL normalization).

        Returns:
            Dictionary with total loss and individual components.
        """
        B = batch_size or logits.size(0)
        probs = torch.softmax(logits, dim=-1)

        l_ce = self.ce_loss(logits, labels)
        # Scale KL by batch_size / dataset_size (per-datapoint ELBO estimate)
        l_kl = kl * (B / self.dataset_size)
        l_ece = self.ece_loss(probs, labels)

        total = l_ce + beta * l_kl + self.gamma * l_ece

        return {
            "total": total,
            "ce": l_ce.detach(),
            "kl": l_kl.detach(),
            "ece": l_ece.detach(),
            "beta": beta,
        }
