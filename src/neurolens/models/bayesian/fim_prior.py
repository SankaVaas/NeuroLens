"""
Fisher Information Matrix (FIM) Prior for Bayesian Neural Networks.

The FIM captures the curvature of the loss landscape around pretrained weights.
Parameters with high Fisher information (high curvature) receive tight priors
(small sigma), preserving important pretrained knowledge.
Parameters with low Fisher information receive broad priors (large sigma),
allowing flexible adaptation to new data.

Mathematical Basis:
    F_ii = E[(d log p(y|x,w) / dw_i)^2]  (empirical Fisher diagonal)
    prior_sigma_i = 1 / sqrt(F_ii + epsilon)

    This is equivalent to a Laplace approximation around the pretrained weights,
    but used as a prior rather than a posterior — the key novel framing.

Reference:
    Kirkpatrick et al. 2017 — Elastic Weight Consolidation (EWC)
    → We adapt FIM from regularization into a principled Bayesian prior.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FisherInformationPrior:
    """
    Estimates the empirical Fisher diagonal from a reference dataset
    and uses it to set individualized priors on VariationalLinear layers.

    Args:
        model: The pretrained model (deterministic ViT-Small).
        dataloader: Small reference DataLoader (100-500 samples sufficient).
        device: Compute device.
        epsilon: Regularization constant to avoid division by zero.
        sigma_max: Maximum prior sigma (floor on precision). Default 1.0.
        sigma_min: Minimum prior sigma (ceiling on precision). Default 1e-4.
        num_samples: Number of batches used for FIM estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epsilon: float = 1e-8,
        sigma_max: float = 1.0,
        sigma_min: float = 1e-4,
        num_samples: int = 100,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.epsilon = epsilon
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_samples = num_samples
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}

    def estimate(self) -> Dict[str, torch.Tensor]:
        """
        Estimate empirical Fisher diagonal over reference dataset.

        Returns:
            Dictionary mapping parameter name → Fisher diagonal tensor.
        """
        logger.info("Estimating Fisher Information Matrix diagonal...")
        self.model.eval()

        # Initialize Fisher accumulator
        fisher_accum = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_samples = 0
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if n_samples >= self.num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            logits = self.model(images)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Sample from predictive distribution (Monte Carlo FIM estimate)
            sampled_labels = torch.distributions.Categorical(
                logits=logits
            ).sample()
            loss = -log_probs[range(len(sampled_labels)), sampled_labels].mean()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.detach() ** 2

            n_samples += images.size(0)

        # Normalize by number of samples
        for name in fisher_accum:
            fisher_accum[name] /= max(n_samples, 1)

        self.fisher_diagonal = fisher_accum
        logger.info(f"FIM estimated over {n_samples} samples.")
        return fisher_accum

    def get_prior_sigma(self, param_name: str) -> Optional[torch.Tensor]:
        """
        Convert Fisher diagonal to prior sigma for a given parameter.

        Args:
            param_name: Named parameter key (from model.named_parameters()).

        Returns:
            Prior sigma tensor. Returns None if FIM not yet estimated.
        """
        if not self.fisher_diagonal:
            raise RuntimeError("Call .estimate() before .get_prior_sigma()")

        if param_name not in self.fisher_diagonal:
            return None

        fisher = self.fisher_diagonal[param_name]
        # sigma = 1 / sqrt(F + epsilon) — clipped for numerical stability
        sigma = 1.0 / torch.sqrt(fisher + self.epsilon)
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        return sigma

    def apply_to_bayes_vit(self, bayes_vit: nn.Module) -> None:
        """
        Apply FIM-informed priors to all VariationalLinear layers in BayesViT.

        This modifies the prior_mu and prior_sigma buffers of each
        VariationalLinear layer in-place.

        Args:
            bayes_vit: The BayesViT model with VariationalLinear layers.
        """
        from neurolens.models.bayesian.variational_linear import VariationalLinear

        if not self.fisher_diagonal:
            raise RuntimeError("Call .estimate() before .apply_to_bayes_vit()")

        applied_count = 0
        for name, module in bayes_vit.named_modules():
            if isinstance(module, VariationalLinear):
                # Match parameter names from pretrained backbone
                weight_key = f"{name}.weight"
                sigma = self.get_prior_sigma(weight_key)

                if sigma is not None:
                    # prior_mu = pretrained weight values (preserve knowledge)
                    prior_mu = self.fisher_diagonal.get(
                        weight_key, torch.zeros_like(module.weight_mu)
                    )
                    module.set_prior(
                        prior_mu=module.weight_mu.data.clone(),
                        prior_sigma=sigma.reshape(module.weight_mu.shape)
                        if sigma.shape != module.weight_mu.shape
                        else sigma,
                    )
                    applied_count += 1

        logger.info(f"FIM prior applied to {applied_count} VariationalLinear layers.")
