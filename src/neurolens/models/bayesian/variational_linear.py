"""
Variational Linear Layer — Core Bayesian Building Block.

Implements Gaussian weight parameterization via the reparameterization trick
as described in Blundell et al. 2015 (Bayes by Backprop).

Mathematical Formulation:
    w ~ N(mu, sigma^2),  sigma = softplus(rho)
    w_sampled = mu + sigma * epsilon,  epsilon ~ N(0, I)

    KL[q(w|mu,sigma) || p(w|prior_mu, prior_sigma)] computed analytically.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VariationalLinear(nn.Module):
    """
    Bayesian linear layer with Gaussian weight distributions.

    Each weight w_ij is parameterized by (mu_ij, rho_ij) where
    sigma_ij = softplus(rho_ij) to ensure positivity.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term (also made variational).
        prior_mu: Prior mean for weights. Default 0.0.
        prior_sigma: Prior std for weights. Set by FIM prior externally.
        device: Compute device.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # ── Variational parameters for weights ─────────────────────────────
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features, device=device)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features, device=device)
        )

        # ── Variational parameters for bias ────────────────────────────────
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, device=device))
            self.bias_rho = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        # ── Prior parameters (set externally by FIMPrior) ──────────────────
        self.register_buffer(
            "prior_mu", torch.full((out_features, in_features), prior_mu)
        )
        self.register_buffer(
            "prior_sigma", torch.full((out_features, in_features), prior_sigma)
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize mu near zero and rho to give sigma ≈ 0.1 initially."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # softplus^{-1}(0.1) ≈ -2.25
        nn.init.constant_(self.weight_rho, -2.25)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -2.25)

    @property
    def weight_sigma(self) -> torch.Tensor:
        """Compute sigma from rho via softplus (ensures positivity)."""
        return F.softplus(self.weight_rho)

    @property
    def bias_sigma(self) -> Optional[torch.Tensor]:
        if self.use_bias:
            return F.softplus(self.bias_rho)
        return None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with reparameterization trick.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            output: Sampled linear output (batch, out_features).
            kl_loss: KL divergence for this layer (scalar).
        """
        # ── Sample weights via reparameterization ──────────────────────────
        epsilon_w = torch.randn_like(self.weight_sigma)
        weight = self.weight_mu + self.weight_sigma * epsilon_w

        bias = None
        if self.use_bias:
            epsilon_b = torch.randn_like(self.bias_sigma)
            bias = self.bias_mu + self.bias_sigma * epsilon_b

        output = F.linear(x, weight, bias)
        kl = self._compute_kl()
        return output, kl

    def _compute_kl(self) -> torch.Tensor:
        """
        Analytical KL divergence: KL[N(mu, sigma^2) || N(prior_mu, prior_sigma^2)].

        KL = log(prior_sigma/sigma) + (sigma^2 + (mu - prior_mu)^2) / (2*prior_sigma^2) - 0.5
        """
        kl = (
            torch.log(self.prior_sigma / self.weight_sigma)
            + (self.weight_sigma**2 + (self.weight_mu - self.prior_mu) ** 2)
            / (2 * self.prior_sigma**2)
            - 0.5
        )
        return kl.sum()

    def set_prior(self, prior_mu: torch.Tensor, prior_sigma: torch.Tensor) -> None:
        """
        Set FIM-informed prior. Called by FisherInformationPrior.

        Args:
            prior_mu: Tensor of shape (out_features, in_features).
            prior_sigma: Tensor of shape (out_features, in_features).
        """
        self.prior_mu.data = prior_mu.to(self.prior_mu.device)
        self.prior_sigma.data = prior_sigma.to(self.prior_sigma.device)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.use_bias}"
