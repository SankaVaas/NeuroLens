"""Unit tests for VariationalLinear layer."""
import pytest
import torch
from neurolens.models.bayesian.variational_linear import VariationalLinear


@pytest.fixture
def layer():
    return VariationalLinear(in_features=64, out_features=32)


def test_output_shape(layer):
    x = torch.randn(4, 64)
    out, kl = layer(x)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"


def test_kl_nonnegative(layer):
    x = torch.randn(4, 64)
    _, kl = layer(x)
    assert kl.item() >= 0, f"KL divergence must be non-negative, got {kl.item()}"


def test_stochasticity(layer):
    """Each forward pass should produce different outputs (stochastic)."""
    x = torch.randn(1, 64)
    out1, _ = layer(x)
    out2, _ = layer(x)
    assert not torch.allclose(out1, out2), "Outputs should differ across forward passes"


def test_sigma_positive(layer):
    """Weight sigma must always be positive (softplus guarantee)."""
    assert (layer.weight_sigma > 0).all(), "All sigma values must be positive"


def test_set_prior(layer):
    prior_mu = torch.zeros(32, 64)
    prior_sigma = torch.ones(32, 64) * 0.5
    layer.set_prior(prior_mu, prior_sigma)
    assert torch.allclose(layer.prior_sigma, prior_sigma)
    assert torch.allclose(layer.prior_mu, prior_mu)
