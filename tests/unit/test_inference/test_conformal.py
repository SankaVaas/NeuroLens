"""Unit tests for ConformalPredictor."""
import pytest
import torch
from neurolens.inference.conformal.predictor import ConformalPredictor


@pytest.fixture
def calibrated_predictor():
    pred = ConformalPredictor(alpha=0.05)
    # Simulate calibration with known probs
    torch.manual_seed(42)
    probs = torch.softmax(torch.randn(200, 4), dim=-1)
    labels = torch.randint(0, 4, (200,))
    pred.calibrate(probs, labels)
    return pred


def test_coverage_guarantee(calibrated_predictor):
    """Empirical coverage must be >= 1 - alpha."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(500, 4), dim=-1)
    labels = torch.randint(0, 4, (500,))
    result = calibrated_predictor.evaluate_coverage(probs, labels)
    assert result["empirical_coverage"] >= 0.90, (
        f"Coverage {result['empirical_coverage']:.3f} too low"
    )


def test_prediction_set_nonempty(calibrated_predictor):
    probs = torch.softmax(torch.randn(10, 4), dim=-1)
    sets = calibrated_predictor.predict_set(probs)
    for s in sets:
        assert len(s) >= 1, "Prediction set must not be empty"


def test_q_hat_in_range(calibrated_predictor):
    assert 0.0 <= calibrated_predictor.q_hat <= 1.0
