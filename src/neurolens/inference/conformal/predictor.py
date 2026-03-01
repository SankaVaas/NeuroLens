"""
Split Conformal Prediction for NeuroLens.

Provides distribution-free coverage guarantee:
    P(y_true ∈ C(x)) >= 1 - alpha

where C(x) is the prediction set and alpha is the miscoverage level.

No distributional assumptions are required — this guarantee holds for any
exchangeable (i.i.d.) dataset. This is the legally defensible confidence
bound required for clinical AI deployment.

Reference:
    Angelopoulos & Bates 2022 — "A Gentle Introduction to Conformal Prediction"
    Venn prediction / Mondrian conformal prediction for class-conditional coverage.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Split Conformal Predictor using softmax probability as conformity score.

    Nonconformity score: s(x, y) = 1 - softmax(f(x))[y]
    (Lower score = more conforming = more confident in label y)

    Calibration:
        Given N calibration samples, compute nonconformity scores s_1,...,s_N.
        Threshold q = quantile at level ceil((N+1)(1-alpha)) / N.

    Prediction:
        C(x) = {y : s(x,y) <= q}  — all labels within threshold.

    Args:
        alpha: Target miscoverage level. Default 0.05 (95% coverage).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None
        self.n_calibration: int = 0

    def calibrate(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Fit conformal threshold on calibration set.

        Args:
            probs: Predicted probabilities from MC inference (N_cal, C).
            labels: True labels (N_cal,).

        Returns:
            q_hat: Calibrated quantile threshold.
        """
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Nonconformity scores: 1 - probability of true label
        scores = 1.0 - probs_np[np.arange(len(labels_np)), labels_np]
        self.calibration_scores = scores
        self.n_calibration = len(scores)

        # Compute quantile with finite-sample correction
        level = min(
            np.ceil((self.n_calibration + 1) * (1 - self.alpha)) / self.n_calibration,
            1.0,
        )
        self.q_hat = float(np.quantile(scores, level))

        empirical_coverage = float(np.mean(scores <= self.q_hat))
        logger.info(
            f"Conformal calibrated: q_hat={self.q_hat:.4f}, "
            f"empirical_coverage={empirical_coverage:.4f} "
            f"(target={1-self.alpha:.4f}), N={self.n_calibration}"
        )
        return self.q_hat

    def predict_set(self, probs: torch.Tensor) -> List[List[int]]:
        """
        Generate prediction sets for test inputs.

        Args:
            probs: Predicted probabilities (B, C).

        Returns:
            List of prediction sets, one per sample.
            Each set contains class indices included at coverage 1-alpha.
        """
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_set()")

        probs_np = probs.cpu().numpy()
        prediction_sets = []

        for i in range(len(probs_np)):
            # Include all classes whose nonconformity score <= q_hat
            scores = 1.0 - probs_np[i]
            included = [int(c) for c in np.where(scores <= self.q_hat)[0]]
            # Guarantee non-empty prediction set
            if not included:
                included = [int(np.argmax(probs_np[i]))]
            prediction_sets.append(included)

        return prediction_sets

    def evaluate_coverage(
        self, probs: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """
        Evaluate empirical coverage on a test set.

        Args:
            probs: Predicted probabilities (N, C).
            labels: True labels (N,).

        Returns:
            Dictionary with coverage, avg_set_size, and set_size_distribution.
        """
        prediction_sets = self.predict_set(probs)
        labels_np = labels.cpu().numpy()

        covered = [
            int(labels_np[i]) in prediction_sets[i]
            for i in range(len(labels_np))
        ]
        set_sizes = [len(s) for s in prediction_sets]

        return {
            "empirical_coverage": float(np.mean(covered)),
            "target_coverage": 1 - self.alpha,
            "coverage_gap": float(np.mean(covered)) - (1 - self.alpha),
            "avg_set_size": float(np.mean(set_sizes)),
            "set_size_distribution": {
                str(s): int(np.sum(np.array(set_sizes) == s))
                for s in sorted(set(set_sizes))
            },
            "n_test": len(labels_np),
        }
