"""Pydantic response schemas for NeuroLens API."""
from pydantic import BaseModel
from typing import Dict, List, Optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class UncertaintyInfo(BaseModel):
    predictive_entropy: float
    epistemic: float
    aleatoric: float
    quality_flag: int   # 0=low, 1=medium_epistemic, 2=high_aleatoric, 3=critical


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float]
    uncertainty: UncertaintyInfo
    conformal_set: List[int]
    conformal_set_labels: List[str]
    heatmap_base64: Optional[str] = None   # Grad-CAM++ overlay
    coverage_guarantee: float = 0.95
