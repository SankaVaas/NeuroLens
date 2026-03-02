"""
NeuroLens FastAPI Backend.

Endpoints:
    POST /api/v1/predict    — Full diagnostic inference with uncertainty
    POST /api/v1/quality    — Image quality screening
    GET  /api/v1/health     — System health check
    GET  /api/v1/model_info — Model metadata

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import io
from PIL import Image
import logging

from api.services.inference_service import InferenceService
from api.schemas.prediction import PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference service (loaded once at startup)
inference_service: InferenceService = None

CLASS_NAMES = {0: "Alzheimer's Disease", 1: "Parkinson's Disease", 2: "MCI", 3: "Healthy"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global inference_service
    logger.info("Loading NeuroLens inference service...")
    inference_service = InferenceService(
        model_path="checkpoints/best_model.pt",
        device=torch.device("cpu"),
        n_mc_samples=20,
        conformal_alpha=0.05,
    )
    logger.info("NeuroLens ready.")
    yield
    logger.info("Shutting down NeuroLens.")


app = FastAPI(
    title="NeuroLens API",
    description="Riemannian-Bayesian Neurological Disease Detection from Retinal Fundus Images",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=inference_service is not None)


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Full neurological disease diagnostic with uncertainty quantification.

    Returns diagnosis, confidence, uncertainty decomposition, conformal set.
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = await inference_service.predict(image)
    return PredictionResponse(**result, class_names=CLASS_NAMES)


@app.get("/api/v1/model_info")
async def model_info():
    return {
        "model": "BayesViT with Geodesic Variational Attention",
        "backbone": "DINO ViT-Small (patch8)",
        "novel_components": ["GeodesicVariationalAttention", "FIMPrior", "ConformalPredictor"],
        "version": "0.1.0",
        "classes": CLASS_NAMES,
        "uncertainty_types": ["epistemic", "aleatoric", "predictive_entropy"],
    }
