"""
FastAPI Prediction Backend — Employee Attrition Predictor
=========================================================
This service exposes REST API endpoints for employee attrition prediction,
model metadata inspection, and service health checks.

Endpoints:
----------
- GET  /health      : Operational status and model readiness.
- GET  /model-info  : Verified architecture metadata, feature schemas, and CV metrics.
- POST /predict     : Validated 30-feature inference with server-side feature engineering.
"""

import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    EmployeePredictionRequest,
    EmployeePredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from backend.model_service import model_service

# Configure clean logging: log operational events without logging employee salary or personal data
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("attrition_api")

app = FastAPI(
    title="Employee Attrition Predictor API",
    description=(
        "Production-grade decision-support API for estimating employee attrition probability "
        "using a Scikit-Learn classification pipeline trained on an 80:20 stratified split."
    ),
    version="1.0.0"
)

# Enable CORS middleware for external client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Probe",
    tags=["Diagnostics"]
)
def get_health() -> HealthResponse:
    """
    Check the operational status of the API and verify that the trained model is loaded.
    Returns HTTP 200 if healthy, or HTTP 503 if model initialization failed.
    """
    health = model_service.get_health()
    if not health.model_loaded:
        logger.error("Health check failed: model artifact is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded or failed initialization."
        )
    return health


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Architecture & Metadata",
    tags=["Diagnostics"]
)
def get_model_info() -> ModelInfoResponse:
    """
    Retrieve verified model architecture, feature schema counts, class definitions,
    and cross-validation evaluation scores.
    """
    try:
        return model_service.get_model_info()
    except Exception as e:
        logger.error("Failed to retrieve model info: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable."
        )


@app.post(
    "/predict",
    response_model=EmployeePredictionResponse,
    summary="Evaluate Employee Attrition Risk",
    tags=["Inference"],
    status_code=status.HTTP_200_OK
)
def predict_attrition(request: EmployeePredictionRequest) -> EmployeePredictionResponse:
    """
    Accepts 30 raw employee features, validates ranges and types, computes tenure
    ratios server-side with zero-safe arithmetic, executes model inference, and
    returns estimated probabilities alongside neutral, non-prescriptive risk status text.

    Note: Employee data is processed in memory and never stored or persisted.
    """
    try:
        # Execute prediction through model service
        response = model_service.predict(request)
        logger.info(
            "Inference completed successfully: prediction=%d, probability=%.4f",
            response.prediction,
            response.probability_attrition
        )
        return response
    except ValueError as ve:
        logger.warning("Inference validation error: %s", str(ve))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except RuntimeError as re:
        logger.error("Service runtime error: %s", str(re))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is temporarily unavailable."
        )
    except Exception as e:
        logger.error("Unexpected inference execution error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction processing."
        )


if __name__ == "__main__":
    import uvicorn
    # Support Render dynamic port assignment via environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
