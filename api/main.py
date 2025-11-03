"""
FastAPI Production Endpoint for Intrusion Detection
Real-time network intrusion detection API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from joblib import load
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.models import (
    TrafficFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from src.config import MODELS

# Initialize FastAPI app
app = FastAPI(
    title="Network Intrusion Detection API",
    description="Real-time network intrusion detection using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
PREPROCESSOR = None
MODEL_NAME = "gradient_boosting"
START_TIME = time.time()

# Threshold for reducing false positives
# 0.5 = default (balanced)
# 0.7 = recommended (82% less false positives)
# 0.9 = ultra-conservative (99% precision)
DECISION_THRESHOLD = 0.7


def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    global MODEL, PREPROCESSOR, MODEL_NAME

    model_path = MODELS / f"{MODEL_NAME}.joblib"
    preprocessor_path = MODELS / "preprocess_ct.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    MODEL = load(model_path)
    PREPROCESSOR = load(preprocessor_path)

    print(f"✓ Loaded model: {MODEL_NAME}")
    print(f"✓ Loaded preprocessor")


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    try:
        load_model_and_preprocessor()
        print("✓ API ready to serve predictions")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def features_to_dataframe(features: TrafficFeatures) -> pd.DataFrame:
    """Convert TrafficFeatures to DataFrame matching preprocessing format"""

    # Create dictionary from features
    data = features.model_dump()

    # Create DataFrame with single row
    df = pd.DataFrame([data])

    return df


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability >= 0.9 or probability <= 0.1:
        return "high"
    elif probability >= 0.7 or probability <= 0.3:
        return "medium"
    else:
        return "low"


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network Intrusion Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_name=MODEL_NAME,
        uptime_seconds=time.time() - START_TIME
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the currently loaded model"""

    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    # Load metrics if available
    metrics_path = MODELS.parent / "reports" / "model_comparison_metrics.csv"
    metrics = {
        "roc_auc": 0.9787,
        "accuracy": 0.924,
        "f1_score": 0.9287
    }

    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path, index_col=0)
            model_display_name = MODEL_NAME.replace('_', ' ').title()
            if model_display_name in df.index:
                metrics = {
                    "roc_auc": float(df.loc[model_display_name, 'ROC AUC']),
                    "accuracy": float(df.loc[model_display_name, 'Accuracy']),
                    "f1_score": float(df.loc[model_display_name, 'F1 Score'])
                }
        except Exception:
            pass

    return ModelInfo(
        name=MODEL_NAME,
        type=type(MODEL).__name__,
        roc_auc=metrics["roc_auc"],
        accuracy=metrics["accuracy"],
        f1_score=metrics["f1_score"],
        trained_date=None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: TrafficFeatures, threshold: float = None):
    """
    Predict whether network traffic is normal or an attack.

    Args:
        features: Network traffic features
        threshold: Decision threshold (optional, default=0.7)
                  - 0.5 = balanced
                  - 0.7 = recommended (82% less false positives)
                  - 0.9 = ultra-conservative (99% precision, ~88 FP/day)

    Returns:
        - prediction: "normal" or "attack"
        - probability: Attack probability (0-1)
        - confidence: "low", "medium", or "high"
    """

    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    # Use provided threshold or default
    decision_threshold = threshold if threshold is not None else DECISION_THRESHOLD

    # Validate threshold
    if not 0 <= decision_threshold <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Threshold must be between 0 and 1"
        )

    try:
        # Convert to DataFrame
        df = features_to_dataframe(features)

        # Preprocess
        X = PREPROCESSOR.transform(df)

        # Get probability
        if hasattr(MODEL, 'predict_proba'):
            probability = float(MODEL.predict_proba(X)[0][1])
        else:
            # For models without predict_proba, use decision function
            score = float(MODEL.decision_function(X)[0])
            # Convert to probability-like score (0-1)
            probability = 1 / (1 + np.exp(-score))

        # Apply custom threshold
        prediction = 1 if probability >= decision_threshold else 0

        # Determine prediction label
        pred_label = "attack" if prediction == 1 else "normal"

        # Get confidence
        confidence = get_confidence_level(probability)

        return PredictionResponse(
            prediction=pred_label,
            probability=probability,
            confidence=confidence,
            model_used=MODEL_NAME,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest, threshold: float = None):
    """
    Predict multiple samples at once (batch prediction).

    Args:
        request: Batch of traffic samples
        threshold: Decision threshold (optional, default=0.7)

    More efficient than calling /predict multiple times.
    """

    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    # Use provided threshold or default
    decision_threshold = threshold if threshold is not None else DECISION_THRESHOLD

    # Validate threshold
    if not 0 <= decision_threshold <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Threshold must be between 0 and 1"
        )

    try:
        start_time = time.time()

        # Convert all samples to DataFrame
        data = [sample.model_dump() for sample in request.samples]
        df = pd.DataFrame(data)

        # Preprocess
        X = PREPROCESSOR.transform(df)

        # Get probabilities
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(X)[:, 1]
        else:
            scores = MODEL.decision_function(X)
            probabilities = 1 / (1 + np.exp(-scores))

        # Apply custom threshold
        predictions = (probabilities >= decision_threshold).astype(int)

        # Create response for each sample
        responses = []
        attacks_detected = 0

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            pred_label = "attack" if pred == 1 else "normal"
            confidence = get_confidence_level(float(prob))

            if pred == 1:
                attacks_detected += 1

            responses.append(PredictionResponse(
                prediction=pred_label,
                probability=float(prob),
                confidence=confidence,
                model_used=MODEL_NAME,
                timestamp=datetime.now()
            ))

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return BatchPredictionResponse(
            predictions=responses,
            total_samples=len(request.samples),
            attacks_detected=attacks_detected,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    # Run with: python -m api.main
    # Or: uvicorn api.main:app --reload
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
