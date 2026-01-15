"""
FastAPI Inference Service for Fraud Detection

Objective: Serve predictions in real-time via REST API.

Features:
- /predict endpoint for fraud detection
- Input validation with Pydantic
- Model artifacts loaded at startup
- Health check endpoint
- Swagger documentation

Author: Your Name
Date: 2026-01-15
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime
import os
import time

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
# Request metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'API request latency in seconds',
    ['method', 'endpoint']
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    'fraud_predictions_total',
    'Total fraud predictions',
    ['prediction', 'confidence']
)

FRAUD_PROBABILITY = Histogram(
    'fraud_probability_score',
    'Distribution of fraud probability scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Error metrics
ERROR_COUNT = Counter(
    'api_errors_total',
    'Total API errors',
    ['error_type']
)

MODEL_ERRORS = Counter(
    'model_prediction_errors_total',
    'Total model prediction errors'
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of active requests'
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for credit card transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to track request metrics
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics for Prometheus."""
    # Skip metrics endpoint
    if request.url.path == "/metrics":
        return await call_next(request)

    # Increment active requests
    ACTIVE_REQUESTS.inc()

    # Track request start time
    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Calculate latency
        latency = time.time() - start_time

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(latency)

        return response

    except Exception as e:
        # Record error
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise

    finally:
        # Decrement active requests
        ACTIVE_REQUESTS.dec()

# Global variables for model artifacts
model = None
scaler = None
threshold = None
feature_names = None


class TransactionFeatures(BaseModel):
    """Input schema for transaction features."""
    
    Time: float = Field(..., description="Time elapsed since first transaction")
    V1: float = Field(..., description="PCA feature 1")
    V2: float = Field(..., description="PCA feature 2")
    V3: float = Field(..., description="PCA feature 3")
    V4: float = Field(..., description="PCA feature 4")
    V5: float = Field(..., description="PCA feature 5")
    V6: float = Field(..., description="PCA feature 6")
    V7: float = Field(..., description="PCA feature 7")
    V8: float = Field(..., description="PCA feature 8")
    V9: float = Field(..., description="PCA feature 9")
    V10: float = Field(..., description="PCA feature 10")
    V11: float = Field(..., description="PCA feature 11")
    V12: float = Field(..., description="PCA feature 12")
    V13: float = Field(..., description="PCA feature 13")
    V14: float = Field(..., description="PCA feature 14")
    V15: float = Field(..., description="PCA feature 15")
    V16: float = Field(..., description="PCA feature 16")
    V17: float = Field(..., description="PCA feature 17")
    V18: float = Field(..., description="PCA feature 18")
    V19: float = Field(..., description="PCA feature 19")
    V20: float = Field(..., description="PCA feature 20")
    V21: float = Field(..., description="PCA feature 21")
    V22: float = Field(..., description="PCA feature 22")
    V23: float = Field(..., description="PCA feature 23")
    V24: float = Field(..., description="PCA feature 24")
    V25: float = Field(..., description="PCA feature 25")
    V26: float = Field(..., description="PCA feature 26")
    V27: float = Field(..., description="PCA feature 27")
    V28: float = Field(..., description="PCA feature 28")
    Amount: float = Field(..., ge=0, description="Transaction amount (must be >= 0)")
    
    @validator('Amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        if v > 1000000:  # Reasonable upper limit
            logger.warning(f"Unusually high transaction amount: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""
    
    is_fraud: int = Field(..., description="Fraud prediction (0=legitimate, 1=fraud)")
    fraud_probability: float = Field(..., description="Probability of fraud (0.0 to 1.0)")
    threshold: float = Field(..., description="Classification threshold used")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": 0,
                "fraud_probability": 0.0234,
                "threshold": 0.9,
                "confidence": "high",
                "timestamp": "2026-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    threshold: Optional[float]
    timestamp: str


@app.on_event("startup")
async def load_model_artifacts():
    """Load model artifacts on startup."""
    global model, scaler, threshold, feature_names

    try:
        logger.info("Loading model artifacts...")

        # Load model
        model_path = 'artifacts/fraud_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded model: {type(model).__name__}")

        # Load scaler
        scaler_path = 'artifacts/scaler.pkl'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Loaded scaler: {type(scaler).__name__}")

        # Set model loaded metric
        MODEL_LOADED.set(1)
        
        # Load threshold
        threshold_path = 'artifacts/threshold.txt'
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"Threshold not found: {threshold_path}")
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        logger.info(f"✓ Loaded threshold: {threshold}")
        
        # Define feature names
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        logger.info("✓ Model artifacts loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        MODEL_LOADED.set(0)
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        threshold=threshold,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionFeatures):
    """
    Predict fraud probability for a credit card transaction.

    **Input:** Transaction features (30 features: Time, V1-V28, Amount)

    **Output:** Fraud prediction with probability and confidence level

    **Example:**
    ```json
    {
        "Time": 0.0,
        "V1": -1.36, "V2": -0.07, ..., "V28": -0.02,
        "Amount": 149.62
    }
    ```
    """
    # Check if model is loaded
    if model is None or scaler is None or threshold is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert input to DataFrame
        features_dict = transaction.dict()
        X = pd.DataFrame([features_dict], columns=feature_names)

        # Scale features
        X_scaled = scaler.transform(X)

        # Preserve feature names for models that need them
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

        # Get prediction probability
        fraud_probability = float(model.predict_proba(X_scaled)[0, 1])

        # Apply threshold
        is_fraud = int(fraud_probability >= threshold)

        # Determine confidence level
        if fraud_probability < 0.3:
            confidence = "high"  # High confidence it's legitimate
        elif fraud_probability < 0.7:
            confidence = "medium"
        else:
            confidence = "high"  # High confidence it's fraud

        # Record metrics
        prediction_label = "fraud" if is_fraud else "legitimate"
        PREDICTION_COUNT.labels(
            prediction=prediction_label,
            confidence=confidence
        ).inc()

        FRAUD_PROBABILITY.observe(fraud_probability)

        # Log prediction
        logger.info(
            f"Prediction: is_fraud={is_fraud}, "
            f"probability={fraud_probability:.4f}, "
            f"amount={transaction.Amount}"
        )

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            threshold=threshold,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        MODEL_ERRORS.inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_fraud_batch(transactions: List[TransactionFeatures]):
    """
    Predict fraud probability for multiple transactions.

    **Input:** List of transaction features

    **Output:** List of fraud predictions
    """
    # Check if model is loaded
    if model is None or scaler is None or threshold is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert inputs to DataFrame
        features_list = [t.dict() for t in transactions]
        X = pd.DataFrame(features_list, columns=feature_names)

        # Scale features
        X_scaled = scaler.transform(X)

        # Preserve feature names
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

        # Get predictions
        fraud_probabilities = model.predict_proba(X_scaled)[:, 1]
        predictions = (fraud_probabilities >= threshold).astype(int)

        # Create responses
        results = []
        for i, (is_fraud, prob) in enumerate(zip(predictions, fraud_probabilities)):
            # Determine confidence
            if prob < 0.3:
                confidence = "high"
            elif prob < 0.7:
                confidence = "medium"
            else:
                confidence = "high"

            results.append(PredictionResponse(
                is_fraud=int(is_fraud),
                fraud_probability=float(prob),
                threshold=threshold,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            ))

        logger.info(f"Batch prediction: {len(transactions)} transactions processed")

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

