"""
FastAPI Model Serving Application by team 49 for MLOps.
Provides RESTful endpoints for real-time inference, batch prediction, Prometheus metrics, and drift monitoring.
"""

import os
import time
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.utils.logger import get_logger
from src.utils.helpers import load_config
from src.data.preprocess import clean_text, extract_lexical_features
from src.models.baseline_model import BaselineClassifier
from src.models.pytorch_model import Vocabulary, BiLSTMAttentionClassifier
from src.monitoring.drift_detector import DriftDetector
from src.api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, DriftCheckRequest, DriftCheckResponse
)

logger = get_logger("api_service")

# Prometheus Metrics Definition
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions broken down by class",
    ["predicted_label"]
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence_score",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

# Global model state
state = {
    "pytorch_model": None,
    "vocab": None,
    "baseline_model": None,
    "drift_detector": None,
    "device": "cpu",
    "active_model": "pytorch",
    "config": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and initialize monitoring on startup."""
    logger.info("Initializing API application and loading model artifacts...")
    try:
        cfg = load_config("config/config.yaml")
        state["config"] = cfg
        state["device"] = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # 1. Load PyTorch model if available
        pt_cfg = cfg["models"]["pytorch_bilstm"]
        if os.path.exists(pt_cfg["artifact_path"]) and os.path.exists(pt_cfg["vocab_path"]):
            logger.info(f"Loading PyTorch model checkpoint from {pt_cfg['artifact_path']}")
            vocab = Vocabulary.load(pt_cfg["vocab_path"])
            lex_cols = ["char_length", "word_count", "avg_word_length", "lexical_diversity", "punctuation_count", "uppercase_ratio"]
            
            model = BiLSTMAttentionClassifier(
                vocab_size=len(vocab),
                embedding_dim=pt_cfg["embedding_dim"],
                hidden_dim=pt_cfg["hidden_dim"],
                num_layers=pt_cfg["num_layers"],
                bidirectional=pt_cfg["bidirectional"],
                dropout=pt_cfg["dropout"],
                num_lexical_features=len(lex_cols)
            ).to(state["device"])

            ckpt = torch.load(pt_cfg["artifact_path"], map_location=state["device"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            state["pytorch_model"] = model
            state["vocab"] = vocab
            state["active_model"] = "pytorch"
            logger.info("PyTorch model loaded successfully.")
        
        # 2. Load Baseline model if available
        base_cfg = cfg["models"]["baseline"]
        if os.path.exists(base_cfg["artifact_path"]) and os.path.exists(base_cfg["tfidf_path"]):
            logger.info(f"Loading baseline model from {base_cfg['artifact_path']}")
            state["baseline_model"] = BaselineClassifier.load(base_cfg["artifact_path"], base_cfg["tfidf_path"])
            if state["pytorch_model"] is None:
                state["active_model"] = "baseline"

        # 3. Load reference data for Drift Detector
        train_path = cfg["data"]["train_path"]
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            state["drift_detector"] = DriftDetector(reference_data=train_df)
            logger.info("Drift detector initialized with reference training dataset.")

    except Exception as e:
        logger.error(f"Error during API startup: {e}", exc_info=True)

    yield
    logger.info("Shutting down API service...")

app = FastAPI(
    title="AI vs Human News Classifier API",
    description="Production-ready MLOps API serving PyTorch Deep Learning & Baseline models with Prometheus monitoring & drift tracking.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _predict_single(text: str) -> PredictResponse:
    """Internal helper to classify a single text."""
    start_time = time.time()
    
    cleaned = clean_text(text)
    lex_features = extract_lexical_features(text)
    
    if state["active_model"] == "pytorch" and state["pytorch_model"] is not None:
        vocab = state["vocab"]
        max_seq_len = state["config"]["models"]["pytorch_bilstm"]["max_seq_len"]
        tokens = torch.tensor([vocab.encode(cleaned, max_len=max_seq_len)], dtype=torch.long).to(state["device"])
        
        lex_array = [
            lex_features["char_length"],
            lex_features["word_count"],
            lex_features["avg_word_length"],
            lex_features["lexical_diversity"],
            lex_features["punctuation_count"],
            lex_features["uppercase_ratio"]
        ]
        lexical_tensor = torch.tensor([lex_array], dtype=torch.float32).to(state["device"])
        
        prob_ai = float(state["pytorch_model"].predict_proba(tokens, lexical_tensor).item())
        model_name = "PyTorch_BiLSTM_Attention"
    elif state["baseline_model"] is not None:
        probs = state["baseline_model"].predict_proba([cleaned])[0]
        prob_ai = float(probs[1])
        model_name = "Baseline_TFIDF_LogisticRegression"
    else:
        # Fallback heuristic if models haven't been trained yet
        lex_div = lex_features["lexical_diversity"]
        prob_ai = 0.70 if lex_div < 0.60 else 0.30
        model_name = "Heuristic_Fallback"

    label = "AI-generated" if prob_ai >= 0.5 else "Human-written"
    confidence = prob_ai if prob_ai >= 0.5 else (1.0 - prob_ai)
    latency_ms = (time.time() - start_time) * 1000.0

    # Prometheus metrics update
    PREDICTION_COUNT.labels(predicted_label=label).inc()
    PREDICTION_CONFIDENCE.observe(confidence)

    return PredictResponse(
        text_snippet=text[:100] + ("..." if len(text) > 100 else ""),
        predicted_label=label,
        confidence_score=round(confidence, 4),
        probability_ai=round(prob_ai, 4),
        latency_ms=round(latency_ms, 2),
        lexical_features=lex_features,
        model_type=model_name,
        model_version="1.0.0"
    )

@app.get("/", summary="Root Index")
def root():
    """Root endpoint returning API details."""
    return {
        "project": "AI vs Human News Classifier MLOps Pipeline",
        "status": "online",
        "docs_url": "/docs",
        "metrics_url": "/metrics",
        "active_model": state.get("active_model", "none")
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Health check endpoint."""
    model_loaded = (state["pytorch_model"] is not None) or (state["baseline_model"] is not None)
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=state.get("active_model", "none"),
        version="1.0.0",
        device=str(state.get("device", "cpu"))
    )

@app.post("/predict", response_model=PredictResponse, summary="Classify Single Text")
def predict(payload: PredictRequest):
    """Classify a single news/text article."""
    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        response = _predict_single(payload.text)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code=200).inc()
        return response

@app.post("/batch-predict", response_model=BatchPredictResponse, summary="Classify Batch of Texts")
def batch_predict(payload: BatchPredictRequest):
    """Classify a batch of news/text articles."""
    with REQUEST_LATENCY.labels(endpoint="/batch-predict").time():
        start_time = time.time()
        predictions = [_predict_single(text) for text in payload.texts]
        total_time_ms = (time.time() - start_time) * 1000.0
        avg_latency = total_time_ms / len(payload.texts) if payload.texts else 0.0

        REQUEST_COUNT.labels(method="POST", endpoint="/batch-predict", status_code=200).inc()
        return BatchPredictResponse(
            total_processed=len(predictions),
            average_latency_ms=round(avg_latency, 2),
            predictions=predictions
        )

@app.post("/drift-check", response_model=DriftCheckResponse, summary="Run On-demand Drift Check")
def check_drift(payload: DriftCheckRequest):
    """Compute statistical feature and prediction drift on incoming texts."""
    detector = state.get("drift_detector")
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized. Reference data missing.")

    report = detector.check_drift_from_texts(payload.texts)
    return DriftCheckResponse(
        num_samples=report["num_incoming_samples"],
        overall_drift_detected=report["overall_drift_detected"],
        feature_drift_summary=report["feature_drift_summary"]
    )

@app.get("/metrics", summary="Prometheus Metrics")
def metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
