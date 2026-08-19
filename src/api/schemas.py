"""
Pydantic Data Schemas for FastAPI Endpoints for team 49.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    """Single text classification request."""
    text: str = Field(
        ...,
        min_length=5,
        example="Artificial intelligence models have shown remarkable advancements in natural language understanding."
    )

class LexicalFeatures(BaseModel):
    """Heuristic features extracted from the text provided."""
    char_length: float
    word_count: float
    avg_word_length: float
    lexical_diversity: float
    punctuation_count: float
    uppercase_ratio: float

class PredictResponse(BaseModel):
    """Single prediction response payload."""
    text_snippet: str
    predicted_label: str
    confidence_score: float
    probability_ai: float
    latency_ms: float
    lexical_features: Dict[str, float]
    model_type: str
    model_version: str

class BatchPredictRequest(BaseModel):
    """Batch prediction request payload."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        example=[
            "This is the first article written by human journalists.",
            "Deep learning enables language models to generate coherent text paragraphs."
        ]
    )

class BatchPredictResponse(BaseModel):
    """Batch prediction response payload."""
    total_processed: int
    average_latency_ms: float
    predictions: List[PredictResponse]

class HealthResponse(BaseModel):
    """Service health response."""
    status: str
    model_loaded: bool
    model_type: str
    version: str
    device: str

class DriftCheckRequest(BaseModel):
    """Drift check request."""
    texts: List[str] = Field(..., min_items=2)

class DriftCheckResponse(BaseModel):
    """Drift check response."""
    num_samples: int
    overall_drift_detected: bool
    feature_drift_summary: Dict[str, Any]
