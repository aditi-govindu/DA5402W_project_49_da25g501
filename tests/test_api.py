"""
Integration and functional tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_predict_endpoint():
    payload = {
        "text": "Artificial intelligence and machine learning models are transforming text generation across the globe."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert data["predicted_label"] in ["AI-generated", "Human-written"]
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert "lexical_features" in data
    assert "latency_ms" in data

def test_batch_predict_endpoint():
    payload = {
        "texts": [
            "This is an article written by a human reporter.",
            "Generative AI algorithms synthesize text from vast corpora."
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
