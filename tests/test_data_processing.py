"""
Unit tests for data ingestion, cleaning, and preprocessing.
"""

import pytest
import pandas as pd
from src.data.ingest import validate_schema
from src.data.preprocess import clean_text, extract_lexical_features

def test_validate_schema_success():
    df = pd.DataFrame({
        "id": [1, 2],
        "text": ["News sample 1", "News sample 2"],
        "label": ["AI-generated", "Human-written"]
    })
    assert validate_schema(df) is True

def test_validate_schema_failure():
    df = pd.DataFrame({"id": [1, 2], "invalid_col": ["a", "b"]})
    with pytest.raises(ValueError):
        validate_schema(df)

def test_clean_text():
    raw_sample = "Check out <p>this</p> article: https://example.com/news! AI is GREAT #tech"
    cleaned = clean_text(raw_sample)
    assert "https" not in cleaned
    assert "<p>" not in cleaned
    assert "#" not in cleaned
    assert "article" in cleaned
    assert cleaned == cleaned.lower()

def test_extract_lexical_features():
    text = "Artificial intelligence and machine learning models are transforming automated content generation."
    features = extract_lexical_features(text)
    
    assert "char_length" in features
    assert "word_count" in features
    assert "avg_word_length" in features
    assert "lexical_diversity" in features
    assert "punctuation_count" in features
    assert "uppercase_ratio" in features
    
    assert features["word_count"] > 5
    assert 0.0 <= features["lexical_diversity"] <= 1.0
