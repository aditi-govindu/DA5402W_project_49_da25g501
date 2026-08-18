"""
Data processing and ingestion module.
"""
from src.data.ingest import ingest_data
from src.data.preprocess import clean_text, extract_lexical_features, preprocess_pipeline

__all__ = ["ingest_data", "clean_text", "extract_lexical_features", "preprocess_pipeline"]
