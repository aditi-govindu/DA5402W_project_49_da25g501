"""
Data Ingestion and Schema Validation Module.
Loads raw AI vs Human news article dataset, validates schema, and reports statistics.
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger("data_ingestion")

REQUIRED_COLUMNS = ["id", "text", "label"]

def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that required columns exist in dataframe."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")
    return True

def ingest_data(raw_path: str = "data/raw/ai_vs_human_text.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingest raw dataset from CSV, validate schema, remove invalid entries, and log summary stats.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
    
    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    initial_rows = len(df)
    validate_schema(df)
    
    # Handle missing/empty text or labels
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    
    cleaned_rows = len(df)
    dropped_rows = initial_rows - cleaned_rows
    
    label_distribution = df["label"].value_counts().to_dict()
    
    stats = {
        "initial_rows": initial_rows,
        "valid_rows": cleaned_rows,
        "dropped_rows": dropped_rows,
        "columns": df.columns.tolist(),
        "label_distribution": label_distribution,
    }
    
    logger.info(f"Ingestion summary: {cleaned_rows} valid records loaded. Dropped {dropped_rows} invalid rows.")
    logger.info(f"Label distribution: {label_distribution}")
    
    return df, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and validate raw dataset.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    raw_path = config["data"]["raw_path"]
    df, stats = ingest_data(raw_path)
    logger.info("Data ingestion completed successfully.")
