"""
Data Preprocessing & Feature Engineering Module.
Contains text cleaning, lexical feature extraction, tokenization, and dataset splitting.
"""

import os
import re
import argparse
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger("preprocessing")

def clean_text(text: str) -> str:
    """
    Clean and normalize raw text:
    - Lowercase
    - Remove URLs and hyperlinks
    - Remove HTML tags
    - Remove excessive punctuation and special symbols
    - Strip leading/trailing whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters except common punctuation
    text = re.sub(r"[^\w\s\.\?!,]", "", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_lexical_features(text: str) -> Dict[str, float]:
    """
    Extract linguistic and statistical features from text:
    - Character length
    - Word count
    - Average word length
    - Lexical diversity (Type-Token Ratio / TTR)
    - Punctuation density
    - Uppercase letter ratio
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "char_length": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0,
            "punctuation_count": 0.0,
            "uppercase_ratio": 0.0
        }
    
    char_len = len(text)
    words = re.findall(r"\b\w+\b", text.lower())
    word_cnt = len(words)
    unique_words = len(set(words))
    
    ttr = unique_words / word_cnt if word_cnt > 0 else 0.0
    avg_word_len = np.mean([len(w) for w in words]) if word_cnt > 0 else 0.0
    punct_cnt = len(re.findall(r"[,\.!?]", text))
    upper_cnt = len(re.findall(r"[A-Z]", text))
    upper_ratio = upper_cnt / char_len if char_len > 0 else 0.0
    
    return {
        "char_length": float(char_len),
        "word_count": float(word_cnt),
        "avg_word_length": float(avg_word_len),
        "lexical_diversity": float(ttr),
        "punctuation_count": float(punct_cnt),
        "uppercase_ratio": float(upper_ratio)
    }

def preprocess_pipeline(
    raw_path: str = "data/raw/ai_vs_human_text.csv",
    output_dir: str = "data/processed",
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full preprocessing pipeline on raw data and save train/val/test splits.
    """
    logger.info(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # 1. Handle missing values
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    
    # 2. Text cleaning
    logger.info("Applying text cleaning and normalization...")
    df["clean_text"] = df["text"].apply(clean_text)
    
    # 3. Feature engineering
    logger.info("Extracting lexical & statistical text features...")
    features_df = df["text"].apply(extract_lexical_features).apply(pd.Series)
    for col in features_df.columns:
        df[col] = features_df[col]
    
    # 4. Target encoding: AI-generated -> 1, Human-written -> 0
    label_map = {"AI-generated": 1, "Human-written": 0}
    df["target"] = df["label"].map(label_map).fillna(0).astype(int)
    
    # 5. Stratified train/val/test split
    logger.info(f"Performing stratified split (Train: {train_split}, Val: {val_split}, Test: {test_split})...")
    
    # Split train vs (val + test)
    val_test_size = val_split + test_split
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_size,
        stratify=df["target"],
        random_state=random_state
    )
    
    # Split val vs test
    val_prop = val_split / val_test_size
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=val_prop,
        stratify=val_test_df["target"],
        random_state=random_state
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.info(f"Saved {len(train_df)} train records to {train_path}")
    logger.info(f"Saved {len(val_df)} validation records to {val_path}")
    logger.info(f"Saved {len(test_df)} test records to {test_path}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    preprocess_pipeline(
        raw_path=cfg["data"]["raw_path"],
        output_dir=cfg["data"]["processed_dir"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        random_state=cfg["data"]["random_state"]
    )
