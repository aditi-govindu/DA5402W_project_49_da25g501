"""
Model Training & MLflow Experiment Tracking Orchestrator for team 49.
Trains and compares:
1. Baseline Statistical Model (TF-IDF + Logistic Regression)
2. PyTorch Deep Learning Model (BiLSTM + Self-Attention)
Logs metrics, parameters, and model artifacts to MLflow.
"""

import os
import sys
import argparse
import copy

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.logger import get_logger
from src.utils.helpers import load_config, set_seed, save_json
from src.data.preprocess import preprocess_pipeline
from src.models.baseline_model import BaselineClassifier
from src.models.pytorch_model import Vocabulary, TextDataset, BiLSTMAttentionClassifier

logger = get_logger("training_pipeline")

def train_baseline_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> dict:
    """Train and evaluate baseline TF-IDF Logistic Regression model with MLflow tracking."""
    logger.info("--- Starting Baseline Model Training ---")
    base_cfg = cfg["models"]["baseline"]
    
    with mlflow.start_run(run_name=base_cfg["name"]):
        # Log parameters
        mlflow.log_param("model_type", base_cfg["name"])
        mlflow.log_param("max_iter", base_cfg["max_iter"])
        mlflow.log_param("c_param", base_cfg["c_param"])
        mlflow.log_param("tfidf_max_features", base_cfg["tfidf_max_features"])
        mlflow.log_param("ngram_range", str(base_cfg["ngram_range"]))
        
        # Train baseline
        clf = BaselineClassifier(
            model_type="logistic_regression",
            max_features=base_cfg["tfidf_max_features"],
            ngram_range=tuple(base_cfg["ngram_range"])
        )
        clf.fit(train_df["clean_text"].tolist(), train_df["target"].values)
        
        # Evaluate on validation set
        val_metrics = clf.evaluate(val_df["clean_text"].tolist(), val_df["target"].values)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        
        # Evaluate on test set
        test_metrics = clf.evaluate(test_df["clean_text"].tolist(), test_df["target"].values)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
            
        logger.info(f"Baseline Validation Metrics: {val_metrics}")
        logger.info(f"Baseline Test Metrics: {test_metrics}")
        
        # Save model and log artifacts
        os.makedirs("models", exist_ok=True)
        clf.save(base_cfg["artifact_path"], base_cfg["tfidf_path"])
        mlflow.log_artifact(base_cfg["artifact_path"])
        mlflow.log_artifact(base_cfg["tfidf_path"])
        
        return {
            "model": "baseline_tfidf_logistic_regression",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

def train_pytorch_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> dict:
    """Train and evaluate PyTorch BiLSTM + Attention deep learning model with MLflow tracking."""
    logger.info("--- Starting PyTorch Deep Learning Model Training ---")
    pt_cfg = cfg["models"]["pytorch_bilstm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    lex_cols = ["char_length", "word_count", "avg_word_length", "lexical_diversity", "punctuation_count", "uppercase_ratio"]

    # 1. Build Vocabulary from training text
    vocab = Vocabulary(max_size=pt_cfg["vocab_size"], min_freq=2)
    vocab.build_vocab(train_df["clean_text"].tolist())
    vocab.save(pt_cfg["vocab_path"])

    # 2. Datasets & DataLoaders
    train_dataset = TextDataset(
        texts=train_df["clean_text"].tolist(),
        labels=train_df["target"].values,
        lexical_features=train_df[lex_cols].values,
        vocab=vocab,
        max_seq_len=pt_cfg["max_seq_len"]
    )
    val_dataset = TextDataset(
        texts=val_df["clean_text"].tolist(),
        labels=val_df["target"].values,
        lexical_features=val_df[lex_cols].values,
        vocab=vocab,
        max_seq_len=pt_cfg["max_seq_len"]
    )
    test_dataset = TextDataset(
        texts=test_df["clean_text"].tolist(),
        labels=test_df["target"].values,
        lexical_features=test_df[lex_cols].values,
        vocab=vocab,
        max_seq_len=pt_cfg["max_seq_len"]
    )

    batch_size = pt_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=pt_cfg["embedding_dim"],
        hidden_dim=pt_cfg["hidden_dim"],
        num_layers=pt_cfg["num_layers"],
        bidirectional=pt_cfg["bidirectional"],
        dropout=pt_cfg["dropout"],
        num_lexical_features=len(lex_cols)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=pt_cfg["learning_rate"], weight_decay=pt_cfg["weight_decay"])

    best_val_f1 = -1.0
    best_model_state = None

    with mlflow.start_run(run_name=pt_cfg["name"]):
        # Log parameters
        mlflow.log_params({
            "model_type": pt_cfg["name"],
            "vocab_size": len(vocab),
            "max_seq_len": pt_cfg["max_seq_len"],
            "embedding_dim": pt_cfg["embedding_dim"],
            "hidden_dim": pt_cfg["hidden_dim"],
            "num_layers": pt_cfg["num_layers"],
            "bidirectional": pt_cfg["bidirectional"],
            "dropout": pt_cfg["dropout"],
            "batch_size": batch_size,
            "learning_rate": pt_cfg["learning_rate"],
            "epochs": pt_cfg["epochs"],
            "device": str(device)
        })

        epochs = pt_cfg["epochs"]
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                tokens = batch["tokens"].to(device)
                lexical = batch["lexical_features"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(tokens, lexical)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(labels)

            train_loss = train_loss / len(train_dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_probs, val_trues = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch["tokens"].to(device)
                    lexical = batch["lexical_features"].to(device)
                    labels = batch["label"].to(device)

                    logits = model(tokens, lexical)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)

                    val_probs.extend(probs)
                    val_preds.extend(preds)
                    val_trues.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_dataset)
            val_acc = accuracy_score(val_trues, val_preds)
            val_prec = precision_score(val_trues, val_preds, zero_division=0)
            val_rec = recall_score(val_trues, val_preds, zero_division=0)
            val_f1 = f1_score(val_trues, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_trues, val_probs)

            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1_score": val_f1,
                "val_roc_auc": val_auc
            }, step=epoch)

            logger.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # state_dict tensors reference the live model parameters; copy them so
                # later training epochs cannot overwrite this best checkpoint.
                best_model_state = copy.deepcopy(model.state_dict())

        # Load best model for test evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate on Test Set
        model.eval()
        test_preds, test_probs, test_trues = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                tokens = batch["tokens"].to(device)
                lexical = batch["lexical_features"].to(device)
                labels = batch["label"].to(device)

                logits = model(tokens, lexical)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                test_probs.extend(probs)
                test_preds.extend(preds)
                test_trues.extend(labels.cpu().numpy())

        test_metrics = {
            "accuracy": float(accuracy_score(test_trues, test_preds)),
            "precision": float(precision_score(test_trues, test_preds, zero_division=0)),
            "recall": float(recall_score(test_trues, test_preds, zero_division=0)),
            "f1_score": float(f1_score(test_trues, test_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_trues, test_probs))
        }

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info(f"PyTorch Test Metrics: {test_metrics}")

        # Save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": pt_cfg,
            "vocab_size": len(vocab),
            "test_metrics": test_metrics
        }
        torch.save(checkpoint, pt_cfg["artifact_path"])
        mlflow.log_artifact(pt_cfg["artifact_path"])
        mlflow.log_artifact(pt_cfg["vocab_path"])

        return {
            "model": "pytorch_bilstm_attention",
            "val_f1": best_val_f1,
            "test_metrics": test_metrics
        }

def run_training_pipeline(config_path: str = "config/config.yaml"):
    """Execute complete training pipeline and compare models."""
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_state"])

    # MLflow Setup
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Check or load processed data
    proc_dir = cfg["data"]["processed_dir"]
    train_path = os.path.join(proc_dir, "train.parquet")
    val_path = os.path.join(proc_dir, "val.parquet")
    test_path = os.path.join(proc_dir, "test.parquet")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        logger.info("Processed datasets not found. Running preprocessing pipeline...")
        train_df, val_df, test_df = preprocess_pipeline(
            raw_path=cfg["data"]["raw_path"],
            output_dir=proc_dir,
            train_split=cfg["data"]["train_split"],
            val_split=cfg["data"]["val_split"],
            test_split=cfg["data"]["test_split"],
            random_state=cfg["data"]["random_state"]
        )
    else:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

    # 1. Baseline Model
    baseline_results = train_baseline_model(train_df, val_df, test_df, cfg)

    # 2. PyTorch Deep Learning Model
    pytorch_results = train_pytorch_model(train_df, val_df, test_df, cfg)

    # Comparison summary
    comparison = {
        "baseline_model": baseline_results,
        "pytorch_model": pytorch_results
    }
    os.makedirs("reports", exist_ok=True)
    save_json(comparison, "reports/model_comparison.json")
    logger.info("Model training and comparison completed successfully.")
    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Training & MLflow Tracking")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_training_pipeline(args.config)
