"""
Comprehensive Model Evaluation & Visualization Module by team 49.
Generates:
- Confusion Matrix plots
- ROC Curves comparison
- Detailed Classification Reports
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from src.utils.logger import get_logger
from src.utils.helpers import load_config, save_json
from src.models.baseline_model import BaselineClassifier
from src.models.pytorch_model import Vocabulary, TextDataset, BiLSTMAttentionClassifier

logger = get_logger("model_evaluation")

def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, output_path: str):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")

def plot_roc_curves(curves: list, output_path: str):
    """Plot ROC curves comparison."""
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    
    for name, fpr, tpr, auc in curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        
    plt.title("ROC Curves Comparison: Baseline vs PyTorch", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves comparison to {output_path}")

def evaluate_models(config_path: str = "config/config.yaml"):
    """Evaluate both baseline and PyTorch models on the test set."""
    cfg = load_config(config_path)
    test_path = cfg["data"]["test_path"]
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    logger.info(f"Loading test set from {test_path}...")
    test_df = pd.read_parquet(test_path)
    y_true = test_df["target"].values
    class_names = ["Human-written", "AI-generated"]

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    roc_curves_data = []
    evaluation_summary = {}

    # 1. Evaluate Baseline Model
    base_cfg = cfg["models"]["baseline"]
    if os.path.exists(base_cfg["artifact_path"]) and os.path.exists(base_cfg["tfidf_path"]):
        logger.info("Evaluating Baseline Model...")
        baseline = BaselineClassifier.load(base_cfg["artifact_path"], base_cfg["tfidf_path"])
        base_preds = baseline.predict(test_df["clean_text"].tolist())
        base_probs = baseline.predict_proba(test_df["clean_text"].tolist())[:, 1]

        base_cm = confusion_matrix(y_true, base_preds)
        base_fpr, base_tpr, _ = roc_curve(y_true, base_probs)
        base_auc = roc_auc_score(y_true, base_probs)
        roc_curves_data.append(("Baseline TF-IDF + LR", base_fpr, base_tpr, base_auc))

        plot_confusion_matrix(
            base_cm, class_names,
            "Baseline Model Confusion Matrix",
            os.path.join(reports_dir, "confusion_matrix_baseline.png")
        )

        evaluation_summary["baseline_model"] = {
            "accuracy": float(accuracy_score(y_true, base_preds)),
            "precision": float(precision_score(y_true, base_preds, zero_division=0)),
            "recall": float(recall_score(y_true, base_preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, base_preds, zero_division=0)),
            "roc_auc": float(base_auc),
            "confusion_matrix": base_cm.tolist(),
            "classification_report": classification_report(y_true, base_preds, target_names=class_names, output_dict=True)
        }

    # 2. Evaluate PyTorch Deep Learning Model
    pt_cfg = cfg["models"]["pytorch_bilstm"]
    if os.path.exists(pt_cfg["artifact_path"]) and os.path.exists(pt_cfg["vocab_path"]):
        logger.info("Evaluating PyTorch Model...")
        device = torch.device("cpu")
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
        ).to(device)

        checkpoint = torch.load(pt_cfg["artifact_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_dataset = TextDataset(
            texts=test_df["clean_text"].tolist(),
            lexical_features=test_df[lex_cols].values,
            vocab=vocab,
            max_seq_len=pt_cfg["max_seq_len"]
        )

        test_probs = []
        with torch.no_grad():
            for item in test_dataset:
                tokens = item["tokens"].unsqueeze(0).to(device)
                lexical = item["lexical_features"].unsqueeze(0).to(device)
                prob = model.predict_proba(tokens, lexical).item()
                test_probs.append(prob)

        test_probs = np.array(test_probs)
        pt_preds = (test_probs >= 0.5).astype(int)

        pt_cm = confusion_matrix(y_true, pt_preds)
        pt_fpr, pt_tpr, _ = roc_curve(y_true, test_probs)
        pt_auc = roc_auc_score(y_true, test_probs)
        roc_curves_data.append(("PyTorch BiLSTM + Attention", pt_fpr, pt_tpr, pt_auc))

        plot_confusion_matrix(
            pt_cm, class_names,
            "PyTorch BiLSTM + Attention Confusion Matrix",
            os.path.join(reports_dir, "confusion_matrix_pytorch.png")
        )

        evaluation_summary["pytorch_model"] = {
            "accuracy": float(accuracy_score(y_true, pt_preds)),
            "precision": float(precision_score(y_true, pt_preds, zero_division=0)),
            "recall": float(recall_score(y_true, pt_preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, pt_preds, zero_division=0)),
            "roc_auc": float(pt_auc),
            "confusion_matrix": pt_cm.tolist(),
            "classification_report": classification_report(y_true, pt_preds, target_names=class_names, output_dict=True)
        }

    # Plot ROC curves comparison
    if roc_curves_data:
        plot_roc_curves(roc_curves_data, os.path.join(reports_dir, "roc_curves_comparison.png"))

    report_path = os.path.join(reports_dir, "evaluation_report.json")
    save_json(evaluation_summary, report_path)
    logger.info(f"Model evaluation report saved to {report_path}")
    return evaluation_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    evaluate_models(args.config)
