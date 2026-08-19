"""
Baseline ML Model: TF-IDF + Logistic Regression.
Serves as an interpretable statistical baseline for comparison against PyTorch deep learning.
"""

import os
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger
from src.utils.helpers import load_config, save_pickle, load_pickle

logger = get_logger("baseline_model")

class BaselineClassifier:
    """Baseline Text Classification Pipeline."""
    def __init__(self, model_type: str = "logistic_regression", max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        if model_type == "logistic_regression":
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "naive_bayes":
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}")

    def fit(self, texts: list, y: np.ndarray):
        """Fit TF-IDF vectorizer and classifier."""
        logger.info(f"Fitting TF-IDF Vectorizer and {self.model_type} model on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)
        return self

    def predict(self, texts: list) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list) -> np.ndarray:
        """Predict probabilities [P(Human), P(AI)]."""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts: list, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate baseline model on given dataset."""
        y_pred = self.predict(texts)
        y_prob = self.predict_proba(texts)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob))
        }
        return metrics

    def save(self, model_path: str, vectorizer_path: str):
        """Save vectorizer and model to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_pickle(self.model, model_path)
        save_pickle(self.vectorizer, vectorizer_path)
        logger.info(f"Saved baseline model to {model_path} and vectorizer to {vectorizer_path}")

    @classmethod
    def load(cls, model_path: str, vectorizer_path: str):
        """Load trained baseline model and vectorizer."""
        instance = cls()
        instance.model = load_pickle(model_path)
        instance.vectorizer = load_pickle(vectorizer_path)
        return instance
