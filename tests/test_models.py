"""
Unit tests for PyTorch deep learning model and Baseline classifier.
"""

import os
import pytest
import numpy as np
import torch
from src.models.pytorch_model import Vocabulary, TextDataset, BiLSTMAttentionClassifier
from src.models.baseline_model import BaselineClassifier

def test_vocabulary_build_and_encode():
    texts = [
        "Machine learning models process natural language.",
        "Deep learning and neural networks are powerful."
    ]
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.build_vocab(texts)
    
    assert len(vocab) > 4
    assert vocab.PAD_TOKEN in vocab.w2i
    assert vocab.UNK_TOKEN in vocab.w2i
    
    encoded = vocab.encode("neural networks learning", max_len=10)
    assert len(encoded) == 10
    assert encoded[0] != vocab.w2i[vocab.PAD_TOKEN]

def test_pytorch_bilstm_forward():
    vocab_size = 50
    batch_size = 4
    seq_len = 16
    num_lexical = 6

    model = BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True,
        num_lexical_features=num_lexical
    )

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    lexical = torch.randn(batch_size, num_lexical)

    logits = model(tokens, lexical)
    assert logits.shape == (batch_size,)

    probs = model.predict_proba(tokens, lexical)
    assert probs.shape == (batch_size,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_baseline_classifier_fit_predict():
    texts = [
        "This is an AI generated response with structured format.",
        "Today I went to the park and met my friends.",
        "ChatGPT generates coherent text based on transformer architecture.",
        "The personal journal entry reflects genuine human sentiments."
    ]
    labels = np.array([1, 0, 1, 0])

    clf = BaselineClassifier(model_type="logistic_regression", max_features=100)
    clf.fit(texts, labels)

    preds = clf.predict(texts)
    assert len(preds) == len(labels)
    
    probs = clf.predict_proba(texts)
    assert probs.shape == (len(labels), 2)
