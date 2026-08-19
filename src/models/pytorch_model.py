"""
PyTorch Deep Learning Model: Bidirectional LSTM with Self-Attention & Lexical Features.
Provides custom vocabulary management, dataset pipeline, and deep neural architecture.
"""

import os
import re
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils.logger import get_logger

logger = get_logger("pytorch_model")

class Vocabulary:
    """Vocabulary mapping tokens to indices with special tokens handling for team 49."""
    PAD_TOKEN = "<PAD>"  # idx 0 - padding
    UNK_TOKEN = "<UNK>"  # idx 1 - unknown token
    SOS_TOKEN = "<SOS>"  # idx 2 - start of stream
    EOS_TOKEN = "<EOS>"  # idx 3 - end of stream

    def __init__(self, max_size: int = 10000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.w2i = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.i2w = {idx: word for word, idx in self.w2i.items()}

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace and regex tokenizer."""
        if not isinstance(text, str):
            return []
        text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of text strings. Limit is 10000."""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Filter by min_freq and max_size
        sorted_tokens = [word for word, count in counter.most_common() if count >= self.min_freq]
        for word in sorted_tokens:
            if len(self.w2i) >= self.max_size:
                break
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

        logger.info(f"Built vocabulary with {len(self.w2i)} unique tokens.")

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        """Convert text string to padded/truncated list of token indices."""
        tokens = self.tokenize(text)
        indices = [self.w2i.get(token, self.w2i[self.UNK_TOKEN]) for token in tokens]
        
        # Truncate if longer than max_len
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            # Pad with PAD_TOKEN (0)
            indices = indices + [self.w2i[self.PAD_TOKEN]] * (max_len - len(indices))
        return indices

    def save(self, file_path: str):
        """Save vocabulary to JSON."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"w2i": self.w2i, "max_size": self.max_size, "min_freq": self.min_freq}, f, indent=2)
        logger.info(f"Saved vocabulary to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """Load vocabulary from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_size=data.get("max_size", 10000), min_freq=data.get("min_freq", 2))
        vocab.w2i = data["w2i"]
        vocab.i2w = {int(v) if isinstance(v, (int, str)) and str(v).isdigit() else v: k for k, v in vocab.w2i.items()}
        # Ensure integer keys
        vocab.i2w = {int(v): k for k, v in vocab.w2i.items()}
        return vocab

    def __len__(self):
        return len(self.w2i)


class TextDataset(Dataset):
    """PyTorch Dataset for news articles classification."""
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        lexical_features: Optional[np.ndarray] = None,
        vocab: Optional[Vocabulary] = None,
        max_seq_len: int = 256
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.lexical_features = lexical_features
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        tokens = self.vocab.encode(text, max_len=self.max_seq_len)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        item = {"tokens": tokens_tensor}

        if self.lexical_features is not None:
            lex_feat = self.lexical_features[idx]
            item["lexical_features"] = torch.tensor(lex_feat, dtype=torch.float32)
        else:
            item["lexical_features"] = torch.zeros(6, dtype=torch.float32)

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


class SelfAttention(nn.Module):
    """Additive Attention Mechanism over LSTM sequence outputs."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: [batch_size, seq_len, hidden_dim]
        energy = self.projection(lstm_output) # [batch_size, seq_len, 1]
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        weights = F.softmax(energy, dim=1) # [batch_size, seq_len, 1]
        context = torch.sum(lstm_output * weights, dim=1) # [batch_size, hidden_dim]
        return context, weights.squeeze(-1)


class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM with Self-Attention and Lexical Feature fusion.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_lexical_features: int = 6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Token Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = SelfAttention(lstm_out_dim)

        # Lexical features projection
        self.lexical_fc = nn.Sequential(
            nn.Linear(num_lexical_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification Head
        combined_dim = lstm_out_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, tokens: torch.Tensor, lexical_features: torch.Tensor) -> torch.Tensor:
        # tokens: [batch_size, seq_len]
        mask = (tokens != 0).long()
        embedded = self.embedding(tokens) # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded) # [batch_size, seq_len, lstm_out_dim]
        
        context, _ = self.attention(lstm_out, mask=mask) # [batch_size, lstm_out_dim]
        lex_embed = self.lexical_fc(lexical_features) # [batch_size, 32]
        
        combined = torch.cat([context, lex_embed], dim=1) # [batch_size, combined_dim]
        logits = self.classifier(combined).squeeze(-1) # [batch_size]
        return logits

    def predict_proba(self, tokens: torch.Tensor, lexical_features: torch.Tensor) -> torch.Tensor:
        """Inference method returning probability of AI-generated class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(tokens, lexical_features)
            probs = torch.sigmoid(logits)
        return probs
