"""
Helper utilities for configuration loading, seed setting, and serialization by team 49.
"""

import os
import random
import json
import pickle
from typing import Any, Dict
import yaml

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    if not os.path.exists(config_path):
        # Check relative to repo root
        root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
        if os.path.exists(root_path):
            config_path = root_path
        else:
            raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def save_json(data: Any, file_path: str) -> None:
    """Save dictionary or list as JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Any:
    """Load JSON from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_pickle(obj: Any, file_path: str) -> None:
    """Save object as pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(file_path: str) -> Any:
    """Load object from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
