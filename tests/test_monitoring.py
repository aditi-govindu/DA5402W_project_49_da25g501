"""
Unit tests for drift detection and statistical tests.
"""

import numpy as np
import pandas as pd
from src.monitoring.drift_detector import DriftDetector

def test_drift_detector_ks_test():
    # Reference data
    ref_df = pd.DataFrame({
        "char_length": np.random.normal(500, 50, 100),
        "word_count": np.random.normal(80, 10, 100),
        "avg_word_length": np.random.normal(6.2, 0.5, 100),
        "lexical_diversity": np.random.normal(0.65, 0.05, 100),
        "punctuation_count": np.random.normal(10, 2, 100),
        "uppercase_ratio": np.random.normal(0.04, 0.01, 100)
    })

    detector = DriftDetector(reference_data=ref_df)

    # Identical distribution (no drift expected)
    similar_df = pd.DataFrame({
        "char_length": np.random.normal(500, 50, 100),
        "word_count": np.random.normal(80, 10, 100),
        "avg_word_length": np.random.normal(6.2, 0.5, 100),
        "lexical_diversity": np.random.normal(0.65, 0.05, 100),
        "punctuation_count": np.random.normal(10, 2, 100),
        "uppercase_ratio": np.random.normal(0.04, 0.01, 100)
    })

    res = detector.compute_ks_drift(similar_df)
    assert "overall_feature_drift" in res
    assert "feature_details" in res

def test_psi_calculation():
    ref_probs = np.random.uniform(0, 1, 500)
    curr_probs = np.random.uniform(0, 1, 500)

    detector = DriftDetector(reference_data=pd.DataFrame())
    psi = detector.compute_psi(ref_probs, curr_probs)
    assert psi >= 0.0
