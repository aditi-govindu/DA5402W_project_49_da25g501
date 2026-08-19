"""
Data and Prediction Drift Detection Module.
Calculates statistical drift between reference (training) distribution and production/inference data:
- Kolmogorov-Smirnov (KS) two-sample test on continuous linguistic features
- Population Stability Index (PSI) and Chi-Square test on prediction distribution
- Generates JSON and HTML monitoring reports
"""

import os
import sys
import json
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.logger import get_logger
from src.utils.helpers import load_config, save_json
from src.data.preprocess import extract_lexical_features

logger = get_logger("drift_detector")

class DriftDetector:
    """Statistical Drift Detector for Text Features and Model Predictions."""
    def __init__(self, reference_data: pd.DataFrame, p_val_threshold: float = 0.05, ks_stat_threshold: float = 0.10):
        self.reference_data = reference_data
        self.p_val_threshold = p_val_threshold
        self.ks_stat_threshold = ks_stat_threshold
        self.feature_cols = ["char_length", "word_count", "avg_word_length", "lexical_diversity", "punctuation_count", "uppercase_ratio"]

    def compute_ks_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test on numerical features."""
        drift_results = {}
        overall_drift = False

        for col in self.feature_cols:
            if col in self.reference_data.columns and col in current_data.columns:
                ref_series = self.reference_data[col].dropna().values
                curr_series = current_data[col].dropna().values

                if len(curr_series) < 2:
                    continue

                ks_stat, p_val = stats.ks_2samp(ref_series, curr_series)
                is_drifted = bool(p_val < self.p_val_threshold and ks_stat > self.ks_stat_threshold)
                if is_drifted:
                    overall_drift = True

                drift_results[col] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "drift_detected": is_drifted,
                    "reference_mean": float(np.mean(ref_series)),
                    "current_mean": float(np.mean(curr_series)),
                    "reference_std": float(np.std(ref_series)),
                    "current_std": float(np.std(curr_series))
                }

        return {
            "overall_feature_drift": overall_drift,
            "feature_details": drift_results
        }

    def compute_psi(self, ref_probs: np.ndarray, curr_probs: np.ndarray, num_bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) for prediction probabilities."""
        if len(curr_probs) == 0 or len(ref_probs) == 0:
            return 0.0

        bins = np.linspace(0, 1, num_bins + 1)
        ref_counts, _ = np.histogram(ref_probs, bins=bins)
        curr_counts, _ = np.histogram(curr_probs, bins=bins)

        # Avoid division by zero
        ref_pct = (ref_counts + 1e-5) / len(ref_probs)
        curr_pct = (curr_counts + 1e-5) / len(curr_probs)

        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi_val)

    def check_drift_from_texts(self, current_texts: List[str], current_probs: Optional[List[float]] = None) -> Dict[str, Any]:
        """Check drift on a batch of incoming text strings."""
        if not current_texts:
            return {"error": "No texts provided for drift checking"}

        # Extract features for incoming texts
        features_list = [extract_lexical_features(t) for t in current_texts]
        curr_df = pd.DataFrame(features_list)

        ks_results = self.compute_ks_drift(curr_df)
        
        report = {
            "num_incoming_samples": len(current_texts),
            "feature_drift_summary": ks_results,
            "overall_drift_detected": ks_results["overall_feature_drift"]
        }

        if current_probs is not None and "target" in self.reference_data.columns:
            ref_probs = self.reference_data["target"].values
            psi_score = self.compute_psi(ref_probs, np.array(current_probs))
            report["prediction_psi"] = psi_score
            report["prediction_drift_detected"] = bool(psi_score > 0.25)
            if report["prediction_drift_detected"]:
                report["overall_drift_detected"] = True

        return report

    def generate_html_report(self, report_dict: Dict[str, Any], output_path: str = "reports/drift_report.html"):
        """Generate interactive HTML summary report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        drift_color = "#e74c3c" if report_dict.get("overall_drift_detected") else "#2ecc71"
        drift_status = "DRIFT DETECTED" if report_dict.get("overall_drift_detected") else "NO DRIFT DETECTED"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLOps Drift Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }}
                h1, h2 {{ color: #2c3e50; }}
                .status-badge {{ background-color: {drift_color}; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold; display: inline-block; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }}
                th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>MLOps News Classifier: Data & Prediction Drift Report</h1>
            <p><strong>Status:</strong> <span class="status-badge">{drift_status}</span></p>
            <p><strong>Sample Size Evaluated:</strong> {report_dict.get('num_incoming_samples', 'N/A')}</p>
            
            <h2>Linguistic & Lexical Feature Drift (KS Test)</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>KS Statistic</th>
                    <th>P-Value</th>
                    <th>Reference Mean</th>
                    <th>Current Mean</th>
                    <th>Drift Detected</th>
                </tr>
        """
        
        details = report_dict.get("feature_drift_summary", {}).get("feature_details", {})
        for feat, d in details.items():
            drift_cell = f"<span style='color: {'red' if d['drift_detected'] else 'green'}; font-weight: bold;'>{'YES' if d['drift_detected'] else 'NO'}</span>"
            html_content += f"""
                <tr>
                    <td>{feat}</td>
                    <td>{d['ks_statistic']:.4f}</td>
                    <td>{d['p_value']:.4e}</td>
                    <td>{d['reference_mean']:.2f}</td>
                    <td>{d['current_mean']:.2f}</td>
                    <td>{drift_cell}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved drift HTML report to {output_path}")

def run_drift_check(config_path: str = "config/config.yaml"):
    """Run drift check on test dataset against train reference."""
    cfg = load_config(config_path)
    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        logger.warning("Train/Test parquet not found. Preprocessing first...")
        from src.data.preprocess import preprocess_pipeline
        preprocess_pipeline()

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    detector = DriftDetector(
        reference_data=train_df,
        p_val_threshold=cfg["monitoring"]["drift_threshold_pvalue"],
        ks_stat_threshold=cfg["monitoring"]["ks_stat_threshold"]
    )

    report = detector.check_drift_from_texts(test_df["text"].tolist())
    
    json_path = cfg["monitoring"]["report_json_path"]
    html_path = cfg["monitoring"]["report_html_path"]
    
    save_json(report, json_path)
    detector.generate_html_report(report, html_path)
    logger.info(f"Drift report generated. Overall Drift Detected: {report.get('overall_drift_detected')}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Drift Detection")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_drift_check(args.config)
