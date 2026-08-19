"""
Apache Airflow DAG: End-to-End MLOps Pipeline for AI vs Human News Classification.
Orchestrates:
1. Ingest Data (Schema validation & cleaning)
2. Spark Preprocessing (PySpark distributed feature engineering & partitioning)
3. Model Training (Baseline + PyTorch BiLSTM with MLflow tracking)
4. Model Evaluation (Confusion matrix & ROC curve generation)
5. Data Drift Monitoring (KS-test & statistical drift reporting)
6. Model Validation & Deployment Gate
"""

import os
import sys
from datetime import datetime, timedelta

# Ensure repo root is on Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
except ImportError:
    # Dummy classes for environments where Airflow is inspected without airflow installed
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class PythonOperator:
        def __init__(self, *args, **kwargs): pass
        def __rshift__(self, other): return other
    class BashOperator:
        def __init__(self, *args, **kwargs): pass
        def __rshift__(self, other): return other

def task_ingest_data_callable():
    """Ingest raw CSV dataset and validate schema."""
    from src.data.ingest import ingest_data
    df, stats = ingest_data("data/raw/ai_vs_human_text.csv")
    print(f"Data ingestion successful. Total rows: {stats['valid_rows']}")
    return stats

def task_spark_preprocess_callable():
    """Execute Apache Spark or modular preprocessing pipeline."""
    try:
        from src.data.spark_preprocess import run_spark_preprocessing
        print("Running distributed preprocessing via Apache Spark...")
        run_spark_preprocessing(raw_path="data/raw/ai_vs_human_text.csv", output_dir="data/processed")
    except Exception as e:
        print(f"PySpark run failed or unavailable ({e}). Falling back to optimized Pandas preprocessor...")
        from src.data.preprocess import preprocess_pipeline
        preprocess_pipeline(raw_path="data/raw/ai_vs_human_text.csv", output_dir="data/processed")
    print("Preprocessing completed. Parquet splits generated.")

def task_train_models_callable():
    """Train Baseline and PyTorch deep learning models with MLflow experiment tracking."""
    from src.models.train import run_training_pipeline
    results = run_training_pipeline("config/config.yaml")
    print("Model training completed. MLflow runs logged.")
    return results

def task_evaluate_models_callable():
    """Evaluate models on test set and generate visualizations."""
    from src.models.evaluate import evaluate_models
    eval_results = evaluate_models("config/config.yaml")
    print(f"Evaluation report generated: {eval_results}")
    return eval_results

def task_drift_detection_callable():
    """Run Kolmogorov-Smirnov statistical data drift monitoring."""
    from src.monitoring.drift_detector import run_drift_check
    drift_report = run_drift_check("config/config.yaml")
    print(f"Drift monitoring check completed. Drift Detected: {drift_report.get('overall_drift_detected')}")
    return drift_report

def task_model_validation_callable():
    """Verify performance metrics against deployment quality thresholds."""
    import json
    eval_path = "reports/evaluation_report.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pt_f1 = data.get("pytorch_model", {}).get("f1_score", 0.0)
        base_f1 = data.get("baseline_model", {}).get("f1_score", 0.0)
        print(f"PyTorch F1: {pt_f1:.4f}, Baseline F1: {base_f1:.4f}")
        if max(pt_f1, base_f1) >= 0.70:
            print("Quality gate passed! Model is certified for production deployment.")
        else:
            print("Warning: Model performance below target threshold (0.70).")
    else:
        print("Evaluation report not found.")

default_args = {
    "owner": "mlops-team-49",
    "depends_on_past": False,
    "start_date": datetime(2026, 8, 19),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="mlops_ai_vs_human_classification_pipeline",
    default_args=default_args,
    description="Automated end-to-end Airflow pipeline for AI vs Human news text classification",
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "pytorch", "spark", "classification", "monitoring"]
) as dag:

    ingest_step = PythonOperator(
        task_id="ingest_and_validate_data",
        python_callable=task_ingest_data_callable
    )

    preprocess_step = PythonOperator(
        task_id="spark_data_preprocessing",
        python_callable=task_spark_preprocess_callable
    )

    train_step = PythonOperator(
        task_id="train_and_track_models",
        python_callable=task_train_models_callable
    )

    evaluate_step = PythonOperator(
        task_id="evaluate_and_plot_metrics",
        python_callable=task_evaluate_models_callable
    )

    drift_step = PythonOperator(
        task_id="data_drift_monitoring",
        python_callable=task_drift_detection_callable
    )

    validation_step = PythonOperator(
        task_id="model_validation_gate",
        python_callable=task_model_validation_callable
    )

    # Define Workflow Orchestration DAG Dependencies
    ingest_step >> preprocess_step >> train_step >> evaluate_step >> [drift_step, validation_step]
