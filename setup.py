from setuptools import setup, find_packages

setup(
    name="ai_vs_human_classifier",
    version="1.0.0",
    description="End-to-End MLOps Pipeline for AI vs Human News Classification",
    author="MLOps Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "prometheus-client>=0.17.0",
        "pyyaml>=6.0",
        "mlflow>=2.5.0",
    ],
)
