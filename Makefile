export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.PHONY: help install ingest preprocess train evaluate drift test lint docker-build docker-up docker-down airflow mlflow clean

help:
	@echo "MLOps AI vs Human News Classification Pipeline"
	@echo "Available commands:"
	@echo "  make install        : Install Python dependencies"
	@echo "  make ingest         : Ingest raw dataset & validate schema"
	@echo "  make preprocess     : Run feature engineering & preprocessing"
	@echo "  make spark-prep     : Run Apache Spark preprocessing"
	@echo "  make train          : Train Baseline & PyTorch models with MLflow"
	@echo "  make evaluate       : Evaluate models and generate confusion/ROC plots"
	@echo "  make drift          : Run data and prediction drift detection"
	@echo "  make pipeline       : Run full end-to-end pipeline (ingest->prep->train->eval->drift)"
	@echo "  make api            : Start FastAPI serving server locally"
	@echo "  make test           : Run automated test suite with coverage"
	@echo "  make lint           : Run code quality checks (flake8, black)"
	@echo "  make docker-build   : Build Docker image for FastAPI service"
	@echo "  make docker-up      : Start all services (API, Airflow, MLflow, Prometheus, Grafana)"
	@echo "  make docker-down    : Stop all docker-compose services"
	@echo "  make clean          : Clean temporary files and caches"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

ingest:
	python3 src/data/ingest.py --config config/config.yaml

preprocess:
	python3 src/data/preprocess.py --config config/config.yaml

spark-prep:
	python3 src/data/spark_preprocess.py --config config/config.yaml

train:
	python3 src/models/train.py --config config/config.yaml

evaluate:
	python3 src/models/evaluate.py --config config/config.yaml

drift:
	python3 src/monitoring/drift_detector.py --config config/config.yaml

pipeline: ingest preprocess train evaluate drift
	@echo "End-to-End MLOps Pipeline executed successfully!"

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest --cov=src --cov-report=term-missing tests/

lint:
	flake8 src/ tests/ --max-line-length=120
	black --check --line-length=120 src/ tests/

docker-build:
	docker build -t ai-news-classifier:latest -f Dockerfile .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
