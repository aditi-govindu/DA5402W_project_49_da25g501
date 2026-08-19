# Production Dockerfile for AI vs Human News Classifier Service
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure directories exist
RUN mkdir -p config src data models reports logs

# Copy source code, config, data, and models
COPY config/ config/
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Install package in editable mode
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
