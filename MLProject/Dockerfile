# Dockerfile for Heart Disease Classification MLflow Project
# Author: Eugenia Grasela Maia

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY conda.yaml .
RUN pip install --no-cache-dir \
    pandas>=1.3.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.0.0 \
    joblib>=1.1.0 \
    mlflow>=2.0.0

# Copy MLflow project files
COPY MLProject .
COPY modelling.py .
COPY heart_disease_preprocessing/ ./heart_disease_preprocessing/

# Create output directory
RUN mkdir -p /app/model /app/mlruns

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Run the training script
CMD ["python", "modelling.py"]
