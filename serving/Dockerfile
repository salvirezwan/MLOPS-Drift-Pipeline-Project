FROM python:3.11-slim

WORKDIR /app

# Install only what's needed for serving (no DVC, GE, Jupyter, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    prometheus-fastapi-instrumentator==7.0.0 \
    prometheus-client \
    mlflow==2.13.0 \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    pyarrow==15.0.2 \
    duckdb==0.10.3 \
    python-dotenv==1.0.1 \
    pyyaml==6.0.1

# Copy application code
COPY serving/ serving/
COPY feature_store/ feature_store/
COPY params.yaml .

# MLflow runs and model registry are mounted at runtime via docker-compose volume
ENV MLFLOW_TRACKING_URI=mlruns
ENV SHADOW_MODE=true

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
